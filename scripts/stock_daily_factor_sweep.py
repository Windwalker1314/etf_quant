"""Prespecified daily-stock factor and risk-budget comparisons, research only.

Run after stock-sync. Outputs are ignored by Git. No live account or site state is read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from steadyquant.composite_ledger import simulate_composite
from steadyquant.composite_study import quarantine_stock_placeholders
from steadyquant.config import ROOT, load_config, write_json
from steadyquant.data import TZ, Cache
from steadyquant.metrics import window_performance
from steadyquant.stock_daily_strategy import _member_mask, _risk_name_mask
from steadyquant.stock_data import load_stock_snapshot
from steadyquant.stock_factors import latest_reports

WINDOWS = {
    "development": ("2016-01-01", "2020-12-31"),
    "validation": ("2021-01-01", "2022-12-31"),
    "recent": ("2023-01-01", None),
    "full": ("2016-01-01", None),
}


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def fundamental_panels(snapshot: dict, dates: pd.DatetimeIndex, symbols: pd.Index):
    values = snapshot["valuations"].copy()
    values["date"] = pd.to_datetime(values.trade_date)
    reports = snapshot["financials"].copy()
    for column in ("ann_date", "end_date"):
        reports[column] = pd.to_datetime(reports[column], errors="coerce")
    rows = []
    for date, group in values.groupby("date"):
        fin = latest_reports(reports, date)
        val = group.drop_duplicates("ts_code").set_index("ts_code")
        frame = pd.DataFrame(index=val.index)
        annualizer = 12 / fin.end_date.dt.month.where(fin.end_date.dt.month.isin([3, 6, 9, 12]))
        frame["roe"] = pd.to_numeric(fin.roe, errors="coerce") * annualizer
        frame["roa"] = pd.to_numeric(fin.roa, errors="coerce") * annualizer
        pe = pd.to_numeric(val.pe_ttm, errors="coerce")
        pb = pd.to_numeric(val.pb, errors="coerce")
        frame["ep"] = 1 / pe.where(pe.gt(0))
        frame["bp"] = 1 / pb.where(pb.gt(0))
        frame["dividend"] = pd.to_numeric(val.dv_ttm, errors="coerce")
        frame["date"] = date
        rows.append(frame.reset_index().rename(columns={"ts_code": "symbol"}))
    factors = pd.concat(rows, ignore_index=True)
    result = {}
    for name in ("roe", "roa", "ep", "bp", "dividend"):
        monthly = factors.pivot(index="date", columns="symbol", values=name).reindex(columns=symbols)
        # Valuation is published after this month's first close: usable next session.
        result[name] = monthly.reindex(dates, method="ffill").shift(1)
    return result


def build_panels(snapshot: dict, dates: pd.DatetimeIndex, market: pd.Series):
    raw = snapshot["daily"]
    bars = raw.loc[(raw[["open", "high", "low", "close"]] > 0).all(axis=1) & raw.vol.gt(0)]
    merged = bars.merge(snapshot["adj_factor"][["ts_code", "trade_date", "adj_factor"]],
                        on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    merged["date"] = pd.to_datetime(merged.trade_date)
    symbols = pd.Index(sorted(merged.ts_code.unique()))

    def panel(col):
        return merged.pivot(index="date", columns="ts_code", values=col).reindex(index=dates, columns=symbols)

    close, factor, amount, volume = (panel(c) for c in ("close", "adj_factor", "amount", "vol"))
    price = close * factor.where(factor.gt(0))
    member, _ = _member_mask(snapshot["members"], dates, symbols)
    risk_name = _risk_name_mask(snapshot["namechange"], dates, symbols)
    listing = snapshot["metadata"].drop_duplicates("ts_code").set_index("ts_code")
    listed = pd.to_datetime(listing.list_date.reindex(symbols), errors="coerce")
    delisted = pd.to_datetime(listing.delist_date.reindex(symbols), errors="coerce")
    life = pd.DataFrame(
        (dates.to_numpy()[:, None] >= listed.to_numpy()[None, :])
        & ((dates.to_numpy()[:, None] < delisted.to_numpy()[None, :]) | delisted.isna().to_numpy()[None, :]),
        index=dates, columns=symbols,
    )
    eligible = member & life & ~risk_name & (price.notna().rolling(270, min_periods=270).sum() >= 253)
    eligible &= (amount * 1000).rolling(20, min_periods=15).mean().ge(30_000_000)
    eligible &= volume.gt(0) & (close * 100).le(200000 / 10)
    ret = price.pct_change(fill_method=None)
    vol = ret.rolling(63, min_periods=50).std() * np.sqrt(252)
    mom = (price.shift(21) / price.shift(126) - 1 + price.shift(21) / price.shift(252) - 1) / 2
    rev = -(price / price.shift(21) - 1)
    fund = fundamental_panels(snapshot, dates, symbols)
    market_ret = market.pct_change(fill_method=None)
    beta = (ret.mul(market_ret, axis=0).rolling(252, min_periods=200).mean()
            - ret.rolling(252, min_periods=200).mean()
            .mul(market_ret.rolling(252, min_periods=200).mean(), axis=0))
    beta = beta.div(market_ret.rolling(252, min_periods=200).var(), axis=0)
    market_mom = (market.shift(21) / market.shift(126) - 1
                  + market.shift(21) / market.shift(252) - 1) / 2
    residual_mom = mom - beta.mul(market_mom, axis=0)
    screened = eligible & vol.gt(0) & vol.lt(1)

    def eligible_rank(frame: pd.DataFrame) -> pd.DataFrame:
        # Rank only the point-in-time eligible universe. Ranking the later
        # historical union would let future constituents change past scores.
        return rank(frame.where(screened))

    quality = (eligible_rank(fund["roe"]) + eligible_rank(fund["roa"])) / 2
    value = (eligible_rank(fund["ep"]) + eligible_rank(fund["bp"])) / 2
    score = {
        "momentum_lowvol": 0.6 * eligible_rank(mom) + 0.4 * eligible_rank(-vol),
        "quality_value": 0.5 * quality + 0.5 * value,
        "quality_lowvol": 0.5 * quality + 0.5 * eligible_rank(-vol),
        "reversal_quality": 0.5 * eligible_rank(rev) + 0.5 * quality,
        "lowvol_dividend": 0.5 * eligible_rank(-vol) + 0.5 * eligible_rank(fund["dividend"]),
        "residual_momentum_lowvol": 0.6 * eligible_rank(residual_mom) + 0.4 * eligible_rank(-vol),
        "momentum_quality": 0.5 * eligible_rank(mom) + 0.5 * quality,
        "momentum_dividend": 0.5 * eligible_rank(mom) + 0.5 * eligible_rank(fund["dividend"]),
    }
    # Momentum-only eligibility applies only to the original momentum recipe.
    score["momentum_lowvol"] = score["momentum_lowvol"].where(mom.gt(0))
    for name in score:
        score[name] = score[name].where(screened)

    ma200 = market.rolling(200, min_periods=200).mean()
    rising = market.gt(ma200)
    market_vol = market.pct_change(fill_method=None).rolling(20, min_periods=20).std() * np.sqrt(252)
    exposure = {
        "always_100": pd.Series(1.0, index=dates),
        "trend_90_30": pd.Series(np.where(rising, 0.9, 0.3), index=dates),
        "trend_90_0": pd.Series(np.where(rising, 0.9, 0.0), index=dates),
    }
    for target_vol in (0.10, 0.12, 0.15):
        exposure[f"trend_vol{int(target_vol * 100)}"] = (
            target_vol / market_vol).clip(upper=0.9).where(rising, 0.0)
    for name, frame in exposure.items():
        if name == "always_100":
            continue
        frame.loc[ma200.isna() | market.isna()] = 0.0
    return score, exposure, close


def choose_targets(score: pd.DataFrame, exposure: pd.Series, close: pd.DataFrame,
                   *, fraction: float = 1.0, capital: float = 200000) -> pd.DataFrame:
    targets = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    held = []
    for date in score.index:
        per_name_budget = capital * fraction * float(exposure.loc[date]) / 10
        affordable = close.loc[date].mul(100).le(per_name_budget)
        ranked = score.loc[date].where(affordable).dropna().sort_values(ascending=False, kind="stable")
        ranks = {symbol: i for i, symbol in enumerate(ranked.index)}
        held = [symbol for symbol in held if ranks.get(symbol, 999999) < 20]
        for symbol in ranked.index:
            if len(held) == 10:
                break
            if symbol not in held:
                held.append(symbol)
        if len(held) == 10:
            targets.loc[date, held] = float(exposure.loc[date]) / 10
    return targets.loc[:, (targets > 0).any(axis=0)]


def main():
    if ("--pure-core" in sys.argv) == ("--pure-expanded" in sys.argv):
        raise SystemExit("Choose exactly one of --pure-core or --pure-expanded")
    is_csi500 = "--stock500" in sys.argv
    root = ROOT / "outputs/stock_daily" / ("sweep500" if is_csi500 else "sweep")
    root = root / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True)
    if is_csi500:
        source = ROOT / "data/stocks500"
        manifest = json.loads((source / "manifest.json").read_text())
        snapshot = {}
        for name, info in manifest["files"].items():
            path = source / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"CSI500 snapshot digest mismatch: {name}")
            snapshot[path.stem] = pd.read_parquet(path)
    else:
        snapshot, manifest = load_stock_snapshot()
    snapshot, quarantined = quarantine_stock_placeholders(snapshot, root)
    cal = pd.read_parquet(ROOT / "data/calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1)
                                 & cal.cal_date.astype(str).between("20140101", manifest["end"]),
                                 "cal_date"])).sort_values().unique()
    bar = Cache().read("510300.SH").set_index("date")
    market = (bar.close * bar.adj_factor).reindex(dates)
    score, budgets, close = build_panels(snapshot, dates, market)
    cfg = load_config(ROOT / "configs/family.yaml")
    cfg.update(initial_cash=200000, rebalance_frequency="daily", rebalance_band=0.02,
               stock_rebalance_band=0.02, commission_bps=3.0, cash_rate=0.0)
    results = []
    for name, frame in score.items():
        for budget_name, budget in budgets.items():
            if "--pure-core" in sys.argv and (
                name not in {"momentum_lowvol", "residual_momentum_lowvol",
                             "momentum_quality", "quality_value"}
                or budget_name not in {"always_100", "trend_90_30", "trend_vol12"}
            ):
                continue
            if "--pure-expanded" in sys.argv and (
                name not in {"quality_lowvol", "reversal_quality",
                             "lowvol_dividend", "momentum_dividend"}
                or budget_name not in {"always_100", "trend_90_30"}
            ):
                continue
            label = name + "__" + budget_name
            target = choose_targets(frame, budget, close)
            if target.shape[1] < 10:
                print(label, "too few symbols", flush=True)
                continue
            result = simulate_composite({}, snapshot, target, cfg, start="2016-01-01")
            metrics = {period: window_performance(result.equity.equity, *window,
                                                  rf=cfg["risk_free_rate"])
                       for period, window in WINDOWS.items()}
            blocks = result.events.type.eq("qualification_block").sum() if not result.events.empty else 0
            row = dict(model=label, trades=len(result.trades), blocks=int(blocks),
                       **{f"{period}_{metric}": value[metric] for period, value in metrics.items()
                          for metric in ("cagr", "sharpe", "max_drawdown")})
            results.append(row)
            pd.DataFrame(results).to_csv(root / "results.csv", index=False)
            print(label, "dev", round(row["development_sharpe"], 3),
                  "val", round(row["validation_sharpe"], 3),
                  "recent", round(row["recent_sharpe"], 3), flush=True)
    write_json(root / "protocol.json", dict(snapshot_end=manifest["end"],
               quarantined=quarantined, factors=list(score), budgets=list(budgets),
               universe="CSI500 historical dated members" if is_csi500 else "CSI300 historical dated main-board members",
               ranking="only point-in-time eligible members; monthly data shifted one session",
               execution="close signal, next-open 100-share lots, T+1, dated costs, price limits, dividends",
               capital=200000, stock_only=True,
               selection="prespecified families; development, validation and recent windows all disclosed; no live promotion by best full-sample Sharpe",
               current_live_policy_unchanged=True))
    print(json.dumps({"output": str(root), "models": len(results)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
