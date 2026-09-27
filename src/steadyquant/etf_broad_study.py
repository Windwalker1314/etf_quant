"""Research-only cross-asset ETF selection under the live class risk budget."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .etf_broad_study_universe import EXTRA, EXTRA_NAMES
from .etf_universe_study import ADDITIONS
from .metrics import window_performance
from .strategy import target_weights

FAMILIES = (
    "base", "leader", "top_two", "positive_top_two", "cn_satellite",
    "low_vol", "risk_adjusted_top_two",
)


def research_config(base: dict) -> dict:
    cfg = {**base, "initial_cash": 200000, "assets": [a.copy() for a in base["assets"]]}
    known = {a["symbol"] for a in cfg["assets"]}
    extras = [(s, name, bucket) for group in ADDITIONS.values() for s, name, bucket in group]
    extras += [(s, EXTRA_NAMES[s], bucket) for s, bucket in EXTRA.items()]
    for symbol, name, bucket in extras:
        if symbol not in known:
            cfg["assets"].append(dict(symbol=symbol, name=name, kind="fund", bucket=bucket, weight=0.0))
            known.add(symbol)
    return cfg


def load_data(cfg: dict) -> dict[str, pd.DataFrame]:
    cache = Cache()
    result = {}
    for asset in cfg["assets"]:
        symbol = asset["symbol"]
        custom = ROOT / "data/etf_broad_research/bars" / f"{symbol}.parquet"
        result[symbol] = pd.read_parquet(custom) if custom.exists() else cache.read(symbol)
        if result[symbol].empty:
            raise ValueError(f"Missing historical bars: {symbol}")
        if result[symbol].date.max() != pd.Timestamp("2026-09-24"):
            raise ValueError(f"Stale historical bars: {symbol}")
    return result


def features(data: dict[str, pd.DataFrame], cfg: dict, dates: pd.DatetimeIndex) -> dict:
    close = pd.DataFrame({s: d.set_index("date").close * d.set_index("date").adj_factor
                          for s, d in data.items()}).reindex(dates)
    amount = pd.DataFrame({s: d.set_index("date").amount for s, d in data.items()}).reindex(dates)
    volume = pd.DataFrame({s: d.set_index("date").volume for s, d in data.items()}).reindex(dates)
    observed = close.notna()
    marked = close.ffill()
    returns = marked.pct_change(fill_method=None)
    momentum = 0.5 * (marked.shift(21) / marked.shift(147) - 1) + 0.5 * (
        marked.shift(21) / marked.shift(273) - 1
    )
    volatility = returns.rolling(126, min_periods=126).std() * np.sqrt(252)
    eligible = observed & (observed.cumsum() >= cfg["min_history"])
    eligible &= observed.rolling(126, min_periods=126).sum() >= 120
    eligible &= amount.rolling(20, min_periods=20).mean() >= cfg["min_amount"]
    eligible &= (volume > 0) & momentum.notna() & volatility.notna()
    return dict(momentum=momentum, eligible=eligible, volatility=volatility,
                trend=marked / marked.rolling(252, min_periods=252).mean() - 1, returns=returns)


def first_session_mask(dates: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    if frequency == "daily":
        return np.ones(len(dates), dtype=bool)
    if frequency not in {"weekly", "monthly"}:
        raise ValueError(frequency)
    periods = dates.to_period("W-SUN" if frequency == "weekly" else "M")
    return np.r_[True, periods[1:] != periods[:-1]]


def broad_targets(
    data: dict[str, pd.DataFrame], core: pd.DataFrame, cfg: dict, family: str,
    frequency: str = "monthly",
) -> pd.DataFrame:
    decisions = first_session_mask(core.index, frequency)
    if family == "base":
        return core.reindex(columns=list(data), fill_value=0.0)
    if family not in FAMILIES:
        raise ValueError(family)
    dates = core.index
    f = features(data, cfg, dates)
    members = {g: [a["symbol"] for a in cfg["assets"] if a["bucket"] == g]
               for g in {a["bucket"] for a in cfg["assets"]}}
    sector = set(EXTRA) & set(members["cn_equity"])
    sector -= {"159905.SZ"}  # dividend exposure is a broad style, not a sector satellite
    weights = pd.DataFrame(0.0, index=dates, columns=list(data))
    for i, date in enumerate(dates):
        if not decisions[i]:
            weights.iloc[i] = weights.iloc[i - 1]
            continue
        row = core.loc[date].reindex(weights.columns, fill_value=0.0).copy()
        for group, symbols in members.items():
            if group == "gold":
                continue
            budget = row[symbols].sum()
            if budget <= 0:
                continue
            if family == "cn_satellite" and group != "cn_equity":
                continue
            allowed = f["eligible"].loc[date, symbols]
            if family == "positive_top_two":
                allowed &= f["trend"].loc[date, symbols] > 0
            if family == "cn_satellite":
                allowed &= pd.Series([s in sector for s in symbols], index=symbols)
            score = f["momentum"].loc[date, symbols]
            if family == "low_vol":
                score = -f["volatility"].loc[date, symbols]
            elif family == "risk_adjusted_top_two":
                score = score / f["volatility"].loc[date, symbols].clip(lower=0.08)
            ranked = score[allowed].sort_values(ascending=False, kind="stable")
            n = 1 if family in {"leader", "cn_satellite", "low_vol"} else 2
            chosen = ranked.index[:n]
            if family == "cn_satellite":
                if len(chosen):
                    row[symbols] *= 0.7
                    row[chosen] += 0.3 * budget
            else:
                row[symbols] = 0.0
                if len(chosen):
                    row[chosen] = min(0.30, budget / len(chosen))
        row = row.clip(lower=0, upper=0.30)
        hist = f["returns"].iloc[max(0, i - 125):i + 1].fillna(0).to_numpy() @ row.to_numpy()
        if len(hist) == 126:
            risk = float(np.std(hist, ddof=1) * np.sqrt(252))
            row *= min(1.0, cfg["target_vol"] / max(risk, 1e-12))
        weights.loc[date] = row
    if (weights < 0).any().any() or weights.sum(axis=1).max() > 1 + 1e-9:
        raise AssertionError("Long-only and no-leverage constraints failed")
    if weights.max().max() > 0.30 + 1e-9:
        raise AssertionError("Single ETF cap failed")
    return weights


def run():
    base = load_active_config()
    cfg = research_config(base)
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    data = load_data(cfg)
    core_data = {a["symbol"]: data[a["symbol"]] for a in base["assets"]}
    core, _ = target_weights(core_data, {**base, "initial_cash": 200000})
    out = ROOT / "outputs/etf_broad_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    write_json(out / "protocol.json", dict(
        families=FAMILIES, symbols={a["symbol"]: a["bucket"] for a in cfg["assets"]},
        design="Phase 1: keep original adaptive asset-class budgets; monthly select strongest 1 or 2 ETFs per class by equal 6m/12m adjusted momentum skipping latest 21 sessions. Third version requires 252-session trend. Fourth uses 30% of China-equity budget for one sector leader and retains 70% core. Phase 2, added after observing phase 1 underperformance: select one lowest-volatility ETF per group or two highest momentum/volatility ETFs. No fitted parameters.",
        eligibility="253 real bars, 120/126 recent sessions, trailing 20-day amount >=10m CNY, positive volume, 126-session volatility. No pre-listing backfill.",
        execution="20w CNY, first monthly trading-session close, next open, 100-share lots, minimum5CNY commission, 5bp slippage, 1% volume cap; double-cost sensitivity.",
        screen="2019-2022 Sharpe >= base+0.05, CAGR >= base, drawdown no worse by 2pp; 2023+ retrospective review only.",
        limitation="Hand-selected currently surviving ETF candidates; not survivorship-free. QDII NAV premium not modelled. All calendar periods have been previously observed; no virgin holdout.",
        protected_sha256=hashes,
    ))
    tables, curves = [], {}
    for family in FAMILIES:
        print(f"Backtest {family}", flush=True)
        w = broad_targets(data, core, cfg, family)
        if family != "base":
            cut = pd.Timestamp("2022-12-30")
            prefix = {s: d[d.date <= cut] for s, d in data.items()}
            old = broad_targets(prefix, core.loc[:cut], cfg, family)
            pd.testing.assert_frame_equal(old, w.loc[:cut], check_freq=False)
        w.to_parquet(out / f"targets_{family}.parquet")
        r = simulate(data, w, cfg, start="2019-01-01", end="2026-09-24")
        curves[family] = r.equity.equity
        r.equity.to_parquet(out / f"equity_{family}.parquet")
        for tag, lo, hi in (("train", "2019-01-01", "2022-12-31"),
                             ("review", "2023-01-01", None), ("all", "2019-01-01", None)):
            tables.append(dict(family=family, mode="carried2019", period=tag, trades=len(r.trades),
                               **window_performance(r.equity.equity, lo, hi, cfg["risk_free_rate"])))
        stress = simulate(data, w, cfg, start="2019-01-01", end="2026-09-24", cost_multiplier=2)
        tables.append(dict(family=family, mode="double_cost", period="all", trades=len(stress.trades),
                           **window_performance(stress.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        fresh = simulate(data, w, cfg, start="2023-01-01", end="2026-09-24")
        tables.append(dict(family=family, mode="fresh2023", period="review", trades=len(fresh.trades),
                           **window_performance(fresh.equity.equity, "2023-01-01", None, cfg["risk_free_rate"])))
    table = pd.DataFrame(tables)
    table.to_csv(out / "comparison.csv", index=False)
    train = table[(table["mode"] == "carried2019") & (table.period == "train")].set_index("family")
    b = train.loc["base"]
    eligible = train[(train.sharpe >= b.sharpe + 0.05) & (train.cagr >= b.cagr)
                     & (train.max_drawdown >= b.max_drawdown - 0.02)]
    selected = str(eligible.sharpe.idxmax()) if not eligible.empty else "base"
    long_rows = []
    for family in ("base", selected) if selected != "base" else ("base",):
        w = pd.read_parquet(out / f"targets_{family}.parquet")
        long = simulate(data, w, cfg, start="2015-01-05", end="2026-09-24")
        for tag, lo, hi in (("pre2019", "2015-01-05", "2018-12-31"),
                             ("full", "2015-01-05", None)):
            long_rows.append(dict(family=family, period=tag,
                                  **window_performance(long.equity.equity, lo, hi, cfg["risk_free_rate"])))
    pd.DataFrame(long_rows).to_csv(out / "long_check.csv", index=False)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in protected)
    write_json(out / "summary.json", dict(selected_on_train=selected, active_unchanged=True,
        prefix_checks=True,
        paired_intervals={family: paired_sharpe_interval(curves[family], curves["base"])
                          for family in FAMILIES if family != "base"}))
    print(table[["family", "mode", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
