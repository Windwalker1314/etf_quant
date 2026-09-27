"""Research-only, point-in-time CSI300 breadth overlays for ETF allocations."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .factors import adjusted_frame
from .metrics import window_performance
from .strategy import target_weights

STOCK_ROOT = ROOT / "data/stocks"
FAMILIES = ("base", "weak_breadth", "selloff_breadth", "price_divergence", "below_trend")


def breadth_signals(
    daily: pd.DataFrame, members: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Use a constituent snapshot only *after* its dated observation.

    A missing/stale snapshot or <200 actually traded members never becomes a
    bearish signal. Stock pct_chg uses the vendor's ex-rights prior close.
    """
    if daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError("Duplicate stock daily bars")
    if members.duplicated(["trade_date", "con_code"]).any():
        raise ValueError("Duplicate constituent observations")
    bars = daily[["ts_code", "trade_date", "pct_chg"]].copy()
    bars["date"] = pd.to_datetime(bars.trade_date.astype(str), format="%Y%m%d")
    returns = bars.pivot(index="date", columns="ts_code", values="pct_chg").reindex(dates)
    # The cached historical stock panel intentionally contains only 60/00
    # main-board members. Never substitute today's CSI300 composition.
    obs = members[members.con_code.isin(returns.columns)].copy()
    if obs.empty:
        raise ValueError("No historical main-board constituents covered")
    obs["date"] = pd.to_datetime(obs.trade_date.astype(str), format="%Y%m%d")
    snapshots = sorted(obs.date.unique())
    groups = {date: set(frame.con_code) for date, frame in obs.groupby("date")}
    columns = list(returns.columns)
    member = pd.DataFrame(False, index=dates, columns=columns)
    for date in dates:
        pos = np.searchsorted(snapshots, date, side="left") - 1
        if pos < 0 or (date - snapshots[pos]).days > 62:
            continue
        member.loc[date, list(groups[snapshots[pos]])] = True
    traded = member & returns.notna()
    count = traded.sum(axis=1)
    valid = count >= 200
    adv = ((returns > 0) & traded).sum(axis=1).div(count).where(valid)
    drop2 = ((returns <= -2) & traded).sum(axis=1).div(count).where(valid)
    # Build ex-rights return indices from today's and past daily changes only.
    # A 126-session warm-up with >=110 real bars protects newly listed names.
    gross = (1 + returns / 100).where(returns.notna(), 1).cumprod()
    warmed = returns.notna().rolling(126, min_periods=126).sum() >= 110
    above = gross > gross.rolling(126, min_periods=126).mean()
    valid_trend = traded & warmed
    trend_count = valid_trend.sum(axis=1)
    above_share = (above & valid_trend).sum(axis=1).div(trend_count).where(trend_count >= 200)
    return pd.DataFrame(
        {
            "constituents_traded": count,
            "advancing_share": adv,
            "advancing_20": adv.rolling(20, min_periods=20).mean(),
            "large_decline_20": drop2.rolling(20, min_periods=20).mean(),
            "above_126": above_share,
        },
        index=dates,
    )


def breadth_targets(base: pd.DataFrame, breadth: pd.DataFrame, cn_proxy: pd.Series) -> dict[str, pd.DataFrame]:
    """Four fixed, downside-only hypotheses; never increase leverage."""
    cn_cols = [s for s in ("510300.SH", "510500.SH", "159915.SZ", "588000.SH") if s in base]
    if not cn_cols:
        raise ValueError("No CN equity ETFs")
    weak = breadth.advancing_20 < 0.48
    selloff = breadth.large_decline_20 > 0.16
    divergence = (cn_proxy / cn_proxy.shift(63) > 1) & weak
    below = breadth.above_126 < 0.40
    flags = dict(weak_breadth=weak, selloff_breadth=selloff,
                 price_divergence=divergence, below_trend=below)
    targets = {"base": base}
    for name, flag in flags.items():
        frame = base.copy()
        frame[cn_cols] = frame[cn_cols].mul(1 - 0.5 * flag.reindex(base.index).fillna(False).astype(float), axis=0)
        if frame.min().min() < 0 or frame.max().max() > 0.30 + 1e-10 or frame.sum(axis=1).max() > 1 + 1e-10:
            raise AssertionError(f"Invalid allocation: {name}")
        targets[name] = frame
    return targets


def run():
    cfg = {**load_active_config(), "initial_cash": 200000}
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    base, _ = target_weights(data, cfg)
    daily = pd.read_parquet(STOCK_ROOT / "daily.parquet")
    members = pd.read_parquet(STOCK_ROOT / "members.parquet")
    end = min(str(d.date.max().date()) for d in data.values())
    end = min(end, pd.to_datetime(daily.trade_date.astype(str), format="%Y%m%d").max().strftime("%Y-%m-%d"))
    base = base.loc[:end]
    dates = base.index
    breadth = breadth_signals(daily, members, dates)
    adjusted = adjusted_frame(data)
    cn_proxy = adjusted[adjusted.symbol == "510300.SH"].set_index("date").close.reindex(dates)
    targets = breadth_targets(base, breadth, cn_proxy)
    cutoff = pd.Timestamp("2022-12-30")
    old_breadth = breadth_signals(
        daily[daily.trade_date.astype(str) <= "20221230"],
        members[members.trade_date.astype(str) <= "20221230"],
        dates[dates <= cutoff],
    )
    pd.testing.assert_frame_equal(old_breadth, breadth.loc[:cutoff], check_freq=False)
    old_targets = breadth_targets(base.loc[:cutoff], old_breadth, cn_proxy.loc[:cutoff])
    for name in FAMILIES:
        pd.testing.assert_frame_equal(old_targets[name], targets[name].loc[:cutoff], check_freq=False)
    out = ROOT / "outputs/etf_breadth_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    source_files = [STOCK_ROOT / "daily.parquet", STOCK_ROOT / "members.parquet"]
    write_json(out / "protocol.json", dict(
        design="Retrospective, research only. Past-dated monthly CSI300 membership, never current constituents retroactively. Signal close to next-session open; only CN ETF bucket reduced by at most 50%, all other targets stay original.",
        rules=dict(weak_breadth="20-session average advancing share <48%",
                   selloff_breadth="20-session average share losing >=2% in one day >16%",
                   price_divergence="CSI300 ETF up over 63 sessions but breadth weak",
                   below_trend="<40% of constituents above their ex-rights 126-session average"),
        selection="2019-2022 Sharpe >= base+0.05 and CAGR >= base and DD no worse by 2pp; 2023+ review only. All dates previously seen in other studies.",
        data_end=end, etf_sha256=cache.snapshot_digest,
        stock_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        protected_sha256=hashes, costs="20万元, 1.5bp/min5 commission, 5bp slippage, 100-share lots",
        limitations="Historical CSI300 members available only as monthly dated snapshots; exact intraday publication not retained, so membership strictly lagged. Vendor historical revisions possible. No virgin holdout.",
    ))
    breadth.to_parquet(out / "breadth.parquet")
    rows, curves = [], {}
    for name in FAMILIES:
        print(f"Breadth {name}", flush=True)
        targets[name].to_parquet(out / f"targets_{name}.parquet")
        result = simulate(data, targets[name], cfg, start="2019-01-01", end=end)
        curves[name] = result.equity.equity
        result.equity.to_parquet(out / f"equity_{name}.parquet")
        for period, lo, hi in [("2019-train", "2019-01-01", "2022-12-31"),
                               ("2023-review", "2023-01-01", None),
                               ("2019-now", "2019-01-01", None)]:
            rows.append(dict(name=name, fresh_start="2019", mode="normal", period=period,
                             trades=len(result.trades),
                             **window_performance(result.equity.equity, lo, hi, cfg["risk_free_rate"])))
        stressed = simulate(data, targets[name], cfg, start="2019-01-01", end=end, cost_multiplier=2)
        rows.append(dict(name=name, fresh_start="2019", mode="double", period="2019-now",
                         trades=len(stressed.trades),
                         **window_performance(stressed.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        long_run = simulate(data, targets[name], cfg, start="2015-01-05", end=end)
        rows.append(dict(name=name, fresh_start="2015", mode="normal", period="2015-now",
                         trades=len(long_run.trades),
                         **window_performance(long_run.equity.equity, "2015-01-05", None, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    train = table[(table.fresh_start == "2019") & (table["mode"] == "normal") &
                  (table.period == "2019-train")].set_index("name")
    b = train.loc["base"]
    passed = train[(train.sharpe >= b.sharpe + .05) & (train.cagr >= b.cagr) &
                   (train.max_drawdown >= b.max_drawdown - .02)].index.tolist()
    intervals = {name: paired_sharpe_interval(curves[name], curves["base"])
                 for name in FAMILIES if name != "base"}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    write_json(out / "summary.json", dict(passed_train=passed, active_unchanged=True,
                                          prefix_checks=True, paired_intervals_unadjusted=intervals))
    print(table[["name", "fresh_start", "mode", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
