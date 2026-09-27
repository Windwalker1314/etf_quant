"""Research-only ETF factor families; never changes the active allocation."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .factors import adjusted_frame
from .metrics import window_performance
from .strategy import target_weights

FAMILIES = (
    "base",
    "trend_quality",
    "downside_guard",
    "volume_confirmation",
    "share_flow",
    "trend_and_flow",
)
SHARE_ROOT = ROOT / "data/etf_share_size"


def load_share_history(symbols: list[str], root: Path = SHARE_ROOT) -> tuple[dict[str, pd.DataFrame], dict]:
    """Never silently substitute another ETF's flow or a missing history."""
    result, manifest = {}, {}
    for symbol in symbols:
        path = root / f"{symbol}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ETF share history for {symbol}")
        frame = pd.read_parquet(path)
        required = {"ts_code", "trade_date", "total_share"}
        if not required.issubset(frame) or frame.empty:
            raise ValueError(f"Invalid ETF share history for {symbol}")
        if not frame.ts_code.eq(symbol).all() or frame.trade_date.duplicated().any():
            raise ValueError(f"Mislabelled or duplicated ETF share history for {symbol}")
        frame["date"] = pd.to_datetime(frame.trade_date.astype(str), format="%Y%m%d")
        frame = frame.sort_values("date").reset_index(drop=True)
        result[symbol] = frame
        manifest[symbol] = dict(
            rows=len(frame), first=str(frame.date.iloc[0].date()), last=str(frame.date.iloc[-1].date()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return result, manifest


def factor_multipliers(
    data: dict[str, pd.DataFrame], shares: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Six predeclared alternatives, all causal and bounded at 0.75..1.

    Price/volume at today's close can trade next open. ETF share observations are
    next-morning publications and are lagged one entire portfolio session.
    Missing observations are neutral, never filled from a later release.
    """
    panel = adjusted_frame(data)
    symbols = list(data)
    close = panel.pivot(index="date", columns="symbol", values="close").reindex(columns=symbols)
    amount = panel.pivot(index="date", columns="symbol", values="amount").reindex_like(close)
    ret = close.ffill().pct_change(fill_method=None)
    vol63 = ret.rolling(63, min_periods=63).std() * np.sqrt(252)
    momentum63 = close / close.shift(63) - 1
    efficiency63 = ((close - close.shift(63)).abs() / close.diff().abs().rolling(63).sum()).clip(0, 1)
    # Suppress coherent falling trends. A choppy negative return has little effect.
    trend_bad = (-momentum63 / (vol63 * np.sqrt(63 / 252))).clip(lower=0)
    trend = 1 - 0.25 * np.tanh(trend_bad * efficiency63)
    drawdown126 = close / close.rolling(126, min_periods=126).max() - 1
    downside = 1 - 0.25 * ((-drawdown126 - 0.05) / 0.15).clip(0, 1)
    # Signed traded value is an OBV-family proxy; use only same-ETF ratios.
    signed = np.sign(ret) * amount
    balance20 = signed.rolling(20, min_periods=20).sum() / amount.rolling(20, min_periods=20).sum()
    volume_guard = 1 - 0.25 * (-balance20).clip(0, 1)
    share_panel = pd.DataFrame(index=close.index, columns=symbols, dtype=float)
    for symbol in symbols:
        frame = shares[symbol]
        values = frame.set_index("date").total_share.astype(float).where(lambda v: v > 0)
        # Up to three sessions can bridge an update gap, but a long gap is missing.
        share_panel[symbol] = values.reindex(close.index).ffill(limit=3)
    log_share = np.log(share_panel)
    flow20 = (log_share - log_share.shift(20)).shift(1)
    # Very large one-day share jumps may be splits or source revisions. Neutralize
    # the affected 20-session window instead of interpreting them as demand.
    discontinuity = log_share.diff().abs().gt(0.30).rolling(21, min_periods=1).max().shift(1)
    flow20 = flow20.mask(discontinuity.astype(bool))
    flow = 1 - 0.25 * (-flow20 / 0.10).clip(0, 1)
    ones = pd.DataFrame(1.0, index=close.index, columns=symbols)
    factors = dict(
        base=ones, trend_quality=trend, downside_guard=downside,
        volume_confirmation=volume_guard, share_flow=flow,
        trend_and_flow=(trend.fillna(1.0) + flow.fillna(1.0)) / 2,
    )
    for name, frame in factors.items():
        factors[name] = frame.reindex_like(close).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.75, 1.0)
    return factors


def run() -> Path:
    cfg = {**load_active_config(), "initial_cash": 200000}
    if cfg["model"] != "adaptive" or cfg["rebalance_frequency"] != "monthly_first_session":
        raise ValueError("This study expects the unchanged monthly adaptive live policy")
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    protected_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    symbols = list(data)
    shares, share_manifest = load_share_history(symbols)
    if min(frame.date.max() for frame in shares.values()) < min(frame.date.max() for frame in data.values()):
        raise ValueError("ETF share series lag the price snapshot")
    end = str(min(frame.date.max() for frame in data.values()).date())
    base, _ = target_weights(data, cfg)
    multipliers = factor_multipliers(data, shares)
    weights = {name: base * multipliers[name] for name in FAMILIES}
    pd.testing.assert_frame_equal(weights["base"], base)
    for name, target in weights.items():
        if target.min().min() < 0 or target.max().max() > 0.30 + 1e-10 or target.sum(axis=1).max() > 1 + 1e-10:
            raise AssertionError(f"Allocation bounds failed for {name}")
    cutoff = pd.Timestamp("2022-12-30")
    old_data = {s: f[f.date <= cutoff] for s, f in data.items()}
    old_shares = {s: f[f.date <= cutoff] for s, f in shares.items()}
    old_base, _ = target_weights(old_data, cfg)
    old_factors = factor_multipliers(old_data, old_shares)
    for name in FAMILIES:
        pd.testing.assert_frame_equal(old_base * old_factors[name], weights[name].loc[:cutoff], check_freq=False)
    out = ROOT / "outputs/etf_factor_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), end=end, families=FAMILIES,
        benchmark="Existing monthly adaptive policy; 20万元; no live changes",
        data_sha256=cache.snapshot_digest, share_manifest=share_manifest,
        factor_rules="Fixed 63-session risk-normalized trend efficiency, 126-session drawdown, 20-session signed amount, 20-session ETF share change; each can only reduce its original target by at most 25%; combined is equal average of trend and flow. No fitted weights or sign search.",
        timing="Price/amount through signal close; ETF shares lag one full portfolio session; next-session open fill. Missing share data neutral; >30% one-day share jumps neutral for affected 20 sessions.",
        selection="2019-2022 train: Sharpe >= base+0.05, CAGR >= base, drawdown no worse than base by 2pp; otherwise base. 2023+ is review only, no reselection. All periods previously observed, not virgin holdout.",
        costs="1.5bp commission, minimum5元, 5bp slippage; double-cost sensitivity; monthly first session.",
        limitations="Surviving 10-ETF universe, revised vendor historical data possible, QDII open premium not reconstructed, ETF distributions approximated as reinvestment, multiple hypotheses and already-seen history.",
        protected_sha256=protected_hashes,
    ))
    rows, equities = [], {}
    periods = {"2019-train": ("2019-01-01", "2022-12-31"),
               "2023-review": ("2023-01-01", None), "2019-now": ("2019-01-01", None)}
    for name in FAMILIES:
        print(f"Simulating {name}", flush=True)
        weights[name].to_parquet(out / f"targets_{name}.parquet")
        result = simulate(data, weights[name], cfg, start="2019-01-01", end=end)
        equities[name] = result.equity.equity
        result.equity.to_parquet(out / f"equity_{name}.parquet")
        for period in ("2019-train", "2023-review", "2019-now"):
            lo, hi = periods[period]
            rows.append(dict(name=name, start_mode="fresh2019", cost="normal", period=period,
                             trades=len(result.trades), **window_performance(result.equity.equity, lo, hi, cfg["risk_free_rate"])))
        stressed = simulate(data, weights[name], cfg, start="2019-01-01", end=end, cost_multiplier=2)
        rows.append(dict(name=name, start_mode="fresh2019", cost="double", period="2019-now",
                         trades=len(stressed.trades), **window_performance(stressed.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        fresh23 = simulate(data, weights[name], cfg, start="2023-01-01", end=end)
        rows.append(dict(name=name, start_mode="fresh2023", cost="normal", period="2023-now",
                         trades=len(fresh23.trades), **window_performance(fresh23.equity.equity, "2023-01-01", None, cfg["risk_free_rate"])))
        long_run = simulate(data, weights[name], cfg, start="2015-01-05", end=end)
        for period, lo, hi in [("2015-2018", "2015-01-01", "2018-12-31"),
                                ("2015-now", "2015-01-01", None)]:
            rows.append(dict(name=name, start_mode="fresh2015", cost="normal", period=period,
                             trades=len(long_run.trades), **window_performance(long_run.equity.equity, lo, hi, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    train = table[(table.start_mode == "fresh2019") & (table.cost == "normal") & (table.period == "2019-train")].set_index("name")
    base_row = train.loc["base"]
    eligible = train[(train.sharpe >= base_row.sharpe + 0.05) &
                     (train.cagr >= base_row.cagr) &
                     (train.max_drawdown >= base_row.max_drawdown - 0.02)]
    selected = str(eligible.sharpe.idxmax()) if not eligible.empty else "base"
    intervals = {name: paired_sharpe_interval(equities[name], equities["base"])
                 for name in FAMILIES if name != "base"}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == protected_hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    write_json(out / "summary.json", dict(selected_on_train=selected, prefix_checks=True,
                                         base_matched=True, active_unchanged=True,
                                         paired_intervals_unadjusted=intervals))
    print(table[["name", "start_mode", "cost", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
