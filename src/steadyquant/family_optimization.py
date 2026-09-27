"""Retrospective, research-only comparisons for the family ETF strategy."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive import GROUP_CAPS, capped_proportions
from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, write_json
from .data import TZ, Cache
from .metrics import window_performance
from .strategy import target_weights


def cap_hk_and_redistribute(weights: pd.DataFrame, cfg: dict, cap: float = 0.05) -> pd.DataFrame:
    """Cap the HK ETF and give excess only to already eligible CN/US ETFs."""
    hk = next(a["symbol"] for a in cfg["assets"] if a["bucket"] == "hk_equity")
    groups = {
        bucket: [a["symbol"] for a in cfg["assets"] if a["bucket"] == bucket]
        for bucket in ("cn_equity", "us_equity")
    }
    priors = pd.Series({a["symbol"]: a["weight"] for a in cfg["assets"]})
    budgets = {bucket: float(priors[symbols].sum()) for bucket, symbols in groups.items()}
    total_budget = sum(budgets.values())
    result = weights.copy()
    excess = (result[hk] - cap).clip(lower=0)
    result[hk] -= excess
    for date in result.index:
        remaining = float(excess.loc[date])
        if remaining <= 0:
            continue
        # The second pass uses spare capacity left by the preferred 20:15 CN/US split.
        for first_pass in (True, False):
            for bucket, symbols in groups.items():
                current = result.loc[date, symbols]
                room_by_etf = (0.30 - current).clip(lower=0)
                active = priors[symbols].where(weights.loc[date, symbols] > 0, 0)
                group_room = max(0.0, min(
                    GROUP_CAPS[bucket] - float(current.sum()),
                    float(room_by_etf[active > 0].sum()),
                ))
                budget = min(remaining * budgets[bucket] / total_budget, group_room) if first_pass else min(remaining, group_room)
                if budget <= 1e-12 or active.sum() <= 0:
                    continue
                added = capped_proportions(active.to_numpy(), room_by_etf.to_numpy(), budget)
                result.loc[date, symbols] += added
                remaining -= float(np.sum(added))
    if result.lt(-1e-10).any().any() or result.gt(0.30 + 1e-9).any().any():
        raise AssertionError("ETF concentration invariant failed")
    if result.sum(axis=1).gt(1 + 1e-9).any():
        raise AssertionError("No-leverage invariant failed")
    if any(result[symbols].sum(axis=1).gt(GROUP_CAPS[bucket] + 1e-9).any()
           for bucket, symbols in groups.items()):
        raise AssertionError("Equity group concentration invariant failed")
    return result


def run_family_optimization() -> Path:
    """Record the complete small candidate set; never change active policy."""
    from .config import load_active_config

    cfg = load_active_config()
    cache = Cache()
    data = cache.load(cfg)
    adaptive, _ = target_weights(data, cfg)
    fixed, _ = target_weights(data, cfg, baseline=True)
    trend, _ = target_weights(data, {**cfg, "trend_budget": True})
    blended, _ = target_weights(data, {**cfg, "risk_mix": 0.5})
    fixed_trend, _ = target_weights(data, {**cfg, "model": "steady"})
    candidates = {
        "current_adaptive": adaptive,
        "fixed": fixed,
        "half_adaptive_half_fixed": 0.5 * adaptive + 0.5 * fixed,
        "adaptive_trend_budget": trend,
        "adaptive_blended_risk": blended,
        "fixed_with_trend_guard": fixed_trend,
        "hk5_reallocated_posthoc": cap_hk_and_redistribute(adaptive, cfg),
    }
    # A truncated data run must reproduce every historical target through the cutoff.
    cutoff = pd.Timestamp("2022-12-30")
    prefix = {symbol: frame.loc[frame.date <= cutoff] for symbol, frame in data.items()}
    prefix_adaptive, _ = target_weights(prefix, cfg)
    pd.testing.assert_frame_equal(prefix_adaptive, adaptive.loc[:cutoff])
    pd.testing.assert_frame_equal(
        cap_hk_and_redistribute(prefix_adaptive, cfg), candidates["hk5_reallocated_posthoc"].loc[:cutoff]
    )

    periods = {
        "2015-2018": ("2015-01-01", "2018-12-31"),
        "2019-2022": ("2019-01-01", "2022-12-31"),
        "2023-now": ("2023-01-01", None),
        "2019-now": ("2019-01-01", None),
    }
    rows, equities = [], {}
    for name, weights in candidates.items():
        result = simulate(data, weights, cfg)
        equities[name] = result.equity.equity
        for period, (start, end) in periods.items():
            stats = window_performance(result.equity.equity, start, end, cfg["risk_free_rate"])
            rows.append({"candidate": name, "period": period, "trades": len(result.trades), **stats})
    doubled_cost = {}
    for name in ("current_adaptive", "fixed", "hk5_reallocated_posthoc"):
        result = simulate(data, candidates[name], cfg, cost_multiplier=2)
        doubled_cost[name] = {
            period: window_performance(result.equity.equity, start, end, cfg["risk_free_rate"])
            for period, (start, end) in periods.items() if period in {"2019-now", "2023-now"}
        }
    intervals = {
        benchmark: {
            period: paired_sharpe_interval(
                equities["hk5_reallocated_posthoc"].loc[start:end], equities[benchmark].loc[start:end]
            )
            for period, (start, end) in periods.items() if period in {"2019-now", "2023-now"}
        }
        for benchmark in ("current_adaptive", "fixed")
    }
    output = ROOT / "outputs/family_optimization" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    write_json(output / "summary.json", {
        "data_sha256": cache.snapshot_digest,
        "periods": periods,
        "doubled_cost": doubled_cost,
        "paired_sharpe_intervals": intervals,
        "prefix_checks": True,
        "decision": "research_only_no_activation",
        "reason": "HK cap was proposed after seeing 2019+ returns, loses in 2015-2018, and has no untouched holdout.",
        "candidate_scope": list(candidates),
    })
    return output
