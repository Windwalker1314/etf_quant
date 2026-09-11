from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import EXECUTION_VERSION, simulate
from .config import ROOT, fingerprint, write_json
from .data import TZ, Cache
from .factors import compute, evaluate
from .framework import engine_comparison
from .metrics import bootstrap_sharpe, performance, window_performance, yearly
from .strategy import target_weights


def run_research(cfg: dict, stress: bool = True) -> Path:
    cache = Cache()
    data = cache.load(cfg)
    digest = cache.snapshot_digest
    run_id = datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + "-" + fingerprint(cfg)[:6]
    output = ROOT / "outputs/research" / run_id
    output.mkdir(parents=True, exist_ok=True)
    # Persist protocol BEFORE inspecting results; subsequent runs retain all historical artifacts.
    write_json(
        output / "protocol.json",
        {
            "execution_version": EXECUTION_VERSION,
            "config": cfg,
            "config_hash": fingerprint(cfg),
            "data_sha256": digest,
            "oos_start": cfg["oos_start"],
            "parameter_selection": "none; fixed ex ante; sensitivity is diagnostic",
            "survivorship": "Present-day liquid ETF universe; not point-in-time universe discovery",
            "holdout": "Retrospective chronological holdout, not genuinely unseen prospective data",
            "corporate_actions": "Total-return approximation: factor changes reinvest into fractional units on ex-date",
            "liquidity": "Daily volume participation proxy; not an opening auction liquidity model",
        },
    )
    calendar_path = cache.root / "calendar.parquet"
    gaps = {}
    if calendar_path.exists():
        calendar = pd.read_parquet(calendar_path)
        sessions = pd.to_datetime(calendar.loc[calendar.is_open.astype(int) == 1, "cal_date"])
        for symbol, df in data.items():
            expected = set(sessions[(sessions >= df.date.min()) & (sessions <= df.date.max())])
            gaps[symbol] = [str(d.date()) for d in sorted(expected - set(df.date))]
    write_json(
        output / "data_quality.json",
        {
            "missing_sessions_by_asset": gaps,
            "policy": "missing bars: mark at last observed close for valuation only; no signal or execution on that symbol",
            "limitation": "Historical gaps may reflect suspension or source omission; not inferred as zero market return",
        },
    )
    print("Calculating causal AKQuant factors...", flush=True)
    weights, strategy_factors = target_weights(data, cfg)
    result = simulate(data, weights, cfg)
    print("Running fixed-allocation benchmark...", flush=True)
    base_weights, _ = target_weights(data, cfg, baseline=True)
    baseline = simulate(data, base_weights, cfg)
    result.equity["baseline"] = baseline.equity.equity
    # Investable raw-price ETF total-return benchmark; initialization cash held during warm-up.
    cn_cfg = copy.deepcopy(cfg)
    cn_weights = weights * 0
    cn_weights["510300.SH"] = 0.99
    cn_weights.loc[cn_weights.index < weights[weights.sum(axis=1) > 0].index.min()] = 0
    cn = simulate(data, cn_weights, cn_cfg)
    result.equity["cn_equity_benchmark"] = cn.equity.equity
    result.equity.to_parquet(output / "equity.parquet")
    result.trades.to_parquet(output / "trades.parquet", index=False)
    result.positions.to_parquet(output / "positions.parquet")
    result.events.to_parquet(output / "events.parquet", index=False)
    weights.to_parquet(output / "targets.parquet")
    strategy_factors.to_parquet(output / "strategy_factors.parquet", index=False)
    eq = result.equity.equity
    statistics = {
        "strategy": performance(eq, cfg["risk_free_rate"]),
        "baseline": performance(baseline.equity.equity, cfg["risk_free_rate"]),
        "oos": window_performance(eq, cfg["oos_start"], None, cfg["risk_free_rate"]),
        "bootstrap_sharpe": bootstrap_sharpe(eq, cfg["risk_free_rate"]),
        "trades": len(result.trades),
        "total_commission": float(result.trades.commission.sum()),
        "total_slippage": float(result.trades.slippage_cost.sum()),
        "average_exposure": float(result.equity.exposure.mean()),
        "run_id": run_id,
    }
    yearly(eq, cfg["risk_free_rate"]).to_csv(output / "yearly.csv", index=False)
    windows = [
        ("2015 equity crash", "2015-06-01", "2015-09-30"),
        ("2018 trade tensions", "2018-01-01", "2018-12-31"),
        ("2020 pandemic", "2020-01-01", "2020-04-30"),
        ("2022 tightening", "2022-01-01", "2022-12-31"),
        ("2024 equity shock", "2024-01-01", "2024-02-29"),
    ]
    first_all = max(df.date.iloc[min(cfg["min_history"] - 1, len(df) - 1)] for df in data.values())
    statistics["all_assets_seasoned"] = window_performance(
        eq, str(first_all.date()), None, cfg["risk_free_rate"]
    )
    statistics["stress_windows"] = {
        name: window_performance(eq, lo, hi, cfg["risk_free_rate"]) for name, lo, hi in windows
    }
    validation = []
    for year in range(2018, eq.index[-1].year + 1):
        cfg_fold = {**cfg, "backtest_start": f"{year}-01-01"}
        folded = simulate(data, weights, cfg_fold, end=f"{year}-12-31")
        validation.append(
            {
                "year": year,
                "kind": "fresh capital yearly forward fold, fixed parameters",
                **performance(folded.equity.equity, cfg["risk_free_rate"]),
            }
        )
    pd.DataFrame(validation).to_csv(output / "forward_folds.csv", index=False)
    experiments = []
    if stress:
        print("Checking costs and parameter perturbations (no selection)...", flush=True)
        for multiple in (1.0, 2.0, 3.0):
            run = result if multiple == 1 else simulate(data, weights, cfg, cost_multiplier=multiple)
            experiments.append(
                {"variant": f"cost_x{multiple:g}", **performance(run.equity.equity, cfg["risk_free_rate"])}
            )
        for windows in ([90, 180], [150, 300]):
            variant = {**cfg, "trend_windows": windows}
            w, _ = target_weights(data, variant)
            run = simulate(data, w, variant)
            experiments.append(
                {
                    "variant": f"trend_{windows[0]}_{windows[1]}",
                    **performance(run.equity.equity, cfg["risk_free_rate"]),
                }
            )
        for weekday in (0, 2):
            variant = {**cfg, "rebalance_weekday": weekday}
            run = simulate(data, weights, variant)
            experiments.append(
                {"variant": f"weekday_{weekday}", **performance(run.equity.equity, cfg["risk_free_rate"])}
            )
    pd.DataFrame(experiments).to_csv(output / "sensitivity.csv", index=False)
    print("Evaluating factors and independent native-engine replay...", flush=True)
    all_factors = compute(data)
    summaries, ic = evaluate(data, all_factors)
    all_factors.to_parquet(output / "factors.parquet", index=False)
    summaries.to_csv(output / "factor_summary.csv", index=False)
    ic.to_parquet(output / "factor_ic.parquet", index=False)
    native, compare = engine_comparison(data, weights, cfg)
    statistics["native_engine_check"] = native
    compare.to_csv(output / "native_comparison.csv")
    # This gate describes evidence, not a promise or an automatically deployed strategy.
    oos = statistics["oos"]
    statistics["research_gate"] = {
        "oos_sharpe_above_0_7": (oos.get("sharpe") or -999) >= 0.7,
        "full_drawdown_under_15pct": statistics["strategy"]["max_drawdown"] >= -0.15,
        "native_engine_verified": native["verified"],
        "status": "research_only; requires prospective paper observation",
    }
    write_json(output / "summary.json", statistics)
    from .reports import research_html

    research_html(output, cfg, statistics, result.equity, result.positions)
    write_json(ROOT / "outputs/latest_research.json", {"path": str(output), "run_id": run_id})
    print(f"Research saved: {output}", flush=True)
    return output
