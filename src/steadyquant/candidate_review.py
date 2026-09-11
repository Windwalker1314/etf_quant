"""Transparent simplicity review after the retained research batches, not an unseen test."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .backtest import EXECUTION_VERSION, simulate
from .config import ROOT, write_json
from .data import TZ, Cache
from .framework import engine_comparison
from .metrics import bootstrap_sharpe, performance, window_performance, yearly
from .optimize import STRESS, render_report
from .strategy import target_weights


def review_simple_candidate(refinement: Path) -> Path:
    source_protocol = json.loads((refinement / "protocol.json").read_text())
    source_summary = json.loads((refinement / "summary.json").read_text())
    chosen = source_protocol["candidates"][source_summary["selected"]]
    cfg = {
        **chosen["core_config"],
        "model": "fixed",
        "name": "Steady Allocation v2 · Monthly Candidate",
        "rebalance_frequency": "monthly",
        "rebalance_band": 0.03,
        "backtest_start": "2015-01-05",
    }
    root = ROOT / "outputs/candidate" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True)
    data = Cache().load(cfg)
    digest = Cache().digest(cfg)
    if any(digest[s] != source_protocol["data_sha256"][s] for s in digest):
        raise ValueError("Candidate review snapshot changed")
    write_json(
        root / "protocol.json",
        dict(
            execution_version=EXECUTION_VERSION,
            config=cfg,
            data_sha256=digest,
            refinement=str(refinement),
            prior_run=source_protocol["prior_run"],
            stage="post-hoc simplicity review; all historical results have been observed",
            decision="Compare the zero-satellite ablation explicitly: lower complexity, lower historical drawdown and higher Sharpe than 25% satellite; not a claim of ex-ante selection or unseen validation",
            qualification="15% drawdown research limit, no leverage, no guaranteed return or future drawdown cap",
            schedule="First calendar Friday of each month if traded; otherwise skip that month. Initial allocation can occur any session. Next-session open fills.",
            tests="cost2x/3x; Monday/Wednesday calendar day; 2%/4% rebalance band; fresh annual capital; native engine; historical prefix",
        ),
    )
    weights, factors = target_weights(data, cfg)
    result = simulate(data, weights, cfg)
    expected = pd.read_parquet(refinement / "core_only_equity.parquet").equity
    if chosen["rebalance_frequency"] == "monthly" and chosen["core_style"] == "fixed":
        pd.testing.assert_series_equal(result.equity.equity, expected)
    name = "fixed_core_monthly"
    folder = root / name
    folder.mkdir()
    weights.to_parquet(folder / "targets.parquet")
    result.equity.to_parquet(folder / "equity.parquet")
    result.trades.to_parquet(folder / "trades.parquet", index=False)
    result.positions.to_parquet(folder / "positions.parquet")
    factors.to_parquet(root / "factors.parquet", index=False)
    yearly(result.equity.equity, cfg["risk_free_rate"]).to_csv(folder / "yearly.csv", index=False)
    (root / "candidate.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    prefix = {s: d[d.date <= pd.Timestamp("2022-12-30")] for s, d in data.items()}
    cut, _ = target_weights(prefix, cfg)
    pd.testing.assert_frame_equal(cut, weights.loc[:"2022-12-30"])
    sensitivity = []
    for multiple in (2, 3):
        run = simulate(data, weights, cfg, cost_multiplier=multiple)
        sensitivity.append(
            dict(variant=f"cost_x{multiple}", **performance(run.equity.equity, cfg["risk_free_rate"]))
        )
    for key, values in [("rebalance_weekday", (0, 2)), ("rebalance_band", (0.02, 0.04))]:
        for value in values:
            run = simulate(data, weights, {**cfg, key: value})
            sensitivity.append(
                dict(variant=f"{key}_{value}", **performance(run.equity.equity, cfg["risk_free_rate"]))
            )
    pd.DataFrame(sensitivity).to_csv(root / "sensitivity.csv", index=False)
    annual = []
    for year in range(2018, weights.index[-1].year + 1):
        run = simulate(data, weights, cfg, start=f"{year}-01-01", end=f"{year}-12-31")
        annual.append(dict(year=year, **performance(run.equity.equity, cfg["risk_free_rate"])))
    pd.DataFrame(annual).to_csv(root / "fresh_annual_folds.csv", index=False)
    native, native_compare = engine_comparison(data, weights, cfg)
    native_compare.to_csv(root / "native_comparison.csv")
    v1 = pd.read_parquet(refinement / "comparison.parquet").v1
    enhanced = pd.read_parquet(refinement / source_summary["selected"] / "equity.parquet").equity
    forward = pd.read_parquet(refinement / "forward_equity.parquet").equity
    compare = pd.concat(
        [
            result.equity.equity.rename(name),
            v1,
            enhanced.rename("25pct_satellite"),
            forward.rename("walk_forward"),
        ],
        axis=1,
    )
    compare.to_parquet(root / "comparison.parquet")
    records = pd.concat(
        [
            pd.read_csv(Path(source_protocol["prior_run"]) / "candidates.csv"),
            pd.read_csv(refinement / "candidates.csv"),
        ]
    ).to_dict("records")
    for label, start, end in [
        ("full", "2015-01-01", None),
        ("selection", "2015-01-01", "2022-12-31"),
        ("test", "2023-01-01", None),
    ]:
        records.append(
            dict(
                candidate=name,
                period=label,
                **window_performance(result.equity.equity, start, end, cfg["risk_free_rate"]),
            )
        )
    pd.DataFrame(records).to_csv(root / "candidates.csv", index=False)
    full = performance(result.equity.equity, cfg["risk_free_rate"])
    test = window_performance(result.equity.equity, "2023-01-01", None, cfg["risk_free_rate"])
    stress = {
        label: {
            key: window_performance(series.dropna(), lo, hi, cfg["risk_free_rate"])
            for key, series in compare.items()
        }
        for label, (lo, hi) in STRESS.items()
    }
    summary = dict(
        run_id=root.name,
        selected=name,
        strategy=full,
        test_2023=test,
        v1_same_start=performance(v1, cfg["risk_free_rate"]),
        v1_test=window_performance(v1, "2023-01-01", None, cfg["risk_free_rate"]),
        walk_forward=source_summary["walk_forward"],
        walk_forward_test=source_summary["walk_forward_test"],
        walk_forward_note="This is the previous-stage adaptive selector, not an unseen test of the final post-hoc simple candidate",
        native_engine_check=native,
        stress_windows=stress,
        bootstrap_sharpe=bootstrap_sharpe(result.equity.equity, cfg["risk_free_rate"]),
        full_target_15pct_reached=full["cagr"] >= 0.15,
        test_target_15pct_reached=test["cagr"] >= 0.15,
        full_drawdown_pass=full["max_drawdown"] >= -0.15,
        test_drawdown_pass=test["max_drawdown"] >= -0.15,
        parameter_stress_drawdown_pass=all(r["max_drawdown"] >= -0.15 for r in sensitivity),
        trades=len(result.trades),
        total_commission=float(result.trades.commission.sum()),
        total_slippage=float(result.trades.slippage_cost.sum()),
        average_exposure=float(result.equity.exposure.mean()),
        refinement=str(refinement),
        prior_run=source_protocol["prior_run"],
        total_candidates_across_rounds=17,
        simplicity_review=True,
        real_prefix_targets_identical=True,
        data_snapshot_verified=True,
        status="Prioritize prospective observation of simple monthly allocation; daily v1 remains active; final selection was a retrospective review.",
    )
    write_json(root / "summary.json", summary)
    render_report(root, summary, compare, records, sensitivity)
    (ROOT / "configs/candidate_v2.yaml").write_text((root / "candidate.yaml").read_text())
    write_json(ROOT / "outputs/latest_candidate.json", dict(path=str(root)))
    return root
