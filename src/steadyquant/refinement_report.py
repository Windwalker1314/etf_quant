"""Risk-first selection, validation and report for the second retained experiment batch."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .backtest import simulate
from .config import ROOT, write_json
from .data import Cache
from .framework import engine_comparison
from .metrics import bootstrap_sharpe, performance, window_performance, yearly
from .optimize import STRESS, render_report
from .strategy import target_weights


def choose_core(
    equities: pd.DataFrame, baseline: pd.Series, start: str, end: str, rf: float
) -> tuple[str | None, list]:
    baseline_sharpe = window_performance(baseline, start, end, rf).get("sharpe")
    records = []
    for name, equity in equities.items():
        stats = window_performance(equity, start, end, rf)
        yy = yearly(equity.loc[start:end], rf)
        worst_year = float(yy.total_return.min())
        valid = (
            stats.get("observations", 0) >= 500
            and stats.get("max_drawdown", -1) >= -0.15
            and worst_year >= -0.05
            and stats.get("sharpe", -99) > (baseline_sharpe if baseline_sharpe is not None else -99)
        )
        score = (stats.get("sharpe") or -99) + 0.25 * min(3, stats.get("calmar") or -99)
        records.append(dict(candidate=name, passed=valid, score=score, worst_year=worst_year, **stats))
    passing = [r for r in records if r["passed"]]
    return (max(passing, key=lambda r: r["score"])["candidate"] if passing else None), records


def finish_refinement(folder: Path) -> dict:
    protocol = json.loads((folder / "protocol.json").read_text())
    source = Path(protocol["prior_run"])
    candidates = protocol["candidates"]
    template = next(iter(candidates.values()))
    cache = Cache()
    data = cache.load(template)
    if cache.snapshot_digest != protocol["data_sha256"]:
        raise ValueError("Cache changed since refinement was frozen")
    first_protocol = json.loads((source / "protocol.json").read_text())
    if first_protocol["data_sha256"] != cache.snapshot_digest:
        raise ValueError("Refinement and first-stage snapshots differ")
    equities = pd.DataFrame(
        {name: pd.read_parquet(folder / name / "equity.parquet").equity for name in candidates}
    )
    baseline = pd.read_parquet(source / "comparison.parquet").v1
    selected, audit = choose_core(equities, baseline, "2015-01-01", "2022-12-31", template["risk_free_rate"])
    write_json(folder / "selection.json", dict(selected=selected, audit=audit))
    if selected is None:
        write_json(folder / "summary.json", {"selected": None, "status": "No refinement passed; retain v1"})
        return {"selected": None}
    cfg = candidates[selected]
    (folder / "candidate.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    weights = pd.read_parquet(folder / selected / "targets.parquet")
    # Verify persisted targets match the production dispatch and a genuine historical prefix.
    recalculated, factors = target_weights(data, cfg)
    pd.testing.assert_frame_equal(recalculated, weights)
    prefix = {s: d[d.date <= pd.Timestamp("2022-12-30")] for s, d in data.items()}
    cut, _ = target_weights(prefix, cfg)
    pd.testing.assert_frame_equal(cut, weights.loc[:"2022-12-30"])
    factors.to_parquet(folder / "factors.parquet", index=False)
    print(f"Risk-first refinement selected: {selected}; running forward and cost audits...", flush=True)
    core_cfg = cfg["core_config"]
    core_data = {a["symbol"]: data[a["symbol"]] for a in core_cfg["assets"]}
    core_weights, _ = target_weights(core_data, core_cfg)
    forward_weights = core_weights.reindex(index=weights.index, columns=weights.columns, fill_value=0)
    forward_weights = forward_weights.astype(float)
    schedules, fold_records = {}, []
    for year in range(2018, weights.index[-1].year + 1):
        chosen, fold_audit = choose_core(
            equities, baseline, f"{year - 3}-01-01", f"{year - 1}-12-31", cfg["risk_free_rate"]
        )
        mask = forward_weights.index.year == year
        if chosen:
            forward_weights.loc[mask] = pd.read_parquet(folder / chosen / "targets.parquet").loc[mask]
            schedules[str(year)] = candidates[chosen]["rebalance_frequency"]
        else:
            schedules[str(year)] = "weekly"
        fold_records.append(
            dict(
                year=year,
                selected=chosen or "v1_fallback",
                selection_through=f"{year - 1}-12-31",
                audit=fold_audit,
            )
        )
    forward_cfg = {**cfg, "schedule_by_year": schedules}
    forward = simulate(data, forward_weights, forward_cfg, start="2018-01-01")
    forward.equity.to_parquet(folder / "forward_equity.parquet")
    forward_weights.to_parquet(folder / "forward_targets.parquet")
    forward.trades.to_parquet(folder / "forward_trades.parquet", index=False)
    write_json(folder / "forward_selection.json", fold_records)
    equity = equities[selected]
    compare = pd.concat(
        [equity.rename(selected), baseline, forward.equity.equity.rename("walk_forward")], axis=1
    )
    compare.to_parquet(folder / "comparison.parquet")
    sensitivity = []
    for multiple in (2, 3):
        run = simulate(data, weights, cfg, cost_multiplier=multiple)
        sensitivity.append(
            dict(variant=f"cost_x{multiple}", **performance(run.equity.equity, cfg["risk_free_rate"]))
        )
    for weekday in (0, 2):
        run = simulate(data, weights, {**cfg, "rebalance_weekday": weekday})
        sensitivity.append(
            dict(variant=f"weekday_{weekday}", **performance(run.equity.equity, cfg["risk_free_rate"]))
        )
    # Adjacent satellite weights are diagnostic only; never re-select after seeing them.
    satellite = pd.read_parquet(source / "quality_v12/targets.parquet")
    base, _ = target_weights(core_data, core_cfg, baseline=cfg["core_style"] == "fixed")
    base = base.reindex(index=weights.index, columns=weights.columns, fill_value=0)
    core_only = simulate(data, base, cfg)
    core_only.equity.to_parquet(folder / "core_only_equity.parquet")
    for fraction in (0.20, 0.30):
        w = base * (1 - fraction) + satellite * fraction
        run = simulate(data, w, cfg)
        sensitivity.append(
            dict(variant=f"satellite_{fraction}", **performance(run.equity.equity, cfg["risk_free_rate"]))
        )
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    native, native_compare = engine_comparison(data, weights, cfg)
    native_compare.to_csv(folder / "native_comparison.csv")
    stress = {
        name: {
            label: window_performance(series.dropna(), lo, hi, cfg["risk_free_rate"])
            for label, series in compare.items()
        }
        for name, (lo, hi) in STRESS.items()
    }
    full = performance(equity, cfg["risk_free_rate"])
    test = window_performance(equity, "2023-01-01", None, cfg["risk_free_rate"])
    result = pd.read_parquet(folder / selected / "equity.parquet")
    trades = pd.read_parquet(folder / selected / "trades.parquet")
    summary = dict(
        run_id=folder.name,
        selected=selected,
        strategy=full,
        test_2023=test,
        v1_same_start=performance(baseline, cfg["risk_free_rate"]),
        v1_test=window_performance(baseline, "2023-01-01", None, cfg["risk_free_rate"]),
        core_only_same_schedule=performance(core_only.equity.equity, cfg["risk_free_rate"]),
        walk_forward=performance(forward.equity.equity, cfg["risk_free_rate"]),
        walk_forward_test=window_performance(
            forward.equity.equity, "2023-01-01", None, cfg["risk_free_rate"]
        ),
        stress_windows=stress,
        bootstrap_sharpe=bootstrap_sharpe(equity, cfg["risk_free_rate"]),
        native_engine_check=native,
        trades=len(trades),
        total_commission=float(trades.commission.sum()),
        total_slippage=float(trades.slippage_cost.sum()),
        average_exposure=float(result.exposure.mean()),
        full_target_15pct_reached=full["cagr"] >= 0.15,
        test_target_15pct_reached=test["cagr"] >= 0.15,
        selection_drawdown_pass=True,
        full_drawdown_pass=full["max_drawdown"] >= -0.15,
        test_drawdown_pass=test["max_drawdown"] >= -0.15,
        cost_stress_drawdown_pass=all(
            r["max_drawdown"] >= -0.15 for r in sensitivity if r["variant"].startswith("cost_")
        ),
        parameter_stress_drawdown_pass=all(r["max_drawdown"] >= -0.15 for r in sensitivity),
        bear_calendar_years_nonnegative=all(
            stress[w][selected]["total_return"] >= 0 for w in ("2018 熊市", "2022 股债压力")
        ),
        data_and_causality_audit=dict(
            snapshot_verified=True,
            dispatch_targets_identical=True,
            real_prefix_targets_identical=True,
            cutoff="2022-12-30",
        ),
        prior_run=str(source),
        total_candidates_across_rounds=16,
        status="Retrospective research candidate. Daily default remains v1. No claim of unseen or prospective performance.",
    )
    write_json(folder / "summary.json", summary)
    records = pd.read_csv(folder / "candidates.csv").to_dict("records")
    render_report(folder, summary, compare, records, sensitivity)
    # Convenient reproducible candidate; saving is not activation.
    (ROOT / "configs/candidate_v2.yaml").write_text((folder / "candidate.yaml").read_text())
    return summary
