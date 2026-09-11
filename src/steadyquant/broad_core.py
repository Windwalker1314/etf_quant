"""User-requested broad ETF extension, preserving total domestic/US equity budgets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .backtest import EXECUTION_VERSION, simulate
from .config import ROOT, load_config, write_json
from .data import TZ, Cache
from .framework import engine_comparison
from .metrics import bootstrap_sharpe, performance, window_performance, yearly
from .optimize import STRESS, render_report
from .refinement_report import choose_core
from .strategy import target_weights


def broad_configs() -> dict[str, dict]:
    original = {
        **load_config(),
        "model": "fixed",
        "name": "Simple Monthly Core",
        "rebalance_frequency": "monthly",
        "rebalance_band": 0.03,
        "backtest_start": "2015-01-05",
    }
    variants = {"fixed_core_monthly": original}
    for fraction in (0.30, 0.50):
        assets = [dict(a) for a in original["assets"]]
        for a in assets:
            if a["symbol"] in {"510300.SH", "513500.SH"}:
                a["weight"] *= 1 - fraction
        for code, name in [
            ("510500.SH", "中证500ETF"),
            ("159915.SZ", "创业板ETF"),
            ("588000.SH", "科创50ETF"),
        ]:
            assets.append(
                dict(symbol=code, name=name, kind="fund", bucket="cn_equity", weight=0.20 * fraction / 3)
            )
        assets.append(
            dict(symbol="513100.SH", name="纳指ETF", kind="fund", bucket="us_equity", weight=0.15 * fraction)
        )
        variants[f"broad_core_{int(fraction * 100)}"] = {
            **original,
            "name": f"Broad Monthly Core {int(fraction * 100)}",
            "assets": assets,
        }
    return variants


def run_broad_core() -> Path:
    configs = broad_configs()
    template = configs["broad_core_50"]
    cache = Cache()
    all_data = cache.load(template)
    folder = ROOT / "outputs/broad_core" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    previous = Path(json.loads((ROOT / "outputs/latest_candidate.json").read_text())["path"])
    prior_summary = json.loads((previous / "summary.json").read_text())
    prior_comparison = pd.read_parquet(previous / "comparison.parquet")
    v1 = prior_comparison.v1
    write_json(
        folder / "protocol.json",
        dict(
            execution_version=EXECUTION_VERSION,
            configs=configs,
            prior_run=str(previous),
            data_sha256=cache.snapshot_digest,
            trigger="User explicitly requested ChiNext, STAR and Nasdaq broad ETFs; ChiNext/Nasdaq were already in research universe, STAR50 newly added",
            allocation="Preserve CN equity20%, US equity15%, HK5%, gold20%, government bond30%, soybean10%; split30% or50% of CN/US budgets into other broad styles, not extra equity exposure",
            selection="Use pre2023 net Sharpe+0.25*min(Calmar,3), drawdown<=15%, worst year>=-5%, Sharpe>v1; if none pass retain simple monthly core",
            holdout="Retrospective; histories already seen. STAR ETF has actual data only since2020-11-16, wait240 observed bars; no synthetic prelisting history",
            schedule="First calendar Friday if traded; initial allocation any session; next-session open; same costs and execution ledger",
        ),
    )
    rows, equities, targets, results = [], pd.DataFrame(), {}, {}
    for name, cfg in configs.items():
        print(f"Broad-core comparison {name}...", flush=True)
        data = {a["symbol"]: all_data[a["symbol"]] for a in cfg["assets"]}
        w, _ = target_weights(data, cfg)
        result = simulate(data, w, cfg)
        targets[name], results[name] = w, result
        equities[name] = result.equity.equity
        p = folder / name
        p.mkdir()
        w.to_parquet(p / "targets.parquet")
        result.equity.to_parquet(p / "equity.parquet")
        result.trades.to_parquet(p / "trades.parquet", index=False)
        result.positions.to_parquet(p / "positions.parquet")
        yearly(result.equity.equity, cfg["risk_free_rate"]).to_csv(p / "yearly.csv", index=False)
        (p / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        if name.startswith("broad_core_"):
            (ROOT / "configs" / f"{name}.yaml").write_text((p / "config.yaml").read_text())
        for period, start, end in [
            ("full", "2015-01-01", None),
            ("selection", "2015-01-01", "2022-12-31"),
            ("test", "2023-01-01", None),
        ]:
            rows.append(
                dict(
                    candidate=name,
                    period=period,
                    **window_performance(result.equity.equity, start, end, cfg["risk_free_rate"]),
                )
            )
    pd.DataFrame(rows).to_csv(folder / "candidates.csv", index=False)
    chosen, selection = choose_core(equities, v1, "2015-01-01", "2022-12-31", 0.02)
    chosen = chosen or "fixed_core_monthly"
    write_json(folder / "selection.json", dict(selected=chosen, audit=selection))
    cfg = configs[chosen]
    data = {a["symbol"]: all_data[a["symbol"]] for a in cfg["assets"]}
    w = targets[chosen]
    (folder / "candidate.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"Chosen by pre2023 metrics: {chosen}; validating costs and execution...", flush=True)
    sensitivity = []
    for multiple in (2, 3):
        run = simulate(data, w, cfg, cost_multiplier=multiple)
        sensitivity.append(dict(variant=f"cost_x{multiple}", **performance(run.equity.equity, 0.02)))
    for weekday in (0, 2):
        run = simulate(data, w, {**cfg, "rebalance_weekday": weekday})
        sensitivity.append(dict(variant=f"weekday_{weekday}", **performance(run.equity.equity, 0.02)))
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    cutoff = pd.Timestamp("2022-12-30")
    prefix = {s: d[d.date <= cutoff] for s, d in data.items()}
    cut, _ = target_weights(prefix, cfg)
    pd.testing.assert_frame_equal(cut, w.loc[:cutoff])
    native, native_compare = engine_comparison(data, w, cfg)
    native_compare.to_csv(folder / "native_comparison.csv")
    profile_checks = {}
    for profile in ("broad_core_30", "broad_core_50"):
        profile_cfg = configs[profile]
        profile_data = {a["symbol"]: all_data[a["symbol"]] for a in profile_cfg["assets"]}
        checked, comparison = engine_comparison(profile_data, targets[profile], profile_cfg)
        profile_checks[profile] = checked
        comparison.to_csv(folder / profile / "native_comparison.csv")
    write_json(folder / "broad_profile_native_checks.json", profile_checks)
    annual = []
    for year in range(2018, w.index[-1].year + 1):
        r = simulate(data, w, cfg, start=f"{year}-01-01", end=f"{year}-12-31")
        annual.append(dict(year=year, **performance(r.equity.equity, 0.02)))
    pd.DataFrame(annual).to_csv(folder / "fresh_annual_folds.csv", index=False)
    compare = pd.concat([equities, v1], axis=1)
    compare.to_parquet(folder / "comparison.parquet")
    stress = {
        label: {key: window_performance(series.dropna(), lo, hi, 0.02) for key, series in compare.items()}
        for label, (lo, hi) in STRESS.items()
    }
    full = performance(equities[chosen], 0.02)
    test = window_performance(equities[chosen], "2023-01-01", None, 0.02)
    result = results[chosen]
    summary = dict(
        run_id=folder.name,
        selected=chosen,
        strategy=full,
        test_2023=test,
        v1_same_start=performance(v1, 0.02),
        v1_test=window_performance(v1, "2023-01-01", None, 0.02),
        native_engine_check=native,
        broad_profile_native_checks=profile_checks,
        stress_windows=stress,
        bootstrap_sharpe=bootstrap_sharpe(equities[chosen], 0.02),
        full_target_15pct_reached=full["cagr"] >= 0.15,
        test_target_15pct_reached=test["cagr"] >= 0.15,
        full_drawdown_pass=full["max_drawdown"] >= -0.15,
        test_drawdown_pass=test["max_drawdown"] >= -0.15,
        parameter_stress_drawdown_pass=all(r["max_drawdown"] >= -0.15 for r in sensitivity),
        trades=len(result.trades),
        total_commission=float(result.trades.commission.sum()),
        total_slippage=float(result.trades.slippage_cost.sum()),
        average_exposure=float(result.equity.exposure.mean()),
        real_prefix_targets_identical=True,
        data_snapshot_verified=True,
        universe_extension=True,
        requested_symbols=["159915.SZ", "588000.SH", "513100.SH"],
        prior_candidate=str(previous),
        prior_run=prior_summary["prior_run"],
        refinement=prior_summary["refinement"],
        status="Retrospective extended broad-ETF study; daily default remains v1; prospective performance unobserved",
    )
    write_json(folder / "summary.json", summary)
    render_report(folder, summary, compare, rows, sensitivity)
    (ROOT / "configs/candidate_v2.yaml").write_text((folder / "candidate.yaml").read_text())
    write_json(ROOT / "outputs/latest_candidate.json", dict(path=str(folder)))
    print(
        pd.DataFrame(rows)[["candidate", "period", "cagr", "sharpe", "max_drawdown"]].to_string(index=False),
        flush=True,
    )
    return folder
