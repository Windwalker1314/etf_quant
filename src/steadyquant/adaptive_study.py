"""Frozen, bounded second-stage study; all history is explicitly retrospective."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import yaml

from .adaptive import adaptive_weights
from .backtest import EXECUTION_VERSION, simulate
from .broad_core import broad_configs
from .config import ROOT, write_json
from .data import TZ, Cache
from .framework import engine_comparison
from .macro import MACRO_ROOT, load_macro
from .metrics import bootstrap_sharpe, performance, window_performance, yearly
from .optimize import STRESS, render_report
from .strategy import target_weights


def adaptive_configs():
    broad = broad_configs()
    base = {
        **broad["broad_core_50"],
        "model": "adaptive",
        "min_history": 253,
        "risk_method": "equal_risk",
        "risk_window": 126,
        "covariance_shrinkage": 0.30,
        "risk_mix": 1.0,
        "target_vol": 0.10,
        "trend_budget": False,
        "value_overlay": False,
        "rate_overlay": False,
    }
    changes = {
        "inverse_vol": dict(risk_method="inverse_vol"),
        "equal_risk": {},
        "trend_risk": dict(trend_budget=True),
        "blended_risk": dict(risk_mix=0.5),
        "value_risk": dict(risk_mix=0.5, value_overlay=True),
        "rate_risk": dict(risk_mix=0.5, rate_overlay=True),
        "value_rate_risk": dict(risk_mix=0.5, value_overlay=True, rate_overlay=True),
        "trend_value_rate": dict(risk_mix=0.5, trend_budget=True, value_overlay=True, rate_overlay=True),
    }
    return {
        "fixed_core_monthly": broad["fixed_core_monthly"],
        "broad_core_50": broad["broad_core_50"],
        **{name: {**base, **delta, "name": name} for name, delta in changes.items()},
    }


def select_adaptive(equities, start="2015-01-01", end="2022-12-31"):
    ref = window_performance(equities.fixed_core_monthly, start, end, 0.02)
    records = []
    for name, equity in equities.items():
        stats = window_performance(equity, start, end, 0.02)
        worst = float(yearly(equity.loc[start:end], 0.02).total_return.min())
        passed = (
            stats["observations"] >= 500
            and stats["max_drawdown"] >= -0.15
            and worst >= -0.05
            and stats["sharpe"] >= ref["sharpe"] + 0.05
            and stats["cagr"] >= ref["cagr"]
        )
        score = stats["sharpe"] + 0.25 * min(3, stats["calmar"])
        records.append(dict(candidate=name, passed=passed, worst_year=worst, score=score, **stats))
    passes = [r for r in records if r["passed"]]
    chosen = max(passes, key=lambda r: r["score"])["candidate"] if passes else "fixed_core_monthly"
    return chosen, records


def run_adaptive_study():
    configs = adaptive_configs()
    template = configs["equal_risk"]
    cache = Cache()
    data = cache.load(template)
    macro = load_macro()
    folder = ROOT / "outputs/adaptive" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    write_json(
        folder / "protocol.json",
        dict(
            execution_version=EXECUTION_VERSION,
            candidates=configs,
            data_sha256=cache.snapshot_digest,
            macro_manifest=json.loads((MACRO_ROOT / "manifest.json").read_text()),
            design="8 predeclared adaptive variants plus simple/broad fixed controls. No parameter grid or candidate addition after results.",
            selection="2015-2022 only; DD<=15%, worst calendar year>=-5%, Sharpe>=simple+0.05 and CAGR>=simple; score Sharpe+0.25*min(Calmar,3); fallback simple. 2023+ does not reselect.",
            history="All price history has been observed in earlier research; this is retrospective, not a new untouched holdout.",
            features="126-session group covariance, 30% diagonal shrinkage, equal-risk or inverse-vol group allocation; trend126/252, trailing1260/min252 PE/PB percentile, Shibor3m change63. Macro lag1 session, <=7 calendar day stale fill, missing is neutral.",
            constraints="No leverage; asset30%, CN35%, US30%, HK15%, gold30%, bond30%, commodity20%, combined equity60%; portfolio volatility cap10%; cash earns0.",
            limitations="Vendor historical fundamentals lack revision vintages; no stock-level profitability or credit factor claimed. ETF dividends approximated reinvestment; QDII premium/open liquidity incomplete.",
        ),
    )
    equities, targets, rows = pd.DataFrame(), {}, []
    for name, cfg in configs.items():
        print(f"Adaptive study: {name}", flush=True)
        subset = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
        w, f = (
            adaptive_weights(subset, cfg, macro)
            if cfg.get("model") == "adaptive"
            else target_weights(subset, cfg)
        )
        result = simulate(subset, w, cfg)
        targets[name] = w
        equities[name] = result.equity.equity
        p = folder / name
        p.mkdir()
        (p / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        for key, frame in dict(
            targets=w, factors=f, equity=result.equity, positions=result.positions, trades=result.trades
        ).items():
            frame.to_parquet(p / f"{key}.parquet")
        yearly(result.equity.equity, 0.02).to_csv(p / "yearly.csv", index=False)
        for label, lo, hi in [
            ("full", "2015-01-01", None),
            ("selection", "2015-01-01", "2022-12-31"),
            ("test", "2023-01-01", None),
        ]:
            rows.append(
                dict(candidate=name, period=label, **window_performance(result.equity.equity, lo, hi, 0.02))
            )
        pd.DataFrame(rows).to_csv(folder / "candidates.csv", index=False)
    equities.to_parquet(folder / "all_equities.parquet")
    chosen, audit = select_adaptive(equities)
    write_json(folder / "selection.json", dict(selected=chosen, audit=audit))
    print(f"Pre2023 selection: {chosen}; verifying candidate", flush=True)
    finish_adaptive(folder, configs, data, macro, equities, targets, chosen, rows)
    print(
        pd.DataFrame(rows)[["candidate", "period", "cagr", "sharpe", "max_drawdown"]].to_string(index=False),
        flush=True,
    )
    print(folder, flush=True)
    return folder


def finish_adaptive(folder, configs, data, macro, equities, targets, chosen, rows):
    cfg = configs[chosen]
    subset = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
    w = targets[chosen]
    cutoff = pd.Timestamp("2022-12-30")
    # Test every adaptive design, including failed ones, against an actual prefix.
    prefix = {s: d[d.date <= cutoff] for s, d in data.items()}
    macro_prefix = {s: d[d.date <= cutoff] for s, d in macro.items()}
    checks = {}
    for key, c in configs.items():
        if c.get("model") == "adaptive":
            cut, _ = adaptive_weights(prefix, c, macro_prefix)
            pd.testing.assert_frame_equal(cut, targets[key].loc[:cutoff])
            checks[key] = True
    sensitivity = []
    for mul in (2, 3):
        r = simulate(subset, w, cfg, cost_multiplier=mul)
        sensitivity.append(dict(variant=f"cost_x{mul}", **performance(r.equity.equity, 0.02)))
    for weekday in (0, 2):
        r = simulate(subset, w, {**cfg, "rebalance_weekday": weekday})
        sensitivity.append(dict(variant=f"weekday_{weekday}", **performance(r.equity.equity, 0.02)))
    if cfg.get("model") == "adaptive":
        for window in (84, 168):
            nw, _ = adaptive_weights(subset, {**cfg, "risk_window": window}, macro)
            r = simulate(subset, nw, cfg)
            sensitivity.append(dict(variant=f"risk_window_{window}", **performance(r.equity.equity, 0.02)))
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    native, compare_native = engine_comparison(subset, w, cfg)
    compare_native.to_csv(folder / "native_comparison.csv")
    annual = []
    for year in range(2018, w.index[-1].year + 1):
        r = simulate(subset, w, cfg, start=f"{year}-01-01", end=f"{year}-12-31")
        annual.append(dict(year=year, **performance(r.equity.equity, 0.02)))
    pd.DataFrame(annual).to_csv(folder / "fresh_annual_folds.csv", index=False)
    # A selector must only read past years; stitched targets are executed with one
    # continuous cash/position ledger, never concatenate idealized model returns.
    folds = []
    for year in range(2018, w.index[-1].year + 1):
        pick, _ = select_adaptive(equities, f"{year - 3}-01-01", f"{year - 1}-12-31")
        folds.append(dict(year=year, selected=pick, selection_through=f"{year - 1}-12-31"))
    # Use union of candidate assets even when chosen is the six-asset fallback.
    full_cfg = configs["equal_risk"]
    fw = targets["fixed_core_monthly"].reindex(columns=list(data), fill_value=0.0).astype(float)
    for fold in folds:
        mask = fw.index.year == fold["year"]
        fw.loc[mask] = targets[fold["selected"]].reindex(columns=fw.columns, fill_value=0).loc[mask]
    forward = simulate(data, fw, full_cfg, start="2018-01-01")
    forward.equity.to_parquet(folder / "forward_equity.parquet")
    fw.to_parquet(folder / "forward_targets.parquet")
    write_json(folder / "forward_selection.json", folds)
    compare = equities[[*dict.fromkeys([chosen, "fixed_core_monthly", "broad_core_50"])]].copy()
    compare["walk_forward"] = forward.equity.equity
    compare.to_parquet(folder / "comparison.parquet")
    stress = {
        label: {name: window_performance(series.dropna(), lo, hi, 0.02) for name, series in compare.items()}
        for label, (lo, hi) in STRESS.items()
    }
    full = performance(equities[chosen], 0.02)
    test = window_performance(equities[chosen], "2023-01-01", None, 0.02)
    baseline_test = window_performance(equities.fixed_core_monthly, "2023-01-01", None, 0.02)
    # This is acceptance, not a second chance to choose a different historical winner.
    accepted = (
        chosen != "fixed_core_monthly"
        and native["verified"]
        and full["max_drawdown"] >= -0.15
        and test["max_drawdown"] >= -0.15
        and test["sharpe"] >= baseline_test["sharpe"]
        and all(r["max_drawdown"] >= -0.15 for r in sensitivity)
    )
    summary = dict(
        selected=chosen,
        strategy=full,
        test_2023=test,
        adaptive_study=True,
        accepted_for_paper=accepted,
        v1_same_start=performance(equities.fixed_core_monthly, 0.02),
        v1_test=baseline_test,
        stress_windows=stress,
        native_engine_check=native,
        bootstrap_sharpe=bootstrap_sharpe(equities[chosen], 0.02),
        walk_forward=performance(forward.equity.equity, 0.02),
        walk_forward_test=window_performance(forward.equity.equity, "2023-01-01", None, 0.02),
        prefix_checks=checks,
        full_drawdown_pass=full["max_drawdown"] >= -0.15,
        test_drawdown_pass=test["max_drawdown"] >= -0.15,
        full_target_15pct_reached=full["cagr"] >= 0.15,
        test_target_15pct_reached=test["cagr"] >= 0.15,
        status="Retrospective research, no daily activation. Acceptance only qualifies a paper candidate, not future performance.",
    )
    write_json(folder / "summary.json", summary)
    (folder / "candidate.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    (ROOT / "configs/adaptive_candidate.yaml").write_text((folder / "candidate.yaml").read_text())
    render_report(folder, summary, compare, rows, sensitivity)
    write_json(ROOT / "outputs/latest_adaptive.json", dict(path=str(folder)))
