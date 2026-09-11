"""Frozen alternative-factor study using the audited stock/ETF execution ledger."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime

import pandas as pd

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .alternative_data import ALT_ROOT, load_alternative
from .alternative_evaluation import factor_tests, portfolio_targets
from .alternative_panel import build_panel
from .composite_ledger import VERSION, simulate_composite
from .composite_study import independent_replay, quarantine_stock_placeholders
from .config import ROOT, load_config, write_json
from .data import Cache
from .metrics import performance, window_performance, yearly
from .stock_data import load_stock_snapshot
from .stock_factors import build_stock_factors, composite_targets

CANDIDATES = ["revision", "surprise", "cash_quality", "balanced"]
NAMES = {
    "etf_core": "自适应ETF研究底仓",
    "revision": "盈利预期修正",
    "surprise": "业绩新增信息",
    "cash_quality": "现金与经营质量",
    "balanced": "三组因子均衡",
    "existing_momentum_defensive_25": "原动量防御25%",
    "matched_price_control": "同行业约束量价对照",
}
NAMES.update({f"{n}_index_control": f"{NAMES[n]}·等仓位指数对照" for n in CANDIDATES})


def prefix_check(snapshot, raw, dates, panel, cut="2024-12-31"):
    cutoff = pd.Timestamp(cut)
    truncated = {}
    for name, d in raw.items():
        if name == "analyst":
            mask = (pd.to_datetime(d.report_date, errors="coerce") <= cutoff) & (
                pd.to_datetime(d.create_time, errors="coerce") <= cutoff + pd.Timedelta(days=1)
            )
        elif "ann_date" in d:
            mask = pd.to_datetime(d.ann_date, errors="coerce") <= cutoff
            if "f_ann_date" in d:
                mask &= pd.to_datetime(d.f_ann_date, errors="coerce") <= cutoff
        elif name == "industry":
            mask = pd.to_datetime(d.in_date, errors="coerce") <= cutoff
        else:
            mask = pd.Series(True, index=d.index)
        truncated[name] = d.loc[mask].copy()
    before = {}
    for name, d in snapshot.items():
        col = "trade_date" if "trade_date" in d else None
        before[name] = d[pd.to_datetime(d[col]) <= cutoff] if col else d
    shorter, _, _, _, _ = build_panel(before, truncated, dates[dates <= cutoff])
    original = panel[panel.date <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(original, shorter.reset_index(drop=True))
    return dict(
        cut=cut,
        verified=True,
        rows=len(shorter),
        note="Future-dated inputs removed and all factors recomputed; vendor historical restatement authenticity remains a separate limitation",
    )


def result_stats(result, etf_data, snapshot, cfg):
    replay = independent_replay(result, etf_data, snapshot, cfg)
    if not replay["verified"]:
        raise AssertionError("Alternative portfolio ledger replay failed")
    blocks = int(result.events.type.eq("qualification_block").sum()) if not result.events.empty else 0
    costs = (
        {
            key: float(result.trades[key].sum())
            for key in ["commission", "stamp_tax", "transfer_fee", "slippage_cost"]
        }
        if not result.trades.empty
        else {}
    )
    return dict(
        full=performance(result.equity.equity),
        early=window_performance(result.equity.equity, "2023-01-01", "2024-12-31", 0.02),
        later=window_performance(result.equity.equity, "2025-01-01", None, 0.02),
        replay=replay,
        qualification_blocks=blocks,
        trades=len(result.trades),
        costs=costs,
        average_stock_exposure=float(result.equity.stock_exposure.mean()),
    )


def run_alternative():
    cfg = {**load_config(ROOT / "configs/adaptive_paper.yaml"), "initial_cash": 200000}
    active_path = ROOT / "configs/active.yaml"
    active_hash = hashlib.sha256(active_path.read_bytes()).hexdigest()
    source_protocol = ROOT / "configs/alternative_research_protocol.json"
    protocol = json.loads(source_protocol.read_text())
    protocol_hash = hashlib.sha256(source_protocol.read_bytes()).hexdigest()
    raw, source_manifest = load_alternative()
    snapshot, stock_manifest = load_stock_snapshot()
    folder = ROOT / "outputs/alternative" / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    snapshot, quarantine_count = quarantine_stock_placeholders(snapshot, folder)
    cache = Cache()
    etf_data = cache.load(cfg)
    etf_data = {s: d[d.date <= pd.Timestamp(protocol["end"])].copy() for s, d in etf_data.items()}
    etf_w, _ = adaptive_weights(etf_data, cfg)
    write_json(
        folder / "protocol.json",
        dict(
            **protocol,
            source_protocol_sha256=protocol_hash,
            alternative_manifest=source_manifest,
            stock_manifest=stock_manifest,
            etf_data_sha256=cache.snapshot_digest,
            execution_version=VERSION,
            active_config_sha256=active_hash,
            cfg=cfg,
            quarantined_placeholders=quarantine_count,
        ),
    )
    source_dir = folder / "source"
    source_dir.mkdir()
    sources = list((ROOT / "src/steadyquant").glob("alternative_*.py")) + [
        ROOT / "src/steadyquant/composite_ledger.py",
        source_protocol,
    ]
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    write_json(
        folder / "implementation.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    )
    print(f"Study output: {folder}", flush=True)
    print("Building dated factor panel", flush=True)
    panel, coverage, market, events, data_audit = build_panel(snapshot, raw, etf_w.index)
    panel.to_parquet(folder / "factors.parquet", index=False)
    coverage.to_csv(folder / "coverage.csv", index=False)
    events.to_parquet(folder / "disclosure_events.parquet", index=False)
    write_json(folder / "data_audit.json", data_audit)
    print("Factor panel ready; checking future-input truncation", flush=True)
    write_json(folder / "prefix_check.json", prefix_check(snapshot, raw, etf_w.index, panel))
    monthly, factor_summary = factor_tests(panel, market, snapshot)
    monthly.to_parquet(folder / "factor_monthly.parquet", index=False)
    factor_summary.to_csv(folder / "factor_summary.csv", index=False)
    print("Factor diagnostics saved; starting fixed portfolio comparisons", flush=True)
    weights = {"etf_core": etf_w}
    selection_audits = {}
    for family in CANDIDATES + ["matched_price_control"]:
        w, picks, audit = portfolio_targets(panel, etf_w, market["price"], etf_data, family)
        weights[family] = w
        dest = folder / family
        dest.mkdir()
        picks.to_csv(dest / "selection.csv", index=False)
        audit.to_csv(dest / "selection_audit.csv", index=False)
        selection_audits[family] = audit
        if family in CANDIDATES:
            budget = w.drop(columns=etf_w.columns).sum(axis=1)
            control = etf_w.mul(1 - budget, axis=0)
            control["510300.SH"] += budget
            weights[f"{family}_index_control"] = control
    prior_factors, _, prior_prices, _ = build_stock_factors(snapshot, etf_w.index)
    prior_w, _, _ = composite_targets(
        prior_factors, etf_w, prior_prices, etf_data, "momentum_defensive", 0.25
    )
    weights["existing_momentum_defensive_25"] = prior_w
    details, curves, metric_rows, runs = {}, {}, [], {}
    for name, w in weights.items():
        r = simulate_composite(etf_data, snapshot, w, cfg, start="2023-01-01")
        runs[name] = r
        dest = folder / name
        dest.mkdir(exist_ok=True)
        for key, frame in dict(
            equity=r.equity, trades=r.trades, positions=r.positions, events=r.events, targets=w
        ).items():
            frame.to_parquet(dest / f"{key}.parquet")
        yearly(r.equity.equity, 0.02).to_csv(dest / "yearly.csv", index=False)
        details[name] = result_stats(r, etf_data, snapshot, cfg)
        curves[name] = r.equity.equity
        for period in ("full", "early", "later"):
            metric_rows.append(dict(model=name, name=NAMES[name], period=period, **details[name][period]))
        print(name, json.dumps(details[name]["full"], ensure_ascii=False), flush=True)
    comparison = pd.DataFrame(curves)
    comparison.to_parquet(folder / "comparison.parquet")
    pd.DataFrame(metric_rows).to_csv(folder / "comparison.csv", index=False)
    qualified = []
    for name in CANDIDATES:
        detail, base = details[name], details["etf_core"]
        interval = paired_sharpe_interval(curves[name], curves["etf_core"])
        detail["paired_sharpe_interval"] = interval
        audit = selection_audits[name]
        available = audit[audit.date >= pd.Timestamp("2023-01-01")]
        adequate = float(available.selected.eq(8).mean())
        detail["complete_selection_fraction"] = adequate
        detail["gates"] = dict(
            full_cagr=bool(detail["full"]["cagr"] >= base["full"]["cagr"]),
            full_sharpe=bool(detail["full"]["sharpe"] >= base["full"]["sharpe"] + 0.05),
            later_cagr=bool(detail["later"]["cagr"] >= base["later"]["cagr"]),
            later_sharpe=bool(detail["later"]["sharpe"] >= base["later"]["sharpe"] + 0.05),
            drawdown=bool(detail["full"]["max_drawdown"] >= -0.15),
            paired_interval=bool(interval["lower_95"] > 0),
            execution=bool(detail["qualification_blocks"] == 0 and detail["replay"]["verified"]),
            coverage=bool(adequate >= 0.75 and available.factor_available.median() >= 24),
        )
        if all(detail["gates"].values()):
            qualified.append(name)
    sensitivity = []
    for name in CANDIDATES:
        extra = simulate_composite(
            etf_data, snapshot, weights[name], cfg, start="2023-01-01", cost_multiplier=2
        )
        xstats = result_stats(extra, etf_data, snapshot, cfg)
        sensitivity.append(
            dict(
                model=name,
                variant="cost_x2",
                **xstats["full"],
                qualification_blocks=xstats["qualification_blocks"],
            )
        )
        extra.equity.to_parquet(folder / name / "cost_x2_equity.parquet")
    for count in (6, 10):
        w, _, _ = portfolio_targets(panel, etf_w, market["price"], etf_data, "balanced", top_n=count)
        extra = simulate_composite(etf_data, snapshot, w, cfg, start="2023-01-01")
        xstats = result_stats(extra, etf_data, snapshot, cfg)
        sensitivity.append(
            dict(
                model="balanced",
                variant=f"holdings_{count}",
                **xstats["full"],
                qualification_blocks=xstats["qualification_blocks"],
            )
        )
    delayed, _, _, _, _ = build_panel(snapshot, raw, etf_w.index, information_delay=1)
    dw, _, _ = portfolio_targets(delayed, etf_w, market["price"], etf_data, "balanced")
    extra = simulate_composite(etf_data, snapshot, dw, cfg, start="2023-01-01")
    xstats = result_stats(extra, etf_data, snapshot, cfg)
    sensitivity.append(
        dict(
            model="balanced",
            variant="information_delay_1session",
            **xstats["full"],
            qualification_blocks=xstats["qualification_blocks"],
        )
    )
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    # Current research observations are visible without becoming account advice.
    last = panel[panel.date == panel.date.max()].copy()
    company_names = snapshot["metadata"].set_index("ts_code").name
    last["name"] = last.symbol.map(company_names)
    last.to_csv(folder / "latest_factor_observations.csv", index=False)
    if hashlib.sha256(active_path.read_bytes()).hexdigest() != active_hash:
        raise AssertionError("Active configuration changed during research")
    if hashlib.sha256(source_protocol.read_bytes()).hexdigest() != protocol_hash:
        raise AssertionError("Protocol changed after outcome tests started")
    summary = dict(
        models=details,
        qualified=qualified,
        reviewed="balanced",
        primary_start="2023-01-01",
        end=protocol["end"],
        capital=200000,
        activation=False,
        status="Research complete; no automatic activation",
        families=CANDIDATES,
        limitations=[
            "Both time folds are retrospective, not unseen forward evidence",
            "Vendor timestamps and financial versions are not a certified point-in-time archive",
            "Analyst EPS conversion and cross-source profit-scope comparisons excluded",
            "2015/2018 crash validation unavailable for the new analyst-data window",
            "Stock native-engine equivalence not claimed; independent fill/event replay used",
        ],
        active_config_sha256=active_hash,
        source_snapshot=str(ALT_ROOT),
    )
    write_json(folder / "summary.json", summary)
    from .alternative_report import render_report

    render_report(folder)
    write_json(ROOT / "outputs/latest_alternative.json", dict(path=str(folder), qualified=qualified))
    print(f"Alternative study complete: {folder}", flush=True)
    return folder
