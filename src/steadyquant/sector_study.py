"""Predeclared sector satellite experiment, with matched equity-budget controls."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime

import pandas as pd

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .commission_study import metrics
from .config import ROOT, write_json
from .framework import engine_comparison
from .metrics import yearly
from .sector_data import SECTOR_ROOT
from .sector_ledger import MarketArrays, replay, simulate_sectors
from .sector_rotation import build_features, choose_sectors, joint_targets, load_snapshot


def run_sector_study():
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    protocol = json.loads((ROOT / "configs/sector_research_protocol.json").read_text())
    out = ROOT / "outputs/sector" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    shutil.copy(ROOT / "configs/sector_research_protocol.json", out / "protocol.json")
    print(f"Output: {out}", flush=True)
    cfg, data, meta, audit, core_hash = load_snapshot()
    cfg = {**cfg, "initial_cash": protocol["primary_capital"]}
    audit.to_csv(out / "download_audit.csv", index=False)
    meta.to_parquet(out / "metadata.parquet")
    (out / "source").mkdir()
    sources = [
        "sector_study.py",
        "sector_rotation.py",
        "sector_ledger.py",
        "sector_data.py",
        "adaptive.py",
        "execution.py",
        "metrics.py",
        "backtest.py",
        "framework.py",
    ]
    for name in sources:
        shutil.copy(ROOT / "src/steadyquant" / name, out / "source" / name)
    write_json(
        out / "provenance.json",
        dict(
            core_sha256=core_hash,
            protected=hashes,
            metadata_sha256=hashlib.sha256((out / "metadata.parquet").read_bytes()).hexdigest(),
            source_sha256={n: hashlib.sha256((out / "source" / n).read_bytes()).hexdigest() for n in sources},
            snapshot=str(SECTOR_ROOT),
        ),
    )
    print("Build causal features across complete downloaded universe", flush=True)
    features = build_features(data, meta)
    core_data = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
    core, _ = adaptive_weights(core_data, cfg)
    picks, coverage = {}, {}
    for family in ("raw", "risk_adjusted"):
        picks[family], coverage[family] = choose_sectors(features, meta, family)
        coverage[family].to_csv(out / f"coverage_{family}.csv", index=False)
        picks[family].map(lambda x: ",".join(x)).to_csv(out / f"picks_{family}.csv")
    # Truncate actual input, recompute every rolling feature and every held state.
    cut = pd.Timestamp("2022-12-30")
    truncated = {s: d[d.date <= cut] for s, d in data.items()}
    prefix = build_features(truncated, meta)
    for family in picks:
        p, _ = choose_sectors(prefix, meta, family)
        pd.testing.assert_series_equal(p.loc[:cut], picks[family].loc[:cut])
    pc, _ = adaptive_weights({s: d[d.date <= cut] for s, d in core_data.items()}, cfg)
    pd.testing.assert_frame_equal(pc, core.loc[pc.index])
    write_json(
        out / "prefix_checks.json", dict(verified=True, cut=str(cut.date()), families=list(picks), core=True)
    )
    del prefix, truncated
    core_symbols = list(core_data)
    symbols = core_symbols + sorted({s for p in picks.values() for v in p for s in v} - set(core_symbols))
    dates = core.index.intersection(features["dates"])
    market = MarketArrays(data, dates, symbols)
    empty = pd.Series([tuple()] * len(dates), index=dates)
    targets = {"core": joint_targets(core, empty, 0, symbols)}
    for name, params in protocol["candidates"].items():
        for control in (False, True):
            targets[name + ("_broad" if control else "")] = joint_targets(
                core, picks[params["family"]], params["fraction"], symbols, control
            )
    constant = pd.Series([("510300.SH", "510500.SH")] * len(dates), index=dates)
    for fraction in (0.1, 0.2):
        targets[f"constant{int(fraction * 100)}"] = joint_targets(core, constant, fraction, symbols, True)

    def simulate(name, capital=200000, start="2015-01-05", end=None, mul=1):
        w, mask, forced, _ = targets[name]
        return simulate_sectors(
            market, w, mask, forced, {**cfg, "initial_cash": capital}, meta, start, end, mul
        )

    def save(r, name, capital, mode, mul=1):
        folder = out / str(capital) / name / mode
        folder.mkdir(parents=True)
        for attr in ("equity", "positions", "trades", "events"):
            getattr(r, attr).to_parquet(folder / f"{attr}.parquet")
        check = replay(r, market, {**cfg, "initial_cash": capital}, meta, mul)
        write_json(folder / "replay.json", check)
        return check

    training = []
    for name in protocol["candidates"]:
        print(f"Selection window ONLY: {name}", flush=True)
        r = simulate(name, end="2019-12-31")
        save(r, name, 200000, "selection_only")
        training.append(dict(model=name, **metrics(r, cfg)))
    chosen = max(training, key=lambda r: r["sharpe"])["model"]
    write_json(
        out / "selection.json",
        dict(chosen_before_2020=chosen, training=training, chosen_at=datetime.now().isoformat()),
    )
    print(f"Frozen selection: {chosen}; now evaluate subsequent windows", flush=True)
    rows, years, stress, results = [], [], [], {}
    checks = 4
    periods = [
        ("full", None, None),
        ("selection", "2015-01-05", "2019-12-31"),
        ("validation", "2020-01-01", "2022-12-31"),
        ("review", "2023-01-01", None),
    ]
    for capital in (200000, 100000, 1000000):
        for name in targets:
            print(f"Main {capital} / {name}", flush=True)
            r = simulate(name, capital)
            save(r, name, capital, "carried")
            checks += 1
            results[capital, name] = r
            local_cfg = {**cfg, "initial_cash": capital}
            for period, lo, hi in periods:
                rows.append(dict(capital=capital, model=name, period=period, **metrics(r, local_cfg, lo, hi)))
            years.extend(
                dict(capital=capital, model=name, **x)
                for x in yearly(r.equity.equity, cfg["risk_free_rate"]).to_dict("records")
            )
            for tag, lo, hi in [
                ("2015crash", "2015-06-01", "2015-12-31"),
                ("2018", "2018-01-01", "2018-12-31"),
                ("2020Q1", "2020-01-01", "2020-03-31"),
                ("2022", "2022-01-01", "2022-12-31"),
                ("2025plus", "2025-01-01", None),
            ]:
                stress.append(dict(capital=capital, model=name, window=tag, **metrics(r, local_cfg, lo, hi)))
        pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    # A pre-existing independently executed baseline must reconcile exactly.
    previous = ROOT / "outputs/commission/20260911-102809/200000/base10/carried/equity.parquet"
    old = pd.read_parquet(previous).equity
    new = results[200000, "core"].equity.equity
    pd.testing.assert_series_equal(old, new, check_exact=False, atol=1e-6, rtol=1e-12)
    write_json(
        out / "baseline_parity.json",
        dict(verified=True, max_diff_cny=float((old - new).abs().max()), previous=str(previous)),
    )
    pd.DataFrame(years).to_csv(out / "yearly.csv", index=False)
    pd.DataFrame(stress).to_csv(out / "stress.csv", index=False)
    table = pd.DataFrame(rows)
    intervals = []
    for capital in (100000, 200000, 1000000):
        for period, lo, hi in periods[2:]:
            for control in ("core", chosen + "_broad"):
                a = results[capital, chosen].equity.equity
                b = results[capital, control].equity.equity
                # Include previous close so the first review-day return is retained.
                start = a.index[a.index < pd.Timestamp(lo)][-1]
                ci = paired_sharpe_interval(a.loc[start:hi], b.loc[start:hi])
                intervals.append(dict(capital=capital, period=period, control=control, **ci))
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    sensitivity = []
    params = protocol["candidates"][chosen]
    monthly, _ = choose_sectors(features, meta, params["family"], "monthly")
    skip_features = build_features(data, meta, skip_recent=5)
    skip, _ = choose_sectors(skip_features, meta, params["family"])
    extras = {"monthly": monthly, "skip5": skip}
    expanded = core_symbols + sorted(
        (set(symbols) | {s for p in extras.values() for v in p for s in v}) - set(core_symbols)
    )
    market = MarketArrays(data, dates, expanded)
    for variant, p in extras.items():
        targets[variant] = joint_targets(
            core, p, params["fraction"], expanded, frequency="monthly" if variant == "monthly" else "weekly"
        )
    for capital in (100000, 200000, 1000000):
        local_cfg = {**cfg, "initial_cash": capital}
        for name in ("core", chosen, chosen + "_broad"):
            for mode, start, mul in (("fresh2023", "2023-01-01", 1), ("double_costs", "2015-01-05", 2)):
                print(f"Robustness {capital} / {name} / {mode}", flush=True)
                r = simulate(name, capital, start=start, mul=mul)
                save(r, name, capital, mode, mul)
                checks += 1
                sensitivity.append(
                    dict(
                        capital=capital,
                        model=name,
                        variant=mode,
                        **metrics(r, local_cfg, cost_multiplier=mul),
                    )
                )
        for name in extras:
            r = simulate(name, capital)
            save(r, name, capital, "carried")
            checks += 1
            for period, lo, hi in periods:
                sensitivity.append(
                    dict(
                        capital=capital,
                        model=chosen,
                        variant=name + "_" + period,
                        **metrics(r, local_cfg, lo, hi),
                    )
                )
    pd.DataFrame(sensitivity).to_csv(out / "sensitivity.csv", index=False)
    # Compare the optimized array ledger, reference ledger, and external engine
    # only on a recent complete corporate-action-free window.
    native = {}
    for name in ("core", chosen, chosen + "_broad"):
        w, mask, forced, _ = targets[name]
        last = dates[-40:]
        used = [s for s in market.symbols if s in w.columns and w.loc[last, s].max() > 0]
        subset = {s: data[s] for s in used}
        m = mask[used].copy()
        m.loc[last[0]] = True
        bands = {s: 0.02 if s in meta.index else cfg["rebalance_band"] for s in used}
        check, compare = engine_comparison(subset, w[used], cfg, m, forced[used], bands)
        if len(compare):
            small = MarketArrays(subset, last, used)
            fast = simulate_sectors(small, w[used], m, forced[used], cfg, meta, str(last[0].date()))
            difference = float((fast.equity.equity - compare.ledger).abs().max())
            check["array_vs_reference_max_diff_cny"] = difference
            check["array_vs_reference_verified"] = difference < 0.01
            check["verified"] = check.get("verified", False) and difference < 0.01
        native[name] = check
        compare.to_csv(out / f"native_{name}.csv")
    write_json(out / "native_checks.json", native)

    def row(name, period):
        return table[(table.capital == 200000) & (table.model == name) & (table.period == period)].iloc[0]

    point_pass = all(
        row(chosen, p).sharpe > row(control, p).sharpe
        for p in ("validation", "review")
        for control in ("core", chosen + "_broad")
    )
    strong = point_pass and all(x["lower_95"] > 0 for x in intervals if x["capital"] == 200000)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in protected)
    summary = dict(
        selected=chosen,
        point_estimate_pass=bool(point_pass),
        strong_evidence=bool(strong),
        active_changed=False,
        protected_files_verified=True,
        replay_checks=checks,
        native_checks=native,
        data_end=protocol["end"],
        downloaded=len(audit),
        usable=len(meta),
        retired_usable=int((meta.current_status == "D").sum()),
        recommendation="keep_existing_core; sector research only",
        limitations=protocol["limitations"],
    )
    write_json(out / "summary.json", summary)
    from .sector_report import render_report

    render_report(out)
    write_json(ROOT / "outputs/latest_sector.json", dict(path=str(out)))
    print(f"Sector research complete: {out}", flush=True)
    return out
