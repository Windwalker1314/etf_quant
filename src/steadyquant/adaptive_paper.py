"""Transparent post-hoc paper candidate review, with no automatic live promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, write_json
from .data import Cache
from .framework import engine_comparison
from .macro import load_macro
from .metrics import performance, window_performance, yearly
from .optimize import STRESS


def review_paper():
    root = Path(json.loads((ROOT / "outputs/latest_adaptive.json").read_text())["path"])
    protocol = json.loads((root / "protocol.json").read_text())
    cfg = {
        **protocol["candidates"]["equal_risk"],
        "rebalance_frequency": "monthly_first_session",
        "name": "Adaptive Equal Risk · Paper Review",
    }
    folder = root / "paper_review"
    folder.mkdir(exist_ok=True)
    write_json(
        folder / "protocol.json",
        dict(
            config=cfg,
            stage="Post-hoc review, not the winner of frozen main batch; primary selection remains simple core.",
            rationale="Equal-risk has lower drawdown than blended-risk; first observed monthly session addresses skipped holiday months. Choice made after historical diagnostics and is potentially overfit.",
            validation="native replay, fresh annual starts, paired bootstrap, costs, lookback84/168, vol8/12, shrink.1/.5, no macro, no relative-band/risk-exit combination",
            activation="Local paper candidate only; steady.yaml and automation unchanged.",
        ),
    )
    cache = Cache()
    data = cache.load(cfg)
    if cache.snapshot_digest != protocol["data_sha256"]:
        raise ValueError("Paper review snapshot changed")
    macro = load_macro()
    w = pd.read_parquet(root / "equal_risk/targets.parquet")
    result = simulate(data, w, cfg)
    expected = pd.read_parquet(root / "diagnostics/equal_risk_first_session_equity.parquet")
    pd.testing.assert_frame_equal(expected, result.equity)
    native, comparison = engine_comparison(data, w, cfg)
    comparison.to_csv(folder / "native_comparison.csv")
    rows = []
    for label, delta, mult in [
        ("cost_x2", {}, 2),
        ("cost_x3", {}, 3),
        ("risk_window_84", dict(risk_window=84), 1),
        ("risk_window_168", dict(risk_window=168), 1),
        ("vol_cap_8", dict(target_vol=0.08), 1),
        ("vol_cap_12", dict(target_vol=0.12), 1),
        ("shrink_10", dict(covariance_shrinkage=0.1), 1),
        ("shrink_50", dict(covariance_shrinkage=0.5), 1),
    ]:
        nw, _ = adaptive_weights(data, {**cfg, **delta}, macro) if delta else (w, None)
        run = simulate(data, nw, cfg, cost_multiplier=mult)
        rows.append(dict(variant=label, **performance(run.equity.equity, 0.02)))
    # Cost feedback changes rebalancing/rounding. Separately show monotone direct
    # extra cost of the exact same fills, without claiming it is a feasible new ledger.
    base_cost = (result.trades.commission + result.trades.slippage_cost).groupby(result.trades.date).sum()
    for mul in (2, 3):
        drag = base_cost.reindex(result.equity.index, fill_value=0).cumsum() * (mul - 1)
        rows.append(dict(variant=f"same_fills_cost_x{mul}", **performance(result.equity.equity - drag, 0.02)))
    pd.DataFrame(rows).to_csv(folder / "sensitivity.csv", index=False)
    annual = []
    for year in range(2018, w.index[-1].year + 1):
        r = simulate(data, w, cfg, start=f"{year}-01-01", end=f"{year}-12-31")
        annual.append(dict(year=year, **performance(r.equity.equity, 0.02)))
    pd.DataFrame(annual).to_csv(folder / "fresh_annual_folds.csv", index=False)
    yearly(result.equity.equity, 0.02).to_csv(folder / "yearly.csv", index=False)
    yearly_base = pd.read_parquet(root / "fixed_core_monthly/equity.parquet").equity
    equities = pd.concat(
        [result.equity.equity.rename("paper_equal_risk"), yearly_base.rename("fixed_core_monthly")], axis=1
    )
    equities.to_parquet(folder / "comparison.parquet")
    for key, frame in dict(
        targets=w, equity=result.equity, positions=result.positions, trades=result.trades
    ).items():
        frame.to_parquet(folder / f"{key}.parquet")
    static_fresh = simulate(data, w, cfg, start="2023-01-01")
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    (ROOT / "configs/adaptive_paper.yaml").write_text((folder / "config.yaml").read_text())
    full = performance(result.equity.equity, 0.02)
    sensitivity_pass = all(r["max_drawdown"] >= -0.15 for r in rows)
    summary = dict(
        strategy=full,
        test_2023=window_performance(result.equity.equity, "2023-01-01", None, 0.02),
        fresh_2023=performance(static_fresh.equity.equity, 0.02),
        native_engine_check=native,
        sensitivity_drawdown_pass=sensitivity_pass,
        paired_sharpe_interval=paired_sharpe_interval(result.equity.equity, yearly_base),
        stress_windows={
            label: {name: window_performance(s, lo, hi, 0.02) for name, s in equities.items()}
            for label, (lo, hi) in STRESS.items()
        },
        trades=len(result.trades),
        average_exposure=float(result.equity.exposure.mean()),
        total_commission=float(result.trades.commission.sum()),
        total_slippage=float(result.trades.slippage_cost.sum()),
        status="Post-hoc paper observation candidate; not a statistically proven improvement; no default activation.",
        paper_qualified=bool(native["verified"] and sensitivity_pass and full["max_drawdown"] >= -0.15),
    )
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return folder
