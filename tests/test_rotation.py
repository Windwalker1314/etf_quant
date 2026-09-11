from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_invariants import bars

from steadyquant.backtest import simulate
from steadyquant.config import ROOT, load_config
from steadyquant.optimize import choose
from steadyquant.rotation import build_features, ridge_predictions, rotation_weights
from steadyquant.strategy import rebalance_needed, scheduled


def inputs(n=420, count=5):
    cfg = load_config(ROOT / "configs/rotation.yaml")
    cfg["assets"] = cfg["assets"][:count]
    cfg["min_amount"] = 0
    data = {a["symbol"]: bars(n, a["symbol"], seed=i + 12) for i, a in enumerate(cfg["assets"])}
    return cfg, data


def test_rotation_prefix_concentration_and_listing():
    cfg, data = inputs()
    late = list(data)[-1]
    data[late] = data[late].iloc[180:]
    full, _ = rotation_weights(data, cfg)
    prefix = {s: d[d.date <= full.index[349]] for s, d in data.items()}
    cut, _ = rotation_weights(prefix, cfg)
    pd.testing.assert_frame_equal(full.iloc[:350], cut)
    assert (full >= 0).all().all()
    assert full.sum(axis=1).max() <= 0.60 + 1e-10  # all five are the same economic group
    assert full.max().max() <= 0.30 + 1e-10
    assert not full[late].any()  # never has the 253 real observations required
    assert not full.iloc[:252].any().any()


def test_future_shock_does_not_change_rotation_signals():
    cfg, data = inputs(n=365, count=4)
    original, _ = rotation_weights(data, cfg)
    changed = {s: d.copy() for s, d in data.items()}
    for d in changed.values():
        d.loc[330:, ["open", "high", "low", "close"]] *= 7
    shocked, _ = rotation_weights(changed, cfg)
    pd.testing.assert_frame_equal(original.iloc[:330], shocked.iloc[:330])


def test_ridge_matured_labels_prefix_and_future_perturbation():
    cfg, data = inputs(n=1100, count=10)
    features = build_features(data, cfg)
    prediction, audit = ridge_predictions(features)
    assert len(audit) > 0 and prediction.notna().any().any()
    assert (audit.last_label_date <= audit.fit_date).all()
    prefix = {s: d.iloc[:950] for s, d in data.items()}
    cut, cut_audit = ridge_predictions(build_features(prefix, cfg))
    pd.testing.assert_frame_equal(prediction.iloc[:950], cut)
    changed = {s: d.copy() for s, d in data.items()}
    for d in changed.values():
        d.loc[950:, ["open", "high", "low", "close"]] *= 5
    shocked, _ = ridge_predictions(build_features(changed, cfg))
    pd.testing.assert_frame_equal(prediction.iloc[:950], shocked.iloc[:950])
    assert (cut_audit.last_label_date <= cut_audit.fit_date).all()


def test_selection_cannot_read_after_cutoff():
    dates = pd.bdate_range("2015-01-01", "2025-12-31")
    rng = np.random.default_rng(57)
    equities = pd.DataFrame(
        {
            "ensemble_v12": np.cumprod(1 + rng.normal(0.0004, 0.006, len(dates))),
            "quality_v16": np.cumprod(1 + rng.normal(0.0003, 0.008, len(dates))),
        },
        index=dates,
    )
    before, audit = choose(equities, "2015-01-01", "2022-12-31", 0.02, 0.2)
    equities.loc["2023":, "quality_v16"] *= np.linspace(1, 100, len(equities.loc["2023":]))
    after, changed_audit = choose(equities, "2015-01-01", "2022-12-31", 0.02, 0.2)
    assert before == after and audit == changed_audit


def test_small_initial_position_is_bought_and_zero_target_fully_exits():
    from test_invariants import config

    cfg = config()
    cfg.update(rebalance_band=0.03, commission_bps=0, minimum_commission=0, slippage_bps=0)
    df = bars(5)
    df.loc[:, ["open", "high", "low", "close"]] = [10.0, 11.0, 9.0, 10.0]
    targets = pd.DataFrame([0.01, 0.01, 0, 0, 0], index=df.date, columns=["510300.SH"])
    result = simulate({"510300.SH": df}, targets, cfg)
    assert result.trades.side.tolist() == ["BUY", "SELL"]
    assert result.trades.quantity.tolist() == [100.0, 100.0]
    assert result.positions.iloc[-1]["510300.SH"] == 0
    assert not rebalance_needed(0.02, 0.021, cfg)
    assert rebalance_needed(0.02, 0, cfg)
    assert rebalance_needed(0, 0.01, cfg)


def test_monthly_schedule_and_per_year_selector():
    cfg = dict(rebalance_weekday=4, rebalance_frequency="monthly")
    assert scheduled(pd.Timestamp("2026-09-04"), cfg, True)
    assert not scheduled(pd.Timestamp("2026-09-11"), cfg, True)
    assert scheduled(pd.Timestamp("2026-09-11"), cfg, False)
    cfg["schedule_by_year"] = {"2026": "weekly"}
    assert scheduled(pd.Timestamp("2026-09-11"), cfg, True)


def test_broad_extension_preserves_risk_budgets_and_includes_requested_etfs():
    from steadyquant.broad_core import broad_configs

    for name, cfg in broad_configs().items():
        totals = {}
        for asset in cfg["assets"]:
            totals[asset["bucket"]] = totals.get(asset["bucket"], 0) + asset["weight"]
        assert totals["cn_equity"] == pytest.approx(0.20)
        assert totals["us_equity"] == pytest.approx(0.15)
        assert sum(totals.values()) == pytest.approx(1.0)
        if name != "fixed_core_monthly":
            symbols = {a["symbol"] for a in cfg["assets"] if a["weight"] > 0}
            assert {"159915.SZ", "588000.SH", "513100.SH"} <= symbols


def test_native_engine_matches_small_position_entry_and_exit():
    from test_invariants import config

    from steadyquant.framework import engine_comparison

    cfg = config()
    cfg["rebalance_band"] = 0.03
    df = bars(80)
    df.loc[:, ["open", "high", "low", "close"]] = [10.0, 11.0, 9.0, 10.0]
    w = pd.DataFrame(0.01, index=df.date, columns=["510300.SH"])
    w.iloc[65:] = 0
    checked, _ = engine_comparison({"510300.SH": df}, w, cfg)
    assert checked["verified"]
    assert checked["native_executions"] == checked["ledger_trades"] == 2
