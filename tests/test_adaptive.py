import numpy as np
import pandas as pd
import pytest
from test_invariants import bars, config

from steadyquant.adaptive import adaptive_weights, capped_proportions, equal_risk
from steadyquant.adaptive_study import adaptive_configs, select_adaptive
from steadyquant.backtest import simulate
from steadyquant.execution import plan_cash_orders
from steadyquant.framework import engine_comparison
from steadyquant.macro import aligned_macro
from steadyquant.strategy import rebalance_needed, risk_exit_due, scheduled


def macro_inputs(dates):
    seq = np.arange(len(dates))
    return {
        "shibor": pd.DataFrame({"date": dates, "3m": 2 + seq / 1000, "on": 1.5, "1y": 3}),
        **{
            s: pd.DataFrame({"date": dates, "pe_ttm": 10 + seq / 100, "pb": 2 + seq / 500})
            for s in ("000300.SH", "000905.SH", "399006.SZ")
        },
    }


def test_macro_lag_stale_and_future_shock():
    dates = pd.bdate_range("2013-01-01", periods=400)
    raw = macro_inputs(dates)
    a = aligned_macro(dates, raw)
    changed = {k: v.copy() for k, v in raw.items()}
    for df in changed.values():
        df.loc[320:, df.columns != "date"] = 10000
    b = aligned_macro(dates, changed)
    pd.testing.assert_frame_equal(a.iloc[:321], b.iloc[:321])  # today's release isn't used today
    old = {k: v.iloc[:300] for k, v in raw.items()}
    stale = aligned_macro(dates, old)
    assert stale.iloc[310:].isna().all().all()


def test_equal_risk_and_caps_have_independent_analytical_solution():
    cov = np.diag([0.01, 0.04, 0.16])
    expected = np.array([1, 0.5, 0.25]) / 1.75
    np.testing.assert_allclose(equal_risk(cov), expected, atol=1e-10)
    x = capped_proportions(np.array([100.0, 1.0, 1.0]), np.array([0.3, 0.3, 0.3]), 0.8)
    np.testing.assert_allclose(x, [0.3, 0.25, 0.25], atol=1e-10)


def test_adaptive_macro_prefix_listing_and_future_invariance():
    cfg = adaptive_configs()["trend_value_rate"]
    cfg["min_amount"] = 0
    data = {a["symbol"]: bars(390, a["symbol"], i + 100) for i, a in enumerate(cfg["assets"])}
    last = list(data)[-1]
    data[last] = data[last].iloc[200:]
    raw = macro_inputs(data[list(data)[0]].date)
    full, _ = adaptive_weights(data, cfg, raw)
    prefix = {s: d[d.date <= full.index[330]] for s, d in data.items()}
    mr = {s: d[d.date <= full.index[330]] for s, d in raw.items()}
    cut, _ = adaptive_weights(prefix, cfg, mr)
    pd.testing.assert_frame_equal(cut, full.iloc[:331])
    for df in data.values():
        df.loc[df.date > full.index[330], ["open", "high", "low", "close"]] *= 3
    changed, _ = adaptive_weights(data, cfg, raw)
    pd.testing.assert_frame_equal(changed.iloc[:331], full.iloc[:331])
    assert not full[last].any()
    assert full.max().max() <= 0.30000001 and full.sum(axis=1).max() <= 1
    equity = [a["symbol"] for a in cfg["assets"] if a["bucket"].endswith("equity")]
    assert full[equity].sum(axis=1).max() <= 0.60000001


def test_adaptive_selector_does_not_see_later_equity():
    dates = pd.bdate_range("2015-01-01", "2026-01-01")
    x = np.arange(len(dates))
    equities = pd.DataFrame(
        {
            "fixed_core_monthly": np.exp(x * 0.0002 + 0.01 * np.sin(x / 20)),
            "test_model": np.exp(x * 0.0003 + 0.01 * np.sin(x / 20)),
        },
        index=dates,
    )
    expected = select_adaptive(equities)
    equities.loc["2023":, "test_model"] *= 1000
    assert select_adaptive(equities) == expected


def test_first_session_schedule_handles_october_holiday():
    cfg = {"rebalance_frequency": "monthly_first_session", "rebalance_weekday": 4}
    assert scheduled(pd.Timestamp("2025-10-09"), cfg, True, pd.Timestamp("2025-09-30"))
    assert not scheduled(pd.Timestamp("2025-10-10"), cfg, True, pd.Timestamp("2025-10-09"))
    with pytest.raises(ValueError, match="previous"):
        scheduled(pd.Timestamp("2025-10-09"), cfg, True)


def test_relative_band_and_risk_exit_do_not_buy_off_schedule():
    cfg = config()
    cfg.update(
        rebalance_frequency="monthly_first_session",
        relative_rebalance_band=0.2,
        risk_exit_ratio=0.5,
        rebalance_band=0.03,
    )
    assert rebalance_needed(0.03, 0.04, cfg)
    assert not rebalance_needed(0.03, 0.032, cfg)
    assert risk_exit_due(0.03, 0.1, cfg) and not risk_exit_due(0.2, 0.1, cfg)
    df = bars(80)
    df.loc[:, ["open", "high", "low", "close"]] = [10.0, 11.0, 9.0, 10.0]
    w = pd.DataFrame(0.2, index=df.date, columns=["510300.SH"])
    w.iloc[50:] = 0.03
    w.iloc[55:] = 0.2  # rebound cannot immediately buy back off schedule
    result = simulate({"510300.SH": df}, w, cfg)
    sold = result.trades.query("side == 'SELL'")
    assert df.date.iloc[51] in set(sold.date)
    assert not result.trades.date.isin(df.date.iloc[56:60]).any()
    checked, _ = engine_comparison({"510300.SH": df}, w, cfg)
    assert checked["verified"]


def test_close_cash_plan_shrinks_request_before_native_risk_check():
    cfg = config()
    cfg.update(
        commission_bps=0,
        minimum_commission=5,
        slippage_bps=0,
        rebalance_frequency="monthly_first_session",
        rebalance_band=0.001,
    )
    plan = plan_cash_orders([dict(symbol="510300.SH", delta=300)], 2050, {"510300.SH": 10}, cfg)
    assert plan[0]["delta"] == 200
    df = bars(80)
    df.loc[:, ["open", "high", "low", "close"]] = [10.0, 11.0, 9.0, 10.0]
    w = pd.DataFrame(0.98, index=df.date, columns=["510300.SH"])
    w.iloc[65:] = 1.0
    checked, _ = engine_comparison({"510300.SH": df}, w, cfg)
    assert checked["verified"] and checked["ledger_trades"] == 2
    # Share requests are fixed at the preceding close, even when the next open
    # becomes cheaper; the lower open cannot retroactively increase the request.
    short = df.iloc[:5].copy()
    short.loc[1, "open"] = 9
    short.loc[1, "low"] = 8
    cfg["initial_cash"] = 2050
    targets = pd.DataFrame(1.0, index=short.date, columns=["510300.SH"])
    run = simulate({"510300.SH": short}, targets, cfg)
    assert run.trades.iloc[0].quantity == 200
