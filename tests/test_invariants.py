from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from steadyquant.backtest import simulate
from steadyquant.config import load_config
from steadyquant.daily import calendar_dates, make_report, notify_local
from steadyquant.data import Cache, DataError, normalize
from steadyquant.factors import compute, validate_expression
from steadyquant.metrics import performance, window_performance
from steadyquant.strategy import target_weights


def bars(n=400, symbol="510300.SH", seed=7):
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    close = 10 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n)))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": close * 0.997,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 20_000_000.0,
            "amount": 200_000_000.0,
            "adj_factor": 1.0,
        }
    )


def config():
    cfg = load_config()
    cfg["assets"] = [
        {"symbol": "510300.SH", "name": "test", "kind": "fund", "bucket": "cn_equity", "weight": 0.3}
    ]
    cfg.update(backtest_start="2020-01-01", initial_cash=100_000.0)
    return cfg


def test_factors_and_targets_prefix_invariant():
    cfg = config()
    df = bars()
    a, _ = target_weights({"510300.SH": df}, cfg)
    b, _ = target_weights({"510300.SH": df.iloc[:310]}, cfg)
    pd.testing.assert_frame_equal(a.iloc[:310], b)
    assert not a.iloc[:239].any().any()
    assert (a.sum(axis=1) <= 0.3).all()


def test_future_perturbation_does_not_change_earlier_signals():
    cfg = config()
    df = bars()
    changed = df.copy()
    for col in ["open", "high", "low", "close"]:
        changed.loc[320:, col] *= 30
    a, _ = target_weights({"510300.SH": df}, cfg)
    b, _ = target_weights({"510300.SH": changed}, cfg)
    pd.testing.assert_frame_equal(a.iloc[:320], b.iloc[:320])


@pytest.mark.parametrize(
    "expr", ["Ref(Close, -1)", "Close.shift(-1)", "Ref(Close, 0)", "Ts_Mean(Close, -5)", "Ref(Close, 5 - 10)"]
)
def test_future_expressions_rejected(expr):
    with pytest.raises(ValueError):
        validate_expression(expr)


def test_native_factor_correctness():
    df = bars(100)
    f = compute({"510300.SH": df}, {"momentum": "Close / Ref(Close, 20) - 1"})
    np.testing.assert_allclose(f.momentum, df.close / df.close.shift(20) - 1, equal_nan=True)


def test_next_open_and_cash_conservation():
    cfg = config()
    cfg.update(commission_bps=0, minimum_commission=5, slippage_bps=0)
    df = bars(5)
    df.loc[:, ["open", "high", "low", "close"]] = [10.0, 21.0, 9.0, 10.0]
    df.loc[1, "open"] = 20.0
    targets = pd.DataFrame(0.3, index=df.date, columns=["510300.SH"])
    result = simulate({"510300.SH": df}, targets, cfg)
    trade = result.trades.iloc[0]
    assert trade.signal_date == df.date.iloc[0]
    assert trade.date == df.date.iloc[1]
    assert trade.price == 20.0
    assert trade.quantity == 3000  # sized at previous close 10, not next open 20
    assert trade.cash_after == 39995.0
    assert result.equity.iloc[1].equity == 69995.0


def test_missing_open_bar_never_filled_with_previous_price():
    cfg = config()
    full = bars(6)
    partial = full.drop(index=1)
    targets = pd.DataFrame(0.3, index=full.date, columns=["510300.SH"])
    result = simulate({"510300.SH": partial}, targets, cfg)
    assert not (result.trades.date == full.date.iloc[1]).any()
    # next-session request expires instead of filling automatically when symbol returns
    assert not (result.trades.date == full.date.iloc[2]).any()
    assert "missing/suspended/locked bar" in result.events.reason.tolist()


def test_locked_bar_does_not_fill():
    cfg = config()
    df = bars(6)
    df.loc[1, ["open", "high", "low", "close"]] = 10.0
    targets = pd.DataFrame(0.3, index=df.date, columns=["510300.SH"])
    result = simulate({"510300.SH": df}, targets, cfg)
    assert not (result.trades.date == df.date.iloc[1]).any()


def test_cash_and_positions_never_negative_under_gap_and_fees():
    cfg = config()
    cfg.update(minimum_commission=50.0, slippage_bps=100.0, initial_cash=12345.0)
    df = bars(100)
    df.loc[1:, ["open", "high", "low", "close"]] *= 20
    targets = pd.DataFrame(0.99, index=df.date, columns=["510300.SH"])
    result = simulate({"510300.SH": df}, targets, cfg)
    assert (result.equity.cash >= 0).all()
    assert (result.positions >= 0).all().all()
    assert (result.positions.sum(axis=1) - 1).abs().max() < 1e-10
    assert (result.trades.quantity % 100 == 0).all()


def test_backtest_prefix_invariant():
    cfg = config()
    df = bars()
    target, _ = target_weights({"510300.SH": df}, cfg)
    full = simulate({"510300.SH": df}, target, cfg)
    prefix = simulate({"510300.SH": df.iloc[:310]}, target.iloc[:310], cfg)
    pd.testing.assert_frame_equal(full.equity.iloc[:310], prefix.equity)
    pd.testing.assert_frame_equal(
        full.trades[full.trades.date <= df.date.iloc[309]].reset_index(drop=True), prefix.trades
    )


def test_no_buying_before_listing():
    cfg = config()
    df = bars(400)
    delayed = df.iloc[100:]
    target = pd.DataFrame(0.3, index=df.date, columns=["510300.SH"])
    result = simulate({"510300.SH": delayed}, target, cfg)
    assert result.trades.date.min() > delayed.date.min()


def test_total_return_reinvestment_conserves_value_on_ex_date():
    cfg = config()
    cfg.update(commission_bps=0, minimum_commission=0, slippage_bps=0)
    df = bars(4)
    df.loc[:, ["open", "high", "low", "close"]] = [10.0, 11.0, 9.0, 10.0]
    df.loc[2:, ["open", "high", "low", "close"]] *= 0.9
    df.loc[2:, "adj_factor"] = 1 / 0.9
    target = pd.DataFrame(0.3, index=df.date, columns=["510300.SH"])
    result = simulate({"510300.SH": df}, target, cfg)
    assert result.equity.equity.iloc[2] == pytest.approx(100_000.0)
    assert len(result.events[result.events.type == "total_return_reinvestment"]) == 1


def test_invalid_ohlc_and_missing_adjustment_fail_closed():
    raw = bars(5).rename(columns={"date": "trade_date", "symbol": "ts_code", "volume": "vol"})
    raw["trade_date"] = raw.trade_date.dt.strftime("%Y%m%d")
    raw.loc[0, "high"] = 0.1
    with pytest.raises(DataError):
        normalize(raw, "510300.SH")
    raw.loc[0, "high"] = 100
    raw.loc[0, "adj_factor"] = np.nan
    with pytest.raises(DataError):
        normalize(raw, "510300.SH")


def test_calendar_weekend_and_next_day(tmp_path):
    cache = Cache(tmp_path)
    pd.DataFrame(
        {"cal_date": ["20260904", "20260905", "20260906", "20260907", "20260908"], "is_open": [1, 0, 0, 1, 1]}
    ).to_parquet(tmp_path / "calendar.parquet")
    now = datetime(2026, 9, 7, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert calendar_dates(cache, now) == ("20260904", "20260907")


def test_stale_data_blocks_order_advice(tmp_path):
    cache = Cache(tmp_path)
    cfg = config()
    df = bars(300)
    cache.save("510300.SH", df)
    last = df.date.iloc[-1] + pd.Timedelta(days=1)
    pd.DataFrame(
        {
            "cal_date": [last.strftime("%Y%m%d"), (last + pd.Timedelta(days=1)).strftime("%Y%m%d")],
            "is_open": [1, 1],
        }
    ).to_parquet(tmp_path / "calendar.parquet")
    report = make_report(
        cfg, cache, now=last.to_pydatetime().replace(hour=19, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    assert not report["actionable"]
    assert report["orders"] == []
    assert report["status"].startswith("暂停")


def test_notification_idempotency_and_failure(tmp_path):
    report = {"signal_date": "2026-09-09", "status": "观察日", "orders": []}
    with patch("steadyquant.daily.sys.platform", "darwin"), patch("steadyquant.daily.subprocess.run") as run:
        run.return_value.returncode = 1
        assert notify_local(report, tmp_path / "report.html", tmp_path)["status"] == "failed"
        run.return_value.returncode = 0
        assert notify_local(report, tmp_path / "report.html", tmp_path)["status"] == "submitted_to_macos"
        assert notify_local(report, tmp_path / "report.html", tmp_path)["status"] == "duplicate_skipped"
        assert run.call_count == 2


def test_year_window_includes_first_day_return():
    series = pd.Series(
        [100.0, 110.0, 121.0], index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03"])
    )
    result = window_performance(series, "2024-01-01", "2024-12-31", 0.02)
    assert result["total_return"] == pytest.approx(0.21)
    assert performance(pd.Series([100.0, 100.0, 100.0], index=series.index))["sharpe"] is None
