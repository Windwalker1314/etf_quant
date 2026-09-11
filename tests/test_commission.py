import pandas as pd
import pytest

from steadyquant.backtest import simulate
from steadyquant.config import load_config
from steadyquant.execution import fee, plan_cash_orders


def test_minimum_commission_economic_threshold_and_exit_exemption():
    cfg = {**load_config(), "commission_bps": 1.5, "minimum_trade_notional": 5000}
    assert fee(3000, cfg) == 5
    assert fee(100000, cfg) == 15
    orders = [dict(symbol="a", delta=300), dict(symbol="b", delta=600)]
    planned = plan_cash_orders(orders, 20000, {"a": 10, "b": 10}, cfg)
    assert [x["symbol"] for x in planned] == ["b"]
    assert plan_cash_orders([dict(symbol="a", delta=-300, cost_exempt=True)], 0, {"a": 10}, cfg)
    assert not plan_cash_orders([dict(symbol="a", delta=-300)], 0, {"a": 10}, cfg)


def test_size_filter_applies_after_cash_rounding():
    cfg = {**load_config(), "commission_bps": 1.5, "minimum_trade_notional": 5000}
    assert not plan_cash_orders([dict(symbol="a", delta=1000)], 4500, {"a": 10}, cfg)
    cfg.pop("minimum_trade_notional")
    assert plan_cash_orders([dict(symbol="a", delta=1000)], 4500, {"a": 10}, cfg)[0]["delta"] == 400


def test_small_complete_exit_is_not_trapped_by_cost_filter():
    cfg = load_config()
    cfg.update(
        initial_cash=100000,
        backtest_start="2024-01-01",
        commission_bps=1.5,
        minimum_trade_notional=5000,
        rebalance_frequency="monthly_first_session",
    )
    cfg["assets"] = [dict(symbol="a", name="a", kind="fund", bucket="cn_equity", weight=0.3)]
    dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    prices = [10, 10, 4, 4]
    data = pd.DataFrame(
        dict(
            date=dates,
            symbol="a",
            open=prices,
            close=prices,
            high=[11, 11, 5, 5],
            low=[9, 9, 3, 3],
            volume=1e7,
            amount=1e8,
            adj_factor=1.0,
        )
    )
    targets = pd.DataFrame({"a": [0.06, 0.06, 0.0, 0.0]}, index=dates)
    result = simulate({"a": data}, targets, cfg)
    assert list(result.trades.side) == ["BUY", "SELL"]
    assert result.trades.iloc[1].notional < 5000
    assert result.positions.iloc[-1]["a"] == pytest.approx(0)


def test_future_open_does_not_change_cost_filter_decision():
    cfg = {**load_config(), "commission_bps": 1.5, "minimum_trade_notional": 5000}
    cfg.update(initial_cash=100000, backtest_start="2024-01-01", rebalance_frequency="monthly_first_session")
    cfg["assets"] = [dict(symbol="a", name="a", kind="fund", bucket="cn_equity", weight=0.3)]
    dates = pd.to_datetime(["2024-01-30", "2024-01-31"])
    bars = pd.DataFrame(
        dict(
            date=dates,
            symbol="a",
            open=[10, 13],
            close=[10, 13],
            high=[11, 14],
            low=[9, 12],
            volume=1e7,
            amount=1e8,
            adj_factor=1.0,
        )
    )
    target = pd.DataFrame({"a": [0.04, 0.04]}, index=dates)
    filtered = simulate({"a": bars}, target, cfg)
    assert filtered.trades.empty
    unfiltered = simulate({"a": bars}, target, {**cfg, "minimum_trade_notional": 0})
    assert unfiltered.trades.iloc[0].quantity == 400
    assert unfiltered.trades.iloc[0].notional > 5000
