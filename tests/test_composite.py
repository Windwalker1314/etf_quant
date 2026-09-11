import numpy as np
import pandas as pd
import pytest

from steadyquant.composite_ledger import prepare_dividends, simulate_composite, stock_fees, tradable_at_open
from steadyquant.stock_factors import latest_reports


def test_dated_stock_taxes_and_directional_limits():
    cfg = dict(minimum_commission=5, commission_bps=3)
    assert stock_fees(10000, 1000, False, pd.Timestamp("2023-08-25"), "600000.SH", cfg) == (5, 10, 0.1)
    assert stock_fees(10000, 1000, False, pd.Timestamp("2023-08-28"), "600000.SH", cfg) == (5, 5, 0.1)
    assert stock_fees(10000, 1000, True, pd.Timestamp("2022-04-28"), "600000.SH", cfg) == (5, 0, 0.2)
    assert not tradable_at_open(11, 11, 9, True)
    assert tradable_at_open(11, 11, 9, False)
    assert not tradable_at_open(9, 11, 9, False)
    assert tradable_at_open(9, 11, 9, True)
    assert not tradable_at_open(10, np.nan, np.nan, True)


def test_original_reports_available_strictly_after_announcement():
    rows = pd.DataFrame(
        [
            dict(ts_code="600000.SH", ann_date="2020-04-01", end_date="2019-12-31", update_flag="0", roe=10),
            dict(ts_code="600000.SH", ann_date="2020-08-01", end_date="2020-06-30", update_flag="0", roe=99),
            dict(ts_code="600000.SH", ann_date="2020-04-02", end_date="2019-12-31", update_flag="1", roe=50),
        ]
    )
    for col in ("ann_date", "end_date"):
        rows[col] = pd.to_datetime(rows[col])
    assert latest_reports(rows, pd.Timestamp("2020-04-01")).empty
    assert latest_reports(rows, pd.Timestamp("2020-04-03")).iloc[0].roe == 10
    assert latest_reports(rows, pd.Timestamp("2020-08-03")).iloc[0].roe == 99


def miniature(dividend=True):
    dates = pd.bdate_range("2016-01-04", periods=10)
    price = np.where(dates >= pd.Timestamp("2016-01-07"), 9.0, 10.0) if dividend else np.full(10, 10.0)
    raw = pd.DataFrame(
        dict(
            ts_code="600000.SH",
            trade_date=dates.strftime("%Y%m%d"),
            open=price,
            close=price,
            high=price + 0.1,
            low=price - 0.1,
            vol=10000,
            amount=10000,
        )
    )
    adj = raw[["ts_code", "trade_date"]].assign(adj_factor=10 / price)
    limits = raw[["ts_code", "trade_date"]].assign(up_limit=price * 1.1, down_limit=price * 0.9)
    div = pd.DataFrame(
        [
            dict(
                ts_code="600000.SH",
                end_date="20151231",
                ann_date="20160101",
                div_proc="实施",
                stk_div=0.0,
                cash_div_tax=1.0,
                record_date="20160106",
                ex_date="20160107",
                pay_date="20160111",
                div_listdate=None,
            )
        ]
    )
    if not dividend:
        div = div.iloc[:0]
    snapshot = dict(
        daily=raw,
        adj_factor=adj,
        stk_limit=limits,
        dividend=div,
        metadata=pd.DataFrame([dict(ts_code="600000.SH", delist_date=None)]),
    )
    targets = pd.DataFrame(0.5, index=dates, columns=["600000.SH"])
    cfg = dict(
        initial_cash=10000,
        cash_rate=0,
        commission_bps=0,
        minimum_commission=0,
        slippage_bps=0,
        participation_rate=0.01,
        rebalance_frequency="monthly_first_session",
        rebalance_band=0.03,
        rebalance_weekday=4,
    )
    return snapshot, targets, cfg


def test_stock_dividends_are_receivables_until_payment_and_taxed_conservatively():
    snapshot, targets, cfg = miniature()
    r = simulate_composite({}, snapshot, targets, cfg)
    assert (r.trades.date > r.trades.signal_date).all()
    assert r.trades.iloc[0].quantity == 500
    ex = r.equity.loc["2016-01-07"]
    pre = r.equity.loc["2016-01-06"]
    assert ex.receivables == 400
    assert ex.cash == pre.cash
    assert ex.equity == pytest.approx(pre.equity - 100)
    pay = r.equity.loc["2016-01-11"]
    assert pay.receivables == 0
    assert pay.cash == pytest.approx(ex.cash + 400)
    assert pay.equity == pytest.approx(ex.equity)


def test_limit_up_prevents_stock_entry_and_order_expires():
    snapshot, targets, cfg = miniature(False)
    snapshot["stk_limit"].loc[1, "up_limit"] = 10
    r = simulate_composite({}, snapshot, targets, cfg)
    assert r.trades.empty
    assert (r.equity.equity == 10000).all()
    assert r.events.iloc[0].type == "unfilled"


def test_delisting_is_written_off_instead_of_infinite_stale_valuation():
    snapshot, targets, cfg = miniature(False)
    snapshot["metadata"]["delist_date"] = "20160111"
    r = simulate_composite({}, snapshot, targets, cfg)
    assert r.equity.loc["2016-01-11", "equity"] < 5001
    assert "delisting_writeoff" in set(r.events.type)


def test_dividend_revisions_do_not_sum_duplicate_schemes_or_use_future_announcements():
    snapshot, targets, _ = miniature()
    d = snapshot["dividend"]
    rows = pd.concat(
        [
            d.assign(ann_date="20160101", cash_div_tax=0.1),
            d.assign(ann_date="20160105", cash_div_tax=0.3),
            d.assign(ann_date="20160201", cash_div_tax=99),
        ],
        ignore_index=True,
    )
    result = prepare_dividends(rows, targets.index)
    assert len(result) == 1
    assert result[0]["cash_div_tax"] == 0.3
    assert not result[0]["reconciliation_error"]
    rows = pd.concat([d.assign(cash_div_tax=0.1), d.assign(cash_div_tax=0.3)], ignore_index=True)
    result = prepare_dividends(rows, targets.index)
    assert result[0]["reconciliation_error"]
    assert np.isnan(result[0]["cash_div_tax"])


def test_suspended_ex_date_does_not_double_count_dividend_value():
    snapshot, targets, cfg = miniature()
    for key in ("daily", "adj_factor", "stk_limit"):
        snapshot[key] = snapshot[key][snapshot[key].trade_date != "20160107"]
    r = simulate_composite({}, snapshot, targets, cfg)
    assert r.equity.loc["2016-01-07", "equity"] == pytest.approx(r.equity.loc["2016-01-06", "equity"] - 100)
    assert r.equity.loc["2016-01-07", "receivables"] == 400
    assert "suspended_ex_reference" in set(r.events.type)
