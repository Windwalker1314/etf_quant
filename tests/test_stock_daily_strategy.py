import numpy as np
import pandas as pd

from steadyquant.stock_daily_strategy import _member_mask, build_daily_targets
from steadyquant.strategy import scheduled


def test_membership_is_prior_observation_and_expires():
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-04-05"])
    members = pd.DataFrame([
        {"trade_date": "20240131", "con_code": "000001.SZ"},
        {"trade_date": "20240201", "con_code": "000002.SZ"},
    ])
    mask, source = _member_mask(members, dates, pd.Index(["000001.SZ", "000002.SZ"]))
    assert not mask.loc["2024-01-31"].any()
    assert mask.at["2024-02-01", "000001.SZ"]
    assert not mask.at["2024-02-01", "000002.SZ"]
    assert not mask.loc["2024-04-05"].any()
    assert source.loc["2024-02-01"] == pd.Timestamp("2024-01-31")


def test_daily_signals_are_unlevered_and_future_prices_do_not_change_prior_targets():
    dates = pd.bdate_range("2020-01-01", periods=370)
    symbols = [f"{i:06d}.SZ" for i in range(1, 13)]
    daily, adjustment, members, metadata = [], [], [], []
    for i, symbol in enumerate(symbols):
        rng = np.random.default_rng(i + 10)
        close = 10 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.003, len(dates))))
        for date, value in zip(dates, close):
            day = date.strftime("%Y%m%d")
            daily.append(dict(ts_code=symbol, trade_date=day, open=value, high=value * 1.01,
                              low=value * 0.99, close=value, vol=100_000, amount=100_000))
            adjustment.append(dict(ts_code=symbol, trade_date=day, adj_factor=1.0))
        metadata.append(dict(ts_code=symbol, list_date="20100101", delist_date=None))
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).last()
    for date in month_ends:
        members.extend(dict(trade_date=date.strftime("%Y%m%d"), con_code=s) for s in symbols)
    snapshot = dict(daily=pd.DataFrame(daily), adj_factor=pd.DataFrame(adjustment),
                    members=pd.DataFrame(members), metadata=pd.DataFrame(metadata),
                    namechange=pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date", "ann_date"]))
    market = pd.Series(np.linspace(10, 15, len(dates)), index=dates)
    first, signals = build_daily_targets(snapshot, dates, market)
    assert first.sum(axis=1).max() <= 0.9 + 1e-9
    assert first.loc[dates[:269]].sum(axis=1).eq(0).all()
    assert signals.selected.max() == 10
    cutoff = dates[320]
    changed = {**snapshot, "daily": snapshot["daily"].copy()}
    later = changed["daily"].trade_date > cutoff.strftime("%Y%m%d")
    for col in ("open", "high", "low", "close"):
        changed["daily"].loc[later, col] *= 2
    second, _ = build_daily_targets(changed, dates, market)
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])
    assert scheduled(pd.Timestamp("2024-01-03"), {"rebalance_frequency": "daily"}, True)
