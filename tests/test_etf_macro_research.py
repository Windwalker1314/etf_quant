"""Causal research-signal checks; no live policy changes."""

import pandas as pd

from steadyquant.etf_breadth_study import breadth_signals, breadth_targets
from steadyquant.etf_yield_study import AAA, GOV, align_yields, overlay_rules


def test_breadth_requires_prior_membership_snapshot_and_future_is_inert():
    dates = pd.bdate_range("2021-01-04", periods=22)
    codes = [f"{i:06d}.SH" for i in range(200)]
    rows = [dict(ts_code=s, trade_date=d.strftime("%Y%m%d"), pct_chg=1.0)
            for d in dates for s in codes]
    daily = pd.DataFrame(rows)
    members = pd.DataFrame([dict(trade_date=dates[0].strftime("%Y%m%d"), con_code=s)
                            for s in codes])
    first = breadth_signals(daily, members, dates)
    assert pd.isna(first.advancing_share.iloc[0])
    assert first.advancing_share.iloc[1] == 1
    future = pd.concat([daily, pd.DataFrame([dict(ts_code=s, trade_date="20210215", pct_chg=-5.0)
                                             for s in codes])], ignore_index=True)
    future_members = pd.concat([members, members.assign(trade_date="20210215")], ignore_index=True)
    pd.testing.assert_frame_equal(first, breadth_signals(future, future_members, dates))


def test_breadth_overlay_only_reduces_cn_etf_targets():
    dates = pd.bdate_range("2021-01-04", periods=65)
    base = pd.DataFrame({"510300.SH": 0.20, "511010.SH": 0.30}, index=dates)
    breadth = pd.DataFrame({"advancing_20": 0.4, "large_decline_20": 0.2,
                            "above_126": 0.3}, index=dates)
    out = breadth_targets(base, breadth, pd.Series(range(1, 66), index=dates))
    for name, target in out.items():
        if name != "base":
            assert target["510300.SH"].max() <= .20
            assert target["511010.SH"].eq(.30).all()


def test_chinabond_alignment_lags_complete_market_session_and_prefix_stable():
    dates = pd.bdate_range("2021-01-04", periods=4)
    rows = []
    for date, y10 in ((dates[0], 3.0), (dates[1], 3.1), (dates[2], 3.2)):
        rows.extend([
            {"曲线名称": GOV, "日期": date, "1年": 2.0, "3年": 2.5, "10年": y10},
            {"曲线名称": AAA, "日期": date, "3年": 3.5},
        ])
    raw = pd.DataFrame(rows)
    full = align_yields(raw, dates)
    assert pd.isna(full.gov_10y.iloc[0])
    assert full.gov_10y.iloc[1] == 3.0
    assert full.gov_10y.iloc[2] == 3.1
    old = align_yields(raw[raw["日期"] <= dates[1]], dates[:2])
    pd.testing.assert_frame_equal(old, full.iloc[:2])
    assert not any(flag.iloc[:2].any() for flag in overlay_rules(full).values())
