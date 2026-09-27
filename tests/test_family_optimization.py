import pandas as pd
import pytest

from steadyquant.family_optimization import cap_hk_and_redistribute


def test_hk_cap_preserves_constraints_and_uses_only_eligible_etfs():
    cfg = {"assets": [
        {"symbol": "510300.SH", "bucket": "cn_equity", "weight": 0.2},
        {"symbol": "513500.SH", "bucket": "us_equity", "weight": 0.15},
        {"symbol": "159920.SZ", "bucket": "hk_equity", "weight": 0.05},
    ]}
    dates = pd.to_datetime(["2026-09-24", "2026-09-25"])
    weights = pd.DataFrame({
        "510300.SH": [0.20, 0.0],
        "513500.SH": [0.15, 0.10],
        "159920.SZ": [0.15, 0.15],
    }, index=dates)
    result = cap_hk_and_redistribute(weights, cfg)
    assert result["159920.SZ"].eq(0.05).all()
    assert result.loc[dates[0]].sum() == pytest.approx(weights.loc[dates[0]].sum())
    assert result.loc[dates[1], "510300.SH"] == 0
    assert result.loc[dates[1], "513500.SH"] > weights.loc[dates[1], "513500.SH"]
    assert result.max().max() <= 0.30
    assert result.sum(axis=1).le(1).all()
    pd.testing.assert_frame_equal(cap_hk_and_redistribute(weights.iloc[:1], cfg), result.iloc[:1])
