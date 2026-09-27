import numpy as np
import pandas as pd
from test_invariants import bars

from steadyquant.config import load_active_config
from steadyquant.etf_cash_sleeve_study import MONEY_ETF, cash_sleeve
from steadyquant.etf_factor_study import factor_multipliers
from steadyquant.etf_premium_study import guarded_targets, publication_dated_premium
from steadyquant.etf_universe_study import ADDITIONS, FAMILIES, expanded_config


def test_share_flow_is_not_tradable_until_a_later_session_and_is_prefix_stable():
    symbol = "510300.SH"
    prices = bars(330, symbol=symbol)
    dates = prices.date
    shares = pd.DataFrame({
        "ts_code": symbol, "trade_date": dates.dt.strftime("%Y%m%d"),
        "date": dates, "total_share": np.where(np.arange(len(dates)) >= 300, 95.0, 100.0),
    })
    full = factor_multipliers({symbol: prices}, {symbol: shares})
    assert full["share_flow"].loc[dates.iloc[300], symbol] == 1.0
    assert full["share_flow"].loc[dates.iloc[301], symbol] < 1.0
    prefix = factor_multipliers({symbol: prices.iloc[:315]}, {symbol: shares.iloc[:315]})
    for name in full:
        pd.testing.assert_frame_equal(prefix[name], full[name].iloc[:315], check_freq=False)
        assert full[name].min().min() >= 0.75
        assert full[name].max().max() <= 1.0


def test_nav_premium_requires_announcement_and_one_complete_session():
    symbol = "513100.SH"
    prices = bars(12, symbol=symbol)
    prices["close"] = 10.0
    dates = prices.date
    nav = pd.DataFrame({
        "ts_code": [symbol, symbol],
        "ann_date": dates.iloc[[4, 9]].dt.strftime("%Y%m%d").to_list(),
        "nav_date": dates.iloc[[2, 7]].dt.strftime("%Y%m%d").to_list(),
        "unit_nav": [8.0, 5.0],
    })
    full = publication_dated_premium({symbol: prices}, {symbol: nav}, pd.DatetimeIndex(dates))
    assert np.isnan(full.loc[dates.iloc[4], symbol])
    assert full.loc[dates.iloc[5], symbol] == 0.25
    prefix = publication_dated_premium(
        {symbol: prices.iloc[:8]}, {symbol: nav.iloc[:1]}, pd.DatetimeIndex(dates.iloc[:8])
    )
    pd.testing.assert_frame_equal(prefix, full.iloc[:8], check_freq=False)
    base = pd.DataFrame(0.3, index=dates, columns=[symbol])
    guarded = guarded_targets(base, full)
    assert guarded.loc[dates.iloc[5], symbol] == 0.15
    assert guarded.max().max() <= 0.3


def test_universe_additions_preserve_asset_class_budgets():
    base = load_active_config()
    original = {bucket: sum(a["weight"] for a in base["assets"] if a["bucket"] == bucket)
                for bucket in {a["bucket"] for a in base["assets"]}}
    for name in FAMILIES:
        cfg = expanded_config(base, name)
        assert len({a["symbol"] for a in cfg["assets"]}) == len(cfg["assets"])
        assert all(a["kind"] == "fund" for a in cfg["assets"])
        for bucket, budget in original.items():
            assert abs(sum(a["weight"] for a in cfg["assets"] if a["bucket"] == bucket) - budget) < 1e-10
    assert len(expanded_config(base, "all_breadth")["assets"]) == len(base["assets"]) + sum(map(len, ADDITIONS.values()))


def test_money_etf_uses_only_spare_budget_and_keeps_cash_reserve():
    cfg = load_active_config()
    dates = pd.bdate_range("2020-01-01", periods=35)
    base = pd.DataFrame({"510300.SH": [0.20] * 35}, index=dates)
    money = bars(35, symbol=MONEY_ETF)
    targets = cash_sleeve(base, money, cfg)
    assert targets[MONEY_ETF].iloc[:19].eq(0).all()
    assert targets[MONEY_ETF].iloc[19:].eq(0.30).all()
    assert targets.sum(axis=1).le(1 - cfg["cash_buffer"] + 1e-10).all()
    pd.testing.assert_series_equal(targets["510300.SH"], base["510300.SH"])
