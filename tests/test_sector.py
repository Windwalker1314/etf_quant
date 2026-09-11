import numpy as np
import pandas as pd

from steadyquant.backtest import simulate
from steadyquant.config import load_active_config
from steadyquant.sector_ledger import MarketArrays, replay, simulate_sectors
from steadyquant.sector_rotation import decision_mask, joint_targets


def fixture():
    dates = pd.bdate_range("2023-01-02", periods=90, name="date")
    data = {}
    for n, s in enumerate(["510300.SH", "512480.SH"]):
        price = 10 + np.arange(len(dates)) * 0.015 * (n + 1)
        factor = np.ones(len(dates))
        if n == 1:
            price[45:] *= 0.9
            factor[45:] = 1 / 0.9
        data[s] = pd.DataFrame(
            dict(
                date=dates,
                symbol=s,
                open=price * 0.998,
                close=price,
                high=price * 1.02,
                low=price * 0.98,
                volume=1e7,
                amount=1e8,
                adj_factor=factor,
            )
        )
    cfg = load_active_config()
    cfg.update(initial_cash=200000, backtest_start="2023-01-02")
    cfg["assets"] = [dict(symbol=s, name=s, kind="fund", bucket="cn_equity", weight=0.2) for s in data]
    meta = pd.DataFrame([dict(ts_code="512480.SH", current_status="L", delist_date=None)]).set_index(
        "ts_code"
    )
    w = pd.DataFrame({"510300.SH": 0.25, "512480.SH": 0.1}, index=dates)
    w.loc[dates[30] :, "510300.SH"] = 0.24
    mask = pd.DataFrame(
        {"510300.SH": decision_mask(dates, "monthly"), "512480.SH": decision_mask(dates)}, index=dates
    )
    mask.loc[dates[30], "510300.SH"] = True
    force = pd.DataFrame(False, index=dates, columns=w.columns)
    force.loc[dates[30], "510300.SH"] = True
    return dates, data, cfg, meta, w, mask, force


def test_array_ledger_matches_independent_reference_masked_funding_and_actions():
    dates, data, cfg, meta, w, mask, force = fixture()
    market = MarketArrays(data, dates, list(data))
    fast = simulate_sectors(market, w, mask, force, cfg, meta, start="2023-01-02")
    slow = simulate(data, w, cfg, rebalance_mask=mask, forced_mask=force, symbol_bands={"512480.SH": 0.02})
    np.testing.assert_allclose(fast.equity.equity, slow.equity.equity, atol=1e-7, rtol=0)
    pd.testing.assert_frame_equal(fast.trades, slow.trades, check_exact=False, atol=1e-8)
    assert replay(fast, market, cfg, meta)["verified"]


def test_sector_prefix_does_not_change_with_later_prices():
    dates, data, cfg, meta, w, mask, force = fixture()
    full = simulate_sectors(
        MarketArrays(data, dates, list(data)), w, mask, force, cfg, meta, start="2023-01-02"
    )
    short = simulate_sectors(
        MarketArrays(data, dates, list(data)),
        w,
        mask,
        force,
        cfg,
        meta,
        start="2023-01-02",
        end=str(dates[59].date()),
    )
    pd.testing.assert_frame_equal(full.equity.iloc[:60], short.equity)


def test_retired_fund_is_written_off_if_not_exited():
    dates, data, cfg, meta, w, mask, force = fixture()
    meta.loc["512480.SH", ["current_status", "delist_date"]] = ["D", str(dates[50].date())]
    data["512480.SH"] = data["512480.SH"].iloc[:50]
    w.loc[dates[50] :, "512480.SH"] = 0
    market = MarketArrays(data, dates, list(data))
    result = simulate_sectors(market, w, mask, force, cfg, meta, start="2023-01-02")
    assert len(result.events[result.events.type == "delisting_writeoff"]) == 1
    assert result.positions.loc[dates[50], "512480.SH"] == 0
    assert replay(result, market, cfg, meta)["verified"]


def test_control_has_same_budget_and_funding_schedule():
    dates = pd.bdate_range("2023-01-02", periods=15, name="date")
    core = pd.DataFrame({"510300.SH": 0.2, "510500.SH": 0.1, "511010.SH": 0.3}, index=dates)
    picks = pd.Series([()] * 5 + [("512480.SH",)] * 5 + [("512480.SH", "512010.SH")] * 5, index=dates)
    w, mask, forced, budget = joint_targets(core, picks, 0.2, ["512480.SH", "512010.SH"])
    control, cm, cf, cb = joint_targets(core, picks, 0.2, ["512480.SH", "512010.SH"], control=True)
    np.testing.assert_allclose(w.sum(axis=1), control.sum(axis=1))
    pd.testing.assert_series_equal(budget, cb)
    pd.testing.assert_frame_equal(forced, cf)
    assert forced.loc[dates[5], list(core)].all()
    assert mask.loc[dates[6], list(core)].eq(False).all()
    assert budget.iloc[-1] == 0.2
