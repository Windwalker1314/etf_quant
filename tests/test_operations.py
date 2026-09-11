from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from test_invariants import bars, config

from steadyquant.backtest import simulate
from steadyquant.daily import make_report, validate_account
from steadyquant.data import Cache, DataError, TushareProvider
from steadyquant.factors import evaluate


def fresh_cache(tmp_path):
    cache = Cache(tmp_path)
    df = bars(400)
    cache.save("510300.SH", df)
    last = pd.Timestamp(df.date.iloc[-1])
    cal = pd.DataFrame(
        {
            "cal_date": [
                last.strftime("%Y%m%d"),
                (last.to_pydatetime() + __import__("datetime").timedelta(days=1)).strftime("%Y%m%d"),
            ],
            "is_open": [1, 1],
        }
    )
    cal.to_parquet(tmp_path / "calendar.parquet")
    now = datetime.combine(last.date(), datetime.min.time()).replace(
        hour=19, tzinfo=ZoneInfo("Asia/Shanghai")
    )
    return cache, last, now


def test_unconfirmed_account_never_creates_orders(tmp_path):
    cache, last, now = fresh_cache(tmp_path)
    report = make_report(
        config(),
        cache,
        now,
        {"confirmed": False, "cash": 100_000, "positions": {}, "as_of": str(last.date())},
    )
    assert report["orders"] == []
    assert not report["actionable"]
    assert len(report["allocations"]) == 2


def test_initial_allocation_uses_actual_confirmed_account(tmp_path):
    cache, last, now = fresh_cache(tmp_path)
    report = make_report(
        config(), cache, now, {"confirmed": True, "cash": 100_000, "positions": {}, "as_of": str(last.date())}
    )
    assert report["actionable"]
    assert all(o["side"] == "BUY" and o["quantity"] % 100 == 0 for o in report["orders"])
    assert sum(o["quantity"] * o["reference_price"] for o in report["orders"]) < 100_000


def test_old_account_snapshot_never_creates_orders(tmp_path):
    cache, _, now = fresh_cache(tmp_path)
    report = make_report(
        config(), cache, now, {"confirmed": True, "cash": 100_000, "positions": {}, "as_of": "2020-01-01"}
    )
    assert report["orders"] == []
    assert "快照" in " ".join(report["notes"])


@pytest.mark.parametrize(
    "account",
    [
        {"cash": -1, "positions": {}},
        {"cash": np.inf, "positions": {}},
        {"cash": 100, "positions": {"600000.SH": 100}},
    ],
)
def test_invalid_accounts_fail_closed(account):
    with pytest.raises(DataError):
        validate_account({**account, "confirmed": True, "as_of": "2026-09-09"}, config())


def test_default_ledger_refuses_unmodelled_stock_tax():
    cfg = config()
    cfg["assets"][0]["kind"] = "stock"
    df = bars(10)
    with pytest.raises(ValueError, match="ETFs only"):
        simulate({"510300.SH": df}, pd.DataFrame(0.3, index=df.date, columns=["510300.SH"]), cfg)


def test_label_does_not_skip_missing_next_session():
    data = {f"{510300 + i}.SH": bars(30, symbol=f"{510300 + i}.SH", seed=i) for i in range(4)}
    first = next(iter(data))
    data[first] = data[first].drop(index=1)
    factors = pd.concat([d[["date", "symbol"]].assign(factor=i + 1) for i, d in enumerate(data.values())])
    _, detail = evaluate(data, factors, horizon=5)
    # At the first signal date only 3 of 4 assets have next-session entry; minimum N=4 rejects IC.
    if not detail.empty:
        assert detail.date.min() > factors.date.min()


def test_remote_exception_bodies_never_enter_error_output(monkeypatch):
    provider = object.__new__(TushareProvider)
    provider._last_call = 0

    class BadService:
        def query(self, *args, **kwargs):
            raise RuntimeError("sensitive-token-example")

    provider.pro = BadService()
    monkeypatch.setattr("steadyquant.data.time.sleep", lambda *_: None)
    with pytest.raises(DataError) as error:
        provider.query("fund_daily")
    assert "sensitive-token-example" not in str(error.value)
    assert "RuntimeError" in str(error.value)
