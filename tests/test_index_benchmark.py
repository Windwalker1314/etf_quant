import pandas as pd
import pytest

from steadyquant.data import Cache, DataError
from steadyquant.index_benchmark import load_shanghai_composite


class FakeProvider:
    def __init__(self, dates):
        self.dates = dates
        self.calls = 0

    def bars(self, symbol, kind, start, end):
        assert (symbol, kind) == ("000001.SH", "index")
        self.calls += 1
        return pd.DataFrame({
            "date": self.dates,
            "symbol": symbol,
            "open": 3000.0, "high": 3010.0, "low": 2990.0, "close": 3000.0,
            "volume": 1_000_000.0, "amount": 1_000_000_000.0, "adj_factor": 1.0,
        })


def test_shanghai_index_requires_same_trading_sessions_and_reuses_cache(tmp_path):
    dates = pd.bdate_range("2026-09-21", periods=4)
    cache = Cache(tmp_path)
    incomplete = FakeProvider(dates.delete(2))
    with pytest.raises(DataError, match="every backtest session"):
        load_shanghai_composite(cache, dates, incomplete)
    assert incomplete.calls == 1
    complete = FakeProvider(dates)
    (cache.root / "benchmarks/000001.SH.parquet").unlink()
    prices = load_shanghai_composite(cache, dates, complete)
    assert prices.index.equals(dates)
    assert complete.calls == 1
    load_shanghai_composite(cache, dates, complete)
    assert complete.calls == 1
