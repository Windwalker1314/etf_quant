"""Read-only Shanghai Composite price-index reference for family backtests."""

from __future__ import annotations

import hashlib

import pandas as pd

from .data import Cache, DataError, TushareProvider, validate_bars

SHANGHAI_COMPOSITE = "000001.SH"


def load_shanghai_composite(
    cache: Cache, dates: pd.DatetimeIndex, provider: TushareProvider | None = None
) -> pd.Series:
    """Return index closes on every portfolio session; never invent missing index bars."""
    if dates.empty:
        raise DataError("No backtest dates for the Shanghai Composite")
    dates = pd.DatetimeIndex(dates)
    path = cache.root / "benchmarks" / f"{SHANGHAI_COMPOSITE}.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if not existing.empty and existing.date.min() > dates.min():
        existing = pd.DataFrame()
    start = str(dates.min().date()).replace("-", "")
    end = str(dates.max().date()).replace("-", "")
    needed = (
        existing.empty or existing.date.max() < dates.max()
    )
    if needed:
        client = provider or TushareProvider()
        fetch_start = (
            str((pd.Timestamp(existing.date.max()) + pd.Timedelta(days=1)).date()).replace("-", "")
            if not existing.empty else start
        )
        fetched = client.bars(SHANGHAI_COMPOSITE, "index", fetch_start, end)
        if fetched.empty:
            raise DataError("Shanghai Composite index history unavailable")
        combined = pd.concat([existing, fetched], ignore_index=True).sort_values("date")
        if combined.date.duplicated().any():
            raise DataError("Duplicate Shanghai Composite dates")
        validate_bars(combined)
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        existing = combined
    closes = existing.set_index("date").close.reindex(dates)
    if closes.isna().any() or (closes <= 0).any():
        raise DataError("Shanghai Composite does not cover every backtest session")
    closes.name = SHANGHAI_COMPOSITE
    return closes


def benchmark_digest(cache: Cache) -> str:
    path = cache.root / "benchmarks" / f"{SHANGHAI_COMPOSITE}.parquet"
    return hashlib.sha256(path.read_bytes()).hexdigest()
