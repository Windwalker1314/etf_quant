"""Daily, stock-only research signals from dated CSI 300 member snapshots.

This module never changes the live ETF configuration or a user's account.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DataError


@dataclass(frozen=True)
class DailyStockPolicy:
    holdings: int = 10
    keep_rank: int = 20
    min_history: int = 253
    min_daily_amount: float = 30_000_000
    initial_cash: float = 200_000
    bull_exposure: float = 0.9
    bear_exposure: float = 0.3


def _member_mask(members: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    observed = members.copy()
    observed["date"] = pd.to_datetime(observed.trade_date)
    snapshots = (
        observed.assign(present=True)
        .pivot(index="date", columns="con_code", values="present")
        .reindex(columns=symbols)
        .notna()
        .sort_index()
    )
    if snapshots.empty:
        raise DataError("No historical constituent snapshots")
    locations = snapshots.index.searchsorted(dates, side="left") - 1
    valid = locations >= 0
    source_dates = pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    source_dates.iloc[np.flatnonzero(valid)] = snapshots.index[locations[valid]].to_numpy()
    valid &= (dates - pd.DatetimeIndex(source_dates)).days <= 62
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    mask[valid] = snapshots.to_numpy(dtype=bool)[locations[valid]]
    source_dates.iloc[np.flatnonzero(~valid)] = pd.NaT
    return pd.DataFrame(mask, index=dates, columns=symbols), source_dates


def _risk_name_mask(names: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
    risk = np.zeros((len(dates), len(symbols)), dtype=bool)
    column = {symbol: i for i, symbol in enumerate(symbols)}
    for row in names.itertuples():
        if row.ts_code not in column or not pd.notna(row.name) or not re.search("ST|退", row.name, re.I):
            continue
        start = pd.to_datetime(row.start_date, errors="coerce")
        announced = pd.to_datetime(row.ann_date, errors="coerce")
        if pd.notna(announced):
            start = max(start, announced) if pd.notna(start) else announced
        end = pd.to_datetime(row.end_date, errors="coerce")
        if pd.isna(start):
            continue
        active = dates >= start
        if pd.notna(end):
            active &= dates <= end
        risk[active, column[row.ts_code]] = True
    return pd.DataFrame(risk, index=dates, columns=symbols)


def build_daily_targets(snapshot: dict, dates: pd.DatetimeIndex, market_price: pd.Series,
                        policy: DailyStockPolicy = DailyStockPolicy()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use today's close for tomorrow's order; never use a same-day member snapshot."""
    if policy.holdings < 1 or policy.keep_rank < policy.holdings:
        raise ValueError("Invalid holding/rank buffer")
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    raw = snapshot["daily"]
    if raw.duplicated(["ts_code", "trade_date"]).any():
        raise DataError("Duplicate stock daily bars")
    bars = raw.loc[(raw[["open", "high", "low", "close"]] > 0).all(axis=1) & raw.vol.gt(0)].copy()
    adjusted = bars.merge(snapshot["adj_factor"][["ts_code", "trade_date", "adj_factor"]],
                          on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    adjusted["date"] = pd.to_datetime(adjusted.trade_date)
    symbols = pd.Index(sorted(adjusted.ts_code.unique()), name="symbol")
    if symbols.empty:
        raise DataError("No usable stock bars")

    def panel(column: str) -> pd.DataFrame:
        return adjusted.pivot(index="date", columns="ts_code", values=column).reindex(
            index=dates, columns=symbols)

    close = panel("close")
    factor = panel("adj_factor")
    price = close * factor.where(factor.gt(0))
    amount = panel("amount") * 1000  # Tushare daily amount is CNY thousands.
    volume = panel("vol")
    member, source_dates = _member_mask(snapshot["members"], dates, symbols)
    risk_name = _risk_name_mask(snapshot["namechange"], dates, symbols)
    listing = snapshot["metadata"].drop_duplicates("ts_code").set_index("ts_code")
    listed = pd.to_datetime(listing.list_date.reindex(symbols), errors="coerce")
    delisted = pd.to_datetime(listing.delist_date.reindex(symbols), errors="coerce")
    life = pd.DataFrame(
        (dates.to_numpy()[:, None] >= listed.to_numpy()[None, :])
        & ((dates.to_numpy()[:, None] < delisted.to_numpy()[None, :]) | delisted.isna().to_numpy()[None, :]),
        index=dates, columns=symbols,
    )
    availability = price.notna().rolling(270, min_periods=270).sum() >= policy.min_history
    liquid = amount.rolling(20, min_periods=15).mean() >= policy.min_daily_amount
    affordable = close * 100 <= policy.initial_cash * policy.bull_exposure / policy.holdings
    eligible = member & life & ~risk_name & availability & liquid & affordable & volume.gt(0)
    momentum = (price.shift(21) / price.shift(126) - 1 + price.shift(21) / price.shift(252) - 1) / 2
    volatility = price.pct_change(fill_method=None).rolling(63, min_periods=50).std() * np.sqrt(252)
    eligible &= momentum.gt(0) & volatility.gt(0) & volatility.lt(1)
    score = 0.6 * momentum.where(eligible).rank(axis=1, pct=True)
    score += 0.4 * (-volatility.where(eligible)).rank(axis=1, pct=True)

    market = market_price.reindex(dates)
    slow = market.rolling(200, min_periods=200).mean()
    exposure = pd.Series(np.where(market.gt(slow), policy.bull_exposure, policy.bear_exposure),
                         index=dates)
    exposure.loc[slow.isna() | market.isna()] = 0.0
    targets = pd.DataFrame(0.0, index=dates, columns=symbols)
    held: list[str] = []
    rows = []
    for date in dates:
        ranked = score.loc[date].dropna().sort_values(ascending=False, kind="stable")
        rank = {symbol: i + 1 for i, symbol in enumerate(ranked.index)}
        held = [symbol for symbol in held if rank.get(symbol, 10**9) <= policy.keep_rank]
        for symbol in ranked.index:
            if len(held) >= policy.holdings:
                break
            if symbol not in held:
                held.append(symbol)
        selected = held if len(held) == policy.holdings else []
        budget = float(exposure.loc[date]) if selected else 0.0
        if selected:
            targets.loc[date, selected] = budget / policy.holdings
        rows.append(dict(date=date, membership_date=source_dates.loc[date], candidates=len(ranked),
                         selected=len(selected), exposure=budget, symbols=",".join(selected)))
    if (targets < 0).any().any() or targets.sum(axis=1).max() > 1 + 1e-8:
        raise AssertionError("Stock targets must be unlevered")
    return targets, pd.DataFrame(rows).set_index("date")
