"""Array ledger for sparse sector holdings; close plans, next-open fills."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .execution import fee, plan_cash_orders


class MarketArrays:
    def __init__(self, data, dates, symbols):
        self.dates, self.symbols = dates, list(symbols)
        self.index = {s: i for i, s in enumerate(self.symbols)}
        for name in ["open", "high", "low", "close", "volume", "adj_factor"]:
            setattr(
                self,
                name,
                pd.DataFrame({s: d.set_index("date")[name] for s, d in data.items()})
                .reindex(index=dates, columns=symbols)
                .to_numpy(float),
            )


def simulate_sectors(
    market, targets, mask, forced, cfg, metadata, start="2015-01-05", end=None, cost_multiplier=1
):
    symbols, dates = market.symbols, market.dates
    selected = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end or dates[-1]))
    steps = np.flatnonzero(selected)
    if len(steps) < 2:
        raise ValueError("At least two sessions needed")
    w = targets.reindex(index=dates, columns=symbols, fill_value=0).to_numpy(float)
    flags = mask.reindex(index=dates, columns=symbols, fill_value=False).to_numpy(bool)
    override = forced.reindex(index=dates, columns=symbols, fill_value=False).to_numpy(bool)
    qty, last, adj = np.zeros(len(symbols)), np.zeros(len(symbols)), np.zeros(len(symbols))
    bands = np.array([0.02 if s in metadata.index else cfg["rebalance_band"] for s in symbols])
    cash, pending = float(cfg["initial_cash"]), []
    nav_rows, holdings, trades, events = [], [], [], []
    terminal = {}
    for s, r in metadata.iterrows():
        if s not in market.index or r.current_status != "D":
            continue
        if pd.notna(r.delist_date):
            terminal[market.index[s]] = (pd.Timestamp(r.delist_date), "source_delist_date")
        else:
            observed = np.flatnonzero(np.isfinite(market.close[:, market.index[s]]))
            if len(observed):
                terminal[market.index[s]] = (
                    dates[observed[-1]] + pd.Timedelta(days=1),
                    "inferred_after_last_bar_not_certified",
                )
    previous = dates[steps[0]]
    for n, i in enumerate(steps):
        date = dates[i]
        cash *= (1 + cfg["cash_rate"]) ** ((date - previous).days / 365.25)
        previous = date
        real = np.isfinite(market.close[i])
        factor_change = real & (adj > 0)
        ratio = np.ones(len(symbols))
        ratio[factor_change] = market.adj_factor[i, factor_change] / adj[factor_change]
        changed = factor_change & (np.abs(ratio - 1) > 1e-8) & (qty > 0)
        for j in np.flatnonzero(changed):
            events.append(
                dict(
                    date=date,
                    symbol=symbols[j],
                    type="total_return_reinvestment",
                    ratio=ratio[j],
                    units_before=qty[j],
                    units_after=qty[j] * ratio[j],
                )
            )
        qty[changed] *= ratio[changed]
        adj[real] = market.adj_factor[i, real]
        for order in sorted(pending, key=lambda o: o["delta"] > 0):
            s, delta = order["symbol"], order["delta"]
            j = market.index[s]
            if (
                not np.isfinite(market.open[i, j])
                or market.volume[i, j] <= 0
                or market.high[i, j] == market.low[i, j]
            ):
                events.append(
                    dict(date=date, symbol=s, type="unfilled", reason="missing/suspended/locked bar")
                )
                continue
            cap = (
                math.floor(market.volume[i, j] * cfg["participation_rate"] / cfg["lot_size"])
                * cfg["lot_size"]
            )
            amount = min(abs(delta), cap)
            sign = 1 if delta > 0 else -1
            price = market.open[i, j] * (1 + sign * cfg["slippage_bps"] / 10000 * cost_multiplier)
            if sign > 0:
                amount = min(
                    amount,
                    math.floor(
                        cash
                        / (price * (1 + cfg["commission_bps"] / 10000 * cost_multiplier))
                        / cfg["lot_size"]
                    )
                    * cfg["lot_size"],
                )
                while amount > 0 and amount * price + fee(amount * price, cfg, cost_multiplier) > cash + 1e-8:
                    amount -= cfg["lot_size"]
            else:
                amount = min(amount, qty[j])
            if amount <= 0:
                events.append(
                    dict(date=date, symbol=s, type="unfilled", reason="cash/volume/position constraint")
                )
                continue
            commission = fee(amount * price, cfg, cost_multiplier)
            cash -= sign * amount * price + commission
            qty[j] += sign * amount
            trades.append(
                dict(
                    signal_date=order["signal_date"],
                    date=date,
                    symbol=s,
                    side="BUY" if sign > 0 else "SELL",
                    quantity=amount,
                    price=price,
                    notional=amount * price,
                    commission=commission,
                    slippage_cost=amount * abs(price - market.open[i, j]),
                    cash_after=cash,
                )
            )
        pending = []
        last[real] = market.close[i, real]
        for j, (day, source) in terminal.items():
            if date >= day and qty[j] > 0:
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="delisting_writeoff",
                        loss_cny=qty[j] * last[j],
                        source=source,
                    )
                )
                qty[j] = 0
        values = qty * last
        nav = cash + values.sum()
        if cash < -1e-6 or qty.min() < -1e-6 or nav <= 0:
            raise AssertionError("Cash/holding conservation failed")
        nav_rows.append(
            dict(
                date=date,
                equity=nav,
                cash=cash,
                exposure=values.sum() / nav,
                satellite_exposure=float(
                    sum(values[market.index[s]] for s in metadata.index if s in market.index) / nav
                ),
            )
        )
        holdings.append(np.r_[values / nav, cash / nav])
        active_flags = flags[i].copy()
        if n == 0:
            active_flags[:] = True
        for j in np.flatnonzero(active_flags & real & (market.volume[i] > 0) & ((w[i] > 0) | (qty > 0))):
            target, current = w[i, j], values[j] / nav
            if not override[i, j] and target > 1e-12 and current > 1e-12 and abs(target - current) < bands[j]:
                continue
            desired = math.floor(nav * target / market.close[i, j] / cfg["lot_size"]) * cfg["lot_size"]
            delta = desired - qty[j]
            cap = (
                math.floor(market.volume[i, j] * cfg["participation_rate"] / cfg["lot_size"])
                * cfg["lot_size"]
            )
            if delta > 0:
                delta = math.floor(min(delta, cap) / cfg["lot_size"]) * cfg["lot_size"]
            elif desired == 0:
                delta = -min(qty[j], cap)
            else:
                delta = -math.floor(min(-delta, cap) / cfg["lot_size"]) * cfg["lot_size"]
            if abs(delta) > 1e-8:
                pending.append(
                    dict(
                        symbol=symbols[j],
                        delta=delta,
                        signal_date=date,
                        cost_exempt=bool(delta < 0 and desired == 0),
                    )
                )
        if pending:
            pending = plan_cash_orders(
                pending,
                cash,
                {o["symbol"]: last[market.index[o["symbol"]]] for o in pending},
                cfg,
                cost_multiplier,
            )
    return BacktestResult(
        pd.DataFrame(nav_rows).set_index("date"),
        pd.DataFrame(
            trades,
            columns=[
                "signal_date",
                "date",
                "symbol",
                "side",
                "quantity",
                "price",
                "notional",
                "commission",
                "slippage_cost",
                "cash_after",
            ],
        ),
        pd.DataFrame(holdings, index=dates[steps], columns=symbols + ["CASH"]),
        pd.DataFrame(events),
    )


def replay(result, market, cfg, metadata, cost_multiplier=1):
    """Audit sparse positions from recorded fills and independent observed-price arrays."""
    fills = {d: g for d, g in result.trades.groupby("date")}
    losses = {}
    if len(result.events) and "type" in result.events:
        losses = {d: g for d, g in result.events[result.events.type == "delisting_writeoff"].groupby("date")}
    holdings, adjustments, last = {}, {}, {}
    cash = float(cfg["initial_cash"])
    max_diff = 0
    previous = result.equity.index[0]
    for date in result.equity.index:
        i = market.dates.get_loc(date)
        cash *= (1 + cfg["cash_rate"]) ** ((date - previous).days / 365.25)
        previous = date
        for s in list(holdings):
            j = market.index[s]
            if np.isfinite(market.close[i, j]):
                factor = market.adj_factor[i, j]
                if s in adjustments and abs(factor / adjustments[s] - 1) > 1e-8:
                    holdings[s] *= factor / adjustments[s]
                adjustments[s] = factor
                last[s] = market.close[i, j]
        for t in fills.get(date, pd.DataFrame()).itertuples():
            j = market.index[t.symbol]
            assert market.dates.get_loc(t.signal_date) == i - 1
            sign = 1 if t.side == "BUY" else -1
            assert (
                abs(t.price - market.open[i, j] * (1 + sign * cfg["slippage_bps"] / 10000 * cost_multiplier))
                < 1e-8
            )
            assert abs(t.commission - fee(t.notional, cfg, cost_multiplier)) < 1e-8
            assert (
                t.quantity
                <= math.floor(market.volume[i, j] * cfg["participation_rate"] / cfg["lot_size"])
                * cfg["lot_size"]
                + 1e-8
            )
            if sign > 0:
                assert abs(t.quantity / cfg["lot_size"] - round(t.quantity / cfg["lot_size"])) < 1e-8
            cash -= sign * t.quantity * t.price + t.commission
            holdings[t.symbol] = holdings.get(t.symbol, 0) + sign * t.quantity
            adjustments[t.symbol] = market.adj_factor[i, j]
            last[t.symbol] = market.close[i, j]
        for t in losses.get(date, pd.DataFrame()).itertuples():
            assert abs(t.loss_cny - holdings.get(t.symbol, 0) * last[t.symbol]) < 1e-5
            holdings[t.symbol] = 0
        nav = cash + sum(q * last[s] for s, q in holdings.items())
        assert cash >= -1e-6 and min([0] + list(holdings.values())) >= -1e-6
        max_diff = max(max_diff, abs(nav - result.equity.loc[date, "equity"]))
    assert max_diff < 1e-5, max_diff
    return dict(
        verified=True,
        max_equity_difference_cny=max_diff,
        sessions=len(result.equity),
        fills=len(result.trades),
        terminal_writeoffs=sum(len(x) for x in losses.values()),
    )
