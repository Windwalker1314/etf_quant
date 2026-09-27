"""Unlevered next-open stock/ETF ledger, with dated taxes and dividend receivables.

Conservative research convention: reserve 20% of every stock cash dividend at
ex-date, irrespective of eventual holding period. Unexplained corporate actions
and missing dividend dates block deployment qualification instead of inventing units.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .strategy import rebalance_needed, scheduled

VERSION = "stock-etf-next-open-v2-ex-reference"


def stock_fees(notional, quantity, buy, date, symbol, cfg, multiplier=1.0):
    commission = max(cfg["minimum_commission"], notional * cfg["commission_bps"] / 10000) * multiplier
    stamp = notional * (0.0005 if date >= pd.Timestamp("2023-08-28") else 0.001) if not buy else 0.0
    if date >= pd.Timestamp("2022-04-29"):
        transfer = notional * 0.00001
    elif date >= pd.Timestamp("2015-08-01"):
        transfer = notional * 0.00002
    else:
        transfer = quantity * 0.0003 if symbol.endswith(".SH") else notional * 0.0000255
    return commission, stamp, transfer


def tradable_at_open(open_price, up, down, buy):
    if not np.isfinite(open_price) or not np.isfinite(up) or not np.isfinite(down):
        return False
    return open_price < up - 1e-8 if buy else open_price > down + 1e-8


def prepare_dividends(raw, dates):
    d = raw[raw.div_proc == "实施"].copy()
    for col in ("record_date", "ex_date", "pay_date", "div_listdate", "ann_date"):
        d[col] = pd.to_datetime(d[col], errors="coerce")
    d = d[d.ex_date.between(dates[0], dates[-1])]
    d = d[d.ann_date.isna() | (d.ann_date <= d.ex_date)].drop_duplicates()
    fields = ["record_date", "cash_div_tax", "stk_div", "pay_date", "div_listdate"]
    resolved = []
    for (symbol, exdate), group in d.groupby(["ts_code", "ex_date"], sort=True):
        row = dict(ts_code=symbol, ex_date=exdate, source_rows=len(group), reconciliation_error=False)
        # Schemes can be revised or combine annual and special dividends. Never
        # sum duplicate schemes. Use the last known pre-ex-date announcement for
        # each field; a null does not override an earlier concrete fact.
        for col in fields:
            known = group.loc[group[col].notna(), ["ann_date", col]]
            if known.empty:
                row[col] = np.nan if col in {"cash_div_tax", "stk_div"} else pd.NaT
                continue
            if known[col].nunique() == 1:
                row[col] = known[col].iloc[0]
                continue
            latest = known.ann_date.max()
            values = known.loc[known.ann_date == latest, col].drop_duplicates()
            if len(values) != 1:
                row["reconciliation_error"] = True
                row[col] = np.nan if col in {"cash_div_tax", "stk_div"} else pd.NaT
            else:
                row[col] = values.iloc[0]
        resolved.append(row)
    return resolved


def simulate_composite(etf_data, snapshot, targets, cfg, start="2016-01-01", end=None, cost_multiplier=1.0):
    dates = targets.index[targets.index >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 2:
        raise ValueError("Insufficient simulation dates")
    if (targets < 0).any().any() or targets.sum(axis=1).max() > 1 + 1e-8:
        raise ValueError("Nonnegative fully funded targets required")
    # Holdings cannot exist in never-targeted symbols; keep matrices bounded.
    symbols = targets.columns[(targets.loc[dates] > 0).any()].tolist()
    if not symbols:
        raise ValueError("Empty target universe")
    n = len(symbols)
    ix = {s: i for i, s in enumerate(symbols)}
    is_stock = np.array([s not in etf_data for s in symbols])
    raw = (
        snapshot["daily"].rename(columns={"ts_code": "symbol", "trade_date": "date", "vol": "volume"}).copy()
    )
    raw["date"] = pd.to_datetime(raw.date)
    raw["volume"] *= 100
    if "pre_close" not in raw:
        raw["pre_close"] = np.nan
    adj = snapshot["adj_factor"].rename(columns={"ts_code": "symbol", "trade_date": "date"}).copy()
    adj["date"] = pd.to_datetime(adj.date)
    limits = snapshot["stk_limit"].rename(columns={"ts_code": "symbol", "trade_date": "date"}).copy()
    limits["date"] = pd.to_datetime(limits.date)
    raw = raw.merge(adj, on=["symbol", "date"], how="left", validate="one_to_one").merge(
        limits, on=["symbol", "date"], how="left", validate="one_to_one"
    )
    raw = pd.concat(
        [raw[raw.symbol.isin(symbols)], *[d for s, d in etf_data.items() if s in ix]], ignore_index=True
    )
    panels = {
        col: raw.pivot(index="date", columns="symbol", values=col)
        .reindex(index=dates, columns=symbols)
        .to_numpy(float)
        for col in (
            "open",
            "close",
            "high",
            "low",
            "volume",
            "adj_factor",
            "up_limit",
            "down_limit",
            "pre_close",
        )
    }
    w = targets.reindex(index=dates, columns=symbols).to_numpy(float)
    dividend_rows = prepare_dividends(snapshot["dividend"][snapshot["dividend"].ts_code.isin(symbols)], dates)
    records, exdates = {}, {}
    for k, row in enumerate(dividend_rows):
        if row["ts_code"] in ix:
            records.setdefault(row["record_date"], []).append((k, row))
            exdates.setdefault(row["ex_date"], []).append((k, row))
    delists = {
        ix[r.ts_code]: pd.Timestamp(r.delist_date)
        for r in snapshot["metadata"].itertuples()
        if r.ts_code in ix and pd.notna(r.delist_date) and str(r.delist_date) not in ("", "None")
    }
    cash = float(cfg["initial_cash"])
    qty = np.zeros(n)
    last = np.zeros(n)
    last_adj = np.zeros(n)
    pending, trades, equity, holdings, events, entitled, receivables = [], [], [], [], [], {}, []
    initialized = False
    known_adjustments = set()
    prior = dates[0]
    for t, date in enumerate(dates):
        o, c, vol = panels["open"][t], panels["close"][t], panels["volume"][t]
        af = panels["adj_factor"][t]
        previous_qty = qty.copy()  # T+1 sellable before today's buys.
        cash *= (1 + cfg["cash_rate"]) ** ((date - prior).days / 365.25)
        corp_today = set()
        for key, row in exdates.get(date, []):
            j = ix[row["ts_code"]]
            corp_today.add(j)
            known_adjustments.add(j)
            if row.get("reconciliation_error") and qty[j] > 0:
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="qualification_block",
                        reason="conflicting same-announcement corporate fields; unresolved value not credited",
                    )
                )
            if pd.isna(row["record_date"]) and qty[j] > 0:
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="qualification_block",
                        reason="missing dividend record date; entitlement not fabricated",
                    )
                )
            units = entitled.get(key, 0.0)
            if units <= 0:
                continue
            if pd.isna(row["cash_div_tax"]) or pd.isna(row["stk_div"]):
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="qualification_block",
                        reason="missing corporate cash/share rate",
                    )
                )
            gross = float(row["cash_div_tax"] or 0)
            bonus = float(row["stk_div"] or 0)
            gross = gross if np.isfinite(gross) else 0.0
            bonus = bonus if np.isfinite(bonus) else 0.0
            # A suspended ex-date still loses its mechanical dividend/split
            # value. Carrying yesterday's unadjusted mark AND the receivable
            # would create fictitious income until the next valid quote.
            if not np.isfinite(c[j]) and last[j] > 0:
                last[j] = max(0.0, (last[j] - gross) / (1 + bonus))
                events.append(
                    dict(date=date, symbol=symbols[j], type="suspended_ex_reference", reference_price=last[j])
                )
            receivables.append(
                dict(
                    j=j,
                    cash=round(units * gross * 0.8, 2),
                    bonus=math.floor(units * bonus + 1e-8),
                    pay=row["pay_date"],
                    listing=row["div_listdate"],
                )
            )
            events.append(
                dict(
                    date=date,
                    symbol=symbols[j],
                    type="dividend_entitlement",
                    units=units,
                    gross_cash=units * gross,
                    tax_reserved=units * gross * 0.2,
                    bonus_shares=math.floor(units * bonus + 1e-8),
                )
            )
            if (gross > 0 and pd.isna(row["pay_date"])) or (bonus > 0 and pd.isna(row["div_listdate"])):
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="qualification_block",
                        reason="missing corporate payment/listing date",
                    )
                )
        for item in receivables:
            if item["cash"] > 0 and pd.notna(item["pay"]) and item["pay"] <= date:
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[item["j"]],
                        type="dividend_cash_payment",
                        amount=item["cash"],
                    )
                )
                cash += item["cash"]
                item["cash"] = 0.0
            if item["bonus"] > 0 and pd.notna(item["listing"]) and item["listing"] <= date:
                events.append(
                    dict(date=date, symbol=symbols[item["j"]], type="bonus_listing", quantity=item["bonus"])
                )
                qty[item["j"]] += item["bonus"]
                previous_qty[item["j"]] += item["bonus"]
                item["bonus"] = 0.0
        valid = np.isfinite(c) & np.isfinite(af)
        changes = (
            valid & (last_adj > 0) & (np.abs(af / np.where(last_adj > 0, last_adj, 1) - 1) > 1e-7) & (qty > 0)
        )
        for j in np.where(changes)[0]:
            if not is_stock[j]:
                ratio = af[j] / last_adj[j]
                events.append(dict(date=date, symbol=symbols[j], type="etf_reinvestment", ratio=ratio))
                qty[j] *= ratio
                previous_qty[j] *= ratio
            elif j not in known_adjustments:
                # Source transitions between 3 and 4 decimals cause <=0.0005
                # absolute factor shifts. Require an unchanged raw previous
                # close too; no money or shares are created by this exception.
                precision_only = abs(af[j] - last_adj[j]) <= 0.0005000001 and np.isclose(
                    panels["pre_close"][t, j], last[j], atol=0.005, rtol=0
                )
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="adjustment_precision" if precision_only else "qualification_block",
                        reason="factor precision change with unchanged raw previous close"
                        if precision_only
                        else "unexplained stock adjustment; raw-price loss retained, no synthetic reinvestment",
                    )
                )
        last_adj[valid] = af[valid]
        known_adjustments.difference_update(np.where(valid)[0])
        for j, delta, signal in sorted(pending, key=lambda x: x[1] > 0):
            buy = delta > 0
            if not np.isfinite(o[j]) or not np.isfinite(vol[j]) or vol[j] <= 0:
                events.append(
                    dict(date=date, symbol=symbols[j], type="unfilled", reason="suspension/missing bar")
                )
                continue
            if is_stock[j] and not tradable_at_open(
                o[j], panels["up_limit"][t, j], panels["down_limit"][t, j], buy
            ):
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="unfilled",
                        reason="directional limit or missing limit",
                    )
                )
                continue
            if not is_stock[j] and panels["high"][t, j] == panels["low"][t, j]:
                continue
            cap = math.floor(vol[j] * cfg["participation_rate"] / 100) * 100
            amount = min(abs(delta), cap)
            if not buy:
                amount = min(amount, previous_qty[j], qty[j])
            slip = cfg["slippage_bps"] * cost_multiplier / 10000
            price = o[j] * (1 + (slip if buy else -slip))
            if is_stock[j]:
                price = (
                    min(price, panels["up_limit"][t, j]) if buy else max(price, panels["down_limit"][t, j])
                )

            def fees(a):
                if is_stock[j]:
                    return stock_fees(a * price, a, buy, date, symbols[j], cfg, cost_multiplier)
                return (
                    max(cfg["minimum_commission"], a * price * cfg["commission_bps"] / 10000)
                    * cost_multiplier,
                    0.0,
                    0.0,
                )

            if buy:
                amount = min(amount, math.floor(cash / price / 100) * 100)
                while amount > 0 and amount * price + sum(fees(amount)) > cash + 1e-8:
                    amount -= 100
            if amount <= 0:
                continue
            fee_parts = fees(amount)
            signed = amount if buy else -amount
            cash -= signed * price + sum(fee_parts)
            qty[j] += signed
            if not buy:
                previous_qty[j] -= amount
            trades.append(
                dict(
                    signal_date=signal,
                    date=date,
                    symbol=symbols[j],
                    kind="stock" if is_stock[j] else "fund",
                    side="BUY" if buy else "SELL",
                    quantity=amount,
                    price=price,
                    notional=amount * price,
                    commission=fee_parts[0],
                    stamp_tax=fee_parts[1],
                    transfer_fee=fee_parts[2],
                    slippage_cost=amount * abs(price - o[j]),
                    cash_after=cash,
                )
            )
        pending = []
        last[np.isfinite(c)] = c[np.isfinite(c)]
        for j, delist in delists.items():
            if date >= delist and qty[j] > 0:
                events.append(
                    dict(
                        date=date,
                        symbol=symbols[j],
                        type="delisting_writeoff",
                        loss=qty[j] * last[j],
                        quantity=qty[j],
                    )
                )
                qty[j] = 0.0
        for key, row in records.get(date, []):
            entitled[key] = qty[ix[row["ts_code"]]]
        claim = sum(item["cash"] + item["bonus"] * last[item["j"]] for item in receivables)
        values = qty * last
        nav = cash + values.sum() + claim
        if cash < -1e-6 or qty.min() < -1e-8 or nav <= 0:
            raise AssertionError("Ledger cash/holdings conservation failure")
        equity.append(
            dict(
                date=date,
                equity=nav,
                cash=cash,
                receivables=claim,
                exposure=values.sum() / nav,
                stock_exposure=values[is_stock].sum() / nav,
            )
        )
        holdings.append(
            dict(
                date=date,
                **{s: values[j] / nav for j, s in enumerate(symbols)},
                CASH=cash / nav,
                RECEIVABLES=claim / nav,
            )
        )
        on_schedule = scheduled(date, cfg, initialized, prior)
        # Stock risk cuts can run daily; they never use tomorrow's prices or volume.
        for j in range(n):
            risk_cut = is_stock[j] and qty[j] > 0 and w[t, j] < values[j] / nav * 0.6
            if not (on_schedule or risk_cut) or not np.isfinite(c[j]) or vol[j] <= 0:
                continue
            band_cfg = {**cfg, "rebalance_band": cfg.get("stock_rebalance_band", 0.005)} if is_stock[j] else cfg
            if not risk_cut and not rebalance_needed(w[t, j], values[j] / nav, band_cfg):
                continue
            desired = math.floor(nav * w[t, j] / c[j] / 100) * 100
            delta = desired - qty[j]
            cap = math.floor(vol[j] * cfg["participation_rate"] / 100) * 100
            if desired == 0:
                delta = -min(qty[j], cap)
            else:
                delta = math.copysign(math.floor(min(abs(delta), cap) / 100) * 100, delta)
            if abs(delta) > 1e-8:
                pending.append((j, delta, date))
        # Reserve cash using only today's close. Failed next-open sells cannot finance buys.
        budget = cash
        planned = []
        for j, delta, signal in sorted(pending, key=lambda x: x[1] > 0):
            buy = delta > 0
            price = c[j] * (1 + (1 if buy else -1) * cfg["slippage_bps"] / 10000 * cost_multiplier)
            amount = abs(delta)

            def estimated_cost(a):
                if is_stock[j]:
                    return sum(stock_fees(a * price, a, buy, date, symbols[j], cfg, cost_multiplier))
                return (
                    max(cfg["minimum_commission"], a * price * cfg["commission_bps"] / 10000)
                    * cost_multiplier
                )

            if buy:
                amount = min(amount, math.floor(budget / price / 100) * 100)
                while amount > 0 and amount * price + estimated_cost(amount) > budget + 1e-8:
                    amount -= 100
            if amount > 0:
                signed = amount if buy else -amount
                budget -= signed * price + estimated_cost(amount)
                planned.append((j, signed, signal))
        pending = planned
        initialized = initialized or w[t].sum() > 0
        prior = date
    return BacktestResult(
        pd.DataFrame(equity).set_index("date"),
        pd.DataFrame(trades),
        pd.DataFrame(holdings).set_index("date"),
        pd.DataFrame(events),
    )
