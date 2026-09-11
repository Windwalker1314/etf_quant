from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import validate_bars
from .execution import fee, plan_cash_orders
from .strategy import rebalance_needed, risk_exit_due, scheduled

EXECUTION_VERSION = "2026-09-11-masked-core-satellite-v5"


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    events: pd.DataFrame


def simulate(
    data: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    cfg: dict,
    cost_multiplier: float = 1.0,
    start: str | None = None,
    end: str | None = None,
    reinvest: bool = True,
    rebalance_mask: pd.DataFrame | None = None,
    forced_mask: pd.DataFrame | None = None,
    symbol_bands: dict | None = None,
) -> BacktestResult:
    """Auditable raw-price ledger. Signals at close; fixed share requests at next open.

    Adjustment-factor changes are modelled as fractional reinvested units at ex-date.
    This is a total-return approximation, NOT actual dividend cash/payment accounting.
    """
    if any(a["kind"] != "fund" for a in cfg["assets"]):
        raise ValueError(
            "Default execution ledger supports ETFs only; stock/individual bond tax and corporate actions require a separate adapter"
        )
    symbols = list(data)
    for df in data.values():
        validate_bars(df)
    if (targets.fillna(0) < 0).any().any() or (targets.fillna(0).sum(axis=1) > 1 + 1e-8).any():
        raise ValueError("Targets must be nonnegative and unlevered")
    dates = targets.index[(targets.index >= pd.Timestamp(start or cfg["backtest_start"]))]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 2:
        raise ValueError("At least two backtest dates are required")
    panels = {s: data[s].set_index("date").reindex(dates) for s in symbols}
    rows = {s: {d: r for d, r in panels[s].iterrows()} for s in symbols}
    cash = float(cfg["initial_cash"])
    qty, last_price, last_adj = ({s: 0.0 for s in symbols} for _ in range(3))
    pending, equities, trades, holdings, events = [], [], [], [], []
    initialized = False
    previous_date = dates[0]
    for date in dates:
        prior_session = previous_date
        cash *= (1 + cfg["cash_rate"]) ** ((date - previous_date).days / 365.25)
        previous_date = date
        # Apply only newly observed adjustment changes, never future factors.
        for s in symbols:
            bar = rows[s][date]
            if np.isfinite(bar.close):
                if last_adj[s] and reinvest:
                    ratio = bar.adj_factor / last_adj[s]
                    if abs(ratio - 1) > 1e-8 and qty[s]:
                        before = qty[s]
                        qty[s] *= ratio
                        events.append(
                            {
                                "date": date,
                                "symbol": s,
                                "type": "total_return_reinvestment",
                                "ratio": ratio,
                                "units_before": before,
                                "units_after": qty[s],
                            }
                        )
                last_adj[s] = bar.adj_factor
        # Expire each order after its next portfolio session, even if symbol is suspended.
        for order in sorted(pending, key=lambda o: o["delta"] > 0):
            s, delta = order["symbol"], order["delta"]
            bar = rows[s][date]
            if not np.isfinite(bar.open) or bar.volume <= 0 or bar.high == bar.low:
                events.append(
                    {"date": date, "symbol": s, "type": "unfilled", "reason": "missing/suspended/locked bar"}
                )
                continue
            # Previous-session volume sizes orders; today's reported volume only caps simulated fills.
            cap = math.floor(bar.volume * cfg["participation_rate"] / cfg["lot_size"]) * cfg["lot_size"]
            amount = min(abs(delta), cap)
            buy = delta > 0
            price = bar.open * (1 + (1 if buy else -1) * cfg["slippage_bps"] / 10000 * cost_multiplier)
            if buy:
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
                amount = min(amount, qty[s])
            if amount <= 0:
                events.append(
                    {
                        "date": date,
                        "symbol": s,
                        "type": "unfilled",
                        "reason": "cash/volume/position constraint",
                    }
                )
                continue
            commission = fee(amount * price, cfg, cost_multiplier)
            signed = amount if buy else -amount
            cash -= signed * price + commission
            qty[s] += signed
            trades.append(
                {
                    "signal_date": order["signal_date"],
                    "date": date,
                    "symbol": s,
                    "side": "BUY" if buy else "SELL",
                    "quantity": amount,
                    "price": price,
                    "notional": amount * price,
                    "commission": commission,
                    "slippage_cost": amount * abs(price - bar.open),
                    "cash_after": cash,
                }
            )
            if cash < -1e-6 or qty[s] < -1e-6:
                raise AssertionError("Cash/holdings conservation violated")
        pending = []
        for s in symbols:
            bar = rows[s][date]
            if np.isfinite(bar.close):
                last_price[s] = bar.close
        values = {s: qty[s] * last_price[s] for s in symbols}
        nav = cash + sum(values.values())
        equities.append({"date": date, "equity": nav, "cash": cash, "exposure": sum(values.values()) / nav})
        holdings.append({"date": date, **{s: v / nav for s, v in values.items()}, "CASH": cash / nav})
        target = targets.loc[date]
        on_schedule = (
            bool(rebalance_mask.loc[date].any())
            if rebalance_mask is not None
            else scheduled(date, cfg, initialized, prior_session)
        )
        if on_schedule or cfg.get("risk_exit_ratio") is not None:
            for s in symbols:
                bar = rows[s][date]
                if not np.isfinite(bar.close) or bar.volume <= 0:
                    continue
                emergency = risk_exit_due(target[s], values[s] / nav, cfg)
                symbol_scheduled = (
                    bool(rebalance_mask.loc[date, s]) if rebalance_mask is not None else on_schedule
                )
                force = bool(forced_mask.loc[date, s]) if forced_mask is not None else False
                if not symbol_scheduled and not emergency:
                    continue
                band_cfg = (
                    {**cfg, "rebalance_band": symbol_bands[s]} if symbol_bands and s in symbol_bands else cfg
                )
                if not force and not emergency and not rebalance_needed(target[s], values[s] / nav, band_cfg):
                    continue
                desired = math.floor(nav * target[s] / bar.close / cfg["lot_size"]) * cfg["lot_size"]
                delta = desired - qty[s]
                cap = math.floor(bar.volume * cfg["participation_rate"] / cfg["lot_size"]) * cfg["lot_size"]
                if delta > 0:
                    delta = math.floor(min(delta, cap) / cfg["lot_size"]) * cfg["lot_size"]
                elif desired == 0:
                    delta = -min(qty[s], cap)
                else:
                    delta = -math.floor(min(-delta, cap) / cfg["lot_size"]) * cfg["lot_size"]
                if abs(delta) > 1e-8:
                    pending.append(
                        {
                            "symbol": s,
                            "delta": delta,
                            "signal_date": date,
                            "cost_exempt": bool(emergency or (delta < 0 and desired == 0)),
                        }
                    )
            initialized = initialized or bool((target > 0).any())
            pending = plan_cash_orders(pending, cash, last_price, cfg, cost_multiplier)
    return BacktestResult(
        pd.DataFrame(equities).set_index("date"),
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
        pd.DataFrame(holdings).set_index("date"),
        pd.DataFrame(events),
    )
