"""Shared close-time order planning; no future opening prices enter share requests."""

from __future__ import annotations

import math


def fee(notional: float, cfg: dict, multiplier: float = 1.0) -> float:
    return max(cfg["minimum_commission"], abs(notional) * cfg["commission_bps"] / 10000) * multiplier


def plan_cash_orders(orders: list[dict], cash: float, prices: dict, cfg: dict, multiplier=1.0) -> list[dict]:
    """Sell first; size buys to estimated cash at observed closing prices.

    Pending sells may fail and prices may gap. The fill engine must still enforce
    actual available cash and liquidity at execution time.
    """
    planned = []
    for order in sorted(orders, key=lambda o: o["delta"] > 0):
        s, delta = order["symbol"], order["delta"]
        buy = delta > 0
        price = prices[s] * (1 + (1 if buy else -1) * cfg["slippage_bps"] / 10000 * multiplier)
        amount = abs(delta)
        if buy:
            amount = min(
                amount,
                math.floor(
                    cash / (price * (1 + cfg["commission_bps"] / 10000 * multiplier)) / cfg["lot_size"]
                )
                * cfg["lot_size"],
            )
            while amount > 0 and amount * price + fee(amount * price, cfg, multiplier) > cash + 1e-8:
                amount -= cfg["lot_size"]
        if amount <= 0:
            continue
        # A close-time economic-size filter. Forced reductions and complete exits
        # are exempt; leaving a small position must never be mandatory to save a fee.
        if not order.get("cost_exempt", False) and amount * prices[s] < cfg.get(
            "minimum_trade_notional", 0.0
        ):
            continue
        signed = amount if buy else -amount
        cash -= signed * price + fee(amount * price, cfg, multiplier)
        planned.append({**order, "delta": signed})
    return planned
