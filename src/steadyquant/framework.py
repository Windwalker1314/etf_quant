"""AKQuant adapter and independent native-engine replay."""

from __future__ import annotations

import math

import akquant as aq
import pandas as pd
from akquant.backtest.fill_mode import NextOpen

from .execution import plan_cash_orders
from .strategy import rebalance_needed, risk_exit_due, scheduled


def native_backtest(
    data: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    cfg: dict,
    rebalance_mask=None,
    forced_mask=None,
    symbol_bands=None,
):
    """Raw-price replay without corporate actions, used as an independent engine check.

    For an exact comparison choose an interval with constant adjustment factors.
    Full-history total returns are calculated by the audited reinvestment ledger.
    """
    selected = {}
    for symbol, df in data.items():
        d = df[df.date >= pd.Timestamp(cfg["backtest_start"])].copy()
        if not d.empty:
            selected[symbol] = d
    price = pd.concat(selected.values()).pivot(index="date", columns="symbol", values="close")
    volume = pd.concat(selected.values()).pivot(index="date", columns="symbol", values="volume")

    class Allocation(aq.Strategy):
        def __init__(self):
            super().__init__()
            self.started_allocation = False
            self.previous_session = None

        def on_cross_section(self, trading_date, timestamp):
            date = pd.Timestamp(trading_date)
            if date not in targets.index or date not in price.index:
                return
            on_schedule = (
                bool(rebalance_mask.loc[date].any())
                if rebalance_mask is not None
                else scheduled(date, cfg, self.started_allocation, self.previous_session)
            )
            self.previous_session = date
            if not on_schedule and cfg.get("risk_exit_ratio") is None:
                return
            target = targets.loc[date]
            portfolio_value = self.equity
            requests = []
            for symbol in selected:
                close = price.loc[date, symbol]
                if pd.isna(close):
                    continue
                position = self.get_position(symbol)
                current = position * close / portfolio_value
                emergency = risk_exit_due(target[symbol], current, cfg)
                symbol_scheduled = (
                    bool(rebalance_mask.loc[date, symbol]) if rebalance_mask is not None else on_schedule
                )
                force = bool(forced_mask.loc[date, symbol]) if forced_mask is not None else False
                if not symbol_scheduled and not emergency:
                    continue
                band_cfg = (
                    {**cfg, "rebalance_band": symbol_bands[symbol]}
                    if symbol_bands and symbol in symbol_bands
                    else cfg
                )
                if not force and not emergency and not rebalance_needed(target[symbol], current, band_cfg):
                    continue
                desired = (
                    math.floor(portfolio_value * target[symbol] / close / cfg["lot_size"]) * cfg["lot_size"]
                )
                delta = desired - position
                # Match the independent ledger's prior-session liquidity sizing.
                cap = (
                    math.floor(volume.loc[date, symbol] * cfg["participation_rate"] / cfg["lot_size"])
                    * cfg["lot_size"]
                )
                if delta > 0:
                    delta = math.floor(min(delta, cap) / cfg["lot_size"]) * cfg["lot_size"]
                elif desired == 0:
                    delta = -min(position, cap)
                else:
                    delta = -math.floor(min(-delta, cap) / cfg["lot_size"]) * cfg["lot_size"]
                if delta:
                    requests.append(
                        dict(
                            delta=delta,
                            symbol=symbol,
                            cost_exempt=bool(emergency or (delta < 0 and desired == 0)),
                        )
                    )
            for request in plan_cash_orders(requests, self.cash, price.loc[date].to_dict(), cfg):
                symbol = request["symbol"]
                self.order_target(symbol, self.get_position(symbol) + request["delta"])
            self.started_allocation |= bool((target > 0).any())

    return aq.run_backtest(
        data=selected,
        strategy=Allocation,
        symbols=list(selected),
        initial_cash=cfg["initial_cash"],
        commission_rate=cfg["commission_bps"] / 10000,
        stamp_tax_rate=0.0,
        transfer_fee_rate=0.0,
        min_commission=cfg["minimum_commission"],
        slippage={"type": "percent", "value": cfg["slippage_bps"] / 10000},
        volume_limit_pct=cfg["participation_rate"],
        t_plus_one=True,
        lot_size=cfg["lot_size"],
        fill_policy=NextOpen(),
        timezone="Asia/Shanghai",
        show_progress=False,
    )


def engine_comparison(
    data: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    cfg: dict,
    rebalance_mask=None,
    forced_mask=None,
    symbol_bands=None,
) -> tuple[dict, pd.DataFrame]:
    from .backtest import simulate

    # Latest 40 common sessions, excluding any factor-changing date; all symbols must have real bars.
    common = sorted(set.intersection(*(set(df.date) for df in data.values())))
    if len(common) < 41:
        return {"verified": False, "reason": "insufficient common sessions"}, pd.DataFrame()
    dates = common[-40:]
    window = {s: df[df.date.isin(dates)].copy() for s, df in data.items()}
    if any(d.adj_factor.nunique() != 1 for d in window.values()):
        return {"verified": False, "reason": "corporate action in comparison window"}, pd.DataFrame()
    local_cfg = {**cfg, "backtest_start": str(pd.Timestamp(dates[0]).date())}
    w = targets.loc[dates]
    ledger = simulate(
        window,
        w,
        local_cfg,
        reinvest=False,
        rebalance_mask=rebalance_mask,
        forced_mask=forced_mask,
        symbol_bands=symbol_bands,
    )
    native = native_backtest(
        window,
        w,
        local_cfg,
        rebalance_mask=rebalance_mask,
        forced_mask=forced_mask,
        symbol_bands=symbol_bands,
    )
    native_series = native.equity_curve_daily
    native_series.index = native_series.index.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    native_series = native_series.groupby(level=0).last()
    # Native daily marks can precede deferred cross-symbol fills. Independently
    # value the engine's actual executions at the same closing marks as the ledger.
    # This does not alter native orders/fills or relax cash constraints.
    executions = native.executions_df.copy()
    dates_index = pd.DatetimeIndex(dates, name="date")
    native_marked = pd.Series(float(cfg["initial_cash"]), index=dates_index)
    if not executions.empty:
        executions["date"] = (
            pd.to_datetime(executions.timestamp)
            .dt.tz_convert("Asia/Shanghai")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        executions["signed"] = executions.quantity.where(
            executions.side.str.lower() == "buy", -executions.quantity
        )
        changes = (
            executions.pivot_table(index="date", columns="symbol", values="signed", aggfunc="sum")
            .reindex(index=dates_index, columns=list(window))
            .fillna(0)
        )
        positions = changes.cumsum()
        executions["cash_change"] = -executions.signed * executions.price - executions.commission
        cash_changes = executions.groupby("date").cash_change.sum().reindex(dates_index, fill_value=0)
        cash = cfg["initial_cash"] + cash_changes.cumsum()
        closes = (
            pd.concat(window.values())
            .pivot(index="date", columns="symbol", values="close")
            .reindex_like(positions)
        )
        native_marked = cash + (positions * closes).sum(axis=1)
    compare = pd.concat(
        [
            ledger.equity.equity.rename("ledger"),
            native_marked.rename("akquant"),
            native_series.rename("akquant_reported"),
        ],
        axis=1,
    ).dropna()
    diff = float((compare.ledger - compare.akquant).abs().max()) if not compare.empty else float("inf")
    result = {
        "framework": "AKQuant 0.3.55",
        "verified": len(compare) == len(dates) and diff < 0.01,
        "start": str(pd.Timestamp(dates[0]).date()),
        "end": str(pd.Timestamp(dates[-1]).date()),
        "compared_sessions": len(compare),
        "max_equity_difference_cny": diff,
        "reported_curve_difference_cny": float((compare.ledger - compare.akquant_reported).abs().max()),
        "ledger_trades": len(ledger.trades),
        "native_executions": len(executions),
        "scope": "40 common sessions, native actual executions revalued at common closing marks; raw prices, no corporate actions; next-open, same fees",
    }
    return result, compare
