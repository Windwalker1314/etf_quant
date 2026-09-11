from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import adjusted_frame, compute


def target_weights(
    data: dict[str, pd.DataFrame], cfg: dict, baseline: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed asset budgets, mild trend filter and trailing portfolio volatility ceiling."""
    if cfg.get("model") == "adaptive" and not baseline:
        from .adaptive import adaptive_weights

        return adaptive_weights(data, cfg)
    if cfg.get("model") == "fixed":
        baseline = True
    if cfg.get("model") == "core_satellite" and not baseline:
        from .core_satellite import core_satellite_weights

        return core_satellite_weights(data, cfg)
    if cfg.get("model") == "rotation" and not baseline:
        from .rotation import rotation_weights

        return rotation_weights(data, cfg)
    windows = cfg["trend_windows"]
    expressions = {f"trend_{w}": f"Close / Ts_Mean(Close, {w}) - 1" for w in windows}
    expressions["volatility"] = f"Ts_Std(Close / Ref(Close, 1) - 1, {cfg['vol_window']})"
    factors = compute(data, expressions)
    adjusted = adjusted_frame(data)
    close = adjusted.pivot(index="date", columns="symbol", values="close").sort_index()
    # Valuation-only fill. Missing bars cannot generate orders or positive target weights.
    returns = close.ffill().pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=close.index, columns=list(data))
    for asset in cfg["assets"]:
        symbol = asset["symbol"]
        df = data[symbol].set_index("date").reindex(close.index)
        f = factors[factors.symbol == symbol].set_index("date").reindex(close.index)
        eligible = (
            df.close.notna()
            & (df.close.notna().cumsum() >= cfg["min_history"])
            & (df.amount.rolling(20, min_periods=20).mean() >= cfg["min_amount"])
            & (df.volume > 0)
        )
        trend = pd.concat([(f[f"trend_{w}"] >= 0).astype(float) for w in windows], axis=1).mean(axis=1)
        scale = cfg["trend_floor"] + (1 - cfg["trend_floor"]) * trend
        if baseline or asset["bucket"] == "bond":
            scale = 1.0
        weights[symbol] = asset["weight"] * (1 - cfg["cash_buffer"]) * scale * eligible
    if not baseline:
        arr = returns.to_numpy()
        symbols = list(returns.columns)
        for i in range(cfg["vol_window"], len(weights)):
            w = weights.iloc[i].reindex(symbols).to_numpy()
            selected = w > 0
            if not selected.any():
                continue
            history = arr[i - cfg["vol_window"] + 1 : i + 1, selected]
            if not np.isfinite(history).all():
                weights.iloc[i] = 0.0
                continue
            port_returns = history @ w[selected]
            vol = np.std(port_returns, ddof=1) * np.sqrt(252)
            if vol > cfg["target_vol"]:
                weights.iloc[i] *= cfg["target_vol"] / vol
    if (weights.sum(axis=1) > 1.000001).any() or (weights < 0).any().any():
        raise AssertionError("Long-only / no-leverage invariant failed")
    return weights, factors


def scheduled(date: pd.Timestamp, cfg: dict, initialized: bool, previous_date=None) -> bool:
    if not initialized:
        return True
    frequency = cfg.get("schedule_by_year", {}).get(str(date.year), cfg.get("rebalance_frequency", "weekly"))
    if frequency == "monthly_first_session":
        if previous_date is None:
            raise ValueError("First-session schedule requires the previous observed session")
        return date.to_period("M") != pd.Timestamp(previous_date).to_period("M")
    if frequency == "monthly":
        return date.weekday() == cfg["rebalance_weekday"] and date.day <= 7
    return date.weekday() == cfg["rebalance_weekday"]


def rebalance_needed(target: float, current: float, cfg: dict) -> bool:
    """The drift band applies to existing positive positions, never entry or full exit."""
    if target <= 1e-12 and current <= 1e-12:
        return False
    if target <= 1e-12 or current <= 1e-12:
        return True
    band = cfg["rebalance_band"]
    if cfg.get("relative_rebalance_band") is not None:
        band = min(
            band, max(cfg.get("minimum_rebalance_band", 0.005), target * cfg["relative_rebalance_band"])
        )
    return abs(target - current) >= band


def risk_exit_due(target: float, current: float, cfg: dict) -> bool:
    """Optional close-time emergency reduction; never authorizes an off-schedule buy."""
    ratio = cfg.get("risk_exit_ratio")
    return ratio is not None and current > 1e-12 and target < current * ratio
