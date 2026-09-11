from __future__ import annotations

import numpy as np
import pandas as pd


def performance(equity: pd.Series, rf: float = 0.02) -> dict:
    equity = equity.dropna()
    if len(equity) < 3:
        return {"observations": len(equity)}
    returns = equity.pct_change().dropna()
    days = (equity.index[-1] - equity.index[0]).days
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (365.25 / max(days, 1)) - 1
    daily_rf = (1 + rf) ** (1 / 252) - 1
    excess = returns - daily_rf
    vol = returns.std(ddof=1) * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    negative = np.minimum(excess, 0)
    downside = np.sqrt((negative**2).mean()) * np.sqrt(252)
    below = dd < -1e-10
    groups = (~below).cumsum()
    max_duration = int(below.groupby(groups).sum().max())

    def finite(x):
        return float(x) if np.isfinite(x) else None

    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "observations": len(returns),
        "total_return": finite(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": finite(annual_return),
        "volatility": finite(vol),
        "sharpe": finite(excess.mean() * 252 / vol) if vol > 1e-12 else None,
        "sortino": finite(excess.mean() * 252 / downside) if downside > 1e-12 else None,
        "max_drawdown": finite(dd.min()),
        "current_drawdown": finite(dd.iloc[-1]),
        "max_underwater_sessions": max_duration,
        "calmar": finite(annual_return / abs(dd.min())) if dd.min() < 0 else None,
        "worst_day": finite(returns.min()),
        "best_day": finite(returns.max()),
        "cvar_95": finite(returns[returns <= returns.quantile(0.05)].mean()),
        "risk_free_rate": rf,
    }


def window_performance(equity: pd.Series, start: str, end: str | None, rf: float) -> dict:
    start_ts = pd.Timestamp(start)
    before = equity[equity.index < start_ts].tail(1)
    selected = equity[equity.index >= start_ts]
    if end:
        selected = selected[selected.index <= pd.Timestamp(end)]
    if selected.empty:
        return {"observations": 0}
    return performance(pd.concat([before, selected]), rf)


def yearly(equity: pd.Series, rf: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": year, **window_performance(equity, f"{year}-01-01", f"{year}-12-31", rf)}
            for year in sorted(set(equity.index.year))
        ]
    )


def bootstrap_sharpe(equity: pd.Series, rf: float, n: int = 500, block: int = 20) -> dict:
    r = equity.pct_change().dropna().to_numpy()
    if len(r) < 252:
        return {"available": False}
    rng = np.random.default_rng(1729)
    values = []
    for _ in range(n):
        starts = rng.integers(0, len(r), size=int(np.ceil(len(r) / block)))
        sample = r[((starts[:, None] + np.arange(block)) % len(r)).ravel()[: len(r)]]
        vol = sample.std(ddof=1)
        if vol > 1e-12:
            values.append((sample.mean() - ((1 + rf) ** (1 / 252) - 1)) / vol * np.sqrt(252))
    low, mid, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "available": True,
        "lower_95": float(low),
        "median": float(mid),
        "upper_95": float(high),
        "block_sessions": block,
        "resamples": n,
        "seed": 1729,
        "note": "Historical block bootstrap, not a prediction",
    }
