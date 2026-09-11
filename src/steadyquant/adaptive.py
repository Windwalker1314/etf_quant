"""Causal group risk budgets with bounded trend, valuation and funding overlays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import adjusted_frame
from .macro import aligned_macro, load_macro

GROUP_CAPS = dict(cn_equity=0.35, us_equity=0.30, hk_equity=0.15, gold=0.30, bond=0.30, commodity=0.20)


def capped_proportions(scores: np.ndarray, caps: np.ndarray, budget: float) -> np.ndarray:
    scores = np.maximum(np.asarray(scores, dtype=float), 0)
    caps = np.asarray(caps, dtype=float)
    if not scores.any():
        return np.zeros_like(scores)
    budget = min(budget, float(caps[scores > 0].sum()))
    lo, hi = 0.0, max(caps[scores > 0] / scores[scores > 0])
    for _ in range(45):
        mid = (lo + hi) / 2
        if np.minimum(scores * mid, caps).sum() < budget:
            lo = mid
        else:
            hi = mid
    return np.minimum(scores * hi, caps)


def equal_risk(cov: np.ndarray) -> np.ndarray:
    """Positive coordinate descent for 0.5*x'C*x - sum(log(x))/n."""
    n = len(cov)
    diagonal = np.diag(cov).clip(1e-10)
    x = 1 / np.sqrt(diagonal)
    x /= np.sqrt(x @ cov @ x)
    for _ in range(40):
        for j in range(n):
            c = cov[j] @ x - diagonal[j] * x[j]
            x[j] = (-c + np.sqrt(c * c + 4 * diagonal[j] / n)) / (2 * diagonal[j])
    return x / x.sum()


def adaptive_weights(data: dict[str, pd.DataFrame], cfg: dict, macro_raw=None):
    symbols = [a["symbol"] for a in cfg["assets"]]
    panel = adjusted_frame(data)
    close = panel.pivot(index="date", columns="symbol", values="close").reindex(columns=symbols).sort_index()
    returns = close.ffill().pct_change(fill_method=None)
    # Real bars only for eligibility. Forward fill is solely a valuation convention.
    amount = panel.pivot(index="date", columns="symbol", values="amount").reindex_like(close)
    volume = panel.pivot(index="date", columns="symbol", values="volume").reindex_like(close)
    eligible = close.notna() & (close.notna().cumsum() >= cfg["min_history"])
    eligible &= (amount.rolling(20).mean() >= cfg["min_amount"]) & (volume > 0)
    trend = sum((close >= close.rolling(w).mean()).astype(float) for w in (126, 252)) / 2
    w = pd.DataFrame(0.0, index=close.index, columns=symbols)
    use_macro = cfg.get("value_overlay") or cfg.get("rate_overlay")
    mf = (
        aligned_macro(close.index, load_macro() if macro_raw is None else macro_raw)
        if use_macro
        else pd.DataFrame(index=close.index)
    )
    base = np.array([a["weight"] for a in cfg["assets"]])
    buckets = np.array([a["bucket"] for a in cfg["assets"]])
    group_names = list(dict.fromkeys(buckets))
    diagnostics = []
    window = cfg.get("risk_window", 126)
    r = returns.to_numpy()
    for i in range(max(window, cfg["min_history"] - 1), len(close)):
        ok = eligible.iloc[i].to_numpy()
        hist = r[i - window + 1 : i + 1]
        ok &= np.isfinite(hist).all(axis=0)
        if not ok.any():
            continue
        groups = [g for g in group_names if (ok & (buckets == g)).any()]
        # Within-group allocation preserves the broad style ratios; adding correlated
        # ETFs never creates an extra independent risk budget.
        mixing = np.zeros((len(symbols), len(groups)))
        priors = []
        for j, g in enumerate(groups):
            mask = ok & (buckets == g)
            mixing[mask, j] = base[mask] / base[mask].sum()
            priors.append(base[buckets == g].sum())
        group_returns = np.nan_to_num(hist) @ mixing
        sample = np.atleast_2d(np.cov(group_returns, rowvar=False)) * 252
        shrink = cfg.get("covariance_shrinkage", 0.3)
        cov = (1 - shrink) * sample + shrink * np.diag(np.diag(sample)) + np.eye(len(groups)) * 1e-8
        scores = 1 / np.sqrt(np.diag(cov)) if cfg["risk_method"] == "inverse_vol" else equal_risk(cov)
        scores /= scores.sum()
        mix = cfg.get("risk_mix", 1.0)
        scores = mix * scores + (1 - mix) * np.array(priors) / sum(priors)
        if cfg.get("trend_budget"):
            scores *= 0.5 + trend.iloc[i].fillna(0).to_numpy() @ mixing
        if use_macro:
            row = mf.iloc[i]
            for j, g in enumerate(groups):
                if g == "cn_equity" and cfg.get("value_overlay"):
                    vals = [
                        row.get(f"{code}_value", np.nan) for code in ("000300.SH", "000905.SH", "399006.SZ")
                    ]
                    vals = [v for v in vals if np.isfinite(v)]
                    if vals:
                        scores[j] *= 1 + 0.25 * np.mean(vals)
                if cfg.get("rate_overlay") and g in {"cn_equity", "bond"}:
                    change = row.get("rate_change", np.nan)
                    if np.isfinite(change):
                        scores[j] *= 1 - 0.20 * np.tanh(change)  # rates are percentage points
        caps = np.array([min(GROUP_CAPS[g], 0.30 * (mixing[:, j] > 0).sum()) for j, g in enumerate(groups)])
        allocation = capped_proportions(scores, caps, 1 - cfg["cash_buffer"])
        equities = np.array([g.endswith("equity") for g in groups])
        if allocation[equities].sum() > 0.60:
            allocation[equities] *= 0.60 / allocation[equities].sum()
        weights = mixing @ allocation
        # Per-ETF cap has precedence; excess remains cash and is never levered.
        weights = np.minimum(weights, 0.30)
        risk = float(np.std(np.nan_to_num(hist) @ weights, ddof=1) * np.sqrt(252))
        scale = min(1.0, cfg["target_vol"] / max(risk, 1e-12))
        weights *= scale
        w.iloc[i] = weights
        diagnostics.append(
            dict(
                date=close.index[i],
                forecast_vol=risk,
                risk_scale=scale,
                available_groups=len(groups),
                cash=1 - weights.sum(),
            )
        )
    if w.max().max() > 0.30000001 or w.sum(axis=1).max() > 1 or (w < 0).any().any():
        raise AssertionError("Adaptive concentration or cash constraint failed")
    diagnostic = (
        pd.DataFrame(diagnostics).set_index("date") if diagnostics else pd.DataFrame(index=close.index)
    )
    diagnostic = diagnostic.join(mf, how="outer")
    return w, diagnostic.reset_index()
