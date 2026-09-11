"""Causal ETF rotation. No fitted transforms use observations after the signal close."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import adjusted_frame


def build_features(data: dict, cfg: dict) -> dict:
    frame = adjusted_frame(data)
    close = (
        frame.pivot(index="date", columns="symbol", values="close").sort_index().reindex(columns=list(data))
    )
    opening = frame.pivot(index="date", columns="symbol", values="open").reindex_like(close)
    # Forward fill is for past-return valuation only, never signal eligibility or execution.
    marked = close.ffill()
    returns = marked.pct_change(fill_method=None)
    vol = returns.rolling(60, min_periods=60).std() * np.sqrt(252)
    downside = returns.clip(upper=0).pow(2).rolling(60).mean().pow(0.5) * np.sqrt(252)
    mom = {w: marked / marked.shift(w) - 1 for w in (21, 63, 126, 252)}
    trend = {w: marked / marked.rolling(w).mean() - 1 for w in (60, 120, 200)}
    efficiency = (marked - marked.shift(63)).abs() / marked.diff().abs().rolling(63).sum()
    drawdown = marked / marked.rolling(126).max() - 1
    eligible = close.notna() & (close.notna().cumsum() >= cfg["min_history"]) & vol.notna()
    for s, df in data.items():
        raw = df.set_index("date").reindex(close.index)
        eligible[s] &= (raw.amount.rolling(20).mean() >= cfg["min_amount"]) & (raw.volume > 0)
    return dict(
        close=close,
        opening=opening,
        returns=returns,
        vol=vol,
        downside=downside,
        mom=mom,
        trend=trend,
        efficiency=efficiency,
        drawdown=drawdown,
        eligible=eligible,
    )


def ridge_predictions(f: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly rolling ridge, fixed penalty, 3-year training, 22-session label embargo.

    Samples are every 21 sessions. A label spans next open through open 22 sessions
    later, and is usable only once its endpoint has occurred. Training transforms
    are fitted solely on eligible, matured labels. No per-symbol identity feature.
    """
    idx, symbols = f["close"].index, f["close"].columns
    vol = f["vol"].clip(lower=0.05)
    features = [f["mom"][w] / vol for w in (21, 63, 126, 252)]
    features += [f["trend"][120] / vol, f["efficiency"], f["drawdown"] / vol, f["downside"] / vol]
    x = np.stack([v.to_numpy() for v in features], axis=-1)
    y = ((f["opening"].shift(-22) / f["opening"].shift(-1) - 1) / vol).to_numpy()
    valid = f["eligible"].to_numpy() & np.isfinite(x).all(axis=2)
    predictions = np.full(valid.shape, np.nan)
    model, audit = None, []
    for i, date in enumerate(idx):
        if i == 0 or date.month != idx[i - 1].month:
            sample = np.arange(max(0, i - 756), max(0, i - 21))
            sample = sample[sample % 21 == 0]
            mask = valid[sample] & np.isfinite(y[sample])
            xx, yy = x[sample][mask], y[sample][mask]
            model = None
            if len(yy) >= 150:
                mean, std = xx.mean(axis=0), np.maximum(xx.std(axis=0), 0.05)
                z = np.clip((xx - mean) / std, -5, 5)
                z = np.column_stack([np.ones(len(z)), z])
                penalty = np.eye(z.shape[1]) * 10.0
                penalty[0, 0] = 0
                beta = np.linalg.solve(z.T @ z + penalty, z.T @ np.clip(yy, -1, 1))
                model = mean, std, beta
                used = sample[mask.any(axis=1)]
                audit.append(
                    dict(
                        fit_date=date,
                        samples=len(yy),
                        last_feature_date=idx[used[-1]],
                        last_label_date=idx[used[-1] + 22],
                        penalty=10.0,
                        coefficients=beta.tolist(),
                    )
                )
        if model is not None:
            mean, std, beta = model
            z = np.clip((x[i] - mean) / std, -5, 5)
            predictions[i] = np.column_stack([np.ones(len(symbols)), z]) @ beta
            predictions[i, ~valid[i]] = np.nan
    return pd.DataFrame(predictions, index=idx, columns=symbols), pd.DataFrame(audit)


def _allocate(
    scores: pd.Series, allowed: pd.Series, volatility: pd.Series, cfg: dict, inverse_vol: bool
) -> pd.Series:
    symbols = list(scores.index)
    weights = pd.Series(0.0, index=symbols)
    buckets = {a["symbol"]: a["bucket"] for a in cfg["assets"]}
    ranked = scores[allowed & scores.notna()].sort_values(ascending=False, kind="stable")
    selected = []
    counts = {}
    # Limit overlapping domestic and US equity funds at selection time as well as allocation time.
    max_per_group = max(1, int(np.ceil(cfg["top_n"] * cfg["max_group_weight"])))
    for s in ranked.index:
        group = buckets[s]
        if counts.get(group, 0) >= max_per_group:
            continue
        selected.append(s)
        counts[group] = counts.get(group, 0) + 1
        if len(selected) == cfg["top_n"]:
            break
    if not selected:
        return weights
    raw = 1 / volatility[selected].clip(lower=0.08) if inverse_vol else pd.Series(1.0, index=selected)
    weights[selected] = (raw / raw.sum() * (1 - cfg["cash_buffer"])).clip(upper=cfg["max_asset_weight"])
    for group in set(buckets.values()):
        members = [s for s in symbols if buckets[s] == group]
        total = weights[members].sum()
        if total > cfg["max_group_weight"]:
            weights[members] *= cfg["max_group_weight"] / total
    return weights


def rotation_weights(
    data: dict, cfg: dict, features: dict | None = None, predictions: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = features if features is not None else build_features(data, cfg)
    family = cfg.get("rotation_family", "ensemble")
    if family not in {"dual", "quality", "ensemble", "ridge"}:
        raise ValueError(f"Unknown rotation family: {family}")
    if family == "ridge" and predictions is None:
        predictions, _ = ridge_predictions(f)
    close, vol = f["close"], f["vol"].clip(lower=0.05)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    bonds = [a["symbol"] for a in cfg["assets"] if a["bucket"] == "bond"]
    rank = {w: f["mom"][w].where(f["eligible"]).rank(axis=1, pct=True) for w in f["mom"]}
    base = (rank[63] + rank[126] + rank[252]) / 3
    specs = [(base, (f["mom"][126] + f["mom"][252]) / 2 > 0, 200)]
    if family == "quality":
        quality = (f["mom"][63] + f["mom"][126] + f["mom"][252]) / (3 * vol)
        score = 0.7 * quality.where(f["eligible"]).rank(axis=1, pct=True)
        score += 0.15 * f["efficiency"].where(f["eligible"]).rank(axis=1, pct=True)
        score += 0.15 * f["drawdown"].where(f["eligible"]).rank(axis=1, pct=True)
        specs = [(score, f["mom"][126] > 0, 120)]
    if family in {"ensemble", "ridge"}:
        specs = [
            ((rank[21] + rank[63]) / 2, f["mom"][63] > 0, 60),
            ((rank[63] + rank[126]) / 2, f["mom"][126] > 0, 120),
            (base, (f["mom"][126] + f["mom"][252]) / 2 > 0, 200),
        ]
        if family == "ridge":
            pr = predictions.rank(axis=1, pct=True)
            specs = [(0.75 * score + 0.25 * pr.fillna(score), gate, window) for score, gate, window in specs]
    returns = f["returns"].to_numpy()
    for i, date in enumerate(close.index):
        if not f["eligible"].iloc[i].any():
            continue
        sleeves = []
        for score, gate, window in specs:
            allowed = f["eligible"].iloc[i] & gate.iloc[i] & (f["trend"][window].iloc[i] > 0)
            allowed.loc[bonds] = False
            sleeves.append(_allocate(score.iloc[i], allowed, vol.iloc[i], cfg, family != "dual"))
        w = sum(sleeves) / len(sleeves)
        # Defensive fill is capped; failed bond trend leaves genuine zero-yield cash.
        remaining = max(0.0, 1 - cfg["cash_buffer"] - w.sum())
        for s in bonds:
            if f["eligible"].loc[date, s] and f["trend"][60].loc[date, s] > 0:
                added = min(remaining, cfg["max_asset_weight"])
                w[s] += added
                remaining -= added
        chosen = w.to_numpy() > 0
        if chosen.any():
            historical = returns[max(0, i - cfg["vol_window"] + 1) : i + 1, chosen]
            if len(historical) < cfg["vol_window"] or not np.isfinite(historical).all():
                w *= 0
            else:
                # Shrink covariance 25% towards diagonal to limit unstable correlation estimates.
                covariance = np.atleast_2d(np.cov(historical, rowvar=False)) * 252
                covariance = 0.75 * covariance + 0.25 * np.diag(np.diag(covariance))
                v = w.to_numpy()[chosen]
                risk = float(np.sqrt(max(0, v @ covariance @ v)))
                if risk > cfg["target_vol"]:
                    w *= cfg["target_vol"] / risk
        weights.iloc[i] = w
    if (weights < -1e-10).any().any() or (weights.sum(axis=1) > 1 + 1e-10).any():
        raise AssertionError("Long-only / no-leverage invariant failed")
    if weights.max().max() > cfg["max_asset_weight"] + 1e-10:
        raise AssertionError("Asset cap failed")
    diagnostics = []
    for s in close.columns:
        d = pd.DataFrame(
            {
                "date": close.index,
                "symbol": s,
                "volatility_60": vol[s],
                "downside_60": f["downside"][s],
                "trend_efficiency_63": f["efficiency"][s],
                "drawdown_126": f["drawdown"][s],
                "eligible": f["eligible"][s],
            }
        )
        for window, values in f["mom"].items():
            d[f"momentum_{window}"] = values[s].to_numpy()
        diagnostics.append(d)
    return weights, pd.concat(diagnostics, ignore_index=True)
