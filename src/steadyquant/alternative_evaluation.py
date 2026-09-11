"""Descriptive factor tests, separate from feature construction and live decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alternative_factors import RAW_FACTORS


def safe_corr(x, y):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def block_mean_interval(series, block, n=1000):
    # Missing monthly slots remain missing, preserving calendar block spacing.
    a = np.asarray(series, float)
    if np.isfinite(a).sum() < 24:
        return (None, None)
    rng = np.random.default_rng(2031)
    means = []
    for _ in range(n):
        starts = rng.integers(0, len(a), size=int(np.ceil(len(a) / block)))
        sample = a[((starts[:, None] + np.arange(block)) % len(a)).ravel()[: len(a)]]
        if np.isfinite(sample).any():
            means.append(np.nanmean(sample))
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def factor_tests(panel, market, snapshot):
    rows = []
    opens = market["adjusted_open"]
    factors = RAW_FACTORS + ["revision", "surprise", "cash_quality", "balanced", "momentum_raw"]
    by_date = {d: f.set_index("symbol") for d, f in panel.groupby("date")}
    for horizon in (20, 60):
        future = opens.shift(-horizon - 1) / opens.shift(-1) - 1
        # Preserve terminal losses in diagnostic labels instead of simply
        # dropping a delisted name whose horizon-end quote is absent.
        for row in snapshot["metadata"].dropna(subset=["delist_date"]).itertuples():
            if row.ts_code not in future or str(row.delist_date) in ("", "None"):
                continue
            terminal = pd.Timestamp(row.delist_date)
            for i, date in enumerate(opens.index[: -horizon - 1]):
                if opens.index[i + 1] < terminal <= opens.index[i + horizon + 1] and pd.notna(
                    opens.iloc[i + 1][row.ts_code]
                ):
                    future.loc[date, row.ts_code] = -1.0
        for date, base in by_date.items():
            base = base.copy()
            base["forward"] = future.loc[date]
            for factor in factors:
                d = base.dropna(subset=[factor, "forward"])
                record = dict(
                    date=date,
                    horizon=horizon,
                    factor=factor,
                    observations=len(d),
                    raw_ic=np.nan,
                    controlled_ic=np.nan,
                    top_mean=np.nan,
                    bottom_mean=np.nan,
                    top_minus_bottom=np.nan,
                    controlled_observations=0,
                )
                if len(d) >= 20:
                    x, y = d[factor].rank(pct=True), d.forward.rank(pct=True)
                    record["raw_ic"] = safe_corr(x, y)
                    quantile = d[factor].rank(pct=True)
                    # Ties stay together. Do not break flat factors using symbols.
                    top, bottom = d.loc[quantile >= 0.8, "forward"], d.loc[quantile <= 0.2, "forward"]
                    if len(top) >= 3 and len(bottom) >= 3 and d[factor].nunique() > 2:
                        record.update(
                            top_mean=float(top.mean()),
                            bottom_mean=float(bottom.mean()),
                            top_minus_bottom=float(top.mean() - bottom.mean()),
                        )
                    control = d.dropna(subset=["industry", "log_market_cap", "momentum_raw"])
                    record["controlled_observations"] = len(control)
                    z = pd.concat(
                        [
                            control[["log_market_cap", "momentum_raw"]].rank(pct=True),
                            pd.get_dummies(control.industry, drop_first=True, dtype=float),
                        ],
                        axis=1,
                    )
                    z.insert(0, "intercept", 1.0)
                    if len(control) >= max(30, z.shape[1] + 10):
                        x = control[factor].rank(pct=True).to_numpy()
                        y = control.forward.rank(pct=True).to_numpy()
                        matrix = z.to_numpy(float)
                        rx = x - matrix @ np.linalg.lstsq(matrix, x, rcond=None)[0]
                        ry = y - matrix @ np.linalg.lstsq(matrix, y, rcond=None)[0]
                        record["controlled_ic"] = safe_corr(rx, ry)
                rows.append(record)
    monthly = pd.DataFrame(rows)
    stats = []
    for (factor, horizon), d in monthly.groupby(["factor", "horizon"]):
        d = d.sort_values("date")
        for name in ["raw_ic", "controlled_ic", "top_minus_bottom"]:
            v = d[name]
            lo, hi = block_mean_interval(v, 3 if horizon == 20 else 6)
            stats.append(
                dict(
                    factor=factor,
                    horizon=horizon,
                    metric=name,
                    months=int(v.notna().sum()),
                    mean=float(v.mean()) if v.notna().any() else None,
                    lower95=lo,
                    upper95=hi,
                    positive_fraction=float(v.dropna().gt(0).mean()) if v.notna().any() else None,
                    median_cross_section=float(d.observations.median()),
                    note="Retrospective monthly diagnostic; open-based adjusted-price labels are not executable short portfolios; circular monthly block interval, not multiplicity-adjusted",
                )
            )
    return monthly, pd.DataFrame(stats)


def portfolio_targets(panel, etf_w, stock_prices, etf_data, family, top_n=8):
    dates = etf_w.index.intersection(stock_prices.index)
    symbols = list(etf_w.columns) + sorted(stock_prices.columns)
    targets = pd.DataFrame(0.0, index=dates, columns=symbols)
    market = etf_data["510300.SH"].set_index("date")
    p = (market.close * market.adj_factor).reindex(dates)
    regime = ((p > p.rolling(126).mean()).astype(float) + (p > p.rolling(252).mean()).astype(float)) / 2
    picks, selection, diagnostics = [], [], []
    groups = {d: f.copy() for d, f in panel.groupby("date")}
    for i, date in enumerate(dates):
        new_month = i == 0 or date.month != dates[i - 1].month
        if new_month:
            f = groups.get(date, pd.DataFrame())
            picks, industry_counts = [], {}
            if not f.empty:
                if family == "matched_price_control":
                    f[family] = 0.6 * f.momentum_raw.rank(pct=True) + 0.4 * (-f.volatility).rank(pct=True)
                usable = f.dropna(subset=[family, "industry"]).sort_values(
                    [family, "symbol"], ascending=[False, True]
                )
                # Flat scores cannot distinguish stocks; avoid turning ticker
                # order into an unadvertised selector.
                if usable[family].nunique() >= 3:
                    for row in usable.itertuples():
                        if industry_counts.get(row.industry, 0) >= 2:
                            continue
                        picks.append(row.symbol)
                        industry_counts[row.industry] = industry_counts.get(row.industry, 0) + 1
                        if len(picks) == top_n:
                            break
                count = len(picks)
                if count < top_n:
                    picks = []
                diagnostics.append(
                    dict(
                        date=date,
                        family=family,
                        factor_available=len(usable),
                        selected=len(picks),
                        insufficient=count < top_n,
                    )
                )
                for rank, symbol in enumerate(picks, 1):
                    row = usable[usable.symbol == symbol].iloc[0]
                    selection.append(
                        dict(
                            date=date,
                            family=family,
                            symbol=symbol,
                            rank=rank,
                            score=float(row[family]),
                            industry=row.industry,
                        )
                    )
        budget = 0.25 * regime.loc[date] if picks else 0.0
        targets.loc[date, etf_w.columns] = etf_w.loc[date] * (1 - budget)
        if picks:
            targets.loc[date, picks] = budget / len(picks)
    return targets, pd.DataFrame(selection), pd.DataFrame(diagnostics)
