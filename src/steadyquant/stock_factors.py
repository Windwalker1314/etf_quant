"""Point-in-time monthly cross-sectional ranks, with explicit missing-data exclusions."""

from __future__ import annotations

import numpy as np
import pandas as pd

FAMILIES = {
    "quality_value": {"value": 0.4, "quality": 0.4, "low_vol": 0.2},
    "momentum_defensive": {"momentum": 0.5, "low_vol": 0.3, "quality": 0.2},
    "ensemble": {"value": 0.25, "quality": 0.25, "momentum": 0.25, "low_vol": 0.25},
}


def latest_reports(reports, date):
    d = reports.copy()
    d = d[(d.ann_date < date) & (d.end_date <= date) & (d.update_flag.astype(str) == "0")]
    d = d[d.end_date >= date - pd.Timedelta(days=550)]
    # Conflicting same-date original reports cannot safely be resolved by row order.
    keys = ["ts_code", "ann_date", "end_date"]
    d = d.drop_duplicates()
    d = d[~d.duplicated(keys, keep=False)]
    return (
        d.sort_values(["end_date", "ann_date"]).drop_duplicates("ts_code", keep="last").set_index("ts_code")
    )


def build_stock_factors(snapshot, dates):
    raw = snapshot["daily"].copy()
    raw["date"] = pd.to_datetime(raw.trade_date)
    adj = snapshot["adj_factor"].copy()
    merged = raw.merge(adj, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
    close = merged.pivot(index="date", columns="ts_code", values="close").reindex(dates)
    af = merged.pivot(index="date", columns="ts_code", values="adj_factor").reindex_like(close)
    price = close * af
    ret = price.pct_change(fill_method=None)
    vol = ret.rolling(63, min_periods=50).std() * np.sqrt(252)
    mom = (price.shift(21) / price.shift(126) - 1 + price.shift(21) / price.shift(252) - 1) / 2
    amount = merged.pivot(index="date", columns="ts_code", values="amount").reindex_like(close) * 1000
    eligible = close.notna() & af.notna() & (price.notna().rolling(270, min_periods=253).sum() >= 253)
    eligible &= amount.rolling(20, min_periods=15).mean() >= 30_000_000
    volume = merged.pivot(index="date", columns="ts_code", values="vol").reindex_like(close)
    eligible &= volume > 0
    # Availability/board permission/affordability are fixed ex-ante rules, not return filters.
    eligible &= close * 100 <= 200000 * 0.25 / 8
    reports = snapshot["financials"].copy()
    for col in ("ann_date", "end_date"):
        reports[col] = pd.to_datetime(reports[col], errors="coerce")
    members = snapshot["members"].copy()
    members["date"] = pd.to_datetime(members.trade_date)
    member_dates = sorted(members.date.unique())
    names = snapshot["namechange"].copy()
    for col in ("start_date", "end_date"):
        names[col] = pd.to_datetime(names[col], errors="coerce")
    values = snapshot["valuations"].copy()
    values["date"] = pd.to_datetime(values.trade_date)
    frames, audits = [], []
    monthly = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    for date in monthly:
        if date < pd.Timestamp("2015-01-01"):
            continue
        prior = [pd.Timestamp(x) for x in member_dates if x < date]
        if not prior or (date - prior[-1]).days > 62:
            audits.append(dict(date=date, status="missing/stale membership", eligible=0))
            continue
        pool = members.loc[members.date == prior[-1], "con_code"]
        ok = eligible.loc[date]
        stocks = [s for s in pool if s in ok and ok[s] and s.startswith(("60", "00"))]
        bad = names[
            (names.start_date <= date)
            & (names.end_date.isna() | (names.end_date >= date))
            & names.name.str.contains("ST|退", case=False, na=False)
        ].ts_code
        stocks = [s for s in stocks if s not in set(bad)]
        val = values[values.date == date].drop_duplicates("ts_code").set_index("ts_code")
        fin = latest_reports(reports, date)
        f = pd.DataFrame(index=pd.Index(stocks, name="symbol"))
        f["ep"] = 1 / pd.to_numeric(val.pe_ttm, errors="coerce").where(lambda x: x > 0)
        f["bp"] = 1 / pd.to_numeric(val.pb, errors="coerce").where(lambda x: x > 0)
        # Reported ratios are year-to-date; normalize the elapsed fiscal months
        # so a newly published Q1 report is not mechanically ranked below FY.
        annualizer = 12 / fin.end_date.dt.month.where(fin.end_date.dt.month.isin([3, 6, 9, 12]))
        f["roe"] = pd.to_numeric(fin.roe, errors="coerce") * annualizer
        f["roa"] = pd.to_numeric(fin.roa, errors="coerce") * annualizer
        f["report_ann_date"] = fin.ann_date
        f["report_end_date"] = fin.end_date
        f["momentum_raw"] = mom.loc[date]
        f["volatility"] = vol.loc[date]
        f = f.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["ep", "bp", "roe", "roa", "momentum_raw", "volatility"]
        )
        f = f[(f.roe > 0) & (f.volatility > 0)]
        f["value"] = (f.ep.rank(pct=True) + f.bp.rank(pct=True)) / 2
        f["quality"] = (f.roe.rank(pct=True) + f.roa.rank(pct=True)) / 2
        f["momentum"] = f.momentum_raw.rank(pct=True)
        f["low_vol"] = (-f.volatility).rank(pct=True)
        for family, weights in FAMILIES.items():
            f[family] = sum(f[col] * w for col, w in weights.items())
        f["date"] = date
        f["membership_date"] = prior[-1]
        frames.append(f.reset_index())
        audits.append(
            dict(
                date=date,
                status="ok" if len(f) >= 8 else "insufficient factors",
                eligible=len(f),
                membership_date=prior[-1],
            )
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audits), price, ret


def composite_targets(factors, etf_targets, stock_prices, etf_data, family, fraction, top_n=8):
    dates = etf_targets.index.intersection(stock_prices.index)
    all_symbols = list(etf_targets.columns) + sorted(stock_prices.columns)
    out = pd.DataFrame(0.0, index=dates, columns=all_symbols)
    index_frame = etf_data["510300.SH"].set_index("date")
    index_price = (index_frame.close * index_frame.adj_factor).reindex(dates)
    regime = (
        (index_price > index_price.rolling(126).mean()).astype(float)
        + (index_price > index_price.rolling(252).mean()).astype(float)
    ) / 2
    picks = []
    rows = []
    for date in dates:
        f = factors[factors.date == date]
        if not f.empty:
            picks = f.sort_values([family, "symbol"], ascending=[False, True]).head(top_n).symbol.tolist()
            if len(picks) < top_n:
                picks = []
            rows.extend(dict(date=date, symbol=s, family=family, rank=i + 1) for i, s in enumerate(picks))
        elif date == dates[0] or date.month != dates[dates.get_loc(date) - 1].month:
            picks = []
        budget = fraction * regime.loc[date] if picks else 0.0
        # In bearish regimes, the unused stock allocation returns to the diversified ETF core.
        out.loc[date, etf_targets.columns] = etf_targets.loc[date] * (1 - budget)
        if picks:
            out.loc[date, picks] = budget / top_n
    return out, pd.DataFrame(rows), regime
