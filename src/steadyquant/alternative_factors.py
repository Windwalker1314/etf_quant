"""Small predeclared non-price factor family, using conservative disclosure times."""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {
    "revision": ["np_revision_63", "revision_breadth_63"],
    "surprise": ["incremental_surprise", "seasonal_sue"],
    "cash_quality": [
        "cash_profitability",
        "negative_accruals",
        "gross_profitability",
        "working_capital_discipline",
    ],
}
RAW_FACTORS = sum(GROUPS.values(), [])
FIN_VALUES = {
    "income": ["revenue", "oper_cost", "operate_profit", "n_income", "n_income_attr_p", "basic_eps"],
    "cashflow": ["net_profit", "n_cashflow_act", "c_pay_acq_const_fiolta"],
    "balance": ["total_assets", "accounts_receiv", "inventories", "goodwill"],
}


def end_of_day(series):
    return (
        pd.to_datetime(series, errors="coerce").dt.normalize()
        + pd.Timedelta(days=1)
        - pd.Timedelta(nanoseconds=1)
    )


def clean_conflicts(d, keys, values):
    """Drop ambiguous same-version facts; deterministic, never use source row order."""
    kept = d.drop_duplicates(keys + values)
    return kept[~kept.duplicated(keys, keep=False)].copy()


def prepare_financial(raw, kind):
    d = raw.copy()
    for col in ("ann_date", "f_ann_date", "end_date"):
        d[col] = pd.to_datetime(d[col], errors="coerce")
    d["available_at"] = end_of_day(d[["ann_date", "f_ann_date"]].max(axis=1))
    rt = d.report_type.astype(str)
    original = rt.isin(["1", "5"]) & d.update_flag.astype(str).eq("0")
    # Type 4 is a comparative statement explicitly published in a later year;
    # it cannot change factors before that later publication.
    comparative = rt.eq("4") & (d.available_at.dt.year > d.end_date.dt.year)
    d = d[(original | comparative) & d.available_at.notna() & d.end_date.notna()]
    d = d[d.available_at > d.end_date]
    for col in FIN_VALUES[kind]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    return clean_conflicts(
        d, ["ts_code", "end_date", "available_at"], FIN_VALUES[kind] + ["comp_type"]
    ).sort_values(["ts_code", "available_at", "end_date"])


def prepare_analysts(raw):
    d = raw.copy()
    d["reported_at"] = pd.to_datetime(d.report_date, errors="coerce")
    d["updated_at"] = pd.to_datetime(d.create_time, errors="coerce")
    d["available_at"] = pd.concat([end_of_day(d.reported_at), d.updated_at], axis=1).max(axis=1)
    lag = (d.updated_at - d.reported_at).dt.total_seconds() / 86400
    good = d.quarter.str.fullmatch(r"\d{4}Q[1-4]", na=False)
    good &= d.updated_at.notna() & d.reported_at.notna() & lag.between(0, 7)
    d = d[good & d.org_name.notna()].copy()
    d["np"] = pd.to_numeric(d["np"], errors="coerce")
    # np is CNY 10,000 in this source; do not manufacture it from EPS.
    d["np_cny"] = d["np"] * 10000
    return clean_conflicts(
        d, ["ts_code", "org_name", "quarter", "reported_at", "available_at"], ["np_cny"]
    ).sort_values(["available_at", "reported_at"])


def latest_analyst(d, cutoff, fiscal_period):
    f = d[
        (d.available_at < cutoff)
        & (d.reported_at >= cutoff.normalize() - pd.Timedelta(days=180))
        & (d.quarter == fiscal_period)
    ]
    # A new report with a missing number supersedes the old number; stale old
    # estimates must not silently survive a new missing-value observation.
    f = f.sort_values(["reported_at", "available_at"]).drop_duplicates(["ts_code", "org_name"], keep="last")
    return f.dropna(subset=["np_cny"])


def analyst_revision(d, date, prior_date):
    fiscal = f"{date.year}Q4"
    cutoff = date.normalize() + pd.Timedelta(hours=18, minutes=30)
    before = prior_date.normalize() + pd.Timedelta(hours=18, minutes=30)
    new = latest_analyst(d, cutoff, fiscal).set_index(["ts_code", "org_name"])
    old = latest_analyst(d, before, fiscal).set_index(["ts_code", "org_name"])
    pairs = new[["np_cny", "available_at"]].join(old[["np_cny"]], lsuffix="_new", rsuffix="_old", how="inner")
    pairs = pairs[pairs.np_cny_old.abs() >= 1_000_000]  # avoid near-zero profit percentage explosions
    pairs["revision"] = (pairs.np_cny_new - pairs.np_cny_old) / pairs.np_cny_old.abs()
    pairs["direction"] = np.sign(pairs.np_cny_new - pairs.np_cny_old)
    if pairs.empty:
        return pd.DataFrame(
            columns=["np_revision_63", "revision_breadth_63", "paired_brokers", "analyst_available_at"]
        )
    result = pairs.groupby(level=0).agg(
        np_revision_63=("revision", "median"),
        revision_breadth_63=("direction", "mean"),
        paired_brokers=("revision", "count"),
        analyst_available_at=("available_at", "max"),
    )
    result.loc[result.paired_brokers < 3, ["np_revision_63", "revision_breadth_63"]] = np.nan
    return result


def financial_asof(d, cutoff):
    f = d[d.available_at < cutoff]
    return f.sort_values("available_at").drop_duplicates(["ts_code", "end_date"], keep="last")


def value_at(frame, period, field):
    if frame is None or period not in frame.index or field not in frame:
        return np.nan
    return float(frame.at[period, field])


def previous_year(period):
    return period - pd.DateOffset(years=1)


def ttm(frame, period, field):
    current = value_at(frame, period, field)
    if period.month == 12:
        return current
    return (
        current
        + value_at(frame, pd.Timestamp(period.year - 1, 12, 31), field)
        - value_at(frame, previous_year(period), field)
    )


def single_quarter(frame, period, field="n_income_attr_p"):
    current = value_at(frame, period, field)
    if period.month == 3:
        return current
    prior = (period.to_period("Q") - 1).end_time.normalize()
    return current - value_at(frame, prior, field)


def seasonal_sue(income, period):
    delta = single_quarter(income, period) - single_quarter(income, previous_year(period))
    changes = []
    for shift in range(1, 9):
        prior = (period.to_period("Q") - shift).end_time.normalize()
        v = single_quarter(income, prior) - single_quarter(income, previous_year(prior))
        if np.isfinite(v):
            changes.append(v)
    if len(changes) < 6:
        return np.nan
    vol = np.std(changes, ddof=1)
    return delta / vol if np.isfinite(delta) and vol > 1_000_000 else np.nan


def financial_factors(tables, cutoff):
    available = {k: financial_asof(tables[k], cutoff) for k in FIN_VALUES}
    grouped = {k: {s: f.set_index("end_date") for s, f in d.groupby("ts_code")} for k, d in available.items()}
    rows = []
    for symbol, inc in grouped["income"].items():
        cf, bs = grouped["cashflow"].get(symbol), grouped["balance"].get(symbol)
        row = dict(ts_code=symbol)
        latest_inc = inc.index.max()
        if (cutoff.normalize() - latest_inc).days <= 550:
            row["seasonal_sue"] = seasonal_sue(inc, latest_inc)
            row["sue_available_at"] = inc.loc[inc.index <= latest_inc, "available_at"].max()
            row["sue_period"] = latest_inc
        if cf is not None and bs is not None:
            common = inc.index.intersection(cf.index).intersection(bs.index)
            if len(common):
                period = common.max()
                previous = previous_year(period)
                if (cutoff.normalize() - period).days <= 550 and str(inc.at[period, "comp_type"]) == "1":
                    assets = (
                        value_at(bs, period, "total_assets") + value_at(bs, previous, "total_assets")
                    ) / 2
                    if assets > 0:
                        cash = ttm(cf, period, "n_cashflow_act")
                        profit = ttm(inc, period, "n_income")
                        gross = ttm(inc, period, "revenue") - ttm(inc, period, "oper_cost")
                        row.update(
                            cash_profitability=cash / assets,
                            negative_accruals=(cash - profit) / assets,
                            gross_profitability=gross / assets,
                        )
                        cur_wc = value_at(bs, period, "accounts_receiv") + value_at(bs, period, "inventories")
                        old_wc = value_at(bs, previous, "accounts_receiv") + value_at(
                            bs, previous, "inventories"
                        )
                        old_sales = value_at(inc, previous, "revenue")
                        if old_wc > 0 and old_sales > 0:
                            row["working_capital_discipline"] = -(
                                cur_wc / old_wc - value_at(inc, period, "revenue") / old_sales
                            )
                        row["quality_period"] = period
                        row["quality_available_at"] = max(
                            f.loc[f.index <= period, "available_at"].max() for f in (inc, cf, bs)
                        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("ts_code") if rows else pd.DataFrame()


def industry_asof(raw, date):
    d = raw.copy()
    d["start"] = pd.to_datetime(d.in_date, errors="coerce")
    d["end"] = pd.to_datetime(d.out_date, errors="coerce")
    d = d[(d.start < date) & (d.end.isna() | (d.end >= date))]
    pairs = d[["ts_code", "l1_code"]].drop_duplicates()
    pairs = pairs[~pairs.ts_code.duplicated(keep=False)]
    return pairs.set_index("ts_code").l1_code


def factor_scores(frame):
    out = frame.copy()
    for col in RAW_FACTORS:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[col + "_rank"] = out[col].rank(pct=True)
    for name, cols in GROUPS.items():
        ranks = out[[c + "_rank" for c in cols]]
        minimum = {"revision": 2, "surprise": 1, "cash_quality": 3}[name]
        out[name] = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= minimum)
    out["balanced"] = out[list(GROUPS)].mean(axis=1).where(out[list(GROUPS)].notna().sum(axis=1) >= 2)
    return out
