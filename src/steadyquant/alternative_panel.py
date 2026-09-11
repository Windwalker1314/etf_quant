"""Disclosure events and monthly point-in-time research panels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .alternative_factors import (
    RAW_FACTORS,
    analyst_revision,
    clean_conflicts,
    end_of_day,
    factor_scores,
    financial_factors,
    industry_asof,
    prepare_analysts,
    prepare_financial,
)


def disclosure_surprises(raw, income, balance):
    """New information beyond prior company guidance; no ambiguous NP-scope join.

    Analyst NP is only used for within-source paired revisions. The analyst and
    express docs do not explicitly certify parent/consolidated profit scope, so
    neither is silently compared with parent-profit guidance/actuals here.
    """
    f = raw["forecast"].copy()
    f["ann"] = pd.to_datetime(f.ann_date, errors="coerce")
    f["first"] = pd.to_datetime(f.first_ann_date, errors="coerce")
    f["end_date"] = pd.to_datetime(f.end_date, errors="coerce")
    f["available_at"] = end_of_day(f.ann)
    dated = f.update_flag.astype(str).eq("0") | (f.ann > f["first"])
    f = f[dated & f.ann.notna() & f.end_date.notna()].copy()
    f["lower"] = pd.to_numeric(f.net_profit_min, errors="coerce") * 10000
    f["upper"] = pd.to_numeric(f.net_profit_max, errors="coerce") * 10000
    f = f[f.lower.notna() & f.upper.notna() & (f.lower <= f.upper)]
    f["source"] = "guidance"
    actual = income[income.report_type.astype(str).isin(["1", "5"])].copy()
    actual["lower"] = actual.n_income_attr_p
    actual["upper"] = actual.n_income_attr_p
    actual["source"] = "initial_parent_actual"
    cols = ["ts_code", "end_date", "available_at", "lower", "upper", "source"]
    events = pd.concat([f[cols], actual[cols]], ignore_index=True).dropna(subset=["lower", "upper"])
    events = clean_conflicts(events, ["ts_code", "end_date", "available_at"], ["lower", "upper"])
    bs = {s: d for s, d in balance.groupby("ts_code")}
    express = raw["express"].copy()
    express["end_date"] = pd.to_datetime(express.end_date, errors="coerce")
    express["available_at"] = end_of_day(express.ann_date)
    quick = {(s, p): d.available_at for (s, p), d in express.groupby(["ts_code", "end_date"])}
    rows = []
    for (symbol, period), group in events.groupby(["ts_code", "end_date"]):
        prior = None
        for event in group.sort_values("available_at").itertuples():
            row = dict(
                symbol=symbol,
                period=period,
                available_at=event.available_at,
                source=event.source,
                lower=event.lower,
                upper=event.upper,
                incremental_surprise=np.nan,
                reason="no_prior_comparable_company_disclosure",
            )
            if prior is not None:
                row["prior_available_at"] = prior.available_at
                row["prior_lower"], row["prior_upper"] = prior.lower, prior.upper
                known_quick = quick.get((symbol, period), pd.Series(dtype="datetime64[ns]"))
                intervening = (
                    (known_quick >= prior.available_at) & (known_quick <= event.available_at)
                ).any()
                if intervening:
                    row["reason"] = "intervening_express_profit_scope_not_certified"
                else:
                    available_bs = bs.get(symbol, pd.DataFrame())
                    if not available_bs.empty:
                        available_bs = available_bs[
                            (available_bs.available_at < event.available_at.normalize())
                            & (available_bs.end_date < event.available_at)
                        ]
                    if not available_bs.empty:
                        asset_row = available_bs.sort_values(["end_date", "available_at"]).iloc[-1]
                        assets = asset_row.total_assets
                        if assets > 0 and (event.available_at.normalize() - asset_row.end_date).days <= 550:
                            # Overlapping guidance/actual intervals contain no
                            # unambiguous incremental sign; score zero, not a midpoint guess.
                            gap = (
                                event.lower - prior.upper
                                if event.lower > prior.upper
                                else event.upper - prior.lower
                                if event.upper < prior.lower
                                else 0.0
                            )
                            row["incremental_surprise"] = gap / assets
                            row["reason"] = "nonoverlap_interval_gap_over_known_assets"
                            row["asset_available_at"] = asset_row.available_at
            rows.append(row)
            prior = event
    return pd.DataFrame(rows).sort_values(["symbol", "available_at", "period"])


def market_panels(snapshot, dates):
    d = snapshot["daily"].copy()
    d["date"] = pd.to_datetime(d.trade_date)
    a = snapshot["adj_factor"].copy()
    a["date"] = pd.to_datetime(a.trade_date)
    d = d.merge(a[["ts_code", "date", "adj_factor"]], on=["ts_code", "date"], validate="one_to_one")
    panel = {
        c: d.pivot(index="date", columns="ts_code", values=c).reindex(dates)
        for c in ["close", "open", "vol", "amount", "adj_factor"]
    }
    panel["price"] = panel["close"] * panel["adj_factor"]
    panel["adjusted_open"] = panel["open"] * panel["adj_factor"]
    price = panel["price"]
    panel["momentum"] = (price.shift(21) / price.shift(126) - 1 + price.shift(21) / price.shift(252) - 1) / 2
    panel["volatility"] = price.pct_change(fill_method=None).rolling(63, min_periods=50).std()
    panel["eligible"] = price.notna().rolling(270, min_periods=253).sum() >= 253
    panel["eligible"] &= panel["amount"].rolling(20, min_periods=15).mean() * 1000 >= 30_000_000
    panel["eligible"] &= (panel["close"] * 100 <= 6250) & (panel["vol"] > 0) & price.notna()
    return panel


def build_panel(snapshot, raw, dates, information_delay=0, start="2023-01-01"):
    prepared = {k: prepare_financial(raw[k], k) for k in ("income", "cashflow", "balance")}
    analysts = prepare_analysts(raw["analyst"])
    events = disclosure_surprises(raw, prepared["income"], prepared["balance"])
    panels = market_panels(snapshot, dates)
    members = snapshot["members"].copy()
    members["date"] = pd.to_datetime(members.trade_date)
    md = sorted(members.date.unique())
    names = snapshot["namechange"].copy()
    names["start"] = pd.to_datetime(names.start_date, errors="coerce")
    names["end"] = pd.to_datetime(names.end_date, errors="coerce")
    vals = snapshot["valuations"].copy()
    vals["date"] = pd.to_datetime(vals.trade_date)
    rows, coverage = [], []
    months = pd.Series(dates, index=dates).groupby(dates.to_period("M")).first()
    for date in months:
        if date < pd.Timestamp(start):
            continue
        loc = dates.get_loc(date)
        if loc < 63 + information_delay:
            continue
        info_date = dates[loc - information_delay]
        cutoff = info_date.normalize() + pd.Timedelta(hours=18, minutes=30)
        observed = [pd.Timestamp(x) for x in md if x < date]
        if not observed or (date - observed[-1]).days > 62:
            continue
        pool = members.loc[members.date == observed[-1], "con_code"]
        ok = panels["eligible"].loc[date]
        stocks = [s for s in pool if s in ok.index and ok[s] and s.startswith(("00", "60"))]
        bad = names[
            (names.start <= date)
            & (names.end.isna() | (names.end >= date))
            & names.name.str.contains("ST|退", case=False, na=False)
        ].ts_code
        stocks = sorted(set(stocks) - set(bad))
        f = pd.DataFrame(index=pd.Index(stocks, name="symbol"))
        rev = analyst_revision(analysts, info_date, dates[loc - information_delay - 63])
        quality = financial_factors(prepared, cutoff)
        f = f.join(rev).join(quality)
        known_events = events[
            (events.available_at < cutoff) & (events.available_at >= cutoff - pd.Timedelta(days=90))
        ]
        known_events = (
            known_events.sort_values(["available_at", "period"])
            .drop_duplicates("symbol", keep="last")
            .set_index("symbol")
        )
        f["incremental_surprise"] = known_events.incremental_surprise
        f["surprise_available_at"] = known_events.available_at
        f["industry"] = industry_asof(raw["industry"], info_date)
        f["momentum_raw"] = panels["momentum"].loc[date]
        f["volatility"] = panels["volatility"].loc[date]
        v = vals[vals.date == date].drop_duplicates("ts_code").set_index("ts_code")
        f["market_cap"] = pd.to_numeric(v.total_mv, errors="coerce") * 10000
        f["log_market_cap"] = np.log(f.market_cap.where(f.market_cap > 0))
        f = factor_scores(f)
        f["date"], f["information_date"], f["membership_date"] = date, info_date, observed[-1]
        for col in [
            "analyst_available_at",
            "sue_available_at",
            "quality_available_at",
            "surprise_available_at",
        ]:
            if col in f and (f[col].dropna() >= cutoff).any():
                raise AssertionError("Future disclosure in factor panel")
        coverage.append(
            dict(
                date=date,
                eligible=len(f),
                industry_known=int(f.industry.notna().sum()),
                **{
                    col: int(f[col].notna().sum())
                    for col in RAW_FACTORS + ["revision", "surprise", "cash_quality", "balanced"]
                },
            )
        )
        rows.append(f.reset_index())
    result = pd.concat(rows, ignore_index=True)
    audit = {
        "financial_rows_raw": {k: len(raw[k]) for k in prepared},
        "financial_rows_usable": {k: len(v) for k, v in prepared.items()},
        "analyst_rows_raw": len(raw["analyst"]),
        "analyst_rows_usable": len(analysts),
        "incremental_event_reasons": events.reason.value_counts().to_dict(),
        "event_profit_scope": "Parent company guidance and original actuals; analyst NP and express numerics not cross-joined due insufficient scope documentation",
    }
    return result, pd.DataFrame(coverage), panels, events, audit
