"""Summarize sampled source quality; this does not evaluate investment returns."""

import json

import pandas as pd

from steadyquant.config import ROOT, write_json

out = ROOT / "data/research/alternative_factors/20260910"
audit = json.loads((out / "audit.json").read_text())
summary = {
    "scope": "Capability and data-quality samples only; no alpha/backtest conclusion",
    "queries": len(audit["samples"]),
    "successful_queries": sum("error" not in s for s in audit["samples"]),
    "unique_apis": len({s["api"] for s in audit["samples"]}),
    "capped_samples": [s["label"] for s in audit["samples"] if s.get("at_requested_cap")],
    "forecast_samples": {},
}
for name in ["rc_2016", "rc_2020", "rc_recent", "rc_single"]:
    f = pd.read_parquet(out / f"{name}.parquet")
    reported = pd.to_datetime(f.report_date, errors="coerce")
    updated = pd.to_datetime(f.create_time, errors="coerce")
    lag = (updated - reported).dt.total_seconds() / 86400
    summary["forecast_samples"][name] = {
        "rows": len(f),
        "stocks": int(f.ts_code.nunique()),
        "eps_missing": int(f.eps.isna().sum()),
        "np_missing": int(f.np.isna().sum()),
        "lag_days_median": float(lag.median()),
        "lag_days_max": float(lag.max()),
        "lag_above_7_days": int((lag > 7).sum()),
        "invalid_quarter": int((~f.quarter.str.fullmatch(r"\d{4}Q[1-4]", na=False)).sum()),
    }
summary["financial_samples"] = {}
for name in ["income", "cashflow", "balancesheet", "forecast"]:
    f = pd.read_parquet(out / f"{name}.parquet")
    summary["financial_samples"][name] = {
        "update_flags": f.update_flag.value_counts(dropna=False).to_dict(),
        "rows": len(f),
    }
f = pd.read_parquet(out / "rc_single.parquet")
f["reported"] = pd.to_datetime(f.report_date)
f["available"] = pd.concat([pd.to_datetime(f.create_time), f.reported], axis=1).max(axis=1)
f = f[(f.quarter == "2026Q4") & f.np.notna()].copy()
keys = ["org_name", "report_date", "quarter", "create_time"]
f = f.drop_duplicates()
f = f[~f.duplicated(keys, keep=False)]
cutoffs = [pd.Timestamp("2026-06-09 18:30"), pd.Timestamp("2026-09-09 18:30")]
snapshots = []
for cutoff in cutoffs:
    valid = f[(f.available < cutoff) & (f.reported >= cutoff.normalize() - pd.Timedelta(days=180))]
    snapshots.append(
        valid.sort_values(["available", "reported"])
        .drop_duplicates("org_name", keep="last")
        .set_index("org_name")
    )
a, b = snapshots
common = a.index.intersection(b.index)
delta = (b.loc[common, "np"] - a.loc[common, "np"]) / a.loc[common, "np"].abs().replace(0, float("nan"))
summary["single_stock_illustration"] = {
    "symbol": "600887.SH",
    "fiscal_period": "2026Q4",
    "as_of": [str(c) for c in cutoffs],
    "matched_brokers": len(common),
    "median_revision": float(delta.median()),
    "up": int((delta > 0).sum()),
    "down": int((delta < 0).sum()),
    "unchanged": int((delta == 0).sum()),
    "purpose": "Data construction illustration; not a recommendation, ranking, or return test",
}
write_json(out / "findings.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
