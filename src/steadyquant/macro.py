"""Dated valuation/rate research inputs; never substitute current fundamentals for history."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import ROOT, write_json
from .data import DataError, TushareProvider

MACRO_ROOT = ROOT / "data/macro"
SOURCES = {
    "shibor": ("shibor", {}),
    **{s: ("index_dailybasic", {"ts_code": s}) for s in ("000300.SH", "000905.SH", "399006.SZ")},
}


def sync_macro(end: str = "20260909") -> dict:
    p = TushareProvider()
    MACRO_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (api, params) in SOURCES.items():
        frames = []
        for year in range(2013, int(end[:4]) + 1):
            lo, hi = f"{year}0101", min(end, f"{year}1231")
            d = p.query(api, start_date=lo, end_date=hi, **params)
            if d.empty:
                raise DataError(f"{name}: empty year {year}; refusing partial macro history")
            date_col = "date" if name == "shibor" else "trade_date"
            if len(d) > 366 or not d[date_col].astype(str).between(lo, hi).all():
                raise DataError(f"{name}: date partition violated")
            if name != "shibor" and not d.ts_code.eq(name).all():
                raise DataError("Macro symbol mismatch")
            d = d.rename(columns={date_col: "date"})
            d["date"] = pd.to_datetime(d.date, format="%Y%m%d")
            frames.append(d)
        result = pd.concat(frames).sort_values("date").reset_index(drop=True)
        if result.date.duplicated().any():
            raise DataError(f"{name}: duplicate macro dates")
        path = MACRO_ROOT / f"{name}.parquet"
        result.to_parquet(path, index=False)
        manifest[name] = dict(
            rows=len(result),
            start=str(result.date.min().date()),
            end=str(result.date.max().date()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        print(f"Macro {name}: {len(result)} observations", flush=True)
    manifest["point_in_time"] = (
        "Dated vendor historical series, lagged one full market session in signals. No revision-vintage guarantee; not verified unrevised point-in-time fundamentals."
    )
    write_json(MACRO_ROOT / "manifest.json", manifest)
    return manifest


def load_macro() -> dict[str, pd.DataFrame]:
    import json

    manifest = json.loads((MACRO_ROOT / "manifest.json").read_text())
    out = {}
    for name in SOURCES:
        path = MACRO_ROOT / f"{name}.parquet"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
            raise DataError(f"Macro checksum mismatch: {name}")
        out[name] = pd.read_parquet(path)
    return out


def aligned_macro(dates: pd.DatetimeIndex, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Backward as-of with <=7 calendar-day age, then one complete session lag."""
    out = pd.DataFrame(index=dates)
    for key, frame in raw.items():
        fields = ["3m", "on", "1y"] if key == "shibor" else ["pe_ttm", "pb"]
        source = frame.set_index("date")[fields].apply(pd.to_numeric, errors="coerce").sort_index()
        a = source.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=7)).shift(1)
        if key == "shibor":
            out["rate_change"] = a["3m"] - a["3m"].shift(63)
            out["funding_stress"] = a["on"] - a["1y"]
        else:
            ranks = []
            for field in fields:
                x = a[field].where(a[field] > 0)
                rank = x.rolling(1260, min_periods=252).rank(pct=True)
                out[f"{key}_{field}_rank"] = rank
                ranks.append(rank)
            # PE and PB each have to be present; no forward or cross-market imputation.
            out[f"{key}_value"] = 1 - 2 * ((ranks[0] + ranks[1]) / 2)
    return out.replace([np.inf, -np.inf], np.nan)
