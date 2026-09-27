"""Cache free ChinaBond historical yield curves through AKShare for research."""

from __future__ import annotations

import hashlib
from datetime import datetime

import akshare as ak
import pandas as pd

from steadyquant.config import ROOT, write_json
from steadyquant.data import TZ

START_YEAR = 2013
END_YEAR = 2026
END_DATE = "2026-09-24"
ROOT_DIR = ROOT / "data/chinabond"


def run():
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in range(START_YEAR, END_YEAR + 1):
        for start, end in ((f"{year}0101", f"{year}0630"),
                           (f"{year}0701", f"{year}1231")):
            if start > END_DATE.replace("-", ""):
                continue
            end = min(end, END_DATE.replace("-", ""))
            print(f"ChinaBond {start}-{end}", flush=True)
            frame = ak.bond_china_yield(start_date=start, end_date=end)
            if frame.empty or not {"曲线名称", "日期", "1年", "10年"}.issubset(frame):
                raise RuntimeError(f"Incomplete yield response: {start}-{end}")
            frame["日期"] = pd.to_datetime(frame["日期"])
            if not frame["日期"].between(pd.Timestamp(start), pd.Timestamp(end)).all():
                raise RuntimeError(f"Yield response ignored dates: {start}-{end}")
            gov = frame[frame["曲线名称"] == "中债国债收益率曲线"]
            if len(gov) < 20:
                raise RuntimeError(f"Government curve missing: {start}-{end}")
            parts.append(frame)
    all_data = pd.concat(parts, ignore_index=True)
    if all_data.duplicated(["曲线名称", "日期"]).any():
        raise RuntimeError("Duplicate curve observations")
    government = all_data[all_data["曲线名称"] == "中债国债收益率曲线"]
    if government[["1年", "10年"]].isna().any().any():
        raise RuntimeError("Government curve has missing required maturity")
    path = ROOT_DIR / "yield_curves.parquet"
    all_data.to_parquet(path, index=False)
    write_json(ROOT_DIR / "manifest.json", dict(
        source="AKShare bond_china_yield / ChinaBond historical curve",
        source_url="https://github.com/akfamily/akshare/blob/main/docs/data/bond/bond.md",
        fetched_at=datetime.now(TZ).isoformat(),
        dates=[str(government["日期"].min().date()), str(government["日期"].max().date())],
        government_rows=len(government), total_rows=len(all_data),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        caveat="Vendor historical history, not original publication vintage. Signals must lag at least one market session.",
    ))
    print(f"Saved {len(all_data)} yield rows, {len(government)} government", flush=True)
    return path


if __name__ == "__main__":
    run()
