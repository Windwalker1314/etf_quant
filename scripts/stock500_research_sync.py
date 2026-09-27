"""Build an isolated, resumable historical CSI500 research snapshot via Tushare."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from steadyquant.config import ROOT, write_json
from steadyquant.data import DataError
from steadyquant.stock_data import batch, load_stock_snapshot, request

END = "20260909"
DEST = ROOT / "data/stocks500"
FIELDS = "ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate"


def batch_fast(specs, label):
    def retry(api, params):
        for attempt in range(3):
            try:
                return request(api, params)
            except DataError:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))

    frames = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(retry, api, params): (api, params) for api, params in specs}
        for i, job in enumerate(as_completed(pending), 1):
            try:
                frame = job.result()
            except Exception:
                api, params = pending[job]
                print("failed", api, params, flush=True)
                raise
            if not frame.empty:
                frames.append(frame)
            if i % 100 == 0 or i == len(specs):
                print(f"{label}: {i}/{len(specs)}", flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save(name: str, frame: pd.DataFrame):
    path = DEST / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    print(name, len(frame), flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    specs = [
        ("index_weight", dict(index_code="000905.SH",
                              start_date=month.start_time.strftime("%Y%m%d"),
                              end_date=min(END, month.end_time.strftime("%Y%m%d"))))
        for month in pd.period_range("2014-01", "2026-09", freq="M")
    ]
    members = batch(specs, "membership").drop_duplicates(["trade_date", "con_code"])
    save("members_source", members)
    # The provider backfills both the old and new ticker for 14 pre-change
    # snapshots. Neither ticker can be traded safely across the conversion
    # without a dated security-ID ledger, so omit this one name until the new
    # code actually becomes valid (2025-02-07 exchange announcement).
    ambiguous = members.con_code.isin(("300114.SZ", "302132.SZ")) & members.trade_date.lt("20250207")
    excluded_rows = int(ambiguous.sum())
    members = members.loc[~ambiguous].copy()
    members.to_parquet(DEST / "members.parquet", index=False)
    counts = members.groupby("trade_date").con_code.nunique()
    weight_sums = members.groupby("trade_date").weight.sum()
    expected = pd.Series(500, index=counts.index)
    expected.loc[expected.index.to_series().between("20231229", "20250127")] = 499
    if counts.empty or not counts.eq(expected).all() or not weight_sums.between(99.5, 100.4).all():
        raise DataError(f"Incomplete CSI500 monthly constituents: {counts.value_counts().to_dict()}")
    print("quarantined ambiguous code-change membership rows", excluded_rows, flush=True)
    symbols = sorted(members.con_code.unique())
    print("CSI500 historical union", len(symbols), flush=True)
    old, old_manifest = load_stock_snapshot()
    if old_manifest["end"] != END:
        raise DataError("CSI300 reference snapshot has a different cutoff")
    old_symbols = set(old["metadata"].ts_code)
    new_symbols = sorted(set(symbols) - old_symbols)
    print("additional symbols", len(new_symbols), flush=True)
    metadata = batch(
        [("stock_basic", dict(list_status=status,
                              fields="ts_code,name,list_date,delist_date,market"))
         for status in ("L", "D", "P")], "metadata")
    metadata = metadata[metadata.ts_code.isin(symbols)].drop_duplicates("ts_code")
    if set(metadata.ts_code) != set(symbols):
        raise DataError("Missing CSI500 historical listing metadata")
    save("metadata", metadata)
    for api in ("daily", "adj_factor", "stk_limit"):
        specs = []
        if api == "stk_limit":
            # Tushare stk_limit silently returns zero rows for comma-separated codes.
            # A single-symbol 2014-2026 request stays below the response row cap.
            specs = [(api, dict(ts_code=s, start_date="20140101", end_date=END))
                     for s in new_symbols]
        else:
            for year in range(2014, 2027):
                for i in range(0, len(new_symbols), 20):
                    specs.append((api, dict(ts_code=",".join(new_symbols[i:i + 20]),
                                            start_date=f"{year}0101", end_date=min(END, f"{year}1231"))))
        new = batch_fast(specs, api)
        combined = pd.concat([old[api][old[api].ts_code.isin(symbols)], new], ignore_index=True)
        if combined.duplicated(["ts_code", "trade_date"]).any() or set(combined.ts_code) != set(symbols):
            raise DataError(f"Incomplete or duplicate {api} CSI500 bars")
        save(api, combined)
    for api, params in [("namechange", {}), ("dividend", dict(fields=FIELDS))]:
        new = batch_fast([(api, dict(ts_code=s, **params)) for s in new_symbols], api)
        combined = pd.concat([old[api][old[api].ts_code.isin(symbols)], new], ignore_index=True)
        save(api, combined)
    # Monthly valuations and quarterly financial queries were cached as all-market
    # responses by the CSI300 snapshot; their existing query keys are reused here.
    calendar = pd.read_parquet(ROOT / "data/calendar.parquet")
    dates = pd.to_datetime(calendar.loc[calendar.is_open.astype(int).eq(1)
                            & calendar.cal_date.between("20150101", END), "cal_date"])
    signal_dates = dates.groupby(dates.dt.to_period("M")).min().tolist()
    valuation = batch([
        ("daily_basic", dict(trade_date=date.strftime("%Y%m%d"),
                             fields="ts_code,trade_date,pe_ttm,pb,dv_ttm,total_mv"))
        for date in signal_dates], "valuation")
    save("valuations", valuation[valuation.ts_code.isin(symbols)])
    reports = batch([
        ("fina_indicator_vip", dict(period=period.end_time.strftime("%Y%m%d"),
                                    fields="ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or,netprofit_yoy,update_flag"))
        for period in pd.period_range("2013Q1", "2026Q2", freq="Q")], "financials")
    save("financials", reports[reports.ts_code.isin(symbols)])
    paths = sorted(DEST.glob("*.parquet"))
    write_json(DEST / "manifest.json", dict(end=END, universe=len(symbols),
               excluded_ambiguous_codes=["300114.SZ", "302132.SZ"],
               excluded_before="20250207", excluded_membership_rows=excluded_rows,
               completed_at=datetime.now().isoformat(),
               files={path.name: dict(rows=len(pd.read_parquet(path)),
                                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                      for path in paths}))
    print("snapshot ready", DEST, flush=True)


if __name__ == "__main__":
    main()
