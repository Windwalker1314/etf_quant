"""Resumable, bounded Tushare research snapshot for historical CSI300 main-board stocks."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import local

import pandas as pd

from .config import ROOT, fingerprint, write_json
from .data import DataError, TushareProvider

STOCK_ROOT = ROOT / "data/stocks"
THREAD = local()
END = "20260909"  # Frozen cross-asset comparison, never use a partial current session.


def request(api, params, cap=6000):
    path = STOCK_ROOT / "requests" / api / f"{fingerprint(params)}.parquet"
    meta = path.with_suffix(".json")
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise DataError("Stock request cache digest mismatch")
        return pd.read_parquet(path)
    if not hasattr(THREAD, "provider"):
        THREAD.provider = TushareProvider()
    if api == "fina_indicator_vip" and "limit" not in params:
        pages, signatures = [], set()
        for offset in range(0, 100000, 5000):
            page = THREAD.provider.query(api, **params, limit=5000, offset=offset)
            signature = fingerprint(page.to_dict("list"))
            if signature in signatures and not page.empty:
                raise DataError("Financial pagination did not advance")
            signatures.add(signature)
            pages.append(page)
            if len(page) < 5000:
                break
        else:
            raise DataError("Financial pagination safety limit")
        d = pd.concat(pages, ignore_index=True)
    else:
        d = THREAD.provider.query(api, **params)
    if len(d) >= cap and not (api == "fina_indicator_vip" and "limit" not in params):
        raise DataError(f"{api}: possible truncation at {len(d)} rows")
    if "trade_date" in d:
        ds = d.trade_date.astype(str)
        lo = params.get("trade_date", params.get("start_date", "19000101"))
        hi = params.get("trade_date", params.get("end_date", END))
        if not ds.between(lo, hi).all():
            raise DataError(f"{api}: date bounds ignored")
    if "ts_code" in params and "ts_code" in d:
        if not set(d.ts_code).issubset(params["ts_code"].split(",")):
            raise DataError(f"{api}: symbol filter ignored")
    if "period" in params and "end_date" in d and not d.end_date.astype(str).eq(params["period"]).all():
        raise DataError(f"{api}: fiscal period filter ignored")
    if "index_code" in params and "index_code" in d and not d.index_code.eq(params["index_code"]).all():
        raise DataError(f"{api}: index filter ignored")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    d.to_parquet(tmp, index=False)
    tmp.replace(path)
    write_json(
        meta, dict(api=api, params=params, rows=len(d), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    )
    return d


def batch(specs, label):
    result = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(request, api, params): (api, params) for api, params in specs}
        for i, job in enumerate(as_completed(jobs), 1):
            result.append(job.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"{label}: {i}/{len(jobs)}", flush=True)
    nonempty = [x for x in result if not x.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def sync_stocks():
    STOCK_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        STOCK_ROOT / "protocol.json",
        dict(
            end=END,
            start="20140101",
            universe="Historical monthly CSI300 399300.SZ; main board prefixes 60/00; membership usable strictly after observation date; exclude snapshots older than 62 days",
            factors="Initial financial reports only (update_flag=0), ann_date strictly before signal; max report age 550 days; value, profitability, momentum, low volatility",
            design="3 fixed stock families x 25/40 percent maximum stock allocation; top 8; trend regime reduces stock risk; ETF remainder existing equal-risk model; no parameter search",
            evaluation="2016-2020 development, 2021-2022 validation, 2023+ retrospective comparison, frozen before outcomes; historical ETF model already selected on this history",
            gate="validation CAGR >= ETF, Sharpe >= ETF + 0.05, maxDD >= -15%, 2018/2022 losses >= -8%; otherwise retain active ETF; historical validation is not unseen forward evidence",
            capital=200000,
            leverage=False,
            activation="research only; active.yaml unchanged",
        ),
    )
    specs = []
    for month in pd.period_range("2014-01", "2026-09", freq="M"):
        specs.append(
            (
                "index_weight",
                dict(
                    index_code="399300.SZ",
                    start_date=month.start_time.strftime("%Y%m%d"),
                    end_date=min(END, month.end_time.strftime("%Y%m%d")),
                ),
            )
        )
    members = batch(specs, "membership").drop_duplicates(["trade_date", "con_code"])
    counts = members.groupby("trade_date").con_code.nunique()
    if (counts != 300).any():
        raise DataError("Historical index snapshot does not contain exactly 300 stocks")
    members.to_parquet(STOCK_ROOT / "members.parquet", index=False)
    symbols = sorted(s for s in members.con_code.unique() if s.startswith(("60", "00")))
    write_json(
        STOCK_ROOT / "universe.json",
        dict(
            symbols=symbols,
            count=len(symbols),
            snapshots=len(counts),
            first=str(counts.index.min()),
            last=str(counts.index.max()),
        ),
    )
    print(f"Historical main-board union: {len(symbols)} stocks, {len(counts)} snapshots", flush=True)
    basic = batch(
        [
            ("stock_basic", dict(list_status=status, fields="ts_code,name,list_date,delist_date,market"))
            for status in ("L", "D", "P")
        ],
        "metadata",
    )
    basic = basic[basic.ts_code.isin(symbols)].drop_duplicates("ts_code")
    if set(basic.ts_code) != set(symbols):
        raise DataError("Missing listing/delisting metadata for historical members")
    basic.to_parquet(STOCK_ROOT / "metadata.parquet", index=False)
    for api in ("daily", "adj_factor", "stk_limit"):
        specs = []
        for i in range(0, len(symbols), 15):
            codes = ",".join(symbols[i : i + 15])
            for year in range(2014, 2027):
                specs.append(
                    (api, dict(ts_code=codes, start_date=f"{year}0101", end_date=min(END, f"{year}1231")))
                )
        d = batch(specs, api)
        if d.duplicated(["ts_code", "trade_date"]).any():
            raise DataError(f"{api}: duplicate symbol dates")
        if set(d.ts_code) != set(symbols):
            raise DataError(f"{api}: incomplete symbol coverage")
        d.to_parquet(STOCK_ROOT / f"{api}.parquet", index=False)
    cal = pd.read_parquet(ROOT / "data/calendar.parquet")
    dates = pd.to_datetime(
        cal.loc[(cal.is_open.astype(int) == 1) & cal.cal_date.between("20150101", END), "cal_date"]
    )
    signal_dates = dates.groupby(dates.dt.to_period("M")).min().tolist()
    # All-market daily_basic avoids silently replacing missing/delisted symbols with today's universe.
    valuation = batch(
        [
            (
                "daily_basic",
                dict(trade_date=d.strftime("%Y%m%d"), fields="ts_code,trade_date,pe_ttm,pb,dv_ttm,total_mv"),
            )
            for d in signal_dates
        ],
        "valuations",
    )
    valuation[valuation.ts_code.isin(symbols)].to_parquet(STOCK_ROOT / "valuations.parquet", index=False)
    fields = "ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or,netprofit_yoy,update_flag"
    reports = batch(
        [
            ("fina_indicator_vip", dict(period=p.end_time.strftime("%Y%m%d"), fields=fields))
            for p in pd.period_range("2013Q1", "2026Q2", freq="Q")
        ],
        "financials",
    )
    reports[reports.ts_code.isin(symbols)].to_parquet(STOCK_ROOT / "financials.parquet", index=False)
    for api, params in [
        ("namechange", {}),
        (
            "dividend",
            dict(
                fields="ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate"
            ),
        ),
    ]:
        d = batch([(api, dict(ts_code=s, **params)) for s in symbols], api)
        d.to_parquet(STOCK_ROOT / f"{api}.parquet", index=False)
    paths = sorted(STOCK_ROOT.glob("*.parquet"))
    write_json(
        STOCK_ROOT / "manifest.json",
        dict(
            completed_at=datetime.now().isoformat(),
            end=END,
            universe=len(symbols),
            files={
                p.name: dict(rows=len(pd.read_parquet(p)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in paths
            },
        ),
    )
    return STOCK_ROOT


def load_stock_snapshot():
    manifest_text = (STOCK_ROOT / "manifest.json").read_text()
    meta = json.loads(manifest_text)
    frames = {}
    for name, info in meta["files"].items():
        path = STOCK_ROOT / name
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise DataError("Stock snapshot changed after manifest")
        frames[Path(name).stem] = pd.read_parquet(io.BytesIO(payload))
    if (STOCK_ROOT / "manifest.json").read_text() != manifest_text:
        raise DataError("Stock manifest changed during snapshot read")
    return frames, meta
