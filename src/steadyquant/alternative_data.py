"""Versioned, resumable alternative-data snapshots; never updates the active account."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import local

import pandas as pd

from .config import ROOT, fingerprint, write_json
from .data import DataError, TushareProvider

ALT_ROOT = ROOT / "data/alternative/20260910"
END = "20260909"
THREAD = local()
FIELDS = {
    "report_rc": "ts_code,report_date,org_name,quarter,np,eps,op_rt,roe,create_time",
    "income_vip": "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,revenue,oper_cost,operate_profit,n_income,n_income_attr_p,basic_eps,update_flag",
    "cashflow_vip": "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,net_profit,n_cashflow_act,c_pay_acq_const_fiolta,update_flag",
    "balancesheet_vip": "ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_assets,accounts_receiv,inventories,goodwill,update_flag",
    "forecast_vip": "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,last_parent_net,first_ann_date,update_flag",
    "express_vip": "ts_code,ann_date,end_date,revenue,operate_profit,total_profit,n_income,total_assets,diluted_eps,yoy_net_profit,is_audit,remark",
    "index_member_all": "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,in_date,out_date,is_new",
}


def request(api, params, page_size=2500):
    params = {**params, "fields": FIELDS[api]}
    key = fingerprint(dict(api=api, params=params, page_size=page_size, version=1))
    path = ALT_ROOT / "requests" / api / f"{key}.parquet"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        payload = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if hashlib.sha256(payload).hexdigest() != meta["sha256"]:
            raise DataError("Alternative source cache checksum mismatch")
        return pd.read_parquet(io.BytesIO(payload))
    if not hasattr(THREAD, "provider"):
        THREAD.provider = TushareProvider()
    pages, signatures, page_counts = [], set(), []
    first_seen = datetime.now().astimezone().isoformat()
    for offset in range(0, 150000, page_size):
        d = THREAD.provider.query(api, **params, limit=page_size, offset=offset)
        signature = fingerprint(d.to_dict("list"))
        if not d.empty and signature in signatures:
            raise DataError(f"{api}: repeated pagination page")
        signatures.add(signature)
        page_counts.append(len(d))
        pages.append(d)
        if len(d) < page_size:
            break
    else:
        raise DataError(f"{api}: pagination exceeded safety cap")
    d = pd.concat(pages, ignore_index=True)
    missing_optional = []
    if not d.empty:
        required = set(FIELDS[api].split(","))
        if api == "express_vip":
            optional = {"is_audit", "remark"}
            missing_optional = sorted(optional - set(d))
            required -= optional
            for col in missing_optional:
                d[col] = None
        if not required.issubset(d):
            raise DataError(f"{api}: fields missing: {sorted(required - set(d))}")
        if "ts_code" in params and not d.ts_code.eq(params["ts_code"]).all():
            raise DataError(f"{api}: symbol filter ignored")
        if "period" in params and not d.end_date.eq(params["period"]).all():
            raise DataError(f"{api}: fiscal period filter ignored")
        if api == "report_rc" and not d.report_date.between(params["start_date"], params["end_date"]).all():
            raise DataError("Analyst forecast date filter ignored")
        if "report_type" in params and not d.report_type.astype(str).eq(params["report_type"]).all():
            raise DataError("Financial report-type filter ignored")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    d.to_parquet(tmp, index=False)
    tmp.replace(path)
    write_json(
        meta_path,
        dict(
            api=api,
            params=params,
            page_counts=page_counts,
            missing_optional_fields=missing_optional,
            first_seen_at=first_seen,
            completed_at=datetime.now().astimezone().isoformat(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=len(d),
        ),
    )
    return d


def batch(specs, label):
    frames = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(request, api, params): (api, params) for api, params in specs}
        for i, job in enumerate(as_completed(jobs), 1):
            frames.append(job.result())
            if i % 10 == 0 or i == len(jobs):
                print(f"{label}: {i}/{len(jobs)} partitions", flush=True)
    return (
        pd.concat([f for f in frames if not f.empty], ignore_index=True)
        if any(not f.empty for f in frames)
        else pd.DataFrame()
    )


def pilot():
    for api in ("income_vip", "cashflow_vip", "balancesheet_vip"):
        for report_type in ("1", "5"):
            d = request(api, dict(period="20231231", report_type=report_type))
            print(
                json.dumps(
                    dict(
                        api=api,
                        report_type=report_type,
                        rows=len(d),
                        flags=d.update_flag.value_counts().to_dict() if not d.empty else {},
                    )
                ),
                flush=True,
            )


def sync():
    if (ALT_ROOT / "manifest.json").exists():
        load_alternative()
        print("Existing complete snapshot verified; no refresh or overwrite", flush=True)
        return
    symbols = set(json.loads((ROOT / "data/stocks/universe.json").read_text())["symbols"])
    # Save all queried market rows in immutable request files, filter only the
    # assembled research tables to the historical union (not today's members).
    months = pd.period_range("2022-01", "2026-09", freq="M")
    specs = [
        (
            "report_rc",
            dict(
                start_date=m.start_time.strftime("%Y%m%d"), end_date=min(END, m.end_time.strftime("%Y%m%d"))
            ),
        )
        for m in months
    ]
    tables = {"analyst": batch(specs, "analyst")}
    for name, api in [
        ("income", "income_vip"),
        ("cashflow", "cashflow_vip"),
        ("balance", "balancesheet_vip"),
    ]:
        specs = [
            (api, dict(period=p.end_time.strftime("%Y%m%d"), report_type=rt))
            for p in pd.period_range("2018Q1", "2026Q2", freq="Q")
            for rt in ("1", "4", "5")
        ]
        tables[name] = batch(specs, name)
    for name, api in [("forecast", "forecast_vip"), ("express", "express_vip")]:
        tables[name] = batch(
            [
                (api, dict(period=p.end_time.strftime("%Y%m%d")))
                for p in pd.period_range("2021Q1", "2026Q2", freq="Q")
            ],
            name,
        )
    tables["industry"] = batch(
        [("index_member_all", dict(ts_code=s, is_new=v)) for s in sorted(symbols) for v in ("Y", "N")],
        "industry",
    )
    info = {}
    for name, d in tables.items():
        d = d[d.ts_code.isin(symbols)].drop_duplicates()
        path = ALT_ROOT / f"{name}.parquet"
        d.to_parquet(path, index=False)
        info[name] = dict(
            rows=len(d), stocks=int(d.ts_code.nunique()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
    write_json(
        ALT_ROOT / "manifest.json",
        dict(
            completed_at=datetime.now().astimezone().isoformat(),
            end=END,
            historical_union=len(symbols),
            tables=info,
            request_count=len(list((ALT_ROOT / "requests").glob("*/*.json"))),
        ),
    )
    print("Alternative snapshot complete", flush=True)


def load_alternative():
    manifest_text = (ALT_ROOT / "manifest.json").read_text()
    meta = json.loads(manifest_text)
    tables = {}
    for name, info in meta["tables"].items():
        payload = (ALT_ROOT / f"{name}.parquet").read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise DataError("Alternative snapshot checksum mismatch")
        tables[name] = pd.read_parquet(io.BytesIO(payload))
    if (ALT_ROOT / "manifest.json").read_text() != manifest_text:
        raise DataError("Alternative manifest changed during read")
    return tables, meta
