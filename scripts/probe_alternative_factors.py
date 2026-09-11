"""Bounded read-only source audit; saves public market data, never credentials."""

import hashlib
from datetime import datetime

from steadyquant.config import ROOT, write_json
from steadyquant.data import DataError, TushareProvider

OUT = ROOT / "data/research/alternative_factors/20260910"
OUT.mkdir(parents=True, exist_ok=True)
report_fields = "ts_code,report_date,org_name,quarter,np,eps,op_rt,roe,create_time"
specs = [
    ("rc_2016", "report_rc", dict(report_date="20160831", fields=report_fields, limit=1000)),
    ("rc_2020", "report_rc", dict(report_date="20200831", fields=report_fields, limit=1000)),
    ("rc_recent", "report_rc", dict(report_date="20260909", fields=report_fields, limit=1000)),
    (
        "rc_single",
        "report_rc",
        dict(
            ts_code="600887.SH", start_date="20250101", end_date="20260909", fields=report_fields, limit=1000
        ),
    ),
    ("forecast", "forecast_vip", dict(period="20260630", limit=1000)),
    ("express", "express_vip", dict(period="20260630", limit=1000)),
    (
        "income",
        "income_vip",
        dict(
            period="20260630",
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,revenue,n_income,n_income_attr_p,update_flag",
            limit=1000,
        ),
    ),
    (
        "cashflow",
        "cashflow_vip",
        dict(
            period="20260630",
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,net_profit,n_cashflow_act,c_pay_acq_const_fiolta,update_flag",
            limit=1000,
        ),
    ),
    (
        "balancesheet",
        "balancesheet_vip",
        dict(
            period="20260630",
            fields="ts_code,ann_date,f_ann_date,end_date,report_type,comp_type,total_assets,accounts_receiv,inventories,goodwill,update_flag",
            limit=1000,
        ),
    ),
    ("repurchase", "repurchase", dict(start_date="20260901", end_date="20260909", limit=1000)),
    (
        "holdertrade",
        "stk_holdertrade",
        dict(
            start_date="20260901",
            end_date="20260909",
            fields="ts_code,ann_date,holder_type,in_de,change_vol,change_ratio,avg_price,begin_date,close_date",
            limit=1000,
        ),
    ),
    ("unlock", "share_float", dict(start_date="20260901", end_date="20260930", limit=1000)),
    (
        "survey",
        "stk_surv",
        dict(
            ts_code="002223.SZ",
            start_date="20260101",
            end_date="20260909",
            fields="ts_code,surv_date,rece_org,rece_mode,org_type",
            limit=400,
        ),
    ),
    ("industry_current", "index_member_all", dict(ts_code="600887.SH", is_new="Y", limit=1000)),
    ("industry_past", "index_member_all", dict(ts_code="600887.SH", is_new="N", limit=1000)),
]
p = TushareProvider()
audit = []
for label, api, params in specs:
    path = OUT / f"{label}.parquet"
    try:
        if path.exists():
            import pandas as pd

            frame = pd.read_parquet(path)
        else:
            frame = p.query(api, **params)
            frame.to_parquet(path, index=False)
        row = dict(
            label=label,
            api=api,
            params=params,
            rows=len(frame),
            columns=list(frame),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            at_requested_cap=len(frame) >= params["limit"],
        )
        row["missing"] = {c: int(frame[c].isna().sum()) for c in frame}
        for col in (
            "ts_code",
            "ann_date",
            "f_ann_date",
            "report_date",
            "create_time",
            "surv_date",
            "end_date",
            "quarter",
            "update_flag",
        ):
            if col in frame and frame[col].notna().any():
                series = frame[col].dropna().astype(str)
                row[col] = dict(unique=int(series.nunique()), min=series.min(), max=series.max())
        audit.append(row)
        print(f"{label}: {len(frame)} rows, cap={row['at_requested_cap']}", flush=True)
    except DataError as exc:
        audit.append(dict(label=label, api=api, params=params, error=str(exc)))
        print(f"{label}: unavailable (sanitized)", flush=True)
    write_json(
        OUT / "audit.json",
        dict(
            created_at=datetime.now().astimezone().isoformat(),
            scope="bounded capability samples; not complete history or alpha evidence",
            samples=audit,
        ),
    )
