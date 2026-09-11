"""Read-only API capability probe; never logs remote exceptions or credentials."""

from steadyquant.config import ROOT, write_json
from steadyquant.data import DataError, TushareProvider

p = TushareProvider()
out = ROOT / "data/stocks/probes"
out.mkdir(parents=True, exist_ok=True)
specs = [
    ("index_weight", dict(index_code="000300.SH", start_date="20131201", end_date="20131231")),
    ("daily_basic", dict(ts_code="600000.SH", start_date="20130101", end_date="20131231")),
    ("stk_limit", dict(ts_code="600000.SH", start_date="20130101", end_date="20131231")),
    (
        "fina_indicator",
        dict(
            ts_code="600000.SH",
            start_date="20130101",
            end_date="20141231",
            fields="ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or,netprofit_yoy,update_flag",
        ),
    ),
    (
        "fina_indicator_vip",
        dict(
            period="20131231",
            fields="ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or,netprofit_yoy,update_flag",
            limit=10,
        ),
    ),
    ("namechange", dict(ts_code="600000.SH")),
    ("stock_basic", dict(list_status="D", fields="ts_code,name,list_date,delist_date,market")),
    ("daily", dict(ts_code="600000.SH,600036.SH", start_date="20260901", end_date="20260909")),
]
audit = []
for i, (api, params) in enumerate(specs):
    try:
        d = p.query(api, **params)
        d.to_parquet(out / f"{i}-{api}.parquet", index=False)
        row = dict(api=api, params=params, rows=len(d), columns=list(d))
        for col in ("trade_date", "ann_date", "ts_code", "con_code"):
            if col in d and not d.empty:
                row[col + "_unique"] = int(d[col].nunique())
        print(row, flush=True)
        audit.append(row)
    except DataError as e:
        print(str(e), flush=True)
        audit.append(dict(api=api, error=str(e)))
write_json(out / "summary.json", audit)
