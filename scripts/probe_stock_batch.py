from steadyquant.config import ROOT, write_json
from steadyquant.data import TushareProvider

p = TushareProvider()
out = ROOT / "data/stocks/probes2"
out.mkdir(parents=True, exist_ok=True)
specs = [
    ("index_weight", dict(index_code="000300.SH", start_date="20150101", end_date="20150131")),
    ("index_weight", dict(index_code="399300.SZ", start_date="20150101", end_date="20150131")),
    ("index_weight", dict(index_code="000300.SH", start_date="20200101", end_date="20200131")),
    ("adj_factor", dict(ts_code="600000.SH,600036.SH", start_date="20260901", end_date="20260909")),
    ("stk_limit", dict(ts_code="600000.SH,600036.SH", start_date="20260901", end_date="20260909")),
    ("daily_basic", dict(trade_date="20150105", fields="ts_code,trade_date,pe_ttm,pb,dv_ttm,total_mv")),
    (
        "dividend",
        dict(
            ts_code="600000.SH",
            fields="ts_code,end_date,ann_date,div_proc,stk_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate",
        ),
    ),
]
rows = []
for i, (api, params) in enumerate(specs):
    d = p.query(api, **params)
    d.to_parquet(out / f"{i}-{api}.parquet", index=False)
    row = dict(api=api, params=params, rows=len(d), columns=list(d))
    for col in ("trade_date", "ts_code", "con_code"):
        if col in d:
            row[col + "_unique"] = int(d[col].nunique())
    rows.append(row)
    print(row, flush=True)
write_json(out / "summary.json", rows)
