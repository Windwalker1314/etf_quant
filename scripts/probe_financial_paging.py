from steadyquant.stock_data import request

for period in ("20131231", "20251231"):
    d = request(
        "fina_indicator_vip",
        dict(
            period=period,
            fields="ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or,netprofit_yoy,update_flag",
            limit=5000,
            offset=0,
        ),
    )
    print(period, len(d), d.ts_code.nunique(), d.update_flag.value_counts(dropna=False).to_dict(), flush=True)
