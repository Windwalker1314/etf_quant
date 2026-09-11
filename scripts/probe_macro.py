from steadyquant.data import DataError, TushareProvider

p = TushareProvider()
for api, params in [
    ("shibor", {}),
    ("index_dailybasic", {"ts_code": "000300.SH"}),
    ("index_dailybasic", {"ts_code": "000905.SH"}),
    ("index_dailybasic", {"ts_code": "399006.SZ"}),
]:
    try:
        d = p.query(api, start_date="20260901", end_date="20260909", **params)
        print(api, params, len(d), list(d), flush=True)
    except DataError as e:
        print(str(e), flush=True)
