"""Minimal authenticated read-only probe; credentials never enter output."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steadyquant.data import DataError, TushareProvider

p = TushareProvider()
queries = [
    ("index_basic", {"limit": 5}),
    ("fund_daily", {"ts_code": "510300.SH", "start_date": "20260901", "end_date": "20260909"}),
    ("fund_adj", {"ts_code": "510300.SH", "start_date": "20260901", "end_date": "20260909"}),
]
if "--gap" in sys.argv:
    queries = [
        ("fund_adj", {"ts_code": s, "trade_date": "20200918"})
        for s in ["513500.SH", "518880.SH", "511010.SH"]
    ]
for api, args in queries:
    try:
        df = p.query(api, **args)
        print(api, "rows=", len(df), "columns=", list(df), flush=True)
        if "trade_date" in df:
            print("date range:", df.trade_date.min(), df.trade_date.max(), flush=True)
        if "--gap" in sys.argv:
            print(df.to_string(index=False), flush=True)
    except DataError as exc:
        print(str(exc), flush=True)
