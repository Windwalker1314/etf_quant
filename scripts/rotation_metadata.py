import pandas as pd

from steadyquant.config import ROOT, load_config
from steadyquant.data import TushareProvider

cfg = load_config(ROOT / "configs/rotation.yaml")
provider = TushareProvider()
rows = []
for a in cfg["assets"]:
    df = provider.query("fund_basic", ts_code=a["symbol"], market="E")
    fields = [c for c in ["ts_code", "name", "list_date", "fund_type", "status"] if c in df]
    df = df.loc[df.ts_code == a["symbol"], fields]
    if len(df) != 1:
        raise RuntimeError(f"Metadata lookup not unique: {a['symbol']}")
    print(df.to_dict("records"), flush=True)
    rows.append(df)
pd.concat(rows).to_json(ROOT / "data/rotation_universe.json", orient="records", force_ascii=False, indent=2)
