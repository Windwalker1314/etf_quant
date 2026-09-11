import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from steadyquant.data import DataError, TushareProvider

if __name__ == "__main__":
    p = TushareProvider()
    for symbol in sys.argv[1:]:
        try:
            p.bars(symbol, "fund", "20130101", "20260909")
            print(symbol, "clean", flush=True)
        except DataError:
            raw = pd.read_parquet(Path("data/raw") / f"{symbol}.parquet")
            columns = ["trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]
            bad = raw[columns].isna().any(axis=1) | (raw[["open", "close"]] <= 0).any(axis=1)
            print(symbol, "rows", len(raw), "invalid", int(bad.sum()), flush=True)
            print(raw.loc[bad, columns].head(30).to_string(index=False), flush=True)
