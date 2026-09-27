"""Refresh a bounded, research-only cross-asset ETF panel."""

import pandas as pd

from steadyquant.config import ROOT
from steadyquant.data import TushareProvider, validate_bars

END = "20260924"
EXTRA = {
    "159905.SZ": "cn_equity",
    "159928.SZ": "cn_equity",
    "512010.SH": "cn_equity",
    "512000.SH": "cn_equity",
    "512800.SH": "cn_equity",
    "512400.SH": "cn_equity",
    "512690.SH": "cn_equity",
    "512760.SH": "cn_equity",
    "513050.SH": "hk_equity",
    "513060.SH": "hk_equity",
    "511260.SH": "bond",
    "513080.SH": "us_equity",
}
SOURCE = ROOT / "data/sector_rotation/20260911/bars"
DEST = ROOT / "data/etf_broad_research/bars"


def run() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    provider = TushareProvider()
    for symbol in EXTRA:
        source = SOURCE / f"{symbol}.parquet"
        old = pd.read_parquet(source) if source.exists() else pd.DataFrame()
        start = (
            (pd.Timestamp(old.date.max()) + pd.Timedelta(days=1)).strftime("%Y%m%d")
            if not old.empty else "20130101"
        )
        new = provider.bars(symbol, "fund", start, END) if start <= END else pd.DataFrame()
        if new.empty and old.empty:
            raise ValueError(f"{symbol}: no history")
        frame = pd.concat([old, new], ignore_index=True).sort_values("date")
        if frame.date.duplicated().any() or frame.date.max() != pd.Timestamp(END):
            raise ValueError(f"{symbol}: duplicate or incomplete end date")
        validate_bars(frame)
        frame.to_parquet(DEST / f"{symbol}.parquet", index=False)
        print(symbol, len(frame), str(frame.date.min().date()), str(frame.date.max().date()), flush=True)


if __name__ == "__main__":
    run()
