"""Revalidate downloaded source frames without repeating authenticated downloads."""

import pandas as pd

from steadyquant.config import ROOT
from steadyquant.data import Cache, normalize, quarantine_missing_adjustments

if __name__ == "__main__":
    for s in ("513500.SH", "518880.SH", "511010.SH"):
        raw = pd.read_parquet(ROOT / "data/raw" / f"{s}.parquet")
        clean = normalize(quarantine_missing_adjustments(raw, s), s)
        Cache().save(s, clean)
        print(s, len(clean), clean.date.max())
