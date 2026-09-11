"""Supplemental reproducible data and causal-signal audit; never changes model selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import write_json
from .data import Cache
from .factors import evaluate
from .rotation import build_features, ridge_predictions, rotation_weights


def audit_run(output: Path) -> dict:
    protocol = json.loads((output / "protocol.json").read_text())
    cfg = protocol["config"]
    cache = Cache()
    data = cache.load(cfg)
    if cache.snapshot_digest != protocol["data_sha256"]:
        raise ValueError("Data changed since frozen optimization protocol; audit the original snapshot")
    metadata = pd.read_json(cache.root / "rotation_universe.json", dtype={"list_date": str})
    calendar = pd.read_parquet(cache.root / "calendar.parquet")
    sessions = pd.to_datetime(calendar.loc[calendar.is_open.astype(int) == 1, "cal_date"])
    rows = []
    for symbol, df in data.items():
        listing = pd.Timestamp(metadata.set_index("ts_code").loc[symbol, "list_date"])
        if df.date.min() < listing:
            raise ValueError(f"{symbol} has prelisting data")
        expected = set(sessions[(sessions >= df.date.min()) & (sessions <= df.date.max())])
        rows.append(
            dict(
                symbol=symbol,
                rows=len(df),
                first=str(df.date.min().date()),
                last=str(df.date.max().date()),
                list_date=str(listing.date()),
                gaps=[str(d.date()) for d in sorted(expected - set(df.date))],
            )
        )
    factors = pd.read_parquet(output / "factors.parquet")
    factors = factors.loc[factors.eligible].drop(columns="eligible")
    summaries = []
    for label, lo, hi in [("selection", "2015-01-01", "2022-12-31"), ("test", "2023-01-01", "2100-01-01")]:
        # Evaluate only labels ending inside this period; never let 2023 returns enter selection diagnostics.
        historical = {s: df[df.date <= pd.Timestamp(hi)] for s, df in data.items()}
        f = factors[factors.date.between(lo, hi)]
        summary, detail = evaluate(historical, f)
        summary["period"] = label
        summaries.append(summary)
        detail.to_parquet(output / f"factor_ic_{label}.parquet", index=False)
    pd.concat(summaries).to_csv(output / "factor_summary.csv", index=False)
    # A real-data truncation audit, independent of synthetic unit tests.
    cutoff = pd.Timestamp("2022-12-30")
    prefix = {s: df[df.date <= cutoff] for s, df in data.items()}
    selected = json.loads((output / "summary.json").read_text())["selected"]
    selected_cfg = protocol["candidates"][selected]
    f = build_features(prefix, cfg)
    predictions, audit = ridge_predictions(f)
    saved_predictions = pd.read_parquet(output / "ridge_predictions.parquet").loc[:cutoff]
    pd.testing.assert_frame_equal(predictions, saved_predictions)
    target, _ = rotation_weights(prefix, selected_cfg, f, predictions)
    saved_targets = pd.read_parquet(output / selected / "targets.parquet").loc[:cutoff]
    pd.testing.assert_frame_equal(target, saved_targets)
    result = dict(
        data_fingerprint_verified=True,
        real_prefix_cutoff=str(cutoff.date()),
        real_prefix_targets_identical=True,
        real_prefix_ridge_predictions_identical=True,
        ridge_fits=len(audit),
        last_label_never_after_fit=bool((audit.last_label_date <= audit.fit_date).all()),
        total_bars=sum(r["rows"] for r in rows),
        data_coverage=rows,
        missing_bar_policy="no signal or execution on missing bar; prior price only used for valuation",
        selected_max_target=float(saved_targets.max().max()),
        selected_max_gross=float(saved_targets.sum(axis=1).max()),
    )
    if not np.isfinite(saved_targets.to_numpy()).all():
        raise ValueError("Invalid targets")
    write_json(output / "audit.json", result)
    return result
