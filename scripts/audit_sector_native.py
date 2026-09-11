"""Diagnose native fill-policy differences without changing the strategy."""

import json
from pathlib import Path

import pandas as pd

from steadyquant.adaptive import adaptive_weights
from steadyquant.backtest import simulate
from steadyquant.config import ROOT, load_active_config, write_json
from steadyquant.data import Cache
from steadyquant.framework import engine_comparison, native_backtest
from steadyquant.sector_ledger import MarketArrays, simulate_sectors
from steadyquant.sector_rotation import joint_targets

out = Path(json.loads((ROOT / "outputs/latest_sector.json").read_text())["path"])
cfg = {**load_active_config(), "initial_cash": 200000}
data = Cache().load(cfg)
core, _ = adaptive_weights(data, cfg)
meta = pd.read_parquet(out / "metadata.parquet")
p = pd.read_csv(out / "picks_risk_adjusted.csv", index_col=0, parse_dates=True).iloc[:, 0]
p = p.fillna("").map(lambda x: tuple(x.split(",")) if x else ())
symbols = list(data) + sorted({s for v in p for s in v} - set(data))
w, mask, forced, _ = joint_targets(core, p, 0.1, symbols)
last = w.index[-40:]
used = [s for s in symbols if w.loc[last, s].max() > 0]
for s in used:
    if s not in data:
        data[s] = pd.read_parquet(ROOT / "data/sector_rotation/20260911/bars" / f"{s}.parquet")
data = {s: data[s][data[s].date.isin(w.index[-41:])] for s in used}
m = mask[used].copy()
m.loc[last[0]] = True
bands = {s: 0.02 if s in meta.index else cfg["rebalance_band"] for s in used}
cfg = {**cfg, "backtest_start": str(last[0].date())}
slow = simulate(data, w.loc[last, used], cfg, rebalance_mask=m, forced_mask=forced[used], symbol_bands=bands)
fast = simulate_sectors(
    MarketArrays(data, last, used), w[used], m, forced[used], cfg, meta, cfg["backtest_start"]
)
difference = float((slow.equity.equity - fast.equity.equity).abs().max())
assert difference < 0.01
pd.testing.assert_frame_equal(slow.trades, fast.trades, check_exact=False, atol=1e-7)
native = native_backtest(data, w.loc[last, used], cfg, m, forced[used], bands)
executions = native.executions_df.copy()
executions["date"] = (
    pd.to_datetime(executions.timestamp).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
)
executions["side"] = executions.side.str.upper()
executions.to_csv(out / "native_risk10_executions.csv", index=False)
slow.trades.to_csv(out / "reference_risk10_trades.csv", index=False)
merged = slow.trades.merge(
    executions[["date", "symbol", "side", "quantity", "price"]],
    on=["date", "symbol", "side"],
    how="outer",
    suffixes=("_reference", "_native"),
)
bad = merged[
    (merged.quantity_reference - merged.quantity_native).abs().gt(1e-7)
    | merged.quantity_reference.isna()
    | merged.quantity_native.isna()
]
bad.to_csv(out / "native_fill_differences.csv", index=False)
checks = {}
for name in ("core", "risk10", "risk10_broad"):
    ww, mm, ff, _ = joint_targets(core, p, 0 if name == "core" else 0.1, symbols, name.endswith("broad"))
    needed = [s for s in used if ww.loc[last, s].max() > 0]
    mm = mm[needed].copy()
    mm.loc[last[0]] = True
    check, compare = engine_comparison(
        {s: data[s] for s in needed}, ww[needed], cfg, mm, ff[needed], {s: bands[s] for s in needed}
    )
    compare.to_csv(out / f"native_{name}.csv")
    check["capital"] = cfg["initial_cash"]
    ar = simulate_sectors(
        MarketArrays({s: data[s] for s in needed}, last, needed),
        ww[needed],
        mm,
        ff[needed],
        cfg,
        meta,
        cfg["backtest_start"],
    )
    delta = float((ar.equity.equity - compare.ledger).abs().max())
    check.update(array_vs_reference_verified=delta < 0.01, array_vs_reference_max_diff_cny=delta)
    checks[name] = check
if not checks["risk10"]["verified"]:
    checks["risk10"]["diagnostic"] = (
        "Native execution gap / cash handling differs from reference partial-affordable round-lot fills; inspect native_fill_differences.csv. Verification remains failed."
    )
write_json(out / "native_checks.json", checks)
summary = json.loads((out / "summary.json").read_text())
summary["native_checks"] = checks
write_json(out / "summary.json", summary)
print(
    bad[
        ["date", "symbol", "side", "quantity_reference", "quantity_native", "price_reference", "price_native"]
    ].to_string(index=False)
)
print("Array/reference exact trade and NAV parity:", difference)
