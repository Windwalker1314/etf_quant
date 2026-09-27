"""Reproducible audit of one stock-only daily recipe, never a live order source."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime

import pandas as pd
from stock_daily_factor_sweep import WINDOWS, build_panels, choose_targets

from steadyquant.composite_ledger import VERSION, simulate_composite
from steadyquant.composite_study import independent_replay, quarantine_stock_placeholders
from steadyquant.config import ROOT, load_config, write_json
from steadyquant.data import TZ, Cache
from steadyquant.metrics import window_performance, yearly
from steadyquant.stock_data import load_stock_snapshot


def load_snapshot(universe: str):
    if universe == "csi300":
        return load_stock_snapshot()
    source = ROOT / "data/stocks500"
    manifest = json.loads((source / "manifest.json").read_text())
    snapshot = {}
    for name, info in manifest["files"].items():
        path = source / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"CSI500 snapshot digest mismatch: {name}")
        snapshot[path.stem] = pd.read_parquet(path)
    return snapshot, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["csi300", "csi500"], required=True)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--budget", required=True)
    args = parser.parse_args()
    output = ROOT / "outputs/stock_daily/pure_audit" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    snapshot, manifest = load_snapshot(args.universe)
    snapshot, quarantined = quarantine_stock_placeholders(snapshot, output)
    cal = pd.read_parquet(ROOT / "data/calendar.parquet")
    dates = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1)
                                 & cal.cal_date.astype(str).between("20140101", manifest["end"]),
                                 "cal_date"])).sort_values().unique()
    signal_etf = Cache().read("510300.SH").set_index("date")
    market = (signal_etf.close * signal_etf.adj_factor).reindex(dates)
    scores, budgets, close = build_panels(snapshot, dates, market)
    targets = choose_targets(scores[args.factor], budgets[args.budget], close)
    cfg = load_config(ROOT / "configs/family.yaml")
    cfg.update(initial_cash=200000, rebalance_frequency="daily", rebalance_band=0.02,
               stock_rebalance_band=0.02, commission_bps=3.0, cash_rate=0.0)
    results = {}
    for multiplier in (1, 2):
        run = simulate_composite({}, snapshot, targets, cfg, start="2016-01-01",
                                 cost_multiplier=multiplier)
        metrics = {period: window_performance(run.equity.equity, *window,
                                              rf=cfg["risk_free_rate"])
                   for period, window in WINDOWS.items()}
        cash_share = run.equity.cash / run.equity.equity
        blocks = int(run.events.type.eq("qualification_block").sum()) if not run.events.empty else 0
        results[f"cost_{multiplier}x"] = dict(metrics=metrics, trades=len(run.trades),
            qualification_blocks=blocks, mean_stock_exposure=float(1 - cash_share.mean()),
            p10_stock_exposure=float((1 - cash_share).quantile(0.1)))
        if multiplier == 1:
            results["cost_1x"]["independent_replay"] = independent_replay(run, {}, snapshot, cfg)
            run.equity.to_parquet(output / "equity.parquet")
            run.trades.to_parquet(output / "trades.parquet", index=False)
            run.events.to_parquet(output / "events.parquet", index=False)
            targets.to_parquet(output / "targets.parquet")
            yearly(run.equity.equity, cfg["risk_free_rate"]).to_csv(output / "yearly.csv", index=False)
    benchmark = {}
    code = "510300.SH" if args.universe == "csi300" else "510500.SH"
    bars = Cache().read(code).set_index("date")
    proxy = (bars.close * bars.adj_factor).reindex(dates)
    for period, window in WINDOWS.items():
        benchmark[period] = window_performance(proxy, *window, rf=cfg["risk_free_rate"])
    stress = simulate_composite({}, snapshot, targets, cfg, start="2015-01-01", end="2015-12-31")
    stress_2015 = window_performance(stress.equity.equity, "2015-01-01", "2015-12-31",
                                     rf=cfg["risk_free_rate"])
    write_json(output / "summary.json", dict(
        universe=args.universe, factor=args.factor, budget=args.budget,
        snapshot_end=manifest["end"], execution_version=VERSION,
        quarantined=quarantined, stock_only=True, benchmark_proxy=code,
        benchmark_proxy_excludes_trading_costs=True, results=results, benchmark=benchmark,
        stress_2015_fresh_capital=stress_2015,
        selection_warning="Selected after exploratory sweep; validation is historical, not unseen forward data",
        live_policy_unchanged=True))
    print(output, flush=True)


if __name__ == "__main__":
    main()
