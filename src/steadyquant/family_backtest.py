"""Compact backtest for the family website; no factor-lab research artifacts."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .backtest import simulate
from .config import ROOT, fingerprint, write_json
from .data import TZ, Cache, DataError
from .index_benchmark import benchmark_digest, load_shanghai_composite
from .strategy import target_weights
from .strategy_catalog import DEFAULT_STRATEGY_ID, research_pointer, strategy_by_id


def run_family_backtest(cfg: dict, cache: Cache | None = None,
                        strategy_id: str = DEFAULT_STRATEGY_ID) -> Path:
    strategy_spec = strategy_by_id(strategy_id)
    cache = cache or Cache()
    data = cache.load(cfg)
    digest = cache.snapshot_digest
    weights, _ = target_weights(data, cfg)
    strategy = simulate(data, weights, cfg)
    baseline_weights, _ = target_weights(data, cfg, baseline=True)
    baseline = simulate(data, baseline_weights, cfg)

    run_id = datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + "-" + strategy_id + "-" + fingerprint(cfg)[:6]
    output = ROOT / "outputs/research" / run_id
    output.mkdir(parents=True, exist_ok=True)
    equity = strategy.equity[["equity"]].copy()
    equity["baseline"] = baseline.equity["equity"]
    benchmark = {"symbol": "000001.SH", "name": "上证综指", "type": "price_index", "status": "unavailable"}
    try:
        index_close = load_shanghai_composite(cache, equity.index)
        equity["shanghai_composite"] = cfg["initial_cash"] * index_close / index_close.iloc[0]
        benchmark.update(status="ok", sha256=benchmark_digest(cache),
                         start=str(index_close.index[0].date()), end=str(index_close.index[-1].date()))
    except (DataError, OSError, ValueError) as exc:
        benchmark["error_type"] = type(exc).__name__
    equity.to_parquet(output / "equity.parquet")
    write_json(output / "protocol.json", {"strategy_id": strategy_id, "config_hash": fingerprint(cfg),
                                           "data_sha256": digest, "benchmark": benchmark})
    pointer = {"path": str(output), "run_id": run_id}
    write_json(research_pointer(ROOT, strategy_spec), pointer)
    if strategy_id == DEFAULT_STRATEGY_ID:
        # Older deployments and local tools still read this pointer.
        write_json(ROOT / "outputs/latest_research.json", pointer)
    return output
