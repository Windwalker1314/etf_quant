"""Research-only ETF universe breadth comparison with fixed asset-class budgets."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .metrics import window_performance
from .strategy import target_weights

ADDITIONS = {
    "cn_breadth": [
        ("510880.SH", "红利ETF", "cn_equity"),
        ("512100.SH", "中证1000ETF", "cn_equity"),
    ],
    "developed_breadth": [
        ("513030.SH", "德国ETF", "us_equity"),
        ("513520.SH", "日经ETF", "us_equity"),
    ],
    "commodity_breadth": [
        ("159980.SZ", "有色期货ETF", "commodity"),
        ("159981.SZ", "能化期货ETF", "commodity"),
    ],
}
FAMILIES = ("base", "cn_breadth", "developed_breadth", "commodity_breadth", "all_breadth")


def expanded_config(base: dict, name: str) -> dict:
    if name not in FAMILIES:
        raise ValueError(name)
    cfg = {**base, "assets": [a.copy() for a in base["assets"]]}
    additions = [a for group, assets in ADDITIONS.items() if name in (group, "all_breadth") for a in assets]
    for symbol, label, bucket in additions:
        cfg["assets"].append(dict(symbol=symbol, name=label, kind="fund", bucket=bucket, weight=0.0))
    budgets = {bucket: sum(a["weight"] for a in base["assets"] if a["bucket"] == bucket)
               for bucket in {a["bucket"] for a in base["assets"]}}
    for bucket in budgets:
        original = [a for a in base["assets"] if a["bucket"] == bucket]
        members = [a for a in cfg["assets"] if a["bucket"] == bucket]
        if len(members) > len(original):
            for asset in members:
                asset["weight"] = budgets[bucket] / len(members)
    if abs(sum(a["weight"] for a in cfg["assets"]) - sum(a["weight"] for a in base["assets"])) > 1e-9:
        raise AssertionError("Asset-class budgets changed")
    return cfg


def run() -> Path:
    base = {**load_active_config(), "initial_cash": 200000}
    if base["model"] != "adaptive" or base["rebalance_frequency"] != "monthly_first_session":
        raise ValueError("Study expects active monthly adaptive policy")
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    configs = {name: expanded_config(base, name) for name in FAMILIES}
    cache = Cache()
    all_data = cache.load(configs["all_breadth"])
    end_dates = {str(d.date.max().date()) for d in all_data.values()}
    if len(end_dates) != 1:
        raise ValueError(f"ETF price histories have inconsistent ends: {end_dates}")
    end = end_dates.pop()
    outputs = ROOT / "outputs/etf_universe_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    outputs.mkdir(parents=True, exist_ok=False)
    write_json(outputs / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), end=end,
        candidates={name: [a["symbol"] for a in cfg["assets"]] for name, cfg in configs.items()},
        design="Three predetermined asset-class breadth changes and their union. Existing CN, developed equity and commodity budget is divided equally within its class; gold, HK and bond budgets unchanged. No ranking by historical returns.",
        grouping="Germany and Japan are conservatively counted under the existing developed/US equity risk bucket, capped at 30%; label is research-only.",
        selection="2019-2022 carried train: Sharpe >= base+0.05, CAGR >= base, drawdown no worse by 2pp; otherwise base. 2023+ review only. Already-seen history, not virgin holdout.",
        costs="20万元, monthly first session, next-open actual-price ledger, 1.5bp commission/minimum5元 and 5bp slippage; doubled-cost sensitivity.",
        limitation="Preselected currently surviving ETF universe; some ETFs list in 2019/2020; historical QDII premium missing; no claim of genuine out-of-sample alpha.",
        data_sha256=cache.snapshot_digest, protected_sha256=hashes,
    ))
    rows, curves = [], {}
    for name in FAMILIES:
        cfg = configs[name]
        data = {a["symbol"]: all_data[a["symbol"]] for a in cfg["assets"]}
        print(f"Universe {name}", flush=True)
        targets, _ = target_weights(data, cfg)
        targets.to_parquet(outputs / f"targets_{name}.parquet")
        cutoff = pd.Timestamp("2022-12-30")
        prefix = {s: d[d.date <= cutoff] for s, d in data.items()}
        old, _ = target_weights(prefix, cfg)
        pd.testing.assert_frame_equal(old, targets.loc[:cutoff], check_freq=False)
        result = simulate(data, targets, cfg, start="2019-01-01", end=end)
        curves[name] = result.equity.equity
        result.equity.to_parquet(outputs / f"equity_{name}.parquet")
        for period, lo, hi in [
            ("2019-train", "2019-01-01", "2022-12-31"),
            ("2023-review", "2023-01-01", None),
            ("2019-now", "2019-01-01", None),
        ]:
            rows.append(dict(name=name, start_mode="fresh2019", cost="normal", period=period,
                             trades=len(result.trades), **window_performance(result.equity.equity, lo, hi, cfg["risk_free_rate"])))
        stressed = simulate(data, targets, cfg, start="2019-01-01", end=end, cost_multiplier=2)
        rows.append(dict(name=name, start_mode="fresh2019", cost="double", period="2019-now",
                         trades=len(stressed.trades), **window_performance(stressed.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        fresh23 = simulate(data, targets, cfg, start="2023-01-01", end=end)
        rows.append(dict(name=name, start_mode="fresh2023", cost="normal", period="2023-now",
                         trades=len(fresh23.trades), **window_performance(fresh23.equity.equity, "2023-01-01", None, cfg["risk_free_rate"])))
        long_run = simulate(data, targets, cfg, start="2015-01-05", end=end)
        for period, lo, hi in [("2015-2018", "2015-01-01", "2018-12-31"),
                                ("2015-now", "2015-01-01", None)]:
            rows.append(dict(name=name, start_mode="fresh2015", cost="normal", period=period,
                             trades=len(long_run.trades), **window_performance(long_run.equity.equity, lo, hi, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(outputs / "comparison.csv", index=False)
    train = table[(table.start_mode == "fresh2019") & (table.cost == "normal") & (table.period == "2019-train")].set_index("name")
    b = train.loc["base"]
    eligible = train[(train.sharpe >= b.sharpe + .05) & (train.cagr >= b.cagr) &
                     (train.max_drawdown >= b.max_drawdown - .02)]
    selected = str(eligible.sharpe.idxmax()) if not eligible.empty else "base"
    intervals = {name: paired_sharpe_interval(curves[name], curves["base"])
                 for name in FAMILIES if name != "base"}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(configs["all_breadth"]) == cache.snapshot_digest
    write_json(outputs / "summary.json", dict(selected_on_train=selected, prefix_checks=True,
                                                active_unchanged=True, paired_intervals_unadjusted=intervals))
    print(table[["name", "start_mode", "cost", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={outputs}", flush=True)
    return outputs
