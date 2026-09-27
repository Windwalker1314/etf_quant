"""Research-only comparison of leaving idle cash versus a money-market ETF."""
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

MONEY_ETF = "511880.SH"


def cash_sleeve(base: pd.DataFrame, money_bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fill only uninvested budget while retaining 1% cash and a 30% ETF cap."""
    money = money_bars.set_index("date").reindex(base.index)
    eligible = money.close.notna() & money.volume.gt(0)
    eligible &= money.amount.rolling(20, min_periods=20).mean().ge(cfg["min_amount"])
    spare = (1 - cfg["cash_buffer"] - base.sum(axis=1)).clip(lower=0, upper=.30)
    result = base.copy()
    result[MONEY_ETF] = spare.where(eligible, 0.0)
    if result.min().min() < 0 or result.max().max() > .30 + 1e-10 or result.sum(axis=1).max() > 1 + 1e-10:
        raise AssertionError("Cash sleeve allocation invariant failed")
    return result


def run() -> Path:
    cfg = {**load_active_config(), "initial_cash": 200000}
    if cfg["model"] != "adaptive" or cfg["rebalance_frequency"] != "monthly_first_session":
        raise ValueError("Study expects active monthly adaptive policy")
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cash_cfg = {**cfg, "assets": [*cfg["assets"], dict(symbol=MONEY_ETF, name="银华货币ETF-A",
                                                  kind="fund", bucket="cash_equivalent", weight=0.0)]}
    cache = Cache()
    data = cache.load(cash_cfg)
    core = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
    end_dates = {str(d.date.max().date()) for d in data.values()}
    if len(end_dates) != 1:
        raise ValueError(f"Price ends disagree: {end_dates}")
    end = end_dates.pop()
    base, _ = target_weights(core, cfg)
    sleeve = cash_sleeve(base, data[MONEY_ETF], cfg)
    cutoff = pd.Timestamp("2022-12-30")
    old_core = {s: d[d.date <= cutoff] for s, d in core.items()}
    old_money = data[MONEY_ETF][data[MONEY_ETF].date <= cutoff]
    old_base, _ = target_weights(old_core, cfg)
    pd.testing.assert_frame_equal(old_base, base.loc[:cutoff], check_freq=False)
    pd.testing.assert_frame_equal(cash_sleeve(old_base, old_money, cfg), sleeve.loc[:cutoff], check_freq=False)
    out = ROOT / "outputs/etf_cash_sleeve_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), end=end, candidate=MONEY_ETF,
        rule="Preserve every original 10-ETF target; allocate at most 30% of the otherwise idle cash to listed money-market ETF 511880.SH, retaining 1% cash. Monthly first-session rebalance only. No selection by historical return.",
        selection="Train 2019-2022: Sharpe >= base+0.05, CAGR >= base, drawdown no worse by 2pp. Review 2023+ without reselection. Already-observed history.",
        costs="20万元, next-open raw-price ledger, 100-share lot (about 1万元), 1.5bp/minimum5元 and 5bp slip; doubled-cost check.",
        limitations="Money-market ETF total return depends on distribution/adjustment data; ETF is not insured bank cash and may have price/liquidity risk. Existing ledger models ETF distributions as reinvestment, not actual settlement.",
        data_sha256=cache.snapshot_digest, protected_sha256=hashes,
    ))
    rows, curves = [], {}
    for name, target, d, c in [("base", base, core, cfg), ("cash_sleeve", sleeve, data, cash_cfg)]:
        print(f"Cash sleeve {name}", flush=True)
        target.to_parquet(out / f"targets_{name}.parquet")
        result = simulate(d, target, c, start="2019-01-01", end=end)
        curves[name] = result.equity.equity
        result.equity.to_parquet(out / f"equity_{name}.parquet")
        for period, lo, hi in [("2019-train", "2019-01-01", "2022-12-31"),
                                ("2023-review", "2023-01-01", None),
                                ("2019-now", "2019-01-01", None)]:
            rows.append(dict(name=name, mode="normal", period=period, trades=len(result.trades),
                             **window_performance(result.equity.equity, lo, hi, cfg["risk_free_rate"])))
        stressed = simulate(d, target, c, start="2019-01-01", end=end, cost_multiplier=2)
        rows.append(dict(name=name, mode="double", period="2019-now", trades=len(stressed.trades),
                         **window_performance(stressed.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        long_run = simulate(d, target, c, start="2015-01-05", end=end)
        for period, lo, hi in [("2015-2018", "2015-01-01", "2018-12-31"),
                                ("2015-now", "2015-01-01", None)]:
            rows.append(dict(name=name, mode="normal", period=period, trades=len(long_run.trades),
                             **window_performance(long_run.equity.equity, lo, hi, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    train = table[(table["mode"] == "normal") & (table.period == "2019-train")].set_index("name")
    b, g = train.loc["base"], train.loc["cash_sleeve"]
    passed = bool(g.sharpe >= b.sharpe + .05 and g.cagr >= b.cagr and
                  g.max_drawdown >= b.max_drawdown - .02)
    interval = paired_sharpe_interval(curves["cash_sleeve"], curves["base"])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cash_cfg) == cache.snapshot_digest
    write_json(out / "summary.json", dict(passed_train=passed, prefix_checks=True,
                                          active_unchanged=True, paired_interval_unadjusted=interval))
    print(table[["name", "mode", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
