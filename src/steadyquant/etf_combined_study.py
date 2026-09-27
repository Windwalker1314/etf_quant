"""Exploratory combinations of three independent ETF risk guards."""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd

from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .etf_breadth_study import STOCK_ROOT, breadth_signals, breadth_targets
from .etf_premium_study import NAV_ROOT, QDII, guarded_targets, publication_dated_premium
from .etf_yield_study import YIELD_ROOT, align_yields, make_targets
from .factors import adjusted_frame
from .metrics import window_performance
from .strategy import target_weights

FAMILIES = ("base", "breadth_yield", "breadth_premium", "yield_premium", "all_three")


def combine(base, breadth_target, yield_target, premium_target):
    """Apply independent downside-only guards to distinct asset buckets."""
    bf = breadth_target.div(base.where(base > 0)).fillna(1).clip(0, 1)
    pf = premium_target.div(base.where(base > 0)).fillna(1).clip(0, 1)
    result = {
        "base": base,
        "breadth_yield": yield_target * bf,
        "breadth_premium": premium_target * bf,
        "yield_premium": yield_target * pf,
        "all_three": yield_target * bf * pf,
    }
    for name, frame in result.items():
        if frame.min().min() < 0 or frame.max().max() > .30 + 1e-10 or frame.sum(axis=1).max() > 1 + 1e-10:
            raise AssertionError(f"Invalid combined targets: {name}")
    return result


def run():
    cfg = {**load_active_config(), "initial_cash": 200000}
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    daily = pd.read_parquet(STOCK_ROOT / "daily.parquet")
    members = pd.read_parquet(STOCK_ROOT / "members.parquet")
    yields = pd.read_parquet(YIELD_ROOT / "yield_curves.parquet")
    nav = {s: pd.read_parquet(NAV_ROOT / f"{s}.parquet") for s in QDII}
    end = min(str(frame.date.max().date()) for frame in data.values())
    end = min(end, pd.to_datetime(daily.trade_date.astype(str), format="%Y%m%d").max().strftime("%Y-%m-%d"))
    end = min(end, str(pd.to_datetime(yields["日期"]).max().date()))
    base, _ = target_weights(data, cfg)
    dates = base.index
    breadth = breadth_signals(daily, members, dates)
    cn_proxy = adjusted_frame(data).query("symbol == '510300.SH'").set_index("date").close.reindex(dates)
    breadth_target = breadth_targets(base, breadth, cn_proxy)["price_divergence"]
    rate = align_yields(yields, dates)
    yield_target = make_targets(data, cfg, rate)[0]["rising_10y"]
    premium = publication_dated_premium(data, nav, dates)
    premium_target = guarded_targets(base, premium)
    targets = combine(base, breadth_target, yield_target, premium_target)
    cutoff = pd.Timestamp("2022-12-30")
    old_data = {s: f[f.date <= cutoff] for s, f in data.items()}
    old_daily = daily[daily.trade_date.astype(str) <= "20221230"]
    old_members = members[members.trade_date.astype(str) <= "20221230"]
    old_yields = yields[pd.to_datetime(yields["日期"]) <= cutoff]
    old_nav = {s: f[pd.to_datetime(f.ann_date.astype(str), format="%Y%m%d") <= cutoff]
               for s, f in nav.items()}
    old_base, _ = target_weights(old_data, cfg)
    old_dates = old_base.index
    old_breadth = breadth_signals(old_daily, old_members, old_dates)
    old_bt = breadth_targets(old_base, old_breadth, cn_proxy.loc[:cutoff])["price_divergence"]
    old_rate = align_yields(old_yields, old_dates)
    old_yt = make_targets(old_data, cfg, old_rate)[0]["rising_10y"]
    old_premium = publication_dated_premium(old_data, old_nav, old_dates)
    old_pt = guarded_targets(old_base, old_premium)
    old_targets = combine(old_base, old_bt, old_yt, old_pt)
    for name in FAMILIES:
        pd.testing.assert_frame_equal(old_targets[name], targets[name].loc[:cutoff], check_freq=False)
    out = ROOT / "outputs/etf_combined_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), end=end,
        design="Exploratory combination after component results were observed; market breadth cuts CN ETF by50% during price/breadth divergence, lagged 10Y yield rise above0.25pp in63 sessions cuts bond cap to15%, publication-dated QDII premium cuts QDII by up to50%. No new asset, leverage or monthly schedule change.",
        selection="2019-2022 train requires Sharpe>=base+0.05, CAGR>=base, DD no worse by2pp. All history already inspected, and components chosen after looking at prior outcomes: no confirmatory holdout.",
        costs="20万元, next-open fills, 1.5bp/min5 commission, 5bp slippage; 100-share lots, double-cost sensitivity",
        etf_sha256=cache.snapshot_digest,
        data_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (STOCK_ROOT / "daily.parquet", STOCK_ROOT / "members.parquet",
                               YIELD_ROOT / "yield_curves.parquet", *(NAV_ROOT / f"{s}.parquet" for s in QDII))},
        protected_sha256=hashes,
    ))
    rows, curves = [], {}
    for name in FAMILIES:
        print(f"Combined {name}", flush=True)
        targets[name].to_parquet(out / f"targets_{name}.parquet")
        result = simulate(data, targets[name], cfg, start="2019-01-01", end=end)
        curves[name] = result.equity.equity
        result.equity.to_parquet(out / f"equity_{name}.parquet")
        for period, lo, hi in (("2019-train", "2019-01-01", "2022-12-31"),
                               ("2023-review", "2023-01-01", None),
                               ("2019-now", "2019-01-01", None)):
            rows.append(dict(name=name, fresh_start="2019", mode="normal", period=period,
                             trades=len(result.trades), **window_performance(result.equity.equity, lo, hi, cfg["risk_free_rate"])))
        double = simulate(data, targets[name], cfg, start="2019-01-01", end=end, cost_multiplier=2)
        rows.append(dict(name=name, fresh_start="2019", mode="double", period="2019-now",
                         trades=len(double.trades),
                         **window_performance(double.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        long = simulate(data, targets[name], cfg, start="2015-01-05", end=end)
        rows.append(dict(name=name, fresh_start="2015", mode="normal", period="2015-now",
                         trades=len(long.trades),
                         **window_performance(long.equity.equity, "2015-01-05", None, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    train = table[(table.fresh_start == "2019") & (table["mode"] == "normal") &
                  (table.period == "2019-train")].set_index("name")
    b = train.loc["base"]
    passed = train[(train.sharpe >= b.sharpe + .05) & (train.cagr >= b.cagr) &
                   (train.max_drawdown >= b.max_drawdown - .02)].index.tolist()
    intervals = {name: paired_sharpe_interval(curves[name], curves["base"])
                 for name in FAMILIES if name != "base"}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    write_json(out / "summary.json", dict(passed_train=passed, active_unchanged=True,
                                          prefix_checks=True, paired_intervals_unadjusted=intervals))
    print(table[["name", "fresh_start", "mode", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
