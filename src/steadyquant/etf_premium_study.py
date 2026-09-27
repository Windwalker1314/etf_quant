"""Research-only, publication-dated QDII premium risk guard."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .metrics import window_performance
from .strategy import target_weights

NAV_ROOT = ROOT / "data/fund_nav"
QDII = ("513100.SH", "513500.SH", "159920.SZ")


def publication_dated_premium(
    data: dict[str, pd.DataFrame], nav: dict[str, pd.DataFrame], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Use the price on the NAV date, only after the NAV announcement is known."""
    output = pd.DataFrame(np.nan, index=dates, columns=list(data), dtype=float)
    for symbol, raw in nav.items():
        frame = raw.copy()
        if not {"ts_code", "ann_date", "nav_date", "unit_nav"}.issubset(frame):
            raise ValueError(f"Incomplete NAV history for {symbol}")
        if not frame.ts_code.eq(symbol).all():
            raise ValueError(f"NAV symbol mismatch for {symbol}")
        frame["ann_date"] = pd.to_datetime(frame.ann_date.astype(str), format="%Y%m%d")
        frame["nav_date"] = pd.to_datetime(frame.nav_date.astype(str), format="%Y%m%d")
        frame = frame[(frame.ann_date >= frame.nav_date) & (frame.unit_nav > 0)]
        frame = frame.sort_values(["ann_date", "nav_date"]).drop_duplicates("ann_date", keep="last")
        close = data[symbol].set_index("date").close
        frame["known_close"] = close.reindex(frame.nav_date).to_numpy()
        frame["premium"] = frame.known_close / frame.unit_nav - 1
        observations = frame.set_index("ann_date").premium.where(lambda v: v.between(-0.5, 0.5))
        observations = observations.dropna()
        # A release on signal day may have arrived after close; one complete
        # market-session lag makes next-open execution unambiguous.
        aligned = pd.merge_asof(
            pd.DataFrame({"date": dates}),
            observations.rename_axis("ann_date").reset_index().sort_values("ann_date"),
            left_on="date", right_on="ann_date", direction="backward",
            tolerance=timedelta(days=7),
        )
        output[symbol] = aligned.premium.shift(1).to_numpy()
    return output


def guarded_targets(base: pd.DataFrame, premium: pd.DataFrame) -> pd.DataFrame:
    """3%-10% observed premium linearly removes at most half QDII exposure."""
    penalty = ((premium - 0.03) / 0.07).clip(0, 1).fillna(0)
    result = base * (1 - 0.5 * penalty)
    if result.min().min() < 0 or result.max().max() > .3 + 1e-10 or result.sum(axis=1).max() > 1 + 1e-10:
        raise AssertionError("Premium guard violates allocation bounds")
    return result


def run() -> Path:
    cfg = {**load_active_config(), "initial_cash": 200000}
    if cfg["model"] != "adaptive" or cfg["rebalance_frequency"] != "monthly_first_session":
        raise ValueError("Study expects active monthly adaptive policy")
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    manifest, nav = {}, {}
    for symbol in QDII:
        path = NAV_ROOT / f"{symbol}.parquet"
        if not path.is_file():
            raise FileNotFoundError(symbol)
        frame = pd.read_parquet(path)
        nav[symbol] = frame
        manifest[symbol] = dict(rows=len(frame), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                latest_announcement=str(frame.ann_date.max()))
    end_dates = {str(frame.date.max().date()) for frame in data.values()}
    if len(end_dates) != 1:
        raise ValueError(f"Price ends disagree: {end_dates}")
    end = end_dates.pop()
    base, _ = target_weights(data, cfg)
    premium = publication_dated_premium(data, nav, base.index)
    targets = {"base": base, "premium_guard": guarded_targets(base, premium)}
    cutoff = pd.Timestamp("2022-12-30")
    old_data = {s: f[f.date <= cutoff] for s, f in data.items()}
    old_nav = {s: f[pd.to_datetime(f.ann_date.astype(str), format="%Y%m%d") <= cutoff] for s, f in nav.items()}
    old_base, _ = target_weights(old_data, cfg)
    old_premium = publication_dated_premium(old_data, old_nav, old_base.index)
    pd.testing.assert_frame_equal(old_base, base.loc[:cutoff], check_freq=False)
    pd.testing.assert_frame_equal(guarded_targets(old_base, old_premium), targets["premium_guard"].loc[:cutoff], check_freq=False)
    out = ROOT / "outputs/etf_premium_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), end=end, qdii=QDII,
        rule="Premium = ETF raw close on NAV date / unit NAV - 1, introduced only on announcement date; announcements lag one further market session. At 3%-10% premium, reduce QDII target by 0%-50%; no added exposure, all other weights unchanged.",
        selection="Train 2019-2022: require Sharpe >= base+0.05, CAGR >= base, DD no worse by 2pp. Review 2023+ without reselection. All histories already seen, not new holdout.",
        costs="20万元, monthly first session, next-open raw-price ledger, 1.5bp/minimum5元 + 5bp slip; doubled-cost check.",
        limitations="Published NAV may be revised; no archival release vintages or exact intraday announcement timestamp. NAV is often 2+ days old for QDII, so this is a lagged premium risk signal, not a current executable arbitrage quote.",
        data_sha256=cache.snapshot_digest, nav_manifest=manifest, protected_sha256=hashes,
    ))
    rows, curves = [], {}
    premium.to_parquet(out / "lagged_premium.parquet")
    for name, target in targets.items():
        print(f"Premium {name}", flush=True)
        target.to_parquet(out / f"targets_{name}.parquet")
        r = simulate(data, target, cfg, start="2019-01-01", end=end)
        curves[name] = r.equity.equity
        r.equity.to_parquet(out / f"equity_{name}.parquet")
        for period, lo, hi in [("2019-train", "2019-01-01", "2022-12-31"),
                                ("2023-review", "2023-01-01", None),
                                ("2019-now", "2019-01-01", None)]:
            rows.append(dict(name=name, mode="normal", period=period, trades=len(r.trades),
                             **window_performance(r.equity.equity, lo, hi, cfg["risk_free_rate"])))
        stressed = simulate(data, target, cfg, start="2019-01-01", end=end, cost_multiplier=2)
        rows.append(dict(name=name, mode="double", period="2019-now", trades=len(stressed.trades),
                         **window_performance(stressed.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
        long_run = simulate(data, target, cfg, start="2015-01-05", end=end)
        for period, lo, hi in [("2015-2018", "2015-01-01", "2018-12-31"),
                                ("2015-now", "2015-01-01", None)]:
            rows.append(dict(name=name, mode="normal", period=period, trades=len(long_run.trades),
                             **window_performance(long_run.equity.equity, lo, hi, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    train = table[(table["mode"] == "normal") & (table.period == "2019-train")].set_index("name")
    b, g = train.loc["base"], train.loc["premium_guard"]
    passed = bool(g.sharpe >= b.sharpe + .05 and g.cagr >= b.cagr and
                  g.max_drawdown >= b.max_drawdown - .02)
    interval = paired_sharpe_interval(curves["premium_guard"], curves["base"])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    write_json(out / "summary.json", dict(passed_train=passed, prefix_checks=True,
                                          active_unchanged=True, paired_interval_unadjusted=interval,
                                          premium_coverage={s: int(premium[s].notna().sum()) for s in QDII}))
    print(table[["name", "mode", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
