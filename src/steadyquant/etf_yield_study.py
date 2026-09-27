"""Research-only ChinaBond rate/credit overlays for ETF allocations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .metrics import window_performance

YIELD_ROOT = ROOT / "data/chinabond"
GOV = "中债国债收益率曲线"
AAA = "中债中短期票据收益率曲线(AAA)"
FAMILIES = ("base", "rising_10y", "rising_and_high_10y", "steepening_curve", "credit_stress")


def align_yields(raw: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Align dated curves and lag one full ETF session after publication."""
    pieces = []
    for label, fields in ((GOV, ["1年", "3年", "10年"]), (AAA, ["3年"])):
        frame = raw[raw["曲线名称"] == label][["日期", *fields]].copy()
        if frame.empty or frame["日期"].duplicated().any():
            raise ValueError(f"Missing/duplicate ChinaBond curve: {label}")
        frame["date"] = pd.to_datetime(frame["日期"])
        frame = frame.set_index("date")[fields].apply(pd.to_numeric, errors="coerce")
        frame = frame.rename(columns={name: f"{label}_{name}" for name in fields})
        pieces.append(frame)
    combined = pd.concat(pieces, axis=1).sort_index()
    joined = pd.merge_asof(
        pd.DataFrame({"date": dates}), combined.reset_index(), on="date",
        direction="backward", tolerance=timedelta(days=7),
    ).set_index("date").shift(1)
    return pd.DataFrame({
        "gov_1y": joined[f"{GOV}_1年"],
        "gov_3y": joined[f"{GOV}_3年"],
        "gov_10y": joined[f"{GOV}_10年"],
        "aaa_3y": joined[f"{AAA}_3年"],
    }, index=dates)


def overlay_rules(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Fixed and economically directed rules; no sign or parameter search."""
    y = features.gov_10y
    slope = y - features.gov_1y
    spread = features.aaa_3y - features.gov_3y
    return {
        "rising_10y": (y - y.shift(63)) > 0.25,
        "rising_and_high_10y": ((y - y.shift(63)) > 0.15) & (y > y.rolling(126).mean()),
        "steepening_curve": ((slope - slope.shift(63)) > 0.25) & ((y - y.shift(63)) > 0.10),
        "credit_stress": ((spread - spread.shift(63)) > 0.20) &
                         (spread > spread.rolling(252, min_periods=126).quantile(0.75)),
    }


def make_targets(data: dict, cfg: dict, curves: pd.DataFrame):
    base, _ = adaptive_weights(data, cfg)
    flags = overlay_rules(curves)
    result = {"base": base}
    for name in FAMILIES[1:]:
        flag = flags[name].reindex(base.index).fillna(False)
        if name == "credit_stress":
            target = base.copy()
            symbols = [a["symbol"] for a in cfg["assets"] if a["bucket"] == "cn_equity"]
            target[symbols] = target[symbols].mul(1 - .25 * flag.astype(float), axis=0)
        else:
            cap = pd.Series(np.where(flag, .15, .30), index=base.index)
            target, _ = adaptive_weights(data, cfg, bond_cap=cap)
            if (target["511010.SH"] > cap + 1e-8).any():
                raise AssertionError("Bond cap violated")
        if target.min().min() < 0 or target.max().max() > .30 + 1e-10 or target.sum(axis=1).max() > 1 + 1e-10:
            raise AssertionError(f"Allocation violation: {name}")
        result[name] = target
    return result, flags


def run():
    cfg = {**load_active_config(), "initial_cash": 200000}
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    path = YIELD_ROOT / "yield_curves.parquet"
    manifest = json.loads((YIELD_ROOT / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("ChinaBond snapshot hash mismatch")
    raw = pd.read_parquet(path)
    end = min(str(d.date.max().date()) for d in data.values())
    end = min(end, str(pd.to_datetime(raw["日期"]).max().date()))
    # Preserve full ETF history for warm-up, but evaluate only through yields.
    old_end = pd.Timestamp("2022-12-30")
    index = next(iter(data.values())).date
    features = align_yields(raw, pd.DatetimeIndex(index))
    targets, flags = make_targets(data, cfg, features)
    old_data = {s: d[d.date <= old_end] for s, d in data.items()}
    old_raw = raw[pd.to_datetime(raw["日期"]) <= old_end]
    old_features = align_yields(old_raw, pd.DatetimeIndex(next(iter(old_data.values())).date))
    old_targets, _ = make_targets(old_data, cfg, old_features)
    for name in FAMILIES:
        pd.testing.assert_frame_equal(old_targets[name], targets[name].loc[:old_end], check_freq=False)
    out = ROOT / "outputs/etf_yield_study" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "protocol.json", dict(
        design="Four fixed ChinaBond rate/credit hypotheses, lagged one full market session; signal close, next open ETF trading. Only bond cap or CN equity target is reduced; no leverage or added symbols.",
        rules=dict(rising_10y="10-year government yield rises >0.25 percentage points over 63 sessions: bond cap 15%",
                   rising_and_high_10y="Yield rises >0.15pp over 63 sessions and above 126-session mean: bond cap 15%",
                   steepening_curve="10Y-1Y slope steepens >0.25pp and 10Y rises >0.10pp over 63 sessions: bond cap 15%",
                   credit_stress="AAA 3Y minus government 3Y spread rises >0.20pp and exceeds its rolling upper quartile: CN equity targets times0.75"),
        selection="2019-2022 Sharpe >= base+0.05, CAGR >= base, DD no worse by 2pp; 2023+ review only. All historical periods are retrospective, not virgin holdout.",
        costs="20万元, 1.5bp/min5 commission, 5bp slippage, 100-share lots; double-cost sensitivity",
        data_end=end, etf_sha256=cache.snapshot_digest, yield_manifest=manifest,
        protected_sha256=hashes,
        limitations="ChinaBond historical vendor curves are not archived publication vintages; one full market-session lag protects timestamp ambiguity, not later data revisions.",
    ))
    features.to_parquet(out / "aligned_yields.parquet")
    pd.DataFrame(flags).to_parquet(out / "regime_flags.parquet")
    rows, equities = [], {}
    for name in FAMILIES:
        print(f"ChinaBond {name}", flush=True)
        targets[name].to_parquet(out / f"targets_{name}.parquet")
        result = simulate(data, targets[name], cfg, start="2019-01-01", end=end)
        equities[name] = result.equity.equity
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
    intervals = {name: paired_sharpe_interval(equities[name], equities["base"])
                 for name in FAMILIES if name != "base"}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    write_json(out / "summary.json", dict(passed_train=passed, active_unchanged=True,
                                          prefix_checks=True, paired_intervals_unadjusted=intervals,
                                          active_days={name: int(flag.sum()) for name, flag in flags.items()}))
    print(table[["name", "fresh_start", "mode", "period", "cagr", "sharpe", "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out
