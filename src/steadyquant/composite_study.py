"""Frozen six-model stock/ETF study; never mutates the active account strategy."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .composite_ledger import VERSION, simulate_composite
from .config import ROOT, load_config, write_json
from .data import Cache, DataError
from .metrics import performance, window_performance, yearly
from .stock_data import STOCK_ROOT, load_stock_snapshot
from .stock_factors import FAMILIES, build_stock_factors, composite_targets

NAMES = {
    "etf_core": "现行ETF等风险底仓",
    "equity_overlay_25": "25%沪深300敞口对照",
    "equity_overlay_40": "40%沪深300敞口对照",
}
for _family, _name in {
    "quality_value": "质量价值",
    "momentum_defensive": "动量防御",
    "ensemble": "多因子均衡",
}.items():
    for _pct in (25, 40):
        NAMES[f"{_family}_{_pct}"] = f"{_name}·个股上限{_pct}%"


def audit_snapshot(snapshot):
    raw = snapshot["daily"]
    fields = ["open", "close", "high", "low", "vol", "amount"]
    if not np.isfinite(raw[fields].to_numpy(float)).all():
        raise DataError("Non-finite stock OHLCV")
    if (raw[["open", "close", "high", "low"]] <= 0).any().any() or (raw[["vol", "amount"]] < 0).any().any():
        raise DataError("Invalid stock prices/volumes")
    if (raw.high < raw[["open", "close", "low"]].max(axis=1) - 1e-8).any() or (
        raw.low > raw[["open", "close", "high"]].min(axis=1) + 1e-8
    ).any():
        raise DataError("Invalid stock OHLC bounds")
    merged = raw.merge(
        snapshot["adj_factor"], on=["ts_code", "trade_date"], how="left", validate="one_to_one"
    ).merge(snapshot["stk_limit"], on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    if snapshot["financials"].groupby("end_date").ts_code.nunique().min() < 100:
        raise DataError("Suspiciously incomplete financial quarter")
    return dict(
        stocks=raw.ts_code.nunique(),
        rows=len(raw),
        first=raw.trade_date.min(),
        last=raw.trade_date.max(),
        membership_snapshots=snapshot["members"].trade_date.nunique(),
        missing_adjustment_rows=int(merged.adj_factor.isna().sum()),
        missing_limit_rows=int(merged.up_limit.isna().sum()),
        original_report_fraction=float(snapshot["financials"].update_flag.astype(str).eq("0").mean()),
        policy="Missing adjustment excludes factor; missing limit prevents execution. Historical financial revisions are not a certified vendor vintage archive.",
    )


def quarantine_stock_placeholders(snapshot, folder):
    """Recognize zero-volume suspended placeholders, without inventing tradable OHLC."""
    raw = snapshot["daily"]
    mask = raw.vol.eq(0) & raw[["open", "high", "low"]].eq(0).all(axis=1) & raw.close.gt(0)
    bad = raw.loc[mask].copy()
    bad["reason"] = "zero-volume placeholder with zero open/high/low; excluded from factors and fills"
    bad.to_parquet(folder / "quarantined_stock_bars.parquet", index=False)
    return {**snapshot, "daily": raw.loc[~mask].copy()}, len(bad)


def independent_replay(result, etf_data, snapshot, cfg):
    """Replay fills and corporate-event book independently of target/order code."""
    dates = result.equity.index
    raw = snapshot["daily"].rename(columns={"ts_code": "symbol", "trade_date": "date"}).copy()
    raw["date"] = pd.to_datetime(raw.date)
    raw = pd.concat([raw, *etf_data.values()], ignore_index=True)
    prices = raw.pivot(index="date", columns="symbol", values="close").reindex(dates)
    trades = {d: f for d, f in result.trades.groupby("date")} if not result.trades.empty else {}
    events = (
        {d: f.to_dict("records") for d, f in result.events.groupby("date")} if not result.events.empty else {}
    )
    cash, claim = float(cfg["initial_cash"]), 0.0
    qty, bonus, marks = {}, {}, {}
    values = []
    prior = dates[0]
    for date in dates:
        cash *= (1 + cfg["cash_rate"]) ** ((date - prior).days / 365.25)
        prior = date
        day_events = events.get(date, [])
        for e in day_events:
            s, kind = e["symbol"], e["type"]
            if kind == "dividend_entitlement":
                claim += round(e["gross_cash"] - e["tax_reserved"], 2)
                bonus[s] = bonus.get(s, 0) + e["bonus_shares"]
            elif kind == "dividend_cash_payment":
                cash += e["amount"]
                claim -= e["amount"]
            elif kind == "bonus_listing":
                bonus[s] = bonus.get(s, 0) - e["quantity"]
                qty[s] = qty.get(s, 0) + e["quantity"]
            elif kind == "etf_reinvestment":
                qty[s] = qty.get(s, 0) * e["ratio"]
            elif kind == "suspended_ex_reference":
                marks[s] = e["reference_price"]
        for tr in trades.get(date, pd.DataFrame()).itertuples():
            signed = tr.quantity if tr.side == "BUY" else -tr.quantity
            qty[tr.symbol] = qty.get(tr.symbol, 0) + signed
            cash -= signed * tr.price + tr.commission + tr.stamp_tax + tr.transfer_fee
        for e in day_events:
            if e["type"] == "delisting_writeoff":
                qty[e["symbol"]] = qty.get(e["symbol"], 0) - e["quantity"]
        marks.update(prices.loc[date].dropna().to_dict())
        nav = (
            cash
            + claim
            + sum(q * marks[s] for s, q in qty.items() if abs(q) > 1e-8)
            + sum(q * marks[s] for s, q in bonus.items() if abs(q) > 1e-8)
        )
        values.append(nav)
    replay = pd.Series(values, index=dates)
    difference = (replay - result.equity.equity).abs()
    return dict(
        verified=bool(difference.max() < 0.01),
        max_abs_difference=float(difference.max()),
        sessions=len(dates),
        note="Independent replay of every fill and corporate event; stock native AKQuant equivalence is not claimed",
    )


def run_composite():
    cfg = {**load_config(ROOT / "configs/adaptive_paper.yaml"), "initial_cash": 200000}
    active_path = ROOT / "configs/active.yaml"
    active_hash = hashlib.sha256(active_path.read_bytes()).hexdigest()
    snapshot, manifest = load_stock_snapshot()
    folder = ROOT / "outputs/composite" / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    write_json(
        folder / "protocol.json",
        dict(
            **json.loads((STOCK_ROOT / "protocol.json").read_text()),
            manifest=manifest,
            etf_config=cfg,
            execution_version=VERSION,
            active_config_sha256=active_hash,
            extra_controls="ETF only and identical stock-risk regime allocated to CSI300 ETF; controls excluded from candidate selection",
            additional_stress="2015 crash as a separate earlier-period fresh-capital replay for every model; added before observing model returns; no retrospective relabeling as a validation fold",
            taxes="historical stamp/transfer; stock cash dividends reserve max20%; ETF adjustment unit approximation retained",
            corporate_actions="actual record/ex/payment/listing dates; missing dates or unexplained held-stock adjustments block qualification; delisted residuals written off",
        ),
    )
    snapshot, quarantined = quarantine_stock_placeholders(snapshot, folder)
    quality = audit_snapshot(snapshot)
    quality["quarantined_zero_volume_placeholders"] = quarantined
    write_json(folder / "data_quality.json", quality)
    cache = Cache()
    etf_data = cache.load(cfg)
    etf_w, _ = adaptive_weights(etf_data, cfg)
    etf_w = etf_w[etf_w.index <= pd.Timestamp(manifest["end"])]
    factors, factor_audit, stock_prices, _ = build_stock_factors(snapshot, etf_w.index)
    factors.to_parquet(folder / "factors.parquet", index=False)
    factor_audit.to_csv(folder / "factor_coverage.csv", index=False)
    # Recompute with all future input rows removed to test point-in-time behavior.
    prefix = {}
    for cut in ("2019-06-28", "2022-06-30"):
        before = {}
        for key, d in snapshot.items():
            col = "trade_date" if "trade_date" in d else "ann_date" if key == "financials" else None
            before[key] = d[pd.to_datetime(d[col], errors="coerce") <= pd.Timestamp(cut)] if col else d
        pf, _, _, _ = build_stock_factors(before, etf_w.index[etf_w.index <= pd.Timestamp(cut)])
        expected = factors[factors.date <= pd.Timestamp(cut)].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, pf.reset_index(drop=True))
        prefix[cut] = True
    write_json(folder / "prefix_checks.json", prefix)
    baseline = simulate_composite(etf_data, snapshot, etf_w, cfg)
    old_baseline = simulate(etf_data, etf_w, cfg, start="2016-01-01")
    delta = float((baseline.equity.equity - old_baseline.equity.equity).abs().max())
    parity = dict(
        verified=delta < 0.01,
        max_abs_difference=delta,
        sessions=len(baseline.equity),
        old_trades=len(old_baseline.trades),
        new_trades=len(baseline.trades),
    )
    write_json(folder / "etf_ledger_parity.json", parity)
    if not parity["verified"]:
        raise AssertionError(f"ETF accounting regression: {delta:.4f}")
    runs = {"etf_core": (baseline, etf_w)}
    for family in FAMILIES:
        for fraction in (0.25, 0.4):
            name = f"{family}_{int(fraction * 100)}"
            w, picks, regime = composite_targets(factors, etf_w, stock_prices, etf_data, family, fraction)
            r = simulate_composite(etf_data, snapshot, w, cfg)
            runs[name] = (r, w)
            dest = folder / name
            dest.mkdir()
            picks.to_csv(dest / "selection.csv", index=False)
            print(name, json.dumps(performance(r.equity.equity), ensure_ascii=False), flush=True)
    for fraction in (0.25, 0.4):
        reference_w = runs[f"quality_value_{int(fraction * 100)}"][1]
        budget = reference_w.drop(columns=etf_w.columns).sum(axis=1)
        cw = etf_w.mul(1 - budget, axis=0)
        cw["510300.SH"] += budget
        runs[f"equity_overlay_{int(fraction * 100)}"] = (simulate_composite(etf_data, snapshot, cw, cfg), cw)
    rows, details, curves, crash_rows = [], {}, {}, []
    for name, (r, w) in runs.items():
        dest = folder / name
        dest.mkdir(exist_ok=True)
        for key, frame in dict(
            equity=r.equity, trades=r.trades, positions=r.positions, events=r.events, targets=w
        ).items():
            frame.to_parquet(dest / f"{key}.parquet")
        yearly(r.equity.equity, 0.02).to_csv(dest / "yearly.csv", index=False)
        stats = {
            "full": performance(r.equity.equity),
            "development": window_performance(r.equity.equity, "2016-01-01", "2020-12-31", 0.02),
            "validation": window_performance(r.equity.equity, "2021-01-01", "2022-12-31", 0.02),
            "retrospective_2023": window_performance(r.equity.equity, "2023-01-01", None, 0.02),
        }
        for period, metric in stats.items():
            rows.append(dict(candidate=name, period=period, **metric))
        blocks = r.events[r.events.type == "qualification_block"] if not r.events.empty else pd.DataFrame()
        replay = independent_replay(r, etf_data, snapshot, cfg)
        if not replay["verified"]:
            raise AssertionError("Independent ledger replay mismatch")
        crash = simulate_composite(etf_data, snapshot, w, cfg, start="2015-01-01", end="2015-12-31")
        crash_stats = performance(crash.equity.equity)
        crash_rows.append(dict(candidate=name, **crash_stats))
        crash.equity.to_parquet(dest / "stress_2015_equity.parquet")
        details[name] = dict(
            metrics=stats,
            stress_2015=crash_stats,
            stress_2015_blocks=int(crash.events.type.eq("qualification_block").sum())
            if not crash.events.empty
            else 0,
            trades=len(r.trades),
            stock_trades=int(r.trades.kind.eq("stock").sum()),
            qualification_blocks=len(blocks),
            replay=replay,
            average_stock_exposure=float(r.equity.stock_exposure.mean()),
            costs={
                key: float(r.trades[key].sum())
                for key in ("commission", "stamp_tax", "transfer_fee", "slippage_cost")
            },
        )
        curves[name] = r.equity.equity
    comparison = pd.DataFrame(curves)
    comparison.to_parquet(folder / "comparison.parquet")
    pd.DataFrame(rows).to_csv(folder / "candidates.csv", index=False)
    pd.DataFrame(crash_rows).to_csv(folder / "stress_2015.csv", index=False)
    base_val = details["etf_core"]["metrics"]["validation"]
    qualified = []
    for name, d in details.items():
        if name in {"etf_core", "equity_overlay_25", "equity_overlay_40"}:
            continue
        val = d["metrics"]["validation"]
        annual = pd.read_csv(folder / name / "yearly.csv").set_index("year")
        gates = dict(
            validation_cagr=val["cagr"] >= base_val["cagr"],
            validation_sharpe=val["sharpe"] >= base_val["sharpe"] + 0.05,
            drawdown=d["metrics"]["full"]["max_drawdown"] >= -0.15,
            crash_2015=d["stress_2015"]["max_drawdown"] >= -0.15,
            bear_years=all(annual.at[y, "total_return"] >= -0.08 for y in (2018, 2022)),
            execution=d["qualification_blocks"] == 0
            and d["stress_2015_blocks"] == 0
            and d["replay"]["verified"],
        )
        d["gates"] = gates
        if all(gates.values()):
            qualified.append(name)
    # Decision uses validation metrics; full-period bear/DD constraints are a
    # deployment safety veto, never evidence that the comparison was unseen.
    preferred = (
        max(qualified, key=lambda name: details[name]["metrics"]["validation"]["sharpe"])
        if qualified
        else "etf_core"
    )
    # Always review the best VALIDATION stock model, including when no model passes.
    stock_names = [x for x in details if x not in {"etf_core", "equity_overlay_25", "equity_overlay_40"}]
    reviewed = max(stock_names, key=lambda name: details[name]["metrics"]["validation"]["sharpe"])
    r, w = runs[reviewed]
    sensitivity = []
    for multiplier in (2, 3):
        extra = simulate_composite(etf_data, snapshot, w, cfg, cost_multiplier=multiplier)
        sensitivity.append(dict(variant=f"cost_x{multiplier}", **performance(extra.equity.equity)))
    family = next(f for f in FAMILIES if reviewed.startswith(f))
    fraction = 0.25 if reviewed.endswith("25") else 0.4
    for count in (6, 10):
        nw, _, _ = composite_targets(factors, etf_w, stock_prices, etf_data, family, fraction, top_n=count)
        extra = simulate_composite(etf_data, snapshot, nw, cfg)
        sensitivity.append(dict(variant=f"holdings_{count}", **performance(extra.equity.equity)))
    pd.DataFrame(sensitivity).to_csv(folder / "sensitivity.csv", index=False)
    fresh = []
    for year in range(2018, 2027):
        extra = simulate_composite(etf_data, snapshot, w, cfg, start=f"{year}-01-01", end=f"{year}-12-31")
        fresh.append(dict(year=year, **performance(extra.equity.equity)))
    pd.DataFrame(fresh).to_csv(folder / "fresh_annual.csv", index=False)
    weights = w.iloc[-1]
    names = {
        **snapshot["metadata"].set_index("ts_code").name.to_dict(),
        **{a["symbol"]: a["name"] for a in cfg["assets"]},
    }
    holdings = pd.DataFrame(
        [
            dict(
                symbol=s,
                name=names.get(s, s),
                kind="fund" if s in etf_data else "stock",
                target_weight=float(v),
            )
            for s, v in weights[weights > 0].items()
        ]
    )
    holdings.to_csv(folder / "research_allocations.csv", index=False)
    summary = dict(
        preferred=preferred,
        reviewed_stock_model=reviewed,
        qualified=qualified,
        models=details,
        paired_sharpe_interval=paired_sharpe_interval(r.equity.equity, baseline.equity.equity),
        etf_ledger_parity=parity,
        prefix_checks=prefix,
        data_quality=quality,
        as_of=str(weights.name.date()),
        research_only=True,
        active_unchanged=active_hash == hashlib.sha256(active_path.read_bytes()).hexdigest(),
        risk_note="15%-20% return is an aspiration, not a gate to fit. All history is retrospective, vendor financial vintages not certified; no brokerage action.",
    )
    if not summary["active_unchanged"]:
        raise AssertionError("Active financial strategy changed during research")
    write_json(folder / "summary.json", summary)
    write_json(ROOT / "outputs/latest_composite.json", dict(path=str(folder)))
    print(
        json.dumps(
            dict(path=str(folder), preferred=preferred, reviewed=reviewed, qualified=qualified),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return folder
