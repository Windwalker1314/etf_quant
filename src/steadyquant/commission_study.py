"""Frozen capital/commission study, using the active ETF rules without activation."""

from __future__ import annotations

import copy
import hashlib
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .config import ROOT, load_active_config, write_json
from .data import Cache
from .execution import fee
from .framework import engine_comparison
from .metrics import performance, window_performance, yearly

CAPITALS = (100000, 200000, 1000000)
NAMES = {
    "base10": "原10只 · 万1.5最低5元",
    "min5000": "10只 · 最低交易5000元",
    "min10000": "10只 · 最低交易10000元",
    "band5": "10只 · 偏离5个百分点",
    "core7": "7只 · A股合并沪深300",
    "core7_min5000": "7只 · 最低交易5000元",
}


def configurations(base):
    actual = copy.deepcopy(base)
    actual.update(commission_bps=1.5, minimum_commission=5.0)
    configs = {name: copy.deepcopy(actual) for name in NAMES}
    configs["min5000"]["minimum_trade_notional"] = 5000
    configs["min10000"]["minimum_trade_notional"] = 10000
    configs["band5"]["rebalance_band"] = 0.05
    for name in ("core7", "core7_min5000"):
        cfg = configs[name]
        total_cn = sum(a["weight"] for a in cfg["assets"] if a["bucket"] == "cn_equity")
        cfg["assets"] = [a for a in cfg["assets"] if a["bucket"] != "cn_equity" or a["symbol"] == "510300.SH"]
        next(a for a in cfg["assets"] if a["symbol"] == "510300.SH")["weight"] = total_cn
    configs["core7_min5000"]["minimum_trade_notional"] = 5000
    return configs


def audit_fills(result, data, cfg, multiplier=1.0):
    """Reconstruct cash and units independently from fills and observed factor ratios."""
    dates = result.equity.index
    qty = {s: 0.0 for s in data}
    last = {s: 0.0 for s in data}
    adj = {s: None for s in data}
    frames = {s: d.set_index("date") for s, d in data.items()}
    grouped = dict(tuple(result.trades.groupby("date"))) if len(result.trades) else {}
    cash, previous, differences = float(cfg["initial_cash"]), dates[0], []
    for date in dates:
        cash *= (1 + cfg["cash_rate"]) ** ((date - previous).days / 365.25)
        previous = date
        for s, d in frames.items():
            if date not in d.index:
                continue
            bar = d.loc[date]
            if adj[s] is not None:
                ratio = bar.adj_factor / adj[s]
                if abs(ratio - 1) > 1e-8:
                    qty[s] *= ratio
            adj[s] = bar.adj_factor
            last[s] = bar.close
        for t in grouped.get(date, pd.DataFrame()).itertuples():
            assert dates.get_loc(t.date) == dates.get_loc(t.signal_date) + 1
            bar = frames[t.symbol].loc[date]
            assert bar.volume > 0 and bar.high != bar.low
            sign = 1 if t.side == "BUY" else -1
            expected_price = bar.open * (1 + sign * cfg["slippage_bps"] / 10000 * multiplier)
            assert abs(t.price - expected_price) < 1e-8
            assert abs(t.commission - fee(t.notional, cfg, multiplier)) < 1e-8
            if sign > 0:
                assert abs(t.quantity / cfg["lot_size"] - round(t.quantity / cfg["lot_size"])) < 1e-8
            cash -= sign * t.quantity * t.price + t.commission
            qty[t.symbol] += sign * t.quantity
            assert cash >= -1e-6 and min(qty.values()) >= -1e-6
        nav = cash + sum(qty[s] * last[s] for s in qty)
        differences.append(abs(nav - result.equity.loc[date, "equity"]))
        assert abs(cash - result.equity.loc[date, "cash"]) < 1e-5
    maximum = float(max(differences))
    assert maximum < 1e-5
    return dict(
        verified=True, sessions=len(dates), fills=len(result.trades), max_equity_difference_cny=maximum
    )


def metrics(result, cfg, start=None, end=None, cost_multiplier=1.0):
    nav = result.equity.equity
    t = result.trades
    if start:
        t = t[t.date >= pd.Timestamp(start)]
    if end:
        t = t[t.date <= pd.Timestamp(end)]
    p = (
        window_performance(nav, start, end, cfg["risk_free_rate"])
        if start
        else performance(nav, cfg["risk_free_rate"])
    )
    floor = (cfg["minimum_commission"] > 0) & (
        t.commission <= cfg["minimum_commission"] * cost_multiplier + 1e-8
    )
    extra = np.maximum(t.commission - t.notional * cfg["commission_bps"] / 10000 * cost_multiplier, 0).sum()
    selected = result.equity.loc[start:end] if start or end else result.equity
    years = max((selected.index[-1] - selected.index[0]).days / 365.25, 1 / 365.25)
    p.update(
        trades=len(t),
        annual_trades=len(t) / years,
        commission_cny=float(t.commission.sum()),
        slippage_cny=float(t.slippage_cost.sum()),
        floor_surcharge_same_fills_cny=float(extra),
        minimum_fee_fraction=float(floor.mean()) if len(t) else 0,
        total_cost_pct_initial=float((t.commission.sum() + t.slippage_cost.sum()) / cfg["initial_cash"]),
        cost_bps_average_nav_per_year=float(
            (t.commission.sum() + t.slippage_cost.sum()) / selected.equity.mean() / years * 10000
        ),
        annual_turnover=float(t.notional.sum() / selected.equity.mean() / years),
        median_ticket_cny=float(t.notional.median()) if len(t) else 0,
        average_cash_weight=float((selected.cash / selected.equity).mean()),
        tiny_ticket_fraction=float((t.notional < 5000).mean()) if len(t) else 0,
    )
    return p


def save_result(out, result, data, cfg, multiplier=1.0):
    out.mkdir(parents=True, exist_ok=True)
    for attr in ("equity", "trades", "positions", "events"):
        getattr(result, attr).to_parquet(out / f"{attr}.parquet")
    check = audit_fills(result, data, cfg, multiplier)
    write_json(out / "replay.json", check)
    write_json(out / "config.json", cfg)
    return check


def acceptable(candidate, baseline):
    return bool(
        candidate["cagr"] >= baseline["cagr"] - 0.002
        and candidate["sharpe"] >= baseline["sharpe"] - 0.03
        and candidate["max_drawdown"] >= baseline["max_drawdown"] - 0.01
    )


def run_commission_study():
    base = load_active_config()
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    raw = cache.load(base)
    end = min(d.date.max() for d in raw.values())
    # Freeze the latest complete cached session, never request intraday data.
    data = {s: d[d.date <= end].copy() for s, d in raw.items()}
    configs = configurations(base)
    out = ROOT / "outputs/commission" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    protocol = dict(
        frozen_at=datetime.now().isoformat(),
        end=str(end.date()),
        capitals=CAPITALS,
        candidates=configs,
        commission="max(notional * 0.00015, CNY5) per simulated fill",
        slippage_bps=base["slippage_bps"],
        start="2015-01-05",
        fresh_start="2023-01-01",
        selection="2015-2022 only: common candidate across all three capitals; CAGR loss <=0.2pp, Sharpe loss<=0.03, drawdown worsening<=1pp; mean trade-count reduction>=20%. Choose greatest mean trade-count reduction, tie by declaration order. If none eligible keep base10.",
        validation="Do not reselect after2023: candidate must retain same tolerances for all capitals in both carried and fresh-capital2023+ runs; full-history drawdown>=-15%. Otherwise keep base10. Historical windows already observed in earlier research; not virgin holdout.",
        sensitivity="base10 and selected candidate (or min5000 if no candidate): all capitals, double fees+slippage, and slippage20bps. Neighbor thresholds are fixed main candidates, no post-result retuning.",
        cost_attribution="Baseline with old3bps, actual1.5bps/no5yuan floor, and zero commission (slippage retained), all capitals full history; path-dependent reruns. Report floor surcharge on identical actual fills separately.",
        execution="close fixed-share plans -> next open, 100 units, 1% volume limit, no leverage. Minimum ticket measured at signal close after liquidity/cash rounding; complete exits/emergency reductions exempt. Risk rules unchanged.",
        universe="Existing10 funds only; core7 removes three A-share satellites and transfers their prior budget to510300; retains both US ETFs.160723 not included.",
        limitations="Historical selected universe; pre-inception funds missing; ETF adjustment-factor reinvestment approximation (not actual dividend receivable dates), no historical QDII premium gate. Fees assumed constant history, actual broker may aggregate fills differently.",
        data_sha256=cache.snapshot_digest,
        protected_sha256=original_hashes,
    )
    write_json(out / "protocol.json", protocol)
    (out / "source").mkdir()
    sources = [
        "commission_study.py",
        "backtest.py",
        "execution.py",
        "strategy.py",
        "adaptive.py",
        "framework.py",
        "metrics.py",
    ]
    for name in sources:
        shutil.copy(ROOT / "src/steadyquant" / name, out / "source" / name)
    write_json(
        out / "implementation.json",
        {n: hashlib.sha256((out / "source" / n).read_bytes()).hexdigest() for n in sources},
    )
    (out / "data").mkdir()
    for s, d in data.items():
        d.to_parquet(out / "data" / f"{s}.parquet", index=False)
    coverage = [
        dict(
            symbol=s,
            first=str(d.date.min().date()),
            last=str(d.date.max().date()),
            bars=len(d),
            earliest_253rd_bar=str(d.iloc[252].date.date()),
        )
        for s, d in data.items()
    ]
    write_json(out / "coverage.json", coverage)
    signals = {}
    for name in ("base10", "core7"):
        print(f"Build frozen targets: {name}", flush=True)
        cfg = configs[name]
        subset = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
        signals[name], _ = adaptive_weights(subset, cfg)
        signals[name].to_parquet(out / f"{name}_targets.parquet")
    # Signal recomputation with a genuinely truncated input verifies causality.
    cut = pd.Timestamp("2022-12-30")
    prefix_checks = {}
    for name in ("base10", "core7"):
        subset = {
            a["symbol"]: data[a["symbol"]][data[a["symbol"]].date <= cut] for a in configs[name]["assets"]
        }
        prefix, _ = adaptive_weights(subset, configs[name])
        pd.testing.assert_frame_equal(prefix, signals[name].loc[prefix.index])
        prefix_checks[name] = dict(verified=True, rows=len(prefix), cut=str(cut.date()))
    write_json(out / "prefix_checks.json", prefix_checks)
    results, rows, year_rows, stress_rows = {}, [], [], []
    for capital in CAPITALS:
        for name, template in configs.items():
            cfg = {**template, "initial_cash": capital}
            subset = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
            w = signals["core7" if name.startswith("core7") else "base10"]
            for mode, start in (("carried", "2015-01-05"), ("fresh2023", "2023-01-01")):
                print(f"Main {capital} / {name} / {mode}", flush=True)
                result = simulate(subset, w, cfg, start=start, end=str(end.date()))
                save_result(out / str(capital) / name / mode, result, subset, cfg)
                results[capital, name, mode] = result
                periods = (
                    (
                        ("full", None, None),
                        ("selection", "2015-01-01", "2022-12-31"),
                        ("review2023", "2023-01-01", None),
                    )
                    if mode == "carried"
                    else (("fresh2023", None, None),)
                )
                for period, lo, hi in periods:
                    rows.append(
                        dict(
                            capital=capital,
                            model=name,
                            name=NAMES[name],
                            period=period,
                            **metrics(result, cfg, lo, hi),
                        )
                    )
                if mode == "carried":
                    year_rows.extend(
                        dict(capital=capital, model=name, **r)
                        for r in yearly(result.equity.equity, cfg["risk_free_rate"]).to_dict("records")
                    )
                    for tag, lo, hi in (
                        ("2015", "2015-06-01", "2015-12-31"),
                        ("2018", "2018-01-01", "2018-12-31"),
                        ("2020Q1", "2020-01-01", "2020-03-31"),
                        ("2022", "2022-01-01", "2022-12-31"),
                        ("2025plus", "2025-01-01", None),
                    ):
                        stress_rows.append(
                            dict(capital=capital, model=name, window=tag, **metrics(result, cfg, lo, hi))
                        )
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(year_rows).to_csv(out / "yearly.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(out / "stress.csv", index=False)

    def row(capital, model, period):
        return table[(table.capital == capital) & (table.model == model) & (table.period == period)].iloc[0]

    selection = []
    for name in list(NAMES)[1:]:
        tolerance = all(
            acceptable(row(c, name, "selection"), row(c, "base10", "selection")) for c in CAPITALS
        )
        reduction = float(
            np.mean(
                [
                    1 - row(c, name, "selection").trades / row(c, "base10", "selection").trades
                    for c in CAPITALS
                ]
            )
        )
        selection.append(
            dict(
                model=name,
                train_tolerances=tolerance,
                mean_trade_reduction=reduction,
                eligible=tolerance and reduction >= 0.20,
            )
        )
    eligible = [s for s in selection if s["eligible"]]
    chosen = max(eligible, key=lambda s: s["mean_trade_reduction"])["model"] if eligible else "base10"
    validation = {
        str(c): {p: acceptable(row(c, chosen, p), row(c, "base10", p)) for p in ("review2023", "fresh2023")}
        for c in CAPITALS
    }
    pass_drawdown = all(row(c, chosen, "full").max_drawdown >= -0.15 for c in CAPITALS)
    accepted = chosen != "base10" and pass_drawdown and all(all(v.values()) for v in validation.values())
    recommendation = chosen if accepted else "base10"
    write_json(
        out / "selection.json",
        dict(
            selection=selection,
            chosen_before_review=chosen,
            validation=validation,
            pass_drawdown=pass_drawdown,
            accepted=accepted,
            recommendation=recommendation,
        ),
    )
    print(f"Frozen selection: {chosen}; accepted={accepted}; recommended={recommendation}", flush=True)

    counterfactual = []
    for c in CAPITALS:
        for label, delta in (
            ("old3bps", dict(commission_bps=3.0)),
            ("no_minimum", dict(minimum_commission=0.0)),
            ("zero_commission", dict(commission_bps=0.0, minimum_commission=0.0)),
        ):
            cfg = {**configs["base10"], "initial_cash": c, **delta}
            print(f"Cost attribution {c} / {label}", flush=True)
            r = simulate(data, signals["base10"], cfg, end=str(end.date()))
            save_result(out / str(c) / "counterfactual" / label, r, data, cfg)
            counterfactual.append(dict(capital=c, variant=label, **metrics(r, cfg)))
    pd.DataFrame(counterfactual).to_csv(out / "cost_counterfactual.csv", index=False)
    sensitivity, native_checks, intervals = [], {}, {}
    examined = list(dict.fromkeys(["base10", chosen if chosen != "base10" else "min5000"]))
    for c in CAPITALS:
        for name in examined:
            cfg = {**configs[name], "initial_cash": c}
            subset = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
            w = signals["core7" if name.startswith("core7") else "base10"]
            for label, delta, mul in (
                ("double_costs", {}, 2.0),
                ("slippage20bps", dict(slippage_bps=20), 1.0),
            ):
                print(f"Sensitivity {c} / {name} / {label}", flush=True)
                scfg = {**cfg, **delta}
                r = simulate(subset, w, scfg, cost_multiplier=mul, end=str(end.date()))
                save_result(out / str(c) / name / label, r, subset, scfg, mul)
                sensitivity.append(
                    dict(capital=c, model=name, variant=label, **metrics(r, scfg, cost_multiplier=mul))
                )
            checked, compare = engine_comparison(subset, w, cfg)
            native_checks[f"{c}_{name}"] = checked
            compare.to_csv(out / str(c) / name / "native.csv")
            if name != "base10":
                intervals[str(c)] = paired_sharpe_interval(
                    results[c, name, "carried"].equity.equity.loc["2023":],
                    results[c, "base10", "carried"].equity.equity.loc["2023":],
                )
    pd.DataFrame(sensitivity).to_csv(out / "sensitivity.csv", index=False)
    write_json(out / "native_checks.json", native_checks)
    write_json(out / "paired_intervals.json", intervals)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == original_hashes[str(p)] for p in protected)
    for c in CAPITALS:
        write_json(out / f"recommended_{c}.json", {**configs[recommendation], "initial_cash": c})
    summary = dict(
        end=str(end.date()),
        recommendation=recommendation,
        candidate=chosen,
        accepted=accepted,
        active_changed=False,
        protected_files_verified=True,
        main_runs=36,
        attribution_runs=9,
        sensitivity_runs=len(sensitivity),
        native_checks=native_checks,
        selected_configs="recommended_*.json",
        limitations=protocol["limitations"],
        historical_validation="All periods retrospective; no fresh forward evidence",
    )
    write_json(out / "summary.json", summary)
    from .commission_report import render_report

    render_report(out)
    write_json(ROOT / "outputs/latest_commission.json", dict(path=str(out), recommendation=recommendation))
    print(f"Commission study complete: {out}", flush=True)
    return out
