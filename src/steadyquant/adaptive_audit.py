"""Post-hoc mechanism/implementation audits; never re-rank the frozen main batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .adaptive import adaptive_weights, capped_proportions
from .backtest import simulate
from .config import ROOT, write_json
from .data import Cache
from .framework import engine_comparison
from .macro import load_macro
from .metrics import performance, window_performance
from .strategy import target_weights


def paired_sharpe_interval(candidate, baseline):
    aligned = pd.concat([candidate.rename("candidate"), baseline.rename("baseline")], axis=1).dropna()
    r = aligned.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(8192)
    differences = []
    for _ in range(500):
        starts = rng.integers(0, len(r), size=int(np.ceil(len(r) / 20)))
        sample = r[((starts[:, None] + np.arange(20)) % len(r)).ravel()[: len(r)]]
        sharpes = (sample.mean(axis=0) - (1.02 ** (1 / 252) - 1)) / sample.std(axis=0, ddof=1) * np.sqrt(252)
        differences.append(sharpes[0] - sharpes[1])
    lo, mid, hi = np.quantile(differences, [0.025, 0.5, 0.975])
    return dict(
        lower_95=float(lo),
        median=float(mid),
        upper_95=float(hi),
        note="Paired20-session circular blocks,500 resamples; historical uncertainty, not multiplicity-adjusted proof.",
    )


def audit_adaptive(folder: Path | None = None):
    folder = folder or Path(json.loads((ROOT / "outputs/latest_adaptive.json").read_text())["path"])
    protocol = json.loads((folder / "protocol.json").read_text())
    configs = protocol["candidates"]
    cache = Cache()
    data = cache.load(configs["equal_risk"])
    if cache.snapshot_digest != protocol["data_sha256"]:
        raise ValueError("Adaptive audit price snapshot changed")
    macro = load_macro()
    out = folder / "diagnostics"
    out.mkdir(exist_ok=True)
    variants = {
        "first_session": dict(rebalance_frequency="monthly_first_session"),
        "relative_band": dict(relative_rebalance_band=0.2),
        "risk_exit": dict(risk_exit_ratio=0.5),
        "combined_execution": dict(
            rebalance_frequency="monthly_first_session", relative_rebalance_band=0.2, risk_exit_ratio=0.5
        ),
    }
    write_json(
        out / "protocol.json",
        dict(
            stage="Post-hoc diagnostic, fixed primary selection remains unchanged. No diagnostic winner automatically promoted.",
            mechanisms="Compare conditional invested fixed budget, and pre2023-mean equal-risk target held during2023+ only; distinguish allocation/exposure changes from dynamic timing.",
            execution_variants=variants,
            risk_sensitivity="For equal-risk and blended-risk: cost2/3, lookback84/168, volcap8/12, covariance shrinkage.1/.5; no reselection.",
            uncertainty="Paired block bootstrap against simple monthly core. All histories already observed.",
        ),
    )
    rows, checks = [], {}
    equities = pd.read_parquet(folder / "all_equities.parquet")
    for model in ("fixed_core_monthly", "equal_risk", "blended_risk"):
        print(f"Mechanism/implementation audit: {model}", flush=True)
        cfg = configs[model]
        subset = {a["symbol"]: data[a["symbol"]] for a in cfg["assets"]}
        w = pd.read_parquet(folder / model / "targets.parquet")
        native, comparison = engine_comparison(subset, w, cfg)
        checks[model] = native
        comparison.to_csv(out / f"{model}_native.csv")
        for name, delta in variants.items():
            adjusted = {**cfg, **delta}
            r = simulate(subset, w, adjusted)
            for period, lo, hi in [
                ("full", "2015-01-01", None),
                ("selection", "2015-01-01", "2022-12-31"),
                ("test", "2023-01-01", None),
            ]:
                rows.append(
                    dict(
                        model=model,
                        variant=name,
                        period=period,
                        trades=len(r.trades),
                        **window_performance(r.equity.equity, lo, hi, 0.02),
                    )
                )
            r.equity.to_parquet(out / f"{model}_{name}_equity.parquet")
            if name == "combined_execution":
                checked, nc = engine_comparison(subset, w, adjusted)
                checks[f"{model}_{name}"] = checked
                nc.to_csv(out / f"{model}_{name}_native.csv")
        if model == "fixed_core_monthly":
            continue
        sens = [
            ("cost_x2", {}, 2),
            ("cost_x3", {}, 3),
            ("risk_window_84", dict(risk_window=84), 1),
            ("risk_window_168", dict(risk_window=168), 1),
            ("vol_cap_8", dict(target_vol=0.08), 1),
            ("vol_cap_12", dict(target_vol=0.12), 1),
            ("shrink_10", dict(covariance_shrinkage=0.1), 1),
            ("shrink_50", dict(covariance_shrinkage=0.5), 1),
        ]
        for name, delta, mul in sens:
            nw, _ = adaptive_weights(subset, {**cfg, **delta}, macro) if delta else (w, None)
            r = simulate(subset, nw, cfg, cost_multiplier=mul)
            rows.append(
                dict(
                    model=model,
                    variant=name,
                    period="full",
                    trades=len(r.trades),
                    **performance(r.equity.equity, 0.02),
                )
            )
    pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)
    write_json(out / "native_checks.json", checks)
    intervals = {
        name: paired_sharpe_interval(equities[name], equities.fixed_core_monthly)
        for name in ("equal_risk", "blended_risk")
    }
    write_json(out / "paired_sharpe_intervals.json", intervals)
    # Keep fixed style budgets but allow unused/missing asset budgets to be invested
    # in the available assets, subject to30% ETF cap; no observed future weights.
    cfg = {**configs["broad_core_50"], "min_history": 253}
    base, _ = target_weights(data, cfg)
    cond = base.copy()
    for date, weights in base.iterrows():
        cond.loc[date] = capped_proportions(weights.to_numpy(), np.full(len(weights), 0.30), 0.99)
    control = simulate(data, cond, cfg)
    control.equity.to_parquet(out / "conditional_fixed_equity.parquet")
    # Static mean is learned through2022 and can only be backtested from2023.
    erc = pd.read_parquet(folder / "equal_risk/targets.parquet")
    mean = erc.loc["2015":"2022"].mean()
    static = pd.DataFrame(np.tile(mean.to_numpy(), (len(erc), 1)), index=erc.index, columns=erc.columns)
    static = static.where(base > 0, 0)
    hold = simulate(data, static, cfg, start="2023-01-01")
    hold.equity.to_parquet(out / "pre2023_mean_2023_forward.parquet")
    attribution = []
    for model in ("fixed_core_monthly", "equal_risk", "blended_risk"):
        nav = pd.read_parquet(folder / model / "equity.parquet").equity
        pos = pd.read_parquet(folder / model / "positions.parquet")
        trades = pd.read_parquet(folder / model / "trades.parquet")
        total = 0
        for a in configs[model]["assets"]:
            s = a["symbol"]
            t = trades[trades.symbol == s]
            contribution = (
                pos[s].iloc[-1] * nav.iloc[-1]
                + t.loc[t.side == "SELL", "notional"].sum()
                - t.loc[t.side == "BUY", "notional"].sum()
                - t.commission.sum()
            )
            total += contribution
            attribution.append(
                dict(model=model, symbol=s, bucket=a["bucket"], profit_cny=float(contribution))
            )
        if abs(total - (nav.iloc[-1] - configs[model]["initial_cash"])) > 1e-5:
            raise AssertionError("Asset P&L doesn't reconcile with total profit")
    pd.DataFrame(attribution).to_csv(out / "asset_profit.csv", index=False)
    write_json(
        out / "mechanisms.json",
        dict(
            conditional_fixed_full=performance(control.equity.equity, 0.02),
            conditional_fixed_test=window_performance(control.equity.equity, "2023-01-01", None, 0.02),
            pre2023_mean_weights=mean.to_dict(),
            pre2023_mean_from2023=performance(hold.equity.equity, 0.02),
            note="Static mean only executes from2023, trained using2015-2022 target averages. It is not valid for a2015 backtest.",
        ),
    )
    print(
        pd.DataFrame(rows)[["model", "variant", "period", "cagr", "sharpe", "max_drawdown"]].to_string(
            index=False
        ),
        flush=True,
    )
    return out
