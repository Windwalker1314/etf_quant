"""Compare monthly, weekly and daily signals on the same broad ETF data."""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from steadyquant.adaptive_audit import paired_sharpe_interval
from steadyquant.backtest import simulate
from steadyquant.config import ROOT, load_active_config, write_json
from steadyquant.data import TZ
from steadyquant.etf_broad_study import broad_targets, first_session_mask, load_data, research_config
from steadyquant.metrics import window_performance
from steadyquant.strategy import target_weights

FAMILIES = ("base", "top_two", "cn_satellite", "low_vol")
FREQUENCIES = ("monthly", "weekly", "daily")
END = "2026-09-24"


def run():
    base = load_active_config()
    cfg = research_config(base)
    data = load_data(cfg)
    original = {a["symbol"]: data[a["symbol"]] for a in base["assets"]}
    core, _ = target_weights(original, {**base, "initial_cash": 200000})
    protected = (ROOT / "configs/active.yaml", ROOT / "data/portfolio.json")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    out = ROOT / "outputs/etf_broad_cadence" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    write_json(out / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), data_end=END, initial_cash=200000,
        universe={a["symbol"]: a["bucket"] for a in cfg["assets"]},
        families=FAMILIES, frequencies=FREQUENCIES,
        schedule="Signal at the close of the first observed trading session in each month/week, or every close; fill at next session open. Weekly W-SUN; holiday shifts to the next observed session.",
        controls="Same 28-ETF data, target rule, 3pp absolute drift band, 10% volatility ceiling, 100-unit lots and 1% volume cap. Only signal/rebalance cadence changes within each family.",
        costs="1.5bp commission, min5CNY; 5bp slippage. Normal and doubled cost; independent 2023-01 fresh start.",
        screen="Retrospective gate: 2019-2022 Sharpe >= monthly base +.05, CAGR>=monthly base, max drawdown no worse by2pp; then inspect 2023+ without reselection. All periods already seen, not virgin out-of-sample.",
        limitations="Current hand-picked ETF survivors; pre-inception assets never backfilled. QDII historical premium absent; adjustment-factor reinvestment approximates dividend timing. Daily next-open model may miss overnight premium/spread changes.",
        protected_sha256=hashes,
    ))
    rows, curves = [], {}
    dates = core.index
    for family in FAMILIES:
        for frequency in FREQUENCIES:
            print(f"{family} / {frequency}", flush=True)
            weights = broad_targets(data, core, cfg, family, frequency)
            mask = pd.DataFrame(
                np.broadcast_to(first_session_mask(dates, frequency)[:, None], weights.shape).copy(),
                index=dates, columns=weights.columns,
            )
            if family != "base":
                cut = pd.Timestamp("2022-12-30")
                prefix = {s: d[d.date <= cut] for s, d in data.items()}
                earlier = broad_targets(prefix, core.loc[:cut], cfg, family, frequency)
                pd.testing.assert_frame_equal(earlier, weights.loc[:cut], check_freq=False)
            weights.to_parquet(out / f"targets_{family}_{frequency}.parquet")
            result = simulate(data, weights, cfg, start="2019-01-01", end=END, rebalance_mask=mask)
            curves[family, frequency] = result.equity.equity
            result.equity.to_parquet(out / f"equity_{family}_{frequency}.parquet")
            for period, lo, hi in (("train", "2019-01-01", "2022-12-31"),
                                   ("review", "2023-01-01", None), ("all", "2019-01-01", None)):
                rows.append(dict(family=family, frequency=frequency, mode="carried2019", period=period,
                                 trades=len(result.trades), commission=float(result.trades.commission.sum()),
                                 slippage=float(result.trades.slippage_cost.sum()),
                                 **window_performance(result.equity.equity, lo, hi, cfg["risk_free_rate"])))
            doubled = simulate(data, weights, cfg, start="2019-01-01", end=END,
                               cost_multiplier=2, rebalance_mask=mask)
            rows.append(dict(family=family, frequency=frequency, mode="double_cost", period="all",
                             trades=len(doubled.trades), commission=float(doubled.trades.commission.sum()),
                             slippage=float(doubled.trades.slippage_cost.sum()),
                             **window_performance(doubled.equity.equity, "2019-01-01", None, cfg["risk_free_rate"])))
            fresh = simulate(data, weights, cfg, start="2023-01-01", end=END, rebalance_mask=mask)
            rows.append(dict(family=family, frequency=frequency, mode="fresh2023", period="review",
                             trades=len(fresh.trades), commission=float(fresh.trades.commission.sum()),
                             slippage=float(fresh.trades.slippage_cost.sum()),
                             **window_performance(fresh.equity.equity, "2023-01-01", None, cfg["risk_free_rate"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison.csv", index=False)
    # The shared monthly baseline must reproduce the preceding universe study exactly.
    prior = pd.read_parquet(
        ROOT / "outputs/etf_broad_study/20260927-014857/equity_base.parquet"
    ).equity
    pd.testing.assert_series_equal(prior, curves["base", "monthly"])
    train = table[(table["mode"] == "carried2019") & (table.period == "train")]
    reference = train[(train.family == "base") & (train.frequency == "monthly")].iloc[0]
    qualified = train[(train.sharpe >= reference.sharpe + .05)
                      & (train.cagr >= reference.cagr)
                      & (train.max_drawdown >= reference.max_drawdown - .02)]
    qualified = qualified[~((qualified.family == "base") & (qualified.frequency == "monthly"))]
    selected = (qualified.sort_values("sharpe", ascending=False).iloc[0][["family", "frequency"]].to_list()
                if len(qualified) else ["base", "monthly"])
    intervals = {}
    for family in FAMILIES:
        for frequency in FREQUENCIES:
            if (family, frequency) != ("base", "monthly"):
                intervals[f"{family}_{frequency}"] = paired_sharpe_interval(
                    curves[family, frequency], curves["base", "monthly"]
                )
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in protected)
    write_json(out / "summary.json", dict(selected_on_train=selected, prefix_checks=True,
        monthly_baseline_parity=True, active_unchanged=True, paired_intervals=intervals))
    figure = go.Figure()
    for family, frequency in (("base", "monthly"), ("base", "weekly"), ("base", "daily"),
                              ("low_vol", "monthly"), ("low_vol", "weekly"), ("low_vol", "daily")):
        nav = curves[family, frequency]
        figure.add_trace(go.Scatter(x=nav.index, y=nav / nav.iloc[0], name=f"{family}/{frequency}"))
    figure.update_layout(title="同一 ETF 池：调仓频率与策略对比", template="plotly_dark",
                         xaxis_title="日期", yaxis_title="净值倍数", hovermode="x unified")
    figure.write_html(out / "comparison.html", include_plotlyjs=True)
    print(table[["family", "frequency", "mode", "period", "cagr", "sharpe",
                 "max_drawdown", "trades"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out


if __name__ == "__main__":
    run()
