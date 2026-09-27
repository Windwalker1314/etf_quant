"""Frozen, retrospective bond-cap research; never activates a strategy."""
from __future__ import annotations

import hashlib
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .adaptive import adaptive_weights
from .adaptive_audit import paired_sharpe_interval
from .backtest import simulate
from .commission_study import metrics, save_result
from .config import ROOT, load_active_config, write_json
from .data import TZ, Cache
from .factors import adjusted_frame
from .metrics import yearly
from .reports import figure_html, shell

POLICIES = {
    "base30": (0.30, None, "原版 · 上限30%"),
    "fixed20": (0.20, None, "固定上限20%"),
    "fixed10": (0.10, None, "固定上限10%"),
    "bull20": (0.30, 0.20, "强趋势20% / 其余30%"),
    "bull10": (0.30, 0.10, "强趋势10% / 其余30%"),
}


def bull_regime(data):
    close = adjusted_frame(data).pivot(index="date", columns="symbol", values="close").sort_index()
    leaders = close.reindex(columns=["510300.SH", "513500.SH", "159920.SZ"])
    strong = (leaders > leaders.rolling(252, min_periods=252).mean()) & (leaders > leaders.shift(126))
    return strong.sum(axis=1) >= 2


def caps_for(data, normal, bull):
    regime = bull_regime(data)
    return pd.Series(np.where(regime & (bull is not None), bull if bull is not None else normal, normal),
                     index=regime.index, dtype=float)


def gap_freeze_mask(data, targets, cfg, start):
    """Sensitivity: freeze all new plans while the bond's 20-bar eligibility input is missing.

    This is not a repaired-price counterfactual. Existing positions are marked using
    the ledger's usual last observed price, and previously submitted orders still expire normally.
    """
    dates = targets.loc[start:].index
    bond = data["511010.SH"].set_index("date").reindex(targets.index)
    healthy = bond.close.notna() & bond.amount.rolling(20, min_periods=20).mean().notna()
    mask = pd.DataFrame(False, index=dates, columns=targets.columns)
    initialized = False
    for i, date in enumerate(dates):
        due = not initialized or (i > 0 and date.to_period("M") != dates[i - 1].to_period("M"))
        if due and healthy.loc[date]:
            mask.loc[date] = True
            initialized = initialized or bool((targets.loc[date] > 0).any())
    return mask


def run():
    cfg = {**load_active_config(), "initial_cash": 200000}
    assert cfg["model"] == "adaptive" and cfg["rebalance_frequency"] == "monthly_first_session"
    assert cfg.get("risk_exit_ratio") is None
    protected = [ROOT / "configs/active.yaml", ROOT / "data/portfolio.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    cache = Cache()
    data = cache.load(cfg)
    ends = {d.date.max() for d in data.values()}
    assert len(ends) == 1
    end = str(next(iter(ends)).date())
    out = ROOT / "outputs/bond_cap" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    write_json(out / "protocol.json", dict(
        frozen_at=datetime.now(TZ).isoformat(), end=end, config=cfg, candidates=POLICIES,
        regimes="At least 2 of 510300/513500/159920 above their 252-observed-session average and above their close126 sessions ago. No hindsight bull-market labels.",
        distribution="Replace only bond group cap BEFORE capped risk allocation. Freed risk budget is redistributed through existing scores/caps; equity60%, perETF30%, volatility10% remain. Some excess can remain cash.",
        fresh_starts=["2015-01-05", "2019-01-01", "2023-01-01"],
        costs="Same 1.5bps minimum5yuan, slippage5bps; rerun all candidates at double fees and slippage for2019/2023.",
        selection="Rank on2019-2022 carried-window net Sharpe only; candidate must improve by>=0.03 to be considered. Then review2023 carried and fresh without reselection. All periods previously observed; retrospective not virgin holdout; no activation.",
        data_gap="Preserve original missing20200918 bond factor. Additional2019 run freezes ALL new plans when bond20-session data input is incomplete, without inventing bars. This is a sensitivity scenario, not corrected truth.",
        limitations="Selected surviving ETF universe; pre-inception absence; factor-reinvestment approximation; no historical QDII premium gate; five correlated hypotheses, bootstrap intervals unadjusted for selection.",
        data_sha256=cache.snapshot_digest, protected_sha256=hashes,
    ))
    (out / "source").mkdir()
    for n in ["bond_study", "adaptive", "backtest", "execution", "strategy", "metrics", "commission_study"]:
        shutil.copyfile(ROOT / f"src/steadyquant/{n}.py", out / f"source/{n}.py")
    (out / "data").mkdir()
    for s, d in data.items():
        d.to_parquet(out / f"data/{s}.parquet", index=False)
    weights, prefixes = {}, {}
    for name, (normal, bull, _) in POLICIES.items():
        print(f"Signals + causality: {name}", flush=True)
        cap = caps_for(data, normal, bull)
        weights[name], _ = adaptive_weights(data, cfg, bond_cap=cap)
        assert (weights[name]["511010.SH"] <= cap + 1e-8).all()
        weights[name].to_parquet(out / f"targets_{name}.parquet")
        cap.to_frame("bond_cap").to_parquet(out / f"cap_{name}.parquet")
        truncated = {s: d[d.date <= "2022-12-30"] for s, d in data.items()}
        prefix, _ = adaptive_weights(truncated, cfg, bond_cap=caps_for(truncated, normal, bull))
        pd.testing.assert_frame_equal(prefix, weights[name].loc[prefix.index])
        prefixes[name] = True
    # Confirm the new optional research argument leaves the active strategy exactly unchanged.
    original, _ = adaptive_weights(data, cfg)
    pd.testing.assert_frame_equal(original, weights["base30"])
    rows, yearly_rows, checks, results, reviews = [], [], {}, {}, []
    jobs = [(start, mode, mul) for start in ["2015-01-05", "2019-01-01", "2023-01-01"]
            for mode, mul in ([("normal", 1), ("double_cost", 2)] if start != "2015-01-05" else [("normal", 1)])]
    jobs.append(("2019-01-01", "gap_freeze", 1))
    for start, mode, mul in jobs:
        for name in POLICIES:
            key = f"{name}_{start[:4]}_{mode}"
            print(f"Simulate + independent ledger replay: {key}", flush=True)
            mask = gap_freeze_mask(data, weights[name], cfg, start) if mode == "gap_freeze" else None
            result = simulate(data, weights[name], cfg, start=start, end=end, cost_multiplier=mul,
                              rebalance_mask=mask)
            checks[key] = save_result(out / key, result, data, cfg, mul)
            results[key] = result
            row = dict(model=name, fresh_start=start, mode=mode, **metrics(result, cfg, cost_multiplier=mul))
            groups = {}
            for bucket in {a["bucket"] for a in cfg["assets"]}:
                symbols = [a["symbol"] for a in cfg["assets"] if a["bucket"] == bucket]
                groups[bucket] = float(result.positions[symbols].sum(axis=1).mean())
            row.update(average_bond=groups["bond"], average_equity=sum(v for k, v in groups.items() if k.endswith("equity")),
                       average_gold=groups["gold"], average_commodity=groups["commodity"],
                       ending_equity=float(result.equity.equity.iloc[-1]), trade_days=result.trades.date.nunique())
            rows.append(row)
            if start == "2019-01-01" and mode == "normal":
                yearly_rows.extend(dict(model=name, **r) for r in yearly(result.equity.equity, .02).to_dict("records"))
                for period, lo, hi in [("train", "2019-01-01", "2022-12-31"), ("review", "2023-01-01", None)]:
                    reviews.append(dict(model=name, period=period, **metrics(result, cfg, lo, hi)))
    table, review = pd.DataFrame(rows), pd.DataFrame(reviews)
    table.to_csv(out / "comparison.csv", index=False)
    review.to_csv(out / "split_review.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(out / "yearly.csv", index=False)
    train = review[review.period == "train"].set_index("model")
    chosen = train.sharpe.idxmax()
    if train.loc[chosen, "sharpe"] < train.loc["base30", "sharpe"] + .03:
        chosen = "base30"
    intervals = {name: {year: paired_sharpe_interval(results[f"{name}_{year}_normal"].equity.equity,
                                                     results[f"base30_{year}_normal"].equity.equity)
                        for year in ["2019", "2023"]} for name in POLICIES if name != "base30"}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert cache.digest(cfg) == cache.snapshot_digest
    summary = dict(end=end, retrospective_candidate_selected_on_pre2023=chosen,
                   active_unchanged=True, original_baseline_matched=True, prefix_checks=prefixes,
                   replay_checks=checks, intervals=intervals, comparison=rows, split_review=reviews)
    write_json(out / "summary.json", summary)
    render(out, table, review, results, intervals, end)
    write_json(ROOT / "outputs/latest_bond_cap.json", dict(path=str(out)))
    print(table[["model", "fresh_start", "mode", "cagr", "sharpe", "max_drawdown", "average_bond", "average_equity"]].to_string(index=False), flush=True)
    print(f"REPORT={out}", flush=True)
    return out


def render(out, table, review, results, intervals, end):
    labels = {name: v[2] for name, v in POLICIES.items()}
    def display(frame):
        d = frame[["model", "cagr", "sharpe", "max_drawdown", "volatility", "average_bond", "average_equity", "average_cash_weight", "trades"]].copy()
        d["model"] = d.model.map(labels)
        for c in ["cagr", "max_drawdown", "volatility", "average_bond", "average_equity", "average_cash_weight"]:
            d[c] = d[c].map(lambda v: f"{v:.2%}")
        d["sharpe"] = d.sharpe.map(lambda v: f"{v:.3f}")
        d.columns = ["方案", "年化收益", "夏普", "最大回撤", "年化波动", "平均债券", "平均股票", "平均现金", "成交笔数"]
        return '<div class="box" style="overflow-x:auto">' + d.to_html(index=False, border=0) + '</div>'
    body = f'<h1>债券比例与夏普对照</h1><p class="sub">20 万本金 · 月频 · 截至 {end} · 5 个预先固定方案</p>'
    body += '<div class="note">固定降低债券上限，或在沪深300、标普500、恒生中至少两只高于252日均线且126日上涨时降低。释放额度按原风险算法再分配，不等于直接加股票。股票合计60%、单ETF30%、目标波动10%限制保持不变。万1.5最低5元，单边滑点5bp，收益已扣费。</div>'
    for start in ["2019-01-01", "2023-01-01", "2015-01-05"]:
        body += f'<h2>{start[:4]} 年起空仓投入20万</h2>' + display(table[(table.fresh_start == start) & (table["mode"] == "normal")])
    for title, mode in [("费用与滑点翻倍（2019起）", "double_cost"), ("数据缺口冻结交易敏感性（2019起）", "gap_freeze")]:
        body += f'<h2>{title}</h2>' + display(table[(table.fresh_start == "2019-01-01") & (table["mode"] == mode)])
    fig, dd = go.Figure(), go.Figure()
    for name in POLICIES:
        nav = results[f"{name}_2019_normal"].equity.equity
        fig.add_trace(go.Scatter(x=nav.index, y=nav / nav.iloc[0], name=labels[name]))
        dd.add_trace(go.Scatter(x=nav.index, y=nav / nav.cummax() - 1, name=labels[name]))
    dd.update_yaxes(tickformat=".0%")
    body += '<h2>净值</h2>' + figure_html(fig) + '<h2>回撤</h2>' + figure_html(dd)
    body += '<h2>验证边界</h2><p class="sub">全部模拟完成独立现金/份额/费用重放，信号截断重算一致。保留原始数据缺口；敏感性组在债券20日数据输入不完整时冻结整个组合的新调仓计划，不补造价格，也不是已修复数据的真值。所有时期都是已见历史，存在资产池选择和多方案比较偏差；复权近似分红再投资，无历史QDII溢价闸门。未修改实盘账户和配置。</p>'
    rows = [dict(model=labels[name], period=year, lower_95=v["lower_95"], upper_95=v["upper_95"])
            for name, years in intervals.items() for year, v in years.items()]
    body += '<h2>相对原版夏普差值的历史重采样95%区间（未作多重检验校正）</h2>' + pd.DataFrame(rows).to_html(index=False, border=0, float_format=lambda v:f"{v:.3f}")
    (out / "report.html").write_text(shell("债券比例研究", body, plotly=True), encoding="utf-8")


if __name__ == "__main__":
    run()
