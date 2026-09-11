"""Offline evidence report for the entire adaptive study and its paper review."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .config import ROOT

NAMES = {
    "fixed_core_monthly": "原月频分散配置",
    "broad_core_50": "宽基固定配置",
    "inverse_vol": "逆波动率配置",
    "equal_risk": "等风险配置",
    "trend_risk": "趋势等风险",
    "blended_risk": "固定与等风险混合",
    "value_risk": "混合＋估值",
    "rate_risk": "混合＋利率",
    "value_rate_risk": "混合＋估值与利率",
    "trend_value_rate": "趋势＋估值与利率",
    "paper_equal_risk": "等风险月初观察版",
}


def render_adaptive_report():
    root = Path(json.loads((ROOT / "outputs/latest_adaptive.json").read_text())["path"])
    p = root / "paper_review"
    s = json.loads((p / "summary.json").read_text())
    m = json.loads((root / "diagnostics/mechanisms.json").read_text())
    summary = s["strategy"]
    test = s["test_2023"]
    eq = pd.read_parquet(root / "all_equities.parquet")[
        ["fixed_core_monthly", "broad_core_50", "equal_risk", "blended_risk"]
    ]
    eq["paper_equal_risk"] = pd.read_parquet(p / "equity.parquet").equity
    fig = go.Figure()
    for name, values in eq.items():
        fig.add_trace(go.Scatter(x=values.index, y=values / values.iloc[0], name=NAMES[name]))
    fig.update_layout(
        template="plotly_dark", height=500, hovermode="x unified", title="2015年同起点 · 已扣成本的历史净值"
    )

    def table(frame):
        return frame.to_html(index=False, float_format=lambda x: f"{x:.4f}")

    candidates = pd.read_csv(root / "candidates.csv")
    candidates["candidate"] = candidates.candidate.map(NAMES)
    controls = pd.DataFrame(
        [
            dict(model="宽基固定：可用预算重新投资", period="2015起", **m["conditional_fixed_full"]),
            dict(model="2022年前平均等风险权重固定持有", period="2023起新资金", **m["pre2023_mean_from2023"]),
            dict(model="等风险月初版", period="2023起新资金", **s["fresh_2023"]),
        ]
    )
    full = candidates[candidates.period == "full"]
    sens = pd.read_csv(p / "sensitivity.csv")
    periods = pd.read_csv(p / "yearly.csv")
    stress = pd.DataFrame(
        [
            dict(window=window, model=NAMES[name], **v)
            for window, models in s["stress_windows"].items()
            for name, v in models.items()
        ]
    )
    native = json.loads((root / "diagnostics/native_checks.json").read_text())
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SteadyQuant 自适应配置研究</title>
<style>body{{font:16px/1.75 system-ui;background:#0b1019;color:#dce7f6;max-width:1400px;margin:35px auto;padding:25px}}h1,h2{{color:#79d5c0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #293b4e;text-align:right}}pre{{white-space:pre-wrap}}a{{color:#79d5c0}}.metrics{{font-size:24px;color:#79d5c0}}.note{{border-left:4px solid #e5b367;padding:15px;background:#172131}}</style>
<h1>自适应配置 · 收益改善有限，风险分配更有价值</h1>
<p class="metrics">观察版年化 {summary["cagr"]:.2%} · 夏普 {summary["sharpe"]:.2f} · 最大回撤 {summary["max_drawdown"]:.2%}</p>
<p>2015-01-05至2026-09-09，无杠杆、下一交易日开盘模拟成交，佣金3bps/最低5元及滑点5bps。2023起回顾性检验：年化{test["cagr"]:.2%}，夏普{test["sharpe"]:.2f}，最大回撤{test["max_drawdown"]:.2%}。</p>
<p class="note">固定主实验8组均未完全达到预设提升门槛，正式选型仍保留原月频配置。等风险月初版是看过历史后的观察候选；其结果不能称为全新样本外验证，也没有替换每日默认策略。配对夏普差异置信区间跨零，稳定优势尚未证明。全历史未达到15%年化。</p>
{fig.to_html(full_html=False, include_plotlyjs=True)}
<h2>主实验：全部结果保留</h2>{table(full[["candidate", "cagr", "sharpe", "max_drawdown"]])}
<p>以资产类别而非ETF数量分配风险；126日协方差向对角矩阵收缩30%，单ETF上限30%、股票合计60%、波动上限10%。等风险配置改善回撤，混合配置收益略高但最差年份更差；估值与利率没有稳定的额外贡献。</p>
<h2>收益与风险改善来自哪里</h2>{table(controls[["model", "period", "cagr", "sharpe", "max_drawdown"]])}
<p>固定预算在资产不可用时保持现金；允许可用预算重新投资即可提高收益，因此不能把全部增益归因于新因子。用2022年前的平均权重固定持有，是检验资产结构与动态调整作用的对照；该平均权重只能用于2023起，不能回填2015。</p>
<h2>年度与熊市</h2>{table(periods[["year", "total_return", "max_drawdown"]])}{table(stress[["window", "model", "total_return", "max_drawdown"]])}
<h2>观察版成本与参数邻域</h2>{table(sens[["variant", "cagr", "sharpe", "max_drawdown"]])}
<p>重跑成本会改变整手数量和后续调仓，所以收益不一定单调；same_fills为相同成交的额外成本扣减，仅用于直接成本诊断。没有根据这些邻域结果重新选参数。</p>
<h2>信息时点与数据限制</h2><p>新增Shibor及沪深300/中证500/创业板PE、PB共四组历史序列，全部滞后一个交易日；估值只用过去1260天、至少252天的滚动分位。缺失超过7个日历日不填造。数据源不保证历史修订版本，不能声称完全未修订的时点基本面。没有纳入缺少可靠公告时间的盈利或信用因子。ETF复权按除权日再投资近似，QDII溢价与开盘流动性仍未充分建模。</p>
<p>方法依据：<a href="https://www.nber.org/papers/w22208">Moreira与Muir：Volatility Managed Portfolios</a>；数据字段：<a href="https://tushare.pro/document/2?doc_id=128">Tushare指数每日指标</a>、<a href="https://tushare.pro/document/2?doc_id=149">Shibor</a>。论文结果不能直接套用到本ETF组合。</p>
<h2>执行核对</h2><p>观察版核对状态：{s["native_engine_check"]["verified"]}；独立账本{s["native_engine_check"]["ledger_trades"]}笔成交，40日最大净值差异{s["native_engine_check"]["max_equity_difference_cny"]:.3g}元。已统一独立账本、原生适配器、每日建议的收盘现金预检，先卖后买并按可用现金缩量，修复此前整单拒绝与事后缩量不一致。以下为修复后的核对；40日验证不等于全历史成交验证，隔夜跳空或卖单未成交仍可能产生引擎差异。</p><pre>{html.escape(json.dumps(native, ensure_ascii=False, indent=2))}</pre>
<h2>观察版完整证据</h2><pre>{html.escape(json.dumps(s, ensure_ascii=False, indent=2))}</pre></html>"""
    (root / "adaptive_report.html").write_text(body)
    return root / "adaptive_report.html"
