"""Standalone report for the complete fixed alternative-factor experiment."""

import html
import json

import pandas as pd
import plotly.graph_objects as go

from .alternative_study import CANDIDATES, NAMES


def render_report(folder):
    summary = json.loads((folder / "summary.json").read_text())
    rows = pd.read_csv(folder / "comparison.csv")
    rows["name"] = rows["model"].map(NAMES)
    eq = pd.read_parquet(folder / "comparison.parquet")
    factor = pd.read_csv(folder / "factor_summary.csv")
    coverage = pd.read_csv(folder / "coverage.csv")
    sensitivity = pd.read_csv(folder / "sensitivity.csv")
    palette = ["#7cddbe", "#ffbc78", "#97b7ff", "#d1a3ff", "#ef8d99"]
    fig = go.Figure()
    for i, name in enumerate(["etf_core"] + CANDIDATES):
        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq[name] / eq[name].iloc[0],
                name=NAMES[name],
                line=dict(color=palette[i], width=2),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111d2b",
        plot_bgcolor="#111d2b",
        height=440,
        margin=dict(t=35, l=30, r=20, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs=True)

    def metric_table(d):
        d = d[["name", "cagr", "sharpe", "max_drawdown"]].copy()
        d.columns = ["模型", "年化收益", "夏普", "最大回撤"]
        for col in ["年化收益", "最大回撤"]:
            d[col] = d[col].map(lambda x: f"{x:.2%}")
        d["夏普"] = d["夏普"].map(lambda x: f"{x:.3f}")
        return d.to_html(index=False, escape=True, border=0)

    primary = rows[
        rows.model.isin(
            ["etf_core"] + CANDIDATES + ["existing_momentum_defensive_25", "matched_price_control"]
        )
    ]
    tabs = "".join(
        f"<h2>{label}</h2>{metric_table(primary[primary.period == period])}"
        for period, label in [
            ("full", "2023起完整比较"),
            ("early", "2023—2024回顾区间"),
            ("later", "2025起较后区间"),
        ]
    )
    evidence = []
    for name in CANDIDATES:
        d = summary["models"][name]
        failed = [k for k, value in d["gates"].items() if not value]
        evidence.append(
            dict(
                模型=NAMES[name],
                平均个股仓位=f"{d['average_stock_exposure']:.1%}",
                满足8股选择月份=f"{d['complete_selection_fraction']:.0%}",
                账本复核=d["replay"]["verified"],
                公司行动阻断=d["qualification_blocks"],
                夏普差95区间=f"[{d['paired_sharpe_interval']['lower_95']:.3f}, {d['paired_sharpe_interval']['upper_95']:.3f}]",
                未通过门槛=", ".join(failed) or "无",
            )
        )
    f = factor[(factor.horizon == 20) & factor.metric.isin(["raw_ic", "controlled_ic"])].copy()
    f = f[["factor", "metric", "months", "mean", "lower95", "upper95", "median_cross_section"]]
    f.columns = ["因子", "检验", "有效月份", "平均IC", "区间下界", "区间上界", "横截面样本中位数"]
    fhtml = f.to_html(index=False, escape=True, border=0, float_format=lambda x: f"{x:.3f}", na_rep="—")
    cv = coverage[
        ["date", "eligible", "industry_known", "revision", "surprise", "cash_quality", "balanced"]
    ].copy()
    sens = sensitivity[["model", "variant", "cagr", "sharpe", "max_drawdown", "qualification_blocks"]].copy()
    for col in ["cagr", "max_drawdown"]:
        sens[col] = sens[col].map(lambda v: f"{v:.2%}")
    qualification = "、".join(NAMES[x] for x in summary["qualified"]) or "没有组合通过全部研究门槛"
    content = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SteadyQuant · 非量价因子研究</title>
<style>body{{margin:0;background:#0b121d;color:#e5edf7;font:15px/1.75 system-ui,-apple-system,sans-serif}}main{{max-width:1300px;margin:auto;padding:40px 28px}}h1{{font-size:34px}}h2{{margin-top:38px;color:#8edcc5}}p{{max-width:1020px}}.note{{padding:20px;background:#152333;border-left:4px solid #7cddbe;border-radius:8px}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:15px 0}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #293749}}th{{color:#9fb5d0}}a{{color:#7cddbe}}small{{color:#92a7c1}}details{{background:#111d2b;padding:18px;margin-top:20px}}summary{{cursor:pointer}}code{{color:#aac8ee}}</style></head>
<body><main><small>STEADYQUANT / ALTERNATIVE FACTORS / 数据来源：Tushare兼容服务</small><h1>盈利预期、业绩信息与经营质量</h1>
<p>2023-01-03 — {summary["end"]} · 各组合20万元起始资金 · 个股上限25% · 8只股票 · 同一行业最多2只 · 次日开盘、整手与历史费用。ETF对照采用自适应研究底仓，账户目前启用的仍是等风险月初配置。</p>
<div class="note"><b>{html.escape(qualification)}</b><br>本报告属于历史研究，未改变实际账户策略。2023—2026已经在此前研究中观察过，不是全新的未见测试集；历史数据更新时间与原始版本也尚非经过认证的时点数据库。</div>
{chart}<div class="table">{tabs}</div>
<h2>门槛、覆盖与独立复核</h2><div class="table">{pd.DataFrame(evidence).to_html(index=False, escape=True, border=0)}</div>
<p>完整收益、较后区间、回撤、配对夏普区间、选股覆盖和成交账本均需达标。相同股票预算转投沪深300ETF的对照结果保存在完整比较中，帮助检查增益是否来自选股。2015/2018的股灾证据仍缺失，不能靠这个较短窗口补足。</p>
<h2>20交易日因子诊断</h2><p>原始Rank IC与控制行业、市值、价格动量后的相关性并列展示；至少20/30只有效股票。区间按月分块，少于24个有效月份不计算区间。价格标签不是实际可卖空组合，交易约束以组合回测为准。</p><div class="table">{fhtml}</div>
<h2>固定敏感性核验</h2><p>费用加倍、均衡组合持股数量变化、信息再延后一交易日。所有结果保留，不把敏感性表现最好的设置替换主模型。</p><div class="table">{sens.to_html(index=False, escape=True, border=0, float_format=lambda x: f"{x:.3f}")}</div>
<details><summary>逐月数据覆盖</summary><div class="table">{cv.to_html(index=False, escape=True, border=0)}</div></details>
<details><summary>包含等仓位指数对照的完整结果</summary><div class="table">{metric_table(rows[rows.period == "full"])}</div></details>
<h2>信息使用规则</h2><p>盈利预测只比较同券商、同财年的净利润预测变化，不把EPS随意换算成净利润。公告新增信息比较同报告期已知的公司预告区间与原始归母利润；快报介于其间且利润口径不能确认时跳过该比较。现金流与利润保持合并口径，金融公司不套用工商业现金流指标。</p>
<p>财报采用可辨认的初始记录以及后来披露的比较报表；修订数字不会提前回填。晚间披露只在下一次决策中使用。缺失因子保持缺失，未凑齐8只满足行业约束的股票时返回ETF底仓。</p>
<small>研究规则、源数据校验、逐笔交易、因子与逐月检验保存在本报告所在目录。所有样本均为历史模拟，未连接券商。</small></main></body></html>"""
    (folder / "report.html").write_text(content)
