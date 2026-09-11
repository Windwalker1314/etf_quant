"""Readable artifacts for capital-size and minimum-commission comparisons."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .commission_study import CAPITALS, NAMES


def render_report(folder: Path):
    summary = json.loads((folder / "summary.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    df = pd.read_csv(folder / "comparison.csv")
    cf = pd.read_csv(folder / "cost_counterfactual.csv")
    sens = pd.read_csv(folder / "sensitivity.csv")
    baseline = df[(df.model == "base10") & (df.period == "full")]
    fresh = df[(df.model == "base10") & (df.period == "fresh2023")]
    candidate = summary["candidate"]
    recommended = summary["recommendation"]
    account_note = (
        "账户持仓规则不变；已把后续费用估算从万3校正为用户实际万1.5、最低5元，现金和持仓未改动。"
        if summary.get("fee_parameter_update")
        else "实际账户配置和持仓文件均未修改。"
    )
    lot_path = folder / "lot_diagnostic.csv"
    lot_html = (
        "<details><summary>整手限制的事后机制诊断</summary><p>将100份整手改为假设1份交易，仅用于解释资金规模差异，不是可交易方案，不参与候选选择。</p><div class='table'>"
        + pd.read_csv(lot_path).to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}")
        + "</div></details>"
        if lot_path.exists()
        else ""
    )
    chosen = df[(df.model == recommended) & (df.period == "full")]

    def table(data, include_model=False):
        columns = (
            ["capital"]
            + (["name"] if include_model else [])
            + [
                "cagr",
                "sharpe",
                "max_drawdown",
                "trades",
                "commission_cny",
                "slippage_cny",
                "average_cash_weight",
            ]
        )
        d = data[columns].copy()
        d.capital = d.capital.map(lambda v: f"{v / 10000:g}万元")
        for col in ("cagr", "max_drawdown", "average_cash_weight"):
            d[col] = d[col].map(lambda v: f"{v:.2%}")
        for col in ("commission_cny", "slippage_cny"):
            d[col] = d[col].map(lambda v: f"{v:,.2f}")
        d.sharpe = d.sharpe.map(lambda v: f"{v:.3f}")
        d.columns = (
            ["初始资金"]
            + (["模型"] if include_model else [])
            + ["年化收益", "夏普", "最大回撤", "成交笔数", "累计佣金/元", "累计滑点/元", "平均现金"]
        )
        return d.to_html(index=False, border=0, escape=True)

    conclusion = (
        f"共同候选「{NAMES[candidate]}」通过预设收益容忍、回撤与成交减少条件；作为研究建议，尚未替换账户策略。"
        if summary["accepted"]
        else f"没有产生通过全部预设条件的新策略；保留原10只ETF规则，费用按实际万1.5、最低5元核算。训练段选中的候选为「{NAMES[candidate]}」，完整选择过程见下文。"
    )
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{c // 10000}万元" for c in CAPITALS])
    compare_model = candidate if candidate != "base10" else "min5000"
    for j, c in enumerate(CAPITALS, 1):
        for name, color in (("base10", "#7cddbe"), (compare_model, "#ffa978")):
            nav = pd.read_parquet(folder / str(c) / name / "carried/equity.parquet").equity
            fig.add_trace(
                go.Scatter(
                    x=nav.index,
                    y=nav / nav.iloc[0],
                    name=NAMES[name],
                    legendgroup=name,
                    showlegend=j == 1,
                    line=dict(color=color, width=1.5),
                ),
                row=1,
                col=j,
            )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111d2b",
        plot_bgcolor="#111d2b",
        height=410,
        margin=dict(t=65, l=35, r=20, b=25),
        legend=dict(orientation="h", y=1.2),
    )
    graph = fig.to_html(full_html=False, include_plotlyjs=True)
    selection_table = pd.DataFrame(selection["selection"])
    selection_table["model"] = selection_table.model.map(NAMES)
    details = "".join(
        f"<h3>{c // 10000}万元 · 六组完整结果</h3><div class='table'>{table(df[(df.capital == c) & (df.period == 'full')], True)}</div>"
        for c in CAPITALS
    )
    other = "".join(
        f"<h3>{label}</h3><div class='table'>{table(df[df.period == p], True)}</div>"
        for p, label in [
            ("selection", "2015—2022选择段"),
            ("review2023", "2023起延续持仓回顾检验"),
            ("fresh2023", "2023起重新投入本金"),
        ]
    )
    floor = baseline[
        [
            "capital",
            "minimum_fee_fraction",
            "floor_surcharge_same_fills_cny",
            "median_ticket_cny",
            "annual_trades",
            "cost_bps_average_nav_per_year",
        ]
    ]
    text = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>资金规模与最低佣金研究</title>
<style>body{{margin:0;background:#0b121d;color:#e5edf7;font:15px/1.75 system-ui,-apple-system,sans-serif}}main{{max-width:1350px;margin:auto;padding:38px 26px}}h1{{font-size:32px}}h2{{color:#8edcc5;margin-top:35px}}.note{{padding:18px;background:#152333;border-left:4px solid #7cddbe;border-radius:7px}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:9px;border-bottom:1px solid #293749;text-align:right}}td:first-child,th:first-child{{text-align:left}}details{{margin:22px 0;background:#111d2b;padding:18px}}summary{{cursor:pointer}}small{{color:#9aadc3}}a{{color:#7cddbe}}</style></head><body><main>
<small>STEADYQUANT / CAPITAL & COMMISSION</small><h1>10万、20万、100万：最低5元是否值得优化？</h1>
<p>完整历史：2015-01-05 — {summary["end"]}。佣金万1.5、每笔最低5元；单边滑点5bp；100份整手；月首收盘信号、次日开盘成交；无杠杆。</p>
<div class="note">{html.escape(conclusion)}</div><h2>原策略按实际费率重跑</h2><div class="table">{table(baseline)}</div>
<p>累计佣金与滑点是整个历史区间的金额，不是单年费用。资金规模改变整手误差、现金余量、最低佣金与后续复利路径，收益不会严格按比例缩放。</p>
<h2>2023年重新投入本金的公平比较</h2><div class="table">{table(fresh)}</div><p>这里三档本金都在2023年首次建仓；延续2015持仓的2023收益另外列出，不能混称为同一笔本金的新账户表现。</p>
{graph}<h2>研究建议</h2><p>{html.escape(NAMES[recommended])}；{account_note}</p><div class="table">{table(chosen)}</div>
<h2>小额交易成本来自哪里</h2><div class="table">{floor.to_html(index=False, border=0, float_format=lambda v: f"{v:.4f}")}</div>
<p>最低收费临界金额=5÷0.00015≈33333元；并不是每笔都要凑到这个金额。floor_surcharge_same_fills_cny只计算真实回测成交记录中的最低收费额外负担；它不包含收益复利或交易路径变化。cost_bps_average_nav_per_year按平均净值与年数计算佣金加滑点，不等于策略收益差。</p>
{lot_html}<details open><summary>六组固定实验</summary>{details}</details>
<details><summary>分段检验与初始资金重置</summary>{other}</details>
<details><summary>预先固定的候选选择与验证</summary><p>仅使用2015—2022数据选择共同候选，三档资金都要求年化收益下降不超过0.2个百分点、夏普下降不超过0.03、回撤恶化不超过1个百分点，且平均成交次数减少至少20%。满足条件者按成交减少最多选取；2023之后只验证，不重新挑选。</p><div class="table">{selection_table.to_html(index=False, border=0)}</div><pre>{html.escape(json.dumps(selection["validation"], ensure_ascii=False, indent=2))}</pre></details>
<details><summary>费率、最低收费与零佣金反事实</summary><div class="table">{cf.to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}")}</div><p>以上均重新执行整手与现金约束，会改变交易路径；zero_commission仍保留滑点，不能当成无摩擦收益。旧费率3bp与真实费率1.5bp比较也不保证净收益严格单调。</p></details>
<details><summary>双倍成本和20bp滑点压力</summary><div class="table">{sens.to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}")}</div></details>
<h2>验证与限制</h2><p>每次回测由成交重新构造现金、份额、复权变动与收盘净值；原生引擎另比较最近40个共同交易日（结果保存在native_checks.json，范围不扩大为全历史）。截去2022年之后的原始输入再算目标，检查前缀一致。所有历史区间在此前研究中已有观察，不是新的未见测试集。</p>
<p>早期未上市基金不会回填，10只基金尚未全部上市的时期不等于完整10基金组合。采用复权因子再投资近似，未还原每笔真实分红到账日期；没有历史QDII溢价过滤；未新增原油160723。费率按用户当前条件恒定回溯，实际券商拆单计费可能不同。降低成交次数可能牺牲跟踪精度或长期留出现金，不能只看省下的佣金。</p>
<small>目录内保留冻结规则、行情快照、完整成交、年度与压力期收益、独立复核和建议配置。仅研究，不连接券商、不提交交易。</small></main></body></html>"""
    (folder / "report.html").write_text(text)
    return folder / "report.html"
