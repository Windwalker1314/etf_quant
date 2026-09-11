import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .commission_study import CAPITALS, NAMES
from .config import ROOT
from .reports import number, pct


def render_page(chart):
    st.markdown('<div class="eyebrow">RESEARCH / CAPITAL & COMMISSION</div>', unsafe_allow_html=True)
    st.title("把最低佣金与资金规模算清楚")
    pointer = ROOT / "outputs/latest_commission.json"
    if not pointer.exists():
        st.info("正在比较10万、20万和100万元的净收益、小额交易与现金余量。")
        return
    folder = Path(json.loads(pointer.read_text())["path"])
    summary = json.loads((folder / "summary.json").read_text())
    st.caption(f"2015-01-05 — {summary['end']} · 万1.5、每笔最低5元 · 次日开盘 · 不加杠杆")
    st.info(
        f"研究建议：{NAMES[summary['recommendation']]}。持仓规则保持原方案；历史分段不等于未见测试集。"
    )
    if summary.get("fee_parameter_update"):
        st.caption("已将账户后续费用估算校正为实际万1.5、每笔最低5元；没有改动现金与持仓。")
    capital = st.selectbox("比较本金", list(CAPITALS), index=1, format_func=lambda c: f"{c // 10000}万元")
    period = st.selectbox(
        "资金与佣金区间",
        ["full", "selection", "review2023", "fresh2023"],
        format_func=lambda p: {
            "full": "2015起完整历史",
            "selection": "2015—2022选择段",
            "review2023": "2023起延续原持仓",
            "fresh2023": "2023起重新投入本金",
        }[p],
    )
    d = pd.read_csv(folder / "comparison.csv")
    chosen = d[(d.capital == capital) & (d.period == period)]
    baseline = chosen[chosen.model == "base10"].iloc[0]
    for col, label, value in zip(
        st.columns(4),
        ["原策略年化", "原策略夏普", "原策略回撤", "累计佣金"],
        [
            pct(baseline.cagr),
            number(baseline.sharpe),
            pct(baseline.max_drawdown),
            f"¥{baseline.commission_cny:,.2f}",
        ],
    ):
        col.metric(label, value)
    frame = chosen[
        [
            "name",
            "cagr",
            "sharpe",
            "max_drawdown",
            "trades",
            "commission_cny",
            "slippage_cny",
            "average_cash_weight",
        ]
    ].copy()
    for col in ["cagr", "max_drawdown", "average_cash_weight"]:
        frame[col] = frame[col].map(pct)
    frame.columns = ["模型", "年化收益", "夏普", "最大回撤", "成交笔数", "累计佣金", "累计滑点", "平均现金"]
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("佣金和滑点是所选区间累计金额；小额交易过滤可能降低跟踪精度或增加现金，并非只省费用。")
    model = st.selectbox("对照模型", list(NAMES), index=1, format_func=lambda k: NAMES[k])
    mode = "fresh2023" if period == "fresh2023" else "carried"
    fig = go.Figure()
    for name in dict.fromkeys(["base10", model]):
        nav = pd.read_parquet(folder / str(capital) / name / mode / "equity.parquet").equity
        if period == "selection":
            nav = nav.loc[:"2022"]
        elif period == "review2023":
            nav = pd.concat([nav.loc[:"2022"].tail(1), nav.loc["2023":]])
        fig.add_trace(go.Scatter(x=nav.index, y=nav / nav.iloc[0], name=NAMES[name]))
    chart(fig, 390)
    with st.expander("最低收费、年度与压力期"):
        st.dataframe(
            chosen[
                [
                    "name",
                    "minimum_fee_fraction",
                    "floor_surcharge_same_fills_cny",
                    "median_ticket_cny",
                    "cost_bps_average_nav_per_year",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        for filename in ["yearly.csv", "stress.csv", "sensitivity.csv", "cost_counterfactual.csv"]:
            values = pd.read_csv(folder / filename)
            st.caption(filename)
            st.dataframe(values[values.capital == capital], hide_index=True, width="stretch")
    with st.expander("规则、选择过程与验证"):
        st.json(json.loads((folder / "selection.json").read_text()))
        st.json(json.loads((folder / "native_checks.json").read_text()))
        st.caption("复权再投资为近似；未加入原油，也未实现历史QDII溢价过滤。早期尚未上市基金不回填。")
    st.download_button(
        "下载资金与佣金研究报告",
        (folder / "report.html").read_bytes(),
        "capital-and-commission.html",
        "text/html",
    )
