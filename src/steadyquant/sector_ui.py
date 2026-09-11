import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import ROOT
from .sector_report import readable


def render_page(chart):
    st.title("行业轮动能否改善组合夏普")
    pointer = ROOT / "outputs/latest_sector.json"
    if not pointer.exists():
        st.info("研究运行中：全市场行业/风格 ETF，固定四个候选与同仓位宽基对照。")
        return
    folder = Path(json.loads(pointer.read_text())["path"])
    summary = json.loads((folder / "summary.json").read_text())
    chosen = summary["selected"]
    st.caption(f"历史截至 {summary['data_end']} · 规则先固定 · 仅研究，未替换现行策略")
    if not summary["point_estimate_pass"]:
        st.info(f"预选 {chosen} 未通过后续区间夏普检验，保留现有核心组合。")
    elif not summary["strong_evidence"]:
        st.info(f"预选 {chosen} 的夏普点估计改善，但置信区间尚不足以确认稳定超额。")
    else:
        st.info(f"预选 {chosen} 通过历史检验，仍需要前瞻观察。")
    capital = st.selectbox(
        "行业研究本金", [100000, 200000, 1000000], index=1, format_func=lambda c: f"{c // 10000}万元"
    )
    period = st.selectbox(
        "行业研究区间",
        ["full", "selection", "validation", "review"],
        format_func=lambda p: {
            "full": "完整历史",
            "selection": "2015–2019 选型",
            "validation": "2020–2022 验证",
            "review": "2023 以后复核",
        }[p],
    )
    table = pd.read_csv(folder / "comparison.csv")
    frame = table[(table.capital == capital) & (table.period == period)]
    st.dataframe(
        readable(frame[["model", "cagr", "sharpe", "max_drawdown", "trades", "commission_cny"]]),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "core 是现行核心组合；_broad 复制该候选的实际预算与调整时点，等分沪深300和中证500；constant 是固定股票预算。"
    )
    fig = go.Figure()
    for name in ("core", chosen, chosen + "_broad"):
        nav = pd.read_parquet(folder / str(capital) / name / "carried/equity.parquet").equity
        lo, hi = {
            "full": (None, None),
            "selection": (None, "2019-12-31"),
            "validation": ("2019-12-31", "2022-12-31"),
            "review": ("2022-12-30", None),
        }[period]
        nav = nav.loc[lo:hi]
        fig.add_trace(go.Scatter(x=nav.index, y=nav / nav.iloc[0], name=name))
    chart(fig, 390)
    with st.expander("夏普差、资金重置、成本与参数敏感性", expanded=True):
        for filename in ("paired_intervals.csv", "sensitivity.csv", "stress.csv"):
            values = pd.read_csv(folder / filename)
            st.caption(filename)
            st.dataframe(readable(values[values.capital == capital]), hide_index=True, width="stretch")
    with st.expander("完整方法与验证证据"):
        st.markdown((folder / "report.md").read_text())
    st.caption("元数据历史版本未认证，包含退市基金仍不等于消除幸存者偏差；历史分段不等于未见样本外。")
    st.download_button(
        "下载行业卫星研究报告", (folder / "report.md").read_bytes(), "sector-research.md", "text/markdown"
    )
