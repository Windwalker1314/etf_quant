import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .alternative_study import CANDIDATES, NAMES
from .config import ROOT
from .reports import number, pct


def render_page(chart):
    st.markdown('<div class="eyebrow">RESEARCH / FUNDAMENTALS</div>', unsafe_allow_html=True)
    st.title("让预期变化与经营质量接受检验")
    pointer = ROOT / "outputs/latest_alternative.json"
    if not pointer.exists():
        st.info("非量价因子研究正在准备；将比较盈利预测修正、业绩新增信息、现金与经营质量。")
        return
    folder = Path(json.loads(pointer.read_text())["path"])
    summary = json.loads((folder / "summary.json").read_text())
    selected = st.selectbox(
        "非量价研究模型",
        ["etf_core"] + CANDIDATES + ["matched_price_control", "existing_momentum_defensive_25"],
        index=4,
        format_func=lambda x: NAMES[x],
    )
    detail = summary["models"][selected]
    metrics = detail["full"]
    cols = st.columns(4)
    for c, label, value in zip(
        cols,
        ["年化收益", "夏普", "最大回撤", "平均个股仓位"],
        [
            pct(metrics["cagr"]),
            number(metrics["sharpe"]),
            pct(metrics["max_drawdown"]),
            pct(detail["average_stock_exposure"]),
        ],
    ):
        c.metric(label, value)
    st.caption(f"2023-01-03 — {summary['end']} · 20万元 · 个股上限25% · 历史研究")
    st.caption("ETF对照采用自适应研究底仓；账户目前启用的仍是等风险月初配置。")
    st.info(
        (
            "历史研究门槛通过：" + "、".join(NAMES[x] for x in summary["qualified"])
            if summary["qualified"]
            else "四组新因子组合均未通过全部研究门槛。"
        )
        + " 两段时间均为回顾检验，尚未自动替换真实账户建议。"
    )
    eq = pd.read_parquet(folder / "comparison.parquet")
    lines = list(
        dict.fromkeys(
            ["etf_core", selected] + ([f"{selected}_index_control"] if selected in CANDIDATES else [])
        )
    )
    fig = go.Figure()
    for name in lines:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[name] / eq[name].iloc[0], name=NAMES[name]))
    chart(fig, 400)
    tabs = st.tabs(["组合对照", "因子与覆盖", "稳健性与证据", "最新研究观察"])
    with tabs[0]:
        period = st.selectbox(
            "非量价比较区间",
            ["full", "early", "later"],
            format_func=lambda x: {"full": "2023起全区间", "early": "2023—2024", "later": "2025起较后区间"}[
                x
            ],
        )
        d = pd.read_csv(folder / "comparison.csv")
        d["name"] = d["model"].map(NAMES)
        d = d[d.period == period][["name", "cagr", "sharpe", "max_drawdown"]].copy()
        for col in ["cagr", "max_drawdown"]:
            d[col] = d[col].map(pct)
        d.columns = ["模型", "年化收益", "夏普", "最大回撤"]
        st.dataframe(d, hide_index=True, width="stretch")
    with tabs[1]:
        horizon = st.selectbox("因子观察长度", [20, 60], format_func=lambda x: f"{x}交易日")
        f = pd.read_csv(folder / "factor_summary.csv")
        st.dataframe(
            f[(f.horizon == horizon) & f.metric.isin(["raw_ic", "controlled_ic"])][
                ["factor", "metric", "months", "mean", "lower95", "upper95", "median_cross_section"]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "控制项为历史行业、市值与价格动量；少于24个有效月份不显示区间。因子价格标签不代表可成交收益。"
        )
        coverage = pd.read_csv(folder / "coverage.csv")
        st.dataframe(
            coverage[["date", "eligible", "industry_known"] + CANDIDATES], hide_index=True, width="stretch"
        )
    with tabs[2]:
        st.dataframe(pd.read_csv(folder / "sensitivity.csv"), hide_index=True, width="stretch")
        if selected in CANDIDATES:
            st.write("研究门槛", detail["gates"])
            st.write("相对ETF夏普差的历史区间", detail["paired_sharpe_interval"])
        st.write("逐笔与公司行动独立复核", detail["replay"])
        st.caption("数据更新时间并非认证的首次公开时间；新因子暂缺2015/2018压力期证据。")
        st.download_button(
            "下载完整非量价研究报告",
            (folder / "report.html").read_bytes(),
            "alternative-factor-research.html",
            "text/html",
        )
    with tabs[3]:
        f = pd.read_csv(folder / "latest_factor_observations.csv")
        st.caption(f"最近月度因子观察：{str(f.date.iloc[0])[:10]}。以下是研究分数，不是买单。")
        f = f.sort_values("balanced", ascending=False).head(20)
        st.dataframe(f[["symbol", "name", "industry"] + CANDIDATES], hide_index=True, width="stretch")
