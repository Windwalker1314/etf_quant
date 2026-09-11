"""Read-only presentation of composite research, isolated from active account flows."""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .composite_study import NAMES
from .config import ROOT
from .reports import number, pct


def render_page(chart):
    st.markdown('<div class="eyebrow">RESEARCH / STOCKS + ETF</div>', unsafe_allow_html=True)
    st.title("让个股增强接受同样严格的检验")
    pointer = ROOT / "outputs/latest_composite.json"
    if not pointer.exists():
        st.info("历史股票池、财务因子与个股成交账本正在准备。当前账户建议仍采用已启用的ETF策略。")
        return
    folder = Path(json.loads(pointer.read_text())["path"])
    if (folder / "superseded.json").exists():
        st.warning("公司行动账本发现需要修正的边界，旧批次已作废，正在重新核验。当前ETF账户建议不受影响。")
        return
    summary = json.loads((folder / "summary.json").read_text())
    model_names = list(summary["models"])
    model = st.selectbox(
        "综合研究模型",
        model_names,
        index=model_names.index(summary["reviewed_stock_model"]),
        format_func=lambda x: NAMES[x],
    )
    s = summary["models"][model]
    m = s["metrics"]["full"]
    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["年化收益", "夏普", "最大回撤", "平均个股仓位"],
        [pct(m["cagr"]), number(m["sharpe"]), pct(m["max_drawdown"]), pct(s["average_stock_exposure"])],
    ):
        col.metric(label, value)
    st.caption(f"{m['start']} — {m['end']} · 20万元起始资金 · 净费用与保守股息税 · 研究结果")
    st.info(
        "综合策略仍处于研究阶段，尚未替换真实账户建议。"
        + (
            "通过全部研究门槛：" + ", ".join(NAMES[x] for x in summary["qualified"])
            if summary["qualified"]
            else "六个个股模型均未满足全部门槛，研究结论保留ETF底仓。"
        )
    )
    eq = pd.read_parquet(folder / "comparison.parquet")
    fig = go.Figure()
    lines = list(
        dict.fromkeys(
            ["etf_core", model, "equity_overlay_25" if model.endswith("25") else "equity_overlay_40"]
        )
    )
    for name in lines:
        fig.add_trace(go.Scatter(x=eq.index, y=eq[name] / eq[name].iloc[0], name=NAMES[name]))
    chart(fig, 420)
    tabs = st.tabs(["模型与区间", "熊市与稳定性", "研究配置", "成交与时点证据"])
    with tabs[0]:
        period = st.selectbox(
            "综合检验区间",
            ["full", "development", "validation", "retrospective_2023"],
            format_func=lambda x: {
                "full": "2016起完整历史",
                "development": "2016—2020开发期",
                "validation": "2021—2022验证期",
                "retrospective_2023": "2023起回顾检验",
            }[x],
        )
        rows = pd.read_csv(folder / "candidates.csv")
        rows["模型"] = rows.candidate.map(NAMES)
        st.dataframe(
            rows.loc[rows.period == period, ["模型", "cagr", "sharpe", "max_drawdown"]],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "质量价值、动量防御、多因子均衡各配25%/40%个股上限；固定6组。沪深300对照帮助区分选股效果与股票风险增加。"
        )
    with tabs[1]:
        crash = pd.read_csv(folder / "stress_2015.csv")
        crash["模型"] = crash.candidate.map(NAMES)
        st.write("2015年新资金股灾压力测试")
        st.dataframe(crash[["模型", "total_return", "max_drawdown"]], hide_index=True, width="stretch")
        st.dataframe(
            pd.read_csv(folder / model / "yearly.csv")[["year", "total_return", "max_drawdown"]],
            hide_index=True,
            width="stretch",
        )
        st.write(
            "下列邻域与每年新资金启动核验，针对验证期夏普最高的个股模型："
            + NAMES[summary["reviewed_stock_model"]]
        )
        st.dataframe(pd.read_csv(folder / "sensitivity.csv"), hide_index=True, width="stretch")
        st.dataframe(
            pd.read_csv(folder / "fresh_annual.csv")[["year", "cagr", "sharpe", "max_drawdown"]],
            hide_index=True,
            width="stretch",
        )
        st.json(summary["paired_sharpe_interval"])
    with tabs[2]:
        w = pd.read_parquet(folder / model / "targets.parquet").iloc[-1]
        meta = pd.read_parquet(ROOT / "data/stocks/metadata.parquet").set_index("ts_code").name.to_dict()
        protocol = json.loads((folder / "protocol.json").read_text())
        meta.update({a["symbol"]: a["name"] for a in protocol["etf_config"]["assets"]})
        st.caption(
            f"{w.name.date()}收盘研究目标权重；不是实际持仓或新买入建议。当前真实账户继续使用ETF策略。"
        )
        st.dataframe(
            pd.DataFrame(
                [{"代码": k, "名称": meta.get(k, k), "权重": f"{v:.2%}"} for k, v in w[w > 0].items()]
            ),
            hide_index=True,
            width="stretch",
        )
        st.write(
            "主板个股限制：当时非ST、充足历史与流动性、可用初始财报，100股成本不超过6250元。创业板和科创板继续通过ETF覆盖。"
        )
        selection_path = folder / model / "selection.csv"
        if selection_path.exists():
            pool = pd.read_csv(selection_path)
            pool = pool[pool.date == pool.date.max()].copy()
            pool["名称"] = pool.symbol.map(meta)
            pool["当前研究权重"] = pool.symbol.map(w).fillna(0)
            st.write("最近的8只个股观察池；零权重表示当前没有建仓信号。")
            st.dataframe(
                pool[["date", "symbol", "名称", "rank", "当前研究权重"]], hide_index=True, width="stretch"
            )
    with tabs[3]:
        st.json(s.get("gates", {"note": "对照模型不参与个股候选筛选"}))
        st.json(s["replay"])
        st.json(summary["etf_ledger_parity"])
        st.json(summary["prefix_checks"])
        st.json(summary["data_quality"])
        st.warning(
            "所有历史均为回顾检验，供应商财报没有完整历史修订档案保证。账本重放不等于券商成交；未知公司行动会阻止实盘资格。"
        )
        if s["qualification_blocks"]:
            events = pd.read_parquet(folder / model / "events.parquet")
            st.dataframe(events[events.type == "qualification_block"], hide_index=True, width="stretch")
    report = folder / "composite_report.html"
    if report.exists():
        st.download_button("下载个股综合研究报告", report.read_bytes(), "composite_report.html", "text/html")
