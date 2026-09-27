"""Local-only Streamlit research workstation. No broker or external publishing."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if os.environ.get("POCKETBAY_DATA_DIR"):
    from scripts.pocketbay_bootstrap import prepare, prepare_environment

    prepare_environment()
    if __name__ == "__main__" and not st.runtime.exists():
        os.execv(sys.executable, prepare())

from steadyquant.config import ROOT, load_active_config, load_config, write_json
from steadyquant.data import Cache, DataError
from steadyquant.reports import number, pct

cloud_mode = bool(os.environ.get("POCKETBAY_DATA_DIR"))
st.set_page_config(
    page_title="逸风ETF调仓助手" if cloud_mode else "SteadyQuant · 个人量化工作台",
    page_icon="🌱" if cloud_mode else "◈",
    layout="wide",
)
if cloud_mode:
    st.markdown(
        """<style>
        .stApp,[data-testid="stAppViewContainer"]{background:#f6faf7;color:#193b34}
        .block-container{max-width:1080px;padding-top:6rem}
        h1,h2,h3,[data-testid="stMarkdownContainer"] p{color:#193b34}
        [data-testid="stCaptionContainer"] p{color:#526a61!important}
        .eyebrow{color:#276c54;font-size:20px;font-weight:750;letter-spacing:.02em;margin-bottom:14px}
        [data-testid="stRadioOption"]{background:#edf5ef;border:1px solid #d7e8dc;
            border-radius:999px;padding:7px 14px}
        [data-testid="stRadioOption"]:has(input:checked){background:#dcefe3;border-color:#87bea1}
        [data-testid="stRadioOption"] p{color:#254d3d!important;font-size:16px;font-weight:650}
        [data-testid="stMetric"]{background:#fff;border:1px solid #dce9de;
            border-radius:14px;padding:16px}
        [data-testid="stMetricValue"]{color:#267b60}
        button[kind="primary"]{background:#2e8b6d;border:0;border-radius:10px}
        button[kind="primary"] p{color:#fff!important}
        [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{
            background:#fff;color:#193b34;border-color:#b9d4c2}
        a{color:#267b60!important}hr{border-color:#dce9de}
        </style>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<style>
.stApp{background:#0b1019;color:#e9eef6}[data-testid="stSidebar"]{background:#101925}
.block-container{padding-top:2.5rem;max-width:1480px}h1{letter-spacing:-1px}
[data-testid="stMetric"]{background:#121d2c;border:1px solid #233247;padding:20px;border-radius:12px}
[data-testid="stMetricValue"]{color:#79d5c0}a{color:#79d5c0!important}
.eyebrow{color:#72d5bc;font-size:12px;letter-spacing:3px}.muted{color:#8da1bc;font-size:14px;line-height:1.8}
button[kind="primary"]{background:#3c9d89;border:0}hr{border-color:#243247}
</style>""",
        unsafe_allow_html=True,
    )
if os.environ.get("POCKETBAY_DATA_DIR"):
    from steadyquant.family_auth import get_user, login, register

    invite_code = os.environ.get("STEADYQUANT_CLOUD_PASSWORD", "")
    if not invite_code:
        st.error("家庭邀请码尚未配置，请联系网站管理员。")
        st.stop()
    cloud_user = get_user(ROOT, st.session_state.get("cloud_user_id", ""))
    if cloud_user is None:
        st.title("🌱 逸风ETF调仓助手")
        st.caption("每人一个账号，持仓和买卖清单只属于自己。行情与历史回测由大家共用。")
        st.caption("旧版共用持仓不会自动带入；注册后请在“我的持仓”重新核对一次。")
        login_tab, signup_tab = st.tabs(["登录", "注册"])
        with login_tab, st.form("family_login"):
            login_name = st.text_input("用户名", key="login_name")
            login_password = st.text_input("密码", type="password", key="login_password")
            if st.form_submit_button("登录", type="primary"):
                user = login(ROOT, login_name, login_password)
                if user:
                    st.session_state.cloud_user_id = user.id
                    st.rerun()
                st.error("用户名或密码错误；连续输错 5 次会暂停登录 15 分钟。")
        with signup_tab, st.form("family_signup"):
            signup_name = st.text_input("新用户名（2—24 个中文、英文、数字或下划线）")
            signup_password = st.text_input("设置密码（至少 8 个字符）", type="password")
            signup_confirm = st.text_input("再输入一次密码", type="password")
            signup_invite = st.text_input("家庭邀请码（原网站访问密码）", type="password")
            if st.form_submit_button("创建账号"):
                if signup_password != signup_confirm:
                    st.error("两次密码不一致。")
                else:
                    try:
                        user = register(ROOT, signup_name, signup_password, signup_invite, invite_code)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state.cloud_user_id = user.id
                        st.rerun()
        st.stop()
    greeting, logout = st.columns([5, 1])
    greeting.caption(f"你好，{cloud_user.name}")
    if logout.button("退出登录"):
        st.session_state.pop("cloud_user_id", None)
        st.rerun()

cfg = load_active_config()
activation_path = ROOT / "data/strategy_activation.json"
activation = json.loads(activation_path.read_text()) if activation_path.exists() else None
cache = Cache()

if os.environ.get("POCKETBAY_DATA_DIR"):
    from steadyquant.parent_ui import render

    render(cfg, cache, cloud_user)
    st.stop()


def chart(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1019",
        plot_bgcolor="#0b1019",
        font={"color": "#9baec6"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=height,
        legend={"orientation": "h", "y": 1.13},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#203046")
    fig.update_yaxes(gridcolor="#203046")
    st.plotly_chart(fig, width="stretch")


def latest(kind: str) -> Path | None:
    path = ROOT / f"outputs/latest_{kind}.json"
    return Path(json.loads(path.read_text())["path"]) if path.exists() else None


with st.sidebar:
    st.markdown("### ◈ SteadyQuant")
    st.caption("个人量化研究工作台")
    page = st.radio(
        "工作区",
        [
            "组合总览",
            "策略优化",
            "自适应研究",
            "个股综合研究",
            "非量价因子研究",
            "资金与佣金",
            "行业卫星研究",
            "策略验证",
            "因子实验室",
            "数据中心",
            "每日调仓",
            "账户与运行",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**运行模式**　研究 / 建议")
    location = "PocketBay 云端" if os.environ.get("POCKETBAY_DATA_DIR") else "本机运行"
    st.caption(
        f"Tushare × AKQuant 0.3.55\n\n{location} · 无杠杆 · "
        + ("月首交易日调仓" if cfg.get("rebalance_frequency") == "monthly_first_session" else "周频调仓")
    )
    if activation:
        st.caption("当前启用：等风险月初配置 · 真实账户建议，尚未连接券商")
    st.caption("回测规则在配置中固定，实验结果不会自动替换默认策略。")

research = latest("research")
daily_path = latest("daily")
stats = json.loads((research / "summary.json").read_text()) if research else None

if page == "组合总览":
    if activation:
        research = Path(activation["research_path"])
        stats = json.loads((research / "summary.json").read_text())
        stats["oos"] = stats["test_2023"]
    st.markdown('<div class="eyebrow">PORTFOLIO / OVERVIEW</div>', unsafe_allow_html=True)
    st.title("让长期投资更有章法")
    st.markdown(
        '<p class="muted">分散风险、控制成本，以可复现的研究支持每一次调仓。</p>', unsafe_allow_html=True
    )
    if activation:
        st.info(
            "已按你的指示启用等风险月初策略。下方收益是历史回测，尚无真实成交收益；每日调仓页使用真实现金与持仓生成清单。"
        )
        if (ROOT / "data/commission_parameter_update.json").exists():
            st.caption("下方沿用原历史报告；按实际万1.5、最低5元重跑的三档本金结果见「资金与佣金」。")
    if not stats:
        if os.environ.get("POCKETBAY_DATA_DIR"):
            from steadyquant.cloud_setup import read_status, start

            state = read_status()
            st.info("云端首次使用需要获取 ETF 历史行情并生成回测，完成后这里会显示收益、净值和目标配置。")
            if state.get("state") in {"queued", "running"}:
                stage = {"starting": "准备中", "sync": "正在获取行情", "backtest": "正在计算回测"}
                st.warning(f"初始化中：{stage.get(state.get('stage'), '处理中')}。可稍后刷新页面。")
                if st.button("刷新进度"):
                    st.rerun()
            else:
                if state.get("state") == "failed":
                    st.error(state.get("message", "初始化未完成"))
                if st.button("获取行情并生成回测", type="primary"):
                    start()
                    st.rerun()
        else:
            st.info("尚无研究结果。在终端运行 .venv/bin/sq sync 与 .venv/bin/sq backtest。")
    else:
        from steadyquant.metrics import performance

        eq = pd.read_parquet(research / "equity.parquet").loc["2019-01-01":].copy()
        if len(eq) < 3:
            st.info("2019 年以后的回测数据不足，暂时无法展示区间指标。")
            st.stop()
        s = performance(eq.equity, cfg["risk_free_rate"])
        overview_oos = performance(eq.equity.loc["2023-01-01":], cfg["risk_free_rate"])
        columns = st.columns(4)
        for col, label, value in zip(
            columns,
            ["年化收益", "夏普（无风险 2%）", "最大回撤", "2023 起回顾检验夏普"],
            [pct(s["cagr"]), number(s["sharpe"]), pct(s["max_drawdown"]), number(overview_oos.get("sharpe"))],
        ):
            col.metric(label, value)
        st.caption(
            f"展示区间 · {s['start']} — {s['end']}（现有回测数据截止日） · 截取既有回测并将起点净值归一，不代表从2019年重新投入本金或真实账户收益"
        )
        st.subheader("净值路径")
        fig = go.Figure()
        for key, label, color in [
            ("equity", "稳健配置", "#73d9bc"),
            ("baseline", "固定配置", "#8395dc"),
            ("cn_equity_benchmark", "沪深300 ETF", "#586c83"),
            ("shanghai_composite", "上证综指 · 价格指数", "#d9a96f"),
        ]:
            if key not in eq:
                continue
            if activation and key == "equity":
                label = "等风险月初策略（历史回测）"
            fig.add_trace(
                go.Scatter(
                    x=eq.index, y=eq[key] / eq[key].iloc[0], name=label, line={"color": color, "width": 2}
                )
            )
        chart(fig)
        if "shanghai_composite" in eq:
            st.caption("上证综指是点位参考线，不含分红和交易成本，不能直接按指数点位买入。")
        left, right = st.columns([1.1, 1])
        with left:
            st.subheader("最近目标配置")
            weights = pd.read_parquet(research / "targets.parquet").iloc[-1]
            allocation_date = str(weights.name.date())
            if daily_path:
                daily_report = json.loads((daily_path / "report.json").read_text())
                if daily_report.get("allocations"):
                    weights = pd.Series(
                        {
                            row["symbol"]: row["target_weight"]
                            for row in daily_report["allocations"]
                            if row["symbol"] != "CASH"
                        }
                    )
                    allocation_date = daily_report["signal_date"]
            st.caption(f"目标截至 {allocation_date} 收盘，不代表实际持仓")
            names = {a["symbol"]: a["name"] for a in cfg["assets"]}
            allocation = pd.Series(
                list(weights) + [max(0.0, 1 - weights.sum())],
                index=[names[s] for s in weights.index] + ["现金"],
            ).sort_values(ascending=False, kind="stable")
            fig = go.Figure(
                go.Bar(
                    y=allocation.index,
                    x=allocation.values,
                    orientation="h",
                    text=[f"{value:.2%}" for value in allocation],
                    textposition="outside",
                    cliponaxis=False,
                    marker_color=["#73859e" if name == "现金" else "#72d5bc" for name in allocation.index],
                    hovertemplate="%{y}：%{x:.2%}<extra></extra>",
                )
            )
            fig.update_layout(showlegend=False, bargap=0.38)
            fig.update_xaxes(
                range=[0, max(float(allocation.max()) * 1.25, 0.05)],
                visible=False,
                fixedrange=True,
            )
            fig.update_yaxes(autorange="reversed", automargin=True, fixedrange=True, showgrid=False)
            chart(fig, 390)
        with right:
            st.subheader("研究状态")
            st.write(
                "规则：按资产类别的波动与相关性调整预算，预测波动上限10%，单ETF上限30%；每月首个交易日收盘调仓，首次建仓例外。"
                if activation
                else "规则：20% A股、15% 美国、5% 香港、20% 黄金、30% 国债、10% 商品；趋势与波动约束可将部分预算留作现金。"
            )
            overview_trades = pd.read_parquet(research / "trades.parquet")
            overview_trades = overview_trades[
                pd.to_datetime(overview_trades.date).between(eq.index[0], eq.index[-1])
            ]
            st.write(f"区间平均投资比例：{pct(eq.exposure.mean())}　区间模拟交易：{len(overview_trades)}")
            native = stats["native_engine_check"]
            st.write("独立引擎核对：" + ("通过" if native["verified"] else "未通过 / 范围受限"))
            st.caption(
                "历史时间切分不等于真正未见数据。当前资产池存在选择偏差；复权收益使用除权日再投资近似。"
            )
            st.download_button(
                "下载独立研究报告",
                (
                    research.parent / "adaptive_report.html" if activation else research / "report.html"
                ).read_bytes(),
                "steadyquant-research.html",
                "text/html",
            )

elif page == "策略优化":
    st.markdown('<div class="eyebrow">RESEARCH / STRATEGY EVOLUTION</div>', unsafe_allow_html=True)
    st.title("先控制回撤，再争取收益")
    st.caption("研究优先级：最大回撤尽量低于15%，其次争取更高夏普与收益。历史回撤不是未来损失上限。")
    optimized = latest("candidate") or latest("refinement")
    if not optimized or not (optimized / "summary.json").exists():
        optimized = latest("optimization")
    elif not json.loads((optimized / "summary.json").read_text()).get("selected"):
        optimized = latest("optimization")
    if not optimized:
        st.info("优化研究进行中，完成后在此展示完整结果。")
        st.code(".venv/bin/sq optimize --max-drawdown 0.15")
    else:
        osum = json.loads((optimized / "summary.json").read_text())
        if (optimized / "superseded.json").exists():
            st.warning("执行规则已修复，正在重算；下列旧结果已标记为被取代，请等待新结果。")
        performance_row = osum["strategy"]
        display_name = {
            "fixed_sat25_monthly": "分散底仓＋25%趋势质量增强（月频）",
            "fixed_core_monthly": "月频分散配置",
            "broad_core_30": "月频宽基分散配置（30%股票预算用于风格扩展）",
            "broad_core_50": "月频宽基分散配置（50%股票预算用于风格扩展）",
        }.get(osum["selected"], osum["selected"])
        columns = st.columns(4)
        for col, label, value in zip(
            columns,
            ["候选年化收益", "候选夏普", "候选最大回撤", "2023起回顾检验年化"],
            [
                pct(performance_row["cagr"]),
                number(performance_row["sharpe"]),
                pct(performance_row["max_drawdown"]),
                pct(osum["test_2023"]["cagr"]),
            ],
        ):
            col.metric(label, value)
        st.caption(f"完整历史：{performance_row['start']} — {performance_row['end']}；夏普采用2%无风险利率。")
        st.write(
            f"候选：**{display_name}**。这是回顾性研究结果。"
            + ("每日建议已按用户指示启用等风险月初策略。" if activation else "默认每日策略仍为首版稳健配置。")
        )
        if osum.get("universe_extension"):
            st.caption(
                "研究中加入创业板（159915）、科创50（588000）、纳指（513100），保持A股20%、美国15%的总预算。3组配置按2022年底之前的收益和风险比较。"
            )
        elif osum.get("simplicity_review"):
            st.caption("消融评审：额外轮动未改善风险调整收益，优先观察简单配置。已保留全部复杂模型。")
        elif osum.get("prior_run"):
            st.caption(
                "第二轮：75%或50%分散底仓搭配轮动增强，比较周频与月频；两轮共保留16组结果。所有检验均为回顾性研究。"
            )
        if not osum["full_drawdown_pass"] or not osum["test_drawdown_pass"]:
            st.warning("候选没有在全部检验区间满足回撤要求，不能据此替换默认策略。")
        if not osum["full_target_15pct_reached"]:
            st.info("全历史年化尚未达到15%；当前候选优先满足历史回撤约束。")
        comp = pd.read_parquet(optimized / "comparison.parquet")
        fig = go.Figure()
        for label, values in comp.items():
            series = values.dropna()
            fig.add_trace(go.Scatter(x=series.index, y=series / series.iloc[0], name=label))
        chart(fig, 420)
        st.caption(
            "各配置按各自起点归一。候选与v1使用2015年相同起点；若显示walk_forward，该线为2018起的历史自适应选型诊断。"
        )
        tabs = st.tabs(["配置与模型对比", "熊市与成本", "逐年验证", "最新研究配置", "模型与数据证据"])
        with tabs[0]:
            rows = pd.read_csv(optimized / "candidates.csv")
            period = st.selectbox(
                "检验区间",
                ["test", "selection", "full"],
                format_func=lambda v: {
                    "test": "2023起：回顾性检验",
                    "selection": "2015—2022：选型区间",
                    "full": "2015起：完整历史",
                }[v],
            )
            st.dataframe(
                rows.loc[
                    rows.period == period, ["candidate", "cagr", "sharpe", "max_drawdown", "volatility"]
                ],
                hide_index=True,
                width="stretch",
            )
            first_path = latest("optimization")
            if first_path:
                with st.expander("查看16只ETF上的轮动与机器学习研究"):
                    first = pd.read_csv(first_path / "candidates.csv")
                    st.dataframe(
                        first.loc[first.period == period, ["candidate", "cagr", "sharpe", "max_drawdown"]],
                        hide_index=True,
                        width="stretch",
                    )
            if osum.get("refinement"):
                with st.expander("查看底仓加轮动增强的8组对比"):
                    previous_rows = pd.read_csv(Path(osum["refinement"]) / "candidates.csv")
                    st.dataframe(
                        previous_rows.loc[
                            previous_rows.period == period, ["candidate", "cagr", "sharpe", "max_drawdown"]
                        ],
                        hide_index=True,
                        width="stretch",
                    )
        with tabs[1]:
            rows = [
                {
                    "window": w,
                    "strategy": c,
                    "return": v.get("total_return"),
                    "drawdown": v.get("max_drawdown"),
                }
                for w, models in osum["stress_windows"].items()
                for c, v in models.items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.dataframe(pd.read_csv(optimized / "sensitivity.csv"), hide_index=True, width="stretch")
            st.caption("这些压力与邻域参数只做诊断，不用于在2023之后重新选型。")
        with tabs[2]:
            if (optimized / "fresh_annual_folds.csv").exists():
                st.dataframe(
                    pd.read_csv(optimized / "fresh_annual_folds.csv"), hide_index=True, width="stretch"
                )
                st.caption(
                    "固定候选规则，每年以新资金单独启动；可使用该年以前的行情。属于历史诊断，不是此前未知的前瞻检验。"
                )
            else:
                folds = json.loads((optimized / "forward_selection.json").read_text())
                st.dataframe(
                    pd.DataFrame(
                        [{k: r[k] for k in ("year", "selected", "selection_through")} for r in folds]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.json(osum["walk_forward"])
                st.caption("每年只看此前三年净收益与回撤选择模型，然后在该年采用该模型当时产生的目标权重。")
        with tabs[3]:
            if osum.get("universe_extension"):
                choices = ["fixed_core_monthly", "broad_core_30", "broad_core_50"]
                profile = st.selectbox(
                    "查看配置",
                    choices,
                    index=choices.index(osum["selected"]),
                    format_func=lambda key: {
                        "fixed_core_monthly": "简单月频配置（评分候选）",
                        "broad_core_30": "宽基扩展：股票预算30%",
                        "broad_core_50": "宽基扩展：股票预算50%",
                    }[key],
                )
                candidate_config_path = optimized / profile / "config.yaml"
            else:
                profile = osum["selected"]
                candidate_config_path = optimized / "candidate.yaml"
            candidate_cfg = load_config(candidate_config_path)
            weights = pd.read_parquet(optimized / profile / "targets.parquet").iloc[-1]
            names = {a["symbol"]: a["name"] for a in candidate_cfg["assets"]}
            rows = [{"代码": s, "名称": names[s], "目标权重": w} for s, w in weights.items() if w > 0]
            rows.append({"代码": "CASH", "名称": "现金", "目标权重": 1 - weights.sum()})
            st.caption(f"截至 {weights.name.date()} 收盘。研究目标权重，不是实际持仓或已成交订单。")
            st.caption(
                "月频规则为每月第一个日历星期五，休市则不补调；首次建仓例外。建仓与清仓不受偏离阈值拦截。"
            )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.download_button("下载候选策略配置", candidate_config_path.read_bytes(), "candidate.yaml")
        with tabs[4]:
            st.json(osum["native_engine_check"])
            st.json(osum["bootstrap_sharpe"])
            st.json(json.loads((optimized / "protocol.json").read_text()))
            st.caption("动量、趋势效率、下行波动与回撤因子；岭回归每月滚动训练，只用收益区间已结束的标签。")
        st.download_button(
            "下载完整优化研究报告",
            (optimized / "report.html").read_bytes(),
            "steadyquant-optimization.html",
            "text/html",
        )

elif page == "自适应研究":
    from steadyquant.adaptive_report import NAMES

    st.markdown('<div class="eyebrow">RESEARCH / ADAPTIVE ALLOCATION</div>', unsafe_allow_html=True)
    st.title("先弄清收益从哪里来")
    study = latest("adaptive")
    if not study or not (study / "paper_review/summary.json").exists():
        st.info("自适应研究及核验正在准备中。")
    else:
        paper = study / "paper_review"
        if (study / "superseded.json").exists():
            st.warning("执行现金预检已更新，下列旧结果正在重算。请以新批次为准。")
        ps = json.loads((paper / "summary.json").read_text())
        s = ps["strategy"]
        cols = st.columns(4)
        for col, label, val in zip(
            cols,
            ["观察版年化", "观察版夏普", "观察版最大回撤", "2023起年化"],
            [pct(s["cagr"]), number(s["sharpe"]), pct(s["max_drawdown"]), pct(ps["test_2023"]["cagr"])],
        ):
            col.metric(label, val)
        st.caption(
            f"{s['start']} — {s['end']} · 净收益 · 等风险预算，每月首个交易日收盘决策，次日开盘模拟成交。"
        )
        st.info(
            "这是事后研究形成的观察版。8组主实验均未完全满足预设提升门槛；夏普差异置信区间跨零，稳定优势尚未证明。"
            + (
                "已按你的指示用于真实账户建议，实际成交由你执行并确认。"
                if activation
                else "每日默认策略保持首版。"
            )
        )
        eq = pd.read_parquet(study / "all_equities.parquet")
        eq["paper_equal_risk"] = pd.read_parquet(paper / "equity.parquet").equity
        fig = go.Figure()
        for name in ("fixed_core_monthly", "equal_risk", "blended_risk", "paper_equal_risk"):
            values = eq[name]
            fig.add_trace(go.Scatter(x=values.index, y=values / values.iloc[0], name=NAMES[name]))
        chart(fig, 420)
        atabs = st.tabs(["模型与区间", "熊市与稳定性", "收益来源", "配置与证据"])
        with atabs[0]:
            period = st.selectbox(
                "自适应检验区间",
                ["full", "selection", "test"],
                format_func=lambda x: {
                    "full": "2015起完整历史",
                    "selection": "2015—2022选型区间",
                    "test": "2023起回顾检验",
                }[x],
            )
            rows = pd.read_csv(study / "candidates.csv")
            rows["candidate"] = rows.candidate.map(NAMES)
            st.dataframe(
                rows.loc[rows.period == period, ["candidate", "cagr", "sharpe", "max_drawdown"]],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "等风险与混合配置的改善较小；估值、利率因子暂未带来稳定增益。观察版的月初规则在事后诊断后保留，不属于上表预设主实验。"
            )
        with atabs[1]:
            st.dataframe(
                pd.read_csv(paper / "yearly.csv")[["year", "total_return", "max_drawdown"]],
                hide_index=True,
                width="stretch",
            )
            st.dataframe(
                pd.read_csv(paper / "sensitivity.csv")[["variant", "cagr", "sharpe", "max_drawdown"]],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "same_fills为相同成交额外扣成本；其余成本组重新运行账本，现金和整手变化会改变调仓路径。所有参数邻域只做诊断。"
            )
            st.json(ps["paired_sharpe_interval"])
        with atabs[2]:
            mechanism = json.loads((study / "diagnostics/mechanisms.json").read_text())
            st.write(
                "资产未上市时将预算保留为现金，与在可用资产间重新分配，会产生明显收益差异。新模型的全部增益不能都算作择时能力。"
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {"对照": "可用预算重新投资的固定配置", **mechanism["conditional_fixed_full"]},
                        {"对照": "2022年前平均权重，2023起固定持有", **mechanism["pre2023_mean_from2023"]},
                        {"对照": "等风险月初版，2023起新资金", **ps["fresh_2023"]},
                    ]
                )[["对照", "start", "cagr", "sharpe", "max_drawdown"]],
                hide_index=True,
                width="stretch",
            )
            st.caption("平均权重只用2022年底之前的数据计算，不能回填到2015年。各对照起点在表中明确列出。")
            profits = pd.read_csv(study / "diagnostics/asset_profit.csv")
            st.dataframe(
                profits.groupby(["model", "bucket"], as_index=False).profit_cny.sum(),
                hide_index=True,
                width="stretch",
            )
        with atabs[3]:
            pc = load_config(paper / "config.yaml")
            w = pd.read_parquet(paper / "targets.parquet").iloc[-1]
            names = {a["symbol"]: a["name"] for a in pc["assets"]}
            allocations = [{"代码": k, "名称": names[k], "目标权重": v} for k, v in w.items()]
            allocations.append({"代码": "CASH", "名称": "现金", "目标权重": 1 - w.sum()})
            st.caption(f"{w.name.date()}收盘研究权重；不代表实际账户持仓或订单。")
            st.dataframe(pd.DataFrame(allocations), hide_index=True, width="stretch")
            st.download_button("下载观察版配置", (paper / "config.yaml").read_bytes(), "adaptive_paper.yaml")
            st.json(ps["native_engine_check"])
            st.caption(
                "已统一三条路径的收盘现金预检，并重新核对成交。相对阈值和快速减仓没有稳定增益，未纳入观察版。历史估值缺少修订版本保证，尚未成为默认策略因子。"
            )
            with st.expander("预设实验协议和信息时点"):
                st.json(json.loads((study / "protocol.json").read_text()))
        if (study / "adaptive_report.html").exists():
            st.download_button(
                "下载自适应研究完整报告",
                (study / "adaptive_report.html").read_bytes(),
                "adaptive_research.html",
                "text/html",
            )

elif page == "个股综合研究":
    from steadyquant.composite_ui import render_page

    render_page(chart)

elif page == "策略验证":
    st.markdown('<div class="eyebrow">VALIDATION / ROBUSTNESS</div>', unsafe_allow_html=True)
    st.title("收益之外，看风险如何发生")
    if not stats:
        st.info("请先运行 sq backtest。")
    else:
        tabs = st.tabs(["回撤与年份", "成本与参数", "前推与压力期", "验证证据"])
        with tabs[0]:
            eq = pd.read_parquet(research / "equity.parquet").equity
            dd = eq / eq.cummax() - 1
            fig = go.Figure(
                go.Scatter(x=dd.index, y=dd, fill="tozeroy", line={"color": "#c2a277"}, name="回撤")
            )
            fig.update_yaxes(tickformat=".0%")
            chart(fig, 280)
            st.dataframe(pd.read_csv(research / "yearly.csv"), hide_index=True, width="stretch")
        with tabs[1]:
            try:
                sensitivity = pd.read_csv(research / "sensitivity.csv")
            except pd.errors.EmptyDataError:
                sensitivity = pd.DataFrame()
            st.dataframe(sensitivity, hide_index=True, width="stretch")
            st.info(
                "这些变体用于检查结果是否脆弱，不用于挑选最佳参数。成本包含佣金和滑点；最低佣金同时随压力倍数增加。"
            )
        with tabs[2]:
            st.write("固定参数，逐年用新资金启动；因子仍可使用该年之前的历史窗口。")
            st.dataframe(pd.read_csv(research / "forward_folds.csv"), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(stats["stress_windows"]).T, width="stretch")
            st.json(stats["bootstrap_sharpe"])
        with tabs[3]:
            st.json(stats["native_engine_check"])
            st.json(stats["research_gate"])
            st.json(json.loads((research / "protocol.json").read_text()))
            st.download_button(
                "下载逐笔交易", pd.read_parquet(research / "trades.parquet").to_csv(index=False), "trades.csv"
            )

elif page == "因子实验室":
    st.markdown('<div class="eyebrow">RESEARCH / FACTOR LAB</div>', unsafe_allow_html=True)
    st.title("把想法变成可检验的因子")
    st.caption("AKQuant 表达式引擎 · 只允许正向历史窗口 · 标签从下一日开盘开始计算")
    if research:
        st.dataframe(pd.read_csv(research / "factor_summary.csv"), hide_index=True, width="stretch")
    st.warning("六只跨资产 ETF 的横截面很小，IC 仅用于方法诊断，不能据此证明个股选股能力。")
    with st.form("factor"):
        name = st.text_input("因子名称", "my_momentum")
        expr = st.text_input("表达式", "Close / Ref(Close, 60) - 1")
        submitted = st.form_submit_button("计算并诊断", type="primary")
    if submitted:
        from steadyquant.factors import compute, evaluate

        try:
            with st.spinner("计算历史因子..."):
                data = cache.load(cfg)
                factor = compute(data, {name: expr})
                summary, detail = evaluate(data, factor)
            st.dataframe(summary, hide_index=True, width="stretch")
            st.dataframe(factor.tail(30), hide_index=True, width="stretch")
            st.download_button("下载因子结果", factor.to_csv(index=False), "factor.csv")
        except Exception as exc:
            st.error(f"表达式计算失败：{type(exc).__name__}。检查函数、字段及历史窗口；不支持负向引用。")
    with st.expander("常用表达式"):
        st.code(
            "Close / Ref(Close, 120) - 1\nClose / Ts_Mean(Close, 120) - 1\nRank(Ts_Mean(Volume, 20))\nTs_Std(Close / Ref(Close, 1) - 1, 60)"
        )

elif page == "数据中心":
    st.markdown('<div class="eyebrow">DATA / LOCAL CACHE</div>', unsafe_allow_html=True)
    st.title("每一个结果，都能追溯到数据")
    coverage = cache.coverage()
    st.dataframe(coverage, hide_index=True, width="stretch")
    st.caption("原始行情与复权因子分别保留；Parquet 存储，DuckDB 查询；按年度下载，增量刷新重叠40天。")
    p = ROOT / "data/last_sync.json"
    if p.exists():
        status = json.loads(p.read_text())
        st.write("最近同步：" + ("完整" if status["ok"] else "部分失败"))
        st.json(status)
    quarantine = list((ROOT / "data/quarantine").glob("*.parquet"))
    if quarantine:
        st.warning("以下源数据记录缺少复权因子，已隔离；未填造数据，不在当日生成信号或成交。")
        st.dataframe(pd.concat([pd.read_parquet(p) for p in quarantine]), hide_index=True, width="stretch")
    if st.button("刷新固定资产池", type="primary"):
        from steadyquant.data import sync

        try:
            with st.spinner("从已配置的 Tushare 服务刷新..."):
                status = sync(cfg)
            if status["ok"]:
                st.success("同步完成")
            else:
                st.error("部分数据未就绪，详见同步日志。")
        except DataError as exc:
            st.error(str(exc))
    if not os.environ.get("POCKETBAY_DATA_DIR"):
        st.code(
            ".venv/bin/sq fetch 000001.SZ --kind stock --start 20200101\n.venv/bin/sq sync --full",
            language="bash",
        )
    st.caption(
        "个股缓存可供因子研究。默认可执行策略使用 ETF；单债和个股账户撮合尚未加入税费、公司行动及历史成分处理。"
    )

elif page == "每日调仓":
    st.markdown('<div class="eyebrow">DAILY / ALLOCATION DESK</div>', unsafe_allow_html=True)
    st.title("先核对，再执行")
    st.caption(
        "每个交易日生成报告。"
        + (
            "每月首个交易日收盘调仓，首次建仓例外；下一交易日开盘参考。"
            if cfg.get("rebalance_frequency") == "monthly_first_session"
            else "默认周五判断调仓，下一交易日开盘参考。"
        )
        + "实际成交由你执行并确认。"
    )
    if st.button("用当前缓存生成简报", type="primary"):
        from steadyquant.daily import daily

        with st.spinner("检查数据和持仓..."):
            daily(cfg, refresh=False)
        st.rerun()
    if daily_path:
        report = json.loads((daily_path / "report.json").read_text())
        st.subheader(report["status"])
        st.caption(f"信号日期 {report['signal_date']} → 下一交易日 {report['execution_date']}")
        for note in report["notes"]:
            st.write(note)
        st.dataframe(pd.DataFrame(report["allocations"]), hide_index=True, width="stretch")
        if report["orders"]:
            st.dataframe(pd.DataFrame(report["orders"]), hide_index=True, width="stretch")
        st.download_button(
            "下载每日简报", (daily_path / "report.html").read_bytes(), "daily-report.html", "text/html"
        )
    else:
        st.info("运行 sq daily 或点击上方按钮生成第一份简报。")

elif page == "非量价因子研究":
    from steadyquant.alternative_ui import render_page

    render_page(chart)

elif page == "行业卫星研究":
    from steadyquant.sector_ui import render_page

    render_page(chart)

elif page == "资金与佣金":
    from steadyquant.commission_ui import render_page

    render_page(chart)

elif page == "账户与运行":
    st.markdown('<div class="eyebrow">OPERATIONS / ACCOUNT</div>', unsafe_allow_html=True)
    st.title("完整持仓，清晰边界")
    st.info(
        "未录入时只给目标权重。录入后按完整账户生成份额清单；账户需确认至当日收盘，系统不会自动假定建议已成交。"
    )
    from steadyquant.daily import calendar_dates

    try:
        signal, _ = calendar_dates(cache)
    except DataError:
        signal = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d")
    account_path = ROOT / "data/portfolio.json"
    try:
        saved_account = json.loads(account_path.read_text()) if account_path.exists() else {}
    except (OSError, ValueError):
        saved_account = {}
    try:
        account_date = pd.Timestamp(saved_account.get("as_of", signal)).date()
    except (ValueError, TypeError):
        account_date = pd.Timestamp(signal).date()
    with st.form("account"):
        as_of = st.date_input("已核对的收盘日期", value=account_date)
        cash = st.number_input("可用现金（元）", min_value=0.0, value=float(saved_account.get("cash", 0)), step=1000.0)
        positions = {}
        columns = st.columns(2)
        for i, asset in enumerate(cfg["assets"]):
            positions[asset["symbol"]] = columns[i % 2].number_input(
                f"{asset['name']} {asset['symbol']}（份）",
                min_value=0,
                value=int(saved_account.get("positions", {}).get(asset["symbol"], 0)),
                step=100,
            )
        confirmed = st.checkbox("这是账户的完整现金与持仓，已核对上述收盘日期")
        save = st.form_submit_button("保存账户快照")
    if save:
        if not confirmed or cash + sum(positions.values()) <= 0:
            st.error("请填写有效账户并确认完整持仓。")
        else:
            account = {"cash": cash, "positions": positions, "as_of": str(as_of), "confirmed": True}
            write_json(ROOT / "data/portfolio.json", account)
            st.success("账户快照已保存在" + ("PocketBay 云端" if os.environ.get("POCKETBAY_DATA_DIR") else "本机") + "。")
    if not os.environ.get("POCKETBAY_DATA_DIR"):
        st.subheader("运行命令")
        st.code(
            ".venv/bin/sq sync\n.venv/bin/sq backtest\n.venv/bin/sq daily --notify\n.venv/bin/sq app",
            language="bash",
        )
        st.caption("定时运行状态见 docs/OPERATIONS.md。Mac 睡眠或关机时无法保证准时刷新；唤醒后可手动补跑。")
