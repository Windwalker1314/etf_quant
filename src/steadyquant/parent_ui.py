"""Small, quantity-first PocketBay interface for a family ETF account."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .cloud_setup import SETUP_VERSION, read_status, start
from .config import ROOT, fingerprint, write_json
from .daily import calendar_dates, make_report
from .data import TZ, Cache, DataError
from .family import advice_is_current, holdings_rows, read_json, save_account
from .family_auth import User
from .metrics import performance
from .reports import number, pct
from .strategy_catalog import (
    DEFAULT_STRATEGY_ID,
    Strategy,
    load_strategy_config,
    research_pointer,
    strategies,
    strategy_account_root,
)


def latest_research(cfg: dict, strategy: Strategy) -> Path | None:
    pointers = [research_pointer(ROOT, strategy)]
    if strategy.id == DEFAULT_STRATEGY_ID:
        pointers.append(ROOT / "outputs/latest_research.json")
    for pointer_path in pointers:
        pointer = read_json(pointer_path, {})
        if not pointer or not pointer.get("path"):
            continue
        path = Path(pointer["path"])
        protocol = read_json(path / "protocol.json", {})
        if (protocol.get("strategy_id", DEFAULT_STRATEGY_ID) == strategy.id
                and protocol.get("config_hash") == fingerprint(cfg)
                and (path / "equity.parquet").exists()):
            return path
    return None


def fresh_market(cache: Cache, signal: str | None) -> bool:
    sync = read_json(cache.root / "last_sync.json", {})
    return bool(signal and sync.get("ok") and sync.get("expected_date") == signal.replace("-", ""))


def current_advice(cfg: dict, cache: Cache, account: dict | None, signal: str | None,
                   account_root: Path, strategy_id: str = DEFAULT_STRATEGY_ID) -> dict | None:
    report = read_json(account_root / "outputs/family_report.json", {})
    if not account or not signal or not fresh_market(cache, signal):
        return None
    if report.get("strategy_id", DEFAULT_STRATEGY_ID) != strategy_id:
        return None
    if not advice_is_current(report, account, signal, cfg):
        return None
    if report.get("status", "").startswith("暂停"):
        return report
    try:
        if report.get("data_sha256") != cache.digest(cfg):
            return None
    except OSError:
        return None
    return report


def _signal(cache: Cache) -> tuple[str | None, str | None]:
    try:
        signal, execution = calendar_dates(cache)
        return str(pd.Timestamp(signal).date()), str(pd.Timestamp(execution).date())
    except DataError:
        return None, None


def _next_monthly_session(cache: Cache, signal: str | None) -> str | None:
    path = cache.root / "calendar.parquet"
    if not path.exists() or not signal:
        return None
    calendar = pd.read_parquet(path)
    open_days = pd.to_datetime(calendar.loc[calendar.is_open.astype(int) == 1, "cal_date"])
    first_of_month = open_days.groupby(open_days.dt.to_period("M")).min()
    firsts = first_of_month[first_of_month > pd.Timestamp(signal)]
    return str(firsts.iloc[0].date()) if len(firsts) else None


def _action_title(cfg: dict) -> str:
    monthly = cfg.get("rebalance_frequency") in {"monthly", "monthly_first_session"}
    return "本月怎么做" if monthly else "本期怎么做"


def _refresh_control(strategy_id: str) -> bool:
    state = read_status()
    if state.get("state") == "failed" and state.get("version", 0) < SETUP_VERSION:
        # One migration retry after deploying a fix for the old initializer.
        state = start()
    running = state.get("state") in {"queued", "running"}
    if running:
        stage = {"starting": "准备中", "sync": "更新 ETF 行情", "backtest": "计算历史回测"}
        st.info(f"正在{stage.get(state.get('stage'), '处理')}，完成后点“刷新页面”。")
        if st.button("刷新页面", key=f"refresh_page_{strategy_id}", width="stretch"):
            st.rerun()
    else:
        if state.get("state") == "failed":
            st.error(state.get("message", "数据更新失败，请稍后重试。"))
            if state.get("finished_at"):
                st.caption(f"本次更新结束于 {state['finished_at'][:19].replace('T', ' ')} UTC")
        if st.button("更新行情与回测", key=f"update_market_{strategy_id}",
                     type="primary", width="stretch"):
            start()
            st.rerun()
    return running


def _advice_page(cfg: dict, cache: Cache, account: dict | None, research: Path | None,
                 account_root: Path, strategy: Strategy):
    action_title = _action_title(cfg)
    st.title(action_title)
    if cfg.get("rebalance_frequency") == "monthly_first_session":
        st.caption("每月首个交易日收盘后看一次；第一次建仓可以在其他交易日生成清单。")
    else:
        st.caption("按本策略的调仓规则查看；第一次建仓可以在其他交易日生成清单。")
    signal, execution = _signal(cache)
    ready = fresh_market(cache, signal)
    if not research or not ready:
        st.info("先更新服务器上的 ETF 行情和回测。第一次会稍久，以后只补最近的数据。")
    elif signal:
        st.success(f"行情已更新至 {signal} 收盘")
    running = _refresh_control(strategy.id)
    if running:
        return

    if not account:
        st.warning("先到“我的持仓”填写实际可用现金和 ETF 份额，才能算出买卖数量。")
    elif signal:
        st.caption(f"账户上次确认：{account['as_of']} · 参考交易日：{execution or '—'}")
        if account["as_of"] < signal:
            st.warning(f"账户还没有确认至 {signal} 收盘。请核对券商账户。")
            if st.button(f"现金和持仓都没变，确认至 {signal} 收盘",
                         key=f"confirm_account_{strategy.id}", disabled=not ready):
                save_account(account_root, cfg, account["cash"], account["positions"], signal)
                st.rerun()
        elif account["as_of"] > signal:
            st.warning("账户日期晚于最新行情，请先更新行情。")

    report = current_advice(cfg, cache, account, signal, account_root, strategy.id)
    can_generate = bool(ready and account and account.get("as_of") == signal)
    if st.button(f"生成{'本月' if action_title == '本月怎么做' else '本期'}买卖清单",
                 key=f"generate_orders_{strategy.id}",
                 disabled=not can_generate, width="stretch"):
        with st.spinner("正在按已保存的行情和持仓计算…"):
            try:
                before = cache.digest(cfg)
                result = make_report(cfg, cache=cache, account=account)
                if before != cache.digest(cfg) or not fresh_market(cache, signal):
                    raise DataError("行情在计算期间发生变化，请再试一次。")
                result["account_fingerprint"] = fingerprint(account)
                result["strategy_id"] = strategy.id
                write_json(account_root / "outputs/family_report.json", result)
            except (DataError, ValueError, OSError):
                st.error("清单生成失败，请先更新行情并重新核对持仓。")
            else:
                st.rerun()

    if report:
        st.subheader(report["status"])
        st.caption(f"根据 {report['signal_date']} 收盘价计算 · {report['execution_date']} 开盘前请核对实时价格")
        if report["status"].startswith("暂停"):
            st.error("行情或账户校验未通过，本次不能按清单交易。")
        elif report.get("orders"):
            st.warning("先卖后买。下表数量单位是“份”，实际成交后请更新持仓和现金。")
        elif report["status"].startswith("观察日"):
            st.success("现在不用调仓，保持当前持仓。")
        elif report["status"].startswith("目标配置"):
            st.info("需要确认当日完整持仓后，才能生成买卖份额。")
        else:
            st.info("本次没有达到需要交易的条件。")
        st.dataframe(pd.DataFrame(holdings_rows(cfg, account, report)), hide_index=True, width="stretch")
        for note in report.get("notes", []):
            st.caption(note)
    elif account:
        st.subheader("当前持仓")
        st.dataframe(pd.DataFrame(holdings_rows(cfg, account)), hide_index=True, width="stretch")

    if signal and cfg.get("rebalance_frequency") == "monthly_first_session":
        next_session = _next_monthly_session(cache, signal)
        if next_session:
            st.caption(f"下个计划调仓判断日：{next_session} 收盘后。平时如出现风险减仓信号，也以新生成的清单为准。")


def _account_page(cfg: dict, cache: Cache, account: dict | None,
                  account_root: Path, strategy: Strategy):
    st.title("我的持仓")
    st.write("照着券商账户填写可用现金和实际持有份额。买卖成交后再更新一次。")
    signal, _ = _signal(cache)
    default_date = pd.Timestamp((account or {}).get("as_of") or signal or datetime.now(TZ).date()).date()
    with st.form(f"family_account_{strategy.id}"):
        as_of = st.date_input("持仓核对日期", value=default_date,
                              max_value=datetime.now(TZ).date(), key=f"account_date_{strategy.id}")
        cash = st.number_input("可用现金（元）", min_value=0.0, value=float((account or {}).get("cash", 0)),
                               step=100.0, format="%.2f", key=f"account_cash_{strategy.id}")
        rows = pd.DataFrame(holdings_rows(cfg, account))
        rows["当前持仓（份）"] = pd.to_numeric(rows["当前持仓（份）"]).fillna(0).astype(int)
        edited = st.data_editor(rows, hide_index=True, width="stretch", num_rows="fixed",
                                key=f"account_positions_{strategy.id}",
                                disabled=["ETF代码", "名称"],
                                column_config={"当前持仓（份）": st.column_config.NumberColumn(
                                    min_value=0, step=1, required=True)})
        confirmed = st.checkbox("我已对照券商账户，确认现金和全部持仓准确",
                                key=f"account_confirmed_{strategy.id}")
        st.caption("如果还持有表格以外的证券，请先找家人核对，不要把它们漏算为可用现金。")
        if st.form_submit_button("保存实际持仓", type="primary"):
            if not confirmed:
                st.error("请先核对并勾选确认。")
            else:
                try:
                    positions = {a["symbol"]: float(edited.iloc[i]["当前持仓（份）"])
                                 for i, a in enumerate(cfg["assets"])}
                    save_account(account_root, cfg, cash, positions, str(as_of))
                    st.success("已保存。旧买卖清单已失效，请回到“本月怎么做”重新生成。")
                except (ValueError, DataError, TypeError):
                    st.error("保存失败：请检查日期、现金和持仓份额。")


def _backtest_page(cfg: dict, research: Path | None, strategy: Strategy):
    st.title("历史回测")
    st.caption("看看策略过去的表现。历史收益不代表实际账户收益，也不能保证以后盈利。")
    if not research:
        st.info("尚无当前策略的回测结果。请先到“本月怎么做”更新行情与回测。")
        return
    equity = pd.read_parquet(research / "equity.parquet").loc["2019-01-01":]
    if len(equity) < 3:
        st.info("2019 年以来的数据不足，暂时无法展示回测。")
        return
    options = [("equity", f"当前策略 · {strategy.name}", "#248967"),
               ("shanghai_composite", "上证综指 · 价格指数", "#5675bb")]
    fig = go.Figure()
    rows = []
    for key, label, color in options:
        if key not in equity:
            continue
        series = equity[key].dropna()
        if len(series) < 3:
            continue
        normalized = series / series.iloc[0]
        fig.add_trace(go.Scatter(x=normalized.index, y=normalized, name=label,
                                 line={"color": color, "width": 3}))
        metrics = performance(series, cfg["risk_free_rate"])
        rows.append({"策略": label, "年化收益": pct(metrics["cagr"]),
                     "最大回撤": pct(metrics["max_drawdown"]),
                     "夏普比率": number(metrics["sharpe"])})
    fig.update_layout(template="plotly_white", paper_bgcolor="#fff", plot_bgcolor="#fff",
                      font={"color": "#315149"}, height=400,
                      margin={"l": 8, "r": 8, "t": 12, "b": 8},
                      legend={"orientation": "h", "y": 1.15})
    fig.update_xaxes(gridcolor="#e5eee6")
    fig.update_yaxes(gridcolor="#e5eee6")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if "shanghai_composite" not in equity:
        st.warning("尚未取得上证综指行情。请到“本月怎么做”点“更新行情与回测”后再看对比。")
    st.caption(f"展示区间：{equity.index[0].date()}—{equity.index[-1].date()}。曲线从同一天归一为 1；上证综指仅为价格点位参考，不含分红和交易成本，也不能直接按指数点位买入。只有“当前策略”用于生成买卖清单。")


def render(cache: Cache, user: User):
    st.markdown("""<style>
    .block-container{max-width:1080px}
    [data-testid="stSidebar"]{display:none}
    [data-testid="stMetric"]{padding:12px}
    button p{font-size:16px!important}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">逸风ETF调仓助手</div>', unsafe_allow_html=True)
    catalog = strategies()
    if len(catalog) > 1:
        selected_id = st.session_state.get("selected_strategy_id")
        strategy = next((item for item in catalog if item.id == selected_id), None)
        if strategy is None:
            st.title("选择策略")
            st.caption("每个策略分别记录持仓和买卖清单；同一笔资金不要在多个策略里重复计算。")
            for item in catalog:
                with st.container(border=True):
                    st.subheader(item.name)
                    st.write(item.summary)
                    if st.button(f"进入 {item.name}", key=f"choose_strategy_{item.id}"):
                        st.session_state.selected_strategy_id = item.id
                        st.rerun()
            return
        if st.button("← 切换策略", key="switch_strategy"):
            st.session_state.pop("selected_strategy_id", None)
            st.rerun()
    else:
        strategy = catalog[0]
    cfg = load_strategy_config(strategy)
    st.caption(f"当前策略：{strategy.name}")
    action_title = _action_title(cfg)
    page = st.radio("页面", [action_title, "我的持仓", "历史回测"], horizontal=True,
                    key=f"strategy_page_{strategy.id}", label_visibility="collapsed")
    account_root = strategy_account_root(ROOT, user, strategy)
    account = read_json(account_root / "data/portfolio.json")
    research = latest_research(cfg, strategy)
    if page == action_title:
        _advice_page(cfg, cache, account, research, account_root, strategy)
    elif page == "我的持仓":
        _account_page(cfg, cache, account, account_root, strategy)
    else:
        _backtest_page(cfg, research, strategy)
    st.divider()
    st.caption("行情和历史报告保存在服务器 · 按本策略规则判断调仓 · 不连接券商，不自动下单")
