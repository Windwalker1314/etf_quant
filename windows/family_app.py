"""Minimal local UI. Never copies suggested trades into confirmed holdings."""
import os
from datetime import datetime
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st
from dotenv import dotenv_values, set_key

from steadyquant.config import ROOT, fingerprint, load_active_config, write_json
from steadyquant.daily import calendar_dates, daily
from steadyquant.data import TZ, Cache, DataError
from steadyquant.family import advice_is_current, holdings_rows, read_json, save_account

st.set_page_config(page_title="ETF 小助手", page_icon="🌱", layout="centered")
st.markdown("""<style>
.block-container{max-width:1000px;padding-top:2rem}
button p{font-size:18px!important} [data-testid="stMetricValue"]{font-size:32px}
header [data-testid="stToolbar"]{display:none}
</style>""", unsafe_allow_html=True)
st.title("🌱 ETF 小助手")
st.caption("等风险月初策略 · 每月首个交易日收盘判断调仓 · 首次建仓例外")
cfg = load_active_config()
account = read_json(ROOT / "data/portfolio.json")
settings = dotenv_values(ROOT / ".env")
for key in ("TUSHARE_TOKEN", "TUSHARE_HTTP_URL"):
    if settings.get(key):
        os.environ[key] = settings[key]
ready = bool(settings.get("TUSHARE_TOKEN") and settings.get("TUSHARE_HTTP_URL"))
page = st.radio("功能", ["今日建议", "我的持仓", "首次设置"], horizontal=True,
                index=0 if ready else 2)
cache = Cache()
try:
    signal, execution = calendar_dates(cache)
    signal = str(pd.Timestamp(signal).date())
except DataError:
    signal = None

if page == "首次设置":
    st.subheader("让家人帮忙设置一次")
    st.write("填入行情服务提供的接口地址和密钥。保存后，平时只需点“更新今日建议”。")
    with st.form("settings"):
        endpoint = st.text_input("行情接口地址（HTTPS）", value=settings.get("TUSHARE_HTTP_URL", ""),
                                 placeholder="填写与密钥配套的服务地址")
        token = st.text_input("行情密钥", type="password",
                              help="已有密钥时留空即可保留；密钥只保存在这台电脑。")
        st.caption("密钥和行情请求会发送到上面填写的地址，请由家人核对地址与密钥是否配套。")
        fee = st.number_input("买卖佣金（万分之几）", min_value=0.0,
                              value=float(cfg["commission_bps"]), step=0.1)
        minimum = st.number_input("每笔最低佣金（元）", min_value=0.0,
                                  value=float(cfg["minimum_commission"]), step=1.0)
        if st.form_submit_button("保存设置", type="primary"):
            parsed = urlsplit(endpoint.strip())
            credential = token.strip() or settings.get("TUSHARE_TOKEN", "")
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                    or parsed.query or parsed.fragment or any(c.isspace() for c in endpoint)):
                st.error("请填写有效的 HTTPS 接口地址，不要在地址中填写密钥。")
            elif not credential or any(c.isspace() for c in credential):
                st.error("请填写行情密钥（不能包含空格或换行）。")
            else:
                import yaml
                ROOT.mkdir(parents=True, exist_ok=True)
                set_key(str(ROOT / ".env"), "TUSHARE_TOKEN", credential)
                set_key(str(ROOT / ".env"), "TUSHARE_HTTP_URL", endpoint.strip())
                cfg.update(commission_bps=fee, minimum_commission=minimum)
                (ROOT / "configs/active.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
                write_json(ROOT / "outputs/family_report.json", {"invalidated": True})
                st.success("已保存。请在“我的持仓”填写你自己的资金和份额，再去更新建议。")
    st.caption(f"本机数据目录：{ROOT}。重装或换电脑前，请由家人备份这个文件夹。")

elif page == "我的持仓":
    st.subheader("我的实际持仓")
    st.write("首次使用填自己的可用现金，空仓时所有份额填 0。买卖后按券商显示的实际持仓修改。")
    if account:
        st.caption(f"上次确认日期：{account['as_of']}")
    with st.form("account"):
        as_of = st.date_input("持仓日期", value=datetime.now(TZ).date(),
                              max_value=datetime.now(TZ).date(),
                              help="填写这个日期的真实持仓。日间成交后可记当天，晚间更新时再计算新建议。")
        cash = st.number_input("账户可用现金（元）", min_value=0.0,
                               value=float((account or {}).get("cash") or 0), step=100.0, format="%.2f")
        rows = pd.DataFrame(holdings_rows(cfg, account))
        rows["当前持仓（份）"] = pd.to_numeric(rows["当前持仓（份）"]).fillna(0).astype(int)
        edited = st.data_editor(rows, hide_index=True, width="stretch", num_rows="fixed",
                               disabled=["ETF代码", "名称"],
                               column_config={"当前持仓（份）": st.column_config.NumberColumn(min_value=0, step=1, required=True)})
        confirmed = st.checkbox("我已核对券商账户，以上是全部真实持仓和可用现金")
        st.caption("若还持有表格外的证券，请先找家人核对，不能遗漏后直接按全部现金建仓。")
        if st.form_submit_button("保存真实持仓", type="primary"):
            if not confirmed:
                st.error("请先核对并勾选确认。")
            else:
                try:
                    positions = {a["symbol"]: float(edited.iloc[i]["当前持仓（份）"])
                                 for i, a in enumerate(cfg["assets"])}
                    save_account(ROOT, cfg, cash, positions, str(as_of))
                    st.success("持仓已保存。原买卖建议已作废；需要时回到“今日建议”重新计算。")
                except (ValueError, DataError, TypeError):
                    st.error("保存失败：请检查现金、日期以及份额，份额必须是非负整数。")

else:
    if account:
        st.metric("可用现金", f"¥{account['cash']:,.2f}")
        st.caption(f"持仓确认于 {account['as_of']}")
    else:
        st.info("先去“我的持仓”填写自己的资金和份额。这里不会使用家人的账户。")
    st.write("建议在交易日晚上 18:30 后打开，点击下面的按钮。首次下载会比较久，请保持联网。")
    if not ready:
        st.info("请先完成“首次设置”。")
    if st.button("更新今日建议", type="primary", disabled=not ready, width="stretch"):
        # Invalidate before fetching, including on unexpected/network failures.
        write_json(ROOT / "outputs/family_report.json", {"invalidated": True})
        try:
            with st.spinner("正在下载行情并计算，请稍候…"):
                report = daily(cfg)
                report["account_fingerprint"] = fingerprint(account)
                write_json(ROOT / "outputs/family_report.json", report)
            st.rerun()
        except Exception:
            st.error("本次更新未完成，旧建议已停用。请检查网络和行情设置后重试；仍失败请联系家人。")
    if account and signal and account["as_of"] < signal:
        st.warning(f"请确认账户截至 {signal} 收盘是否仍然不变。有成交、转账或分红，请先修改持仓。")
        if st.button(f"现金、持仓都没变，确认至 {signal} 收盘"):
            save_account(ROOT, cfg, account["cash"], account["positions"], signal)
            st.rerun()
    report = read_json(ROOT / "outputs/family_report.json", {})
    current = advice_is_current(report, account, signal, cfg)
    if current:
        st.caption(f"行情截至 {report['signal_date']} 收盘 · 参考交易日 {report.get('execution_date', '—')}")
        if report["status"].startswith("暂停"):
            st.error("行情未完整更新，暂时不能生成买卖建议。请稍后重试。")
        elif report.get("orders"):
            st.warning("有买卖参考清单；以下价格和份额按收盘行情估算，交易前需核对实时价格。")
        elif report["status"].startswith("观察日"):
            st.success("今天不需要操作，保持当前持仓。")
        else:
            st.info(report["status"])
        if account and account["as_of"] != report["signal_date"]:
            st.info("持仓日期与行情日期不同，当前只显示持仓。今日已成交的持仓请在今晚更新后再计算。")
    else:
        st.info("暂无有效的最新建议。下面只展示实际持仓，请更新行情后再看买卖清单。")
    st.dataframe(pd.DataFrame(holdings_rows(cfg, account, report if current else None)),
                 hide_index=True, width="stretch")
    st.caption("买卖数量单位都是“份”。软件不自动下单；实际成交后，到“我的持仓”更新份额和剩余现金。")
st.divider()
st.caption("本地运行 · 历史研究不能保证未来收益 · 初次使用请由家人陪同核对")
