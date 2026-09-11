from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

STYLE = """
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0b1019;color:#e9eef6;font:15px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif}main{max-width:1240px;margin:auto;padding:42px 32px}nav{font-size:12px;color:#78dcca;letter-spacing:3px;border-bottom:1px solid #243043;padding-bottom:22px}h1{font-size:38px;letter-spacing:-1px;margin-bottom:10px}h2{font-size:20px;font-weight:550;margin-top:38px}.sub{color:#93a3b9;line-height:1.8}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.card{padding:22px;background:#121d2c;border:1px solid #223146;border-radius:12px}.label{color:#9baec5;font-size:13px}.value{font-size:30px;margin:12px 0 2px;font-variant-numeric:tabular-nums}.accent{color:#72d5bc}.box{border:1px solid #23324a;border-radius:12px;background:#101925;padding:20px;margin:20px 0}.note{border-left:3px solid #b6a077;background:#1c2027;padding:16px 20px;color:#c6cbd4;line-height:1.8}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:13px 12px;border-bottom:1px solid #253043;text-align:right}th:first-child,td:first-child{text-align:left}th{color:#849bb9;font-weight:500}a{color:#71d7c0}.tag{font-size:12px;padding:5px 10px;border:1px solid #3b615d;color:#80d2bc;border-radius:30px}.footer{font-size:12px;color:#71839b;margin:32px 0;line-height:1.9}@media(max-width:700px){main{padding:20px 14px}.cards{grid-template-columns:1fr 1fr}h1{font-size:28px}.value{font-size:23px}}
"""


def pct(value):
    return "—" if value is None else f"{value:.2%}"


def number(value):
    return "—" if value is None else f"{value:.2f}"


def figure_html(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101925",
        plot_bgcolor="#101925",
        font=dict(family="-apple-system, PingFang SC, sans-serif", color="#9caec6"),
        margin=dict(l=30, r=20, t=30, b=35),
        height=350,
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#223047")
    fig.update_yaxes(gridcolor="#223047")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})


def shell(title: str, body: str, plotly: bool = False) -> str:
    script = f"<script>{get_plotlyjs()}</script>" if plotly else ""
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style>{script}</head><body><main><nav>STEADYQUANT / PERSONAL RESEARCH</nav>{body}<div class="footer">本地运行 · 数据与凭证保留在本机 · 收盘信号 / 次日执行 · 历史研究结果不能保证未来表现</div></main></body></html>'


def research_html(output: Path, cfg: dict, stats: dict, equity: pd.DataFrame, positions: pd.DataFrame):
    s = stats["strategy"]
    fig = go.Figure()
    for col, name, color in [
        ("equity", "稳健配置", "#72d5bc"),
        ("baseline", "固定配置基准", "#879adb"),
        ("cn_equity_benchmark", "沪深300 ETF", "#647084"),
    ]:
        values = equity[col] / equity[col].iloc[0]
        fig.add_trace(go.Scatter(x=values.index, y=values, name=name, line=dict(color=color, width=2)))
    dd = equity.equity / equity.equity.cummax() - 1
    drawdown = go.Figure(
        go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color="#b89a72"), name="回撤")
    )
    drawdown.update_yaxes(tickformat=".0%")
    weights = go.Figure()
    names = {a["symbol"]: a["name"] for a in cfg["assets"]} | {"CASH": "现金"}
    for col in positions:
        weights.add_trace(
            go.Scatter(
                x=positions.index, y=positions[col], name=names[col], stackgroup="one", line=dict(width=0.5)
            )
        )
    weights.update_yaxes(tickformat=".0%", range=[0, 1])
    cards = [
        ("年化收益", pct(s["cagr"])),
        ("夏普 · 无风险 2%", number(s["sharpe"])),
        ("最大回撤", pct(s["max_drawdown"])),
        ("样本外夏普 · 2023 起", number(stats["oos"].get("sharpe"))),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{label}</div><div class="value accent">{value}</div></div>'
        for label, value in cards
    )
    annual = pd.read_csv(output / "yearly.csv")
    annual = annual[["year", "total_return", "sharpe", "max_drawdown", "volatility"]]
    for col in ("total_return", "max_drawdown", "volatility"):
        annual[col] = annual[col].map(pct)
    annual["sharpe"] = annual.sharpe.map(number)
    annual.columns = ["年份", "收益", "夏普", "最大回撤", "波动率"]
    bootstrap = stats["bootstrap_sharpe"]
    confidence = f"历史区块重采样夏普 95% 区间：{number(bootstrap.get('lower_95'))} 至 {number(bootstrap.get('upper_95'))}。"
    native = stats["native_engine_check"]
    body = f'<h1>以稳健为先的资产配置</h1><p class="sub">{s["start"]} — {s["end"]} · 6 类资产敞口 · 无杠杆 · 周频调仓 <span class="tag">研究版本 v1</span></p><div class="cards">{card_html}</div>'
    body += '<div class="note">回测前固定规则；样本外为历史时间切分，仍存在当前 ETF 池的幸存者偏差。分红按复权因子变化近似为除权日再投资份额，未模拟真实派息到账日。日线成交量限制也不等于开盘实际可成交量。</div>'
    body += f'<h2>净值与基准</h2><div class="box">{figure_html(fig)}</div><h2>回撤路径</h2><div class="box">{figure_html(drawdown)}</div><h2>实际模拟持仓</h2><div class="box">{figure_html(weights)}</div><h2>逐年表现</h2><div class="box">{annual.to_html(index=False, escape=True, border=0)}</div>'
    body += f'<p class="sub">{confidence}原生引擎核对：{html.escape(str(native))}</p>'
    body += '<p class="sub">同目录含逐笔交易、因子值、IC、年度前推验证、成本与参数敏感性、配置与数据指纹。默认规则不会根据这些结果自动调参。</p>'
    (output / "report.html").write_text(shell("SteadyQuant · 研究报告", body, plotly=True))


def daily_html(report: dict) -> str:
    rows = pd.DataFrame(report.get("allocations", []))
    if not rows.empty:
        for col in ("target_weight", "current_weight", "difference"):
            if col in rows:
                rows[col] = rows[col].map(lambda v: "—" if v is None else pct(v))
        rows = rows.rename(
            columns={
                "symbol": "代码",
                "name": "资产",
                "target_weight": "目标权重",
                "current_weight": "当前权重",
                "difference": "偏离",
                "close": "参考收盘价",
            }
        )
    table = rows.to_html(index=False, border=0, escape=True) if not rows.empty else "数据未就绪"
    orders = pd.DataFrame(report.get("orders", []))
    order_html = (
        orders.to_html(index=False, border=0, escape=True)
        if not orders.empty
        else "本次没有可执行的调仓清单。"
    )
    reasons = "<br>".join(html.escape(s) for s in report.get("notes", []))
    body = f'<h1>每日组合简报</h1><p class="sub">信号日期 {html.escape(report.get("signal_date", "—"))} · 下一交易日 {html.escape(report.get("execution_date", "—"))}</p><div class="note">{html.escape(report["status"])}<br>{reasons}</div><h2>目标配置</h2><div class="box">{table}</div><h2>调仓建议</h2><div class="box">{order_html}</div>'
    return shell("SteadyQuant · 每日简报", body)
