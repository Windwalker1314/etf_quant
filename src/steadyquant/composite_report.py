"""Self-contained stock/ETF research report, without any brokerage instruction."""

import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .composite_study import NAMES
from .config import ROOT


def table(frame):
    d = frame.copy()
    for col in ("cagr", "max_drawdown", "total_return", "target_weight", "average_stock_exposure"):
        if col in d:
            d[col] = d[col].map(lambda x: f"{x:.2%}")
    if "sharpe" in d:
        d["sharpe"] = d.sharpe.map(lambda x: f"{x:.2f}")
    if "candidate" in d:
        d["candidate"] = d.candidate.map(NAMES)
    return d.to_html(index=False, escape=True, border=0)


def render_composite(folder):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text())
    rows = pd.read_csv(folder / "candidates.csv")
    reviewed = summary["reviewed_stock_model"]
    picks = pd.read_csv(folder / reviewed / "selection.csv")
    picks = picks[picks.date == picks.date.max()].copy()
    weights = pd.read_parquet(folder / reviewed / "targets.parquet").iloc[-1]
    names = pd.read_parquet(ROOT / "data/stocks/metadata.parquet").set_index("ts_code").name
    picks["name"] = picks.symbol.map(names)
    picks["target_weight"] = picks.symbol.map(weights).fillna(0)
    picks[["date", "symbol", "name", "rank", "target_weight"]].to_csv(
        folder / "stock_watchlist.csv", index=False
    )
    eq = pd.read_parquet(folder / "comparison.parquet")
    fig = go.Figure()
    control = "equity_overlay_25" if reviewed.endswith("25") else "equity_overlay_40"
    for name in ("etf_core", reviewed, control):
        fig.add_trace(go.Scatter(x=eq.index, y=eq[name] / eq[name].iloc[0], name=NAMES[name]))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111d2c",
        plot_bgcolor="#111d2c",
        height=440,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=35, b=25),
        legend=dict(orientation="h"),
    )
    chart = fig.to_html(full_html=False, include_plotlyjs=True)
    dd = go.Figure()
    for name in ("etf_core", reviewed, control):
        dd.add_trace(go.Scatter(x=eq.index, y=eq[name] / eq[name].cummax() - 1, name=NAMES[name]))
    dd.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111d2c",
        plot_bgcolor="#111d2c",
        height=320,
        hovermode="x unified",
        yaxis_tickformat=".0%",
    )
    stock = summary["models"][reviewed]["metrics"]["full"]
    title = "个股 + ETF 综合量化研究"
    html_text = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:#09111c;color:#dbe6f1;margin:0}}main{{max-width:1260px;margin:auto;padding:48px 28px}}h1{{font-size:38px}}h2{{margin-top:42px;font-size:24px}}p,li{{line-height:1.85;color:#a9bbcd}}.eyebrow{{letter-spacing:3px;color:#6ed8bd;font-size:12px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card,.box{{background:#111d2c;border:1px solid #26384a;border-radius:12px;padding:22px}}.value{{font-size:32px;color:#75dbc1}}.warning{{border-left:3px solid #ddb864;padding:16px 22px;background:#202331}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #26384a}}th{{color:#75dbc1}}.scroll{{overflow:auto}}a{{color:#75dbc1}}code{{color:#a3d7f5}}@media(max-width:720px){{.grid{{grid-template-columns:1fr}}}}</style><main>
<div class="eyebrow">STEADYQUANT / EQUITY + MULTI-ASSET / RESEARCH</div><h1>{title}</h1>
<p>历史沪深300主板股票池 · 20万元起始资金 · 不加杠杆 · 数据截至 {summary["as_of"]}</p>
<div class="warning">本页为研究结果及研究权重。当前真实账户建议仍采用已启用的ETF策略，未发送个股订单。综合模型选择仅依据预设验证区间；所有历史均属回顾研究，财务数据缺少完整的原始版本档案。</div>
<h2>重点复核：{NAMES[reviewed]}</h2><p>按2021—2022验证期夏普选择需要详细复核的个股模型。{"通过研究门槛的模型：" + ", ".join(NAMES[x] for x in summary["qualified"]) if summary["qualified"] else "六个综合模型均未通过全部研究门槛，研究结论保留ETF底仓。"}</p>
<div class="grid"><div class="card">完整区间年化<div class="value">{stock["cagr"]:.2%}</div></div><div class="card">夏普，扣2%无风险收益<div class="value">{stock["sharpe"]:.2f}</div></div><div class="card">最大回撤<div class="value">{stock["max_drawdown"]:.2%}</div></div></div>
{chart}{dd.to_html(full_html=False, include_plotlyjs=False)}
<h2>所有模型，同一起点与资金</h2><p>2016起主区间。三个选股族各配25%和40%上限，共六个候选；ETF及沪深300敞口对照不参与选型。实际个股仓位会随市场趋势降为一半或零。其余预算由原等风险ETF底仓承担。</p>
<div class="scroll">{table(rows[rows.period == "full"][["candidate", "cagr", "sharpe", "max_drawdown"]])}</div>
<h2>验证期与2023起回顾</h2><p>验证期：2021—2022；2023后结果作为回顾比较，不能称为从未看过的样本外证明。15%—20%年化不是用来反向调参的通过条件。</p>
<div class="scroll">{table(rows[rows.period.isin(["validation", "retrospective_2023"])][["candidate", "period", "cagr", "sharpe", "max_drawdown"]])}</div>
<h2>熊市、成本与持股数邻域</h2><p>先看单独的2015年新资金股灾压力测试：</p><div class="scroll">{table(pd.read_csv(folder / "stress_2015.csv")[["candidate", "total_return", "max_drawdown"]])}</div><p>主区间逐年结果：</p><div class="scroll">{table(pd.read_csv(folder / reviewed / "yearly.csv")[["year", "total_return", "max_drawdown"]])}</div>
<div class="scroll">{table(pd.read_csv(folder / "sensitivity.csv")[["variant", "cagr", "sharpe", "max_drawdown"]])}</div>
<p>成本压力提高佣金和滑点；法定印花税与过户费按历史日期计提。改变持股数仅作敏感性检验，不重新选赢家。每年新资金启动结果见 fresh_annual.csv。</p>
<h2>研究配置快照</h2><p>{summary["as_of"]}收盘的模型目标权重，未按今日行情刷新，不是实际持仓，也不是新买入清单。</p><div class="scroll">{table(pd.read_csv(folder / "research_allocations.csv"))}</div>
<p>最近一次选出的8只个股观察池：{"当前市场趋势开关关闭，个股目标权重全部为零。" if picks.target_weight.sum() == 0 else "以下权重仅用于研究观察。"}</p><div class="scroll">{table(picks[["date", "symbol", "name", "rank", "target_weight"]])}</div>
<h2>模型与成交规则</h2><ul><li>历史成分记录严格晚于观察日期才进入股票池，超过62天未更新则停用；不拿今天的成分股回填。</li><li>质量：按财政月份折算的ROE、ROA；价值：正盈利收益率、账面市值比；动量：剔除最近一个月的半年/一年表现；防御：63日低波动。只使用公告日早于信号日、初始版本标志为0的财报。</li><li>剔除当时ST/退市风险名称、流动性不足、历史不足、因子缺失、100股成本超过6250元的股票；主板为主，创业板和科创板敞口通过ETF保留。</li><li>每月首个交易日收盘选股，次日开盘按固定股数尝试成交；卖出先执行，现金不足会缩量；涨停不买、跌停不卖、停牌无成交，不预知翌日价格。</li><li>现金分红按登记日权益、除息日应收、支付日入账，保守预扣20%红利税；送股待上市日可卖。未知公司行动保留原始价格损失并阻止实盘资格，退市残值按零处理。</li><li>ETF原账本与新账本全区间比较，所有个股成交和公司事件另行重放核算；不把账本重放等同于券商成交或AKQuant个股原生验证。</li></ul>
<h2>验证证据与限制</h2><pre class="box">{html.escape(json.dumps(dict(data=summary["data_quality"], etf_parity=summary["etf_ledger_parity"], prefix=summary["prefix_checks"], reviewed_replay=summary["models"][reviewed]["replay"], reviewed_gates=summary["models"][reviewed]["gates"], paired_sharpe_interval=summary["paired_sharpe_interval"]), ensure_ascii=False, indent=2))}</pre>
<p>估值日线与历史财报来自当前供应商数据库，没有逐日保存的历史修订版本保证。所有比较都保留这一限制。行业历史分类尚未纳入，相关个股仍可能集中在同类行业；20万元整手约束也会留下现金。</p>
<p>规则参考：<a href="https://tushare.pro/document/2?doc_id=96">历史指数成分</a> · <a href="https://tushare.pro/document/2?doc_id=79">财务指标接口</a> · <a href="https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988866.shtml">2015过户费规则</a> · <a href="https://www.chinaclear.cn/zdjs/gszb/202204/f89e788c65a241e88e7f0d0348de586f.shtml">2022过户费通知</a> · <a href="https://www.chinatax.gov.cn/chinatax/n810341/n810765/n1465977/n1466017/c1967339/content.html">差别化红利税</a></p>
</main></html>"""
    path = folder / "composite_report.html"
    path.write_text(html_text)
    return path
