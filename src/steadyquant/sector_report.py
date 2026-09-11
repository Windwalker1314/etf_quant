"""Readable results from the frozen sector experiment."""

import json

import pandas as pd

from .config import ROOT


def md_table(frame):
    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    )


def readable(frame):
    frame = frame.copy()
    for col in ("cagr", "max_drawdown", "volatility", "total_return", "average_cash_weight"):
        if col in frame:
            frame[col] = frame[col].map(lambda x: f"{x:.2%}")
    for col in ("sharpe", "lower_95", "upper_95", "median", "annual_turnover"):
        if col in frame:
            frame[col] = frame[col].map(lambda x: f"{x:.3f}")
    for col in ("commission_cny", "slippage_cny"):
        if col in frame:
            frame[col] = frame[col].map(lambda x: f"{x:,.0f}")
    return frame


def render_report(out):
    s = json.loads((out / "summary.json").read_text())
    p = json.loads((out / "protocol.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    t = pd.read_csv(out / "comparison.csv")
    ci = pd.read_csv(out / "paired_intervals.csv")
    sens = pd.read_csv(out / "sensitivity.csv")
    chosen = s["selected"]
    models = ["core", chosen, chosen + "_broad"]
    primary = t[(t.capital == 200000) & t.model.isin(models)]
    coverage = pd.read_csv(out / "coverage_raw.csv", parse_dates=["date"])
    coverage = (
        coverage.assign(year=coverage.date.dt.year)
        .groupby("year")[["eligible_etfs", "eligible_groups", "held"]]
        .mean()
        .round(1)
        .reset_index()
    )
    audit = pd.read_csv(out / "download_audit.csv")
    verdict = (
        "历史点估计通过，但尚不能认定稳定超额"
        if s["point_estimate_pass"]
        else "未通过预设的后续区间夏普检验，保留现有核心组合"
    )
    if s["strong_evidence"]:
        verdict = "后续区间夏普差及置信区间通过，仍需前瞻验证后再决定是否启用"
    text = f"""# 全市场行业/风格 ETF 卫星研究

{verdict}。仅用 2015–2019 年夏普预选的候选是 **{chosen}**。实际账户和现行配置均未更改。

数据截至 {s["data_end"]}；主资金 20 万，另检验 10 万和 100 万。{s["downloaded"]} 只下载候选中 {s["usable"]} 只行情可用，其中已退市 {s["retired_usable"]} 只。此处“全市场”是当前数据源可识别的行业、主题、风格 ETF 候选覆盖，不是全体证券或已认证的历史全集。

## 主结果：收益与风险一起看

`raw10/raw20` 为原始动量、最多 10%/20% 卫星预算；`risk10/risk20` 为波动率调整动量、同样两种预算。预选候选每只目标权重最多 5%，实际权重会随行情和交易偏离阈值漂移。

{md_table(readable(primary[["model", "period", "cagr", "sharpe", "max_drawdown", "volatility"]]))}

core 为当前多资产自适应等风险组合；候选用最多 10% 或 20% 预算持有两个不同分组 ETF；`_broad` 将候选实际占用的同一预算等分投入沪深 300 和中证 500，并复制资金调整时点。它用于检验行业选择是否胜过增加同等股票仓位。空出的卫星预算回归核心。

## 四个预先声明的候选全部公开

{md_table(readable(t[(t.capital == 200000) & t.model.isin(p["candidates"])][["model", "period", "cagr", "sharpe", "max_drawdown"]]))}

候选选型只比较 2015–2019 年：

{md_table(readable(pd.DataFrame(selection["training"])[["model", "cagr", "sharpe", "max_drawdown"]]))}

2020–2022 年为后续验证，2023 年以后为复核。所有区间均属于已发生、且可能在此前研究中被观察过的历史，不称为全新样本外。后续表现不重新选型、不追加寻优。回撤允许增加，主判断为选中候选在两个后续区间均优于核心和匹配宽基的夏普。

## 夏普差的历史不确定性

以下为选中候选减对照的夏普差，20 日成块配对重采样 500 次、95% 区间。只有两个后续区间相对两种对照的下界均大于零，才称为较强历史证据。没有多重检验校正，不能据此保证未来收益。

{md_table(readable(ci[ci.capital == 200000][["period", "control", "lower_95", "median", "upper_95"]]))}

## 资金规模和成本

{md_table(readable(t[t.model.isin(models) & (t.period == "full")][["capital", "model", "cagr", "sharpe", "max_drawdown", "trades", "commission_cny", "slippage_cny", "annual_turnover"]]))}

## 敏感性：不以结果重新调参

{md_table(readable(sens[sens.capital == 200000][["model", "variant", "cagr", "sharpe", "max_drawdown"]]))}

`fresh2023` 从 2023 年重新投入本金；`double_costs` 佣金（包括最低收费）和滑点同时翻倍；`monthly` 改为月度检查；`skip5` 动量跳过最近五个观测。完整三个资金规模结果见 sensitivity.csv。逐年、2015 年股灾、2018、2020 年一季度、2022、2025 以后压力期见 yearly.csv 和 stress.csv。

## 数据与实现检查

各年每个检查日平均符合条件的 ETF 数、分组数和持有数（原始动量）：

{md_table(coverage)}

早期可交易品种明显较少，因此不能将最近才丰富的行业 ETF 池当作整个历史一直可用。数据源有 {int((~audit.usable).sum())} 只无原始行情；共隔离 {int(audit.quarantined.fillna(0).sum())} 行异常/缺失复权记录，未补造价格。

- 动量使用过去 63/126 日，风险调整版本除以过去 63 日波动（8% 下限）；至少 127 个真实行情观测，最近 126 日至少 120 日有数据，20 日平均成交额至少 2000 万。
- 要求 126 日动量为正、在 126 日均线上方、相对沪深 300 的平均动量为正；每周首个交易日收盘检查，次日开盘执行。
- 按跟踪指数去重、再按语义分组；已有合格代表优先保留，组内其他代表按同期流动性/强度选择；持有前五名缓冲，最多两组，过去 126 日相关系数低于 0.85。
- 全部 {s["replay_checks"]} 次结果逐笔重放检查现金和持仓；纯核心与上一次佣金研究逐日净值对齐；真实数据截断至 2022 年重算信号一致。
- 外部引擎只核对最近 40 个共同交易日，具体通过状态为 {json.dumps({k: v.get("verified") for k, v in s["native_checks"].items()}, ensure_ascii=False)}。这不等于外部引擎认证整个历史或上线验证。
- 上述为主资金 20 万的引擎检查。额外 100 万检查中，行业候选遇到开盘跳涨和现金不足：本地账本按可买整手部分成交，AKQuant 拒绝整笔买入、后续检查日才重新下单，因此未通过一致性检查。详见 native_checks_1000000.json 和 native_fill_differences_1000000.csv；不能把本轮结果称为全资金规模通过外部引擎验证。
- 原始分页参数、源响应与哈希保存在数据快照；本轮协议、代码快照、元数据哈希、净值、成交、仓位、事件和复核证据均可追溯。

当前元数据没有认证的历史版本，名称/分类变更、数据源未收录基金和未知类别可能带来偏差；虽包含退市记录，仍不能称为消除了幸存者偏差。无上市前指数回填。缺失复权或异常行情隔离；停牌/无成交/一字价格不成交。退市持仓按终止日归零保守处理，缺失终止日期时用最后行情后一天推断并留痕，不作为提前卖出的信号。分红按复权因子份额再投资近似处理；没有历史 QDII 溢价门槛。当前佣金按全历史恒定计算。

首轮因核心资产按代码重新排序、现金受限时下单优先级发生变化，未通过纯核心对账，已标记作废。修复为现行配置顺序后全量重跑，训练段仍选择 risk10；选股规则与参数未改变。正确结果的纯核心已与此前研究逐日对齐。

产物目录：`{out}`。
"""
    (out / "report.md").write_text(text)
    (ROOT / "docs/SECTOR_RESULTS.md").write_text(text)
