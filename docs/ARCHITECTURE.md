# 设计与开源框架选择

## 框架选择：2026-09-10核实

选择 [AKQuant](https://github.com/akfamily/akquant)，锁定[0.3.55发布版](https://github.com/akfamily/akquant/releases/tag/v0.3.55)，官方发布时间2026-09-03；MIT许可证、macOS ARM64 wheel。主分支当时标为0.3.58，但没有直接追随未锁定的主分支。

也查看了 [RQAlpha](https://github.com/ricequant/rqalpha)、[backtesting.py](https://github.com/kernc/backtesting.py/blob/master/CHANGELOG.md)、[CZSC](https://github.com/waditu/czsc/releases) 和几个新建的 Tushare 投研项目。AKQuant 的多资产事件撮合、Python入口和表达式因子引擎适合本地轻量研究；无需安装完整机器学习/LLM技术栈。backtesting.py偏单标的；RQAlpha需要更深入的数据源/中国市场接入；CZSC近期有较大的Rust迁移与候选版本变化。

**准确的适配关系**：未把 AKQuant 说成内置了现成的 Tushare 连接器。它能接受 DataFrame/Parquet，项目新增 Tushare 适配层，将标准化数据接入其 `FactorEngine` 和原生 `run_backtest`。上游通过版本锁定依赖使用，没有复制一整套上游仓库再形成难以更新的分叉。

## 数据流

```text
.env → TushareProvider → 原始快照 / 隔离异常
                      → 标准化Parquet + SHA256 → DuckDB覆盖查询
                      → AKQuant FactorEngine → 固定规则每日目标权重
                          ├→ 原始价格资金账本 → 回测/前推/敏感性/报告
                          ├→ AKQuant NextOpen → 独立短窗口核对
                          └→ 已确认真实账户 → 每日调仓参考 → 本机通知
```

## 代码边界

| 文件 | 责任 |
|---|---|
| `data.py` | Tushare SDK实例配置、限速重试、年度分区、价格校验、Parquet缓存、快照校验 |
| `factors.py` | AKQuant表达式、无前视校验、历史复权价格、标签和非重叠IC诊断 |
| `strategy.py` | 固定资产桶预算、趋势预算缩减、历史波动约束、调仓日判断 |
| `backtest.py` | 原始价格、份额、现金、费用、次日成交、分红再投资近似、逐笔账本 |
| `framework.py` | AKQuant多资产策略适配、NextOpen、独立原生引擎核对 |
| `metrics.py` | CAGR、夏普、回撤、年度/压力期、区块重采样 |
| `research.py` | 先保存协议、后执行完整研究；不可变运行目录、哈希与证据 |
| `daily.py` | 锁、日期新鲜度、真实账户校验、整手参考清单、本机幂等通知 |
| `reports.py` | 自包含离线HTML、每日与研究报告 |
| `app.py` | 本地可视化工作台，使用相同核心逻辑 |

## 为什么保留独立账本

需要显式区分原始可成交价格、信号复权价格和ETF总回报近似。账本保存每笔现金变化和再投资事件，可核查未来函数、交易成本和缺失行情处理。AKQuant 原生撮合以相同价格和成本在没有公司行动的共同窗口进行逐日净值核对。40日一致性只证明该窗口/撮合口径，不能代替全部历史公司行动核对。

## 时间与审计约定

- `date` 是中国市场交易日，时间戳不代表日内真实撮合时刻；本机任务用 `Asia/Shanghai`。
- 复权价格乘以截至当日的因子，除以每个标的首条因子。历史前缀不使用未来截面分位数、全样本归一化或后来发布的财务数值。
- 复权缺失日不按未来已知因子向后填；缓存仍保留原始缺失记录和原因。
- 下一日开盘跳空不会改变前一日已生成的订单数量，只影响执行价格及现金约束后的成交量。
- 指标在净值路径上计算；年度切片包含前一年底最后净值，避免丢掉首个交易日收益。
- 回测CAGR用真实日历天数；波动/夏普按252个交易日年化，无风险收益2%，现金收益0%。
- 历史前推折不训练模型、不重新选参数；不声称有真正未见的前瞻测试集。

## 扩展路线

新增价格因子可直接复用表达式引擎。若加入财报因子，需要引入公告/可用时间并按当时已知信息做as-of join；不能把最新报表回填到过去。若加入个股策略，先补历史可交易股票池、退市/ST/停复牌、实际涨跌停、交易税和真实公司行动；若加入单债，先补票息、应计利息、净价/全价、到期和违约逻辑。实时自动交易需要另一个独立执行适配器，当前没有接入。

## 数据接口来源

- [Tushare ETF日线说明](https://tushare.pro/document/2?doc_id=127)：字段、手/千元单位、单次上限。
- [Tushare交易日历](https://tushare.pro/document/2?doc_id=26)。
- [AKQuant因子示例](https://github.com/akfamily/akquant/blob/v0.3.55/examples/19_factor_expression.py)。
- [AKQuant撮合模式源码](https://github.com/akfamily/akquant/blob/v0.3.55/python/akquant/backtest/fill_mode.py)。
