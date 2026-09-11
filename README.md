# SteadyQuant · ETF 量化研究工作台

在本机运行的 Python / Streamlit 研究项目，使用 Tushare 兼容行情接口和 AKQuant，支持多资产 ETF 配置、因子实验、回测、成本分析与每日调仓参考。不连接券商，不自动下单。

本仓库发布源码、测试和研究参数示例，不包含行情、真实账户、凭证或本机回测结果。首次启动时，尚未运行的研究页面会显示准备提示；克隆仓库不会自动创建定时任务或启用真实账户策略。

## 安装与启动

需要 Python 3.12+。

```bash
git clone https://github.com/Windwalker1314/etf_quant.git
cd etf_quant
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
cp .env.example .env
```

在本机编辑 `.env`，填写自己的 Token。支持 `tushare_token` 和 `TUSHARE_TOKEN`。`TUSHARE_HTTP_URL` 指定数据服务地址，仅允许 HTTPS；示例使用的 Tushare 兼容服务不是 Tushare 官方站点，运行下载前应自行确认服务与凭证匹配。不要把 Token 粘贴到公开 issue、截图或提交中。

```bash
.venv/bin/sq app
```

浏览器打开 <http://127.0.0.1:8501>。macOS 也可双击 `start.command`；终端保持运行即可。服务只监听本机地址，并关闭 Streamlit 使用统计。

工作台包含组合总览、策略优化、自适应研究、个股综合研究、非量价因子研究、资金与佣金、行业卫星研究、策略验证、因子实验室、数据中心、每日调仓、账户与运行。

## 数据与基本回测

以下下载命令会使用已配置的数据服务及其配额，需要相应 API 权限。

```bash
.venv/bin/sq --help
.venv/bin/sq sync
.venv/bin/sq status
.venv/bin/sq backtest
```

行情与运行状态写入 `data/`，研究及每日简报写入 `outputs/`，均已忽略。公开仓库不会提供数据缓存；不同数据源、日期、权限及数据修订可能影响复现结果。

`configs/steady.yaml` 是基础研究配置；`configs/adaptive_paper.yaml` 是自适应等风险配置示例。日常命令优先读取本机 `configs/active.yaml`，该文件不存在时回退到基础配置。可按需在本机复制示例，再核对费用、资产池和风险参数：

```bash
cp configs/adaptive_paper.yaml configs/active.yaml
```

`initial_cash` 是研究本金。每日份额参考使用另行确认的真实账户快照，不把模拟本金当成真实资产，也不把调仓建议当作已经成交。

## 研究模块

- `sq study`：策略参数及分段比较。
- `sq macro-sync`、`sq adaptive`：宏观缓存与自适应配置研究。
- `sq stock-sync`、`sq composite`：个股与 ETF 组合研究。
- `sq alternative-sync`、`sq alternative`：非量价因子研究。
- `sq commission-study`：多本金、最低佣金与交易频率比较。
- `sq sector-study`：固定行业卫星候选与同预算宽基对照。

部分高级研究仍引用固定历史窗口和前序本地快照，不能在空仓库中直接获得完整结果。先阅读相应模块和脚本的数据依赖；不会为缺失的历史数据自动补造结果。`scripts/` 中的历史诊断与重算入口同样需要本机数据。

因子表达式示例：

```bash
.venv/bin/sq factor 'Close / Ref(Close, 120) - 1' --name momentum_120
```

## 每日参考

```bash
.venv/bin/sq daily
# macOS 可选本机通知
.venv/bin/sq daily --notify
```

在“账户与运行”页面录入并确认完整现金、持仓和日期，保存到被忽略的 `data/portfolio.json`。账户缺失或过期时，不生成可执行的份额参考。定时调度需自行配置；任何建议都不会自动执行交易。

## 验证与边界

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src scripts tests app.py
python3 scripts/check_public_source.py
```

测试主要用合成数据核对时间顺序、现金约束、费用、持仓和数据校验。历史回测需要独立生成并核验。回测采用收盘信号、下一交易日开盘模拟成交、整手和流动性限制；ETF 复权因子再投资近似不等于实际分红到账。历史数据版本、资产池选择、停牌与 QDII 溢价等限制不能被回测收益指标替代。

结构见 [架构说明](docs/ARCHITECTURE.md)，公开仓库文件范围见 [上传与隐私](docs/PUBLIC_REPOSITORY.md)。
