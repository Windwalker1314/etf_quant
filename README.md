# SteadyQuant · ETF 量化研究工作台

在本机运行的 Python / Streamlit 研究项目，使用 Tushare 兼容行情接口和 AKQuant，支持多资产 ETF 配置、因子实验、回测、成本分析与每日调仓参考。不连接券商，不自动下单。

本仓库发布源码、测试和研究参数示例，不包含行情、真实账户、凭证或本机回测结果。首次启动时，尚未运行的研究页面会显示准备提示；克隆仓库不会自动创建定时任务或启用真实账户策略。

## 安装与启动

### Windows 家庭版（简化界面）

家庭版使用同一套等风险月初策略，只保留今日建议、我的持仓、首次设置。分发包内置运行环境，完整解压后双击 `启动ETF小助手.cmd`，不需要安装 Python 或 Git。适用于 Windows 10/11 x64；Windows 实机验证尚待完成。

首次由家人填写配套的行情地址、密钥、券商费用和使用者自己的现金与持仓。数据保存到 `%LOCALAPPDATA%\SteadyQuant`，与研究工作台独立。软件包不含密钥、账户或行情；不创建自动任务。保存账户后旧建议立即失效，更新失败也不会继续显示旧买卖清单。

详见 [家庭版说明](windows/先看这里.txt)。开发者构建方法见 [Windows 构建说明](windows/BUILD.md)。

### PocketBay 家庭网站

网站首次使用时，每位家人分别注册用户名和密码；注册时填写家庭邀请码（服务器环境变量 `STEADYQUANT_CLOUD_PASSWORD`，即旧版网站访问密码）。之后各自用用户名和密码登录。ETF 行情共用；每个策略有独立的历史回测，每位用户在每个策略下的现金、ETF 份额、持仓历史与买卖清单分别保存。当前只发布已在使用的“等风险月初”策略，因此登录后直接进入；发布第二个策略后才会出现选择页。

旧版共用持仓不会自动分配给新用户。首次登录后，请每人到“我的持仓”对照自己的券商账户重新填写；旧文件留在服务器上，不在新页面展示。用户密码只保存加盐哈希，数据库和账户数据都不进入源码包或 Git。页面刷新后需重新登录；目前没有自助找回密码功能。

网站策略清单在 `src/steadyquant/strategy_catalog.py`。新增可实际使用的策略时，在清单中登记稳定的策略 ID、名称、说明和已审定的配置文件；历史行情会按全部已发布策略的 ETF 并集更新，回测分别生成。现有策略继续读取原用户目录中的真实持仓与清单，后续策略写入该用户目录下的 `strategies/<策略ID>/`；回测指针则分别写入 `outputs/strategies/<策略ID>/`。同一个券商账户里的资金不能在不同策略里重复登记。

### 完整研究工作台

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
- `scripts/run_stock_daily_study.py`：独立的个股日频策略研究，使用已保存的沪深 300 历史成分、股票日线、复权因子、涨跌停价及公司行动数据。每天收盘评分，次日开盘模拟交易；结果写入被忽略的 `outputs/stock_daily/`，不影响家庭网站或真实持仓。
- `sq alternative-sync`、`sq alternative`：非量价因子研究。
- `sq commission-study`：多本金、最低佣金与交易频率比较。
- `sq sector-study`：固定行业卫星候选与同预算宽基对照。

部分高级研究仍引用固定历史窗口和前序本地快照，不能在空仓库中直接获得完整结果。先阅读相应模块和脚本的数据依赖；不会为缺失的历史数据自动补造结果。`scripts/` 中的历史诊断与重算入口同样需要本机数据。

个股日频研究的运行方式：

```bash
.venv/bin/python scripts/run_stock_daily_study.py
```

该入口要求本机已有完整的 `data/stocks/` 快照、交易日历和 `510300.SH` 基准行情；输出会分别列出全期、开发期、验证期和近期结果，以及同仓位基准、双倍交易成本和公司行动校验情况。现有个股快照是冻结研究数据，不能据此生成当日买卖建议；只有独立验证通过后才应考虑发布到家庭网站。

纯个股的后续实验与 ETF 配置分开：`scripts/stock_daily_factor_sweep.py --pure-core` 用沪深 300 历史成分股比较预先写定的因子与仓位规则；`scripts/stock500_research_sync.py` 可另建中证 500 历史快照，然后用 `scripts/stock_daily_factor_sweep.py --stock500 --pure-core` 按相同规则重跑。`scripts/stock_daily_pure_audit.py --universe csi300 --factor residual_momentum_lowvol --budget always_100` 会逐笔复核一个股票组合及双倍成本。风险资产只持有股票；510300 行情只用于市场信号及对照，不进入股票策略持仓。以上均需本机完整历史数据和接口权限，产物保存在被忽略的 `data/`、`outputs/`，不修改 ETF 策略或网站。

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
