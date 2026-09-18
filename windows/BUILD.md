# Windows x64 便携版构建

使用官方 CPython 3.13.15 embeddable x64 包及 PyPI Windows wheels；可在 Mac 上组装，但必须区分跨平台打包检查与 Windows 原生验收。

1. 从 [Python 官方发布页](https://www.python.org/downloads/release/python-31315/) 下载 Windows embeddable package (64-bit)，保存到 `build/python-3.13.15-embed-amd64.zip`。构建脚本校验发布页的 SHA-256。
2. 在已有的开发环境中下载依赖（不安装进开发环境）：

```bash
.venv/bin/python -m pip --isolated download --index-url https://pypi.org/simple \
  --dest build/windows-wheels --platform win_amd64 --python-version 3.13 \
  --implementation cp --abi cp313 --only-binary=:all: -r windows/requirements.txt
.venv/bin/python scripts/build_windows.py \
  --runtime build/python-3.13.15-embed-amd64.zip --output dist/windows
```

构建脚本按 Windows 条件校验全部依赖及 extras，按白名单复制应用文件。不会包含 `.env`、真实账户、缓存、历史结果或本机配置。保持依赖内的 LICENSE / dist-info。`build-manifest.json` 记录依赖包与源码 SHA-256。重复构建请使用新的 output 目录，防止旧文件混入。

家庭版冻结配置是 `configs/family.yaml`，首次启动复制到使用者数据目录；以后不会覆盖现有配置。仅加载每日等风险策略，不打包 AKQuant 研究引擎；共享因子模块在实际运行研究计算时才导入 AKQuant。

## 验证

开发机运行 `pytest`、Streamlit AppTest 和 `windows/self_test.py`，覆盖账户保存、旧建议失效、更新失败、独立数据目录和策略生成。Windows 锁分支有模拟测试，不能代替操作系统实测。

Windows 10/11 x64 原生验收清单：

- 中文且包含空格的目录解压，双击 `检查运行环境.cmd`，结果通过。
- 双击 `启动ETF小助手.cmd`，自动打开浏览器；再次双击能复用运行中的窗口。
- 首次页没有其他人的持仓、现金或凭证；录入自己的配置。
- 记录资金与份额，重启后内容保留；填写现金需准确到实际余额。
- 更新行情失败时不展示旧买卖清单；完整下载成功后日期为最近完整收盘。
- 在符合时间条件的账户快照下，与同一配置的开发机输出逐项核对。
- 关闭启动窗口后重开、重新解压新软件包，账户仍保留。

本次开发机没有 Windows，manifest 中 `native_windows_validation` 保持 `pending`。软件不会自动运行检查、自动调仓或接入券商。
