# 上传与隐私

仓库只存放源码、合成测试、研究参数示例、依赖清单和通用说明。

以下内容保留在本机：

- `.env`、密钥、凭证和 Streamlit secrets。
- `data/`：行情、原始响应、账户、交易日历、同步及通知状态。
- `outputs/`、日志、数据库、表格、压缩包和回测产物。
- `configs/active.yaml`：本机实际启用配置。
- 本机研究结果文档和运行记录；原有说明备份为被忽略的 `docs/LOCAL_README.md`。
- 虚拟环境、缓存及本地编辑器配置。

研究配置中的模拟本金和策略参数属于示例，不是账户快照。`.env.example` 只有占位符，不含可用 Token。

`.gitignore` 采用顶层允许列表，并对常见凭证与数据文件增加递归排除规则。新增顶层目录需明确加入允许列表。不要使用 `git add -f` 强制添加私人内容。

提交前检查：

```bash
git status --short
git diff --cached --stat
python3 scripts/check_public_source.py
```

检查脚本只输出文件名和命中类型，不打印秘密值；检查暂存区中的敏感路径、常见密钥格式和本机 `.env` 中的凭证值。它是辅助检查，不能保证识别所有秘密。`.gitignore` 不会删除既有 Git 历史中的敏感内容。
