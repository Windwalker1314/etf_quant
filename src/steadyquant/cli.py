from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import ROOT, load_active_config, load_config, write_json
from .data import Cache, DataError, TushareProvider, sync


def main():
    parser = argparse.ArgumentParser(description="SteadyQuant · 本地量化研究工作台")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sync", help="同步固定资产池，保留最近40天重叠校验")
    p.add_argument("--full", action="store_true")
    p = sub.add_parser("fetch", help="额外标的行情缓存（个股/ETF/指数研究）")
    p.add_argument("symbol")
    p.add_argument("--kind", choices=["fund", "stock", "index"], default="stock")
    p.add_argument("--start", default="20130101")
    p.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    p = sub.add_parser("backtest", help="回测、因子诊断、敏感性、原生引擎核对")
    p.add_argument("--quick", action="store_true", help="跳过敏感性组，仅用于开发验证")
    p = sub.add_parser("optimize", help="固定8组策略实验、历史切分、向前选择和压力验证")
    p.add_argument("--max-drawdown", type=float, default=0.15)
    sub.add_parser("refine", help="基于上一轮结果检验分散底仓与低换手增强")
    sub.add_parser("adaptive", help="自适应风险预算、历史估值及利率的固定8组研究")
    sub.add_parser("adaptive-audit", help="自适应配置的归因、成本、执行与稳定性诊断")
    sub.add_parser("stock-sync", help="缓存个股综合研究的历史成分、行情、估值、财报和公司行动")
    sub.add_parser("composite", help="个股与ETF综合策略：固定六模型、成交约束、压力与稳定性验证")
    sub.add_parser("alternative-sync", help="缓存本轮非量价研究的预测、财报、预告与历史行业数据")
    sub.add_parser("alternative", help="固定非量价因子、分组检验、11组组合对照与独立成交复核")
    sub.add_parser("commission-study", help="10万/20万/100万：真实费率、六组策略与独立成交复核")
    sub.add_parser("sector-study", help="固定四组行业ETF卫星与匹配宽基对照研究")
    p = sub.add_parser("macro-sync", help="缓存历史指数估值与Shibor，截止指定日或最近行情日")
    p.add_argument("--end", help="YYYYMMDD；省略时取沪深300ETF缓存最后交易日")
    sub.add_parser("study", help="重建完整研究：首版基准、16只ETF模型、底仓增强、消融和宽基扩展")
    p = sub.add_parser("daily", help="更新行情并生成每日简报")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--notify", action="store_true")
    p.add_argument("--scheduled", action="store_true", help="仅18:30后且当天为交易日时推送")
    sub.add_parser("status", help="缓存状态")
    p = sub.add_parser("factor", help="运行自定义 AKQuant 因子表达式")
    p.add_argument("expression")
    p.add_argument("--name", default="custom")
    p.add_argument("--symbols", help="逗号分隔的已缓存代码；省略时使用策略资产池")
    sub.add_parser("app", help="启动仅本机可访问的研究工作台")
    args = parser.parse_args()
    cfg = (
        load_active_config()
        if args.config is None and args.command in {"daily", "sync"}
        else load_config(args.config)
    )
    try:
        if args.command == "sync":
            raise SystemExit(0 if sync(cfg, full=args.full)["ok"] else 1)
        if args.command == "status":
            print(Cache().coverage().to_string(index=False))
        elif args.command == "fetch":
            data = TushareProvider().bars(args.symbol, args.kind, args.start, args.end)
            if data.empty:
                raise DataError("No data returned")
            Cache().save(args.symbol, data)
            print(f"{args.symbol}: {len(data)} rows")
        elif args.command == "backtest":
            from .research import run_research

            run_research(cfg, stress=not args.quick)
        elif args.command == "optimize":
            from .optimize import run_optimization

            research_cfg = cfg if args.config else load_config(ROOT / "configs/rotation.yaml")
            if not 0 < args.max_drawdown <= 0.5:
                raise ValueError("Drawdown research limit must be between 0 and 0.5")
            run_optimization(research_cfg, args.max_drawdown)
        elif args.command == "refine":
            from .refinement import run_refinement

            run_refinement()
        elif args.command == "adaptive":
            from .adaptive_audit import audit_adaptive
            from .adaptive_paper import review_paper
            from .adaptive_report import render_adaptive_report
            from .adaptive_study import run_adaptive_study

            folder = run_adaptive_study()
            audit_adaptive(folder)
            review_paper()
            print(render_adaptive_report())
        elif args.command == "stock-sync":
            from .stock_data import sync_stocks

            print(sync_stocks())
        elif args.command == "composite":
            from .composite_report import render_composite
            from .composite_study import run_composite

            print(render_composite(run_composite()))
        elif args.command == "adaptive-audit":
            from .adaptive_audit import audit_adaptive

            audit_adaptive()
        elif args.command == "alternative-sync":
            from .alternative_data import sync as sync_alternative

            sync_alternative()
        elif args.command == "alternative":
            from .alternative_study import run_alternative

            print(run_alternative())
        elif args.command == "sector-study":
            from .sector_study import run_sector_study

            print(run_sector_study())
        elif args.command == "commission-study":
            from .commission_study import run_commission_study

            print(run_commission_study())
        elif args.command == "macro-sync":
            from .macro import sync_macro

            end = args.end
            if end is None:
                df = Cache().read("510300.SH")
                if df.empty:
                    raise DataError("Sync price history first or specify --end YYYYMMDD")
                end = df.date.max().strftime("%Y%m%d")
            if len(end) != 8 or not end.isdigit():
                raise ValueError("Macro end date must be YYYYMMDD")
            sync_macro(end)
        elif args.command == "study":
            from .broad_core import run_broad_core
            from .candidate_review import review_simple_candidate
            from .optimize import run_optimization
            from .refinement import run_refinement
            from .research import run_research

            run_research(load_config())
            first = run_optimization(load_config(ROOT / "configs/rotation.yaml"), 0.15)
            second = run_refinement(first)
            review_simple_candidate(second)
            run_broad_core()
        elif args.command == "daily":
            from .daily import daily

            result = daily(cfg, refresh=not args.no_sync, notify=args.notify, scheduled_run=args.scheduled)
            if result["status"].startswith("暂停"):
                raise SystemExit(1)
        elif args.command == "factor":
            from .factors import compute, evaluate

            data = (
                {s: Cache().read(s) for s in args.symbols.split(",")} if args.symbols else Cache().load(cfg)
            )
            if any(df.empty for df in data.values()):
                raise DataError("One or more selected symbols have no cached data")
            result = compute(data, {args.name: args.expression})
            summary, _ = evaluate(data, result)
            output = ROOT / "outputs/factors"
            output.mkdir(parents=True, exist_ok=True)
            # Fixed filenames avoid treating user-controlled factor names as paths.
            result.to_parquet(output / "custom.parquet", index=False)
            write_json(output / "custom.json", {"name": args.name, "expression": args.expression})
            print(summary.to_string(index=False))
            print(output / "custom.parquet")
        elif args.command == "app":
            import subprocess
            import sys

            raise SystemExit(
                subprocess.call(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        str(ROOT / "app.py"),
                        "--server.address",
                        "127.0.0.1",
                        "--server.port",
                        "8501",
                        "--server.headless",
                        "true",
                        "--browser.gatherUsageStats",
                        "false",
                    ]
                )
            )
    except DataError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
