"""Reproducible, research-only daily stock strategy backtest."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .composite_ledger import VERSION as LEDGER_VERSION
from .composite_ledger import simulate_composite
from .composite_study import independent_replay, quarantine_stock_placeholders
from .config import ROOT, fingerprint, load_config, write_json
from .data import TZ, Cache, DataError
from .metrics import window_performance, yearly
from .stock_daily_strategy import DailyStockPolicy, build_daily_targets
from .stock_data import load_stock_snapshot

SOURCE = Path(__file__).resolve().parents[2]
WINDOWS = {
    "full": ("2016-01-01", None),
    "development": ("2016-01-01", "2020-12-31"),
    "validation": ("2021-01-01", "2022-12-31"),
    "recent": ("2023-01-01", None),
}


def _benchmark_price(cache: Cache) -> pd.Series:
    bars = cache.read("510300.SH")
    if bars.empty or bars.duplicated("date").any():
        raise DataError("Missing or duplicate CSI 300 ETF benchmark bars")
    return (bars.set_index("date").close * bars.set_index("date").adj_factor).sort_index()


def run_stock_daily_study() -> Path:
    snapshot, manifest = load_stock_snapshot()
    root = ROOT / "outputs/stock_daily"
    run_id = datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + "-" + fingerprint(DailyStockPolicy())[:6]
    output = root / run_id
    output.mkdir(parents=True, exist_ok=False)
    snapshot, quarantined = quarantine_stock_placeholders(snapshot, output)
    calendar = pd.read_parquet(ROOT / "data/calendar.parquet")
    end = str(manifest["end"])
    dates = pd.DatetimeIndex(pd.to_datetime(
        calendar.loc[(calendar.is_open.astype(int) == 1)
                     & calendar.cal_date.astype(str).between("20140101", end), "cal_date"]
    )).sort_values().unique()
    if dates.empty or dates[-1].strftime("%Y%m%d") != end:
        raise DataError("Stock snapshot end is not a complete trading session")
    cache = Cache()
    benchmark = _benchmark_price(cache)
    policy = DailyStockPolicy()
    targets, signals = build_daily_targets(snapshot, dates, benchmark, policy)
    cfg = load_config(SOURCE / "configs/family.yaml")
    cfg.update(initial_cash=policy.initial_cash, rebalance_frequency="daily",
               rebalance_band=0.02, stock_rebalance_band=0.02,
               commission_bps=3.0, cash_rate=0.0)
    selected = targets.columns[(targets > 0).any(axis=0)]
    if len(selected) < policy.holdings:
        raise DataError("Daily strategy produced too few tradable stocks")
    targets = targets.loc[:, selected]
    result = simulate_composite({}, snapshot, targets, cfg, start="2016-01-01")
    stress = simulate_composite({}, snapshot, targets, cfg, start="2016-01-01", cost_multiplier=2)
    replay = independent_replay(result, {}, snapshot, cfg)
    if not replay["verified"]:
        raise DataError(f"Stock ledger replay mismatch: {replay}")
    held = result.equity.equity
    index = benchmark.reindex(held.index)
    if index.isna().any():
        raise DataError("Benchmark has gaps in simulation window")
    baseline = policy.initial_cash * index / index.iloc[0]
    prior_exposure = signals.exposure.reindex(held.index).shift(1).fillna(0.0)
    matched_returns = prior_exposure * index.pct_change(fill_method=None).fillna(0.0)
    matched_baseline = policy.initial_cash * (1.0 + matched_returns).cumprod()
    curves = pd.DataFrame({"stock_daily": held, "csi300_etf": baseline,
                           "csi300_exposure_matched": matched_baseline,
                           "stock_daily_cost_x2": stress.equity.equity})
    metrics = {
        name: {label: window_performance(series, *window, rf=cfg["risk_free_rate"])
               for label, window in WINDOWS.items()}
        for name, series in curves.items()
    }
    trades = result.trades
    events = result.events
    qualification_blocks = (int(events.type.eq("qualification_block").sum())
                            if not events.empty else 0)
    validation_worse = (metrics["stock_daily"]["validation"]["cagr"]
                        < metrics["csi300_exposure_matched"]["validation"]["cagr"])
    deployment_issues = ["Live-data freshness and forward validation have not been established"]
    if validation_worse:
        deployment_issues.append("Validation CAGR trails the exposure-matched benchmark")
    if qualification_blocks:
        deployment_issues.append(f"{qualification_blocks} unresolved corporate-action adjustments")
    summary = {
        "snapshot_end": end,
        "snapshot_manifest_sha256": hashlib.sha256(
            (ROOT / "data/stocks/manifest.json").read_bytes()).hexdigest(),
        "model": "12-1/6-1 momentum + low volatility rank, 10 stocks, rank-20 retention buffer, "
                 "CSI300 ETF 200-day regime 90%/30% stock budget, daily close decisions",
        "research_only": True,
        "deployable": False,
        "deployability_issues": deployment_issues,
        "ledger_version": LEDGER_VERSION,
        "independent_replay": replay,
        "config": cfg,
        "policy": policy.__dict__,
        "quarantined_placeholder_bars": quarantined,
        "symbols_ever_targeted": len(selected),
        "signal_days": len(signals),
        "trades": len(trades),
        "stock_trade_notional_cny": float(trades.notional.sum()) if not trades.empty else 0.0,
        "qualification_blocks": qualification_blocks,
        "metrics": metrics,
        "benchmark_note": "510300.SH adjusted close, no transaction cost; exposure-matched variant uses prior-close stock budget; stock ledger includes fees/slippage",
        "limitations": [
            "Historical CSI300 constituents are monthly snapshots and used only after their dated observation.",
            "The saved stock snapshot ends on 2026-09-09; this run does not provide current live orders.",
            "Vendor revisions and announced corporate-action fields are not a certified point-in-time archive.",
            "Daily bar volume caps do not prove executable auction fills.",
        ],
    }
    curves.to_parquet(output / "equity.parquet")
    signals.to_parquet(output / "signals.parquet")
    result.trades.to_parquet(output / "trades.parquet", index=False)
    result.events.to_parquet(output / "events.parquet", index=False)
    result.positions.to_parquet(output / "holdings.parquet")
    yearly(held, cfg["risk_free_rate"]).to_csv(output / "stock_yearly.csv", index=False)
    yearly(baseline, cfg["risk_free_rate"]).to_csv(output / "benchmark_yearly.csv", index=False)
    write_json(output / "summary.json", summary)
    lines = [
        "# 个股日频研究结果",
        "",
        f"数据截至 {end[:4]}-{end[4:6]}-{end[6:]}; 历史研究，不提供今日买卖清单。",
        "",
        "策略：历史沪深300主板成分，6—12个月动量与低波动排名；最多10只，",
        "200日趋势决定股票预算30%或90%；收盘决策、次日开盘模拟成交。",
        "",
        "| 模型 | 时段 | 年化 | 夏普 | 最大回撤 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    labels = {"stock_daily": "个股策略（扣费）", "csi300_etf": "沪深300ETF（无费用）",
              "csi300_exposure_matched": "沪深300ETF（同仓位、无费用）",
              "stock_daily_cost_x2": "个股策略（双倍交易成本）"}
    period_labels = {"full": "全期", "development": "开发期", "validation": "验证期",
                     "recent": "近期"}
    for name, windows in metrics.items():
        for period, values in windows.items():
            lines.append(f"| {labels[name]} | {period_labels[period]} | {values['cagr']:.2%} | "
                         f"{values['sharpe']:.2f} | {values['max_drawdown']:.2%} |")
    chinese_issues = ["实时数据未更新、尚无前瞻验证"]
    if validation_worse:
        chinese_issues.append("验证期年化收益落后于同仓位基准")
    if qualification_blocks:
        chinese_issues.append(f"{qualification_blocks} 次公司行动复权变化尚未解释")
    lines.extend([
        "",
        f"成交 {len(trades)} 笔；公司行动校验阻断 {qualification_blocks} 次；"
        f"独立账本重放最大误差 ¥{replay['max_abs_difference']:.6f}。",
        "",
        "结论：目前暂不适合上线。" + "；".join(chinese_issues) + "。",
        "",
        "明细见 summary.json、equity.parquet、signals.parquet、trades.parquet 和 events.parquet。",
        "",
    ])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "latest.json", {"path": str(output), "run_id": run_id})
    print(json.dumps({"output": str(output), "snapshot_end": end,
                      "full": metrics["stock_daily"]["full"],
                      "benchmark_full": metrics["csi300_etf"]["full"],
                      "qualification_blocks": qualification_blocks}, ensure_ascii=False))
    return output
