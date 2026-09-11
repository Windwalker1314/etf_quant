"""Bounded, fully retained strategy tournament and chronological forward selector."""

from __future__ import annotations

import copy
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yaml

from .backtest import EXECUTION_VERSION, simulate
from .config import ROOT, fingerprint, load_config, write_json
from .data import TZ, Cache
from .framework import engine_comparison
from .metrics import bootstrap_sharpe, performance, window_performance, yearly
from .rotation import build_features, ridge_predictions, rotation_weights
from .strategy import target_weights

STRESS = {
    "2015 股灾": ("2015-06-01", "2015-09-30"),
    "2018 熊市": ("2018-01-01", "2018-12-31"),
    "2020 疫情": ("2020-01-01", "2020-04-30"),
    "2022 股债压力": ("2022-01-01", "2022-12-31"),
    "2024 年初": ("2024-01-01", "2024-02-29"),
}


def variants(cfg: dict) -> dict[str, dict]:
    return {
        f"{family}_v{int(vol * 100)}": {**copy.deepcopy(cfg), "rotation_family": family, "target_vol": vol}
        for family in ("dual", "quality", "ensemble", "ridge")
        for vol in (0.12, 0.16)
    }


def choose(equities: pd.DataFrame, start: str, end: str, rf: float, max_dd: float) -> tuple[str, list]:
    """Selection only consumes returns <= end. Stable tie order from protocol."""
    records = []
    for name in equities:
        s = window_performance(equities[name], start, end, rf)
        valid = s.get("observations", 0) >= 500 and s.get("max_drawdown", -1) >= -max_dd
        score = (s.get("sharpe") or -99) + 0.25 * min(3, s.get("calmar") or -99)
        records.append(dict(candidate=name, passed=valid, score=score, **s))
    passing = [r for r in records if r["passed"]]
    # If no model clears the risk gate, use the predeclared conservative ensemble, never silently relax the gate.
    selected = max(passing, key=lambda r: r["score"])["candidate"] if passing else "ensemble_v12"
    return selected, records


def run_optimization(cfg: dict, max_drawdown: float = 0.15) -> Path:
    cache = Cache()
    data = cache.load(cfg)
    candidates = variants(cfg)
    run_id = datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + "-" + fingerprint([cfg, max_drawdown])[:6]
    output = ROOT / "outputs/optimization" / run_id
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "execution_version": EXECUTION_VERSION,
        "created_at": datetime.now(TZ).isoformat(),
        "config": cfg,
        "data_sha256": cache.snapshot_digest,
        "return_target": [0.15, 0.20],
        "drawdown_research_limit": max_drawdown,
        "candidates": candidates,
        "number_of_candidates": len(candidates),
        "selection": "2015-2022 net-of-cost Sharpe + 0.25 * min(Calmar, 3); >=500 observations and drawdown gate; fallback ensemble_v12",
        "test": "2023 onwards: retrospective diagnostic only, excluded from candidate selection; prior v1 results already viewed",
        "forward_selector": "each January from 2018 chooses using previous three years only; follows selected targets through December",
        "ridge": "monthly rolling 756-session training, sample every21 sessions, next-open to open+22 label, endpoint<=fit close, penalty10, >=150 samples; 25% score blend",
        "execution": "signal close -> next local session open; weekly; unlevered long only; lot100; 3bps fee/min5 +5bps slip; cash yield0; simulated capital1m",
        "universe": f"{len(cfg['assets'])} present-day representative ETF products, selected by market/asset class not historical returns; not a survivorship-free historical universe",
        "eligibility": "253 observed bars + trailing20-session amount>=10m; no prelisting/synthetic fund history",
        "boundaries": [
            "same existing total-return reinvestment approximation",
            "QDII traded-price premium is included but NAV/premium risk not independently modelled",
            "no leverage or short instruments; no guarantee of bear-market gains",
            "no automatic replacement of daily default",
        ],
        "diagnostics_not_selection": [
            "cost2x/3x",
            "weekday Monday/Wednesday",
            "top_n3/5",
            "risk12%/16%",
            "bootstrap",
            "bear windows",
            "native engine replay",
        ],
    }
    write_json(output / "protocol.json", protocol)
    print("Protocol frozen; calculating features and matured-label ridge...", flush=True)
    features = build_features(data, cfg)
    predictions, fit_audit = ridge_predictions(features)
    predictions.to_parquet(output / "ridge_predictions.parquet")
    fit_audit.to_json(output / "ridge_fits.json", orient="records", date_format="iso", indent=2)
    results, weights, equities = {}, {}, pd.DataFrame()
    records = []
    for name, variant in candidates.items():
        print(f"Running predeclared candidate {name}...", flush=True)
        w, factors = rotation_weights(data, variant, features, predictions)
        result = simulate(data, w, variant)
        weights[name], results[name] = w, result
        equities[name] = result.equity.equity
        folder = output / name
        folder.mkdir()
        w.to_parquet(folder / "targets.parquet")
        result.equity.to_parquet(folder / "equity.parquet")
        result.trades.to_parquet(folder / "trades.parquet", index=False)
        result.positions.to_parquet(folder / "positions.parquet")
        yearly(result.equity.equity, cfg["risk_free_rate"]).to_csv(folder / "yearly.csv", index=False)
        for period, start, end in [
            ("full", cfg["backtest_start"], None),
            ("selection", cfg["backtest_start"], "2022-12-31"),
            ("test", "2023-01-01", None),
        ]:
            records.append(
                dict(
                    candidate=name,
                    period=period,
                    **window_performance(result.equity.equity, start, end, cfg["risk_free_rate"]),
                )
            )
    factors.to_parquet(output / "factors.parquet", index=False)
    pd.DataFrame(records).to_csv(output / "candidates.csv", index=False)
    selected, selection_audit = choose(
        equities, cfg["backtest_start"], "2022-12-31", cfg["risk_free_rate"], max_drawdown
    )
    write_json(output / "selection.json", {"selected": selected, "audit": selection_audit})
    chosen_cfg = candidates[selected]
    (output / "candidate.yaml").write_text(yaml.safe_dump(chosen_cfg, allow_unicode=True, sort_keys=False))
    print(f"Selected using pre-2023 data only: {selected}. Checking walk-forward and stress...", flush=True)
    forward_weights = weights["ensemble_v12"].copy()
    fold_audit = []
    for year in range(2018, equities.index[-1].year + 1):
        chosen, audit = choose(
            equities, f"{year - 3}-01-01", f"{year - 1}-12-31", cfg["risk_free_rate"], max_drawdown
        )
        mask = forward_weights.index.year == year
        forward_weights.loc[mask] = weights[chosen].loc[mask]
        fold_audit.append(
            dict(year=year, selected=chosen, selection_through=f"{year - 1}-12-31", audit=audit)
        )
    write_json(output / "forward_selection.json", fold_audit)
    forward_weights.to_parquet(output / "forward_targets.parquet")
    forward = simulate(data, forward_weights, cfg, start="2018-01-01")
    forward.equity.to_parquet(output / "forward_equity.parquet")
    forward.trades.to_parquet(output / "forward_trades.parquet", index=False)
    old_cfg = load_config()
    old_data = {a["symbol"]: data[a["symbol"]] for a in old_cfg["assets"]}
    old_weights, _ = target_weights(old_data, old_cfg)
    old = simulate(old_data, old_weights, {**old_cfg, "backtest_start": cfg["backtest_start"]})
    broad_weights, _ = target_weights(data, cfg, baseline=True)
    broad = simulate(data, broad_weights, cfg)
    equity = results[selected].equity.equity
    compare = pd.concat(
        [
            equity.rename(selected),
            old.equity.equity.rename("v1"),
            broad.equity.equity.rename("expanded_fixed"),
            forward.equity.equity.rename("walk_forward"),
        ],
        axis=1,
    )
    compare.to_parquet(output / "comparison.parquet")
    sensitivity = []
    for multiple in (2, 3):
        run = simulate(data, weights[selected], chosen_cfg, cost_multiplier=multiple)
        sensitivity.append(
            dict(variant=f"cost_x{multiple}", **performance(run.equity.equity, cfg["risk_free_rate"]))
        )
    for key, values in [("rebalance_weekday", (0, 2)), ("top_n", (3, 5))]:
        for value in values:
            variant = {**chosen_cfg, key: value}
            w = (
                weights[selected]
                if key == "rebalance_weekday"
                else rotation_weights(data, variant, features, predictions)[0]
            )
            run = simulate(data, w, variant)
            sensitivity.append(
                dict(variant=f"{key}_{value}", **performance(run.equity.equity, cfg["risk_free_rate"]))
            )
    pd.DataFrame(sensitivity).to_csv(output / "sensitivity.csv", index=False)
    native, native_compare = engine_comparison(data, weights[selected], chosen_cfg)
    native_compare.to_csv(output / "native_comparison.csv")
    stress = {
        name: {
            label: window_performance(series.dropna(), start, end, cfg["risk_free_rate"])
            for label, series in compare.items()
        }
        for name, (start, end) in STRESS.items()
    }
    current = performance(equity, cfg["risk_free_rate"])
    retrospective = window_performance(equity, "2023-01-01", None, cfg["risk_free_rate"])
    selected_record = next(r for r in selection_audit if r["candidate"] == selected)
    summary = {
        "run_id": run_id,
        "selected": selected,
        "config_path": str(output / "candidate.yaml"),
        "strategy": current,
        "test_2023": retrospective,
        "v1_same_start": performance(old.equity.equity, cfg["risk_free_rate"]),
        "expanded_fixed": performance(broad.equity.equity, cfg["risk_free_rate"]),
        "walk_forward": performance(forward.equity.equity, cfg["risk_free_rate"]),
        "walk_forward_test": window_performance(
            forward.equity.equity, "2023-01-01", None, cfg["risk_free_rate"]
        ),
        "stress_windows": stress,
        "bootstrap_sharpe": bootstrap_sharpe(equity, cfg["risk_free_rate"]),
        "native_engine_check": native,
        "trades": len(results[selected].trades),
        "average_exposure": float(results[selected].equity.exposure.mean()),
        "full_target_15pct_reached": current["cagr"] >= 0.15,
        "test_target_15pct_reached": retrospective["cagr"] >= 0.15,
        "selection_drawdown_pass": selected_record["passed"],
        "full_drawdown_pass": current["max_drawdown"] >= -max_drawdown,
        "test_drawdown_pass": retrospective["max_drawdown"] >= -max_drawdown,
        "bear_calendar_years_nonnegative": all(
            stress[w][selected]["total_return"] >= 0 for w in ("2018 熊市", "2022 股债压力")
        ),
        "status": "research candidate; daily default remains v1; prospective performance unobserved",
    }
    write_json(output / "summary.json", summary)
    from .optimization_audit import audit_run

    summary["data_and_causality_audit"] = audit_run(output)
    write_json(output / "summary.json", summary)
    render_report(output, summary, compare, records, sensitivity)
    write_json(ROOT / "outputs/latest_optimization.json", {"path": str(output), "run_id": run_id})
    print(
        json.dumps(
            {"output": str(output), "selected": selected, "full": current, "test": retrospective},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return output


def render_report(output: Path, summary: dict, compare: pd.DataFrame, records: list, sensitivity: list):
    fig = go.Figure()
    for label, series in compare.items():
        series = series.dropna()
        fig.add_trace(go.Scatter(x=series.index, y=series / series.iloc[0], name=label))
    fig.update_layout(
        template="plotly_dark", title="净值比较（各自起点归一；向前选择从2018开始）", height=520
    )
    table = pd.DataFrame(records)[["candidate", "period", "cagr", "sharpe", "max_drawdown"]]
    selection_text = (
        "经回顾性消融评审优先观察" if summary.get("simplicity_review") else "按2022年底之前的收益与风险选择"
    )
    stress = [
        {"window": w, "candidate": c, **v}
        for w, models in summary["stress_windows"].items()
        for c, v in models.items()
    ]
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SteadyQuant 策略优化</title>
    <style>body{{font:16px/1.7 system-ui;background:#0b1019;color:#dce7f6;max-width:1400px;margin:36px auto;padding:24px}}h1,h2{{color:#79d5c0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #293b4e;text-align:right}}pre{{white-space:pre-wrap}}a{{color:#79d5c0}}</style>
    <h1>SteadyQuant · 策略优化研究</h1><p>{selection_text}：<b>{html.escape(summary["selected"])}</b>。无杠杆，下一交易日开盘模拟成交。</p>
    <p>2015起年化 {summary["strategy"]["cagr"]:.2%}，夏普 {summary["strategy"]["sharpe"]:.2f}，最大回撤 {summary["strategy"]["max_drawdown"]:.2%}；2023起回顾性检验年化 {summary["test_2023"]["cagr"]:.2%}。</p>
    <p>风险优先：历史最大回撤尽量低于15%，收益目标可以降低。2023以后的行情并非真正未知数据。报告保留{table.candidate.nunique()}组结果，默认每日策略仍为v1；历史回撤不代表未来损失上限。</p>
    {fig.to_html(full_html=False, include_plotlyjs=True)}<h2>全部候选</h2>{table.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
    <h2>熊市与压力窗口</h2>{pd.DataFrame(stress)[["window", "candidate", "total_return", "max_drawdown"]].to_html(index=False, float_format=lambda x: f"{x:.4f}")}
    <h2>成本与邻域参数（不重新挑选）</h2>{pd.DataFrame(sensitivity)[["variant", "cagr", "sharpe", "max_drawdown"]].to_html(index=False, float_format=lambda x: f"{x:.4f}")}
    <h2>研究依据与限制</h2><p>多周期趋势与波动控制的研究依据：<a href="https://www.aqr.com/insights/research/journal-article/a-century-of-evidence-on-trend-following-investing">AQR 原始研究</a>。其多空期货策略与本策略的多头ETF不同，不能直接套用论文收益。</p>
    <p>现今基金池存在选择与存续偏差。缺失行情不成交；上市前无虚拟历史；复权变化按除权日再投资近似；QDII溢价和开盘流动性存在未充分建模的风险。现金收益按0计。</p>
    <h2>核验摘要</h2><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></html>"""
    (output / "report.html").write_text(body)
