from __future__ import annotations

import fcntl
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, fingerprint, write_json
from .data import TZ, Cache, DataError, sync
from .execution import plan_cash_orders
from .reports import daily_html
from .strategy import rebalance_needed, risk_exit_due, scheduled, target_weights


@contextmanager
def job_lock(root: Path = ROOT):
    path = root / "data/daily.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DataError("A daily job is already running") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def calendar_dates(cache: Cache, now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(TZ)
    path = cache.root / "calendar.parquet"
    if not path.exists():
        raise DataError("Trading calendar missing")
    cal = pd.read_parquet(path)
    cutoff = now.date() if now.hour >= 18 else now.date() - timedelta(days=1)
    open_days = sorted(cal.loc[cal.is_open.astype(int) == 1, "cal_date"].astype(str))
    past = [d for d in open_days if d <= cutoff.strftime("%Y%m%d")]
    future = [d for d in open_days if d > cutoff.strftime("%Y%m%d")]
    if not past or not future:
        raise DataError("Trading calendar expired; sync before generating advice")
    return past[-1], future[0]


def load_account(cfg: dict, path: Path | None = None) -> dict | None:
    path = path or ROOT / "data/portfolio.json"
    if not path.exists():
        return None
    account = json.loads(path.read_text())
    return validate_account(account, cfg)


def validate_account(account: dict, cfg: dict) -> dict | None:
    if not account.get("confirmed"):
        return None
    symbols = {a["symbol"] for a in cfg["assets"]}
    if set(account.get("positions", {})) - symbols:
        raise DataError(
            "Account contains unsupported positions; include all holdings in the configured universe"
        )
    values = [account.get("cash", -1), *account.get("positions", {}).values()]
    if not all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in values):
        raise DataError("Invalid account cash/positions")
    if not account.get("as_of"):
        raise DataError("Account as_of date is required")
    return account


def make_report(
    cfg: dict, cache: Cache | None = None, now: datetime | None = None, account: dict | None = None
) -> dict:
    cache = cache or Cache()
    signal, execution = calendar_dates(cache, now)
    report = {
        "signal_date": str(pd.Timestamp(signal).date()),
        "execution_date": str(pd.Timestamp(execution).date()),
        "created_at": (now or datetime.now(TZ)).isoformat(),
        "status": "",
        "notes": [],
        "config_hash": fingerprint(cfg),
        "allocations": [],
        "orders": [],
        "actionable": False,
    }
    data = cache.load(cfg)
    data = {s: df[df.date <= pd.Timestamp(signal)].copy() for s, df in data.items()}
    bad = [
        s
        for s, df in data.items()
        if df.empty or df.date.max() != pd.Timestamp(signal) or df.iloc[-1].volume <= 0
    ]
    if bad:
        report.update(
            status="暂停建议：数据不完整或未更新",
            notes=[f"未就绪标的：{', '.join(bad)}", "不会沿用过期数据生成订单。"],
        )
        return report
    targets, _ = target_weights(data, cfg)
    weights = targets.loc[pd.Timestamp(signal)]
    account = validate_account(account, cfg) if account is not None else load_account(cfg)
    nav, cash, positions = None, None, {}
    if account:
        if account["as_of"] != report["signal_date"]:
            report["notes"].append(
                f"持仓快照截至 {account['as_of']}；需确认至 {report['signal_date']} 收盘后才能生成份额清单。"
            )
            account = None
        else:
            cash, positions = float(account["cash"]), account.get("positions", {})
            nav = cash + sum(positions.get(s, 0) * df.iloc[-1].close for s, df in data.items())
            if nav <= 0:
                raise DataError("Account equity must be positive")
    requests = []
    prior_dates = targets.index[targets.index < pd.Timestamp(signal)]
    prior_session = prior_dates[-1] if len(prior_dates) else None
    on_schedule = scheduled(pd.Timestamp(signal), cfg, any(positions.values()), prior_session)
    for asset in cfg["assets"]:
        s = asset["symbol"]
        close = float(data[s].iloc[-1].close)
        current = positions.get(s, 0) * close / nav if nav else None
        target = float(weights[s])
        report["allocations"].append(
            {
                "symbol": s,
                "name": asset["name"],
                "target_weight": target,
                "current_weight": current,
                "difference": target - current if current is not None else None,
                "close": close,
            }
        )
        emergency = bool(nav and risk_exit_due(target, current, cfg))
        if nav and (on_schedule or emergency) and (emergency or rebalance_needed(target, current, cfg)):
            desired = math.floor(nav * target / close / cfg["lot_size"]) * cfg["lot_size"]
            delta = desired - positions.get(s, 0)
            cap = (
                math.floor(data[s].iloc[-1].volume * cfg["participation_rate"] / cfg["lot_size"])
                * cfg["lot_size"]
            )
            quantity = math.floor(min(abs(delta), cap) / cfg["lot_size"]) * cfg["lot_size"]
            if delta < 0 and desired == 0:
                quantity = min(positions.get(s, 0), cap)
            if quantity:
                requests.append(
                    {
                        "symbol": s,
                        "side": "BUY" if delta > 0 else "SELL",
                        "quantity": quantity,
                        "reference_price": close,
                        "target_weight": target,
                        "cost_exempt": bool(emergency or (delta < 0 and desired == 0)),
                    }
                )
    report["allocations"].append(
        {
            "symbol": "CASH",
            "name": "现金",
            "target_weight": float(1 - weights.sum()),
            "current_weight": cash / nav if nav else None,
            "difference": None,
            "close": None,
        }
    )
    if not account:
        report["status"] = "目标配置已生成 · 尚无当日已确认持仓"
        report["notes"].append(
            "请在工作台录入真实现金与完整持仓。当前只展示权重，不使用回测持仓代替真实账户。"
        )
    elif not on_schedule and not requests:
        report["status"] = "观察日 · 保持持仓"
        frequency = (
            "每月首个交易日"
            if cfg.get("rebalance_frequency") == "monthly_first_session"
            else "每月第一个日历星期五（休市不补调）"
            if cfg.get("rebalance_frequency") == "monthly"
            else "每周五"
        )
        report["notes"].append(f"{frequency}收盘后判断调仓；首次建仓可在任一交易日生成建议。")
    else:
        plan = [{**req, "delta": req["quantity"] * (1 if req["side"] == "BUY" else -1)} for req in requests]
        prices = {req["symbol"]: req["reference_price"] for req in requests}
        for req in plan_cash_orders(plan, cash, prices, cfg):
            delta = req.pop("delta")
            req.pop("cost_exempt", None)
            report["orders"].append({**req, "quantity": abs(delta)})
        report["status"] = "调仓日 · 次日开盘参考清单" if report["orders"] else "调仓日 · 偏离未达到阈值"
        if not on_schedule and report["orders"]:
            report["status"] = "风险减仓 · 次日开盘参考清单"
        report["actionable"] = bool(report["orders"])
    report["notes"].append(
        "价格与数量以昨收估算；先卖后买，开盘需重核现金、停牌、涨跌停与 QDII 溢价。系统不连接券商、不下单。"
    )
    report["data_sha256"] = cache.snapshot_digest
    return report


def notify_local(report: dict, output: Path, root: Path = ROOT) -> dict:
    key = fingerprint(
        {k: report.get(k) for k in ("signal_date", "status", "orders", "config_hash", "data_sha256")}
    )
    sent = root / "data/notifications" / f"{key}.json"
    if sent.exists():
        return {"status": "duplicate_skipped", "key": key}
    if sys.platform != "darwin":
        return {"status": "unsupported_os", "report": str(output)}
    message = f"{report.get('signal_date', '')} · {report['status']}。报告：{output.name}"
    script = 'on run argv\ndisplay notification (item 1 of argv) with title "SteadyQuant 每日简报"\nend run'
    completed = subprocess.run(
        ["osascript", "-e", script, message], capture_output=True, text=True, timeout=15
    )
    if completed.returncode:
        return {"status": "failed", "reason": "macOS notification command failed"}
    delivery = {
        "status": "submitted_to_macos",
        "at": datetime.now(TZ).isoformat(),
        "report": str(output),
        "key": key,
    }
    write_json(sent, delivery)
    return delivery


def daily(cfg: dict, refresh: bool = True, notify: bool = False, scheduled_run: bool = False) -> dict:
    with job_lock():
        try:
            if scheduled_run and (datetime.now(TZ).hour, datetime.now(TZ).minute) < (18, 30):
                return {"status": "尚未到收盘后运行时间", "actionable": False, "orders": []}
            if refresh:
                status = sync(cfg)
                if not status["ok"]:
                    raise DataError("Refresh incomplete; see data/last_sync.json")
            if scheduled_run:
                signal, _ = calendar_dates(Cache())
                if signal != datetime.now(TZ).strftime("%Y%m%d"):
                    print("休市日，跳过简报推送", flush=True)
                    return {"status": "休市日", "actionable": False, "orders": []}
            report = make_report(cfg)
        except (DataError, ValueError) as exc:
            report = {
                "signal_date": datetime.now(TZ).date().isoformat(),
                "execution_date": "—",
                "status": "暂停建议：数据或账户校验失败",
                "notes": [str(exc)],
                "allocations": [],
                "orders": [],
                "actionable": False,
            }
        output = ROOT / "outputs/daily" / report["signal_date"]
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "report.json", report)
        write_json(output / f"snapshot-{fingerprint(report)}.json", report)
        (output / "report.html").write_text(daily_html(report))
        lines = [
            "# SteadyQuant 每日简报",
            "",
            f"信号日期：{report['signal_date']}",
            f"状态：{report['status']}",
            "",
            *report["notes"],
        ]
        for allocation in report.get("allocations", []):
            lines.append(f"- {allocation['name']} {allocation['symbol']}: {allocation['target_weight']:.2%}")
        (output / "report.md").write_text("\n".join(lines))
        write_json(ROOT / "outputs/latest_daily.json", {"path": str(output), "status": report["status"]})
        if notify:
            delivery = notify_local(report, output / "report.html")
            write_json(output / "notification.json", delivery)
            print(json.dumps(delivery, ensure_ascii=False))
        print(f"{report['status']} — {output / 'report.html'}", flush=True)
        return report
