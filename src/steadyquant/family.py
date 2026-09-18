"""Account and display rules for the simple desktop edition."""
from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

from .config import fingerprint, write_json
from .daily import validate_account
from .data import TZ


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_account(root: Path, cfg: dict, cash: float, positions: dict, as_of: str) -> dict:
    if date.fromisoformat(as_of) > datetime.now(TZ).date():
        raise ValueError("持仓日期不能晚于今天")
    if any(not math.isfinite(v) or v < 0 or int(v) != v for v in positions.values()):
        raise ValueError("持仓份额必须是非负整数")
    account = dict(confirmed=True, cash=float(cash), positions=positions, as_of=as_of,
                   source="家庭版：用户录入并确认真实持仓及现金",
                   updated_at=datetime.now(TZ).isoformat())
    validate_account(account, cfg)
    stamp = datetime.now(TZ).strftime("%Y%m%dT%H%M%S%f")
    old = read_json(root / "data/portfolio.json")
    if old is not None:
        write_json(root / f"data/account_history/{stamp}-before.json", old)
    write_json(root / "data/portfolio.json", account)
    write_json(root / f"data/account_history/{stamp}.json", account)
    # A saved balance changes the input. Old orders must not stay visible as current advice.
    write_json(root / "outputs/family_report.json", {"invalidated": True})
    return account


def advice_is_current(report: dict, account: dict | None, signal: str, cfg: dict) -> bool:
    return bool(report and not report.get("invalidated")
                and report.get("signal_date") == signal
                and report.get("account_fingerprint") == fingerprint(account)
                and report.get("config_hash") == fingerprint(cfg))


def holdings_rows(cfg: dict, account: dict | None, report: dict | None = None) -> list[dict]:
    orders = {o["symbol"]: o for o in (report or {}).get("orders", [])}
    result = []
    for asset in cfg["assets"]:
        symbol = asset["symbol"]
        quantity = account.get("positions", {}).get(symbol, 0) if account else None
        order = orders.get(symbol)
        delta = order["quantity"] * (1 if order["side"] == "BUY" else -1) if order else 0
        row = {"ETF代码": symbol.split(".")[0], "名称": asset["name"],
               "当前持仓（份）": quantity}
        if report is not None:
            row.update({"建议操作": ("买入" if delta > 0 else "卖出") if delta else "不调整",
                        "买卖数量（份）": abs(delta),
                        "调整后（份）": quantity + delta if quantity is not None else None})
        result.append(row)
    return result
