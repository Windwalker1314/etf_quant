"""First-run cloud data preparation; never imports or uploads a local account."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import ROOT, write_json
from .strategy_catalog import load_strategy_config, market_sync_config, strategies

STATUS = ROOT / "data/cloud_setup.json"
LOG = ROOT / "data/cloud_setup.log"
SETUP_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_status() -> dict:
    if not STATUS.exists():
        return {}
    try:
        status = json.loads(STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"state": "failed", "message": "初始化状态文件无法读取"}
    if status.get("state") == "running" and not _pid_alive(status.get("pid")):
        return {"state": "failed", "message": "初始化进程已停止，可重试。"}
    if status.get("state") == "queued":
        try:
            started = datetime.fromisoformat(status["started_at"])
            if datetime.now(timezone.utc) - started > timedelta(minutes=2):
                return {"state": "failed", "message": "初始化未能启动，可重试。"}
        except (KeyError, ValueError):
            return {"state": "failed", "message": "初始化状态不完整，可重试。"}
    return status


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start() -> dict:
    if not os.environ.get("POCKETBAY_DATA_DIR"):
        raise RuntimeError("Cloud setup only runs on PocketBay")
    current = read_status()
    if current.get("state") == "queued":
        return current
    if current.get("state") == "running" and _pid_alive(current.get("pid")):
        return current
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    write_json(STATUS, {"state": "queued", "stage": "starting", "started_at": _now(), "version": SETUP_VERSION})
    try:
        with LOG.open("ab") as log:
            subprocess.Popen(
                [sys.executable, "-m", "steadyquant.cloud_setup"],
                cwd=Path(__file__).resolve().parents[2],
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError:
        write_json(STATUS, {"state": "failed", "message": "无法启动云端初始化进程", "version": SETUP_VERSION})
    return read_status()


def run() -> int:
    from .data import sync
    from .family_backtest import run_family_backtest

    pid = os.getpid()
    started_at = _now()
    try:
        entries = [(strategy, load_strategy_config(strategy)) for strategy in strategies()]
        write_json(STATUS, {"state": "running", "stage": "sync", "pid": pid, "started_at": started_at,
                            "version": SETUP_VERSION})
        result = sync(market_sync_config(entries))
        if not result["ok"]:
            write_json(
                STATUS,
                {
                    "state": "failed",
                    "stage": "sync",
                    "message": "部分 ETF 行情未更新成功，请稍后重试。",
                    "finished_at": _now(),
                    "started_at": started_at,
                    "version": SETUP_VERSION,
                },
            )
            return 1
        write_json(STATUS, {"state": "running", "stage": "backtest", "pid": pid, "started_at": started_at,
                            "version": SETUP_VERSION})
        outputs = {strategy.id: run_family_backtest(cfg, strategy_id=strategy.id).name
                   for strategy, cfg in entries}
        write_json(
            STATUS,
            {"state": "done", "stage": "backtest", "run_id": outputs[entries[0][0].id],
             "run_ids": outputs, "started_at": started_at,
             "finished_at": _now(), "version": SETUP_VERSION},
        )
        return 0
    except Exception as exc:
        stage = read_status().get("stage", "unknown")
        detail = ""
        if isinstance(exc, FileNotFoundError) and exc.filename:
            # A path is enough to diagnose missing deployment assets; never show provider responses.
            detail = f"（缺少 {Path(exc.filename).name}）"
        write_json(
            STATUS,
            {
                "state": "failed",
                "stage": stage,
                "message": f"{'行情更新' if stage == 'sync' else '历史回测'}失败：{type(exc).__name__}{detail}。请稍后重试。",
                "started_at": started_at,
                "finished_at": _now(),
                "version": SETUP_VERSION,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(run())
