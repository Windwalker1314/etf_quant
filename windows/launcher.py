"""Portable Windows entry point. No installer, shell commands, or administrator rights."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]


def prepare_home(home: Path):
    (home / "configs").mkdir(parents=True, exist_ok=True)
    for name in ("active.yaml", "steady.yaml"):
        target = home / "configs" / name
        if not target.exists():
            shutil.copyfile(BUNDLE / "configs/family.yaml", target)
    (home / "logs").mkdir(exist_ok=True)


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as r:
            return r.status == 200 and r.read().strip() == b"ok"
    except (OSError, ValueError):
        return False


def main():
    if sys.platform != "win32":
        raise SystemExit("此启动器供 Windows 10/11 64 位电脑使用。")
    import msvcrt

    home = Path(os.environ["LOCALAPPDATA"]) / "SteadyQuant"
    prepare_home(home)
    state = home / "running.json"
    lock = (home / "launcher.lock").open("a+b")
    if lock.seek(0, 2) == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        for _ in range(20):
            try:
                port = int(json.loads(state.read_text(encoding="utf-8"))["port"])
                if healthy(port):
                    webbrowser.open(f"http://127.0.0.1:{port}")
                    return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1)
        raise SystemExit("软件正在启动，请等待片刻；若一直没打开，请关闭原启动窗口再重试。")

    child = None
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        state.write_text(json.dumps({"port": port}), encoding="utf-8")
        env = dict(os.environ, STEADYQUANT_HOME=str(home), PYTHONUTF8="1")
        args = [sys.executable, "-X", "utf8", "-m", "streamlit", "run",
                str(BUNDLE / "windows/family_app.py"), "--server.address=127.0.0.1",
                f"--server.port={port}", "--server.headless=true",
                "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
        print("正在启动 ETF 小助手，请稍候…", flush=True)
        with (home / "logs/startup.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(args, cwd=home, env=env, stdout=log, stderr=log)
            for _ in range(90):
                if child.poll() is not None:
                    raise RuntimeError("启动失败，请把首次启动错误界面交给家人处理。日志位于数据目录 logs/startup.log。")
                if healthy(port):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("启动超时，请关闭窗口后重试。")
            url = f"http://127.0.0.1:{port}"
            webbrowser.open(url)
            print(f"已打开。浏览器没有出现时，请手动打开：{url}\n使用期间保留此窗口；退出按 Ctrl+C。", flush=True)
            child.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        state.unlink(missing_ok=True)
        lock.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("启动失败。请联系家人，检查是否完整解压、电脑是否为 Windows 10/11 64 位。", flush=True)
        sys.exit(1)
