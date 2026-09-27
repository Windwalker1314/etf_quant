"""PocketBay web entry: persistent data under /data; secrets remain runtime mounts."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

SOURCE = Path(__file__).resolve().parents[1]


def prepare_environment():
    data_dir = os.environ.get("POCKETBAY_DATA_DIR")
    if not data_dir:
        raise RuntimeError("POCKETBAY_DATA_DIR is required for persistent cloud storage")
    home = Path(data_dir) / "steadyquant"
    (home / "configs").mkdir(parents=True, exist_ok=True)
    for name in (".env", ".pocketbay-cloud.env"):
        path = SOURCE / name
        if path.exists():
            load_dotenv(path, override=False)
    active = home / "configs/active.yaml"
    if not active.exists():
        source_config = SOURCE / "configs/active.yaml"
        if not source_config.exists():
            source_config = SOURCE / "configs/family.yaml"
        shutil.copyfile(source_config, active)
    baseline = home / "configs/steady.yaml"
    if not baseline.exists():
        shutil.copyfile(SOURCE / "configs/steady.yaml", baseline)
    os.environ["STEADYQUANT_HOME"] = str(home)
    os.environ["PYTHONPATH"] = str(SOURCE / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    if str(SOURCE / "src") not in sys.path:
        sys.path.insert(0, str(SOURCE / "src"))


def prepare():
    prepare_environment()
    port = int(os.environ["PORT"])
    if not 1 <= port <= 65535:
        raise ValueError("Invalid PORT")
    return [sys.executable, "-m", "streamlit", "run", str(SOURCE / "app.py"),
            "--server.address=0.0.0.0", f"--server.port={port}",
            "--server.headless=true", "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none"]


if __name__ == "__main__":
    subprocess.run(prepare(), cwd=SOURCE, check=True)
