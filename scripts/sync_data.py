import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steadyquant.config import load_config
from steadyquant.data import sync

if __name__ == "__main__":
    result = sync(load_config(), full="--full" in sys.argv)
    raise SystemExit(0 if result["ok"] else 1)
