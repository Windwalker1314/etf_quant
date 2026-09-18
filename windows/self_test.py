"""Offline smoke test for the packaged runtime; never touches a real account."""
import os
import tempfile
from pathlib import Path


def main():
    with tempfile.TemporaryDirectory(prefix="steadyquant-test-") as folder:
        os.environ["STEADYQUANT_HOME"] = folder
        from datetime import datetime

        import numpy as np
        import pandas as pd
        import streamlit
        import tushare

        from steadyquant.config import load_config
        from steadyquant.daily import job_lock, make_report
        from steadyquant.data import TZ, Cache
        from steadyquant.family import save_account

        cfg = load_config(Path(__file__).resolve().parents[1] / "configs/family.yaml")
        cfg["assets"] = cfg["assets"][:1]
        dates = pd.bdate_range("2024-01-01", periods=300)
        rng = np.random.default_rng(17)
        close = 10 * np.exp(rng.normal(0, .004, len(dates)).cumsum())
        frame = pd.DataFrame(dict(date=dates, symbol="510300.SH", open=close, high=close,
                                  low=close, close=close, volume=1e8, amount=1e9, adj_factor=1.0))
        cache = Cache(Path(folder) / "data")
        cache.save("510300.SH", frame)
        pd.DataFrame({"cal_date": [dates[-1].strftime("%Y%m%d"),
                                   (dates[-1] + pd.offsets.BDay()).strftime("%Y%m%d")],
                      "is_open": [1, 1]}).to_parquet(cache.root / "calendar.parquet")
        account = save_account(Path(folder), cfg, 100000, {}, str(dates[-1].date()))
        now = datetime.combine(dates[-1].date(), datetime.min.time()).replace(hour=19, tzinfo=TZ)
        with job_lock(Path(folder)):
            report = make_report(cfg, cache, now, account)
        assert report["actionable"] and report["orders"][0]["quantity"] > 0
        print(f"检查通过：Python、行情组件、中文文件、日期、数据存储、文件锁和策略计算。\nStreamlit {streamlit.__version__} / Tushare {tushare.__version__}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"检查未通过（{type(exc).__name__}）。请联系家人核对软件包与系统版本。")
        raise SystemExit(1) from None
