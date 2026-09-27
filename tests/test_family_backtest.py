import json
from pathlib import Path

import numpy as np
import pandas as pd

from steadyquant import config, data, factors, family_backtest, strategy_catalog
from steadyquant.strategy_catalog import Strategy, research_pointer


def test_cloud_backtest_needs_no_factor_lab_config(tmp_path, monkeypatch):
    cfg = config.load_config(Path(__file__).resolve().parents[1] / "configs/family.yaml")
    cfg["backtest_start"] = "2018-01-01"
    dates = pd.bdate_range("2018-01-01", periods=400)
    for module in (config, data, factors, family_backtest):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    benchmark_dates = dates
    monkeypatch.setattr(family_backtest, "load_shanghai_composite",
                        lambda _cache, sessions: pd.Series(
                            3000.0 + np.arange(len(benchmark_dates)), index=benchmark_dates
                        ).reindex(sessions))
    monkeypatch.setattr(family_backtest, "benchmark_digest", lambda _cache: "test-index-digest")
    cache = data.Cache(tmp_path / "data")
    for i, asset in enumerate(cfg["assets"]):
        rng = np.random.default_rng(i + 1)
        close = 10 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(dates))))
        cache.save(asset["symbol"], pd.DataFrame({
            "date": dates, "symbol": asset["symbol"],
            "open": close * 0.997, "high": close * 1.02,
            "low": close * 0.98, "close": close,
            "volume": 20_000_000.0, "amount": 200_000_000.0, "adj_factor": 1.0,
        }))
    assert not (tmp_path / "configs/factors.yaml").exists()
    output = family_backtest.run_family_backtest(cfg, cache)
    equity = pd.read_parquet(output / "equity.parquet")
    assert len(equity) == len(dates)
    assert set(equity) == {"equity", "baseline", "shanghai_composite"}
    assert equity.shanghai_composite.iloc[0] == cfg["initial_cash"]
    assert np.isclose(equity.shanghai_composite.iloc[-1] / cfg["initial_cash"],
                      (3000 + len(dates) - 1) / 3000)
    assert (tmp_path / "outputs/latest_research.json").exists()
    assert research_pointer(tmp_path, Strategy("original", "等风险月初", "现行策略")).exists()
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["strategy_id"] == "original"
    assert protocol["config_hash"] == config.fingerprint(cfg)

    second = Strategy("test_second", "测试策略", "仅用于测试隔离")
    monkeypatch.setattr(strategy_catalog, "STRATEGIES", (*strategy_catalog.STRATEGIES, second))
    second_output = family_backtest.run_family_backtest(cfg, cache, strategy_id=second.id)
    assert second_output != output
    assert json.loads(research_pointer(tmp_path, second).read_text())["path"] == str(second_output)
    assert json.loads((tmp_path / "outputs/latest_research.json").read_text())["path"] == str(output)
