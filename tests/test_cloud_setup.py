from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from steadyquant import cloud_setup
from steadyquant.strategy_catalog import Strategy


def test_cloud_setup_launches_once_and_records_result(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POCKETBAY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cloud_setup, "STATUS", tmp_path / "cloud_setup.json")
    monkeypatch.setattr(cloud_setup, "LOG", tmp_path / "cloud_setup.log")

    with patch("steadyquant.cloud_setup.subprocess.Popen") as launch:
        assert cloud_setup.start()["state"] == "queued"
        assert cloud_setup.start()["state"] == "queued"
        launch.assert_called_once()

    with (
        patch("steadyquant.cloud_setup.load_strategy_config", return_value={
            "model": "adaptive", "data_start": "20130101",
            "assets": [{"symbol": "510300.SH", "kind": "fund"}]}),
        patch("steadyquant.data.sync", return_value={"ok": True}),
        patch("steadyquant.family_backtest.run_family_backtest", return_value=tmp_path / "research-run"),
    ):
        assert cloud_setup.run() == 0

    status = json.loads(cloud_setup.STATUS.read_text())
    assert status["state"] == "done"
    assert status["run_id"] == "research-run"
    assert status["run_ids"] == {"original": "research-run"}


def test_cloud_setup_syncs_shared_market_once_and_backtests_each_strategy(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud_setup, "STATUS", tmp_path / "cloud_setup.json")
    original = Strategy("original", "等风险月初", "现行策略")
    second = Strategy("test_second", "测试策略", "仅用于测试隔离")
    monkeypatch.setattr(cloud_setup, "strategies", lambda: (original, second))
    configs = {
        "original": {"data_start": "20150101", "assets": [{"symbol": "510300.SH", "kind": "fund"}]},
        "test_second": {"data_start": "20130101", "assets": [
            {"symbol": "510300.SH", "kind": "fund"}, {"symbol": "518880.SH", "kind": "fund"}]},
    }
    monkeypatch.setattr(cloud_setup, "load_strategy_config", lambda item: configs[item.id])
    with (
        patch("steadyquant.data.sync", return_value={"ok": True}) as sync,
        patch("steadyquant.family_backtest.run_family_backtest",
              side_effect=lambda _cfg, strategy_id: tmp_path / strategy_id) as backtest,
    ):
        assert cloud_setup.run() == 0
    sync.assert_called_once()
    shared = sync.call_args.args[0]
    assert shared["data_start"] == "20130101"
    assert {item["symbol"] for item in shared["assets"]} == {"510300.SH", "518880.SH"}
    assert backtest.call_count == 2
    status = json.loads(cloud_setup.STATUS.read_text())
    assert status["run_ids"] == {"original": "original", "test_second": "test_second"}


def test_cloud_setup_reports_missing_file_and_stage(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cloud_setup, "STATUS", tmp_path / "cloud_setup.json")
    missing = tmp_path / "configs" / "factors.yaml"
    with (
        patch("steadyquant.cloud_setup.load_strategy_config", return_value={
            "model": "adaptive", "data_start": "20130101",
            "assets": [{"symbol": "510300.SH", "kind": "fund"}]}),
        patch("steadyquant.data.sync", side_effect=FileNotFoundError(2, "missing", str(missing))),
    ):
        try:
            cloud_setup.run()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Missing file should fail setup")

    status = json.loads(cloud_setup.STATUS.read_text())
    assert status["state"] == "failed"
    assert status["stage"] == "sync"
    assert status["version"] == cloud_setup.SETUP_VERSION
    assert "factors.yaml" in status["message"]
