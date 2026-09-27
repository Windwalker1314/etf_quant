from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from steadyquant import cloud_setup


def test_cloud_setup_launches_once_and_records_result(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("POCKETBAY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cloud_setup, "STATUS", tmp_path / "cloud_setup.json")
    monkeypatch.setattr(cloud_setup, "LOG", tmp_path / "cloud_setup.log")

    with patch("steadyquant.cloud_setup.subprocess.Popen") as launch:
        assert cloud_setup.start()["state"] == "queued"
        assert cloud_setup.start()["state"] == "queued"
        launch.assert_called_once()

    with (
        patch("steadyquant.cloud_setup.load_active_config", return_value={"model": "adaptive"}),
        patch("steadyquant.data.sync", return_value={"ok": True}),
        patch("steadyquant.family_backtest.run_family_backtest", return_value=tmp_path / "research-run"),
    ):
        assert cloud_setup.run() == 0

    status = json.loads(cloud_setup.STATUS.read_text())
    assert status["state"] == "done"
    assert status["run_id"] == "research-run"


def test_cloud_setup_reports_missing_file_and_stage(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cloud_setup, "STATUS", tmp_path / "cloud_setup.json")
    missing = tmp_path / "configs" / "factors.yaml"
    with (
        patch("steadyquant.cloud_setup.load_active_config", return_value={"model": "adaptive"}),
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
