import shutil

from steadyquant import config


def test_active_daily_config_does_not_replace_frozen_research_baseline(tmp_path, monkeypatch):
    original = config.ROOT
    (tmp_path / "configs").mkdir()
    shutil.copy(original / "configs/steady.yaml", tmp_path / "configs/steady.yaml")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    baseline = config.load_config()
    assert config.load_active_config() == baseline
    shutil.copy(original / "configs/adaptive_paper.yaml", tmp_path / "configs/active.yaml")
    assert config.load_active_config()["model"] == "adaptive"
    assert config.load_config() == baseline
