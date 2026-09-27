import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from steadyquant import parent_ui
from steadyquant.config import fingerprint, load_config, write_json
from steadyquant.family_auth import User, user_root
from steadyquant.strategy_catalog import Strategy, research_pointer, strategy_account_root


class FakeCache:
    def __init__(self, root: Path):
        self.root = root
        self.digest_value = {"510300.SH": "new"}

    def digest(self, _cfg):
        return self.digest_value


def test_old_market_or_changed_account_cannot_show_buy_orders(tmp_path, monkeypatch):
    monkeypatch.setattr(parent_ui, "ROOT", tmp_path)
    account_root = user_root(tmp_path, User("0" * 32, "甲"))
    cache = FakeCache(tmp_path / "data")
    cfg = {"name": "example"}
    account = {"cash": 10000, "positions": {}, "as_of": "2026-09-25", "confirmed": True}
    report = {
        "signal_date": "2026-09-25",
        "account_fingerprint": fingerprint(account),
        "config_hash": fingerprint(cfg),
        "data_sha256": cache.digest(cfg),
        "status": "调仓日 · 次日开盘参考清单",
        "orders": [{"symbol": "510300.SH", "side": "BUY", "quantity": 100}],
    }
    write_json(cache.root / "last_sync.json", {"ok": True, "expected_date": "20260925"})
    write_json(account_root / "outputs/family_report.json", report)
    assert parent_ui.current_advice(cfg, cache, account, "2026-09-25", account_root) == report
    assert parent_ui.current_advice(cfg, cache, account, "2026-09-25", account_root,
                                    "test_second") is None
    assert parent_ui.current_advice(cfg, cache, {**account, "cash": 9000}, "2026-09-25", account_root) is None
    cache.digest_value = {"510300.SH": "revised"}
    assert parent_ui.current_advice(cfg, cache, account, "2026-09-25", account_root) is None
    write_json(cache.root / "last_sync.json", {"ok": True, "expected_date": "20260924"})
    assert parent_ui.current_advice(cfg, cache, account, "2026-09-25", account_root) is None


def test_next_monthly_session_skips_remaining_days_this_month(tmp_path):
    cache = FakeCache(tmp_path / "data")
    cache.root.mkdir()
    pd.DataFrame({"cal_date": ["20260925", "20260928", "20260929", "20261008", "20261009"],
                  "is_open": [1, 1, 1, 1, 1]}).to_parquet(cache.root / "calendar.parquet")
    assert parent_ui._next_monthly_session(cache, "2026-09-25") == "2026-10-08"


def test_cloud_parent_navigation_starts_without_market_data(tmp_path, monkeypatch):
    monkeypatch.setattr(parent_ui, "ROOT", tmp_path)
    monkeypatch.setattr(parent_ui, "read_status", lambda: {})
    monkeypatch.setattr(parent_ui, "load_strategy_config", lambda _: load_config("configs/family.yaml"))
    source = f"""
from pathlib import Path
from steadyquant.data import Cache
from steadyquant.family_auth import User
from steadyquant.parent_ui import render
render(Cache(Path({str(tmp_path / 'data')!r})), User('0' * 32, 'test'))
"""
    ui = AppTest.from_string(source).run()
    assert not ui.exception
    assert ui.radio[0].options == ["本月怎么做", "我的持仓", "历史回测"]
    ui.radio[0].set_value("历史回测").run()
    assert not ui.exception


def test_strategy_selection_keeps_existing_account_and_isolates_new_one(tmp_path, monkeypatch):
    monkeypatch.setattr(parent_ui, "ROOT", tmp_path)
    monkeypatch.setattr(parent_ui, "read_status", lambda: {})
    cfg = load_config("configs/family.yaml")
    monkeypatch.setattr(parent_ui, "load_strategy_config", lambda _: cfg)
    original = Strategy("original", "等风险月初", "现行策略")
    second = Strategy("test_second", "测试策略", "仅用于测试隔离")
    monkeypatch.setattr(parent_ui, "strategies", lambda: (original, second))
    user = User("0" * 32, "test")
    old_root = user_root(tmp_path, user)
    write_json(old_root / "data/portfolio.json",
               {"cash": 12345, "positions": {}, "as_of": "2026-09-25", "confirmed": True})
    assert strategy_account_root(tmp_path, user, original) == old_root
    assert strategy_account_root(tmp_path, user, second) != old_root
    source = f"""
from pathlib import Path
from steadyquant.data import Cache
from steadyquant.family_auth import User
from steadyquant.parent_ui import render
render(Cache(Path({str(tmp_path / 'data')!r})), User('0' * 32, 'test'))
"""
    ui = AppTest.from_string(source).run()
    assert not ui.exception
    assert [item.value for item in ui.title] == ["选择策略"]
    assert not ui.radio
    next(button for button in ui.button if button.label == "进入 等风险月初").click().run()
    assert not ui.exception
    ui.radio[0].set_value("我的持仓").run()
    assert ui.number_input[0].value == 12345
    next(button for button in ui.button if button.label == "← 切换策略").click().run()
    next(button for button in ui.button if button.label == "进入 测试策略").click().run()
    ui.radio[0].set_value("我的持仓").run()
    assert not ui.exception
    assert ui.number_input[0].value == 0
    assert not (strategy_account_root(tmp_path, user, second) / "data/portfolio.json").exists()


def test_research_pointer_cannot_show_another_strategys_backtest(tmp_path, monkeypatch):
    monkeypatch.setattr(parent_ui, "ROOT", tmp_path)
    cfg = {"model": "example"}
    original = Strategy("original", "等风险月初", "现行策略")
    second = Strategy("test_second", "测试策略", "仅用于测试隔离")
    run = tmp_path / "outputs/research/example"
    run.mkdir(parents=True)
    (run / "equity.parquet").write_bytes(b"test")
    write_json(run / "protocol.json", {"strategy_id": "original", "config_hash": fingerprint(cfg)})
    write_json(research_pointer(tmp_path, original), {"path": str(run)})
    write_json(research_pointer(tmp_path, second), {"path": str(run)})
    assert parent_ui.latest_research(cfg, original) == run
    assert parent_ui.latest_research(cfg, second) is None


def test_authenticated_cloud_app_reaches_parent_navigation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(POCKETBAY_DATA_DIR=str(tmp_path / "persistent"),
               STEADYQUANT_CLOUD_PASSWORD="test-pass", PYTHONPATH=str(root / "src"))
    script = f"""
from streamlit.testing.v1 import AppTest
from pathlib import Path
from scripts.pocketbay_bootstrap import prepare_environment
from steadyquant.family_auth import register
prepare_environment()
user = register(Path({str(tmp_path / 'persistent/steadyquant')!r}), 'tester', 'test-password-123', 'test-pass', 'test-pass')
ui = AppTest.from_string('import app')
ui.run(timeout=30)
assert not ui.exception, ui.exception
assert not ui.radio
assert [item.label for item in ui.tabs] == ['登录', '注册']
next(item for item in ui.text_input if item.label == '用户名').set_value('tester')
next(item for item in ui.text_input if item.label == '密码').set_value('test-password-123')
next(item for item in ui.button if item.label == '登录').click().run(timeout=30)
assert not ui.exception, ui.exception
assert ui.session_state['cloud_user_id'] == user.id
assert [item.value for item in ui.title] == ['本月怎么做']
assert ui.radio[0].options == ['本月怎么做', '我的持仓', '历史回测']
"""
    subprocess.run([sys.executable, "-c", script], cwd=root, env=env, check=True,
                   capture_output=True, text=True)
