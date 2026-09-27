import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from steadyquant import parent_ui
from steadyquant.config import fingerprint, write_json
from steadyquant.family_auth import User, user_root


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
    source = f"""
from pathlib import Path
from steadyquant.config import load_config
from steadyquant.data import Cache
from steadyquant.family_auth import User
from steadyquant.parent_ui import render
render(load_config('configs/family.yaml'), Cache(Path({str(tmp_path / 'data')!r})), User('0' * 32, 'test'))
"""
    ui = AppTest.from_string(source).run()
    assert not ui.exception
    assert ui.radio[0].options == ["本月怎么做", "我的持仓", "历史回测"]
    ui.radio[0].set_value("历史回测").run()
    assert not ui.exception


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
