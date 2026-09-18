import shutil
import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from steadyquant import config, daily, data
from steadyquant.config import fingerprint
from steadyquant.family import advice_is_current, holdings_rows, read_json, save_account

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def family_home(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    shutil.copyfile(REPO / "configs/family.yaml", tmp_path / "configs/active.yaml")
    for module in (config, data, daily):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    return tmp_path, config.load_active_config()


def test_account_save_preserves_exact_shares_and_invalidates_advice(family_home):
    home, cfg = family_home
    positions = {cfg["assets"][0]["symbol"]: 1300}
    first = save_account(home, cfg, 2300.25, positions, "2026-09-17")
    second = save_account(home, cfg, 1900.10, positions, "2026-09-18")
    assert read_json(home / "data/portfolio.json") == second
    snapshots = list((home / "data/account_history").glob("*-before.json"))
    assert read_json(snapshots[0]) == first
    assert read_json(home / "outputs/family_report.json")["invalidated"]
    assert holdings_rows(cfg, second)[0]["当前持仓（份）"] == 1300


@pytest.mark.parametrize("cash,positions,day", [
    (-1, {}, "2026-09-18"),
    (float("nan"), {}, "2026-09-18"),
    (100, {"510300.SH": 1.5}, "2026-09-18"),
    (100, {"999999.SH": 100}, "2026-09-18"),
    (100, {}, "2099-01-01"),
])
def test_invalid_account_never_overwrites(family_home, cash, positions, day):
    home, cfg = family_home
    before = save_account(home, cfg, 100, {}, "2026-09-17")
    with pytest.raises((ValueError, data.DataError)):
        save_account(home, cfg, cash, positions, day)
    assert read_json(home / "data/portfolio.json") == before


def test_old_account_or_date_or_config_never_reuses_advice(family_home):
    home, cfg = family_home
    account = save_account(home, cfg, 10000, {}, "2026-09-17")
    report = dict(signal_date="2026-09-17", account_fingerprint=fingerprint(account), config_hash=fingerprint(cfg))
    assert advice_is_current(report, account, "2026-09-17", cfg)
    assert not advice_is_current(report, {**account, "cash": 9999}, "2026-09-17", cfg)
    assert not advice_is_current(report, account, "2026-09-18", cfg)
    assert not advice_is_current(report, account, "2026-09-17", {**cfg, "commission_bps": 3})
    assert not advice_is_current({"invalidated": True}, account, "2026-09-17", cfg)


def test_windows_lock_releases_after_error(tmp_path, monkeypatch):
    calls = []
    fake = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=lambda fd, op, n: calls.append((op, n)))
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "msvcrt", fake)
        m.setattr(sys, "platform", "win32")
        with pytest.raises(RuntimeError):
            with daily.job_lock(tmp_path):
                raise RuntimeError("test")
    assert calls == [(1, 1), (2, 1)]
    assert (tmp_path / "data/daily.lock").read_bytes() == b"0"


def test_job_lock_refuses_concurrent_run(tmp_path):
    with daily.job_lock(tmp_path):
        with pytest.raises(data.DataError, match="already running"):
            with daily.job_lock(tmp_path):
                pass
    with daily.job_lock(tmp_path):
        pass


def test_fresh_ui_has_no_copied_account_and_can_save_cash(family_home):
    home, _ = family_home
    at = AppTest.from_file(str(REPO / "windows/family_app.py")).run()
    assert not at.exception
    assert at.radio[0].value == "首次设置"
    assert not (home / "data/portfolio.json").exists()
    at.radio[0].set_value("我的持仓").run()
    assert not at.exception
    assert at.number_input[0].value == 0
    at.number_input[0].set_value(85000)
    at.checkbox[0].check()
    at.button[0].click().run()
    assert not at.exception
    account = read_json(home / "data/portfolio.json")
    assert account["cash"] == 85000 and not any(account["positions"].values())
    at.radio[0].set_value("今日建议").run()
    assert not at.exception
    assert at.button[0].disabled
    assert at.metric[0].value == "¥85,000.00"


def test_ui_failed_refresh_removes_old_orders(family_home, monkeypatch):
    home, cfg = family_home
    (home / ".env").write_text("TUSHARE_TOKEN=synthetic-test-key\nTUSHARE_HTTP_URL=https://example.invalid/\n")
    account = save_account(home, cfg, 10000, {}, "2026-09-17")
    config.write_json(home / "outputs/family_report.json", dict(
        account_fingerprint=fingerprint(account), config_hash=fingerprint(cfg),
        signal_date="2026-09-17", status="调仓日", orders=[{"symbol":"510300.SH","side":"BUY","quantity":100}]))
    monkeypatch.setattr(daily, "calendar_dates", lambda *_: ("20260917", "20260918"))
    def failed(*_):
        raise data.DataError("synthetic-test-key must not appear in UI")
    monkeypatch.setattr(daily, "daily", failed)
    at = AppTest.from_file(str(REPO / "windows/family_app.py")).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.error and "synthetic-test-key" not in at.error[0].value
    assert read_json(home / "outputs/family_report.json")["invalidated"]
    assert "买卖数量（份）" not in at.dataframe[0].value.columns
