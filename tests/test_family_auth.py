import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from steadyquant.config import write_json
from steadyquant.family import read_json, save_account
from steadyquant.family_auth import get_user, login, register, user_root


def test_family_accounts_are_independent_and_legacy_account_is_not_claimed(tmp_path):
    cfg = {"assets": [{"symbol": "510300.SH"}]}
    write_json(tmp_path / "data/portfolio.json", {"cash": 999999, "legacy": True})
    father = register(tmp_path, "爸爸", "a-strong-pass-123", "family-code", "family-code")
    child = register(tmp_path, "child", "another-pass-123", "family-code", "family-code")
    assert user_root(tmp_path, father) != user_root(tmp_path, child)
    assert read_json(user_root(tmp_path, father) / "data/portfolio.json") is None
    assert login(tmp_path, "爸爸", "a-strong-pass-123") == father
    assert login(tmp_path, "CHILD", "another-pass-123") == child
    save_account(user_root(tmp_path, father), cfg, 10000, {"510300.SH": 100}, "2026-09-18")
    save_account(user_root(tmp_path, child), cfg, 20000, {"510300.SH": 200}, "2026-09-18")
    assert read_json(user_root(tmp_path, father) / "data/portfolio.json")["cash"] == 10000
    assert read_json(user_root(tmp_path, child) / "data/portfolio.json")["cash"] == 20000
    assert read_json(tmp_path / "data/portfolio.json")["legacy"] is True
    assert get_user(tmp_path, father.id) == father
    assert get_user(tmp_path, "../bad") is None
    database = tmp_path / "data/family_users.sqlite3"
    contents = database.read_bytes()
    assert b"a-strong-pass-123" not in contents
    assert b"family-code" not in contents
    if os.name != "nt":
        assert database.stat().st_mode & 0o777 == 0o600


def test_registration_checks_invite_and_unique_username(tmp_path):
    with pytest.raises(ValueError, match="邀请码"):
        register(tmp_path, "first", "a-strong-pass-123", "bad", "family-code")
    assert not (tmp_path / "data/family_users.sqlite3").exists()
    register(tmp_path, "First", "a-strong-pass-123", "family-code", "family-code")
    with pytest.raises(ValueError, match="已被使用"):
        register(tmp_path, "first", "different-pass-123", "family-code", "family-code")
    with pytest.raises(ValueError, match="密码"):
        register(tmp_path, "second", "123", "family-code", "family-code")
    assert register(tmp_path, "second", "12345678", "family-code", "family-code")
    with pytest.raises(ValueError, match="密码"):
        register(tmp_path, "third", "1234567", "family-code", "family-code")


def test_five_bad_logins_lock_account_temporarily(tmp_path):
    register(tmp_path, "father", "a-strong-pass-123", "family-code", "family-code")
    for _ in range(5):
        assert login(tmp_path, "father", "wrong-password") is None
    assert login(tmp_path, "father", "a-strong-pass-123") is None
    with sqlite3.connect(tmp_path / "data/family_users.sqlite3") as db:
        db.execute("UPDATE users SET locked_until = ? WHERE username_key = ?",
                   ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(), "father"))
    assert login(tmp_path, "father", "a-strong-pass-123") is not None
