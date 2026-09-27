"""Small, server-side family accounts for the PocketBay edition."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

USERNAME = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,24}\Z")
ITERATIONS = 600_000
MAX_FAILURES = 5
LOCK_MINUTES = 15


@dataclass(frozen=True)
class User:
    id: str
    name: str


def _db_path(root: Path) -> Path:
    return root / "data/family_users.sqlite3"


@contextmanager
def _connect(root: Path):
    path = _db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    except FileExistsError:
        pass
    if os.name != "nt":
        path.chmod(0o600)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username_key TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        salt BLOB NOT NULL,
        password_hash BLOB NOT NULL,
        failed_count INTEGER NOT NULL DEFAULT 0,
        locked_until TEXT,
        created_at TEXT NOT NULL
    )""")
    try:
        with db:
            yield db
    finally:
        db.close()


def _username(name: str) -> tuple[str, str]:
    display = name.strip()
    if not USERNAME.fullmatch(display):
        raise ValueError("用户名须为 2—24 个中文、英文、数字或下划线字符。")
    return display.casefold(), display


def _password(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if not 8 <= len(password) <= 128 or len(encoded) > 512:
        raise ValueError("密码至少需要 8 个字符。")
    return encoded


def _digest(password: bytes, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password, salt, ITERATIONS)


def register(root: Path, name: str, password: str, invite: str, expected_invite: str) -> User:
    if not expected_invite or not hmac.compare_digest(invite, expected_invite):
        raise ValueError("家庭邀请码不正确。")
    key, display = _username(name)
    encoded = _password(password)
    salt = os.urandom(16)
    password_hash = _digest(encoded, salt)
    user = User(uuid.uuid4().hex, display)
    with _connect(root) as db:
        try:
            db.execute("""INSERT INTO users
                (id, username_key, display_name, salt, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                       (user.id, key, display, salt, password_hash, datetime.now(timezone.utc).isoformat()))
        except sqlite3.IntegrityError:
            raise ValueError("用户名已被使用，请换一个。") from None
    return user


def login(root: Path, name: str, password: str) -> User | None:
    try:
        key, _ = _username(name)
    except ValueError:
        return None
    encoded = password.encode("utf-8")[:512]
    now = datetime.now(timezone.utc)
    with _connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM users WHERE username_key = ?", (key,)).fetchone()
        if row is None:
            _digest(encoded, b"\0" * 16)
            return None
        if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > now:
            return None
        valid = hmac.compare_digest(_digest(encoded, row["salt"]), row["password_hash"])
        if not valid:
            failures = row["failed_count"] + 1
            locked = (now + timedelta(minutes=LOCK_MINUTES)).isoformat() if failures >= MAX_FAILURES else None
            db.execute("UPDATE users SET failed_count = ?, locked_until = ? WHERE id = ?",
                       (failures, locked, row["id"]))
            return None
        db.execute("UPDATE users SET failed_count = 0, locked_until = NULL WHERE id = ?", (row["id"],))
        return User(row["id"], row["display_name"])


def get_user(root: Path, user_id: str) -> User | None:
    try:
        uuid.UUID(hex=user_id)
    except (ValueError, AttributeError, TypeError):
        return None
    with _connect(root) as db:
        row = db.execute("SELECT id, display_name FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(row["id"], row["display_name"]) if row else None


def user_root(root: Path, user: User) -> Path:
    # Directory names come only from server-generated UUIDs, never from user input.
    uuid.UUID(hex=user.id)
    return root / "users" / user.id
