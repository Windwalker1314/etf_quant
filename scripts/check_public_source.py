"""Check staged public content without displaying any credential values."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    files = git("ls-files", "-z").decode().split("\0")
    secrets = []
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            key, sep, value = line.partition("=")
            value = value.strip().strip("\"'")
            if sep and re.search(r"token|secret|password|api.?key", key, re.I) and len(value) >= 8:
                secrets.append(value.encode())
    patterns = [
        ("private key", rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        ("GitHub credential", rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
        ("AWS access key", rb"AKIA[0-9A-Z]{16}"),
        ("credential URL", rb"https?://[^\s/:]+:[^\s/@]+@"),
        ("personal machine path", rb"/Users/[A-Za-z0-9._-]+/"),
    ]
    blocked_parts = {"data", "outputs", "logs", ".venv", ".codex", ".agents", "__pycache__"}
    blocked_suffixes = {".parquet", ".csv", ".tsv", ".xlsx", ".db", ".pem", ".key", ".log", ".zip"}
    failures, total = [], 0
    for name in filter(None, files):
        path = Path(name)
        if (
            blocked_parts.intersection(path.parts)
            or path.suffix in blocked_suffixes
            or (path.name.startswith(".env") and name != ".env.example")
            or name in {"configs/active.yaml", ".streamlit/secrets.toml"}
        ):
            failures.append((name, "private/data path"))
        content = git("show", f":{name}")
        total += len(content)
        if any(secret in content for secret in secrets):
            failures.append((name, "local credential value"))
        for label, pattern in patterns:
            if re.search(pattern, content):
                failures.append((name, label))
        if len(content) > 1_000_000:
            failures.append((name, "unexpected large file"))
    for name, label in failures:
        print(f"BLOCKED {name}: {label}")
    print(f"Checked {len(list(filter(None, files)))} staged files, {total:,} bytes; findings={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
