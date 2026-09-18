"""Build a self-contained Windows x64 ZIP from official embedded Python and wheels.

Fetch wheels with pip --isolated download --only-binary=:all: --platform win_amd64
--python-version 3.13 --implementation cp --abi cp313 -r windows/requirements.txt.
Python: https://www.python.org/downloads/release/python-31315/
Only the explicit application allowlist below is copied; never account data or .env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SHA256 = "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf"
MODULES = ("__init__", "config", "data", "daily", "execution", "strategy", "adaptive",
           "factors", "macro", "reports", "family")


def check_dependencies(wheels: Path):
    metadata = {}
    for wheel in wheels.glob("*.whl"):
        with zipfile.ZipFile(wheel) as z:
            meta = Parser().parsestr(z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode())
            name = canonicalize_name(meta["Name"])
            if name in metadata:
                raise ValueError(f"Multiple versions of {name}")
            metadata[name] = meta
    environment = default_environment()
    environment.update(os_name="nt", sys_platform="win32", platform_system="Windows",
                       platform_machine="AMD64", python_version="3.13", python_full_version="3.13.15", extra="")
    requirements = [Requirement(line) for line in (ROOT / "windows/requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]
    visited = set()
    while requirements:
        req = requirements.pop()
        name = canonicalize_name(req.name)
        meta = metadata.get(name)
        if not meta or not req.specifier.contains(meta["Version"]):
            raise ValueError(f"Missing/incompatible Windows dependency: {req}")
        if (name, tuple(sorted(req.extras))) in visited:
            continue
        visited.add((name, tuple(sorted(req.extras))))
        for spec in meta.get_all("Requires-Dist", []):
            child = Requirement(spec)
            if not child.marker or any(child.marker.evaluate({**environment, "extra": extra})
                                       for extra in {"", *req.extras}):
                requirements.append(child)


def unpack(archive: Path, destination: Path, wheel=False):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename:
                raise ValueError("Unsafe archive path")
            if wheel and path.parts[0].endswith(".data"):
                if path.parts[1] not in {"purelib", "platlib"}:
                    continue  # Console entry scripts aren't used by this application.
                path = PurePosixPath(*path.parts[2:])
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(item))


def build(runtime_zip: Path, wheels: Path, output: Path) -> Path:
    check_dependencies(wheels)
    if hashlib.sha256(runtime_zip.read_bytes()).hexdigest() != PYTHON_SHA256:
        raise ValueError("Python archive SHA256 does not match python.org")
    output.mkdir(parents=True, exist_ok=True)
    staging = output / "ETF小助手-Windows"
    if staging.exists():
        raise ValueError("Staging directory exists; use a new --output directory")
    staging.mkdir()
    unpack(runtime_zip, staging / "runtime")
    (staging / "runtime/python313._pth").write_text(
        "python313.zip\n.\n../packages\n../src\n../windows\nimport site\n", encoding="utf-8")
    manifests = []
    wheel_files = sorted(wheels.glob("*.whl"))
    if not wheel_files:
        raise ValueError("Windows wheel directory is empty")
    for wheel in wheel_files:
        if not (wheel.name.endswith("win_amd64.whl") or wheel.name.endswith("none-any.whl")):
            raise ValueError("Non-Windows wheel in bundle")
        unpack(wheel, staging / "packages", wheel=True)
        manifests.append(dict(file=wheel.name, sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()))
    sources = [f"src/steadyquant/{name}.py" for name in MODULES]
    sources += ["windows/launcher.py", "windows/family_app.py", "windows/self_test.py", "windows/requirements.txt", "configs/family.yaml"]
    # Check local credentials without including or printing their contents.
    from dotenv import dotenv_values
    secrets = [v.encode() for k, v in dotenv_values(ROOT / ".env").items()
               if v and len(v) >= 8 and any(s in k.lower() for s in ("token", "key", "password", "secret"))]
    for name in sources:
        content = (ROOT / name).read_bytes()
        if any(secret in content for secret in secrets):
            raise ValueError("Private credential detected in application source")
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    shutil.copyfile(ROOT / "windows/先看这里.txt", staging / "先看这里.txt")
    (staging / "启动ETF小助手.cmd").write_bytes(
        b'@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\n'
        b'"%~dp0runtime\\python.exe" -X utf8 "%~dp0windows\\launcher.py"\r\n'
        b'if errorlevel 1 pause\r\n')
    (staging / "检查运行环境.cmd").write_bytes(
        b'@echo off\r\nsetlocal\r\nchcp 65001 >nul\r\n'
        b'"%~dp0runtime\\python.exe" -X utf8 "%~dp0windows\\self_test.py"\r\npause\r\n')
    (staging / "build-manifest.json").write_text(json.dumps({
        "python_sha256": PYTHON_SHA256, "dependencies": manifests,
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "contains_credentials_or_accounts": False,
        "native_windows_validation": "pending",
    }, indent=2), encoding="utf-8")
    target = output / "ETF小助手-Windows-x64.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(output))
    print(f"Created {target} ({target.stat().st_size / 1024**2:.1f} MiB)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--wheels", type=Path, default=ROOT / "build/windows-wheels")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/windows")
    args = parser.parse_args()
    build(args.runtime, args.wheels, args.output)
