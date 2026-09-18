from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(os.environ.get("STEADYQUANT_HOME", Path(__file__).resolve().parents[2]))


def load_active_config() -> dict:
    """Daily/account configuration, separate from the frozen research baseline."""
    path = ROOT / "configs/active.yaml"
    return load_config(path if path.exists() else None)


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv(ROOT / ".env", override=False)
    cfg = yaml.safe_load(Path(path or ROOT / "configs/steady.yaml").read_text(encoding="utf-8"))
    weights = [a["weight"] for a in cfg["assets"]]
    if len({a["symbol"] for a in cfg["assets"]}) != len(weights):
        raise ValueError("Duplicate asset symbols")
    if any(w < 0 or w > 0.30 for w in weights) or sum(weights) > 1.000001:
        raise ValueError("Long-only policy: allocation <= 100%, single asset <= 30%")
    for key in ("commission_bps", "minimum_commission", "slippage_bps", "cash_rate", "rebalance_band"):
        if cfg[key] < 0:
            raise ValueError(f"Negative {key}")
    if not 0 < cfg["target_vol"] <= 0.2 or not 0 <= cfg["trend_floor"] <= 1:
        raise ValueError("Invalid risk limits")
    if not 0 < cfg["participation_rate"] <= 0.1 or cfg["lot_size"] < 1:
        raise ValueError("Invalid execution limits")
    if cfg.get("model") in {"rotation", "core_satellite"}:
        if not 1 <= cfg["top_n"] <= len(weights):
            raise ValueError("Invalid rotation holding count")
        if (
            not 0 < cfg["max_asset_weight"] <= 0.30
            or not cfg["max_asset_weight"] <= cfg["max_group_weight"] <= 1
        ):
            raise ValueError("Invalid rotation concentration limits")
        if cfg["min_history"] < 253:
            raise ValueError("Rotation requires at least 253 observed bars")
    if cfg.get("model") == "core_satellite":
        if not 0 <= cfg["satellite_fraction"] <= 0.50:
            raise ValueError("Satellite fraction exceeds policy")
        if cfg["core_style"] not in {"steady", "fixed"} or cfg["core_config"].get("model"):
            raise ValueError("Invalid core model")
    if cfg.get("rebalance_frequency", "weekly") not in {"weekly", "monthly", "monthly_first_session"}:
        raise ValueError("Unknown rebalance frequency")
    if cfg.get("model") == "adaptive":
        if cfg.get("risk_method") not in {"equal_risk", "inverse_vol"} or cfg["min_history"] < 253:
            raise ValueError("Invalid adaptive method/history")
        if not 0 <= cfg.get("risk_mix", 1) <= 1 or not 0 <= cfg.get("covariance_shrinkage", 0.3) <= 1:
            raise ValueError("Invalid adaptive risk mix/shrinkage")
        if cfg.get("risk_window", 126) < 20:
            raise ValueError("Insufficient risk window")
    for name in ("relative_rebalance_band", "risk_exit_ratio"):
        if cfg.get(name) is not None and not 0 < cfg[name] < 1:
            raise ValueError(f"Invalid {name}")
    if not 0 <= cfg.get("minimum_trade_notional", 0) < float("inf"):
        raise ValueError("Invalid minimum_trade_notional")
    return cfg


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, value: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
