"""Published family-site strategies and their isolated storage locations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import load_active_config, load_config
from .family_auth import User, user_root

SOURCE = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_ID = "original"


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    summary: str
    config_file: str | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", self.id):
            raise ValueError("Invalid strategy ID")


# Add only strategies with reviewed trading rules and a usable historical backtest.
STRATEGIES = (
    Strategy("original", "等风险月初", "现行 ETF 策略 · 每月首个交易日判断调仓"),
)


def strategies() -> tuple[Strategy, ...]:
    return STRATEGIES


def strategy_by_id(strategy_id: str) -> Strategy:
    for strategy in strategies():
        if strategy.id == strategy_id:
            return strategy
    raise ValueError("Unknown strategy")


def load_strategy_config(strategy: Strategy) -> dict:
    if strategy.id == DEFAULT_STRATEGY_ID:
        return load_active_config()
    if not strategy.config_file:
        raise ValueError("Strategy config is missing")
    return load_config(SOURCE / strategy.config_file)


def strategy_account_root(root: Path, user: User, strategy: Strategy) -> Path:
    account = user_root(root, user)
    # Keep existing users' real holdings and advice at their original paths.
    if strategy.id == DEFAULT_STRATEGY_ID:
        return account
    return account / "strategies" / strategy.id


def research_pointer(root: Path, strategy: Strategy) -> Path:
    return root / "outputs/strategies" / strategy.id / "latest_research.json"


def market_sync_config(entries: list[tuple[Strategy, dict]]) -> dict:
    """Sync the union of ETF data once, then backtest each strategy separately."""
    if not entries:
        raise ValueError("No published strategies")
    assets: dict[str, dict] = {}
    for _, cfg in entries:
        for asset in cfg["assets"]:
            symbol = asset["symbol"]
            if symbol in assets and assets[symbol]["kind"] != asset["kind"]:
                raise ValueError(f"Conflicting data kind for {symbol}")
            assets.setdefault(symbol, asset)
    return {**entries[0][1], "data_start": min(cfg["data_start"] for _, cfg in entries),
            "assets": list(assets.values())}
