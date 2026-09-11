"""Fixed core budgets with a bounded, independently specified momentum satellite."""

from __future__ import annotations

import pandas as pd

from .rotation import rotation_weights


def core_satellite_weights(data: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .strategy import target_weights

    core = cfg["core_config"]
    core_data = {a["symbol"]: data[a["symbol"]] for a in core["assets"]}
    base, _ = target_weights(core_data, core, baseline=cfg["core_style"] == "fixed")
    rotation_cfg = {**cfg, "model": "rotation", "rotation_family": "quality", "target_vol": 0.12}
    satellite, factors = rotation_weights(data, rotation_cfg)
    base = base.reindex(index=satellite.index, columns=satellite.columns, fill_value=0)
    fraction = cfg["satellite_fraction"]
    weights = (1 - fraction) * base + fraction * satellite
    return weights, factors
