import numpy as np
import pandas as pd
import pytest
from test_invariants import bars

from steadyquant.adaptive import adaptive_weights
from steadyquant.bond_study import bull_regime, caps_for
from steadyquant.config import ROOT, load_config


def sample():
    cfg = load_config(ROOT / "configs/family.yaml")
    cfg["min_amount"] = 0
    data = {a["symbol"]: bars(340, a["symbol"], i + 50) for i, a in enumerate(cfg["assets"])}
    return cfg, data


def test_default_cap_equivalence_and_reduced_cap_constraints():
    cfg, data = sample()
    original, _ = adaptive_weights(data, cfg)
    explicit, _ = adaptive_weights(data, cfg, bond_cap=.3)
    pd.testing.assert_frame_equal(original, explicit)
    lower, _ = adaptive_weights(data, cfg, bond_cap=.1)
    assert lower["511010.SH"].max() <= .10000001
    assert lower.max().max() <= .30000001
    stocks = [a["symbol"] for a in cfg["assets"] if a["bucket"].endswith("equity")]
    assert lower[stocks].sum(axis=1).max() <= .60000001
    assert lower.sum(axis=1).max() <= 1 and lower.min().min() >= 0
    for bad in [-.1, .31, float("nan")]:
        with pytest.raises(ValueError):
            adaptive_weights(data, cfg, bond_cap=bad)


def test_bull_regime_requires_two_markets_and_no_future():
    _, data = sample()
    for s in ["510300.SH", "513500.SH", "159920.SZ"]:
        prices = np.arange(340, dtype=float) + 100
        data[s].loc[:, ["open", "high", "low", "close"]] = prices[:, None] * np.ones((1, 4))
        data[s]["adj_factor"] = 1.0
    regime = bull_regime(data)
    assert not regime.iloc[:251].any() and regime.iloc[251:].all()
    cap = caps_for(data, .3, .1)
    assert cap.iloc[250] == .3 and cap.iloc[-1] == .1
    cut = regime.index[300]
    truncated = {s: d[d.date <= cut] for s, d in data.items()}
    pd.testing.assert_series_equal(caps_for(truncated, .3, .1), cap.loc[:cut])
    for s in ["510300.SH", "513500.SH"]:
        data[s].loc[data[s].date > cut, "close"] = .001
    pd.testing.assert_series_equal(caps_for(data, .3, .1).loc[:cut], cap.loc[:cut])
    assert not bull_regime(data).iloc[-1]
