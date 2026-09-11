from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd
import polars as pl
import yaml
from akquant.data import ParquetDataCatalog
from akquant.factor import FactorEngine

from .config import ROOT


def adjusted_frame(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol, raw in data.items():
        df = raw.copy()
        ratio = df.adj_factor / df.adj_factor.iloc[0]
        for col in ("open", "high", "low", "close"):
            df[col] *= ratio
        frames.append(df)
    return pd.concat(frames).sort_values(["symbol", "date"])


def validate_expression(expr: str):
    """Prevent accidental forward references and arbitrary code in user formulas."""
    if len(expr) > 1000:
        raise ValueError("Factor expression too long")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.ListComp, ast.DictComp)):
            raise ValueError("Only AKQuant expressions are supported")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid factor function")
            if node.func.id in {"Ref", "Delay", "Delta"} or node.func.id.startswith("Ts_"):
                # For all time operators the last argument must be a positive literal.
                if len(node.args) > 1:
                    window = node.args[-1]
                    if (
                        not isinstance(window, ast.Constant)
                        or not isinstance(window.value, (int, float))
                        or window.value < 1
                    ):
                        raise ValueError(
                            "Lookback windows must be positive literals; future references forbidden"
                        )


def compute(data: dict[str, pd.DataFrame], expressions: dict[str, str] | None = None) -> pd.DataFrame:
    expressions = expressions or yaml.safe_load((ROOT / "configs/factors.yaml").read_text())
    engine = FactorEngine(ParquetDataCatalog(str(ROOT / "data/akquant_catalog")))
    frame = adjusted_frame(data)
    # AKQuant maps Close/Open/etc to lowercase columns.
    lazy = pl.from_pandas(frame).lazy()
    out = frame[["date", "symbol"]].copy()
    for name, expr in expressions.items():
        validate_expression(expr)
        values = engine.run_on_data(lazy, expr).to_pandas().rename(columns={"factor_value": name})
        out = out.merge(values, on=["date", "symbol"], validate="one_to_one")
    return out.replace([np.inf, -np.inf], np.nan)


def evaluate(
    data: dict[str, pd.DataFrame], factor_data: pd.DataFrame, horizon: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-overlapping cross-sectional rank IC, labelled research, never used to tune defaults."""
    raw = adjusted_frame(data)
    # Shared calendar: do not silently move a missing next-session entry to a later bar.
    opens = raw.pivot(index="date", columns="symbol", values="open").sort_index()
    labels = (
        (opens.shift(-(horizon + 1)) / opens.shift(-1) - 1)
        .stack(future_stack=True)
        .rename("label")
        .reset_index()
    )
    merged = factor_data.merge(labels, on=["date", "symbol"])
    factor_names = [c for c in factor_data if c not in {"date", "symbol"}]
    dates = sorted(merged.date.unique())[::horizon]
    records = []
    for date, group in merged[merged.date.isin(dates)].groupby("date"):
        for name in factor_names:
            g = group[[name, "label"]].dropna()
            if len(g) < 4 or g[name].nunique() < 2 or g.label.nunique() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ic = g[name].corr(g.label, method="spearman")
            records.append({"date": date, "factor": name, "rank_ic": ic, "assets": len(g)})
    detail = pd.DataFrame(records)
    summary = []
    if not detail.empty:
        for name, g in detail.groupby("factor"):
            s = g.rank_ic.dropna()
            summary.append(
                {
                    "factor": name,
                    "mean_rank_ic": s.mean(),
                    "ic_std": s.std(),
                    "positive_fraction": (s > 0).mean(),
                    "observations": len(s),
                    "horizon": horizon,
                    "sampling": "non-overlapping; small cross-asset universe",
                }
            )
    return pd.DataFrame(summary), detail
