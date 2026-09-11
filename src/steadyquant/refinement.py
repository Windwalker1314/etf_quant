"""Second bounded research stage; all previously seen results are explicitly disclosed."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from steadyquant.backtest import EXECUTION_VERSION, simulate
from steadyquant.config import ROOT, load_config, write_json
from steadyquant.data import TZ, Cache
from steadyquant.metrics import window_performance, yearly
from steadyquant.strategy import target_weights


def run_refinement(source: Path | None = None) -> Path:
    source = source or Path(json.loads((ROOT / "outputs/latest_optimization.json").read_text())["path"])
    base_cfg = json.loads((source / "protocol.json").read_text())["config"]
    core_cfg = load_config()
    data = Cache().load(base_cfg)
    folder = ROOT / "outputs/refinement" / datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True)
    candidates = {
        f"{style}_sat{int(fraction * 100)}_{frequency}": dict(
            base_cfg,
            model="core_satellite",
            core_config=core_cfg,
            core_style=style,
            satellite_fraction=fraction,
            rebalance_frequency=frequency,
            rebalance_band=0.03,
        )
        for style in ("steady", "fixed")
        for fraction in (0.25, 0.5)
        for frequency in ("weekly", "monthly")
    }
    write_json(
        folder / "protocol.json",
        dict(
            execution_version=EXECUTION_VERSION,
            prior_run=str(source),
            candidates=candidates,
            reason="Round1 all candidates failed 15% selection drawdown. Evaluate diversification and reduced turnover, not another parameter search.",
            satellite="quality_v12: strongest pre2023 family; use lower12% risk because v16 score improvement<0.01",
            selection="pre2023 drawdown<=15%, worst calendar year>=-5%, Sharpe higher than v1; maximize Sharpe+0.25*min(Calmar,3); if none pass retain v1",
            holdout="All histories have been viewed in round1; these are retrospective refinements, NOT fresh out-of-sample tests.",
            data_sha256=Cache().digest(base_cfg),
        ),
    )
    core_data = {a["symbol"]: data[a["symbol"]] for a in core_cfg["assets"]}
    core_targets = {
        s: target_weights(core_data, core_cfg, baseline=s == "fixed")[0] for s in ("steady", "fixed")
    }
    sat = pd.read_parquet(source / "quality_v12/targets.parquet")
    rows = []
    for name, cfg in candidates.items():
        print("Refinement", name, flush=True)
        core = core_targets[cfg["core_style"]].reindex(index=sat.index, columns=sat.columns, fill_value=0)
        w = core * (1 - cfg["satellite_fraction"]) + sat * cfg["satellite_fraction"]
        run = simulate(data, w, cfg)
        p = folder / name
        p.mkdir()
        w.to_parquet(p / "targets.parquet")
        run.equity.to_parquet(p / "equity.parquet")
        run.trades.to_parquet(p / "trades.parquet", index=False)
        run.positions.to_parquet(p / "positions.parquet")
        (p / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        yy = yearly(run.equity.equity, cfg["risk_free_rate"])
        yy.to_csv(p / "yearly.csv", index=False)
        for period, start, end in [
            ("full", "2015-01-01", None),
            ("selection", "2015-01-01", "2022-12-31"),
            ("test", "2023-01-01", None),
        ]:
            rows.append(
                dict(
                    candidate=name,
                    period=period,
                    **window_performance(run.equity.equity, start, end, cfg["risk_free_rate"]),
                )
            )
    pd.DataFrame(rows).to_csv(folder / "candidates.csv", index=False)
    from .refinement_report import finish_refinement

    finish_refinement(folder)
    write_json(ROOT / "outputs/latest_refinement.json", dict(path=str(folder)))
    print(
        pd.DataFrame(rows)[["candidate", "period", "cagr", "sharpe", "max_drawdown"]].to_string(index=False),
        flush=True,
    )

    return folder
