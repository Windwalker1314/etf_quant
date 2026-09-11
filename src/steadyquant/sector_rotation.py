"""Causal sector selection and independently scheduled core/satellite targets."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .config import ROOT, load_active_config
from .data import Cache, DataError, validate_bars
from .sector_data import END, SECTOR_ROOT


def load_snapshot():
    meta = pd.read_parquet(SECTOR_ROOT / "metadata.parquet")
    cfg = load_active_config()
    cache = Cache()
    core = cache.load(cfg)
    core = {s: d[d.date <= pd.Timestamp(END)].copy() for s, d in core.items()}
    records, data, audit = [], dict(core), []
    for row in meta[meta.download_candidate].to_dict("records"):
        s = row["ts_code"]
        info = SECTOR_ROOT / "bars" / f"{s}.json"
        if not info.exists():
            raise DataError(f"Download not complete: {s}")
        status = json.loads(info.read_text())
        audit.append({**status, "name": row["csname"], "current_status": row["current_status"]})
        if not status.get("complete"):
            raise DataError(f"Unresolved download failure: {s}")
        if not status.get("usable"):
            continue
        path = info.with_suffix(".parquet")
        if hashlib.sha256(path.read_bytes()).hexdigest() != status["sha256"]:
            raise DataError(f"ETF snapshot hash changed: {s}")
        d = pd.read_parquet(path)
        validate_bars(d)
        if s in core:
            continue
        data[s] = d
        records.append(row)
    return cfg, data, pd.DataFrame(records).set_index("ts_code"), pd.DataFrame(audit), cache.snapshot_digest


def build_features(data, metadata, skip_recent=0):
    symbols = list(data)
    calendar = pd.read_parquet(ROOT / "data/calendar.parquet")
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            calendar.loc[
                (calendar.is_open.astype(int) == 1) & calendar.cal_date.astype(str).between("20130101", END),
                "cal_date",
            ].sort_values()
        ),
        name="date",
    )
    close = pd.DataFrame({s: d.set_index("date").close for s, d in data.items()}).reindex(dates)
    adj = pd.DataFrame({s: d.set_index("date").adj_factor for s, d in data.items()}).reindex(dates)
    amount = pd.DataFrame({s: d.set_index("date").amount for s, d in data.items()}).reindex(dates)
    volume = pd.DataFrame({s: d.set_index("date").volume for s, d in data.items()}).reindex(dates)
    marked = (close * adj).ffill()
    returns = marked.pct_change(fill_method=None)
    observed = close.notna()
    shifted = marked.shift(skip_recent)
    mom63, mom126 = shifted / shifted.shift(63) - 1, shifted / shifted.shift(126) - 1
    vol = returns.rolling(63, min_periods=63).std() * np.sqrt(252)
    mean_amount = amount.rolling(20, min_periods=20).mean()
    eligible = observed & (observed.cumsum() >= 127 + skip_recent)
    eligible &= observed.rolling(126, min_periods=126).sum() >= 120
    eligible &= (mean_amount >= 20_000_000) & (volume > 0) & (vol > 0)
    eligible &= (marked >= marked.rolling(126, min_periods=126).mean()) & (mom126 > 0)
    relative = 0.5 * (mom63.sub(mom63["510300.SH"], axis=0) + mom126.sub(mom126["510300.SH"], axis=0))
    eligible &= relative > 0
    for s, row in metadata.iterrows():
        eligible[s] &= dates >= pd.Timestamp(row.effective_list_date)
        # A delisting date only masks that date and later; no anticipatory sale.
        if pd.notna(row.delist_date):
            eligible[s] &= dates < pd.Timestamp(row.delist_date)
    eligible.loc[:, [s for s in symbols if s not in metadata.index]] = False
    raw_score = 0.5 * (mom63 + mom126)
    return dict(
        dates=dates,
        close=close,
        returns=returns,
        eligible=eligible,
        amount20=mean_amount,
        raw=raw_score,
        risk_adjusted=raw_score / vol.clip(lower=0.08),
        volatility=vol,
    )


def decision_mask(dates, frequency="weekly"):
    periods = dates.to_period("W-SUN" if frequency == "weekly" else "M")
    return np.r_[True, periods[1:] != periods[:-1]]


def choose_sectors(features, metadata, family, frequency="weekly", entry_count=2, retain_rank=5):
    dates = features["dates"]
    rebalance = decision_mask(dates, frequency)
    eligible = features["eligible"]
    score, amount = features[family], features["amount20"]
    held, histories, diagnostics = [], [], []
    for i, date in enumerate(dates):
        if rebalance[i]:
            valid = metadata.index.intersection(eligible.columns[eligible.loc[date]])
            d = metadata.loc[valid, ["group", "index_key"]].copy()
            d["score"], d["amount20"] = score.loc[date, valid], amount.loc[date, valid]
            d = d.dropna().sort_values(["amount20"], ascending=False, kind="stable")
            # Prefer the already-held tradable share in an index/group; otherwise
            # choose contemporaneous liquidity within an index, strength within a group.
            d["incumbent"] = d.index.isin(held)
            d = d.sort_values(["incumbent", "amount20"], ascending=False, kind="stable")
            indexes = len(d.index_key.unique())
            d = d.drop_duplicates("index_key")
            d = d.sort_values(["incumbent", "score"], ascending=False, kind="stable").drop_duplicates("group")
            d = d.sort_values("score", ascending=False, kind="stable")
            rank = {s: k + 1 for k, s in enumerate(d.index)}
            proposed = [s for s in held if rank.get(s, 9999) <= retain_rank]
            proposed += [s for s in d.index if s not in proposed]
            chosen = []
            for s in proposed:
                if len(chosen) == entry_count:
                    break
                if chosen:
                    window = features["returns"].iloc[max(0, i - 125) : i + 1][chosen + [s]]
                    corr = window.corr(min_periods=100).loc[s, chosen]
                    if corr.isna().any() or (corr >= 0.85).any():
                        continue
                chosen.append(s)
            held = chosen
            diagnostics.append(
                dict(
                    date=date,
                    eligible_etfs=len(valid),
                    eligible_indexes=indexes,
                    eligible_groups=len(d),
                    held=len(held),
                )
            )
        histories.append(tuple(held))
    return pd.Series(histories, index=dates, name="picks"), pd.DataFrame(diagnostics)


def joint_targets(core_weights, picks, fraction, symbols, control=False, frequency="weekly"):
    dates = core_weights.index.intersection(picks.index)
    core_symbols = list(core_weights.columns)
    columns = core_symbols + sorted(set(symbols) - set(core_symbols))
    w = pd.DataFrame(0.0, index=dates, columns=columns)
    mask = pd.DataFrame(False, index=dates, columns=columns)
    forced = mask.copy()
    month = decision_mask(dates, "monthly")
    week = decision_mask(dates, frequency)
    budgets = pd.Series([fraction * len(picks.loc[d]) / 2 for d in dates], index=dates, name="budget")
    previous = 0.0
    sector_symbols = list(set(columns) - set(core_symbols))
    for i, date in enumerate(dates):
        budget = budgets.loc[date]
        w.loc[date, core_symbols] = core_weights.loc[date] * (1 - budget)
        if control:
            w.loc[date, ["510300.SH", "510500.SH"]] += budget / 2
        else:
            for s in picks.loc[date]:
                w.loc[date, s] = fraction / 2
        changed = abs(budget - previous) > 1e-10
        if month[i] or changed:
            mask.loc[date, core_symbols] = True
        if changed:
            # Funding the satellite is part of the decision, not accidental cash
            # left after a 3pp drift band. Apply the identical policy to controls.
            forced.loc[date, core_symbols] = True
        if week[i]:
            mask.loc[date, sector_symbols] = True
            if control:
                mask.loc[date, ["510300.SH", "510500.SH"]] = True
        previous = budget
    assert (w.sum(axis=1) <= 1 + 1e-8).all() and (w >= 0).all().all()
    assert w[sector_symbols].max().max() <= fraction / 2 + 1e-8 if sector_symbols else True
    return w, mask, forced, budgets
