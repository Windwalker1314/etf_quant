from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

from .config import ROOT, write_json

TZ = ZoneInfo("Asia/Shanghai")


class DataError(RuntimeError):
    pass


class TushareProvider:
    """Only sends the token to the configured Tushare service, never logs it."""

    def __init__(self):
        import tushare as ts
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        token = os.getenv("TUSHARE_TOKEN") or os.getenv("tushare_token")
        if not token:
            raise DataError("Missing tushare_token in .env")
        url = os.getenv("TUSHARE_HTTP_URL", "https://t.xiaodefa.top/")
        if not url.startswith("https://"):
            raise DataError("Tushare endpoint must use HTTPS")
        self.pro = ts.pro_api(token, timeout=25)
        self.pro._DataApi__http_url = url
        self._last_call = 0.0

    def query(self, api: str, **params) -> pd.DataFrame:
        for attempt in range(3):
            time.sleep(max(0.0, 0.25 - (time.monotonic() - self._last_call)))
            try:
                self._last_call = time.monotonic()
                frame = self.pro.query(api, **params)
                if not isinstance(frame, pd.DataFrame):
                    raise DataError("Invalid response type")
                return frame
            except Exception as exc:
                if attempt == 2:
                    # Remote exception bodies can contain credentials. Keep only the class.
                    raise DataError(
                        f"{api} failed after 3 attempts ({type(exc).__name__}); check endpoint/permission"
                    ) from None
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def calendar(self, start: str, end: str) -> pd.DataFrame:
        df = self.query("trade_cal", exchange="SSE", start_date=start, end_date=end)
        if df.empty or not {"cal_date", "is_open"}.issubset(df):
            raise DataError("Trading calendar unavailable")
        return df.sort_values("cal_date")

    def bars(self, symbol: str, kind: str, start: str, end: str) -> pd.DataFrame:
        if kind not in {"fund", "stock", "index"}:
            raise DataError("Supported kinds: fund, stock, index (bond exposure through ETFs)")
        endpoint = {"fund": "fund_daily", "stock": "daily", "index": "index_daily"}[kind]
        frames = []
        # Calendar-year partitions are strictly below provider row caps; never trust a truncated response.
        for year in range(int(start[:4]), int(end[:4]) + 1):
            lo, hi = max(start, f"{year}0101"), min(end, f"{year}1231")
            raw = self.query(endpoint, ts_code=symbol, start_date=lo, end_date=hi)
            if raw.empty:
                if frames and year < int(end[:4]):
                    raise DataError(
                        f"{symbol}: empty interior calendar year {year}; refusing partial history"
                    )
                continue
            if len(raw) > 366:
                raise DataError(f"{symbol}: response ignored date bounds")
            if not raw.trade_date.astype(str).between(lo, hi).all():
                raise DataError(f"{symbol}: returned dates outside requested partition")
            if kind != "index":
                adj = self.query(
                    "fund_adj" if kind == "fund" else "adj_factor", ts_code=symbol, start_date=lo, end_date=hi
                )
                if adj.empty:
                    raise DataError(f"{symbol}: adjustment factors unavailable; refusing unadjusted backtest")
                raw = raw.merge(
                    adj[["ts_code", "trade_date", "adj_factor"]],
                    on=["ts_code", "trade_date"],
                    how="left",
                    validate="one_to_one",
                )
            else:
                raw["adj_factor"] = 1.0
            frames.append(raw)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        raw_path = ROOT / "data/raw" / f"{symbol}.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(raw_path, index=False)
        combined = quarantine_missing_adjustments(combined, symbol)
        return normalize(combined, symbol)


def quarantine_missing_adjustments(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """No invented factors: omit affected bars and preserve evidence for audit."""
    missing = raw.adj_factor.isna()
    if missing.any():
        quarantine = ROOT / "data/quarantine" / f"{symbol}.parquet"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        bad = raw.loc[missing].copy()
        bad["reason"] = "source_missing_adjustment_factor; excluded from signals and fills"
        if quarantine.exists():
            bad = pd.concat([pd.read_parquet(quarantine), bad]).drop_duplicates(
                ["ts_code", "trade_date"], keep="last"
            )
        bad.to_parquet(quarantine, index=False)
        print(f"{symbol}: quarantined {int(missing.sum())} bars with missing adjustment factors", flush=True)
    return raw.loc[~missing].copy()


def normalize(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount", "adj_factor"}
    if not required.issubset(raw):
        raise DataError(f"{symbol}: missing fields {sorted(required - set(raw))}")
    df = raw.rename(columns={"trade_date": "date", "ts_code": "symbol", "vol": "volume"}).copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for col in ["open", "high", "low", "close", "volume", "amount", "adj_factor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["symbol"].ne(symbol).any() or df["date"].duplicated().any():
        raise DataError(f"{symbol}: symbol mismatch or duplicate dates")
    # vol is lots of 100 units; amount is CNY thousands for fund_daily/daily.
    df["volume"] *= 100
    df["amount"] *= 1000
    validate_bars(df)
    return (
        df[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "adj_factor"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def validate_bars(df: pd.DataFrame):
    values = df[["open", "high", "low", "close", "volume", "amount", "adj_factor"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise DataError("Non-finite OHLCV / adjustment factor")
    if (values[["open", "high", "low", "close", "adj_factor"]] <= 0).any().any():
        raise DataError("Nonpositive prices / adjustment factors")
    if (values[["volume", "amount"]] < 0).any().any():
        raise DataError("Negative volume / amount")
    if (
        (df.high + 1e-8 < df[["open", "close", "low"]].max(axis=1))
        | (df.low - 1e-8 > df[["open", "close", "high"]].min(axis=1))
    ).any():
        raise DataError("Invalid OHLC bounds")


class Cache:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or ROOT / "data")
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, symbol: str) -> Path:
        if not __import__("re").fullmatch(r"[0-9]{6}\.(SH|SZ|BJ)", symbol):
            raise DataError("Invalid symbol")
        return self.root / "bars" / f"{symbol}.parquet"

    def read(self, symbol: str) -> pd.DataFrame:
        path = self.path(symbol)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def save(self, symbol: str, df: pd.DataFrame):
        validate_bars(df)
        if df.date.duplicated().any():
            raise DataError("Duplicate cache dates")
        path = self.path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        write_json(
            path.with_suffix(".json"),
            {
                "symbol": symbol,
                "rows": len(df),
                "start": str(df.date.min().date()),
                "end": str(df.date.max().date()),
                "updated_at": datetime.now(TZ).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source": "tushare",
                "adjustment": "raw OHLC + point-in-time adj_factor",
            },
        )

    def coverage(self) -> pd.DataFrame:
        files = sorted((self.root / "bars").glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        with duckdb.connect(":memory:") as db:
            return db.execute(
                "SELECT symbol, count(*) AS rows, min(date) AS first_date, max(date) AS last_date FROM read_parquet(?) GROUP BY symbol ORDER BY symbol",
                [[str(p) for p in files]],
            ).df()

    def load(self, cfg: dict) -> dict[str, pd.DataFrame]:
        before = self.digest(cfg) if all(self.path(a["symbol"]).exists() for a in cfg["assets"]) else None
        result = {a["symbol"]: self.read(a["symbol"]) for a in cfg["assets"]}
        missing = [s for s, d in result.items() if d.empty]
        if missing:
            raise DataError(f"No cached data for {missing}; run sq sync")
        after = self.digest(cfg)
        if before != after:
            raise DataError("Cache changed during snapshot; retry after sync completes")
        self.snapshot_digest = after
        return result

    def digest(self, cfg: dict) -> dict:
        return {
            a["symbol"]: hashlib.sha256(self.path(a["symbol"]).read_bytes()).hexdigest()
            for a in cfg["assets"]
        }


def sync(cfg: dict, full: bool = False, cache: Cache | None = None) -> dict:
    cache, provider = cache or Cache(), TushareProvider()
    now = datetime.now(TZ)
    # Daily jobs wait until 18:30. Before 18:00 never cache incomplete current-day bars.
    end = (now.date() if now.hour >= 18 else now.date() - timedelta(days=1)).strftime("%Y%m%d")
    calendar = provider.calendar(cfg["data_start"], (now.date() + timedelta(days=40)).strftime("%Y%m%d"))
    calendar.to_parquet(cache.root / "calendar.parquet", index=False)
    expected = calendar.loc[
        (calendar.is_open.astype(int) == 1) & (calendar.cal_date <= end), "cal_date"
    ].max()
    report = {"at": now.isoformat(), "expected_date": expected, "assets": [], "ok": True}
    for asset in cfg["assets"]:
        symbol = asset["symbol"]
        old = cache.read(symbol)
        start = (
            cfg["data_start"]
            if full or old.empty
            else max(cfg["data_start"], (old.date.max() - pd.Timedelta(days=40)).strftime("%Y%m%d"))
        )
        try:
            new = provider.bars(symbol, asset["kind"], start, end)
            if new.empty:
                raise DataError("Empty response for requested refresh window")
            merged = (
                new
                if full or old.empty
                else pd.concat([old[old.date < pd.Timestamp(start)], new], ignore_index=True)
            )
            merged = merged.sort_values("date").reset_index(drop=True)
            cache.save(symbol, merged)
            latest = merged.date.max().strftime("%Y%m%d")
            item = {"symbol": symbol, "rows": len(merged), "last_date": latest, "fresh": latest == expected}
            report["ok"] &= item["fresh"]
        except DataError as exc:
            item = {"symbol": symbol, "error": str(exc), "fresh": False}
            report["ok"] = False
        report["assets"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    write_json(cache.root / "last_sync.json", report)
    return report
