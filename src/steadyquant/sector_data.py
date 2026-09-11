"""Isolated, auditable market-wide ETF metadata and daily-bar snapshots."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from .config import ROOT, write_json
from .data import DataError, TushareProvider, normalize

SECTOR_ROOT = ROOT / "data/sector_rotation/20260911"
END = "20260910"
START = "20130101"

# Semantic coverage, not a list of historically winning symbols. Classification
# uses current metadata and is explicitly not a certified historical taxonomy.
THEMES = [
    ("semiconductor", "半导体|芯片|集成电路"),
    ("health", "医药|医疗|医械|创新药|生物|疫苗|中药|制药|药物|养老|健康"),
    ("new_energy", "新能源|电池|电动|光伏|风电|储能|氢能|锂|稀土|碳中和|碳能|碳达峰|清洁能源|绿色电力"),
    ("vehicle", "汽车|智能驾驶|车联网"),
    (
        "technology",
        "科技|信息|计算机|软件|互联网|通信|通讯|电子|5G|6G|人工智能|AI|云计算|数据|数字|智能|机器人|物联网|网络|信创|消费电子",
    ),
    ("defense", "军工|国防|航空|航天|卫星|低空|通用航空"),
    ("finance", "银行|证券|券商|保险|金融|财富管理|金融科技"),
    (
        "consumption",
        "消费|食品|饮料|酒|家电|家居|家用|畜牧|养殖|农业|农牧|旅游|文旅|休闲|餐饮|零售|纺织|服装|商贸",
    ),
    ("media", "传媒|游戏|影视|文化|娱乐|教育|出版|动漫|体育"),
    ("resources", "有色|金属|钢铁|煤炭|石油|石化|化工|油气|能源|矿业|材料|黄金股|资源|稀有|化纤"),
    (
        "infrastructure",
        "建筑|建材|基建|地产|房地产|工程|机械|制造|工业|装备|水泥|运输|交通|物流|电力|电网|公用|环保|水务|电气|港口|航运|交运",
    ),
    ("dividend", "红利|股息"),
    ("value_quality", "低波|价值|质量|现金流|基本面"),
    ("growth", "成长|创新|新兴|创成长|硬科技|新质|科创综指"),
    ("ownership_esg", "央企|国企|国资|治理|ESG|责任|可持续|一带一路|改革|民企|民营"),
    (
        "regional",
        "长三角|湾区|京津冀|京津|成渝|浙江|江苏|山东|四川|湖北|上海国企|上海改革|北京50|浙江凤凰|粤港澳",
    ),
]


def classify(text):
    text = str(text)
    # Avoid confusing physical commodity/bond funds with equity sectors.
    if re.search(
        "期货|债券|国债|政金债|信用债|城投债|短融|货币|现金添|现金宝|保证金|黄金ETF|黄金交易|黄金现货|白银",
        text,
    ):
        return None
    for group, pattern in THEMES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return group
    return None


def build_metadata():
    root = SECTOR_ROOT / "probes"
    calendar = pd.read_parquet(ROOT / "data/calendar.parquet")
    sessions = (
        calendar.loc[(calendar.is_open.astype(int) == 1) & (calendar.cal_date.astype(str) <= END), "cal_date"]
        .astype(str)
        .sort_values()
    )
    latest_listing = sessions.iloc[-127]
    e = pd.concat([pd.read_parquet(root / f"etf_basic_{s}.parquet") for s in ("L", "D")], ignore_index=True)
    f = pd.concat([pd.read_parquet(root / f"fund_basic_{s}.parquet") for s in ("L", "D")], ignore_index=True)
    if e.ts_code.duplicated().any() or f.ts_code.duplicated().any():
        raise DataError("Duplicate metadata symbol")
    listed = e[e.ts_code.str.fullmatch(r"\d{6}\.(SH|SZ)")].copy()
    of = e[e.ts_code.str.endswith(".OF")].set_index(e[e.ts_code.str.endswith(".OF")].ts_code.str[:6])
    fund = f.set_index("ts_code")
    rows = []
    for r in listed.to_dict("records"):
        s = r["ts_code"]
        other = of.loc[s[:6]].to_dict() if s[:6] in of.index else {}
        if other and str(other.get("csname")) != str(r.get("csname")):
            other = {}  # Do not join ambiguous share classes on digits alone.
        for key in ("index_code", "index_name", "list_date"):
            if pd.isna(r.get(key)) and pd.notna(other.get(key)):
                r[key] = other[key]
        fm = fund.loc[s].to_dict() if s in fund.index else {}
        dates = [
            str(v)
            for v in (r.get("list_date"), fm.get("list_date"))
            if pd.notna(v) and re.fullmatch(r"\d{8}", str(v))
        ]
        # If listing dates disagree, wait for the later one, never backfill before it.
        r["effective_list_date"] = max(dates) if dates else None
        r["listing_date_conflict"] = len(set(dates)) > 1
        r["delist_date"] = fm.get("delist_date")
        r["fund_type"] = fm.get("fund_type")
        r["fund_status"] = fm.get("status")
        r["benchmark"] = fm.get("benchmark")
        r["group"] = classify(str(r.get("index_name", "")) + " " + str(r["csname"]))
        r["is_theme"] = bool(r["group"] and "ETF" in str(r["csname"]).upper())
        # Six calendar months are a download bound only; >=127 real observations
        # and all trailing windows are required again at every historical signal.
        r["download_candidate"] = bool(
            r["is_theme"] and r["effective_list_date"] and r["effective_list_date"] <= latest_listing
        )
        r["current_status"] = "D" if r["list_status"] == "D" or fm.get("status") == "D" else "L"
        r["index_key"] = str(r.get("index_code")) if pd.notna(r.get("index_code")) else "unknown:" + s
        rows.append(r)
    meta = pd.DataFrame(rows)
    meta.to_parquet(SECTOR_ROOT / "metadata.parquet", index=False)
    meta[
        [
            "ts_code",
            "csname",
            "index_name",
            "group",
            "effective_list_date",
            "delist_date",
            "current_status",
            "download_candidate",
        ]
    ].to_csv(SECTOR_ROOT / "universe_review.csv", index=False)
    write_json(
        SECTOR_ROOT / "universe_audit.json",
        dict(
            listed_exchange_codes=len(meta),
            theme_candidates=int(meta.is_theme.sum()),
            download_candidates=int(meta.download_candidate.sum()),
            retired_download_candidates=int(((meta.current_status == "D") & meta.download_candidate).sum()),
            missing_listing=int(meta.effective_list_date.isna().sum()),
            listing_conflicts=int(meta.listing_date_conflict.sum()),
            unknown_theme=int((~meta.is_theme).sum()),
            taxonomy_rules=THEMES,
            limitation="Current source metadata, not certified historical taxonomy; full exchange archive completeness unverified. Include retired records; uncertain metadata disclosed rather than calling this survivorship-free.",
        ),
    )
    return meta


class SnapshotClient:
    def __init__(self):
        self.local = threading.local()
        self.lock = threading.Lock()
        self.last = 0.0

    def query(self, api, **params):
        key = hashlib.sha256(json.dumps([api, params], sort_keys=True).encode()).hexdigest()
        path = SECTOR_ROOT / "requests" / f"{key}.parquet"
        info = path.with_suffix(".json")
        if path.exists() and info.exists():
            meta = json.loads(info.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                raise DataError("Snapshot hash mismatch")
            return pd.read_parquet(path)
        if not hasattr(self.local, "provider"):
            self.local.provider = TushareProvider()
        with self.lock:
            time.sleep(max(0, 0.28 - (time.monotonic() - self.last)))
            self.last = time.monotonic()
        d = self.local.provider.query(api, **params)
        path.parent.mkdir(exist_ok=True)
        d.to_parquet(path, index=False)
        write_json(
            info,
            dict(
                api=api,
                params=params,
                rows=len(d),
                first_seen_at=datetime.now().isoformat(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
        )
        return d

    def paged(self, api, symbol, start, end, limit=1500):
        pieces, seen = [], set()
        for offset in range(0, 15000, limit):
            d = self.query(api, ts_code=symbol, start_date=start, end_date=end, offset=offset, limit=limit)
            if d.empty:
                break
            if not {"ts_code", "trade_date"}.issubset(d):
                raise DataError(f"{api}: missing keys")
            if not d.ts_code.eq(symbol).all() or not d.trade_date.astype(str).between(start, end).all():
                raise DataError(f"{api}: ignored filters")
            keys = set(d.trade_date.astype(str))
            if len(keys) != len(d) or seen.intersection(keys):
                raise DataError(f"{api}: duplicate/repeated page")
            seen.update(keys)
            pieces.append(d)
            if len(d) < limit:
                break
        else:
            raise DataError("Pagination did not terminate")
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def download_one(client, record):
    symbol = record["ts_code"]
    out = SECTOR_ROOT / "bars" / f"{symbol}.parquet"
    detail = SECTOR_ROOT / "bars" / f"{symbol}.json"
    if detail.exists():
        prior = json.loads(detail.read_text())
        if prior.get("complete") and (
            not prior.get("usable")
            or (out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == prior.get("sha256"))
        ):
            return prior
    try:
        start = max(START, record["effective_list_date"])
        raw = client.paged("fund_daily", symbol, start, END)
        if raw.empty:
            result = dict(symbol=symbol, complete=True, usable=False, reason="no_raw_bars", rows=0)
        else:
            adj = client.paged("fund_adj", symbol, start, END)
            if adj.empty:
                raise DataError("No adjustment history")
            if "adj_factor" not in adj:
                raise DataError("Adjustment column missing")
            joined = raw.merge(
                adj[["ts_code", "trade_date", "adj_factor"]],
                on=["ts_code", "trade_date"],
                how="left",
                validate="one_to_one",
            )
            numeric = joined[["open", "high", "low", "close", "vol", "amount", "adj_factor"]].apply(
                pd.to_numeric, errors="coerce"
            )
            valid = np.isfinite(numeric.to_numpy()).all(axis=1)
            valid &= (numeric[["open", "high", "low", "close", "adj_factor"]] > 0).all(axis=1)
            valid &= (numeric[["vol", "amount"]] >= 0).all(axis=1)
            valid &= numeric.high + 1e-8 >= numeric[["open", "close", "low"]].max(axis=1)
            valid &= numeric.low - 1e-8 <= numeric[["open", "close", "high"]].min(axis=1)
            bad = joined.loc[~valid]
            if len(bad):
                (SECTOR_ROOT / "quarantine").mkdir(exist_ok=True)
                bad.to_parquet(SECTOR_ROOT / "quarantine" / f"{symbol}.parquet", index=False)
            clean = normalize(joined.loc[valid], symbol)
            out.parent.mkdir(exist_ok=True)
            clean.to_parquet(out, index=False)
            result = dict(
                symbol=symbol,
                complete=True,
                usable=len(clean) >= 127,
                rows=len(clean),
                raw_rows=len(raw),
                quarantined=len(bad),
                start=str(clean.date.min()),
                end=str(clean.date.max()),
                sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            )
        detail.parent.mkdir(exist_ok=True)
        write_json(detail, result)
        return result
    except DataError as exc:
        result = dict(symbol=symbol, complete=False, usable=False, reason=str(exc))
        detail.parent.mkdir(exist_ok=True)
        write_json(detail, result)
        return result


def sync_sector(pilot=False):
    meta = build_metadata()
    records = meta[meta.download_candidate].sort_values("ts_code").to_dict("records")
    if pilot:
        # Two old funds (pagination), two thematic funds and retired examples.
        codes = ["510880.SH", "512010.SH", "512480.SH", "515030.SH"]
        records = [r for r in records if r["ts_code"] in codes] + [
            r for r in records if r["current_status"] == "D"
        ][:3]
    client = SnapshotClient()
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(download_one, client, r) for r in records]
        for task in as_completed(pending):
            result = task.result()
            results.append(result)
            if not result["complete"] or len(results) % 25 == 0 or pilot:
                print(
                    json.dumps(dict(done=len(results), total=len(records), **result), ensure_ascii=False),
                    flush=True,
                )
            write_json(
                SECTOR_ROOT / ("pilot_progress.json" if pilot else "progress.json"),
                dict(done=len(results), total=len(records), failed=sum(not r["complete"] for r in results)),
            )
    write_json(SECTOR_ROOT / ("pilot_summary.json" if pilot else "download_summary.json"), results)
    return results
