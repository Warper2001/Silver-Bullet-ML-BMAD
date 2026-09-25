#!/usr/bin/env python3
"""ETFTM-1 A0.2 — read-only depth / adjustment probe (market-data GETs only; places nothing).

Auth: reads the CURRENT token from .access_token on every request and never refreshes, so the live
bots' shared credentials (.env, .access_token) are never written. The live bots refresh the file
every ~10 minutes.

For each symbol:
  TradeStation: Daily barcharts, barsback=5000, then page back with lastdate until no older bars.
  Yahoo chart API (range=max, events=div,split): first date, dividend count, splits.
  Adjustment test on the overlap: TS close vs Yahoo 'close' (split-adjusted, price-only) and
  'adjclose' (split + dividend adjusted).
Writes _bmad-output/etftm1_depth_probe_<date>.json.

Run: .venv/bin/python tools/probe_etf_depth.py [SYM ...]
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.etftm1_universe import ALL_SYMBOLS, AUDIT_ONLY, TRADING_ALTERNATES  # noqa: E402

MD = "https://api.tradestation.com/v3/marketdata"
YF = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
YF_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
              "Accept": "application/json"}


def token() -> str:
    return (ROOT / ".access_token").read_text().strip()


def ts_get(sym: str, params: dict) -> list[dict]:
    for attempt in range(3):
        r = requests.get(f"{MD}/barcharts/{sym}", headers={"Authorization": f"Bearer {token()}"},
                         params={"interval": 1, "unit": "Daily", **params}, timeout=60)
        if r.status_code == 401 and attempt < 2:
            time.sleep(20)                 # a live bot is mid-refresh; re-read the file
            continue
        r.raise_for_status()
        return r.json().get("Bars", [])
    return []


def ts_history(sym: str, max_pages: int = 6) -> tuple[pd.DataFrame, int]:
    frames, pages = [], 0
    bars = ts_get(sym, {"barsback": 5000})
    while bars and pages < max_pages:
        pages += 1
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["TimeStamp"], utc=True).dt.tz_convert("America/New_York").dt.date
        frames.append(df)
        earliest = df["TimeStamp"].min()
        older = ts_get(sym, {"barsback": 5000, "lastdate": earliest})
        older = [b for b in older if b["TimeStamp"] < earliest]
        bars = older
    if not frames:
        return pd.DataFrame(), pages
    out = pd.concat(frames).drop_duplicates("TimeStamp").sort_values("TimeStamp")
    out["close"] = out["Close"].astype(float)
    return out[["date", "close"]].reset_index(drop=True), pages


def yahoo(sym: str) -> tuple[pd.DataFrame, dict]:
    r = requests.get(YF.format(sym=sym), headers=YF_HEADERS, timeout=60,
                     params={"interval": "1d", "period1": 0, "period2": int(time.time()),
                             "events": "div,split"})   # range=max silently returns MONTHLY bars
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    df = pd.DataFrame({"date": pd.to_datetime(res["timestamp"], unit="s", utc=True)
                       .tz_convert("America/New_York").date, "close": q["close"], "adjclose": adj})
    ev = res.get("events", {})
    return df.dropna(subset=["close"]), {"n_dividends": len(ev.get("dividends", {})),
                                         "splits": [f"{pd.to_datetime(int(k), unit='s').date()} {v.get('splitRatio')}"
                                                    for k, v in ev.get("splits", {}).items()]}


def probe(sym: str) -> dict:
    ts, pages = ts_history(sym)
    yf, ev = yahoo(sym)
    out = {"sym": sym, "ts_pages": pages, "ts_rows": len(ts),
           "ts_first": str(ts["date"].min()) if len(ts) else None,
           "ts_last": str(ts["date"].max()) if len(ts) else None,
           "yahoo_first": str(yf["date"].min()), "yahoo_rows": len(yf), **ev}
    if len(ts):
        m = ts.merge(yf, on="date", suffixes=("_ts", "_yf"))
        r_close = (m["close_ts"] / m["close_yf"] - 1).abs()
        r_adj = (m["close_ts"] / m["adjclose"] - 1).abs()
        out.update({"overlap_days": len(m),
                    "median_absdiff_vs_yahoo_close": round(float(r_close.median()), 6),
                    "p99_absdiff_vs_yahoo_close": round(float(r_close.quantile(0.99)), 6),
                    "median_absdiff_vs_yahoo_adjclose": round(float(r_adj.median()), 6),
                    "early_absdiff_vs_adjclose": round(float(r_adj.head(250).median()), 6)})
    return out


def main() -> int:
    syms = sys.argv[1:] or (ALL_SYMBOLS + TRADING_ALTERNATES + AUDIT_ONLY)
    rows = []
    for s in syms:
        try:
            rows.append(probe(s))
        except Exception as e:                       # record, keep going
            rows.append({"sym": s, "error": repr(e)[:300]})
        print(json.dumps(rows[-1]), flush=True)
        time.sleep(0.5)
    if not sys.argv[1:]:
        p = ROOT / f"_bmad-output/etftm1_depth_probe_{date.today():%Y%m%d}.json"
        p.write_text(json.dumps(rows, indent=1))
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
