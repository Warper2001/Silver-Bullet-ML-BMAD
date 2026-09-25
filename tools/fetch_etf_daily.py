#!/usr/bin/env python3
"""ETFTM-1 A0.3 — fetch raw daily data (read-only market-data GETs; places nothing).

Writes, per symbol:
  data/etf_daily/raw/ts/<SYM>.csv     TradeStation Daily bars (split-adjusted, price-only; paged past 5,000)
  data/etf_daily/raw/yahoo/<SYM>.csv  Yahoo daily close / adjclose
  data/etf_daily/raw/yahoo/<SYM>_div.csv, <SYM>_split.csv   Yahoo dividend and split events
and the risk-free series data/etf_daily/raw/yahoo/IRX.csv (^IRX, 13-week T-bill yield, %).
Token handling as in tools/probe_etf_depth.py: reads .access_token per request, never refreshes.

Run: .venv/bin/python tools/fetch_etf_daily.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.etftm1_universe import ALL_SYMBOLS, AUDIT_ONLY, RISK_FREE_YAHOO, TRADING_ALTERNATES  # noqa: E402
from tools.probe_etf_depth import YF, YF_HEADERS, ts_get  # noqa: E402

OUT = Path("/root/Silver-Bullet-ML-BMAD/data/etf_daily/raw")


def ts_raw(sym: str, max_pages: int = 8) -> pd.DataFrame:
    frames, bars, pages = [], ts_get(sym, {"barsback": 5000}), 0
    while bars and pages < max_pages:
        pages += 1
        frames.append(pd.DataFrame(bars))
        earliest = frames[-1]["TimeStamp"].min()
        bars = [b for b in ts_get(sym, {"barsback": 5000, "lastdate": earliest}) if b["TimeStamp"] < earliest]
    df = pd.concat(frames).drop_duplicates("TimeStamp").sort_values("TimeStamp")
    df["date"] = pd.to_datetime(df["TimeStamp"], utc=True).dt.tz_convert("America/New_York").dt.date
    cols = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "TotalVolume": "volume"}
    df = df.rename(columns=cols)
    return df[["date", "TimeStamp", *cols.values()]]


def yahoo_raw(sym: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = requests.get(YF.format(sym=sym), headers=YF_HEADERS, timeout=60,
                     params={"interval": "1d", "period1": 0, "period2": int(time.time()), "events": "div,split"})
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    d = lambda ts: pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").date  # noqa: E731
    px = pd.DataFrame({"date": d(res["timestamp"]), "close": q["close"], "adjclose": adj}).dropna(subset=["close"])
    ev = res.get("events", {})
    div = pd.DataFrame([{"date": d([int(k)])[0], "amount": v["amount"]} for k, v in ev.get("dividends", {}).items()],
                       columns=["date", "amount"])
    spl = pd.DataFrame([{"date": d([int(k)])[0], "ratio": v.get("splitRatio"),
                         "numerator": v.get("numerator"), "denominator": v.get("denominator")}
                        for k, v in ev.get("splits", {}).items()],
                       columns=["date", "ratio", "numerator", "denominator"])
    return px, div.sort_values("date"), spl.sort_values("date")


def main() -> int:
    (OUT / "ts").mkdir(parents=True, exist_ok=True)
    (OUT / "yahoo").mkdir(parents=True, exist_ok=True)
    for sym in ALL_SYMBOLS + TRADING_ALTERNATES + AUDIT_ONLY:
        ts = ts_raw(sym)
        ts.to_csv(OUT / "ts" / f"{sym}.csv", index=False)
        px, div, spl = yahoo_raw(sym)
        px.to_csv(OUT / "yahoo" / f"{sym}.csv", index=False)
        div.to_csv(OUT / "yahoo" / f"{sym}_div.csv", index=False)
        spl.to_csv(OUT / "yahoo" / f"{sym}_split.csv", index=False)
        print(f"{sym:5} ts {len(ts):5} ({ts['date'].min()}..{ts['date'].max()})  yahoo {len(px):5}  div {len(div):3}  split {len(spl)}", flush=True)
        time.sleep(0.5)
    irx, _, _ = yahoo_raw(RISK_FREE_YAHOO)
    irx[["date", "close"]].rename(columns={"close": "yield_pct"}).to_csv(OUT / "yahoo" / "IRX.csv", index=False)
    print(f"^IRX {len(irx)} rows ({irx['date'].min()}..{irx['date'].max()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
