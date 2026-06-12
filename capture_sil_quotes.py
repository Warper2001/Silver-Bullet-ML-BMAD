"""
SIL slippage measurement — quote capture daemon (prereg addendum 3, 2026-06-12).

Polls TradeStation /v3/marketdata/quotes/SILN26,SIN26 every 5 s during
09:25-16:00 ET Mon-Fri and appends rows to data/quotes/sil_quote_capture.csv.
Outside the window it sleeps. Auto-exits after END_DATE (before the ~Jul 17
SIN26/SILN26 roll). Append-mode + per-row flush: safe to restart anytime.

Token handling: re-reads .access_token on every poll (stale-token lesson,
commit 988c393 — another process keeps the token file fresh).

Usage:
  nohup .venv/bin/python capture_sil_quotes.py > logs/sil_quote_capture.log 2>&1 &

Analysis (after >=5 qualifying sessions, frozen rule):
  .venv/bin/python analyze_sil_quotes.py
"""
import asyncio
import csv
import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from src.data.auth_v3 import TradeStationAuthV3

SYMBOLS = "SILN26,SIN26"
OUT_PATH = Path("data/quotes/sil_quote_capture.csv")
ET = ZoneInfo("America/New_York")
POLL_SECONDS = 5
WINDOW_START = time(9, 25)
WINDOW_END = time(16, 0)
END_DATE = date(2026, 6, 23)  # hard stop; roll is ~Jul 17
FIELDS = ["ts_utc", "symbol", "bid", "ask", "bid_size", "ask_size", "last", "spread"]

URL = f"https://api.tradestation.com/v3/marketdata/quotes/{SYMBOLS}"


def in_window(now_et: datetime) -> bool:
    return now_et.weekday() < 5 and WINDOW_START <= now_et.time() <= WINDOW_END


async def poll_once(client: httpx.AsyncClient, writer, fh) -> int:
    auth = TradeStationAuthV3.from_file(".access_token")  # re-read every poll
    token = await auth.authenticate()
    resp = await client.get(URL, headers={"Authorization": f"Bearer {token}"},
                            timeout=15.0)
    if resp.status_code != 200:
        print(f"{datetime.utcnow().isoformat()} HTTP {resp.status_code}: "
              f"{resp.text[:120]}", flush=True)
        return 0
    n = 0
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for q in resp.json().get("Quotes", []):
        try:
            bid, ask = q.get("Bid"), q.get("Ask")
            spread = (float(ask) - float(bid)) if bid and ask else ""
            writer.writerow({
                "ts_utc": ts, "symbol": q.get("Symbol"),
                "bid": bid, "ask": ask,
                "bid_size": q.get("BidSize"), "ask_size": q.get("AskSize"),
                "last": q.get("Last"), "spread": spread,
            })
            n += 1
        except Exception as e:
            print(f"{ts} parse error: {e}", flush=True)
    fh.flush()
    return n


async def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT_PATH.exists()
    rows_today, last_day = 0, None
    with open(OUT_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()
        print(f"capture started → {OUT_PATH} (auto-stop {END_DATE})", flush=True)
        async with httpx.AsyncClient() as client:
            while True:
                now_et = datetime.now(ET)
                if now_et.date() > END_DATE:
                    print(f"{now_et} past END_DATE — exiting", flush=True)
                    return
                if in_window(now_et):
                    if now_et.date() != last_day:
                        print(f"=== session {now_et.date()} ===", flush=True)
                        last_day, rows_today = now_et.date(), 0
                    try:
                        rows_today += await poll_once(client, writer, fh)
                    except Exception as e:
                        print(f"{now_et} poll error: {e}", flush=True)
                    if rows_today and rows_today % 1200 == 0:
                        print(f"{now_et} rows today: {rows_today}", flush=True)
                    await asyncio.sleep(POLL_SECONDS)
                else:
                    await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
