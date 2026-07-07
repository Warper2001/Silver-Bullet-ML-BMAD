#!/usr/bin/env python3
"""Account order watchdog — tripwire for orders nobody placed.

Background (2026-07-06): an untagged external order (#3229490103) flattened a live
MIM-NB position and canceled its stop; no local process placed it and the bot took
29 minutes to notice (and misbooked it). Alex's decision: park the anomaly, but
alarm on any recurrence. This tool IS that alarm.

Read-only. Polls ProjectX Order/search for the combine account, diffs every order
ID against local records (MIM's orders.csv + order IDs greppable from both bots'
logs + a persistent seen-file), and appends any unrecognized order to
data/combine_joint/order_watchdog_alerts.csv while logging loudly.

Usage:
    .venv/bin/python tools/account_order_watchdog.py            # single pass
    .venv/bin/python tools/account_order_watchdog.py --loop 300 # poll every 300s

Known-benign IDs (the parked 07-06 incident) live in _ACKNOWLEDGED below.
Not deployed as a service yet — deployment pending Alex's go.
"""
import argparse
import asyncio
import csv
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.research.projectx_auth import ProjectXAuth  # noqa: E402

ACCOUNT_ID = 23884932
MIM_ORDERS_CSV = ROOT / "data/mim_nb/orders.csv"
BOT_LOGS = [ROOT / "logs/mim_nb_live.log", ROOT / "logs/yank_streaming_working.log"]
ALERTS_CSV = ROOT / "data/combine_joint/order_watchdog_alerts.csv"
SEEN_FILE = ROOT / "data/combine_joint/order_watchdog_seen.txt"
LOOKBACK_HOURS = 48

# Parked 2026-07-06 incident (halt_review_mim_nb_parity_20260707.md) — acknowledged.
_ACKNOWLEDGED = {3229490103}

_ID_RE = re.compile(r"(3\d{9})")


def local_known_ids() -> set:
    known = set(_ACKNOWLEDGED)
    if MIM_ORDERS_CSV.exists():
        with open(MIM_ORDERS_CSV) as f:
            for row in csv.DictReader(f):
                if str(row.get("order_id", "")).isdigit():
                    known.add(int(row["order_id"]))
    for logf in BOT_LOGS:
        if logf.exists():
            with open(logf, errors="ignore") as f:
                for line in f:
                    for m in _ID_RE.findall(line):
                        known.add(int(m))
    if SEEN_FILE.exists():
        for line in SEEN_FILE.read_text().split():
            if line.isdigit():
                known.add(int(line))
    return known


async def fetch_orders() -> list:
    auth = ProjectXAuth.from_file(str(ROOT / ".projectx_api_key"))
    tok = await auth.authenticate()
    start = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://api.topstepx.com/api/Order/search",
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            json={"accountId": ACCOUNT_ID, "startTimestamp": start},
        )
        r.raise_for_status()
        return r.json().get("orders", [])


def alert(order: dict):
    is_new = not ALERTS_CSV.exists()
    with open(ALERTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["detected_utc", "order_id", "created", "type", "side", "size",
                        "limit", "stop", "fill_vol", "fill_px", "status", "tag"])
        w.writerow([datetime.now(timezone.utc).isoformat(), order["id"],
                    order.get("creationTimestamp"), order.get("type"), order.get("side"),
                    order.get("size"), order.get("limitPrice"), order.get("stopPrice"),
                    order.get("fillVolume"), order.get("filledPrice"),
                    order.get("status"), order.get("customTag")])
    print(f"🚨 UNRECOGNIZED ORDER on acct {ACCOUNT_ID}: id={order['id']} "
          f"{order.get('creationTimestamp')} side={order.get('side')} size={order.get('size')} "
          f"fill_px={order.get('filledPrice')} tag={order.get('customTag')}", flush=True)


def run_pass() -> int:
    known = local_known_ids()
    orders = asyncio.run(fetch_orders())
    unrec = [o for o in orders if o["id"] not in known]
    for o in unrec:
        alert(o)
    if unrec:
        with open(SEEN_FILE, "a") as f:
            f.write("".join(f"{o['id']}\n" for o in unrec))
    print(f"{datetime.now(timezone.utc).isoformat()} pass: {len(orders)} orders "
          f"({LOOKBACK_HOURS}h window), {len(unrec)} unrecognized", flush=True)
    return len(unrec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="poll continuously at this interval (default: single pass)")
    args = ap.parse_args()
    if not args.loop:
        sys.exit(1 if run_pass() else 0)
    while True:
        try:
            run_pass()
        except Exception as e:  # keep the tripwire alive through transient API errors
            print(f"watchdog pass error (retrying next cycle): {e}", flush=True)
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
