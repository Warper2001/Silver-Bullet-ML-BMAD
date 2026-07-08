#!/usr/bin/env python3
"""Best-effort backfill REPORT of realized TS SIM fills for gap-fade's historical
trades (pre-fills.csv era). Order IDs were not persisted after each trade closed,
so this matches heuristically: TradeStation historical orders on SIM2797251F,
filtered to the gap-fade symbol and each trade date from data/gap_fade/trades.csv.

Writes data/gap_fade/fills_backfill_report.csv (NOT the hash-chained fills.csv —
the forward chain stays pure) and prints a per-trade summary. The shared SIM
account also carries MIM/YANK mirror orders on MNQ, so rows are candidates for
manual confirmation, not gospel: gap-fade entries are market orders at ~09:31 ET.

    .venv/bin/python tools/gap_fade_fill_backfill.py
"""
import asyncio
import csv
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

ACCOUNT = "SIM2797251F"
TRADES = ROOT / "data/gap_fade/trades.csv"
OUT = ROOT / "data/gap_fade/fills_backfill_report.csv"
HIST_URL = f"https://sim-api.tradestation.com/v3/brokerage/accounts/{ACCOUNT}/historicalorders"


async def main():
    with open(TRADES) as f:
        trades = list(csv.DictReader(f))
    if not trades:
        sys.exit("no gap-fade trades to backfill")
    since = min(t["date_et"] for t in trades)
    dates = {t["date_et"] for t in trades}

    auth = TradeStationAuthV3.from_file(".access_token")
    await auth.authenticate()
    token = await auth.authenticate()
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.get(HIST_URL, params={"since": since},
                           headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        orders = r.json().get("Orders", [])

    rows = []
    for o in orders:
        leg = (o.get("Legs") or [{}])[0]
        sym = leg.get("Symbol") or o.get("Symbol") or ""
        opened = (o.get("OpenedDateTime") or "")[:10]
        if not sym.startswith("MNQ") or opened not in dates:
            continue
        rows.append({
            "date_et": opened,
            "order_id": o.get("OrderID"),
            "opened": o.get("OpenedDateTime"),
            "closed": o.get("ClosedDateTime"),
            "type": o.get("OrderType"),
            "side": leg.get("BuyOrSell"),
            "qty": leg.get("QuantityOrdered"),
            "status": o.get("StatusDescription") or o.get("Status"),
            "exec_price": leg.get("ExecutionPrice"),
        })

    rows.sort(key=lambda x: (x["date_et"], str(x["opened"])))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["date_et", "order_id"])
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} candidate MNQ orders across {len(dates)} gap-fade trade dates "
          f"→ {OUT.relative_to(ROOT)}")
    for d in sorted(dates):
        day = [x for x in rows if x["date_et"] == d]
        print(f"  {d}: {len(day)} orders " +
              "; ".join(f"#{x['order_id']} {x['side']} {x['type']} @{x['exec_price']} ({x['status']})"
                        for x in day[:6]))


if __name__ == "__main__":
    asyncio.run(main())
