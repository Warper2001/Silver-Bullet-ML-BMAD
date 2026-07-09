#!/usr/bin/env python3
"""Flatten a stray position on the shared TradeStation SIM account.

One-shot operator tool (written for the 2026-07-06 stranded MNQU26 long: the
external flatten closed the live combine side but the pre-fix mirror never
closed the SIM copy). Places a single market order and verifies the resulting
position, printing before/after.

    .venv/bin/python tools/ts_sim_flatten_stray.py --symbol MNQU26 --qty 1 --action SELL
"""
import argparse
import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

ACCOUNT = "SIM2797251F"
ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"
BASE = f"https://sim-api.tradestation.com/v3/brokerage/accounts/{ACCOUNT}"


async def main(symbol: str, qty: int, action: str):
    auth = TradeStationAuthV3.from_file(".access_token")
    token = await auth.authenticate()
    h = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60) as http:
        pos = (await http.get(f"{BASE}/positions", headers=h)).json()
        match = [p for p in pos.get("Positions", []) if p.get("Symbol") == symbol]
        if not match:
            sys.exit(f"no open {symbol} position on {ACCOUNT} — nothing to flatten")
        p = match[0]
        print(f"before: {symbol} {p.get('LongShort')} {p.get('Quantity')} "
              f"@ {p.get('AveragePrice')} unreal ${p.get('UnrealizedProfitLoss')}")
        if abs(int(float(p.get("Quantity", 0)))) < qty:
            sys.exit(f"refusing: requested qty {qty} exceeds open position {p.get('Quantity')}")

        r = await http.post(ORDERS_URL, headers=h, json={
            "AccountID": ACCOUNT, "Symbol": symbol, "Quantity": str(qty),
            "OrderType": "Market", "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"}, "Route": "Intelligent",
        })
        r.raise_for_status()
        oid = r.json().get("Orders", [{}])[0].get("OrderID")
        print(f"placed market {action} {qty} {symbol} → order #{oid}; waiting 5s...")
        await asyncio.sleep(5)

        st = (await http.get(f"{BASE}/orders/{oid}", headers=h)).json()
        o = (st.get("Orders") or [{}])[0]
        leg = (o.get("Legs") or [{}])[0]
        print(f"order #{oid}: {o.get('StatusDescription')} exec={leg.get('ExecutionPrice')}")
        pos2 = (await http.get(f"{BASE}/positions", headers=h)).json()
        left = [p for p in pos2.get("Positions", []) if p.get("Symbol") == symbol]
        print("after:", f"{symbol} {left[0].get('LongShort')} {left[0].get('Quantity')} still open"
              if left else f"{symbol} FLAT ✓")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--action", choices=["BUY", "SELL"], required=True)
    a = ap.parse_args()
    asyncio.run(main(a.symbol, a.qty, a.action))
