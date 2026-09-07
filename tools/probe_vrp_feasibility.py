"""VRP path-2 probe: how deep is VIX/VX history, and is there a full term structure?"""
import asyncio
import sys

import requests

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

MD = "https://api.tradestation.com/v3/marketdata"


def bars(tok, sym, n, unit="Daily"):
    try:
        r = requests.get(f"{MD}/barcharts/{sym}",
                         headers={"Authorization": f"Bearer {tok}"},
                         params={"unit": unit, "barsback": n}, timeout=60)
    except Exception as e:  # noqa: BLE001
        return None, f"EXC {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:90]}"
    b = r.json().get("Bars", [])
    if not b:
        return 0, "0 bars"
    return len(b), f"{len(b):5d} bars, {b[0]['TimeStamp'][:10]} → {b[-1]['TimeStamp'][:10]}"


async def main():
    auth = TradeStationAuthV3.from_file("/root/Silver-Bullet-ML-BMAD/.access_token")
    tok = await auth.authenticate()
    print("AUTH OK\n")

    print("=== History depth (daily) ===")
    for sym in ["$VIX.X", "$VXN.X", "@VX", "QQQ", "$SPX.X"]:
        for n in (5000, 2500):
            cnt, msg = bars(tok, sym, n)
            print(f"  {sym:8} barsback={n:5d}  {msg}")
            if cnt:
                break

    print("\n=== VX term structure: contract ladder ===")
    months = [("U", 26), ("V", 26), ("X", 26), ("Z", 26),
              ("F", 27), ("G", 27), ("H", 27), ("J", 27), ("K", 27)]
    live = 0
    for m, y in months:
        sym = f"VX{m}{y}"
        cnt, msg = bars(tok, sym, 400)
        print(f"  {sym:8} {msg}")
        if cnt:
            live += 1
    print(f"  -> {live} VX contracts return data (term structure needs >=3-4)")

    print("\n=== Intraday availability (for a same-day signal) ===")
    for sym in ["$VIX.X", "@VX"]:
        cnt, msg = bars(tok, sym, 300, unit="Minute")
        print(f"  {sym:8} 1-min  {msg}")


asyncio.run(main())
