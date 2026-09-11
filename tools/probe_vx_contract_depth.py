import asyncio
import sys

import requests

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402


def bars(tok, sym, n=400):
    r = requests.get(f"https://api.tradestation.com/v3/marketdata/barcharts/{sym}",
                     headers={"Authorization": f"Bearer {tok}"},
                     params={"unit": "Daily", "barsback": n}, timeout=45)
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {r.text[:70]}"
    b = r.json().get("Bars", [])
    return (f"{len(b):4d} bars, {b[0]['TimeStamp'][:10]} → {b[-1]['TimeStamp'][:10]}"
            if b else "0 bars")


async def main():
    auth = TradeStationAuthV3.from_file("/root/Silver-Bullet-ML-BMAD/.access_token")
    tok = await auth.authenticate()
    print("AUTH OK\n")

    print("=== EXPIRED full-VX contracts (needed for a roll-yield study) ===")
    for sym in ["VXQ26", "VXN26", "VXM26", "VXJ26", "VXF26",
                "VXU25", "VXM25", "VXZ24", "VXM23", "VXH20"]:
        print(f"  {sym:8} {bars(tok, sym)}")

    print("\n=== EXPIRED Mini-VIX contracts ===")
    for sym in ["VXMQ26", "VXMN26", "VXMM26", "VXMU25", "VXMZ24"]:
        print(f"  {sym:8} {bars(tok, sym)}")

    print("\n=== Continuous back-adjusted variants ===")
    for sym in ["@VX", "@VX=", "@VX2", "@VXM", "@VX.D"]:
        print(f"  {sym:8} {bars(tok, sym, 200)}")


asyncio.run(main())
