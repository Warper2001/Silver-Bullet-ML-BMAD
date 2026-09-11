"""Fetch per-contract-month daily settle history + expirations for TSC-1.

Pre-registration: _bmad-output/preregistration_term_structure_carry.md

Pulls TradeStation `barcharts/{contract_symbol}` (Daily) and
`symbols/{contract_symbol}` (for ExpirationDate) for every plausible
contract-month symbol of each frozen-universe root, across the frozen
sample window. Writes one row per (root, contract_symbol, date) to
data/term_structure/raw_contract_bars.csv and one row per (root,
contract_symbol) to data/term_structure/contract_meta.csv.

Read-only against the broker (market data GETs only, no orders). Uses the
main checkout's .access_token directly (no client id/secret needed for
existing-token market-data reads).
"""
from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

import httpx

MD_BASE = "https://api.tradestation.com/v3/marketdata"
TOKEN_FILE = "/root/Silver-Bullet-ML-BMAD/.access_token"

MONTH_CODES = "FGHJKMNQUVXZ"  # Jan..Dec

# Frozen universe (preregistration_term_structure_carry.md)
UNIVERSE = {
    "MGC": list(range(1, 13)),          # all 12 months
    "SIL": [3, 5, 7, 9, 12],            # H,K,N,U,Z
    "MHG": list(range(1, 13)),          # all 12 months
    "PL": [1, 4, 7, 10],                # F,J,N,V
    "MNQ": [3, 6, 9, 12],               # H,M,U,Z
}

YEARS = range(2020, 2027)  # probe a little before the 2021 dev-window start
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "term_structure"
BARS_CSV = OUT_DIR / "raw_contract_bars.csv"
META_CSV = OUT_DIR / "contract_meta.csv"

CONCURRENCY = 8


def contract_symbol(root: str, month: int, year: int) -> str:
    return f"{root}{MONTH_CODES[month - 1]}{year % 100:02d}"


async def fetch_one(http: httpx.AsyncClient, headers: dict, root: str, sym: str,
                     sem: asyncio.Semaphore) -> tuple[str, str, list[dict], str | None, str | None]:
    async with sem:
        bars: list[dict] = []
        expiry = None
        point_value = None
        try:
            r = await http.get(f"{MD_BASE}/barcharts/{sym}",
                                params={"interval": "1", "unit": "Daily", "barsback": "900"},
                                headers=headers)
            if r.status_code == 200:
                bars = r.json().get("Bars", [])
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: bars EXC {e!r}")
        if bars:
            try:
                r2 = await http.get(f"{MD_BASE}/symbols/{sym}", headers=headers)
                if r2.status_code == 200:
                    syms = r2.json().get("Symbols", [])
                    if syms:
                        expiry = syms[0].get("ExpirationDate")
                        point_value = (syms[0].get("PriceFormat") or {}).get("PointValue")
            except Exception as e:  # noqa: BLE001
                print(f"  {sym}: meta EXC {e!r}")
        return root, sym, bars, expiry, point_value


async def main() -> None:
    token = Path(TOKEN_FILE).read_text().strip()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = []
    async with httpx.AsyncClient(timeout=30) as http:
        for root, months in UNIVERSE.items():
            for year in YEARS:
                for month in months:
                    sym = contract_symbol(root, month, year)
                    tasks.append(fetch_one(http, headers, root, sym, sem))
        print(f"Probing {len(tasks)} contract-month symbols across {len(UNIVERSE)} roots...")
        results = await asyncio.gather(*tasks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_bars = 0
    n_contracts_with_data = 0
    with BARS_CSV.open("w", newline="") as fb, META_CSV.open("w", newline="") as fm:
        bw = csv.writer(fb)
        bw.writerow(["root", "symbol", "date", "close"])
        mw = csv.writer(fm)
        mw.writerow(["root", "symbol", "expiration_date", "point_value", "n_bars", "first_date", "last_date"])
        for root, sym, bars, expiry, point_value in results:
            if not bars:
                continue
            n_contracts_with_data += 1
            dates = []
            for b in bars:
                ts = b.get("TimeStamp", "")
                date = ts[:10] if ts else ""
                close = b.get("Close")
                if not date or close in (None, ""):
                    continue
                bw.writerow([root, sym, date, close])
                dates.append(date)
                n_bars += 1
            if dates:
                mw.writerow([root, sym, expiry or "", point_value or "", len(dates), min(dates), max(dates)])

    print(f"Contracts with data: {n_contracts_with_data} / {len(results)} probed")
    print(f"Total bar-rows written: {n_bars}")
    print(f"-> {BARS_CSV}")
    print(f"-> {META_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
