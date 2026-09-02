"""Pull authoritative broker fills from TopstepX/ProjectX for the §A8.2 Part A sample.

**Read-only.** Places no orders. Only calls `/Account/search` and `/Trade/search`,
both of which are searches.

Why this exists
---------------
Part A compares a simulator's fill price against a *real* fill at a *real* instant, so
it needs a fill record with a trustworthy timestamp. The candidates on disk are not
equal:

* ``data/mim_nb/orders.csv`` -- the mim-nb bot's own order log. PLACE/FILL/CANCEL rows
  with microsecond timestamps and executed prices. **Authoritative**: every one of the
  10 mim-nb fills ProjectX still retains matches this file's ``order_id`` exactly.
* ``data/trades.db`` -- bar-level trade rows. Prices are broadly right, but the
  **timestamps are not fill times**: the 2026-06-25 mim-nb row records 19:34 @ 29318.50
  while ``orders.csv`` shows that trade filled at 14:00 @ 29359.5. Simulating the
  recorded minute prices the wrong moment. Not usable for Part A.
* ``data/gap_fade/fills.csv`` -- date resolution only, no fill time. Not usable.
* **ProjectX** (this tool) -- the broker's own execution records. The only source of
  authoritative fills for ``trader-yank``, which keeps no local order log.

Validation performed 2026-09-02: 12 of 12 ProjectX fills fall inside the CME trade tape
(schema ``mbo``, action ``T``) within +/-2 s of their own timestamp -- 0 misses. There is
no price-basis offset between the broker records and the CME data.

Retention caveat
----------------
ProjectX returns fills only for the **currently active** account, and only recent ones:
as of 2026-09-02 that is 2026-08-13 -> 08-28 on account 26556101. The pre-reset combine
accounts (23884932 and earlier) return zero rows. So this **supplements**
``orders.csv``; it cannot replace it for the earlier parity window.

Usage
-----
    PYTHONPATH=. .venv/bin/python tools/pull_broker_fills.py \
        --start 2026-06-11 --end 2026-08-29 --out data/mim_nb/projectx_fills.json

Credentials come from ``.projectx_api_key`` (username on line 1, API key on line 2;
``#`` comments ignored) -- gitignored, never logged by this tool.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from src.research.projectx_auth import ProjectXAuth

BASE_URL = "https://api.topstepx.com/api"
KEY_FILE = ".projectx_api_key"


async def _accounts(client: httpx.AsyncClient, headers: dict) -> list[dict]:
    resp = await client.post(
        f"{BASE_URL}/Account/search",
        json={"onlyActiveAccounts": False},
        headers=headers,
    )
    resp.raise_for_status()
    accounts: list[dict] = resp.json().get("accounts", [])
    return accounts


async def _fills(
    client: httpx.AsyncClient, headers: dict, account_id: int, start: str, end: str
) -> list[dict]:
    resp = await client.post(
        f"{BASE_URL}/Trade/search",
        json={
            "accountId": int(account_id),
            "startTimestamp": start,
            "endTimestamp": end,
        },
        headers=headers,
    )
    if resp.status_code != 200:
        print(
            f"  account {account_id}: /Trade/search HTTP {resp.status_code}",
            file=sys.stderr,
        )
        return []
    payload = resp.json()
    fills: list[dict] = payload.get(
        "trades", payload if isinstance(payload, list) else []
    )
    return fills


async def run(start: str, end: str, out_path: Path) -> int:
    auth = ProjectXAuth.from_file(KEY_FILE)
    token = await auth.authenticate()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    collected: dict[int, dict] = {}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            accounts = await _accounts(client, headers)
            print(f"{len(accounts)} account(s) visible")
            for account in accounts:
                account_id = account.get("id")
                fills = await _fills(client, headers, int(account_id), start, end)
                print(
                    f"  {account_id} ({account.get('name')}): {len(fills)} fill(s) "
                    f"{start[:10]}..{end[:10]}"
                )
                for fill in fills:
                    collected[int(fill["id"])] = fill
    finally:
        await auth.cleanup()

    ordered = sorted(collected.values(), key=lambda f: str(f["creationTimestamp"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(ordered)} unique fill(s) -> {out_path}")
    if ordered:
        print(
            f"span: {ordered[0]['creationTimestamp'][:19]} .. "
            f"{ordered[-1]['creationTimestamp'][:19]}"
        )
        print(f"contracts: {sorted({f['contractId'] for f in ordered})}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", default="2026-06-11", metavar="YYYY-MM-DD")
    parser.add_argument("--end", default="2026-08-29", metavar="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        default="data/mim_nb/projectx_fills.json",
        metavar="PATH",
        help="JSON array of fill records (default: data/mim_nb/projectx_fills.json)",
    )
    args = parser.parse_args(argv)
    return asyncio.run(
        run(f"{args.start}T00:00:00Z", f"{args.end}T00:00:00Z", Path(args.out))
    )


if __name__ == "__main__":
    raise SystemExit(main())
