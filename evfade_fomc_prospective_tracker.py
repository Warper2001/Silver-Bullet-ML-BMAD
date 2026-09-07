#!/usr/bin/env python3
"""
EVFADE-FOMC — prospective accrual tracker (OBSERVATION ONLY)

Pre-registration: _bmad-output/preregistration_evfade_fomc_prospective.md (sealed 2026-09-07,
BEFORE the first eligible event 2026-09-16).

WHY THIS SCRIPT REPORTS ALMOST NOTHING
--------------------------------------
Same principle as gap_velocity_prospective_tracker.py: this is a pure observation
instrument. It accrues the raw per-event facts (date, impulse, entry, exit, net P&L)
and reports ONLY the sample count and progress to target. It deliberately reports NO
running mean, NO profit factor and NO verdict, because watching those accrue is how a
decision rule gets chosen to fit the data it will be tested on.

The decision rule is already fixed in the seal (§5) and is evaluated ONLY at the N=15
interim look or the N>=30 final look — by a human, deliberately, not by this script.

WHAT IS BEING ACCRUED, AND THE HONEST CAVEAT
--------------------------------------------
The FOMC-fade direction is a POST-HOC SUBGROUP selected from N~9 scout observations
(seal §1). Its in-sample PF 4.66 is the observation that motivated the test, not
evidence for it. Worse, the seal's own power note (§6) records that N=30 can only
resolve a per-event effect of roughly $200+, while the scout measured +$59.63 --
so a null here will NOT be informative about a modest true effect.

Target N=30 is reached around Q2 2030 at 8 FOMC/year. ACCRUAL IS NOT PROGRESS.
A partial ledger is not evidence of anything.

Frozen spec (seal §3): reference = close of 14:00 ET bar; impulse over K=3 min;
FADE that impulse at the close of 14:03; hold M=30 min; exit at close of 14:33.
1 MNQ contract, $2/pt, cost $2.24/round-turn.

Reads TradeStation market data read-only. PLACES NO ORDERS. Touches no live trading
state. Idempotent: natural key = event date.

Usage:
    .venv/bin/python evfade_fomc_prospective_tracker.py
"""
from __future__ import annotations

import asyncio
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data.auth_v3 import TradeStationAuthV3  # noqa: E402

ET = ZoneInfo("America/New_York")
MD = "https://api.tradestation.com/v3/marketdata"

# ── Frozen by the seal — do not edit without a new pre-registration ──────────────
SYMBOL_ROOT = "MNQ"
POINT_VALUE = 2.0
K_MIN = 3                    # impulse window
M_MIN = 30                   # hold
COST_RT = 2.24               # scout cost basis (NOT a fresh measurement — seal §8.3)
EVENT_HHMM = (14, 0)         # FOMC statement 14:00 ET
N_INTERIM = 15               # PASS-only look
N_FINAL = 30
STOP_DATE = date(2030, 12, 31)
WINDOW_OPEN = date(2026, 9, 8)   # sealed 2026-09-07; nothing on/before may enter

CALENDAR = ROOT / "data/macro/fomc_calendar_forward.csv"
LEDGER = ROOT / "data/evfade_fomc/prospective_events.csv"
FIELDS = ["event_date", "ref_close", "impulse_pts", "direction",
          "entry_price", "exit_price", "gross_pnl_usd", "net_pnl_usd", "accrued_at"]


def load_ledger() -> dict[str, dict]:
    if not LEDGER.exists():
        return {}
    with LEDGER.open() as f:
        return {r["event_date"]: r for r in csv.DictReader(f)}


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def scheduled_events() -> list[date]:
    if not CALENDAR.exists():
        return []
    out = []
    with CALENDAR.open() as f:
        for r in csv.DictReader(f):
            if r.get("event") == "FOMC":
                out.append(date.fromisoformat(r["date"]))
    return sorted(out)


def _third_friday(y: int, m: int) -> date:
    d = date(y, m, 1)
    # weekday(): Mon=0 … Fri=4
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(days=14)


def front_contract(d: date) -> str:
    """MNQ front month on the quarterly cycle (H,M,U,Z).

    The front contract is the first quarterly month whose expiry (3rd Friday) is on or
    after `d`. A naive day-of-month roll gets this wrong for events in the ~2 weeks
    before expiry -- e.g. 2026-09-16 is still U26, not Z26 -- and the failure is silent
    because the back month happily returns plausible bars. Same class of error as the
    dead-front-month slippage artefact (MHGN26/PLN26): never infer the front contract
    from a proxy when the real rule is cheap to compute.
    """
    codes = {3: "H", 6: "M", 9: "U", 12: "Z"}
    for y in (d.year, d.year + 1):
        for mm in sorted(codes):
            if _third_friday(y, mm) >= d:
                return f"{SYMBOL_ROOT}{codes[mm]}{y % 100:02d}"
    raise ValueError(f"no front contract resolved for {d}")


def fetch_minutes(tok: str, sym: str, d: date) -> dict[datetime, float]:
    """1-min closes for the event afternoon, keyed by ET timestamp."""
    end = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
    try:
        r = requests.get(
            f"{MD}/barcharts/{sym}",
            headers={"Authorization": f"Bearer {tok}"},
            params={"unit": "Minute", "interval": 1, "barsback": 600,
                    "lastdate": end.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")},
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  fetch error {sym} {d}: {type(e).__name__}")
        return {}
    if r.status_code != 200:
        print(f"  fetch HTTP {r.status_code} for {sym} {d}")
        return {}
    out = {}
    for b in r.json().get("Bars", []):
        ts = datetime.fromisoformat(b["TimeStamp"].replace("Z", "+00:00")).astimezone(ET)
        out[ts.replace(second=0, microsecond=0)] = float(b["Close"])
    return out


def evaluate(closes: dict[datetime, float], d: date) -> dict | None:
    """Apply the frozen rule. Returns None if the day is ineligible (seal §3)."""
    def at(mins_after: int) -> float | None:
        t = datetime(d.year, d.month, d.day, EVENT_HHMM[0], EVENT_HHMM[1], tzinfo=ET) \
            + timedelta(minutes=mins_after)
        return closes.get(t)

    ref, imp, ext = at(0), at(K_MIN), at(K_MIN + M_MIN)
    if ref is None or imp is None or ext is None:
        return None

    impulse = imp - ref
    if impulse == 0:
        return None                      # no impulse to fade
    direction = "SHORT" if impulse > 0 else "LONG"
    sign = -1.0 if impulse > 0 else 1.0   # fade
    gross = (ext - imp) * sign * POINT_VALUE
    return {
        "event_date": d.isoformat(),
        "ref_close": f"{ref:.2f}",
        "impulse_pts": f"{impulse:.2f}",
        "direction": direction,
        "entry_price": f"{imp:.2f}",
        "exit_price": f"{ext:.2f}",
        "gross_pnl_usd": f"{gross:.2f}",
        "net_pnl_usd": f"{gross - COST_RT:.2f}",
        "accrued_at": datetime.now(ET).isoformat(timespec="seconds"),
    }


async def main() -> int:
    today = datetime.now(ET).date()
    ledger = load_ledger()
    events = scheduled_events()

    if not events:
        print("EVFADE-FOMC: calendar missing or empty — cannot accrue. "
              f"Expected {CALENDAR}")
        return 1

    # events that have already happened, are in the sealed window, and are unrecorded
    due = [d for d in events
           if d >= WINDOW_OPEN and d < today and d.isoformat() not in ledger]

    tok = None
    for d in due:
        if tok is None:
            auth = TradeStationAuthV3.from_file(str(ROOT / ".access_token"))
            tok = await auth.authenticate()
        sym = front_contract(d)
        row = evaluate(fetch_minutes(tok, sym, d), d)
        if row is None:
            print(f"  {d}: ineligible (bars missing across 14:00–14:33 ET) — not accrued")
            continue
        append_ledger(row)
        ledger[d.isoformat()] = row
        print(f"  {d}: accrued ({sym})")

    n = len(ledger)
    future = [d for d in events if d >= today]
    nxt = future[0].isoformat() if future else "NONE — extend the calendar"
    last_sched = events[-1]

    # ── Report ONLY count and progress. No mean, no PF, no verdict. (seal §7) ──
    print("\nEVFADE-FOMC prospective accrual (observation only)")
    print(f"  accrued events      : {n}")
    print(f"  interim look at     : {N_INTERIM}   ({'reached' if n >= N_INTERIM else 'not reached'})")
    print(f"  final look at       : {N_FINAL}   ({'reached' if n >= N_FINAL else 'not reached'})")
    print(f"  next scheduled FOMC : {nxt}")
    print(f"  stopping date       : {STOP_DATE}")
    if n >= N_FINAL:
        print("  ** N target reached — a HUMAN evaluates the sealed rule (seal §5). **")
    elif n >= N_INTERIM:
        print("  ** interim N reached — PASS-only look available (seal §4). **")
    if last_sched < today + timedelta(days=120):
        print(f"  !! calendar runs out {last_sched} — extend {CALENDAR.name} from "
              f"federalreserve.gov before then.")
    print("  (no mean, no PF, no verdict by design — accrual is not progress)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
