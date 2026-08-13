#!/usr/bin/env python3
"""One-shot, append-only repair of the gap-fade ledger rows lost on 2026-08-06.

WHAT WAS LOST
-------------
`data/gap_fade/{trades,decisions,fills}.csv` were git-tracked until 84bb7b3. At
2026-08-06 20:00:36Z a branch checkout (reflog: main -> fix/s26-combine-unit-drift,
20:00:39Z) reverted all three to commit 1910acb, whose content stopped at 2026-08-04.
Everything appended on 08-05 and 08-06 was destroyed. `84bb7b3` untracked the files
74 minutes later, so the trigger is gone, but the hole was never filled.

Lost, and recovered here from sources that survived independently:

  decisions.csv  2026-08-05  NO_SETUP           reconstructed from TS 1-min bars
  decisions.csv  2026-08-06  ENTERED long       corroborated by trades.db metadata
  trades.csv     2026-08-06  +$646.00 TP fill   verbatim from trades.db (contemporaneous)
  fills.csv      2026-08-06  broker executions  from TS SIM historicalorders

WHY THESE NUMBERS CAN BE TRUSTED
--------------------------------
TradeStation revises its own 1-minute history, so a bar-derived reconstruction is
not evidence by itself: re-deriving the *2026-08-04* decision today yields rth_open
29224.5 against the 29223.25 the bot recorded live — 1.25pt of drift on a row we can
check. So bar data is used only where it cannot change an outcome:

  2026-08-06 — every recomputed field (gap_pct 1.089, gap_abs 323.0, prior_close
    29667.5, rth_open 29344.5, target 29667.5, stop 28698.5) matched the metadata
    trades.db recorded live at 15:01:01Z, to the cent. Those values are therefore
    taken from trades.db, not from bars, and re-checked against it at apply time.

  2026-08-05 — no trade, so no contemporaneous record exists and bars are all there
    is. The verdict is robust anyway: gap_pct 0.288% against a 0.500% threshold, a
    margin ~170x the observed 1.25pt revision drift. The action (NO_SETUP) is safe;
    the two gap figures in that row are reconstructed and are marked as such in the
    manifest.

WHAT THIS DOES NOT DO
---------------------
It does not rewrite, reorder, re-chain, or delete anything. Recovered rows are
appended at the tail, so they appear after 2026-08-12 in date terms — that
out-of-order position is deliberate provenance: an append-only file that shows an
08-05 row arriving last is telling the truth about when it was written.

In particular it does NOT remove the duplicate 2026-06-25 row in trades.csv
(+$1,390.50 counted twice). A row cannot be withdrawn from an append-only chain
without destroying the chain, which is the one thing the chain is for. After this
repair trades.csv still overstates by $1,390.50 and `data/trades.db` remains the
arithmetic authority; `tools/verify_chain.py --reconcile` reports the duplicate as
a registered permanent scar.

Usage:
    .venv/bin/python tools/gap_fade_ledger_repair.py            # dry run (default)
    .venv/bin/python tools/gap_fade_ledger_repair.py --apply

Exit codes: 0 = clean (or dry run) · 1 = refused / verification failed
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from tools.verify_chain import verify  # noqa: E402  (read-only chain walker)

DATA = BASE / "data" / "gap_fade"
SERVICE = "trader-gap-fade"
MANIFEST = DATA / "ledger_repair_20260812.csv"
INCIDENT = "_bmad-output/ledger_incident_20260806_gap_fade.md"

# Field order must match ChainedCsv construction in src/research/gap_fade_live.py.
SCHEMA = {
    "decisions.csv": ["date_et", "dow", "gap_pct", "gap_abs_pts",
                      "prior_close", "rth_open", "action", "detail"],
    "trades.csv":    ["date_et", "dir", "gap_pct", "gap_abs_pts",
                      "entry", "exit_px", "target", "stop",
                      "outcome", "pnl_pts", "pnl_usd"],
    "fills.csv":     ["date_et", "dir", "outcome", "entry_id", "entry_exec",
                      "exit_role", "exit_id", "exit_exec", "qty",
                      "realized_pnl_usd", "modeled_pnl_usd", "delta_usd"],
}

# (file, row, source, note) — values recorded exactly as the bot would have written
# them (round(x, 2); gap_pct as round(pct * 100, 3)).
RECOVERED = [
    ("decisions.csv",
     {"date_et": "2026-08-05", "dow": "Wed", "gap_pct": 0.288, "gap_abs_pts": 86.0,
      "prior_close": 29875.0, "rth_open": 29961.0, "action": "NO_SETUP", "detail": ""},
     "reconstructed from TradeStation 1-min bars (MNQU26, RTH 09:30-16:00 ET)",
     "gap 0.288% < 0.500% threshold; verdict robust to the observed 1.25pt bar drift"),

    ("decisions.csv",
     {"date_et": "2026-08-06", "dow": "Thu", "gap_pct": 1.089, "gap_abs_pts": 323.0,
      "prior_close": 29667.5, "rth_open": 29344.5, "action": "ENTERED", "detail": "long"},
     "trades.db metadata (contemporaneous, written 2026-08-06T15:01:01Z)",
     "every field independently reproduced from bars; exact match"),

    ("trades.csv",
     {"date_et": "2026-08-06", "dir": "L", "gap_pct": 1.089, "gap_abs_pts": 323.0,
      "entry": 29344.5, "exit_px": 29667.5, "target": 29667.5, "stop": 28698.5,
      "outcome": "fill", "pnl_pts": 323.0, "pnl_usd": 646.0},
     "trades.db row id for trader-gap-fade @ 2026-08-06 (contemporaneous)",
     "re-verified against trades.db at apply time"),

    ("fills.csv",
     {"date_et": "2026-08-06", "dir": "L", "outcome": "fill",
      "entry_id": "965760604", "entry_exec": 29368.0,
      "exit_role": "tp", "exit_id": "965760598", "exit_exec": 29667.75, "qty": 1.0,
      "realized_pnl_usd": 599.5, "modeled_pnl_usd": 646.0, "delta_usd": -46.5},
     "TradeStation SIM historicalorders, account SIM2797251F",
     "market entry 13:30:02Z @29368.0; TP limit 15:00:03Z @29667.75; "
     "StopMarket sibling 965760601 UROut. 23.5pt entry slippage vs the modeled open"),
]


def refuse(msg: str) -> int:
    print(f"REFUSED: {msg}")
    return 1


def chain_state() -> dict:
    """{filename: (rows, first_bad_rownum)} for the three ledgers."""
    out = {}
    for name in SCHEMA:
        n, bad_n, _bad_key, err = verify(DATA / name)
        if err:
            out[name] = ("ERR", err)
        else:
            out[name] = (n, bad_n)
    return out


def db_trade_2026_08_06():
    con = sqlite3.connect(f"file:{BASE / 'data/trades.db'}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT direction, entry_price, exit_price, pnl, exit_reason, metadata "
        "FROM trades WHERE trader_id='trader-gap-fade' AND date(timestamp)='2026-08-06'"
    ).fetchall()
    return rows


def cross_check_trade(row: dict) -> list:
    """Return a list of mismatch strings between the recovered row and trades.db."""
    rows = db_trade_2026_08_06()
    if len(rows) != 1:
        return [f"expected exactly 1 trades.db row for 2026-08-06, found {len(rows)}"]
    direction, entry, exit_px, pnl, reason, meta_json = rows[0]
    meta = json.loads(meta_json or "{}")
    checks = [
        ("dir", row["dir"], "L" if direction == "L" else "S"),
        ("entry", row["entry"], entry),
        ("exit_px", row["exit_px"], exit_px),
        ("pnl_usd", row["pnl_usd"], pnl),
        ("outcome", row["outcome"], reason),
        ("gap_pct", row["gap_pct"], meta.get("gap_pct")),
        ("gap_abs_pts", row["gap_abs_pts"], meta.get("gap_abs_pts")),
        ("target", row["target"], meta.get("target")),
        ("stop", row["stop"], meta.get("stop")),
    ]
    return [f"{k}: recovered {a!r} != trades.db {b!r}" for k, a, b in checks if a != b]


def existing_keys(name: str) -> set:
    path = DATA / name
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {r.get("date_et") for r in csv.DictReader(fh)}


def append_row(name: str, row: dict) -> str:
    """Append one row using the on-disk head. Returns the new chain value."""
    import hashlib
    path = DATA / name
    fields = SCHEMA[name] + ["chain"]
    head = "GENESIS"
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            head = r.get("chain") or head
    payload = "|".join(str(row.get(k, "")) for k in SCHEMA[name])
    head = hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]
    out = dict(row)
    out["chain"] = head
    with path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore").writerow(out)
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually append (default is a dry run that writes nothing)")
    args = ap.parse_args()

    # ── Safety gates. All of these fail closed. ──────────────────────────────
    active = subprocess.run(["systemctl", "is-active", "--quiet", SERVICE]).returncode == 0
    state = DATA / "state.json"
    open_trade = False
    if state.exists():
        try:
            open_trade = bool(json.loads(state.read_text()).get("trade_open"))
        except Exception:
            return refuse(f"{state} is unreadable — cannot prove the bot is flat")

    if args.apply:
        if active:
            return refuse(
                f"{SERVICE} is ACTIVE. Appending now would leave its in-memory chain "
                f"head stale and break the chain on its next write — the exact defect "
                f"this repair exists to undo.\n"
                f"  sudo systemctl stop {SERVICE}\n"
                f"  .venv/bin/python tools/gap_fade_ledger_repair.py --apply\n"
                f"  sudo systemctl start {SERVICE}")
        if open_trade:
            return refuse("state.json reports an OPEN trade — repair only between sessions")

    # ── Baseline: record existing chain breaks so we can prove we added none ──
    before = chain_state()
    print("=== chain state BEFORE ===")
    for k, v in before.items():
        print(f"  {k:15s} {v}")

    # ── Cross-check the one row with a contemporaneous counterpart ───────────
    trade_row = next(r for f, r, _s, _n in RECOVERED if f == "trades.csv")
    mismatches = cross_check_trade(trade_row)
    if mismatches:
        print("\nCROSS-CHECK FAILED against data/trades.db:")
        for m in mismatches:
            print(f"  - {m}")
        return refuse("recovered trade row does not match trades.db; refusing to append")
    print("\ncross-check: recovered 2026-08-06 trade row matches trades.db on all 9 fields")

    # ── Plan ─────────────────────────────────────────────────────────────────
    planned = []
    for name, row, source, note in RECOVERED:
        if row["date_et"] in existing_keys(name):
            print(f"  SKIP  {name:15s} {row['date_et']} — already present (idempotent)")
            continue
        planned.append((name, row, source, note))
        print(f"  {'APPEND' if args.apply else 'WOULD APPEND'}  {name:15s} "
              f"{row['date_et']}  {row.get('action') or row.get('outcome')}")

    if not planned:
        print("\nNothing to do — ledger already repaired.")
        return 0

    if not args.apply:
        print(f"\nDry run. {len(planned)} row(s) would be appended. Re-run with --apply "
              f"(service must be stopped).")
        return 0

    # ── Apply ────────────────────────────────────────────────────────────────
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_rows = []
    for name, row, source, note in planned:
        chain = append_row(name, row)
        manifest_rows.append({
            "repaired_at_utc": stamp, "file": name, "date_et": row["date_et"],
            "chain": chain, "source": source, "note": note,
            "values": json.dumps(row, sort_keys=True),
        })
        print(f"  appended {name} {row['date_et']} -> chain {chain}")

    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nmanifest: {MANIFEST.relative_to(BASE)}")

    # ── Prove we introduced no new break ─────────────────────────────────────
    after = chain_state()
    print("\n=== chain state AFTER ===")
    ok = True
    for name in SCHEMA:
        b_rows, b_bad = before[name]
        a_rows, a_bad = after[name]
        added = a_rows - b_rows if isinstance(a_rows, int) and isinstance(b_rows, int) else "?"
        verdict = "OK" if a_bad == b_bad else "NEW BREAK"
        if a_bad != b_bad:
            ok = False
        print(f"  {name:15s} rows {b_rows} -> {a_rows} (+{added})  "
              f"first_break {b_bad} -> {a_bad}  [{verdict}]")

    if not ok:
        print("\nA NEW chain break appeared. Do not trust this file; see " + INCIDENT)
        return 1

    print(f"\nRepair complete. Pre-existing scars are unchanged and remain registered:\n"
          f"  - decisions.csv chain break at row 30 (2026-08-07) — permanent\n"
          f"  - trades.csv duplicate 2026-06-25 (+$1,390.50) — permanent\n"
          f"data/trades.db remains the arithmetic authority. See {INCIDENT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
