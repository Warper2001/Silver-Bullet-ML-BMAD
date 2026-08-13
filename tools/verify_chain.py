#!/usr/bin/env python3
"""Verify the SHA-256 hash chains on the bots' append-only CSV audit trails.

Written 2026-08-08, after V9: `data/gap_fade/decisions.csv` had been carrying a
broken chain since 2026-08-06 and nothing checked it. A hash chain that nobody
verifies is a long file with an extra column.

What a passing chain does and does not mean — read this before trusting a PASS:

  PROVES     no row present in the file was edited or reordered after it was written.
  PROVES NOT that the file is complete.

`trades.csv` verified cleanly on 2026-08-08 while **missing the 2026-08-06 trade** and
**double-counting 2026-06-25**. It had been reverted wholesale to an earlier commit, and
an earlier state of an append-only chain is itself a valid chain prefix. Completeness is
a separate question, answered by reconciling against `data/trades.db` (see --reconcile).

`decisions.csv` DID fail — because the live process kept appending after the revert,
chaining onto an in-memory head the file no longer contained. That break is the
fingerprint of the loss. `trades.csv` hid the same event only because no trade happened
to be appended afterwards.

The missing rows were recovered on 2026-08-12 by tools/gap_fade_ledger_repair.py, and
ChainedCsv now re-reads its head from disk before every append so an outside writer can
no longer corrupt the chain. Two injuries are permanent, because undoing either would
mean rewriting an append-only file: the decisions.csv break at row 30, and the duplicate
2026-06-25 trade. Both are registered in KNOWN_SCARS below and reported as [SCAR] rather
than as findings — a monitor that cries every run stops being read. Anything NOT in that
registry still fails. Use --strict to see the raw, unregistered truth.

Read-only. Never repairs, never rewrites, never re-chains — a chain you rewrite when it
is inconvenient was never evidence.

Usage:
    .venv/bin/python tools/verify_chain.py
    .venv/bin/python tools/verify_chain.py --reconcile
    .venv/bin/python tools/verify_chain.py --reconcile --strict
    .venv/bin/python tools/verify_chain.py --file data/gap_fade/trades.csv

Exit codes:  0 = all chains verify · 1 = a chain is broken · 2 = a file is unreadable
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GENESIS = "GENESIS"

OK, BROKEN, UNREADABLE = 0, 1, 2

# Files whose chain is defined by "every column except `chain`, in header order".
DEFAULT_FILES = [
    "data/gap_fade/trades.csv",
    "data/gap_fade/decisions.csv",
    "data/gap_fade/fills.csv",
]

# relpath -> (trader_id, date column) for the completeness cross-check
RECONCILE = {"data/gap_fade/trades.csv": ("trader-gap-fade", "date_et")}

# Permanent, documented injuries. Each one is damage that already happened and cannot
# be undone without rewriting an append-only file. Registering a scar suppresses its
# exit code, never its output — it still prints, tagged [SCAR], with the incident that
# explains it. Adding an entry here is a claim that the damage is understood and
# accepted; do not add one to quiet a finding you have not investigated.
INCIDENT_20260806 = "_bmad-output/ledger_incident_20260806_gap_fade.md"
KNOWN_SCARS = {
    "data/gap_fade/decisions.csv": {
        "break_at": 30,
        "why": ("git checkout reverted this file on 2026-08-06 while the bot held a "
                "stale in-memory head; row 30 (2026-08-07) chains onto a head the "
                "file no longer contains. Unfixable — re-chaining would destroy the "
                "evidence the chain exists to be."),
        "incident": INCIDENT_20260806,
    },
    "data/gap_fade/trades.csv": {
        "duplicates": ["2026-06-25"],
        "why": ("first-day double append across a restart, before the double-entry "
                "guard existed (7c9bc0a). Overstates this file by $1,390.50; a row "
                "cannot be withdrawn from an append-only chain, so data/trades.db is "
                "the arithmetic authority."),
        "incident": INCIDENT_20260806,
    },
}


def verify(path: Path):
    """Return (rows, first_bad_rownum, first_bad_key, error).

    `first_bad_rownum` is None when the chain verifies. The walk continues past a
    mismatch (adopting the stored head) so a single break does not cascade into a
    misleading "everything after this is corrupt" report.
    """
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return 0, None, None, "empty file (no header)"
            fields = [f for f in reader.fieldnames if f != "chain"]
            head, n, bad_n, bad_key = GENESIS, 0, None, None
            for row in reader:
                n += 1
                payload = "|".join(str(row.get(k, "")) for k in fields)
                head = hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]
                stored = row.get("chain")
                if stored != head:
                    if bad_n is None:
                        bad_n, bad_key = n, row.get(fields[0], "?")
                    head = stored          # resync; report the first break only
            return n, bad_n, bad_key, None
    except OSError as exc:
        return 0, None, None, str(exc)


def missing_vs_db(path: Path, trader_id: str, date_col: str):
    """Dates present in trades.db but absent from the chained file (completeness)."""
    db = BASE / "data" / "trades.db"
    if not db.exists() or not path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT DISTINCT date(timestamp) FROM trades WHERE trader_id = ?",
            (trader_id,),
        ).fetchall()
        db_dates = {r[0] for r in rows}
        with path.open(newline="") as fh:
            file_dates = [r.get(date_col) for r in csv.DictReader(fh)]
        dupes = sorted({d for d in file_dates if file_dates.count(d) > 1})
        return sorted(db_dates - set(file_dates)), dupes
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", action="append", dest="files",
                    help="chained CSV to verify (repeatable; default: the gap-fade set)")
    ap.add_argument("--reconcile", action="store_true",
                    help="also cross-check completeness against data/trades.db")
    ap.add_argument("--strict", action="store_true",
                    help="ignore KNOWN_SCARS — report every break and duplicate as a finding")
    args = ap.parse_args()

    worst = OK
    scars_seen = []

    def scar(rel: str, kind: str, detail: str) -> bool:
        """Print a registered scar and return True if its exit code is suppressed."""
        reg = {} if args.strict else KNOWN_SCARS.get(rel, {})
        known = (reg.get("break_at") == detail if kind == "break"
                 else detail in reg.get("duplicates", []))
        if not known:
            return False
        scars_seen.append(f"{rel} ({kind} {detail})")
        print(f"         [SCAR] known and accepted: {reg['why']}")
        print(f"                see {reg['incident']}")
        return True
    for rel in (args.files or DEFAULT_FILES):
        path = BASE / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            print(f"[ MISS ] {rel}: not found")
            worst = max(worst, UNREADABLE)
            continue
        n, bad_n, bad_key, err = verify(path)
        if err:
            print(f"[ ERR  ] {rel}: {err}")
            worst = max(worst, UNREADABLE)
            continue
        if bad_n is None:
            print(f"[  OK  ] {rel}: {n} rows, chain verifies")
        else:
            print(f"[BROKEN] {rel}: {n} rows, first mismatch at row {bad_n} ({bad_key})")
            if not scar(rel, "break", bad_n):
                worst = max(worst, BROKEN)

        if args.reconcile and rel in RECONCILE:
            trader, col = RECONCILE[rel]
            res = missing_vs_db(path, trader, col)
            if res is None:
                print("         reconcile: unavailable")
                continue
            missing, dupes = res
            if missing:
                print(f"         INCOMPLETE: in trades.db but not in this file: {missing}")
                worst = max(worst, BROKEN)
            for d in dupes:
                print(f"         DUPLICATED rows for: {d}")
                if not scar(rel, "dupe", d):
                    worst = max(worst, BROKEN)
            if not missing and not dupes:
                print("         reconcile: complete and duplicate-free vs trades.db")

    if scars_seen:
        print(f"\n{len(scars_seen)} registered scar(s) suppressed from the exit code: "
              f"{', '.join(scars_seen)}")
        print("Re-run with --strict to have them counted as findings again.")
    if worst == OK:
        print("\nAll chains verify. Note: a verifying chain does NOT prove completeness "
              "— run --reconcile for that.")
    return worst


if __name__ == "__main__":
    sys.exit(main())
