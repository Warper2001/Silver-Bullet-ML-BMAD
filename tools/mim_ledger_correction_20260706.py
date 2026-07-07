#!/usr/bin/env python3
"""One-shot ledger correction — 2026-07-06 MIM-NB trade (halt-review 2026-07-07).

The 07-06 close was misbooked as CAT_STOP -$500. Broker truth (ProjectX
Trade/Order search, acct 23884932): the bot's stop #3228110588 @29762.25 was
CANCELED unfilled; external order #3229490103 flattened at 29940.5, realized
P&L -$165.00 (entry fill 30023.0). This script corrects, atomically and with
backups:

  1. data/mim_nb/trades.csv  — rewrite the 07-06 row with broker-true values
     (entry_px uses the REAL fill 30023.00, deviating from the mark-model
     convention of other rows; documented in the halt-review doc), then
     re-chain that row and every subsequent row (rolling SHA-256, GENESIS
     seed, ChainedCsv semantics from mim_nb_live.py).
  2. data/trades.db          — UPDATE the matching trader-mim-nb row.
  3. data/mim_nb/state.json  — refresh chains.trades to the new head.

Run from the repo root that owns the LIVE data (the main checkout):
    .venv/bin/python tools/mim_ledger_correction_20260706.py [--apply]

Without --apply it is a dry run. Restart trader-mim-nb afterwards (while
flat) so the in-memory chain head reloads from the corrected CSV.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADES_CSV = ROOT / "data/mim_nb/trades.csv"
STATE_JSON = ROOT / "data/mim_nb/state.json"
TRADES_DB = ROOT / "data/trades.db"
BACKUP = ROOT / "data/mim_nb/trades.csv.bak-20260707"

FIELDS = ["day", "dir", "entry_t", "entry_px", "exit_t", "exit_px",
          "reason", "pnl_pts", "pnl_usd", "day_pnl_usd"]

OLD_ROW_DAY = "2026-07-06"
OLD_CHAIN = "b7164bd9e2b33b6d"
CORRECTED = {
    "day": "2026-07-06", "dir": "1", "entry_t": "10:30", "entry_px": "30023.00",
    "exit_t": "13:30", "exit_px": "29940.50", "reason": "EXTERNAL_FLATTEN",
    "pnl_pts": "-82.50", "pnl_usd": "-165.00", "day_pnl_usd": "-165.00",
}
DB_NOTE = ("20260707 halt-review correction: external flatten order 3229490103 "
           "(bot stop 3228110588 canceled unfilled); was misbooked CAT_STOP -500")


def rechain(rows):
    head = "GENESIS"
    out = []
    for row in rows:
        payload = "|".join(str(row.get(k, "")) for k in FIELDS)
        head = hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]
        row = dict(row)
        row["chain"] = head
        out.append(row)
    return out, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    with open(TRADES_CSV) as f:
        rows = list(csv.DictReader(f))

    # pre-flight: current chain must be internally valid
    _, cur_head = rechain([{k: r[k] for k in FIELDS} for r in rows])
    if cur_head != rows[-1]["chain"]:
        sys.exit("ABORT: existing chain does not recompute — file already modified?")

    targets = [i for i, r in enumerate(rows)
               if r["day"] == OLD_ROW_DAY and r["chain"] == OLD_CHAIN]
    if len(targets) != 1:
        sys.exit(f"ABORT: expected exactly one 07-06 row with chain {OLD_CHAIN}, found {len(targets)}")
    idx = targets[0]

    new_rows = [dict(r) for r in rows]
    new_rows[idx].update(CORRECTED)
    chained, new_head = rechain(new_rows)

    print(f"row {idx}: {rows[idx]['reason']} {rows[idx]['pnl_usd']} -> "
          f"{CORRECTED['reason']} {CORRECTED['pnl_usd']}")
    for i in range(idx, len(chained)):
        print(f"  re-chained row {i} ({chained[i]['day']}): "
              f"{rows[i]['chain']} -> {chained[i]['chain']}")
    print(f"new head: {new_head}")

    # DB row
    con = sqlite3.connect(TRADES_DB)
    cur = con.execute(
        "SELECT id, pnl, exit_reason FROM trades WHERE trader_id='trader-mim-nb' "
        "AND timestamp LIKE '2026-07-06%' AND exit_reason='CAT_STOP'")
    db_rows = cur.fetchall()
    if len(db_rows) != 1:
        sys.exit(f"ABORT: expected exactly one matching trades.db row, found {db_rows}")
    db_id = db_rows[0][0]
    print(f"trades.db id {db_id}: pnl {db_rows[0][1]} -> -165.0, exit_reason -> EXTERNAL_FLATTEN")

    if not args.apply:
        con.close()
        print("\nDRY RUN — rerun with --apply to write.")
        return

    shutil.copy2(TRADES_CSV, BACKUP)
    with open(TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["chain"])
        w.writeheader()
        w.writerows(chained)

    con.execute(
        "UPDATE trades SET pnl=-165.0, entry_price=30023.0, exit_price=29940.5, "
        "exit_reason='EXTERNAL_FLATTEN', metadata=? WHERE id=?",
        (json.dumps({"pnl_pts": -82.5, "correction": DB_NOTE}), db_id))
    con.commit()
    con.close()

    state = json.loads(STATE_JSON.read_text())
    state.setdefault("chains", {})["trades"] = new_head
    STATE_JSON.write_text(json.dumps(state))

    # post-verify
    with open(TRADES_CSV) as f:
        vrows = list(csv.DictReader(f))
    _, vhead = rechain([{k: r[k] for k in FIELDS} for r in vrows])
    assert vhead == vrows[-1]["chain"] == new_head, "post-write chain verify FAILED"
    print(f"\nAPPLIED. backup={BACKUP.name}, new trades head={new_head}, chain verified.")


if __name__ == "__main__":
    main()
