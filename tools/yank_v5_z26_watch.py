"""V5 watcher: record YANK's first real order on MNQZ26 (roll prereg f18b0cd, check V5).

Read-only and idempotent. Scans YANK's log incrementally (byte offset in a state file)
and the live ledger, and the first time it sees a REAL order after the roll it appends
the V5 result to _bmad-output/roll_z26_verdict.md and stops looking. Places no orders,
calls no broker, touches no trader state. Safe to run on a timer forever.

Run:  .venv/bin/python tools/yank_v5_z26_watch.py [--status]
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "logs/yank_streaming_working.log"
DB = REPO / "data/trades.db"
STATE = REPO / "data/yank/v5_watch_state.json"
VERDICT = REPO / "_bmad-output/roll_z26_verdict.md"

# YANK's installed unit switched to MNQZ26 at this instant (roll_z26_verdict.md).
ROLL_UTC = "2026-09-15 20:10:14"
# Real-order lines only. Shadow (👻) and SIM-only chatter are not V5 evidence.
PATTERNS = re.compile(
    r"TIER 2 LIMIT PLACED|Limit entry FILLED|TP/SL placed on fill|"
    r"ProjectX order rejected|TS SIM entry \|"
)
TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"offset": 0, "found": False}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def scan_log(offset: int) -> tuple[list[str], int]:
    """Return (matching lines after the roll, new offset). Never raises on a missing file."""
    hits = []
    if not LOG.exists():
        return hits, offset
    size = LOG.stat().st_size
    if offset > size:          # log rotated or truncated — rescan from the start
        offset = 0
    with LOG.open("r", errors="replace") as fh:
        fh.seek(offset)
        for line in fh:
            if "👻" in line or not PATTERNS.search(line):
                continue
            m = TS.match(line)
            if m and m.group(1) >= ROLL_UTC:
                hits.append(line.rstrip()[:300])
        return hits, fh.tell()


def scan_ledger() -> list[tuple]:
    if not DB.exists():
        return []
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        rows = con.execute(
            "select timestamp, symbol, direction, entry_price, exit_price, pnl "
            "from trades where trader_id='trader-yank' and write_mode='realtime' "
            "and symbol like '%Z26%' order by timestamp limit 5"
        ).fetchall()
        con.close()
        return rows
    except sqlite3.Error:
        return []


def record(evidence: list[str], rows: list[tuple]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = [f"\n## V5 — YANK's first order on MNQZ26 (recorded automatically {now})\n",
            "Detected by `tools/yank_v5_z26_watch.py`, which reads only YANK's own log and",
            "the live ledger. Verify the contract and the TS SIM mirror before calling V5 passed.\n"]
    if evidence:
        body.append("**Log evidence:**\n")
        body += [f"- `{line}`" for line in evidence[:6]]
    if rows:
        body.append("\n**Ledger rows:**\n")
        body += [f"- {r[0]} | {r[1]} | {r[2]} | entry {r[3]} | exit {r[4]} | pnl {r[5]}" for r in rows]
    body.append("")
    with VERDICT.open("a") as fh:
        fh.write("\n".join(body) + "\n")


def main() -> int:
    st = load_state()
    if "--status" in sys.argv:
        print(json.dumps(st, indent=2))
        return 0
    if st.get("found"):
        return 0
    hits, offset = scan_log(int(st.get("offset", 0)))
    rows = scan_ledger()
    st["offset"] = offset
    st["last_check"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if hits or rows:
        st.update({"found": True, "found_at": st["last_check"],
                   "evidence": hits[:6], "ledger_rows": [list(r) for r in rows]})
        if VERDICT.exists():
            record(hits, rows)
        print(f"V5 EVIDENCE FOUND — {len(hits)} log line(s), {len(rows)} ledger row(s); "
              f"recorded in {VERDICT.name}")
        for line in hits[:3]:
            print(" ", line)
    save_state(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
