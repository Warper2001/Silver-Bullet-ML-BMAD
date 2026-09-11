#!/usr/bin/env python3
"""
Restore the Thursday-short ledger rows lost to the 2026-09-08 stale-tree clobber.

WHY THIS EXISTS (second reconstruction of the same data)
--------------------------------------------------------
The 2026-08-27 trade was already reconstructed once, on 2026-09-02, by
tools/reconstruct_thursday_aug27.py (see _bmad-output/reconstruction_note_thursday_20260827.md);
that work merged to main as 00203e1 -> e6c8682.

It was then destroyed again. Commit 27e4e1e ("record thursday-short live trades
through 2026-09-03", 2026-09-08) wrote the 09-03 rows onto a parent tree that
PREDATED the 08-27 reconstruction, and never merged into main
(`git merge-base --is-ancestor 27e4e1e origin/main` is false). The live files on
disk today are byte-identical to 27e4e1e's stale result plus the bot's own later
09-10 appends. Net effect:

  decisions.csv  - lost the 08-27 ENTERED row AND the 09-03 ENTERED row
  trades.csv     - lost both 08-27 legs (the 09-03 legs were restored by 27e4e1e)
  counterfactuals.csv - lost nothing recoverable (see below)

This is the same "git-tracked while live-appended" failure class named in
ChainedCsv's own docstring, recurring for at least the fourth time (gap-fade
2026-08-06, thursday-short 2026-08-27, and now this) -- this time via a parallel
orphaned commit rather than a branch checkout.

WHY THE CHAIN STILL PASSED WHILE ROWS WERE MISSING
--------------------------------------------------
tools/verify_chain.py proves no row was EDITED, not that the file is COMPLETE.
ChainedCsv._read_tail() re-reads the head from disk before every append, so when
the bot wrote 09-10 it correctly chained onto the stale on-disk tail. The result
is an internally consistent chain over an incomplete file. Completeness is a
separate question, and nothing was asking it.

APPEND ORDER
------------
These rows are appended at the END of each file, after the 09-10 rows, so they
are not in chronological order. That is deliberate: an append-only hash chain
records the order rows were WRITTEN, and rewriting the file to interleave them
would invalidate every subsequent chain hash -- destroying the tamper-evidence
to cosmetically fix sort order. The `thursday` column is the authoritative date.

SOURCE OF EVERY VALUE (logs/thursday_short.log)
-----------------------------------------------
2026-08-27 (unchanged from the 2026-09-02 reconstruction, lines 50975-52767):
  Entry confirm  50988: "SHORT confirmed | MBTU26 @ 79315.0 | METU26 @ 2516.0"
  Entry sizing   50983: "1 MBTU26 (~$7,937) + 32 METU26 (~$8,059) short"
  Exit (MBT)     52756: "MBT MBTU26 EXIT: 79315.00->79130.00 +23.3bps $+18.50"
  Exit (MET)     52760: "MET METU26 EXIT: 2516.00->2501.50 +57.6bps $+46.40"
  Exit reason    52752: "THURSDAY SHORT -- EXITING (shutdown)"
2026-09-03 (decisions row only; its trades/counterfactuals rows are already present):
  Entry banner   58614: "THURSDAY SHORT - ENTERING SHORT POSITIONS"
  Entry sizing   58620: "1 MBTU26 (~$7,758) + 32 METU26 (~$7,678) short"
  Entry confirm  58626: "SHORT confirmed | MBTU26 @ 77575.0 | METU26 @ 2399.0"
  LR slopes      recovered from the surviving 09-03 rows in trades.csv
                 (lr_slope20_bpd=117.717, lr_slope40_bpd=62.325) -- the live bot
                 writes the same pair to both files from one fetch, so this is a
                 copy of the real value, not a recomputation.

WHAT IS NOT RECONSTRUCTED, AND WHY
-----------------------------------
  08-27 counterfactuals.csv -- reason="shutdown" defers the counterfactual write
    until a later poll resolves it against the eventual 23:05 mark (_exit()'s
    cf_pending path). That in-memory pending state died in the same restart, and
    no 23:05 mark for MBTU26/METU26 that day appears anywhere in the log. The real
    bot could not have resolved it either. Expected final row counts are therefore
    7 decisions / 14 trades / 12 counterfactuals -- the counterfactual file is
    legitimately two rows shorter.
  08-27 lr_slope20_bpd / lr_slope40_bpd -- left blank. fetch_btc_lr_slopes() logs
    only on FAILURE, so a successful fetch's values were never in the log. Blank is
    what the live code itself writes on failure, and is the honest answer here.

Idempotent. trades.csv dedupes on (thursday, symbol); decisions.csv has no dedupe
key by design, so this script checks for an existing ENTERED row per Thursday.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, str(REPO))

from thursday_short import ChainedCsv  # noqa: E402  (reuse the real chain logic)

TDIR = REPO / "data" / "thursday_ts"

DECISION_FIELDS = ["ts_utc", "thursday", "mbt_sym", "met_sym", "mark_btc", "mark_eth",
                   "n_mbt", "n_met", "lr_slope20_bpd", "lr_slope40_bpd", "action", "detail"]
TRADE_FIELDS = ["thursday", "symbol", "dir", "entry_t", "entry_px", "exit_t", "exit_px",
                "qty", "ret_bps", "pnl_usd", "reason", "lr_slope20_bpd", "lr_slope40_bpd"]

# (thursday, ts_utc, mbt_px, met_px, n_mbt, n_met, lr20, lr40)
DECISIONS = [
    ("2026-08-27", "2026-08-27T00:03:41.229000+00:00", 79315.0, 2516.0, 1, 32, "", ""),
    ("2026-09-03", "2026-09-03T00:02:01.528000+00:00", 77575.0, 2399.0, 1, 32, "117.717", "62.325"),
]

# 2026-08-27 legs only -- 09-03's legs survived in trades.csv.
TRADES = [
    # symbol, entry_px, exit_px, qty, label
    ("MBTU26", 79315.0, 79130.0, 1, "MBT"),
    ("METU26", 2516.0, 2501.5, 32, "MET"),
]
TRADE_THURSDAY = "2026-08-27"


def decision_exists(thursday: str) -> bool:
    path = TDIR / "decisions.csv"
    if not path.exists():
        return False
    with open(path) as f:
        return any(r.get("thursday") == thursday and r.get("action") == "ENTERED"
                   for r in csv.DictReader(f))


def main() -> int:
    decisions_log = ChainedCsv(TDIR / "decisions.csv", DECISION_FIELDS, key_fields=None)
    trades_log = ChainedCsv(TDIR / "trades.csv", TRADE_FIELDS,
                            key_fields=("thursday", "symbol"))

    wrote: list[tuple[str, bool]] = []

    for thursday, ts_utc, mbt_px, met_px, n_mbt, n_met, lr20, lr40 in DECISIONS:
        if decision_exists(thursday):
            print(f"decisions.csv already has an ENTERED row for {thursday} "
                  f"-- skipping (idempotent).")
            continue
        ok = decisions_log.append({
            "ts_utc": ts_utc, "thursday": thursday,
            "mbt_sym": "MBTU26", "met_sym": "METU26",
            "mark_btc": mbt_px, "mark_eth": met_px,
            "n_mbt": n_mbt, "n_met": n_met,
            "lr_slope20_bpd": lr20, "lr_slope40_bpd": lr40,
            "action": "ENTERED", "detail": "",
        })
        wrote.append((f"decisions.csv ENTERED {thursday}", ok))

    for symbol, entry_px, exit_px, qty, label in TRADES:
        ret_bps = round((entry_px - exit_px) / entry_px * 10_000, 2)
        pnl = round((entry_px - exit_px) * qty * 0.1, 2)
        ok = trades_log.append({
            "thursday": TRADE_THURSDAY, "symbol": symbol, "dir": "short",
            "entry_t": "00:03", "entry_px": entry_px,
            "exit_t": "06:49", "exit_px": exit_px, "qty": qty,
            "ret_bps": ret_bps, "pnl_usd": pnl, "reason": "shutdown",
            "lr_slope20_bpd": "", "lr_slope40_bpd": "",
        })
        wrote.append((f"trades.csv {label} {symbol} {TRADE_THURSDAY}", ok))
        print(f"{label} {symbol}: {entry_px:.2f}->{exit_px:.2f}  "
              f"{ret_bps:+.2f}bps  ${pnl:+.2f}  reason=shutdown  written={ok}")

    print("\n=== summary ===")
    for what, ok in wrote:
        print(f"  {'wrote' if ok else 'SKIPPED (duplicate)'}: {what}")
    if not wrote:
        print("  nothing to do -- ledger already complete.")

    total = round((79315.0 - 79130.0) * 1 * 0.1 + (2516.0 - 2501.5) * 32 * 0.1, 2)
    print(f"\n2026-08-27 net P&L restored: ${total:+.2f} (early shutdown-exit at "
          f"06:49 UTC, NOT the intended 23:05 hold -- understated versus a full hold).")
    print("True traded history is now 7 Thursdays. Note that per the 2026-09-11 "
          "restart pre-registration these Thursdays are VOID for the accrual; this "
          "reconstruction is for audit completeness only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
