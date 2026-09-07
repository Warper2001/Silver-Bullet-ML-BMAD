#!/usr/bin/env python3
"""
One-shot reconstruction of the 2026-08-27 Thursday-short trade into
data/thursday_ts/{decisions,trades}.csv.

WHY THIS EXISTS
---------------
thursday_short.py is a systemd service running a live, unattended trading
loop, restarted 06:49:54 UTC on 2026-08-27 mid-position (its shutdown handler
force-closed the open MBTU26/METU26 short, reason="shutdown", NOT the
intended scheduled_exit at 23:05). The service's own decisions.csv/trades.csv
show only the four Thursdays through 2026-07-23, despite a confirmed real
entry and exit that day (logs/thursday_short.log, lines 50975-52767) --
these files are git-tracked while being live-appended, the same failure
class already documented in ChainedCsv's own docstring and responsible for
the 2026-08-06 gap-fade incident.

This script reconstructs ONLY what the log actually proves happened. It
imports the live ChainedCsv class directly rather than reimplementing it, so
the hash chain this produces is byte-for-byte what the real bot would have
written, and the running service's next real append will chain onto it
cleanly.

SOURCE OF EVERY VALUE (logs/thursday_short.log line numbers)
  Entry confirm  50988: "SHORT confirmed | MBTU26 @ 79315.0 | METU26 @ 2516.0"
  Entry sizing   50983: "1 MBTU26 (~$7,937) + 32 METU26 (~$8,059) short"
  Exit (MBT)     52756: "MBT MBTU26 EXIT: 79315.00->79130.00 +23.3bps $+18.50"
  Exit (MET)     52760: "MET METU26 EXIT: 2516.00->2501.50 +57.6bps $+46.40"
  Exit reason    52752: "THURSDAY SHORT -- EXITING (shutdown)"

WHAT IS NOT RECONSTRUCTED, AND WHY
  lr_slope20_bpd / lr_slope40_bpd -- fetch_btc_lr_slopes() only logs on
    FAILURE (a WARNING line, absent here), never on success, so its actual
    values were never written to the log and cannot be recovered. Left
    blank -- exactly what the live code writes when the fetch itself fails,
    which is indistinguishable from "succeeded but unlogged" after the fact,
    and blank is the honest answer either way.
  counterfactuals.csv -- reason="shutdown" defers the counterfactual write
    until a later poll resolves it against the eventual 23:05 mark (see
    _exit()'s cf_pending path); that in-memory pending state was itself lost
    in the same restart, so the real bot could not have resolved it either.
    Not reconstructed: there is no log line giving a 23:05 mark for
    MBTU26/METU26 that day to reconstruct it from.

Idempotent: the (thursday, symbol) key on trades.csv means a second run is a
harmless no-op (ChainedCsv.append refuses the duplicate). decisions.csv has
no dedupe key by design (repeat decisions are legitimate) -- this script
checks for an existing 2026-08-27 ENTERED row itself before appending.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/root/Silver-Bullet-ML-BMAD")
sys.path.insert(0, str(REPO))

from thursday_short import ChainedCsv  # noqa: E402  (reuse the real chain logic)

TDIR = REPO / "data" / "thursday_ts"
THURSDAY = "2026-08-27"


def already_recorded_decision() -> bool:
    path = TDIR / "decisions.csv"
    if not path.exists():
        return False
    import csv
    with open(path) as f:
        return any(r.get("thursday") == THURSDAY for r in csv.DictReader(f))


def main() -> int:
    trades_log = ChainedCsv(
        TDIR / "trades.csv",
        ["thursday", "symbol", "dir", "entry_t", "entry_px", "exit_t", "exit_px",
         "qty", "ret_bps", "pnl_usd", "reason", "lr_slope20_bpd", "lr_slope40_bpd"],
        key_fields=("thursday", "symbol"),
    )
    decisions_log = ChainedCsv(
        TDIR / "decisions.csv",
        ["ts_utc", "thursday", "mbt_sym", "met_sym", "mark_btc", "mark_eth",
         "n_mbt", "n_met", "lr_slope20_bpd", "lr_slope40_bpd", "action", "detail"],
        key_fields=None,
    )

    wrote = []

    if already_recorded_decision():
        print(f"decisions.csv already has a {THURSDAY} row -- skipping (idempotent).")
    else:
        ok = decisions_log.append({
            "ts_utc": "2026-08-27T00:03:41.229000+00:00",
            "thursday": THURSDAY,
            "mbt_sym": "MBTU26", "met_sym": "METU26",
            "mark_btc": 79315.0, "mark_eth": 2516.0,
            "n_mbt": 1, "n_met": 32,
            "lr_slope20_bpd": "", "lr_slope40_bpd": "",
            "action": "ENTERED", "detail": "",
        })
        wrote.append(("decisions.csv ENTERED", ok))

    for symbol, entry_px, exit_px, qty, label in [
        ("MBTU26", 79315.0, 79130.0, 1, "MBT"),
        ("METU26", 2516.0, 2501.5, 32, "MET"),
    ]:
        ret_bps = round((entry_px - exit_px) / entry_px * 10_000, 2)
        pnl = round((entry_px - exit_px) * qty * 0.1, 2)
        ok = trades_log.append({
            "thursday": THURSDAY, "symbol": symbol, "dir": "short",
            "entry_t": "00:03", "entry_px": entry_px,
            "exit_t": "06:49", "exit_px": exit_px, "qty": qty,
            "ret_bps": ret_bps, "pnl_usd": pnl, "reason": "shutdown",
            "lr_slope20_bpd": "", "lr_slope40_bpd": "",
        })
        wrote.append((f"trades.csv {label} {symbol}", ok))
        print(f"{label} {symbol}: {entry_px:.2f}->{exit_px:.2f}  "
              f"{ret_bps:+.2f}bps  ${pnl:+.2f}  reason=shutdown  written={ok}")

    print("\n=== summary ===")
    for what, ok in wrote:
        print(f"  {'wrote' if ok else 'SKIPPED (duplicate)'}: {what}")

    total_pnl = round(
        (79315.0 - 79130.0) * 1 * 0.1 + (2516.0 - 2501.5) * 32 * 0.1, 2
    )
    print(f"\nreconstructed 2026-08-27 net P&L: ${total_pnl:+.2f} "
          f"(early shutdown-exit at 06:49 UTC, NOT the intended 23:05 hold -- "
          f"this Thursday's realized result is understated relative to what a "
          f"full 24h hold would have shown, per the strategy's own design)")
    print("N toward the N>=30 decision rule is now 5 of 30 Thursdays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
