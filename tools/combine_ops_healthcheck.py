#!/usr/bin/env python3
"""Combine operations heartbeat — read-only liveness + safety check for the live
YANK + MIM-NB Topstep 50K combine (acct 23884932).

Built per the 2026-06-17 party-mode ops review (Winston): with ~8 services live on a
real combine and research parked ~5 weeks on S25 data, the dominant risk is a SILENT
operational failure (a bot 401-looping, a dead floor monitor, a HALT flag nobody saw)
quietly costing the combine while attention is elsewhere. This is the cheap heartbeat
that catches those. It NEVER trades, halts, or mutates state — it only reads.

Checks, by dollar consequence on the combine:
  CRITICAL (failure can blow the combine):
    - trader-yank, trader-mim-nb, combine-floor-monitor are systemd-active
    - HALT flag present (data/combine_joint/HALT) -> monitor already halted; human needed
    - floor monitor is actually polling (monitor.csv fresh) — a dead monitor = no circuit breaker
    - distance-to-floor headroom (early warning well before the $500 derived trigger)
  INFO (paper / non-combine bots):
    - btc-carry, s26-combine, s26, s27, sil-quote-capture active + logs fresh

Exit codes (for cron / alerting):  0 = all good · 1 = warning · 2 = CRITICAL

Usage:
    .venv/bin/python tools/combine_ops_healthcheck.py [--max-stale SEC] [--quiet]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

BASE = Path(__file__).resolve().parent.parent
JOINT = BASE / "data" / "combine_joint"
HALT_FILE = JOINT / "HALT"
MONITOR_CSV = JOINT / "monitor.csv"
FLOOR_STATE = JOINT / "floor_state.json"

HALT_DISTANCE = 500.0      # mirrors combine_floor_monitor.HALT_DISTANCE (derived trigger)
WARN_DISTANCE = 1500.0     # heartbeat early-warning: well above the hard trigger

# name -> (critical?, log filename, max staleness seconds, window_or_None)
# window = ((start_h, start_m), (end_h, end_m)) in ET, weekdays only. When the current
# time is OUTSIDE a service's window, the service is EXPECTED to idle-sleep, so we check
# only that it is active and skip the freshness alarm (avoids crying wolf — a monitor
# that false-alarms gets ignored, which defeats its whole purpose).
SERVICES = {
    "trader-yank":           (True,  "yank_streaming_working.log", 240, None),
    "trader-mim-nb":         (True,  "mim_nb_live.log",            240, None),
    "combine-floor-monitor": (True,  "combine_floor_monitor.log",  120, None),
    "trader-btc-carry":      (False, "btc_carry_executor.log",     900, None),
    "trader-s26-combine":    (False, "btc_s26_combine.log",        900, None),
    "trader-s26":            (False, "s26_soft_fvg_streaming.log",  900, None),
    "trader-s27":            (False, "s27_squeeze_streaming.log",   900, None),
    # SIL capture only runs 09:25-16:00 ET Mon-Fri (capture_sil_quotes.py); idle otherwise.
    "sil-quote-capture":     (False, "sil_quote_capture.log",       900, ((9, 25), (16, 0))),
}


def in_window(window) -> bool:
    """True if now (ET) is inside the service's weekday capture window."""
    if window is None:
        return True
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    (sh, sm), (eh, em) = window
    return dtime(sh, sm) <= now.time() <= dtime(eh, em)

CRIT, WARN, OK = 2, 1, 0


def is_active(svc: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", f"{svc}.service"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def log_age(fname: str):
    p = BASE / "logs" / fname
    if not p.exists():
        return None
    return time.time() - p.stat().st_mtime


def last_monitor_row():
    if not MONITOR_CSV.exists():
        return None
    try:
        with MONITOR_CSV.open() as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-stale", type=int, default=None,
                    help="override the per-service log staleness threshold (seconds)")
    ap.add_argument("--quiet", action="store_true", help="only print on warning/critical")
    args = ap.parse_args()

    worst = OK
    lines: list[str] = []

    def emit(level: int, msg: str):
        nonlocal worst
        worst = max(worst, level)
        tag = {OK: "  OK ", WARN: " WARN", CRIT: "CRIT!"}[level]
        lines.append(f"[{tag}] {msg}")

    # 1) HALT flag — check first; if the monitor already halted, that dominates.
    if HALT_FILE.exists():
        try:
            info = json.loads(HALT_FILE.read_text())
            emit(CRIT, f"HALT FLAG PRESENT — {info.get('reason','?')} @ {info.get('ts','?')} "
                       f"(both bots stopped; human review required)")
        except Exception:
            emit(CRIT, f"HALT FLAG PRESENT at {HALT_FILE} (unparseable; human review required)")
    else:
        emit(OK, "no HALT flag")

    # 2) Services: active + log freshness (freshness only inside the service's window)
    for svc, (critical, log, max_stale, window) in SERVICES.items():
        lvl_if_bad = CRIT if critical else WARN
        if not is_active(svc):
            emit(lvl_if_bad, f"{svc}: NOT active")
            continue
        if not in_window(window):
            emit(OK, f"{svc}: active, idle (outside capture window — freshness not checked)")
            continue
        age = log_age(log)
        thresh = args.max_stale if args.max_stale is not None else max_stale
        if age is None:
            emit(WARN, f"{svc}: active but log {log} missing")
        elif age > thresh:
            emit(lvl_if_bad if critical else WARN,
                 f"{svc}: active but log stale {age:.0f}s > {thresh}s (possible silent stall / 401-loop)")
        else:
            emit(OK, f"{svc}: active, log fresh ({age:.0f}s)")

    # 3) Floor monitor data freshness + distance-to-floor headroom
    row = last_monitor_row()
    if row is None:
        emit(CRIT, "monitor.csv missing/empty — floor circuit breaker has no data")
    else:
        try:
            equity = float(row["equity"]); floor = float(row["floor"])
            dist = equity - floor
            n = row.get("n_trades", "0")
            pf = row.get("combined_pf", "") or "n/a"
            if dist <= HALT_DISTANCE:
                emit(CRIT, f"distance-to-floor ${dist:,.0f} <= ${HALT_DISTANCE:.0f} TRIGGER "
                           f"(equity ${equity:,.0f}, floor ${floor:,.0f})")
            elif dist <= WARN_DISTANCE:
                emit(WARN, f"distance-to-floor ${dist:,.0f} approaching ${HALT_DISTANCE:.0f} trigger "
                           f"(equity ${equity:,.0f})")
            else:
                emit(OK, f"distance-to-floor ${dist:,.0f} (equity ${equity:,.0f}, floor ${floor:,.0f}, "
                         f"combined_pf {pf}, n_trades {n})")
        except Exception as e:
            emit(WARN, f"monitor.csv last row unparseable: {e}")

    header = {OK: "ALL OK", WARN: "WARNINGS", CRIT: "CRITICAL"}[worst]
    if not (args.quiet and worst == OK):
        print(f"=== Combine ops healthcheck: {header} ===")
        for ln in lines:
            print(ln)
    return worst


if __name__ == "__main__":
    sys.exit(main())
