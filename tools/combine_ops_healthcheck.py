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
from datetime import datetime, time as dtime, timedelta, timezone
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

# name -> (critical?, log filename, max staleness seconds, window_or_None, fresh_path_or_None)
# window = ((start_h, start_m), (end_h, end_m)) in ET, weekdays only. When the current
# time is OUTSIDE a service's window, the service is EXPECTED to idle-sleep, so we check
# only that it is active and skip the freshness alarm (avoids crying wolf — a monitor
# that false-alarms gets ignored, which defeats its whole purpose).
# fresh_path = BASE-relative file whose mtime stands in for liveness (None => logs/<log>).
# Use it when a service's journal log is written far less often than it actually does work
# (e.g. a heartbeat line every N rows) but it continuously appends to a data file.
SERVICES = {
    "trader-yank":           (True,  "yank_streaming_working.log", 240, None, None),
    "trader-mim-nb":         (True,  "mim_nb_live.log",            240, None, None),
    "combine-floor-monitor": (True,  "combine_floor_monitor.log",  120, None, None),
    "trader-btc-carry":      (False, "btc_carry_executor.log",     900, None, None),
    "trader-s26-combine":    (False, "btc_s26_combine.log",        900, None, None),
    "trader-s26":            (False, "s26_soft_fvg_streaming.log",  900, None, None),
    "trader-s27":            (False, "s27_squeeze_streaming.log",   900, None, None),
    # SIL capture only runs 09:25-16:00 ET Mon-Fri (capture_sil_quotes.py); idle otherwise.
    # Its log only prints a heartbeat every 1200 rows (~53 min), so check the CSV it flushes
    # every 5s poll instead — kills false flapping AND catches a real 401-loop stall in minutes.
    "sil-quote-capture":     (False, "sil_quote_capture.log",       300, ((9, 25), (16, 0)),
                              "data/quotes/sil_quote_capture.csv"),
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


def secs_into_window(window):
    """Seconds since today's window opened (ET), or None if no window / outside it.

    Note in_window(None) is True ("always on"), so the explicit None check is required
    before unpacking the window bounds below.
    """
    if window is None or not in_window(window):
        return None
    now = datetime.now(ET)
    (sh, sm), _ = window
    open_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    return (now - open_dt).total_seconds()


CRIT, WARN, OK = 2, 1, 0

# --- YANK evaluation heartbeat (spec: _bmad-output/spec_yank_evaluation_heartbeat.md) ---
# The heartbeat proves the strategy loop is EVALUATING bars, not merely that the process
# is alive — log mtime stays fresh even in a 401-loop (the 06-07 failure class).
HEARTBEAT_FILE = BASE / "logs" / "yank_heartbeat.json"
HB_SERVICE = "trader-yank"
HB_GRACE_SEC = 600        # startup grace before a missing/corrupt file is CRITICAL
HB_TS_STALE_SEC = 300     # loop-frozen threshold; one lost cycle + one slow one at 60s cadence
HB_BAR_LAG_SEC = 300      # matches the bot's own _check_stale 300s convention
HB_POLL_FAILS_CRIT = 3    # network is transient; 3 consecutive = 401-loop / stale token


def in_globex_window(now_et: datetime) -> bool:
    """True inside CME Globex hours: Sun 18:00 ET -> Fri 17:00 ET, minus the 17:00-18:00
    daily break. Deliberately no holiday calendar — a false page on a half-day costs a
    human 30 seconds; a holiday module costs maintenance forever."""
    wd, t = now_et.weekday(), now_et.time()
    if wd == 5:                       # Saturday
        return False
    if wd == 6:                       # Sunday: opens 18:00 ET
        return t >= dtime(18, 0)
    if wd == 4:                       # Friday: closes 17:00 ET
        return t < dtime(17, 0)
    return not (dtime(17, 0) <= t < dtime(18, 0))   # Mon-Thu daily break


def secs_since_globex_reopen(now_et: datetime):
    """Seconds since the most recent Globex session open (18:00 ET), or None outside the
    window. Used to grace the bar-lag alarm right after a reopen (first bar isn't instant)."""
    if not in_globex_window(now_et):
        return None
    open_dt = now_et.replace(hour=18, minute=0, second=0, microsecond=0)
    if now_et.time() < dtime(18, 0):
        open_dt -= timedelta(days=1)
    while open_dt.weekday() in (4, 5):   # no 18:00 open on Friday or Saturday
        open_dt -= timedelta(days=1)
    return (now_et - open_dt).total_seconds()


def service_uptime_secs(svc: str):
    """Seconds since the systemd unit last entered active state, or None if unknown."""
    try:
        r = subprocess.run(
            ["systemctl", "show", f"{svc}.service", "-p", "ActiveEnterTimestamp", "--value"],
            capture_output=True, text=True, timeout=10)
        raw = r.stdout.strip()
        if not raw:
            return None
        # systemd format: "Mon 2026-06-30 06:02:34 UTC"
        parts = raw.split()
        dt = datetime.strptime(" ".join(parts[1:3]), "%Y-%m-%d %H:%M:%S")
        return max(0.0, time.time() - dt.timestamp())
    except Exception:
        return None


def read_heartbeat(path: Path = HEARTBEAT_FILE):
    """Parsed heartbeat dict, or None if missing/torn/corrupt (caller decides severity)."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _parse_iso(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def evaluate_heartbeat(hb, now_utc: datetime, uptime_secs) -> list:
    """Pure evaluation of a heartbeat payload -> [(level, message), ...].

    Kept side-effect-free (no filesystem, no clock, no systemctl) so fixture tests can
    drive every branch offline before this ever judges the real-money bot.
    """
    out = []
    if hb is None:
        if uptime_secs is not None and uptime_secs < HB_GRACE_SEC:
            out.append((OK, f"yank heartbeat: absent but service started {uptime_secs:.0f}s ago (grace {HB_GRACE_SEC}s)"))
        else:
            out.append((CRIT, "yank heartbeat: missing/unparseable past startup grace — "
                              "cannot prove the strategy loop is evaluating bars"))
        return out

    # 1) Loop frozen? ts must tick 24/7 (heartbeat writes in the market-closed branch too).
    ts = _parse_iso(hb.get("ts") or "")
    if ts is None:
        out.append((CRIT, "yank heartbeat: 'ts' missing/unparseable"))
        return out
    age = (now_utc - ts).total_seconds()
    if age > HB_TS_STALE_SEC:
        out.append((CRIT, f"yank heartbeat: stale {age:.0f}s > {HB_TS_STALE_SEC}s — "
                          f"event loop frozen/deadlocked (loop_seq {hb.get('loop_seq')})"))
        return out   # everything else in the payload is equally stale; one alarm is enough

    # 2) Poll failures (401-loop / stale token / PX fetch errors). Transient -> counter.
    fails = int(hb.get("consec_poll_failures") or 0)
    if fails >= HB_POLL_FAILS_CRIT:
        out.append((CRIT, f"yank heartbeat: {fails} consecutive poll failures "
                          f"(last: {hb.get('poll_error')}) — data feed down / token stale"))

    # 3) Detect errors: sticky (total>0 alarms until restart). A logic fault in
    #    _detect_and_enter silently skips entry decisions and will not self-heal.
    detect_total = int(hb.get("detect_errors_total") or 0)
    if detect_total > 0:
        out.append((CRIT, f"yank heartbeat: {detect_total} _detect_and_enter exception(s) since start "
                          f"— entries silently skipped (last: {hb.get('last_exception')})"))

    # 4) Bar lag — only meaningful when the bot believes the market is open, we're inside
    #    Globex hours, backfill is done, and the session has been open long enough.
    now_et = now_utc.astimezone(ET)
    reopen = secs_since_globex_reopen(now_et)
    if (hb.get("market_open") and not hb.get("is_backfill")
            and reopen is not None and reopen > HB_BAR_LAG_SEC):
        last_bar = _parse_iso(hb.get("last_bar_ts") or "")
        if last_bar is None:
            out.append((CRIT, "yank heartbeat: market open but no bar processed yet this session"))
        else:
            lag = (now_utc - last_bar).total_seconds()
            if lag > HB_BAR_LAG_SEC:
                out.append((CRIT, f"yank heartbeat: last bar {lag:.0f}s old > {HB_BAR_LAG_SEC}s "
                                  f"in Globex hours — data gap / silent stall"))

    if not out:
        out.append((OK, f"yank heartbeat: fresh ({age:.0f}s), loop_seq {hb.get('loop_seq')}, "
                        f"{hb.get('bars_evaluated_total')} bars evaluated, "
                        f"src {hb.get('data_source')}, contract {hb.get('contract')}"))
    return out


def is_active(svc: str) -> bool:
    try:
        r = subprocess.run(["systemctl", "is-active", f"{svc}.service"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:
        return False


def file_age(relpath: str):
    """Seconds since the BASE-relative file was last modified, or None if absent."""
    p = BASE / relpath
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

    # 2) Services: active + freshness (freshness only inside the service's window).
    # Freshness is measured on fresh_path when set, else the journal log.
    for svc, (critical, log, max_stale, window, fresh_path) in SERVICES.items():
        lvl_if_bad = CRIT if critical else WARN
        if not is_active(svc):
            emit(lvl_if_bad, f"{svc}: NOT active")
            continue
        if not in_window(window):
            emit(OK, f"{svc}: active, idle (outside capture window — freshness not checked)")
            continue
        fresh_rel = fresh_path or f"logs/{log}"
        thresh = args.max_stale if args.max_stale is not None else max_stale
        # Window-open grace: right after the window opens the producer may not have written
        # yet (it was idle-sleeping), so its file still carries the prior session's mtime.
        # Suppress the staleness alarm until we're at least `thresh` seconds into the window
        # — long enough that a healthy producer would have written by now.
        into = secs_into_window(window)
        if into is not None and into < thresh:
            emit(OK, f"{svc}: active, window just opened ({into:.0f}s) — freshness not yet checked")
            continue
        age = file_age(fresh_rel)
        if age is None:
            emit(WARN, f"{svc}: active but {fresh_rel} missing")
        elif age > thresh:
            emit(lvl_if_bad if critical else WARN,
                 f"{svc}: active but {fresh_rel} stale {age:.0f}s > {thresh}s "
                 f"(possible silent stall / 401-loop)")
        else:
            emit(OK, f"{svc}: active, fresh ({age:.0f}s)")

    # 2b) YANK evaluation heartbeat — proves the strategy loop is evaluating bars.
    # Only judged while trader-yank is active (the service check above already alarms
    # on not-active; a stopped service must not double-page as a missing heartbeat).
    if is_active(HB_SERVICE):
        for lvl, msg in evaluate_heartbeat(read_heartbeat(),
                                           datetime.now(timezone.utc),
                                           service_uptime_secs(HB_SERVICE)):
            emit(lvl, msg)

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
