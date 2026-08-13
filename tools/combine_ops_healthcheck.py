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
  WARN (running but not working):
    - structural silence: confirmed H1 sweep + M15 CHoCH but zero FVG (the 2026-08-07
      YANK failure class — every other check green while the gap gates were unsatisfiable)
    - ledger integrity: gap-fade's hash-chained audit trail is broken or incomplete
      (the 2026-08-06 failure class — a git checkout reverted the live ledgers and
      destroyed two sessions; nobody noticed for six days because nothing checked)
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
import re
from datetime import datetime, time as dtime, timedelta
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
    # Added 2026-08-08. GAP-1 was the ONLY bot absent from this dict, and on
    # 2026-08-08 it died at 12:58:52 UTC in a TradeStation auth outage and stayed
    # dead nine hours with nothing alerting — while being the bot specced for a
    # funded account. Threshold 420s: it polls 60s inside RTH but 300s outside, so
    # anything tighter cries wolf every evening.
    # FLIP TO CRITICAL when it moves to a funded account (buildspec §6).
    "trader-gap-fade":       (False, "gap_fade_live.log",           420, None, None),
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


# --- "active but achieving nothing" -----------------------------------------------
# Every check above answers "is the process alive?". None of them answer "is it doing
# anything?". trader-s26-combine polled the EXPIRED contract MBTN26 from 2026-07-30,
# taking ~3,400 HTTP 404s a day with zero trades, while systemd reported active and its
# log stayed fresh — because writing 404s to a log IS freshness. Nothing flagged it for a
# week. These markers are the bot's own distress signals; a service emitting them
# repeatedly is running, not working. Generic on purpose: the same blind spot applies to
# every bot here, not just the one that hit it.
UNPRODUCTIVE_MARKERS = re.compile(
    r"STALE_DATA|AUTOROLL FAILED|contract may be expired|HTTP 4\d\d from", re.I)
UNPRODUCTIVE_WINDOW_S = 3600
UNPRODUCTIVE_MIN_HITS = 10      # a handful of 4xx is normal API noise; sustained is not
UNPRODUCTIVE_TAIL_BYTES = 512 * 1024   # bounded read — these logs reach tens of MB
_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")


def recent_unproductive_hits(relpath: str):
    """Count distress markers logged in the last UNPRODUCTIVE_WINDOW_S seconds.

    Returns None when the log is missing/unreadable (the freshness check already covers
    that case, and double-reporting one fault as two findings trains people to skim).
    """
    p = BASE / relpath
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2)
            start = max(0, fh.tell() - UNPRODUCTIVE_TAIL_BYTES)
            fh.seek(start)
            tail = fh.read().decode("utf-8", "replace").splitlines()
        # Only the first line of a MID-FILE read can be a fragment. Dropping it
        # unconditionally would silently discard a real line from any log smaller than
        # the tail window.
        if start > 0 and tail:
            tail = tail[1:]
    except OSError:
        return None
    cutoff = datetime.now() - timedelta(seconds=UNPRODUCTIVE_WINDOW_S)
    hits = 0
    for line in tail:
        if not UNPRODUCTIVE_MARKERS.search(line):
            continue
        m = _LOG_TS.match(line)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1).replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts >= cutoff:
            hits += 1
    return hits


# --- "achieving nothing, quietly" (structural silence) -----------------------------
# The markers above catch a bot that is LOUD about failing (404s, stale contracts).
# They cannot catch a bot that fails SILENTLY. Found 2026-08-07: YANK went 17 days with
# zero trades while systemd was active, its log fresh, its token refreshing, and its
# upstream stages logging healthy structure — because detect_fvg's two gap gates had
# become mutually unsatisfiable (0.25 x H1_ATR > $60/$2 = 30pts whenever H1 ATR > 120,
# which was the MEDIAN hour in 2026-06 and 2026-07). Every existing check was green.
#
# The signature of this failure class is a funnel that confirms upstream and then never
# produces downstream. tier2_bar_decisions.csv records exactly that, per bar, so the
# check reads the funnel rather than guessing from prose. See tools/fvg_feasibility.py
# for the arithmetic behind WHY the window closes.
SILENCE_LOG = "logs/tier2_bar_decisions.csv"
SILENCE_SVC = "trader-yank"
SILENCE_WINDOW_DAYS = 5          # ~1 trading week of sessions
SILENCE_MIN_CONFIRMED = 200      # bars of confirmed structure before silence means anything
SILENCE_TAIL_BYTES = 8 * 1024 * 1024   # bounded read — this file reaches >1 GB


def _csv_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def structural_silence(relpath: str = SILENCE_LOG):
    """Look for confirmed upstream structure with zero downstream signal.

    Returns ``(confirmed_bars, fvg_bars, sweep_bars)`` over the last
    SILENCE_WINDOW_DAYS, or None when the funnel log is missing/unreadable/empty.

    ``confirmed`` = H1 sweep active AND M15 CHoCH confirmed on the same bar: the point
    past which the only remaining gate is the FVG gap test. If that count is healthy
    and ``fvg`` is zero, the bot is not waiting on the market — it cannot fire.
    """
    p = BASE / relpath
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2)
            start = max(0, fh.tell() - SILENCE_TAIL_BYTES)
            fh.seek(start)
            tail = fh.read().decode("utf-8", "replace").splitlines()
        # Mid-file reads start on a fragment; a read from byte 0 starts on the header.
        if tail:
            tail = tail[1:]
    except OSError:
        return None

    cutoff = datetime.now(ET) - timedelta(days=SILENCE_WINDOW_DAYS)
    confirmed = fvg = sweep = 0
    seen = False
    # bar_timestamp,h1_sweep_active,kill_zone_active,vol_regime_blocked,m15_confirmed,
    # fvg_detected,action
    for row in csv.reader(tail):
        if len(row) < 6:
            continue
        try:
            ts = datetime.fromisoformat(row[0])
        except ValueError:
            continue
        if ts.tzinfo is None:
            continue
        if ts < cutoff:
            continue
        seen = True
        h1, m15, has_fvg = _csv_bool(row[1]), _csv_bool(row[4]), _csv_bool(row[5])
        if h1:
            sweep += 1
        if h1 and m15:
            confirmed += 1
        if has_fvg:
            fvg += 1
    return (confirmed, fvg, sweep) if seen else None


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
            # Alive and writing — but is it working? A log full of 404s is "fresh".
            hits = recent_unproductive_hits(f"logs/{log}")
            if hits is not None and hits >= UNPRODUCTIVE_MIN_HITS:
                emit(WARN,
                     f"{svc}: active and fresh but NOT WORKING — {hits} distress markers "
                     f"in the last {UNPRODUCTIVE_WINDOW_S // 60}m "
                     f"(stale contract / repeated API errors); check logs/{log}")
            else:
                emit(OK, f"{svc}: active, fresh ({age:.0f}s)")

    # 3) Structural silence — active, fresh, error-free, and unable to fire.
    sil = structural_silence()
    if sil is None:
        emit(WARN, f"{SILENCE_SVC}: {SILENCE_LOG} missing/empty — cannot check for "
                   f"structural silence (the 2026-08-07 failure class)")
    else:
        confirmed, fvg, sweep = sil
        win = f"last {SILENCE_WINDOW_DAYS}d"
        if confirmed >= SILENCE_MIN_CONFIRMED and fvg == 0:
            emit(WARN,
                 f"{SILENCE_SVC}: STRUCTURALLY SILENT — {confirmed} bars with H1 sweep + "
                 f"M15 CHoCH confirmed and ZERO FVG detected ({win}). The bot is not "
                 f"waiting on the market; check whether the gap gates can be satisfied: "
                 f".venv/bin/python tools/fvg_feasibility.py")
        elif confirmed < SILENCE_MIN_CONFIRMED:
            emit(OK, f"{SILENCE_SVC}: only {confirmed} confirmed-structure bars ({win}, "
                     f"sweep={sweep}) — below the {SILENCE_MIN_CONFIRMED} needed to judge silence")
        else:
            emit(OK, f"{SILENCE_SVC}: funnel productive — {confirmed} confirmed-structure "
                     f"bars, {fvg} FVG detected ({win})")

    # 4) Floor monitor data freshness + distance-to-floor headroom
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

    # 5) Ledger integrity — the 2026-08-06 failure class.
    # Delegated to tools/verify_chain.py so the chain logic has one owner. Registered,
    # documented scars are suppressed there; anything reaching us is new.
    try:
        vc = subprocess.run(
            [sys.executable, str(BASE / "tools/verify_chain.py"), "--reconcile"],
            capture_output=True, text=True, timeout=30, cwd=BASE)
        if vc.returncode == 0:
            emit(OK, "gap-fade ledger: chains verify, complete vs trades.db")
        else:
            findings = [ln.strip() for ln in vc.stdout.splitlines()
                        if ln.startswith("[BROKEN]") or "INCOMPLETE" in ln
                        or "DUPLICATED" in ln]
            emit(WARN, "gap-fade ledger: " + ("; ".join(findings) or "verify_chain failed") +
                       " — run .venv/bin/python tools/verify_chain.py --reconcile")
    except Exception as e:
        emit(WARN, f"gap-fade ledger: verify_chain.py did not run ({e})")

    header = {OK: "ALL OK", WARN: "WARNINGS", CRIT: "CRITICAL"}[worst]
    if not (args.quiet and worst == OK):
        print(f"=== Combine ops healthcheck: {header} ===")
        for ln in lines:
            print(ln)
    return worst


if __name__ == "__main__":
    sys.exit(main())
