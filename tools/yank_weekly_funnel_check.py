#!/usr/bin/env python3
"""
YANK weekly funnel checkpoint — read-only status report, no trading action.

Runs analyze_filter_funnel.py over the trailing 7 days for trader-yank, then adds the
three questions that funnel alone can't answer: did any REAL entry fire, is the shadow
bullish ledger (the sole evidence base for the drafted, unsealed bidirectional
pre-registration) any closer to its sealing gate, and did the bot run the week without
crashing (which would make the funnel unrepresentative rather than informative).

Baseline (2026-09-11, the day this checkpoint was requested):
  - Month funnel (2026-08-18 -> 09-11, 14,831 bars): 6,383 sweeps -> 2,982 CHoCH ->
    2 FVGs -> 0 past the LR regime filter -> 0 entries.
  - Single-session funnel (2026-09-11, 1,003 bars): 0 entries; binding gate that day
    was M15 CHoCH itself (3 H1 sweeps, 0 confirms) — the 37% volatility-blocked share
    lost zero sweep+CHoCH bars upstream, confirmed from the raw structure flags.
  - Shadow bullish ledger: N=3, total -$253.00, PF 0.260 (two SL losses, one TIME_STOP
    win). Sealing gate for _bmad-output/preregistration_yank_bidirectional_DRAFT.md is
    N>=15 AND PF>=1.00 — not close.

Usage: .venv/bin/python tools/yank_weekly_funnel_check.py
Exit code is always 0 — this never signals failure to systemd; the report is the output.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRADES_DB = BASE / "data" / "trades.db"
SHADOW_LEDGER = BASE / "logs" / "yank_shadow_bullish_trades.csv"
ACTIVE_STATE = BASE / "logs" / "active_trade_state.json"
YANK_LOG = BASE / "logs" / "yank_streaming_working.log"

BASELINE_DATE = "2026-09-11"
BASELINE_SHADOW_N = 3
BASELINE_SHADOW_PNL = -253.00
BASELINE_SHADOW_PF = 0.260
SEAL_MIN_N = 15
SEAL_MIN_PF = 1.00

WINDOW_DAYS = 7


def hr(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def section_funnel() -> str:
    out = [hr(f"1. FILTER FUNNEL — trailing {WINDOW_DAYS} days, trader-yank")]
    try:
        r = subprocess.run(
            [str(BASE / ".venv" / "bin" / "python"), str(BASE / "analyze_filter_funnel.py"),
             "--days", str(WINDOW_DAYS), "--trader", "trader-yank"],
            cwd=BASE, capture_output=True, text=True, timeout=120,
        )
        out.append(r.stdout.strip() or "(no output)")
        if r.returncode != 0:
            out.append(f"\n[analyze_filter_funnel.py exited {r.returncode}]\n{r.stderr[-2000:]}")
    except Exception as exc:
        out.append(f"FAILED to run analyze_filter_funnel.py: {exc}")
    return "\n".join(out)


def section_real_entries() -> tuple[str, int]:
    out = [hr(f"2. REAL ENTRIES — trailing {WINDOW_DAYS} days")]
    cutoff = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat()
    n = 0
    try:
        with sqlite3.connect(f"file:{TRADES_DB}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT timestamp, direction, entry_price, exit_price, pnl, exit_reason "
                "FROM trades WHERE trader_id='trader-yank' AND write_mode='realtime' "
                "AND timestamp >= ? ORDER BY timestamp",
                (cutoff,),
            ).fetchall()
        n = len(rows)
        out.append(f"trades.db rows (write_mode='realtime', timestamp >= {cutoff[:19]}Z): {n}")
        for row in rows:
            out.append(f"  {row}")
    except Exception as exc:
        out.append(f"FAILED to query trades.db: {exc}")

    try:
        placed = subprocess.run(
            ["grep", "-c", "TIER 2 LIMIT PLACED", str(YANK_LOG)],
            capture_output=True, text=True,
        )
        out.append(f"\n'TIER 2 LIMIT PLACED' occurrences in the full log "
                   f"(not window-filtered, sanity cross-check): "
                   f"{placed.stdout.strip() or '0'}")
    except Exception as exc:
        out.append(f"grep failed: {exc}")

    try:
        state = ACTIVE_STATE.read_text().strip()
        out.append(f"\nlogs/active_trade_state.json: {state}")
        out.append(f"(baseline on {BASELINE_DATE}: last_trading_date was still 2026-08-17)")
    except Exception as exc:
        out.append(f"could not read active_trade_state.json: {exc}")

    return "\n".join(out), n


def section_shadow_ledger() -> tuple[str, int, float, float]:
    out = [hr("3. SHADOW BULLISH LEDGER — evidence base for the unsealed bidirectional draft")]
    n = 0
    pnl_total = 0.0
    pf = float("nan")
    try:
        import csv
        with SHADOW_LEDGER.open() as f:
            rows = list(csv.DictReader(f))
        n = len(rows)
        pnls = [float(r["pnl_usd"]) for r in rows]
        pnl_total = sum(pnls)
        wins = sum(p for p in pnls if p > 0)
        losses = sum(p for p in pnls if p < 0)
        pf = wins / abs(losses) if losses < 0 else float("inf") if wins > 0 else float("nan")

        out.append(f"N = {n}   total pnl = ${pnl_total:+.2f}   PF = {pf:.3f}")
        for r in rows:
            out.append(f"  {r['timestamp_entry']}  {r['direction']:5s}  "
                       f"pnl=${float(r['pnl_usd']):+.2f}  {r['exit_reason']}")

        d_n = n - BASELINE_SHADOW_N
        d_pnl = pnl_total - BASELINE_SHADOW_PNL
        out.append(f"\nvs baseline ({BASELINE_DATE}): N {BASELINE_SHADOW_N}->{n} ({d_n:+d}), "
                   f"pnl ${BASELINE_SHADOW_PNL:+.2f}->${pnl_total:+.2f} ({d_pnl:+.2f}), "
                   f"PF {BASELINE_SHADOW_PF:.3f}->{pf:.3f}")

        sealable = n >= SEAL_MIN_N and pf >= SEAL_MIN_PF
        out.append(f"\nSealing gate: N>={SEAL_MIN_N} AND PF>={SEAL_MIN_PF:.2f} "
                   f"-> {'MET' if sealable else 'not met'} "
                   f"(N {'OK' if n >= SEAL_MIN_N else f'needs {SEAL_MIN_N - n} more'}, "
                   f"PF {'OK' if pf >= SEAL_MIN_PF else 'below bar'})")
    except FileNotFoundError:
        out.append("shadow ledger file not found")
    except Exception as exc:
        out.append(f"FAILED to read shadow ledger: {exc}")
    return "\n".join(out), n, pnl_total, pf


def section_health() -> tuple[str, bool]:
    out = [hr(f"4. BOT HEALTH — trailing {WINDOW_DAYS} days")]
    healthy = True
    try:
        active = subprocess.run(["systemctl", "is-active", "trader-yank"],
                                capture_output=True, text=True).stdout.strip()
        out.append(f"systemctl is-active trader-yank: {active}")
        healthy = healthy and active == "active"
    except Exception as exc:
        out.append(f"systemctl check failed: {exc}")
        healthy = False

    try:
        since = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
        errs = subprocess.run(
            ["bash", "-c",
             f"awk -F'|' '$1 >= \"{since}\"' {YANK_LOG} 2>/dev/null | "
             f"grep -icE 'traceback|exception|fatal' || true"],
            capture_output=True, text=True,
        ).stdout.strip()
        n_err = int(errs or 0)
        out.append(f"ERROR/Traceback/Exception lines in-window: {n_err}")
        healthy = healthy and n_err == 0
    except Exception as exc:
        out.append(f"log error-scan failed: {exc}")

    out.append(f"\noverall: {'CLEAN' if healthy else 'CHECK NEEDED'}")
    return "\n".join(out), healthy


def main() -> int:
    now = datetime.now(timezone.utc)
    report = [
        f"YANK WEEKLY FUNNEL CHECKPOINT",
        f"generated: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"window: trailing {WINDOW_DAYS} days",
        f"baseline for comparison: {BASELINE_DATE} (single-session) and the "
        f"2026-08-18..09-11 month-long funnel",
    ]

    report.append(section_funnel())
    entries_txt, n_entries = section_real_entries()
    report.append(entries_txt)
    shadow_txt, shadow_n, shadow_pnl, shadow_pf = section_shadow_ledger()
    report.append(shadow_txt)
    health_txt, healthy = section_health()
    report.append(health_txt)

    sealable = shadow_n >= SEAL_MIN_N and shadow_pf >= SEAL_MIN_PF
    report.append(hr("VERDICT"))
    report.append(f"REAL_ENTRIES: {n_entries}")
    report.append(f"SHADOW_LEDGER: N={shadow_n} pnl=${shadow_pnl:+.2f} PF={shadow_pf:.3f}")
    report.append(f"SEALABLE: {'yes' if sealable else 'no'}")
    report.append(f"BOT_HEALTH: {'clean' if healthy else 'check needed'}")
    report.append(
        f"SUMMARY: {n_entries} real entr{'y' if n_entries == 1 else 'ies'} this week; "
        f"shadow bullish ledger at N={shadow_n}/PF={shadow_pf:.3f} against a "
        f"N>={SEAL_MIN_N}/PF>={SEAL_MIN_PF:.2f} sealing gate "
        f"({'MET' if sealable else 'not met'}); bot ran "
        f"{'clean' if healthy else 'with issues — see section 4'}."
    )

    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
