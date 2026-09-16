#!/usr/bin/env python3
"""
YANK live PF checkpoint — read-only status report, no trading action.

Why this exists: the ML meta-label filter was disabled live on 2026-09-15 23:53 UTC
(seal da82cfc, `_bmad-output/results_yank_frontmonth_revalidation.md`), because the
front-month re-validation returned INCONCLUSIVE under seal 138cab1's own minimum-sample
rule. YANK now trades unfiltered, and its unfiltered arm is the weaker of the two on
corrected bars (census PF 0.71 vs 0.98). Seal 138cab1 left a forward stop:

    "Disable ML and re-review if live YANK PF < 0.90 after N >= 20 live trades."

ML is already disabled, so the remaining half of that rule is the re-review. This
checkpoint reports the live count and PF against it. It **observes and reports only**:
it never edits config, never touches the bot, and its thresholds are quoted from that
seal, not invented here (AGENTS.md: monitors observe and report until a threshold is
derived from a sweep).

What it reports:
  1. live ledger, all time and since the ML-disable cutover (PF, net, N, wins)
  2. forward-stop status against the sealed N>=20 / PF<0.90 rule
  3. config drift: the live ML gate must still read 0.0
  4. bot health: unit state, restarts, last trade age

Baseline at setup (2026-09-16): 5 live trades since 2026-07-13, +$259.00, PF 1.63,
all of them BEFORE the cutover; 0 trades since. So the forward stop is 15 trades away.

Usage: .venv/bin/python tools/yank_live_pf_check.py [--json]
Exit code is always 0 — this never signals failure to systemd; the report is the output.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRADES_DB = BASE / "data" / "trades.db"
THRESHOLD_JSON = BASE / "models" / "xgboost" / "tier2_threshold.json"
CONFIG_YAML = BASE / "strategy_config.yaml"
YANK_LOG = BASE / "logs" / "yank_streaming_working.log"

TRADER_ID = "trader-yank"
# The restart that disabled the ML filter (results_yank_frontmonth_revalidation.md).
ML_OFF_UTC = datetime(2026, 9, 15, 23, 53, 4, tzinfo=timezone.utc)
# Quoted from seal 138cab1's forward stop. Not derived here; do not tune.
STOP_MIN_N = 20
STOP_PF = 0.90
EXPECTED_GATE = 0.0          # the live ML gate after seal da82cfc


def hr(title: str) -> str:
    return f"\n{'=' * 72}\n{title}\n{'=' * 72}"


def pf_of(pnls: list[float]) -> float | None:
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return None          # undefined: no losing trade yet
    return gains / losses


def fmt_pf(pf: float | None, n: int | None = None) -> str:
    if pf is not None:
        return f"{pf:.3f}"
    return "n/a (no trades)" if n == 0 else "n/a (no losing trade)"


class LedgerUnavailable(Exception):
    """The trade ledger could not be read — distinct from 'no trades'."""


def live_trades() -> list[tuple[datetime, float, str, str]]:
    """(timestamp, pnl, symbol, exit_reason) for real live YANK trades, oldest first.

    write_mode='realtime' matters: most of this table is backfilled backtest rows.
    Timestamps are mixed-format, so each is parsed individually.
    """
    if not TRADES_DB.exists():
        raise LedgerUnavailable(f"{TRADES_DB} not found — run this from the live checkout")
    con = sqlite3.connect(f"file:{TRADES_DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT timestamp, pnl, symbol, exit_reason FROM trades "
            "WHERE trader_id = ? AND write_mode = 'realtime' AND pnl IS NOT NULL",
            (TRADER_ID,),
        ).fetchall()
    finally:
        con.close()
    out = []
    for ts, pnl, sym, reason in rows:
        t = datetime.fromisoformat(str(ts))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        out.append((t.astimezone(timezone.utc), float(pnl), sym or "?", reason or "?"))
    return sorted(out)


def window_stats(trades: list[tuple[datetime, float, str, str]]) -> dict:
    pnls = [t[1] for t in trades]
    return {"n": len(pnls), "net": sum(pnls), "pf": pf_of(pnls),
            "wins": sum(1 for p in pnls if p > 0), "losses": sum(1 for p in pnls if p < 0),
            "first": trades[0][0].isoformat() if trades else None,
            "last": trades[-1][0].isoformat() if trades else None}


def section_ledger(all_t, since_t) -> tuple[str, dict, dict]:
    a, s = window_stats(all_t), window_stats(since_t)
    lines = [hr("1. LIVE LEDGER (trades.db, write_mode='realtime')")]
    for label, st in (("all live trades", a), (f"since ML off ({ML_OFF_UTC:%Y-%m-%d %H:%M} UTC)", s)):
        lines.append(f"  {label}: N={st['n']}  net=${st['net']:+,.2f}  PF={fmt_pf(st['pf'])}  "
                     f"W/L={st['wins']}/{st['losses']}")
        if st["first"]:
            lines.append(f"    first {st['first']}   last {st['last']}")
    if since_t:
        lines.append("\n  Trades since the cutover:")
        for t, pnl, sym, reason in since_t:
            lines.append(f"    {t:%Y-%m-%d %H:%M} UTC  {sym:8} {pnl:+9.2f}  {reason}")
    else:
        lines.append("\n  No live trades since the cutover yet.")
    return "\n".join(lines), a, s


def section_forward_stop(a: dict, s: dict) -> tuple[str, str]:
    """Status against seal 138cab1's forward stop. Reporting only."""
    lines = [hr("2. FORWARD STOP (seal 138cab1, quoted; ML already disabled by da82cfc)")]
    lines.append(f"  Rule: re-review if live PF < {STOP_PF:.2f} after N >= {STOP_MIN_N} live trades.")
    verdicts = {}
    for label, st in (("since-ML-off", s), ("all-live", a)):
        n, pf = st["n"], st["pf"]
        if n < STOP_MIN_N:
            v = f"WAITING ({n}/{STOP_MIN_N} trades)"
        elif pf is None or pf >= STOP_PF:
            v = f"OK (PF {fmt_pf(pf)} >= {STOP_PF:.2f} at N={n})"
        else:
            v = f"REVIEW TRIGGERED (PF {pf:.3f} < {STOP_PF:.2f} at N={n})"
        verdicts[label] = v
        lines.append(f"  {label:14}: {v}")
    lines.append("\n  The since-ML-off window is the operative one: it is the only record of the "
                 "\n  configuration YANK is actually running. The all-live row is context.")
    lines.append("  PF at N=20 is noisy, and per-trade dispersion on corrected bars is ~$400, so "
                 "\n  this rule is a review trigger, not evidence of an edge either way.")
    lines.append("  This monitor takes no action of its own (AGENTS.md).")
    return "\n".join(lines), verdicts["since-ML-off"]


def section_config() -> tuple[str, str]:
    """Returns (text, status) where status is 'clean', 'DRIFT' or 'UNKNOWN'."""
    lines = [hr("3. CONFIG DRIFT (the live ML gate must still read 0.0)")]
    drift, unknown, gate = False, False, None
    try:
        gate = float(json.loads(THRESHOLD_JSON.read_text())["threshold"])
        d = abs(gate - EXPECTED_GATE) > 1e-12
        drift = drift or d
        lines.append(f"  tier2_threshold.json threshold = {gate}  "
                     f"{'OK' if not d else f'DRIFT — expected {EXPECTED_GATE}'}")
    except Exception as e:
        unknown = True
        lines.append(f"  tier2_threshold.json UNREADABLE: {e}")
    try:
        yaml_line = next((ln for ln in CONFIG_YAML.read_text().splitlines()
                          if ln.strip().startswith("ml_threshold:")), "")
        val = float(yaml_line.split(":", 1)[1].split("#")[0])
        d = abs(val - EXPECTED_GATE) > 1e-12
        drift = drift or d
        lines.append(f"  strategy_config.yaml ml_threshold = {val}  "
                     f"{'OK' if not d else f'DRIFT — expected {EXPECTED_GATE}'} (documentation only)")
    except Exception as e:
        unknown = True
        lines.append(f"  strategy_config.yaml ml_threshold UNREADABLE: {e}")
    if gate is not None and abs(gate - EXPECTED_GATE) > 1e-12:
        lines.append("  A non-zero gate means the ML filter is back on. That needs a pre-registration; "
                     "\n  if it happened without one, treat it as unsealed drift.")
    if unknown:
        lines.append("  UNREADABLE is not the same as clean: run this from the live checkout "
                     "(/root/Silver-Bullet-ML-BMAD), where these files exist.")
    return "\n".join(lines), ("DRIFT" if drift else "UNKNOWN" if unknown else "clean")


def _systemctl(*args: str) -> str:
    try:
        return subprocess.run(["systemctl", *args], capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def section_health(all_t) -> tuple[str, bool]:
    lines = [hr("4. BOT HEALTH")]
    active = _systemctl("is-active", "trader-yank")
    restarts = _systemctl("show", "trader-yank", "-p", "NRestarts", "--value")
    since = _systemctl("show", "trader-yank", "-p", "ActiveEnterTimestamp", "--value")
    ok = active == "active"
    lines.append(f"  trader-yank: {active or 'unknown'} (restarts={restarts or '?'}, since {since or '?'})")
    if all_t:
        age = datetime.now(timezone.utc) - all_t[-1][0]
        lines.append(f"  last live trade: {all_t[-1][0]:%Y-%m-%d %H:%M} UTC ({age.days}d ago)")
    if YANK_LOG.exists():
        mtime = datetime.fromtimestamp(YANK_LOG.stat().st_mtime, timezone.utc)
        stale = datetime.now(timezone.utc) - mtime > timedelta(hours=6)
        ok = ok and not stale
        lines.append(f"  log last written {mtime:%Y-%m-%d %H:%M} UTC{'  — STALE (>6h)' if stale else ''}")
    return "\n".join(lines), ok


def main() -> int:
    header = [f"YANK live PF checkpoint — {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
              "Report only. No trading action, no config change."]
    try:
        all_t = live_trades()
    except LedgerUnavailable as e:
        cfg_txt, cfg_status = section_config()
        msg = "\n".join(header + [hr("1. LIVE LEDGER"), f"  LEDGER UNAVAILABLE: {e}", cfg_txt,
                                  hr("VERDICT"),
                                  "LIVE_ALL: unknown (ledger unavailable)",
                                  "LIVE_SINCE_ML_OFF: unknown (ledger unavailable)",
                                  "FORWARD_STOP: UNKNOWN (ledger unavailable)",
                                  f"CONFIG: {cfg_status}", "BOT_HEALTH: not checked",
                                  f"SUMMARY: checkpoint could not read the trade ledger ({e}); "
                                  "nothing about live PF can be concluded from this run."])
        print(json.dumps({"error": str(e)}, indent=2) if "--json" in sys.argv else msg)
        return 0
    since_t = [t for t in all_t if t[0] >= ML_OFF_UTC]
    ledger_txt, a, s = section_ledger(all_t, since_t)
    stop_txt, stop_verdict = section_forward_stop(a, s)
    cfg_txt, cfg_status = section_config()
    health_txt, healthy = section_health(all_t)

    report = header + [ledger_txt, stop_txt, cfg_txt, health_txt, hr("VERDICT")]
    report.append(f"LIVE_ALL: N={a['n']} net=${a['net']:+,.2f} PF={fmt_pf(a['pf'], a['n'])}")
    report.append(f"LIVE_SINCE_ML_OFF: N={s['n']} net=${s['net']:+,.2f} PF={fmt_pf(s['pf'], s['n'])}")
    report.append(f"FORWARD_STOP: {stop_verdict}")
    report.append(f"CONFIG: {cfg_status}{'' if cfg_status == 'clean' else ' — see section 3'}")
    report.append(f"BOT_HEALTH: {'clean' if healthy else 'check needed'}")
    report.append(
        f"SUMMARY: {s['n']} live trade{'' if s['n'] == 1 else 's'} since the ML filter was disabled "
        f"({'PF ' + fmt_pf(s['pf'], s['n']) if s['n'] else 'no PF yet'}); forward stop {stop_verdict}; "
        f"config {cfg_status}; bot {'clean' if healthy else 'needs a look'}."
    )
    text = "\n".join(report)
    if "--json" in sys.argv:
        print(json.dumps({"generated": datetime.now(timezone.utc).isoformat(), "all_live": a,
                          "since_ml_off": s, "forward_stop": stop_verdict,
                          "config": cfg_status, "bot_healthy": healthy}, indent=2, default=str))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
