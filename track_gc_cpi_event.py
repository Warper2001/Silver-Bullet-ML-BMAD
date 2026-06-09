#!/usr/bin/env python3
"""
track_gc_cpi_event.py — Prospective tracker for GC/MGC CPI post-catalyst test.

Usage (after each monthly CPI release):
    .venv/bin/python track_gc_cpi_event.py --date 2026-06-11

What this does:
  1. Verifies gc_cpi_config.yaml hash matches the pre-registration seal
  2. Loads GC 1-min data covering the event date
  3. Applies frozen parameters to compute the trade (or "did not qualify")
  4. Appends one row to data/reports/gc_cpi_prospective_log.csv
  5. Reports running cumulative stats and early-stop status

Prerequisites:
  - GC 1-min data must cover the event date + 2h30m post-release
    Update gc_1min_2025_2026.csv via download_gc_1min.py (update end date + contract)
  - Pre-registration commit must predate this event date

Protocol:
  After running this script, commit the updated log:
    git add data/reports/gc_cpi_prospective_log.csv
    git commit -m "prospective CPI log: YYYY-MM-DD result"
"""
import argparse
import csv
import hashlib
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT        = Path(__file__).parent
CONFIG_PATH = ROOT / "gc_cpi_config.yaml"
GC_PATH     = ROOT / "data/processed/dollar_bars/1_minute/gc_1min_2025_2026.csv"
LOG_PATH    = ROOT / "data/reports/gc_cpi_prospective_log.csv"
PREREG_PATH = ROOT / "_bmad-output/preregistration_gc_cpi_prospective.md"

LOG_FIELDS = [
    "date", "qualified", "direction", "net_move_pts", "min_move_threshold_pts",
    "entry_price", "stop_price", "tp_price", "stop_usd", "atr_at_entry",
    "exit_reason", "bars_held", "pnl_usd",
    # running cumulative stats
    "cum_n_events", "cum_n_qualifying", "cum_wins", "cum_wr",
    "cum_pf", "cum_avg_pnl", "early_stop_triggered",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_config_hash() -> bool:
    """Check that gc_cpi_config.yaml hash matches the one embedded in the pre-registration doc."""
    if not PREREG_PATH.exists():
        print("WARNING: Pre-registration doc not found — cannot verify config hash.")
        print("         Proceed only if you trust the config is unchanged since pre-reg commit.")
        return True
    prereg_text = PREREG_PATH.read_text()
    current_hash = _sha256(CONFIG_PATH)
    if current_hash[:16] in prereg_text:
        print(f"✅ Config hash verified ({current_hash[:16]}…)")
        return True
    else:
        print(f"❌ CONFIG HASH MISMATCH — gc_cpi_config.yaml has changed since pre-registration!")
        print(f"   Current hash: {current_hash[:64]}")
        print(f"   This result is NOT valid under the pre-registered protocol.")
        print(f"   If parameters were changed intentionally, start a new pre-registration.")
        return False


def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    return df.set_index("timestamp").sort_index()


def compute_event(gc: pd.DataFrame, cfg: dict, event_date: str) -> dict:
    """
    Apply frozen parameters to one CPI event.
    Returns a result dict with all log fields populated.
    """
    event_ts = pd.Timestamp(f"{event_date} {cfg['event_time_et']}",
                            tz="America/New_York")

    idx_arr = gc.index
    pos = idx_arr.searchsorted(event_ts)

    if pos >= len(gc):
        return {"error": "No GC bars found at or after event time — data may not cover this date."}
    actual_ts = idx_arr[pos]
    if abs((actual_ts - event_ts).total_seconds()) > 120:
        return {"error": f"Nearest bar ({actual_ts}) is >2 min from event time ({event_ts})."}
    if pos == 0:
        return {"error": "Event is the very first bar — no pre-event reference available."}

    atr_win   = cfg["atr_window"]
    wait_bars = cfg["wait_bars"]
    mm_atr    = cfg["min_move_atr"]
    sm        = cfg["stop_atr_mult"]
    tp_mult   = cfg["tp_mult"]
    hold_max  = cfg["hold_max_bars"]
    mgc_pv    = cfg["mgc_pv"]
    commission = cfg["commission"]
    stop_cap  = cfg["stop_cap_usd"]

    cl = gc["close"].values
    hi = gc["high"].values
    lo = gc["low"].values

    pre_close = cl[pos - 1]

    # ATR at pre-event bar (rolling window)
    atr_start = max(0, pos - 1 - atr_win)
    atr_vals  = gc["high"].values[atr_start:pos] - gc["low"].values[atr_start:pos]
    pre_atr   = float(atr_vals.mean()) if len(atr_vals) >= atr_win // 2 else float("nan")
    if np.isnan(pre_atr) or pre_atr <= 0:
        return {"error": "Insufficient bars to compute ATR before event."}

    entry_pos = pos + wait_bars
    if entry_pos >= len(gc):
        return {"error": f"Data does not extend {wait_bars} bars past event time."}

    entry_close = cl[entry_pos]
    entry_atr_start = max(0, entry_pos - atr_win)
    entry_atr_vals  = gc["high"].values[entry_atr_start:entry_pos + 1] - \
                      gc["low"].values[entry_atr_start:entry_pos + 1]
    entry_atr = float(entry_atr_vals.mean()) if len(entry_atr_vals) >= atr_win // 2 else pre_atr

    net_move    = entry_close - pre_close
    min_move_pts = mm_atr * pre_atr

    if abs(net_move) < min_move_pts:
        return {
            "date": event_date, "qualified": False,
            "net_move_pts": round(net_move, 3),
            "min_move_threshold_pts": round(min_move_pts, 3),
            "direction": None, "entry_price": None, "stop_price": None,
            "tp_price": None, "stop_usd": None, "atr_at_entry": round(entry_atr, 3),
            "exit_reason": None, "bars_held": None, "pnl_usd": None,
        }

    direction = 1 if net_move > 0 else -1
    entry     = entry_close
    stop_dist_raw = sm * entry_atr
    stop_usd_raw  = stop_dist_raw * mgc_pv
    stop_usd  = min(stop_usd_raw, stop_cap)
    stop_dist = stop_usd / mgc_pv
    stop_p    = entry - direction * stop_dist
    tp_p      = entry + direction * stop_dist * tp_mult

    # scan forward
    exit_reason = "TIME"; exit_price = None; bars_held = hold_max
    for k in range(entry_pos + 1, min(entry_pos + 1 + hold_max, len(gc))):
        hi_k = hi[k]; lo_k = lo[k]
        hit_tp   = (direction ==  1 and hi_k >= tp_p) or (direction == -1 and lo_k <= tp_p)
        hit_stop = (direction ==  1 and lo_k <= stop_p) or (direction == -1 and hi_k >= stop_p)
        if hit_tp:
            exit_reason = "TP"; exit_price = tp_p; bars_held = k - entry_pos; break
        if hit_stop:
            exit_reason = "STOP"; exit_price = stop_p; bars_held = k - entry_pos; break

    if exit_price is None:
        last_k = min(entry_pos + hold_max, len(gc) - 1)
        exit_price = cl[last_k]

    pnl = (exit_price - entry) * direction * mgc_pv - commission

    return {
        "date": event_date, "qualified": True,
        "direction": "LONG" if direction == 1 else "SHORT",
        "net_move_pts": round(net_move, 3),
        "min_move_threshold_pts": round(min_move_pts, 3),
        "entry_price": round(entry, 2), "stop_price": round(stop_p, 2),
        "tp_price": round(tp_p, 2), "stop_usd": round(stop_usd, 2),
        "atr_at_entry": round(entry_atr, 3),
        "exit_reason": exit_reason, "bars_held": bars_held,
        "pnl_usd": round(pnl, 2),
    }


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return list(csv.DictReader(f))


def compute_cumulative(past_rows: list[dict], new_result: dict) -> dict:
    """Add cumulative stats to the new result row."""
    all_rows = past_rows + [new_result]
    n_events     = len(all_rows)
    qualifying   = [r for r in all_rows if str(r.get("qualified", "")).lower() in ("true", "1")]
    n_qual       = len(qualifying)
    wins         = sum(1 for r in qualifying if float(r.get("pnl_usd", 0)) > 0)
    wr           = wins / n_qual if n_qual > 0 else 0.0
    gross_w      = sum(float(r["pnl_usd"]) for r in qualifying if float(r["pnl_usd"]) > 0)
    gross_l      = abs(sum(float(r["pnl_usd"]) for r in qualifying if float(r["pnl_usd"]) < 0))
    pf           = gross_w / max(1e-9, gross_l)
    avg_pnl      = sum(float(r["pnl_usd"]) for r in qualifying) / n_qual if n_qual > 0 else 0.0

    new_result["cum_n_events"]     = n_events
    new_result["cum_n_qualifying"] = n_qual
    new_result["cum_wins"]         = wins
    new_result["cum_wr"]           = round(wr, 4)
    new_result["cum_pf"]           = round(pf, 4)
    new_result["cum_avg_pnl"]      = round(avg_pnl, 2)

    # early stop check (from config)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    early_stop = (n_qual >= cfg["early_stop_n"]) and (pf < cfg["early_stop_pf"])
    new_result["early_stop_triggered"] = early_stop

    return new_result


def append_to_log(row: dict) -> None:
    exists = LOG_PATH.exists() and LOG_PATH.stat().st_size > 0
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def print_summary(past_rows: list[dict], new_result: dict, cfg: dict) -> None:
    qualifying = [r for r in past_rows + [new_result]
                  if str(r.get("qualified", "")).lower() in ("true", "1")]
    n_qual = len(qualifying)

    print()
    print("=" * 70)
    print(f"PROSPECTIVE CPI RESULT — {new_result['date']}")
    print("=" * 70)

    if not new_result.get("qualified"):
        move = new_result.get("net_move_pts", "?")
        thr  = new_result.get("min_move_threshold_pts", "?")
        print(f"  DID NOT QUALIFY: |net_move| = {move} pts < {thr} pts ({cfg['min_move_atr']}×ATR)")
        print(f"  No trade taken. Event counted toward calendar total but not toward N_qualifying.")
    else:
        win = new_result.get("pnl_usd", 0) > 0
        print(f"  Direction:   {new_result['direction']}")
        print(f"  Net move:    {new_result['net_move_pts']:+.2f} pts (threshold: ≥{new_result['min_move_threshold_pts']:.2f})")
        print(f"  Entry:       {new_result['entry_price']:.2f}")
        print(f"  Stop/TP:     {new_result['stop_price']:.2f} / {new_result['tp_price']:.2f}")
        print(f"  Stop $:      ${new_result['stop_usd']:.2f}/MGC")
        print(f"  Exit:        {new_result['exit_reason']} after {new_result['bars_held']} bars")
        print(f"  P&L:         ${new_result['pnl_usd']:+.2f}  ({'WIN ✅' if win else 'LOSS ❌'})")

    print()
    print("  Cumulative prospective stats:")
    print(f"    Calendar events logged:   {new_result['cum_n_events']}")
    print(f"    Qualifying trades:        {n_qual}  "
          f"(need {cfg['n_min_qualifying']} for final verdict)")
    if n_qual > 0:
        print(f"    Win rate:                 {new_result['cum_wr']:.1%}")
        print(f"    Profit factor:            {new_result['cum_pf']:.3f}")
        print(f"    Avg net P&L:              ${new_result['cum_avg_pnl']:.2f}/trade")

    print()
    if new_result.get("early_stop_triggered"):
        print(f"  ⛔ EARLY STOP TRIGGERED: PF={new_result['cum_pf']:.3f} < {cfg['early_stop_pf']} "
              f"after N={n_qual} qualifying events.")
        print(f"     Declare FAIL. CPI edge is not present prospectively.")
    elif n_qual >= cfg["n_min_qualifying"]:
        pf_ok  = new_result["cum_pf"] >= cfg["final_pf_threshold"]
        ev_ok  = new_result["cum_avg_pnl"] > cfg["final_ev_threshold"]
        wr_ok  = new_result["cum_wr"] >= cfg["final_wr_threshold"]
        passed = pf_ok and ev_ok and wr_ok
        print(f"  {'✅ FINAL VERDICT: PASS' if passed else '❌ FINAL VERDICT: FAIL'}")
        print(f"     PF {new_result['cum_pf']:.3f} {'≥' if pf_ok else '<'} {cfg['final_pf_threshold']}  "
              f"EV ${new_result['cum_avg_pnl']:.2f} {'>' if ev_ok else '≤'} $0  "
              f"WR {new_result['cum_wr']:.1%} {'≥' if wr_ok else '<'} {cfg['final_wr_threshold']:.0%}")
        if passed:
            print(f"     Proceed to Gate 1 full-combine backtest.")
        else:
            print(f"     CPI subgroup was a noise artefact. Record and close this thread.")
    else:
        remaining = cfg["n_min_qualifying"] - n_qual
        print(f"  🕐 Tracking: {remaining} more qualifying event(s) needed for final verdict.")
        # early stop check status
        early_n  = cfg["early_stop_n"]
        early_pf = cfg["early_stop_pf"]
        if n_qual < early_n:
            print(f"     Early stop check activates at N_qualifying = {early_n} "
                  f"(if PF < {early_pf}).")
        else:
            print(f"     Early stop: PF={new_result['cum_pf']:.3f} ≥ {early_pf} — continue.")

    print("=" * 70)
    print(f"  Log updated: {LOG_PATH}")
    print(f"  Next: git add {LOG_PATH} && git commit")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Log one prospective CPI event.")
    parser.add_argument("--date", required=True,
                        help="CPI release date (YYYY-MM-DD)")
    parser.add_argument("--gc-data", default=str(GC_PATH),
                        help="Path to GC 1-min CSV (default: %(default)s)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip config hash verification (use only in emergencies)")
    args = parser.parse_args()

    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: --date must be YYYY-MM-DD (got: {args.date})")
        return 1

    print(f"GC/MGC CPI Prospective Tracker — {args.date}")
    print()

    # Verify config
    if not args.no_verify:
        if not _verify_config_hash():
            print("Aborting. Use --no-verify only if you understand the implications.")
            return 1

    # Check pre-reg exists
    if not PREREG_PATH.exists():
        print("ERROR: Pre-registration document not found.")
        print(f"  Expected: {PREREG_PATH}")
        print("  Run prereg_gc_cpi_prospective.py first and commit the result.")
        return 1

    # Load config
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    # Check that this date is after the pre-registration commit
    # (we can't automatically verify this, but we warn if the date seems early)
    print(f"  Config:  gc_cpi_config.yaml v{cfg.get('version', 'unknown')}")
    print(f"  Event:   {cfg['event_type']} at {cfg['event_time_et']} ET")
    print(f"  Params:  WAIT={cfg['wait_bars']}bar  MM={cfg['min_move_atr']}×ATR  "
          f"SM={cfg['stop_atr_mult']}×  TP={cfg['tp_mult']}×  HOLD={cfg['hold_max_bars']}bar")
    print()

    # Load GC data
    gc_data_path = Path(args.gc_data)
    if not gc_data_path.exists():
        print(f"ERROR: GC data not found at {gc_data_path}")
        print(f"  Run download_gc_1min.py with an updated end date to fetch recent data.")
        return 1

    print(f"Loading GC data from {gc_data_path}…")
    gc = load_et(gc_data_path)
    print(f"  Bars: {len(gc):,}  ({gc.index[0].date()} → {gc.index[-1].date()})")

    # Check data covers event date
    event_date_ts = pd.Timestamp(f"{args.date} {cfg['event_time_et']}",
                                  tz="America/New_York")
    if gc.index[-1] < event_date_ts:
        print(f"ERROR: GC data ends at {gc.index[-1].date()}, before event date {args.date}.")
        print(f"  Update GC data: run download_gc_1min.py with end date ≥ {args.date}.")
        return 1

    # Process event
    result = compute_event(gc, cfg, args.date)
    if "error" in result:
        print(f"ERROR processing event: {result['error']}")
        return 1

    # Load existing log and check for duplicate
    past_rows = load_log()
    if any(r["date"] == args.date for r in past_rows):
        print(f"WARNING: {args.date} is already in the log. Skipping to avoid duplicate.")
        print(f"  Use --no-verify and manually edit {LOG_PATH} if you need to correct it.")
        return 0

    # Add cumulative stats
    result = compute_cumulative(past_rows, result)

    # Print summary
    print_summary(past_rows, result, cfg)

    # Append to log
    append_to_log(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
