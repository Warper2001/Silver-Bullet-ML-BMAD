"""
YANK Compressed-Cascade Phase 2 — Prospective OOS Tracker
Pre-registration: _bmad-output/preregistration_yank_compressed_cascade.md (Phase 2, sec 4)
Phase 1 verdict (PASS): _bmad-output/yank_compressed_cascade_phase1_verdict.md

Runs the same candidate cascade (M15 sweep / M5 CHoCH / M1 FVG) against live
bars logged by the already-running shadow-parity process
(logs/yank_shadow_parity.csv, TradeStation columns) and records any trade
whose ENTRY timestamp falls on or after the seal commit -- the fresh
prospective window the seal requires, never the spent 2026-03-01/05-19
holdout.

Idempotent: re-running as the shadow-parity log grows only appends trades
not already in the ledger (natural key = entry_ts), matching this project's
established fix for the trades.db double-logging class of bug.

No orders are placed. Paper/shadow only -- reads an existing log, writes only
to data/yank_compressed_cascade/phase2_trades.csv.

Usage:
    .venv/bin/python yank_compressed_cascade_phase2_tracker.py \
        --shadow-log /path/to/yank_shadow_parity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from yank_compressed_cascade_phase1 import BASELINE_CONFIG, _precompute_gates, _run_cascade

SEAL_COMMIT = "445a9ba"
SEAL_TS = pd.Timestamp("2026-08-19T16:04:27+00:00")  # git log -1 --format=%aI 445a9ba
PHASE2_N_TARGET = 30
LEDGER_PATH = Path("data/yank_compressed_cascade/phase2_trades.csv")
LEDGER_FIELDS = ["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "exit_reason", "pnl"]


def _load_shadow_bars(csv_path: str) -> pd.DataFrame:
    """Build canonical AR9 bars (tz-aware America/New_York) from the TS columns
    of the shadow-parity log. TS was the designated signal source during the
    ProjectX shadow migration (project_projectx_data_migration memory) -- if
    that has since flipped, re-point this at whichever column pair is live."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["minute"], utc=True)
    df = df.rename(columns={"ts_open": "open", "ts_high": "high", "ts_low": "low", "ts_close": "close", "ts_vol": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0).astype("int64")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="first")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    return df


def _load_ledger() -> set[str]:
    if not LEDGER_PATH.exists():
        return set()
    with open(LEDGER_PATH) as f:
        return {row["entry_ts"] for row in csv.DictReader(f)}


def _append_ledger(new_trades: list[dict]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LEDGER_PATH.exists()
    with open(LEDGER_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if is_new:
            writer.writeheader()
        for t in new_trades:
            writer.writerow(t)


def run(shadow_log_path: str) -> dict:
    bars = _load_shadow_bars(shadow_log_path)
    if len(bars) < 200:
        return {"error": f"only {len(bars)} bars available -- need warmup history before any signal can fire"}

    config = BASELINE_CONFIG
    gates = _precompute_gates(bars, config, "15min", "5min")
    result = _run_cascade(bars, config, gates)

    all_trades = result.trades or []
    prospective = [t for t in all_trades if pd.Timestamp(t["entry_ts"]) >= SEAL_TS]

    already_logged = _load_ledger()
    new_trades = [t for t in prospective if t["entry_ts"] not in already_logged]
    _append_ledger(new_trades)

    n_total_logged = len(already_logged) + len(new_trades)
    return {
        "seal_commit": SEAL_COMMIT,
        "seal_ts": str(SEAL_TS),
        "bars_available_through": str(bars.index[-1]),
        "prospective_trades_found_this_run": len(prospective),
        "new_trades_appended": len(new_trades),
        "n_logged_total": n_total_logged,
        "n_target": PHASE2_N_TARGET,
        "n_remaining": max(0, PHASE2_N_TARGET - n_total_logged),
        "ledger": str(LEDGER_PATH),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-log", required=True)
    args = parser.parse_args()

    if not Path(args.shadow_log).exists():
        print(f"ERROR: {args.shadow_log} not found", file=sys.stderr)
        sys.exit(1)

    print(run(args.shadow_log))


if __name__ == "__main__":
    main()
