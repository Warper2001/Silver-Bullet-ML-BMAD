"""
YANK Compressed-Cascade — LR Counter-Trend Filter Tracker (child of Phase 2)
Pre-registration: _bmad-output/preregistration_compressed_cascade_lr_filter.md

Runs the same unmodified compressed-cascade candidate (M15 sweep / M5 CHoCH / M1 FVG,
BASELINE_CONFIG) the parent Phase 2 tracker uses, against the same shadow-parity bars.
For every trade with entry_ts >= SEAL_TS (this seal's own timestamp, distinct from and
later than the parent Phase 2 seal), records whether the LR counter-trend regime filter
(fast_len=195, slow_len=2925 -- frozen by the seal, not further tunable) would have kept
or dropped it, alongside the trade itself.

Does NOT touch or duplicate data/yank_compressed_cascade/phase2_trades.csv -- this is an
additional label on the same underlying trade stream, not a second trade-generation path.

Idempotent: re-running as the shadow-parity log grows only appends trades not already in
this ledger (natural key = entry_ts), matching this project's established double-logging
fix.

No orders are placed. Paper/shadow only -- reads an existing log, writes only to
data/yank_compressed_cascade/lr_filter_trades.csv.

Usage:
    .venv/bin/python yank_compressed_cascade_lr_filter_tracker.py \
        --shadow-log /path/to/yank_shadow_parity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from yank_compressed_cascade_phase1 import BASELINE_CONFIG, _precompute_gates, _run_cascade
from yank_compressed_cascade_phase2_tracker import _load_shadow_bars
from src.ml.regime_detection.lr_channel_detector import LRChannelRegimeDetector

# Seal timestamp for THIS experiment -- set at implementation/commit time, matching the
# pre-registration document's §4. Deliberately distinct from (and later than) the parent
# Phase 2 seal (445a9ba, 2026-08-19) -- the exploratory grid search that motivated this
# experiment touched data through 2026-08-24, so nothing before this timestamp may count
# as fresh evidence for THIS seal, even trades already logged in the parent's ledger.
SEAL_TS = pd.Timestamp("2026-08-25T21:17:06+00:00")
FAST_LEN = 195
SLOW_LEN = 2925
N_TARGET = 30
LEDGER_PATH = Path("data/yank_compressed_cascade/lr_filter_trades.csv")
LEDGER_FIELDS = ["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "exit_reason", "pnl", "lr_kept"]


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
    if len(bars) < SLOW_LEN:
        return {"error": f"only {len(bars)} bars available -- need >= {SLOW_LEN} for LR warmup"}

    config = BASELINE_CONFIG
    gates = _precompute_gates(bars, config, "15min", "5min")
    result = _run_cascade(bars, config, gates)
    all_trades = result.trades or []

    prospective = [t for t in all_trades if pd.Timestamp(t["entry_ts"]) >= SEAL_TS]

    if prospective:
        closes = bars["close"].values
        bar_idx = bars.index
        detector = LRChannelRegimeDetector(fast_len=FAST_LEN, slow_len=SLOW_LEN)
        regimes = detector.fit_predict(closes)

        entry_ts_idx = pd.DatetimeIndex([pd.Timestamp(t["entry_ts"]) for t in prospective])
        bar_pos = bar_idx.searchsorted(entry_ts_idx, side="right") - 1
        for t, pos in zip(prospective, bar_pos):
            regime = regimes[pos] if pos >= 0 else "SIDEWAYS"
            # bearish_only cascade: counter-trend passes unless regime says DOWN
            t["lr_kept"] = bool(regime != "DOWN")

    already_logged = _load_ledger()
    new_trades = [t for t in prospective if t["entry_ts"] not in already_logged]
    _append_ledger(new_trades)

    n_total_logged = len(already_logged) + len(new_trades)

    return {
        "seal_ts": str(SEAL_TS),
        "fast_len": FAST_LEN,
        "slow_len": SLOW_LEN,
        "bars_available_through": str(bars.index[-1]),
        "prospective_trades_found_this_run": len(prospective),
        "new_trades_appended": len(new_trades),
        "n_logged_total": n_total_logged,
        "n_target": N_TARGET,
        "n_remaining": max(0, N_TARGET - n_total_logged),
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
