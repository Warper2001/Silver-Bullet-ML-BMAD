"""
LRC strategy Phase 2 — prospective shadow tracker.

Pre-registration: _bmad-output/preregistration_lrc_strategy.md
Primary candidate (Amendment 2, Alex's call 2026-08-20): lookback=150,
15min, slope-only, SL5/TP8, gap_ratio=0.35.

Same mechanism as yank_compressed_cascade_phase2_tracker.py: reads the
already-running shadow-parity log (logs/yank_shadow_parity.csv) rather than
standing up a new live bot, filters to trades with entry_ts on or after the
seal commit, and appends new ones to an idempotent ledger (natural key =
entry_ts) so re-running as the log grows only adds what's new.

SEAL_TS is the original seal commit (1b6aedf, 2026-08-20T02:47:12Z), not
Amendment 2's later commit -- lookback=150's exact parameters were already
fully specified in the original seal (as the documented sibling); only its
*primary* label changed later the same day. Nothing about what counts as a
"lookback=150 trade" changed between the two commits.

No orders placed. Paper/shadow only.

Usage:
    .venv/bin/python yank_lrc_phase2_tracker.py --shadow-log /path/to/yank_shadow_parity.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from src.research.strategy_lrc import LRCConfig, precompute_base_gates, precompute_regression_features, run_lrc_cascade
from yank_lrc_grid_search import BASE_CONFIG
from yank_compressed_cascade_phase2_tracker import _load_shadow_bars

SEAL_COMMIT = "1b6aedf"
SEAL_TS = pd.Timestamp("2026-08-20T02:47:12+00:00")
PHASE2_N_TARGET = 30
LEDGER_PATH = Path("data/lrc_strategy/phase2_trades.csv")
LEDGER_FIELDS = ["entry_ts", "exit_ts", "direction", "entry_price", "exit_price", "exit_reason", "pnl"]

PRIMARY = LRCConfig(
    regression_lookback=150,
    band_k=1.5,
    regression_timeframe="15min",
    gate_mode="slope",
    sl_multiplier=5.0,
    tp_multiplier=8.0,
    min_gap_atr_ratio=0.35,
)


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

    gates = precompute_base_gates(bars, BASE_CONFIG)
    reg = precompute_regression_features(bars, PRIMARY.regression_lookback, PRIMARY.regression_timeframe)
    result = run_lrc_cascade(bars, BASE_CONFIG, gates, reg, PRIMARY)

    all_trades = getattr(result, "trades", None) or []
    prospective = [t for t in all_trades if pd.Timestamp(t["entry_ts"]) >= SEAL_TS] if all_trades else []

    already_logged = _load_ledger()
    new_trades = [t for t in prospective if t["entry_ts"] not in already_logged]
    _append_ledger(new_trades)

    n_total_logged = len(already_logged) + len(new_trades)
    return {
        "seal_commit": SEAL_COMMIT,
        "seal_ts": str(SEAL_TS),
        "primary": "lookback=150/15min/slope/SL5/TP8/gap0.35",
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
