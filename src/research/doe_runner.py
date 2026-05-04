#!/usr/bin/env python3
"""DOE runner for Tier 2 strategy parameter sweep (story 6-6).

Runs an L9 Taguchi fractional-factorial design across three axes:
  A — SL/TP ATR multiplier   (L1=2.0, L2=3.5, L3=5.0)
  B — Session window level   (L1=extended, L2=baseline, L3=morning-afternoon)
  C — FVG min size (ATR mult) (L1=0.10, L2=0.25, L3=0.50)

Axis D (instrument universe) is fixed at L1=MNQ only (only MNQ data available).

Acceptance gates per run (raw backtest — no ML):
  n_trades     >= 300
  profit_factor >= 1.10
  win_rate      >= 0.40

Output:
  data/reports/doe_summary.csv   — all runs with gate_pass flag
  data/reports/doe_main_effects.csv — mean PF per axis level
"""

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Axis Definitions ─────────────────────────────────────────────────────────── #

AXIS_A = [2.0, 3.5, 5.0]          # SL/TP multiplier
AXIS_B = ["extended", "baseline", "morning-afternoon"]   # session windows
AXIS_C = [0.10, 0.25, 0.50]       # ATR threshold (FVG min size)

# Standard L9 Taguchi orthogonal array (indices 0-based into each axis)
L9 = [
    (0, 0, 0),
    (0, 1, 1),
    (0, 2, 2),
    (1, 0, 1),
    (1, 1, 2),
    (1, 2, 0),
    (2, 0, 2),
    (2, 1, 0),
    (2, 2, 1),
]

# Gates
GATE_N_TRADES = 300
GATE_PF = 1.10
GATE_WIN_RATE = 0.40

REPORT_DIR = Path("data/reports")
ML_DIR = Path("data/ml_training")
PYTHON = ".venv/bin/python"
BACKTEST = "src/research/backtest_zero_bias_optimized.py"


def _parse_stdout(text: str) -> dict:
    """Extract n_trades, win_rate, profit_factor from backtest stdout."""
    n_trades = 0
    win_rate = 0.0
    profit_factor = 0.0

    m = re.search(r"Total Trades:\s+(\d+)", text)
    if m:
        n_trades = int(m.group(1))

    m = re.search(r"Win Rate:\s+([\d.]+)%", text)
    if m:
        win_rate = float(m.group(1))

    m = re.search(r"Profit Factor:\s+([\d.]+)", text)
    if m:
        profit_factor = float(m.group(1))

    m = re.search(r"Total P&L:\s+\$?([-\d.]+)", text)
    sharpe = float("nan")  # not printed by backtest; placeholder

    return {"n_trades": n_trades, "win_rate": win_rate, "profit_factor": profit_factor, "sharpe": sharpe}


def run_doe():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ML_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for run_idx, (a_i, b_i, c_i) in enumerate(L9, start=1):
        sl_mult = AXIS_A[a_i]
        tp_mult = AXIS_A[a_i]
        session = AXIS_B[b_i]
        atr_thresh = AXIS_C[c_i]

        export_features = ML_DIR / f"doe_run_{run_idx:02d}_features.csv"
        export_history = ML_DIR / f"doe_run_{run_idx:02d}_history.csv"
        report_file = REPORT_DIR / f"doe_run_{run_idx:02d}.txt"

        cmd = [
            PYTHON, BACKTEST,
            "--sl-mult", str(sl_mult),
            "--tp-mult", str(tp_mult),
            "--atr-threshold", str(atr_thresh),
            "--session-windows", session,
            "--export",
            "--export-path", str(export_features),
            "--history", str(export_history),
        ]

        print(f"\n[Run {run_idx:02d}/09] SL/TP={sl_mult}x | session={session} | ATR={atr_thresh}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined = result.stdout + result.stderr
        report_file.write_text(combined)

        if result.returncode != 0:
            print(f"  ERROR (exit {result.returncode}): {result.stderr[:200]}")

        metrics = _parse_stdout(result.stdout)
        gate = (
            metrics["n_trades"] >= GATE_N_TRADES
            and metrics["profit_factor"] >= GATE_PF
            and metrics["win_rate"] >= GATE_WIN_RATE
        )

        status = "PASS" if gate else "FAIL"
        print(f"  n={metrics['n_trades']} WR={metrics['win_rate']:.1f}% PF={metrics['profit_factor']:.3f} → {status}")

        rows.append({
            "run_id": run_idx,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "atr_threshold": atr_thresh,
            "session_level": session,
            "instrument_level": "MNQ",
            "n_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "sharpe": metrics["sharpe"],
            "gate_pass": gate,
            "features_csv": str(export_features),
            "history_csv": str(export_history),
        })

    summary = pd.DataFrame(rows)
    summary_path = REPORT_DIR / "doe_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n✅ DOE summary written to {summary_path}")

    # Gate-pass summary
    survivors = summary[summary["gate_pass"]]
    print(f"\n{'='*60}")
    print(f"SURVIVORS (gate_pass=True): {len(survivors)} / {len(summary)}")
    if len(survivors) > 0:
        print(survivors[["run_id", "sl_mult", "session_level", "atr_threshold", "n_trades", "win_rate", "profit_factor"]].to_string(index=False))
    else:
        print("  No survivors — all runs failed gates. Present results to Alex before proceeding.")
    print(f"{'='*60}")

    return summary


def compute_main_effects(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean PF per axis level (controlling for others via orthogonal design)."""
    axis_map = {
        "sl_mult":       ("A", AXIS_A),
        "session_level": ("B", AXIS_B),
        "atr_threshold": ("C", AXIS_C),
    }

    records = []
    for col, (axis_name, levels) in axis_map.items():
        for level in levels:
            subset = summary[summary[col] == level]
            mean_pf = subset["profit_factor"].mean()
            mean_n = subset["n_trades"].mean()
            records.append({
                "axis": axis_name,
                "column": col,
                "level": level,
                "mean_profit_factor": round(mean_pf, 4),
                "mean_n_trades": round(mean_n, 1),
                "n_runs": len(subset),
            })

    effects = pd.DataFrame(records)
    effects_path = Path("data/reports/doe_main_effects.csv")
    effects.to_csv(effects_path, index=False)

    print("\n📊 Main Effects (mean PF per axis level):")
    print(effects.to_string(index=False))
    print(f"\n✅ Main effects written to {effects_path}")

    return effects


def identify_best_config(summary: pd.DataFrame) -> dict | None:
    survivors = summary[summary["gate_pass"]]
    if survivors.empty:
        return None
    best = survivors.loc[survivors["profit_factor"].idxmax()]
    print(f"\n⭐ Best surviving config: Run {int(best['run_id'])}")
    print(f"   SL/TP={best['sl_mult']}x | session={best['session_level']} | ATR={best['atr_threshold']}")
    print(f"   n_trades={int(best['n_trades'])} | WR={best['win_rate']:.1f}% | PF={best['profit_factor']:.3f}")
    print(f"   Features CSV: {best['features_csv']}")
    print(f"   History CSV:  {best['history_csv']}")
    return best.to_dict()


if __name__ == "__main__":
    summary = run_doe()
    effects = compute_main_effects(summary)
    best = identify_best_config(summary)

    if best is None:
        print("\n⚠️  No DOE survivors. Report to Alex and halt before proceeding to ML training.")
        sys.exit(1)

    print("\n✅ DOE complete. Proceed to Task 3 (LR model training) using the best config above.")
