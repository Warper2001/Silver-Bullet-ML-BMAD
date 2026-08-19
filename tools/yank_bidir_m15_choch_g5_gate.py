#!/usr/bin/env python3
"""§5 acceptance gate — bidirectional M15 CHoCH.

Runs the gate from
_bmad-output/preregistration_yank_bidirectional_m15_choch.md §5 (sealed
2026-08-19). Config: bearish_only=False, m15_confirmation=True (the REAL
bidirectional-with-a-working-gate run — not §0.1's M15-off exploratory
config), everything else at today's live values including
max_gap_atr_ratio=0.426. Pre-holdout derivation window only
(2025-01-01..2026-02-28), hard-enforced, data/sealed_holdout/ never read.

Gates (pre-committed, reused unmodified from the 2026-05-23 Epic 2 battery's
own bar — not fit to the §0.1 exploratory number):
    G1  bullish-subset N                                    >= 15
    G2  bullish-subset PF                                    > 1.3
    G3  bearish-subset PF vs today's bearish-only baseline   within 10%
    G4  no single calendar month > 40% of bullish N          (no artifact)

RAW pre-ML population (BacktestEngine doesn't apply ml_threshold=0.50) —
same caveat as the gap-ceiling gate; the gate tests structural population
shape, not live P&L.

Usage:
    .venv/bin/python tools/yank_bidir_m15_choch_g5_gate.py
"""
from __future__ import annotations

import dataclasses
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.backtest_engine import BacktestEngine  # noqa: E402
from src.research.config_loader import load_strategy_config  # noqa: E402
from src.research.strategy_core import calc_profit_factor  # noqa: E402
from tools.yank_gap_ceiling_backtest import load_and_write_temp_csv, MAX_GAP_ATR_RATIO  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / "strategy_config.yaml"

# Baseline bearish PF against which G3 is measured — today's live config + new
# ceiling, bearish_only=True (tools/yank_gap_ceiling_backtest.py's NEW row,
# 2026-08-19: N=46, PF=1.053). Recomputed fresh below rather than hardcoded,
# so a drifted baseline fails loudly instead of silently comparing against a
# stale number.


def summarize(trades) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "pf": float("nan"), "net": 0.0}
    pnls = [t.pnl_usd for t in trades]
    return {"n": n, "pf": calc_profit_factor(pnls), "net": sum(pnls)}


def main() -> int:
    print("Loading derivation-window bars (2025-01-01..2026-02-28, HARD-filtered)...", file=sys.stderr)
    tmp_csv, dmin, dmax, n_bars = load_and_write_temp_csv()
    print(f"  {n_bars} bars, {dmin} .. {dmax}", file=sys.stderr)

    live_cfg = load_strategy_config(CONFIG_PATH)
    baseline_cfg = dataclasses.replace(live_cfg, max_gap_atr_ratio=MAX_GAP_ATR_RATIO)
    bidir_cfg = dataclasses.replace(baseline_cfg, bearish_only=False)  # m15_confirmation stays True

    try:
        print("Running BASELINE (bearish_only=True, today's live config + new ceiling)...", file=sys.stderr)
        baseline_trades = BacktestEngine(str(tmp_csv), baseline_cfg).run()
        print(f"  {len(baseline_trades)} trades", file=sys.stderr)

        print("Running BIDIRECTIONAL (bearish_only=False, m15_confirmation=True — the real gate)...", file=sys.stderr)
        bidir_trades = BacktestEngine(str(tmp_csv), bidir_cfg).run()
        print(f"  {len(bidir_trades)} trades", file=sys.stderr)
    finally:
        tmp_csv.unlink(missing_ok=True)

    baseline_s = summarize(baseline_trades)

    bearish = [t for t in bidir_trades if t.direction == "BEARISH"]
    bullish = [t for t in bidir_trades if t.direction == "BULLISH"]
    bearish_s = summarize(bearish)
    bullish_s = summarize(bullish)

    # G1
    g1_pass = bullish_s["n"] >= 15

    # G2
    g2_pass = bool(bullish_s["n"] and bullish_s["pf"] == bullish_s["pf"] and bullish_s["pf"] > 1.3)

    # G3 — bearish subset (with the bidirectional wiring active) vs baseline (bearish_only=True)
    g3_pass = False
    g3_pct_diff = float("nan")
    if baseline_s["pf"] == baseline_s["pf"] and baseline_s["pf"] > 0 and bearish_s["pf"] == bearish_s["pf"]:
        g3_pct_diff = abs(bearish_s["pf"] - baseline_s["pf"]) / baseline_s["pf"] * 100.0
        g3_pass = g3_pct_diff <= 10.0

    # G4 — no single calendar month > 40% of bullish N
    g4_pass = False
    month_counts = Counter()
    if bullish_s["n"] > 0:
        month_counts = Counter(t.timestamp_entry.strftime("%Y-%m") for t in bullish)
        worst_month, worst_count = month_counts.most_common(1)[0]
        g4_pct = worst_count / bullish_s["n"] * 100.0
        g4_pass = g4_pct <= 40.0
    else:
        worst_month, g4_pct = None, float("nan")

    all_pass = g1_pass and g2_pass and g3_pass and g4_pass

    print()
    print("=== §5 acceptance gate — bidirectional M15 CHoCH ===")
    print(f"derivation window: {dmin} .. {dmax}")
    print(f"baseline (bearish_only=True): N={baseline_s['n']}  PF={baseline_s['pf']:.3f}  net=${baseline_s['net']:.2f}")
    print(f"bidirectional run: N={len(bidir_trades)}  ({bearish_s['n']} bearish, {bullish_s['n']} bullish)")
    print()
    g2_pf_str = f"{bullish_s['pf']:.3f}" if bullish_s["pf"] == bullish_s["pf"] else "n/a"
    print(f"G1 (bullish N >= 15)              : N={bullish_s['n']}  -> {'PASS' if g1_pass else 'FAIL'}")
    print(f"G2 (bullish PF > 1.3)             : PF={g2_pf_str}  net=${bullish_s['net']:.2f}  -> {'PASS' if g2_pass else 'FAIL'}")
    print(f"G3 (bearish PF within 10% of base): bearish PF={bearish_s['pf']:.3f} vs baseline PF={baseline_s['pf']:.3f}  "
          f"diff={g3_pct_diff:.2f}%  -> {'PASS' if g3_pass else 'FAIL'}")
    print(f"G4 (no month > 40% of bullish N)  : worst month={worst_month} ({g4_pct:.1f}%)  -> {'PASS' if g4_pass else 'FAIL'}")
    print()
    if bullish_s["n"] > 0:
        print("bullish trades by month:")
        for m, c in sorted(month_counts.items()):
            print(f"  {m}  {c}")
    print()
    print(f"VERDICT: {'ALL GATES PASS — H1 CONFIRMED' if all_pass else 'AT LEAST ONE GATE FAILED — H0, Response B'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
