"""
LRC strategy — random-null test on the grid search's stable winners.

Tests whether the two configs that kept winning across three widened grids
(lookback=100/15min and lookback=150/15min, both slope-only, SL5/TP8,
gap_ratio=0.35) beat a direction-matched random-entry control sharing every
structure/regime gate -- same S12 methodology as
yank_compressed_cascade_phase1.py's Phase 1 test.

2026 data only (same merged dataset as the grid search), matching Alex's
dev-phase discipline. Not pre-registered/sealed -- exploratory, per Alex's
explicit direction to skip the OOS gate for now.

Usage:
    .venv/bin/python yank_lrc_null_test.py
"""

from __future__ import annotations

import numpy as np

from src.research.strategy_core import calc_profit_factor
from src.research.strategy_lrc import LRCConfig, precompute_base_gates, precompute_regression_features, run_lrc_cascade
from yank_lrc_grid_search import BASE_CONFIG, load_2026_bars

N_SEEDS = 100

CANDIDATES = [
    LRCConfig(regression_lookback=100, band_k=1.5, regression_timeframe="15min", gate_mode="slope", sl_multiplier=5.0, tp_multiplier=8.0, min_gap_atr_ratio=0.35),
    LRCConfig(regression_lookback=150, band_k=1.5, regression_timeframe="15min", gate_mode="slope", sl_multiplier=5.0, tp_multiplier=8.0, min_gap_atr_ratio=0.35),
]


def test_one(bars, gates, lrc: LRCConfig) -> dict:
    reg = precompute_regression_features(bars, lrc.regression_lookback, lrc.regression_timeframe)

    real = run_lrc_cascade(bars, BASE_CONFIG, gates, reg, lrc)
    if real.n_trades == 0:
        return {"lrc": lrc, "error": "zero trades"}
    real_pf = calc_profit_factor(real.pnls)
    p_enter = real.n_trades / real.n_candidate_bars if real.n_candidate_bars > 0 else 0.0

    null_pfs = []
    for seed in range(N_SEEDS):
        res = run_lrc_cascade(bars, BASE_CONFIG, gates, reg, lrc, random_seed=seed, p_enter=p_enter)
        if res.n_trades > 0:
            pf = calc_profit_factor(res.pnls)
            if np.isfinite(pf):
                null_pfs.append(pf)

    null_arr = np.array(null_pfs)
    median = float(np.median(null_arr)) if len(null_arr) else float("nan")
    p90 = float(np.percentile(null_arr, 90)) if len(null_arr) else float("nan")
    pct_rank = float((null_arr < real_pf).mean() * 100) if len(null_arr) else float("nan")

    if real_pf < median:
        verdict = "PIVOT"
    elif real_pf > p90:
        verdict = "PATTERNS SURVIVE"
    else:
        verdict = "AMBIGUOUS = TREATED AS FAIL"

    return {
        "lookback": lrc.regression_lookback,
        "timeframe": lrc.regression_timeframe,
        "gate_mode": lrc.gate_mode,
        "sl": lrc.sl_multiplier,
        "tp": lrc.tp_multiplier,
        "gap_ratio": lrc.min_gap_atr_ratio,
        "real_pf": real_pf,
        "real_n_trades": real.n_trades,
        "real_n_candidate_bars": real.n_candidate_bars,
        "p_enter": p_enter,
        "n_null_sims_with_trades": len(null_arr),
        "null_median": median,
        "null_p90": p90,
        "pct_rank": pct_rank,
        "verdict": verdict,
    }


def main() -> None:
    bars = load_2026_bars()
    print(f"bars: {len(bars)}  {bars.index[0]} -> {bars.index[-1]}", flush=True)
    gates = precompute_base_gates(bars, BASE_CONFIG)
    print("base gates precomputed", flush=True)

    for lrc in CANDIDATES:
        print(f"\n--- testing lookback={lrc.regression_lookback} timeframe={lrc.regression_timeframe} ---", flush=True)
        result = test_one(bars, gates, lrc)
        print(result, flush=True)


if __name__ == "__main__":
    main()
