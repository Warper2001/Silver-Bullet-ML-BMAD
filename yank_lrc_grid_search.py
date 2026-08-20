"""
LRC (Linear-Regression-Channel) strategy — grid search, 2026 data only.

New strategy line, not a YANK amendment (Alex, 2026-08-19): from-scratch
optimization of SL/TP/gap-ratio + a new regression-channel regime gate, on
top of the compressed cascade (M15 sweep / M5 CHoCH / M1 FVG) that Phase 1
showed real in-sample signal for. Dev-phase discipline per Alex's explicit
direction: no OOS gate right now, 2026 calendar year only, aggressive/
exploratory -- this is NOT a pre-registered or sealed test.

Data: merges data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv
(2026-01-01 -> ~2026-06-11) with logs/yank_shadow_parity.csv's TS columns
(2026-06-16 -> present, live shadow-parity log). There's a ~5-day gap
(06-11 -> 06-16) between the two sources -- disclosed, not patched.

Usage:
    .venv/bin/python yank_lrc_grid_search.py --out data/reports/lrc_grid_search.csv
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd

from src.research.strategy_core import StrategyConfig, calc_profit_factor
from src.research.strategy_lrc import (
    LRCConfig,
    precompute_base_gates,
    precompute_regression_features,
    run_lrc_cascade,
)
from yank_compressed_cascade_phase1 import _load_bars
from yank_compressed_cascade_phase2_tracker import _load_shadow_bars

YTD_CSV = "/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
SHADOW_LOG = "/root/Silver-Bullet-ML-BMAD/logs/yank_shadow_parity.csv"

BASE_CONFIG = StrategyConfig(
    entry_pct=0.5,
    atr_threshold=0.5,
    max_gap_dollars=60.0,
    max_gap_atr_ratio=0.426,
    max_hold_bars=60,
    max_pending_bars=240,
    contracts_per_trade=5,
    max_daily_loss=-750.0,
    vol_regime_lookback=120,
    vol_regime_threshold=0.75,
    bearish_only=True,
    h1_sweep_lookback=6,
    commission_per_roundtrip=4.0,
    enable_kill_zone_filter=False,
    m15_confirmation=True,
    tuesday_exclusion=True,
)

# v3 grid -- widened around v2's two winning clusters (lookback=20/5min and
# lookback=100/15min, both slope-only, both at the high end of SL/TP/gap
# tested so far). band_k is a no-op under gate_mode="slope" (see v1/v2
# results) so it's excluded from the slope combos entirely rather than
# wastefully repeated -- see _iter_combos.
GRID = {
    "regression_lookback": [10, 20, 100, 150],
    "band_k": [1.5, 2.0, 2.5],
    "regression_timeframe": ["15min", "5min"],
    "gate_mode": ["slope", "position", "combined"],
    "sl_multiplier": [3.0, 5.0, 7.0],
    "tp_multiplier": [6.0, 8.0, 10.0],
    "min_gap_atr_ratio": [0.25, 0.35, 0.45],
}


def _iter_combos(grid: dict) -> list[dict]:
    """band_k only matters for gate_mode in {position, combined} -- skip the
    redundant repetition for slope (would otherwise 3x every slope cell for
    zero new information, as v1/v2 both showed)."""
    fixed_keys = ["regression_lookback", "regression_timeframe", "sl_multiplier", "tp_multiplier", "min_gap_atr_ratio"]
    fixed_vals = [grid[k] for k in fixed_keys]
    combos = []
    for fixed_combo in itertools.product(*fixed_vals):
        base = dict(zip(fixed_keys, fixed_combo))
        # slope: band_k irrelevant, one placeholder row
        combos.append({**base, "gate_mode": "slope", "band_k": grid["band_k"][0]})
        for gate_mode in ("position", "combined"):
            for bk in grid["band_k"]:
                combos.append({**base, "gate_mode": gate_mode, "band_k": bk})
    return combos


def load_2026_bars() -> pd.DataFrame:
    ytd = _load_bars(YTD_CSV)
    shadow = _load_shadow_bars(SHADOW_LOG)
    combined = pd.concat([ytd, shadow])
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/Silver-Bullet-ML-BMAD/data/reports/lrc_grid_search.csv")
    args = parser.parse_args()

    t0 = time.time()
    bars = load_2026_bars()
    print(f"bars: {len(bars)}  {bars.index[0]} -> {bars.index[-1]}  (loaded {time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    gates = precompute_base_gates(bars, BASE_CONFIG)
    print(f"base gates precomputed {time.time()-t0:.1f}s", flush=True)

    reg_cache: dict[tuple[int, str], object] = {}
    combos = _iter_combos(GRID)
    print(f"grid size: {len(combos)}", flush=True)

    rows = []
    t_start = time.time()
    for n_done, params in enumerate(combos):
        reg_key = (params["regression_lookback"], params["regression_timeframe"])
        if reg_key not in reg_cache:
            reg_cache[reg_key] = precompute_regression_features(bars, *reg_key)
        reg = reg_cache[reg_key]

        lrc = LRCConfig(
            regression_lookback=params["regression_lookback"],
            band_k=params["band_k"],
            regression_timeframe=params["regression_timeframe"],
            gate_mode=params["gate_mode"],
            sl_multiplier=params["sl_multiplier"],
            tp_multiplier=params["tp_multiplier"],
            min_gap_atr_ratio=params["min_gap_atr_ratio"],
        )
        result = run_lrc_cascade(bars, BASE_CONFIG, gates, reg, lrc)
        pf = calc_profit_factor(result.pnls) if result.n_trades > 0 else float("nan")
        total_pnl = sum(result.pnls)
        rows.append({**params, "n_trades": result.n_trades, "n_candidate_bars": result.n_candidate_bars, "pf": pf, "total_pnl": total_pnl})

        if (n_done + 1) % 10 == 0 or n_done == len(combos) - 1:
            elapsed = time.time() - t_start
            rate = (n_done + 1) / elapsed
            eta = (len(combos) - n_done - 1) / rate if rate > 0 else float("nan")
            print(f"[{n_done+1}/{len(combos)}] elapsed={elapsed:.0f}s eta={eta:.0f}s last_pf={pf:.3f} n_trades={result.n_trades}", flush=True)

    df = pd.DataFrame(rows).sort_values("pf", ascending=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("\nTop 10 by PF (n_trades >= 10):")
    print(df[df["n_trades"] >= 10].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
