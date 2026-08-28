"""Stage 1 EXPLORATION — sweep YANK's min_gap_atr_ratio (the FVG floor).

NOT a validation. Runs on data YANK's derivation has already seen (2025), to
locate the floor's frequency/edge trade-off before anything is sealed. Per the
project rule (feedback_derive_dont_assert_one_knob): sweep first, then seal the
choice, then test it out-of-sample on data the sweep never saw.

Diagnosis this follows from (2026-08-28): of 6,936 structural 3-bar bearish gaps
in the live shadow log, the CEILING (gap <= 0.426*H1_ATR) passes 99.6% and the
FLOOR (gap >= 0.25*H1_ATR) passes 1.5%. The floor sits near the 98th percentile
of the actual gap/H1_ATR distribution (median 0.034). The floor is the binding
gate; the ceiling is not.

One knob. Everything else is BASE_CONFIG, unchanged.
"""
from __future__ import annotations
import argparse, sys, time
from dataclasses import replace
import pandas as pd

sys.path.insert(0, "/root/Silver-Bullet-ML-BMAD")
from yank_compressed_cascade_phase1 import _load_bars, _precompute_gates, _run_cascade
from yank_lrc_grid_search import BASE_CONFIG

FLOORS = [0.25, 0.20, 0.175, 0.15, 0.125, 0.10, 0.075, 0.05, 0.025]
NOTIONAL_PER_CT, FEE_BPS = 59000.0, 0.51


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
    ap.add_argument("--label", default="2025 (derivation era — EXPLORATION)")
    a = ap.parse_args()

    cfg = replace(BASE_CONFIG, sl_multiplier=2.0, tp_multiplier=8.0)  # live YAML
    bars = _load_bars(a.csv)
    print(f"bars {len(bars):,}  {bars.index[0]} -> {bars.index[-1]}")
    print(f"window: {a.label}\n", flush=True)

    t0 = time.time()
    gates = _precompute_gates(bars, cfg, "1h", "15min")   # live YANK cascade
    print(f"gates precomputed in {time.time()-t0:.0f}s (floor does not affect them)\n", flush=True)

    print(f"{'floor':>7}{'N':>7}{'freq/day':>10}{'PF':>8}{'net$':>12}{'mean$':>9}"
          f"{'$/ct':>8}{'bps':>8}{'ratio':>8}")
    print("-"*77)
    days = (bars.index[-1] - bars.index[0]).days or 1
    rows = []
    for fl in FLOORS:
        res = _run_cascade(bars, replace(cfg, min_gap_atr_ratio=fl), gates)
        pnls = res.pnls
        n = len(pnls)
        if n == 0:
            print(f"{fl:>7.3f}{0:>7}{'-':>10}{'-':>8}{'-':>12}{'-':>9}{'-':>8}{'-':>8}{'-':>8}")
            continue
        net = sum(pnls)
        gp = sum(p for p in pnls if p > 0); gl = -sum(p for p in pnls if p < 0)
        pf = gp/gl if gl else float("inf")
        mean = net/n
        per_ct = mean/cfg.contracts_per_trade
        bps = 10000*per_ct/NOTIONAL_PER_CT
        rows.append((fl, n, pf, net, mean, bps, bps/FEE_BPS))
        print(f"{fl:>7.3f}{n:>7}{n/days:>10.3f}{pf:>8.3f}{net:>12,.0f}{mean:>9.1f}"
              f"{per_ct:>8.1f}{bps:>8.2f}{bps/FEE_BPS:>7.1f}x", flush=True)

    print("-"*77)
    print(f"  contracts={cfg.contracts_per_trade} (backtest); live YANK runs 2 — bps and PF are")
    print(f"  size-invariant, dollar columns are not.  friction {FEE_BPS} bps, project bar 3x.")
    print(f"\n  EXPLORATION ONLY. No value here is validated. Any choice must be sealed and")
    print(f"  then tested on data this sweep never saw.")


if __name__ == "__main__":
    main()
