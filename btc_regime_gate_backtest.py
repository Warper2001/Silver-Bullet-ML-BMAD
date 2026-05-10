#!/usr/bin/env python3
"""Sprint 1 Gate Check: BTC Silver Bullet with ATR-Percentile Volatility Regime Gate.

Runs the raw backtest (no ML) with the vol regime gate applied and prints
a PASS/FAIL verdict against the Sprint 1 decision framework criteria:

  Gate 1: Holdout raw PF  >= 1.40
  Gate 2: Holdout trades  >= 20
  Gate 3: Train trades after regime filter >= 100

Config: NY AM 09:00-10:00 CDT, stop_mult=0.75, uncapped target.
Split:  Train  < Nov 8 2025 | Holdout >= Nov 8 2025.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtest_btc_silver_bullet import (
    BTC_CONTRACT_VALUE,
    COMMISSION,
    LIMIT_CANCEL_BARS,
    MAX_HOLD_BARS,
    MIN_RR,
    POSITION_SIZE,
    _cdt_date,
    _find_next_liquidity_pool,
    detect_confluence,
    detect_fvg_setups,
    detect_mss_events,
    detect_swing_points,
    load_csv_as_bars,
    round_tick,
)
from optimize_btc_silver_bullet import filter_setups_by_zones, execute_param
from train_btc_ml import build_vol_regime_map

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "data/kraken/PF_XBTUSD_1min.csv"
SPLIT_TS = datetime(2025, 11, 8, tzinfo=timezone.utc)
NY_AM_ZONE = [("NY AM", 9, 0, 10, 0)]
STOP_MULT = 0.75
TARGET_CAP_RR = 0  # uncapped

# Sprint 1 gate thresholds (from decision framework)
GATE_HOLDOUT_PF = 1.40
GATE_HOLDOUT_TRADES = 20
GATE_TRAIN_TRADES = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("backtest_btc_silver_bullet").setLevel(logging.WARNING)
logging.getLogger("optimize_btc_silver_bullet").setLevel(logging.WARNING)
logging.getLogger("train_btc_ml").setLevel(logging.WARNING)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def profit_factor(pnls: list) -> float:
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return wins / losses if losses > 0 else float("inf")


def win_rate(pnls: list) -> float:
    return sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0


def _metrics_str(label: str, trades: list) -> str:
    pnls = [t["pnl"] for t in trades]
    if not pnls:
        return f"  {label}: trades=0  PF=N/A  WR=N/A  P&L=$0"
    pf = profit_factor(pnls)
    wr = win_rate(pnls)
    total = sum(pnls)
    return (
        f"  {label}: trades={len(pnls):3d}  PF={pf:.3f}  "
        f"WR={wr:.1%}  P&L=${total:+,.0f}  avg=${total/len(pnls):+.2f}"
    )


# ── Regime-gated setup filter ─────────────────────────────────────────────────

def filter_by_regime(setups: list, bars: list, vol_regime_map: dict) -> list:
    """Return only setups whose CDT date is classified 'medium'."""
    return [
        s for s in setups
        if vol_regime_map.get(_cdt_date(bars[s["index"]].timestamp), "medium") == "medium"
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("BTC Silver Bullet — Sprint 1 Regime Gate Backtest")
    logger.info(f"Config: NY AM 09:00-10:00 CDT | stop_mult={STOP_MULT} | uncapped")
    logger.info(f"Split: train < {SPLIT_TS.date()} | holdout >= {SPLIT_TS.date()}")

    # ── Load bars ─────────────────────────────────────────────────────────────
    logger.info("Loading bars...")
    bars = load_csv_as_bars(DATA_PATH)
    logger.info(f"Bars loaded: {len(bars):,}")

    # ── Build regime map ──────────────────────────────────────────────────────
    logger.info("Building volatility regime map...")
    vol_regime_map = build_vol_regime_map(bars)
    n_total = len(vol_regime_map)
    n_medium = sum(1 for v in vol_regime_map.values() if v == "medium")
    n_low = sum(1 for v in vol_regime_map.values() if v == "low")
    n_extreme = sum(1 for v in vol_regime_map.values() if v == "extreme")
    logger.info(
        f"Regime days — medium: {n_medium} ({n_medium/n_total:.0%})  "
        f"low: {n_low} ({n_low/n_total:.0%})  "
        f"extreme: {n_extreme} ({n_extreme/n_total:.0%})"
    )

    # ── Detect patterns once ──────────────────────────────────────────────────
    logger.info("Detecting patterns (runs once)...")
    swing_highs, swing_lows = detect_swing_points(bars)
    mss_events = detect_mss_events(bars, swing_highs, swing_lows)
    fvg_setups = detect_fvg_setups(bars)
    confluences = detect_confluence(mss_events, fvg_setups)
    logger.info(f"Confluences: {len(confluences)}")

    # ── Kill zone filter ──────────────────────────────────────────────────────
    all_setups = filter_setups_by_zones(confluences, NY_AM_ZONE)
    logger.info(f"NY AM setups: {len(all_setups)}")

    # ── Apply regime filter ───────────────────────────────────────────────────
    regime_setups = filter_by_regime(all_setups, bars, vol_regime_map)
    logger.info(
        f"After regime filter: {len(regime_setups)} setups "
        f"({len(all_setups) - len(regime_setups)} removed)"
    )

    # ── Split train / holdout ─────────────────────────────────────────────────
    train_setups = [s for s in regime_setups if s["timestamp"] < SPLIT_TS]
    holdout_setups = [s for s in regime_setups if s["timestamp"] >= SPLIT_TS]
    logger.info(f"Train setups: {len(train_setups)} | Holdout setups: {len(holdout_setups)}")

    # ── Execute ───────────────────────────────────────────────────────────────
    logger.info("Executing train backtest...")
    train_trades = execute_param(bars, train_setups, stop_mult=STOP_MULT, target_cap_rr=TARGET_CAP_RR)
    logger.info("Executing holdout backtest...")
    holdout_trades = execute_param(bars, holdout_setups, stop_mult=STOP_MULT, target_cap_rr=TARGET_CAP_RR)

    # ── Also run baseline (no regime gate) for comparison ────────────────────
    train_setups_base = [s for s in all_setups if s["timestamp"] < SPLIT_TS]
    holdout_setups_base = [s for s in all_setups if s["timestamp"] >= SPLIT_TS]
    train_base = execute_param(bars, train_setups_base, stop_mult=STOP_MULT, target_cap_rr=TARGET_CAP_RR)
    holdout_base = execute_param(bars, holdout_setups_base, stop_mult=STOP_MULT, target_cap_rr=TARGET_CAP_RR)

    # ── Report ────────────────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("BTC SILVER BULLET — SPRINT 1 REGIME GATE REPORT")
    from train_btc_ml import VOL_REGIME_LOW, VOL_REGIME_HIGH
    print(f"Regime thresholds: P{int(VOL_REGIME_LOW*100):.0f}–P{int(VOL_REGIME_HIGH*100):.0f}  "
          f"(medium = {n_medium}/{n_total} days, {n_medium/n_total:.0%})")
    print(sep)

    print("\nBASELINE (no regime gate):")
    print(_metrics_str("TRAIN   ", train_base))
    print(_metrics_str("HOLDOUT ", holdout_base))

    print("\nWITH REGIME GATE:")
    print(_metrics_str("TRAIN   ", train_trades))
    print(_metrics_str("HOLDOUT ", holdout_trades))

    # ── Sprint 1 gate evaluation ──────────────────────────────────────────────
    holdout_pnls = [t["pnl"] for t in holdout_trades]
    holdout_pf = profit_factor(holdout_pnls)
    n_holdout = len(holdout_trades)
    n_train = len(train_trades)

    gate1_pass = holdout_pf >= GATE_HOLDOUT_PF
    gate2_pass = n_holdout >= GATE_HOLDOUT_TRADES
    gate3_pass = n_train >= GATE_TRAIN_TRADES

    if n_holdout < GATE_HOLDOUT_TRADES:
        logger.warning(
            f"Holdout trade count {n_holdout} < {GATE_HOLDOUT_TRADES}. "
            "Consider widening regime thresholds to P90/P10 before re-running."
        )
        print(
            f"\nHALT: only {n_holdout} holdout trades after regime filter "
            f"(minimum {GATE_HOLDOUT_TRADES}). "
            "Widen VOL_REGIME_LOW/HIGH to 0.10/0.90 and re-run before evaluating gates."
        )
        sys.exit(1)
    if n_train < GATE_TRAIN_TRADES:
        logger.warning(
            f"Train trade count {n_train} < {GATE_TRAIN_TRADES}. "
            "Consider widening regime thresholds to P90/P10 before re-running."
        )

    print(f"\n{sep}")
    print("SPRINT 1 GATE EVALUATION")
    print(sep)
    status = lambda ok: "PASS ✓" if ok else "FAIL ✗"
    print(f"  Gate 1 — Holdout raw PF  >= {GATE_HOLDOUT_PF:.2f}: "
          f"{holdout_pf:.3f}  →  {status(gate1_pass)}")
    print(f"  Gate 2 — Holdout trades  >= {GATE_HOLDOUT_TRADES:d}:   "
          f"{n_holdout:3d}    →  {status(gate2_pass)}")
    print(f"  Gate 3 — Train trades after regime filter >= {GATE_TRAIN_TRADES:d}:  "
          f"{n_train:3d}    →  {status(gate3_pass)}")
    print(sep)

    all_pass = gate1_pass and gate2_pass and gate3_pass
    verdict = "SPRINT 1: PASS — proceed to Sprints 2 and 3" if all_pass else \
              "SPRINT 1: FAIL — investigate fold 3 date range; do not proceed"
    print(f"\n  *** {verdict} ***\n")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
