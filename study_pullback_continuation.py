"""study_pullback_continuation.py — PBC Gate 0: Pullback-Continuation Study.

Stage 0 DIAGNOSTIC — hypothesis generation only, NOT validation.

PRIMARY SPEC (frozen before running, 2026-06-08):
  PULLBACK_MIN_FRAC = 0.5  (50% retrace of impulse)
  MAX_PULLBACK_BARS = 20
  TP_R_MULT = 1.5
  No volume-absorption filter

SETUP CLASS: "the retest IS the entry"
  Trigger → measure impulse → wait for 50% pullback → enter on resumption →
  stop below pullback extreme → 1.5R target.

Two tracks measured independently:
  Track A — ORB Retest Continuation
    trigger = ORB boundary break (build_opening_range + detect_extension)
    reference level L = broken ORB boundary
    impulse X = extension extreme (running high/low up to detection bar)

  Track B — VWAP Extension Pullback
    trigger = first 1-min close beyond VWAP ± 2σ
    reference level L = VWAP at trigger
    impulse X = extension close
    (trade direction = WITH the extension, not against it)

GATE 0 (per track, on primary spec — pre-committed):
  - Tradeable WR ≥ 50%
  - Median structural stop ≤ $150/contract
  - Frequency ≥ 1.0 setups/day
  - No single calendar month WR < 35%  (variance guard)

In-sample: 2025-01-01 → 2026-02-28.  OOS holdout ≥ 2026-03-01 stays sealed.

Usage:
    .venv/bin/python study_pullback_continuation.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.research.sorm_core import (
    POINT_VALUE_USD,
    TICK_SIZE,
    Direction,
    SORMConfig,
    build_opening_range,
    detect_extension,
    load_bars_et,
)
from src.research.strategy_core import calc_atr

# ── Constants ─────────────────────────────────────────────────────────────────

UTC = timezone.utc

START_UTC = datetime(2025, 1, 1, tzinfo=UTC)
END_UTC   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

POINT_VALUE = POINT_VALUE_USD   # $2/pt
STOP_CAP_USD = 150.0            # skip if structural stop > this
STOP_ATR_BUFFER = 0.25          # fraction of 20-bar M1 ATR added beyond pullback extreme
TP_R_MULT = 1.5                 # target in R
HARD_CLOSE_ET = time(15, 55)

# ORB config (Track A)
ORB_CFG = SORMConfig(
    extension_threshold=0.25,   # same as ORBM-2: 0.25× ORB size beyond boundary
    orb_min_size_points=5.0,
)

# VWAP config (Track B)
VWAP_SESSION_OPEN_ET  = time(9, 30)
VWAP_DETECT_START_ET  = time(9, 45)
VWAP_DETECT_END_ET    = time(14, 30)
VWAP_MIN_BARS         = 15       # need this many session bars before σ is reliable
VWAP_SIGMA_K_ENTRY    = 2.0      # σ-band level for extension detection (Track B)

# Grid values (DISCLOSED SENSITIVITY — not the primary spec)
PULLBACK_MIN_FRACS  = [0.3, 0.5]
MAX_PULLBACK_BARS_S = [10, 20]

# Primary spec (FROZEN, evaluated first)
PRIMARY_PULLBACK_FRAC = 0.5
PRIMARY_MAX_BARS      = 20

# Gate 0 thresholds (pre-committed)
GATE0_WR_MIN         = 0.50
GATE0_STOP_USD_MAX   = 150.0
GATE0_FREQ_MIN       = 1.0
GATE0_WORST_MONTH_WR = 0.35


# ── VWAP bands (shared helper, same formula as study_vwap_reversion_rate.py) ──

def compute_session_vwap_bands(
    sess_df: pd.DataFrame,
    k_values: list[float],
) -> pd.DataFrame:
    """Running session-anchored VWAP and volume-weighted σ bands (no look-ahead)."""
    df = sess_df.copy()
    df["typ"] = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    cum_vol = vol.cumsum()
    cum_tv  = (df["typ"] * vol).cumsum()
    df["vwap"] = cum_tv / cum_vol.replace(0, np.nan)
    sq_dev     = vol * (df["typ"] - df["vwap"]) ** 2
    cum_sq     = sq_dev.cumsum()
    vw_var     = cum_sq / cum_vol.replace(0, np.nan)
    df["sigma"] = np.sqrt(vw_var.clip(lower=0.0))
    for k in k_values:
        df[f"upper_{k}"] = df["vwap"] + k * df["sigma"]
        df[f"lower_{k}"] = df["vwap"] - k * df["sigma"]
    return df


# ── Setup record ──────────────────────────────────────────────────────────────

@dataclass
class PBCSetup:
    """One pullback-continuation setup, fully specified."""
    date_et:      date
    track:        str          # "A" (ORB) or "B" (VWAP)
    direction:    str          # "LONG" or "SHORT"
    trigger_close: float
    ref_level:    float        # L — boundary or VWAP
    impulse_pts:  float        # |X − L|
    pullback_frac_spec: float
    max_pb_bars_spec:   int
    # filled by simulation
    trade_taken:  bool = False
    entry_price:  float = 0.0
    stop_price:   float = 0.0
    tp_price:     float = 0.0
    stop_pts:     float = 0.0
    stop_usd:     float = 0.0
    skip_stop_cap: bool = False
    win:          bool = False
    exit_reason:  str = ""


# ── Shared pullback-continuation engine ──────────────────────────────────────

def simulate_pullback_continuation(
    post_trigger_df: pd.DataFrame,   # bars AFTER the trigger bar, RTH only
    direction: str,                   # "LONG" or "SHORT"
    trigger_close: float,             # X (impulse endpoint: extension close or extreme)
    ref_level: float,                 # L (boundary or VWAP at trigger)
    impulse_pts: float,               # |X − L| > 0
    pullback_min_frac: float,
    max_pullback_bars: int,
    atr_at_trigger: float,
    hard_close_str: str,
) -> tuple[bool, float, float, float, float, bool, str]:
    """Core pullback-continuation engine.

    Returns:
        (trade_taken, entry, stop, tp, stop_pts, win, exit_reason)
    """
    if post_trigger_df.empty or impulse_pts <= 0:
        return False, 0.0, 0.0, 0.0, 0.0, False, "NO_SETUP"

    pullback_threshold = impulse_pts * pullback_min_frac

    # Phase 1: hunt for the pullback within max_pullback_bars
    pb_bars = post_trigger_df.iloc[:max_pullback_bars]

    pullback_achieved = False
    pb_extreme_price  = trigger_close  # best (deepest) counter-D price seen
    pb_extreme_low    = trigger_close  # for stop: worst intrabar low
    pb_extreme_high   = trigger_close  # for stop: worst intrabar high

    for ts, row in pb_bars.iterrows():
        bar_close = float(row["close"])
        bar_low   = float(row["low"])
        bar_high  = float(row["high"])

        if direction == "LONG":
            # Pullback = price drops toward ref_level
            if bar_low < pb_extreme_low:
                pb_extreme_low = bar_low
            if bar_close < pb_extreme_price:
                pb_extreme_price = bar_close
            if trigger_close - bar_close >= pullback_threshold:
                pullback_achieved = True
        else:  # SHORT
            # Pullback = price rises toward ref_level
            if bar_high > pb_extreme_high:
                pb_extreme_high = bar_high
            if bar_close > pb_extreme_price:
                pb_extreme_price = bar_close
            if bar_close - trigger_close >= pullback_threshold:
                pullback_achieved = True

    if not pullback_achieved:
        return False, 0.0, 0.0, 0.0, 0.0, False, "NO_PULLBACK"

    # Phase 2: find the resumption entry — first bar that resumes in direction D
    # after the pullback extreme is set.
    # We re-walk from the first bar that achieves pullback depth, tracking the running
    # pullback extreme, and enter on the first bar that reverses back past the extreme.
    running_extreme_close = trigger_close
    running_extreme_low   = trigger_close if direction == "LONG" else float("inf")
    running_extreme_high  = float("inf") if direction == "LONG" else trigger_close
    entry_found           = False
    entry_price           = 0.0
    entry_pos             = -1

    for i, (ts, row) in enumerate(post_trigger_df.iterrows()):
        bar_close = float(row["close"])
        bar_low   = float(row["low"])
        bar_high  = float(row["high"])

        if direction == "LONG":
            # Update running pullback extreme
            if bar_close < running_extreme_close:
                running_extreme_close = bar_close
            if bar_low < running_extreme_low:
                running_extreme_low = bar_low

            # Check if pullback depth achieved
            if trigger_close - running_extreme_close >= pullback_threshold:
                # Now look for resumption: current close > running pullback extreme close
                if bar_close > running_extreme_close and i > 0:
                    # Confirm we've seen the pullback first (not just the first bar)
                    prev_close = float(post_trigger_df.iloc[i - 1]["close"])
                    if bar_close > prev_close:
                        entry_price = bar_close
                        entry_found = True
                        entry_pos   = i
                        break
        else:  # SHORT
            if bar_close > running_extreme_close:
                running_extreme_close = bar_close
            if bar_high > running_extreme_high:
                running_extreme_high = bar_high

            if running_extreme_close - trigger_close >= pullback_threshold:
                if bar_close < running_extreme_close and i > 0:
                    prev_close = float(post_trigger_df.iloc[i - 1]["close"])
                    if bar_close < prev_close:
                        entry_price = bar_close
                        entry_found = True
                        entry_pos   = i
                        break

    if not entry_found:
        return False, 0.0, 0.0, 0.0, 0.0, False, "NO_RESUMPTION"

    # Stop: beyond the pullback extreme (bar extreme, not close)
    atr_buffer = STOP_ATR_BUFFER * atr_at_trigger
    if direction == "LONG":
        stop_price = running_extreme_low - atr_buffer
        stop_pts   = entry_price - stop_price
    else:
        stop_price = running_extreme_high + atr_buffer
        stop_pts   = stop_price - entry_price

    stop_pts = max(stop_pts, TICK_SIZE)
    tp_price = (
        entry_price + TP_R_MULT * stop_pts if direction == "LONG"
        else entry_price - TP_R_MULT * stop_pts
    )

    # Phase 3: walk bars from AFTER entry to hard close
    walk_df = post_trigger_df.iloc[entry_pos + 1:]
    walk_df = walk_df.between_time("00:00", hard_close_str, inclusive="left")

    exit_reason = "TIME_STOP"
    win = False

    for ts, row in walk_df.iterrows():
        bar_high  = float(row["high"])
        bar_low   = float(row["low"])

        if direction == "LONG":
            if bar_low <= stop_price:
                exit_reason = "SL"
                break
            if bar_high >= tp_price:
                exit_reason = "TP"
                win = True
                break
        else:
            if bar_high >= stop_price:
                exit_reason = "SL"
                break
            if bar_low <= tp_price:
                exit_reason = "TP"
                win = True
                break

    return True, entry_price, stop_price, tp_price, stop_pts, win, exit_reason


# ── Track A: ORB Retest Continuation ─────────────────────────────────────────

def process_session_track_a(
    date_et: date,
    sess_df: pd.DataFrame,
    grid_cells: list[tuple[float, int]],  # (pullback_frac, max_pb_bars)
) -> list[PBCSetup]:
    """Detect ORB extension and simulate pullback-continuation for each grid cell."""
    results: list[PBCSetup] = []

    orb = build_opening_range(sess_df, ORB_CFG)
    if orb is None:
        return results

    ext = detect_extension(sess_df, orb, ORB_CFG)
    if ext is None:
        return results

    # Determine trade direction and reference level
    if ext.direction == Direction.BEARISH:  # broke ABOVE orb.high → LONG continuation
        direction  = "LONG"
        ref_level  = orb.high
        trigger_close = ext.extreme_price   # running high = impulse endpoint
    else:                                   # broke BELOW orb.low → SHORT continuation
        direction  = "SHORT"
        ref_level  = orb.low
        trigger_close = ext.extreme_price   # running low

    impulse_pts = abs(trigger_close - ref_level)

    # Post-trigger bars within RTH
    hard_close_str = HARD_CLOSE_ET.strftime("%H:%M")
    post_df = sess_df.loc[sess_df.index > ext.detection_bar_ts]
    post_df = post_df.between_time("00:00", hard_close_str, inclusive="left")

    if post_df.empty or impulse_pts < TICK_SIZE:
        return results

    # ATR from bars up to the detection bar
    bars_to_trigger = sess_df.loc[sess_df.index <= ext.detection_bar_ts]
    atr_val = calc_atr(bars_to_trigger) if len(bars_to_trigger) >= 20 else 10.0

    for pb_frac, max_pb_bars in grid_cells:
        setup = PBCSetup(
            date_et=date_et,
            track="A",
            direction=direction,
            trigger_close=trigger_close,
            ref_level=ref_level,
            impulse_pts=impulse_pts,
            pullback_frac_spec=pb_frac,
            max_pb_bars_spec=max_pb_bars,
        )

        taken, entry, stop, tp, stop_pts, win, reason = simulate_pullback_continuation(
            post_trigger_df=post_df,
            direction=direction,
            trigger_close=trigger_close,
            ref_level=ref_level,
            impulse_pts=impulse_pts,
            pullback_min_frac=pb_frac,
            max_pullback_bars=max_pb_bars,
            atr_at_trigger=atr_val,
            hard_close_str=hard_close_str,
        )

        setup.trade_taken  = taken
        setup.entry_price  = entry
        setup.stop_price   = stop
        setup.tp_price     = tp
        setup.stop_pts     = stop_pts
        setup.stop_usd     = stop_pts * POINT_VALUE
        setup.skip_stop_cap = (stop_pts * POINT_VALUE > STOP_CAP_USD)
        setup.win          = win
        setup.exit_reason  = reason
        results.append(setup)

    return results


# ── Track B: VWAP Extension Pullback ─────────────────────────────────────────

def process_session_track_b(
    date_et: date,
    sess_df: pd.DataFrame,
    grid_cells: list[tuple[float, int]],
) -> list[PBCSetup]:
    """Detect VWAP ±2σ extension and simulate pullback-continuation for each cell."""
    results: list[PBCSetup] = []

    rth_df = sess_df.between_time(
        VWAP_SESSION_OPEN_ET.strftime("%H:%M"), HARD_CLOSE_ET.strftime("%H:%M"),
        inclusive="both",
    )
    if len(rth_df) < VWAP_MIN_BARS + 5:
        return results

    vwap_df = compute_session_vwap_bands(rth_df, [VWAP_SIGMA_K_ENTRY])

    detect_df = vwap_df.between_time(
        VWAP_DETECT_START_ET.strftime("%H:%M"), VWAP_DETECT_END_ET.strftime("%H:%M"),
        inclusive="both",
    )
    if detect_df.empty:
        return results

    session_positions = {ts: i for i, ts in enumerate(vwap_df.index)}
    upper_col = f"upper_{VWAP_SIGMA_K_ENTRY}"
    lower_col = f"lower_{VWAP_SIGMA_K_ENTRY}"
    hard_close_str = HARD_CLOSE_ET.strftime("%H:%M")

    # Find first extension
    trigger_ts    = None
    trigger_close = None
    vwap_at_trig  = None
    direction: Optional[str] = None

    for ts, row in detect_df.iterrows():
        pos = session_positions.get(ts, -1)
        if pos < VWAP_MIN_BARS:
            continue

        close = float(row["close"])
        sigma = float(row["sigma"])
        vwap  = float(row["vwap"])

        if sigma <= 0 or math.isnan(sigma) or math.isnan(vwap):
            continue

        upper = float(row[upper_col])
        lower = float(row[lower_col])

        if math.isnan(upper) or math.isnan(lower):
            continue

        if close > upper:
            direction     = "LONG"   # extension upward → trade continuation LONG
            trigger_close = close
            vwap_at_trig  = vwap
            trigger_ts    = ts
            break
        elif close < lower:
            direction     = "SHORT"  # extension downward → trade continuation SHORT
            trigger_close = close
            vwap_at_trig  = vwap
            trigger_ts    = ts
            break

    if trigger_ts is None or direction is None:
        return results

    ref_level = vwap_at_trig
    impulse_pts = abs(trigger_close - ref_level)

    if impulse_pts < TICK_SIZE:
        return results

    # ATR from bars to trigger
    pos_of_trigger = session_positions.get(trigger_ts, -1)
    bars_to_trigger = vwap_df.iloc[:pos_of_trigger + 1]
    atr_val = calc_atr(bars_to_trigger) if len(bars_to_trigger) >= 20 else 10.0

    # Post-trigger bars (RTH)
    post_df = vwap_df.loc[vwap_df.index > trigger_ts]
    post_df = post_df.between_time("00:00", hard_close_str, inclusive="left")

    if post_df.empty:
        return results

    for pb_frac, max_pb_bars in grid_cells:
        setup = PBCSetup(
            date_et=date_et,
            track="B",
            direction=direction,
            trigger_close=trigger_close,
            ref_level=ref_level,
            impulse_pts=impulse_pts,
            pullback_frac_spec=pb_frac,
            max_pb_bars_spec=max_pb_bars,
        )

        taken, entry, stop, tp, stop_pts, win, reason = simulate_pullback_continuation(
            post_trigger_df=post_df,
            direction=direction,
            trigger_close=trigger_close,
            ref_level=ref_level,
            impulse_pts=impulse_pts,
            pullback_min_frac=pb_frac,
            max_pullback_bars=max_pb_bars,
            atr_at_trigger=atr_val,
            hard_close_str=hard_close_str,
        )

        setup.trade_taken   = taken
        setup.entry_price   = entry
        setup.stop_price    = stop
        setup.tp_price      = tp
        setup.stop_pts      = stop_pts
        setup.stop_usd      = stop_pts * POINT_VALUE
        setup.skip_stop_cap = (stop_pts * POINT_VALUE > STOP_CAP_USD)
        setup.win           = win
        setup.exit_reason   = reason
        results.append(setup)

    return results


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _tradeable(setups: list[PBCSetup]) -> list[PBCSetup]:
    return [s for s in setups if s.trade_taken and not s.skip_stop_cap]


def _grid_stats(setups: list[PBCSetup]) -> dict:
    """Stats for a slice of tradeable setups."""
    n = len(setups)
    if n == 0:
        return {"n": 0, "wr": float("nan"), "stop_med": float("nan"), "stop_p75": float("nan")}
    wins  = sum(1 for s in setups if s.win)
    stops = sorted(s.stop_usd for s in setups)
    return {
        "n": n,
        "wr": wins / n,
        "stop_med": float(np.median(stops)),
        "stop_p75": float(np.percentile(stops, 75)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("PBC Gate 0: Pullback-Continuation Study — Track A (ORB) + Track B (VWAP)")
    print(f"In-sample: {START_UTC.date()} → {END_UTC.date()}")
    print(f"Primary spec: pullback_frac=0.5, max_pb_bars=20, TP=1.5R, no vol filter")
    print("=" * 72)

    # ── Load bars ────────────────────────────────────────────────────────────
    print("\nLoading 1-min bars…", end=" ", flush=True)
    df = load_bars_et([CSV_2025, CSV_2026], START_UTC, END_UTC)
    if df.empty:
        print("ERROR: no bars loaded")
        sys.exit(1)
    print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")

    # ── Build grid of (pb_frac, max_pb_bars) cells ───────────────────────────
    grid_cells: list[tuple[float, int]] = [
        (f, m) for f in PULLBACK_MIN_FRACS for m in MAX_PULLBACK_BARS_S
    ]

    # ── Session loop ─────────────────────────────────────────────────────────
    df["_date"] = df.index.date
    sessions = list(df.groupby("_date"))
    trading_days = 0

    all_a: list[PBCSetup] = []
    all_b: list[PBCSetup] = []

    print(f"\nScanning {len(sessions)} calendar days…", end=" ", flush=True)
    for date_et, sess_df in sessions:
        sess_df = sess_df.drop(columns=["_date"])
        if date_et.weekday() >= 5:
            continue
        rth_check = sess_df.between_time("09:30", "15:55", inclusive="both")
        if len(rth_check) < 30:
            continue
        trading_days += 1

        all_a.extend(process_session_track_a(date_et, sess_df, grid_cells))
        all_b.extend(process_session_track_b(date_et, sess_df, grid_cells))

    print(f"done.\n  {trading_days} trading days.")
    print(f"  Track A raw setups (across all cells): {len(all_a)}")
    print(f"  Track B raw setups (across all cells): {len(all_b)}")

    def _gate0_report(track_name: str, all_setups: list[PBCSetup]) -> None:
        print()
        print("═" * 72)
        print(f"TRACK {track_name} — {'ORB Retest Continuation' if track_name=='A' else 'VWAP Extension Pullback'}")
        print("═" * 72)

        # ── Grid table ───────────────────────────────────────────────────────
        print()
        print("─" * 72)
        print("GRID  (tradeable = stop ≤ $150; win = TP before stop; freq = trade_taken/day)")
        print("─" * 72)
        print(f"  {'pb_frac':>7}  {'max_bars':>8}  {'triggers/d':>10}  {'taken/d':>8}  "
              f"{'N':>5}  {'WR':>6}  {'stop_med $':>10}  {'stop_p75 $':>10}")
        print(f"  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*10}")

        for pb_frac, max_pb_bars in grid_cells:
            cell = [s for s in all_setups
                    if s.pullback_frac_spec == pb_frac and s.max_pb_bars_spec == max_pb_bars]
            triggered = len(cell)  # sessions where trigger fired
            taken_c   = [s for s in cell if s.trade_taken]
            tradeable_c = [s for s in taken_c if not s.skip_stop_cap]
            freq_trigger = triggered / trading_days if trading_days > 0 else 0.0
            freq_taken   = len(taken_c) / trading_days if trading_days > 0 else 0.0
            st = _grid_stats(tradeable_c)

            primary_mark = " ◀ PRIMARY" if (pb_frac == PRIMARY_PULLBACK_FRAC and
                                             max_pb_bars == PRIMARY_MAX_BARS) else ""
            wr_flag = "✅" if st["wr"] >= GATE0_WR_MIN else "❌" if not math.isnan(st["wr"]) else "  "
            print(
                f"  {pb_frac:>7.1f}  {max_pb_bars:>8}  {freq_trigger:>9.2f}/d  {freq_taken:>7.2f}/d  "
                f"{st['n']:>5}  {st['wr']*100:5.1f}%{wr_flag}  "
                f"${st['stop_med']:>8.1f}  ${st['stop_p75']:>8.1f}{primary_mark}"
            )

        # ── Primary spec detail ───────────────────────────────────────────────
        primary_all  = [s for s in all_setups
                        if s.pullback_frac_spec == PRIMARY_PULLBACK_FRAC
                        and s.max_pb_bars_spec == PRIMARY_MAX_BARS]
        primary_taken = [s for s in primary_all if s.trade_taken]
        primary_t     = [s for s in primary_taken if not s.skip_stop_cap]

        n_trigger  = len(primary_all)
        n_taken    = len(primary_taken)
        n_tradeable = len(primary_t)
        n_skipped   = sum(1 for s in primary_taken if s.skip_stop_cap)

        freq_trigger = n_trigger / trading_days if trading_days > 0 else 0.0
        freq_taken   = n_taken  / trading_days if trading_days > 0 else 0.0

        st = _grid_stats(primary_t)
        wr         = st["wr"]
        stop_med   = st["stop_med"]
        n_t        = st["n"]

        print()
        print("─" * 72)
        print(f"PRIMARY SPEC FUNNEL  (pb_frac=0.5, max_pb_bars=20)")
        print("─" * 72)
        print(f"  Trigger sessions:      {n_trigger:>5}  ({freq_trigger:.2f}/day)")
        print(f"  Trade taken:           {n_taken:>5}  ({freq_taken:.2f}/day)  "
              f"[needed pullback + resumption]")
        print(f"  Stop skipped (>$150):  {n_skipped:>5}  "
              f"({n_skipped/max(n_taken,1)*100:.0f}%)")
        print(f"  Tradeable:             {n_tradeable:>5}")

        exit_breakdown = defaultdict(int)
        for s in primary_t:
            exit_breakdown[s.exit_reason] += 1
        print(f"  Exit: TP={exit_breakdown['TP']} SL={exit_breakdown['SL']} "
              f"TIME={exit_breakdown['TIME_STOP']}")

        # By-month
        monthly_days: dict[str, int] = defaultdict(int)
        monthly_setups: dict[str, list] = defaultdict(list)
        for d_et, _ in sessions:
            if d_et.weekday() >= 5:
                continue
            mk = f"{d_et.year}-{d_et.month:02d}"
            monthly_days[mk] += 1
        for s in primary_t:
            mk = f"{s.date_et.year}-{s.date_et.month:02d}"
            monthly_setups[mk].append(s)

        print()
        print("─" * 72)
        print("BY-MONTH  (primary spec, tradeable only)")
        print("─" * 72)
        print(f"  {'Month':<10}  {'N':>4}  {'WR':>6}  {'stop_med':>9}  {'freq/d':>7}  {'var'}")
        worst_month_wr = 1.0
        for mk in sorted(monthly_days.keys()):
            days  = monthly_days[mk]
            recs  = monthly_setups.get(mk, [])
            n_m   = len(recs)
            if n_m == 0:
                print(f"  {mk:<10}  {n_m:>4}   n/a    n/a         n/a   ")
                continue
            wins_m = sum(1 for r in recs if r.win)
            wr_m   = wins_m / n_m
            s_m    = np.median([r.stop_usd for r in recs])
            fr_m   = n_m / days if days > 0 else 0.0
            worst_month_wr = min(worst_month_wr, wr_m)
            var_flag = "✅" if wr_m >= GATE0_WORST_MONTH_WR else "❌"
            print(f"  {mk:<10}  {n_m:>4}  {wr_m*100:5.1f}%  ${s_m:7.1f}  {fr_m:6.2f}/d  {var_flag}")

        # ── Gate 0 verdict ────────────────────────────────────────────────────
        wr_pass       = not math.isnan(wr) and wr >= GATE0_WR_MIN
        stop_pass     = not math.isnan(stop_med) and stop_med <= GATE0_STOP_USD_MAX
        freq_pass     = freq_taken >= GATE0_FREQ_MIN
        variance_pass = worst_month_wr >= GATE0_WORST_MONTH_WR

        print()
        print("═" * 72)
        print(f"GATE 0 VERDICT — Track {track_name} primary spec")
        print("═" * 72)
        print(f"  {'✅ PASS' if wr_pass else '❌ FAIL'}  Win rate ≥ {GATE0_WR_MIN*100:.0f}%"
              f"              [measured: {wr*100:.1f}%  (N={n_t})]")
        print(f"  {'✅ PASS' if stop_pass else '❌ FAIL'}  Median stop ≤ ${GATE0_STOP_USD_MAX:.0f}/contract"
              f"     [measured: ${stop_med:.1f}]")
        print(f"  {'✅ PASS' if freq_pass else '❌ FAIL'}  Frequency ≥ {GATE0_FREQ_MIN:.1f} trades/day"
              f"       [measured: {freq_taken:.2f}/day  (N_taken={n_taken})]")
        print(f"  {'✅ PASS' if variance_pass else '❌ FAIL'}  Worst-month WR ≥ {GATE0_WORST_MONTH_WR*100:.0f}%"
              f"            [measured: {worst_month_wr*100:.1f}%]")

        if wr_pass and stop_pass and freq_pass and variance_pass:
            print()
            print(f"  ✅ GATE 0 PASS — Track {track_name}. Proceed to Stage 1 pre-registration.")
        else:
            failing = []
            if not wr_pass:     failing.append(f"WR {wr*100:.1f}% < {GATE0_WR_MIN*100:.0f}%")
            if not stop_pass:   failing.append(f"Median stop ${stop_med:.1f} > ${GATE0_STOP_USD_MAX:.0f}")
            if not freq_pass:   failing.append(f"Frequency {freq_taken:.2f}/day < {GATE0_FREQ_MIN:.1f}")
            if not variance_pass: failing.append(f"Worst-month WR {worst_month_wr*100:.1f}% < {GATE0_WORST_MONTH_WR*100:.0f}%")
            print()
            print(f"  ❌ GATE 0 FAIL — Track {track_name}. Failing criteria:")
            for f in failing:
                print(f"     • {f}")

    _gate0_report("A", all_a)
    _gate0_report("B", all_b)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    def _pass_check(setups: list[PBCSetup]) -> bool:
        primary_t = [s for s in setups
                     if s.pullback_frac_spec == PRIMARY_PULLBACK_FRAC
                     and s.max_pb_bars_spec == PRIMARY_MAX_BARS
                     and s.trade_taken and not s.skip_stop_cap]
        if not primary_t:
            return False
        n = len(primary_t)
        wr = sum(1 for s in primary_t if s.win) / n
        n_taken = sum(1 for s in setups
                      if s.pullback_frac_spec == PRIMARY_PULLBACK_FRAC
                      and s.max_pb_bars_spec == PRIMARY_MAX_BARS
                      and s.trade_taken)
        freq = n_taken / trading_days if trading_days > 0 else 0.0
        stop_med = np.median([s.stop_usd for s in primary_t])
        monthly: dict[str, list] = defaultdict(list)
        for s in primary_t:
            monthly[f"{s.date_et.year}-{s.date_et.month:02d}"].append(s)
        worst_wr = min(
            (sum(1 for s in v if s.win) / len(v) for v in monthly.values() if v),
            default=0.0,
        )
        return (wr >= GATE0_WR_MIN and stop_med <= GATE0_STOP_USD_MAX
                and freq >= GATE0_FREQ_MIN and worst_wr >= GATE0_WORST_MONTH_WR)

    a_pass = _pass_check(all_a)
    b_pass = _pass_check(all_b)

    print(f"  Track A (ORB Retest):         {'✅ GATE 0 PASS' if a_pass else '❌ GATE 0 FAIL'}")
    print(f"  Track B (VWAP Ext Pullback):  {'✅ GATE 0 PASS' if b_pass else '❌ GATE 0 FAIL'}")
    print()
    if a_pass or b_pass:
        passing = [t for t, p in [("A", a_pass), ("B", b_pass)] if p]
        print(f"  → Proceed to Stage 1 pre-registration for track(s): {', '.join(passing)}")
        print("  → Pre-reg must disclose: continuation thesis derived from prior ORB/VWAP")
        print("    studies (data-observation); primary spec fixed before running this study.")
    else:
        print("  → Both tracks fail. PBC continuation not viable at combine limits.")
        print("  → Do not pre-register. Update memory; return to brainstorm.")
    print("=" * 72)


if __name__ == "__main__":
    main()
