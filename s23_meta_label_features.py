"""
S23: Meta-label feature generator — H1·M15·M1·g0.25
Program C Phase 2 ML meta-labeling

Runs the H1·M15·M1·g0.25 cascade (identical to S22) on 2025 pre-cutoff
data and records features + outcomes for every executed trade.

NO holdout access. NO pre-registration gate.
Data source: data/processed/mnq_1min_2025.csv  (pre-cutoff only)
Output:      data/ml_training/s23_meta_labels_2025.csv

Features recorded at M1 FVG detection time (no lookahead):
  gap_atr_ratio       — gap / h1_atr  (continuous value of ≥0.25 filter)
  gap_dollars         — gap * MNQ_DOLLAR
  gap_m1_atr_ratio    — gap / m1_atr  (gap quality vs M1 noise)
  h1_atr              — H1 ATR at entry
  m1_atr              — M1 ATR at entry
  h1_vol_pct          — percentile of H1 ATR in 120-bar history (< 0.75 to pass filter)
  hour_et             — hour of day Eastern Time (0-23)
  dow_et              — day of week Eastern Time (0=Mon, ..., 4=Fri; Tue blocked)
  m1_bars_since_sweep — M1 bars elapsed since H1 sweep fired
  m1_bars_since_choch — M1 bars elapsed since M15 CHoCH fired

Outcome columns:
  label       — 1 if TP hit, 0 if SL or time-stop
  exit_type   — "TP", "SL", or "TIME"
  pnl_1x      — P&L in dollars per contract (1x)
"""

import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── Frozen parameters (identical to S22 / pre-registration) ──────────────────
SL_MULT           = 5.0
TP_MULT           = 6.0
ATR_PERIOD        = 20
VOL_LOOKBACK      = 120
VOL_THRESH        = 0.75
ATR_THRESHOLD     = 0.5
MIN_GAP_ATR_RATIO = 0.25
MAX_GAP_DOLLARS   = 60.0
MNQ_DOLLAR        = 2.0
ENTRY_PCT         = 0.5
MNQ_TICK          = 0.25
H1_BAR_CAP        = 3000
CHOCH_ATR_MULT    = 0.3
SWING_RADIUS      = 2
SWEEP_TF_HOURS    = 1
CONFIRM_TF_MIN    = 15
MAX_HOLD_MIN      = 60
MAX_PENDING_MIN   = 240

DATA_PATH   = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
OUTPUT_PATH = Path("data/ml_training/s23_meta_labels_2025.csv")
ET_TZ       = pytz.timezone("US/Eastern")

FEATURE_COLS = [
    "entry_ts", "exit_ts", "label", "exit_type", "pnl_1x",
    "entry_price", "sl", "tp",
    "gap_atr_ratio", "gap_dollars", "gap_m1_atr_ratio",
    "h1_atr", "m1_atr", "h1_vol_pct",
    "hour_et", "dow_et",
    "m1_bars_since_sweep", "m1_bars_since_choch",
]

from collections import namedtuple
Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close"])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bars(path: Path) -> list:
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        sys.exit(1)
    bars = []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bars.append(Bar(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            ))
    bars.sort(key=lambda b: b.timestamp)
    return bars


def resample_bars(bars: list, tf_min: int) -> list:
    df = pd.DataFrame({
        "timestamp": [b.timestamp for b in bars],
        "open":  [b.open  for b in bars],
        "high":  [b.high  for b in bars],
        "low":   [b.low   for b in bars],
        "close": [b.close for b in bars],
    }).set_index("timestamp")
    resampled = df.resample(f"{tf_min}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    result = []
    for ts, row in resampled.iterrows():
        ts_dt = ts.to_pydatetime()
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        result.append(Bar(
            timestamp=ts_dt,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        ))
    return result


def build_completed_idx(bars_1min: list, bars_tf: list, tf_min: int) -> list:
    result = [-1] * len(bars_1min)
    j = 0
    n_tf = len(bars_tf)
    tf_delta = timedelta(minutes=tf_min)
    for i, b in enumerate(bars_1min):
        while j < n_tf and bars_tf[j].timestamp + tf_delta <= b.timestamp:
            j += 1
        result[i] = j - 1
    return result


# ── Utilities (identical to S22) ──────────────────────────────────────────────

def snap_tick(price: float) -> float:
    return round(round(price / MNQ_TICK) * MNQ_TICK, 4)


def is_market_open(ts: datetime) -> bool:
    wd, h = ts.weekday(), ts.hour
    if wd == 5: return False
    if wd == 6: return h >= 23
    if wd == 4: return h < 22
    return h != 22


def is_tuesday_et(ts: datetime) -> bool:
    return ts.astimezone(ET_TZ).weekday() == 1


def calc_atr(bars: list, end_idx: int, period: int = ATR_PERIOD) -> float:
    if end_idx < period + 1:
        return 10.0
    trs = []
    for i in range(end_idx - period, end_idx):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs))


# ── Vol regime (extended to expose percentile) ────────────────────────────────

class VolRegimeTracker:
    def __init__(self):
        self._atr_history: list = []
        self.current_h1_atr: float = 0.0
        self._last_fed: int = -1

    def feed_up_to(self, h1_bars: list, up_to: int) -> None:
        for j in range(self._last_fed + 1, up_to + 1):
            if j < ATR_PERIOD + 1:
                continue
            trs = []
            for k in range(j - ATR_PERIOD, j):
                h = h1_bars[k].high; l = h1_bars[k].low; pc = h1_bars[k-1].close
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            self.current_h1_atr = float(np.mean(trs))
            self._atr_history.append(self.current_h1_atr)
            if len(self._atr_history) > VOL_LOOKBACK:
                self._atr_history.pop(0)
        self._last_fed = max(self._last_fed, up_to)

    def is_high(self) -> bool:
        if len(self._atr_history) < 20 or self.current_h1_atr <= 0:
            return False
        pct = sum(v < self.current_h1_atr for v in self._atr_history) / len(self._atr_history)
        return pct > VOL_THRESH

    def current_pct(self) -> float:
        """Return the percentile rank of the current H1 ATR (0-1)."""
        if len(self._atr_history) < 20 or self.current_h1_atr <= 0:
            return 0.5
        return sum(v <= self.current_h1_atr for v in self._atr_history) / len(self._atr_history)


# ── Sweep and CHoCH (identical to S22) ───────────────────────────────────────

def detect_bearish_sweep(htf_bars: list, up_to_idx: int):
    n = up_to_idx + 1
    if n < 5:
        return False, None
    start = max(0, up_to_idx - H1_BAR_CAP)
    bars  = htf_bars[start:up_to_idx + 1]
    n     = len(bars)
    if n < 5:
        return False, None
    last_bar   = bars[-1]
    last_high  = last_bar.high
    last_close = last_bar.close
    last_ts    = last_bar.timestamp
    cutoff_ts  = last_ts - timedelta(hours=2)
    for i in range(2, n - 3):
        h = bars[i].high
        if not (h > bars[i-1].high and h > bars[i-2].high
                and h > bars[i+1].high and h > bars[i+2].high):
            continue
        if bars[i].timestamp >= cutoff_ts:
            continue
        if last_high > h and last_close < h:
            return True, last_ts
    return False, None


def get_latest_swing_low(bars: list, up_to_idx: int):
    r = SWING_RADIUS
    for i in range(up_to_idx - r, r - 1, -1):
        lo = bars[i].low
        if all(bars[i + k].low >= lo for k in range(-r, r + 1) if k != 0):
            return lo
    return None


def choch_fired(bars_m15: list, conf_idx: int, atr_m15: float) -> bool:
    r = SWING_RADIUS
    if conf_idx < r * 2 + 1:
        return False
    swing_low = get_latest_swing_low(bars_m15, conf_idx - 1)
    if swing_low is None:
        return False
    return bars_m15[conf_idx].close < swing_low - CHOCH_ATR_MULT * atr_m15


# ── Strategy with feature recording ──────────────────────────────────────────

def run_strategy_with_features(bars_1min: list) -> list:
    """
    Run H1·M15·M1·g0.25 cascade (identical to S22) and record
    features + outcomes for every executed trade.
    Returns list of dicts, one per executed trade.
    """
    bars_m15 = resample_bars(bars_1min, CONFIRM_TF_MIN)
    bars_h1  = resample_bars(bars_1min, 60)

    m15_idx = build_completed_idx(bars_1min, bars_m15, CONFIRM_TF_MIN)
    h1_idx  = build_completed_idx(bars_1min, bars_h1,  60)

    vol_tracker  = VolRegimeTracker()
    last_h1_vol  = -1
    last_htf_idx = -1

    sweep_active       = False
    sweep_expires_ts   = datetime.min.replace(tzinfo=timezone.utc)
    sweep_fire_m1_bar  = 0    # M1 bar index when sweep fired
    confirm_done       = False
    choch_fire_m1_bar  = 0    # M1 bar index when CHoCH fired
    last_m15_conf      = -1

    pending       = False
    active        = False
    entry_price   = sl_price = tp_price = 0.0
    pending_bars  = active_bars = 0
    pending_rec   = None   # feature dict for the trade currently pending/active

    records = []

    for i, bar in enumerate(bars_1min):
        cur_ts = bar.timestamp

        cur_h1 = h1_idx[i]
        if cur_h1 > last_h1_vol and cur_h1 >= 0:
            last_h1_vol = cur_h1
            vol_tracker.feed_up_to(bars_h1, cur_h1)

        regime_high = vol_tracker.is_high()

        # ── Advance active trade ──────────────────────────────────────────────
        if active:
            active_bars += 1
            if bar.high >= sl_price:
                pnl_1x = (entry_price - sl_price) * MNQ_DOLLAR
                pending_rec["exit_ts"]   = cur_ts.isoformat()
                pending_rec["exit_type"] = "SL"
                pending_rec["pnl_1x"]    = round(pnl_1x, 2)
                pending_rec["label"]     = 0
                records.append(pending_rec)
                pending_rec = None
                active = False; active_bars = 0
            elif bar.low <= tp_price:
                pnl_1x = (entry_price - tp_price) * MNQ_DOLLAR
                pending_rec["exit_ts"]   = cur_ts.isoformat()
                pending_rec["exit_type"] = "TP"
                pending_rec["pnl_1x"]    = round(pnl_1x, 2)
                pending_rec["label"]     = 1
                records.append(pending_rec)
                pending_rec = None
                active = False; active_bars = 0
            elif active_bars >= MAX_HOLD_MIN:
                pnl_1x = (entry_price - bar.close) * MNQ_DOLLAR
                pending_rec["exit_ts"]   = cur_ts.isoformat()
                pending_rec["exit_type"] = "TIME"
                pending_rec["pnl_1x"]    = round(pnl_1x, 2)
                pending_rec["label"]     = 0
                records.append(pending_rec)
                pending_rec = None
                active = False; active_bars = 0
            if active:
                continue

        # ── Advance pending limit order ───────────────────────────────────────
        if pending:
            pending_bars += 1
            if bar.high >= entry_price:
                active = True; pending = False; pending_bars = 0; active_bars = 0
                continue
            elif pending_bars >= MAX_PENDING_MIN:
                pending = False; pending_bars = 0; pending_rec = None
            else:
                continue

        if not is_market_open(cur_ts):
            continue
        if is_tuesday_et(cur_ts):
            continue
        if regime_high:
            continue

        # ── Expire sweep ──────────────────────────────────────────────────────
        if sweep_active and cur_ts >= sweep_expires_ts:
            sweep_active  = False
            confirm_done  = False
            last_m15_conf = -1
            last_htf_idx  = -1

        # ── Detect H1 sweep ───────────────────────────────────────────────────
        if not sweep_active:
            if cur_h1 >= 4 and cur_h1 > last_htf_idx:
                last_htf_idx = cur_h1
                detected, sweep_ts = detect_bearish_sweep(bars_h1, cur_h1)
                if detected:
                    sweep_active      = True
                    sweep_expires_ts  = sweep_ts + timedelta(hours=SWEEP_TF_HOURS * 6)
                    sweep_fire_m1_bar = i
                    confirm_done      = False
                    last_m15_conf     = -1

        if not sweep_active:
            continue

        # ── M15 CHoCH confirmation ────────────────────────────────────────────
        if not confirm_done:
            cur_m15 = m15_idx[i]
            if cur_m15 >= SWING_RADIUS * 2 + 1 and cur_m15 > last_m15_conf:
                last_m15_conf = cur_m15
                atr_m15 = calc_atr(bars_m15, cur_m15 + 1)
                if choch_fired(bars_m15, cur_m15, atr_m15):
                    confirm_done     = True
                    choch_fire_m1_bar = i

        if not confirm_done:
            continue

        # ── M1 FVG detection ──────────────────────────────────────────────────
        if i < 2:
            continue

        c1 = bars_1min[i - 2]
        c2 = bars_1min[i - 1]
        c3 = bars_1min[i]

        if not (c1.low > c3.high and c2.close < c2.open):
            continue
        gap = c1.low - c3.high
        if gap <= 0:
            continue
        m1_atr = calc_atr(bars_1min, i + 1)
        if gap < ATR_THRESHOLD * m1_atr:
            continue
        if gap * MNQ_DOLLAR > MAX_GAP_DOLLARS:
            continue
        h1_atr = vol_tracker.current_h1_atr
        if h1_atr > 0 and gap < MIN_GAP_ATR_RATIO * h1_atr:
            continue

        entry_price = snap_tick(c3.high + gap * ENTRY_PCT)
        sl_price    = snap_tick(entry_price + gap * SL_MULT)
        tp_price    = snap_tick(entry_price - gap * TP_MULT)
        pending     = True
        pending_bars = 0

        ts_et = cur_ts.astimezone(ET_TZ)
        pending_rec = {
            "entry_ts":            cur_ts.isoformat(),
            "exit_ts":             None,
            "label":               None,
            "exit_type":           None,
            "pnl_1x":              None,
            "entry_price":         entry_price,
            "sl":                  sl_price,
            "tp":                  tp_price,
            "gap_atr_ratio":       round(gap / h1_atr, 6) if h1_atr > 0 else 0.0,
            "gap_dollars":         round(gap * MNQ_DOLLAR, 4),
            "gap_m1_atr_ratio":    round(gap / m1_atr, 6) if m1_atr > 0 else 0.0,
            "h1_atr":              round(h1_atr, 4),
            "m1_atr":              round(m1_atr, 4),
            "h1_vol_pct":          round(vol_tracker.current_pct(), 4),
            "hour_et":             ts_et.hour,
            "dow_et":              ts_et.weekday(),
            "m1_bars_since_sweep": i - sweep_fire_m1_bar,
            "m1_bars_since_choch": i - choch_fire_m1_bar,
        }

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("S23: Meta-label feature generator — H1·M15·M1·g0.25")
    print("Running on PRE-CUTOFF data only (2025) — no holdout access")
    print(f"Loading {DATA_PATH} …")

    bars = load_bars(DATA_PATH)
    print(f"Loaded {len(bars):,} bars  "
          f"({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})")

    print("Running H1·M15·M1·g0.25 cascade with feature recording …")
    records = run_strategy_with_features(bars)

    executed = [r for r in records if r["label"] is not None]
    print(f"\nExecuted trades: {len(executed)}")
    if executed:
        wins  = sum(1 for r in executed if r["label"] == 1)
        gp    = sum(r["pnl_1x"] for r in executed if r["pnl_1x"] > 0)
        gl    = abs(sum(r["pnl_1x"] for r in executed if r["pnl_1x"] < 0))
        pf    = gp / gl if gl > 0 else float("inf")
        print(f"Win rate:        {wins/len(executed)*100:.1f}%")
        print(f"Profit factor:   {pf:.4f}")
        print(f"Net P&L (1x):    ${sum(r['pnl_1x'] for r in executed):,.2f}")
        etype_counts = {}
        for r in executed:
            etype_counts[r["exit_type"]] = etype_counts.get(r["exit_type"], 0) + 1
        print(f"Exit types:      {etype_counts}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLS)
        writer.writeheader()
        writer.writerows(executed)

    print(f"\nWrote {len(executed)} labeled trades → {OUTPUT_PATH}")
    print("Next step: run s24_ml_meta_filter.py")


if __name__ == "__main__":
    main()
