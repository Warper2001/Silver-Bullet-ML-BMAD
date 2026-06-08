"""study_vwap_reversion_rate.py — VWAPR Gate 0: VWAP-deviation mean reversion study.

Stage 0 DIAGNOSTIC — hypothesis generation only, NOT validation.

** DISCLOSED SEQUENCE (data-observation record) **
Phase 1 (bar-extreme stop): primary spec FAILED Gate 0 — 16.1% WR (need 55%).
Root cause: bar extreme + ATR buffer stop (~$14/contract) is too tight for a
2σ VWAP fade; momentum continuation knocks off the trade before reversal.

Phase 2 (σ-band stop): structural redesign. Stop at the NEXT σ band (e.g.,
VWAP + 3σ for a 2σ entry) is the canonical VWAP-fade stop — 3σ is the
institutional "tail event" threshold, not a fitted parameter. Stop_pts ≈ 1σ ≈
$20-$40/contract. Approved by user 2026-06-08 as a disclosed redesign.

GATE 0 criteria for σ-band stop spec (primary: k_entry=2.0, k_stop=3.0, VWAP,
all-day, no vol filter):
  - Tradeable reversion win rate  ≥ 55%
  - Median structural stop        ≤ $150/contract  (≤ 75 index pts)
  - Frequency                     ≥ 1.0 setups/day

In-sample: 2025-01-01 → 2026-02-28  (OOS holdout ≥ 2026-03-01 remains sealed)

Usage:
    .venv/bin/python study_vwap_reversion_rate.py
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.research.sorm_core import POINT_VALUE_USD, TICK_SIZE, load_bars_et
from src.research.strategy_core import calc_atr

# ── Constants ─────────────────────────────────────────────────────────────────

UTC = timezone.utc
ET = "America/New_York"

# In-sample: DO NOT touch holdout ≥ 2026-03-01
START_UTC = datetime(2025, 1, 1, tzinfo=UTC)
END_UTC   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

# VWAP configuration
SESSION_OPEN_ET   = time(9, 30)   # VWAP accumulation starts here
DETECT_START_ET   = time(9, 45)   # earliest extension detection (allow σ to build)
MIN_BARS_FOR_SIGMA = 15           # skip session if fewer bars have accumulated
DETECT_END_ET     = time(14, 30)  # last valid entry bar (leaves ≥90 min for reversion)
HARD_CLOSE_ET     = time(15, 55)  # walk bars until here

# Grid: k values to scan (in σ-bands)
K_VALUES = [1.5, 2.0, 2.5]

# Stop: extension bar extreme + ATR buffer (Phase 1 — bar-extreme architecture)
STOP_ATR_BUFFER = 0.25            # fraction of 20-bar M1 ATR added to bar extreme

# σ-band stop grid (Phase 2 — σ-band architecture)
# k_stop must be > k_entry for each pairing tested
SIGMA_STOP_GRID = [
    # (k_entry, k_stop)
    (1.5, 2.0),
    (1.5, 2.5),
    (2.0, 2.5),
    (2.0, 3.0),
    (2.0, 3.5),
    (2.5, 3.0),
    (2.5, 3.5),
]

# Sizing (for stop-$ reporting only; not used in win-rate calc)
STOP_CAP_USD   = 150.0            # skip-if-stop-too-wide threshold (reporting)
POINT_VALUE    = POINT_VALUE_USD  # $2/point for MNQ

# Volume-exhaustion detection: bar volume > VOL_EXHAUST_MULT × 20-bar trailing avg
VOL_EXHAUST_MULT = 1.5

# Gate 0 thresholds (pre-committed, evaluated on primary spec only)
GATE0_WIN_RATE_MIN = 0.55
GATE0_STOP_USD_MAX = 150.0
GATE0_FREQ_MIN     = 1.0


# ── VWAP + σ computation ─────────────────────────────────────────────────────

def compute_session_vwap_bands(sess_df: pd.DataFrame, k_values: list[float]) -> pd.DataFrame:
    """Compute running session-anchored VWAP and volume-weighted σ bands.

    No look-ahead: all values are cumulative sums from the start of the session.
    Formula:
        typ   = (H + L + C) / 3
        VWAP  = cumsum(typ × vol) / cumsum(vol)
        vw_var = cumsum(vol × (typ − VWAP)²) / cumsum(vol)
        σ     = sqrt(vw_var)
        bands  = VWAP ± k × σ
    """
    df = sess_df.copy()
    df["typ"] = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)

    cum_vol  = vol.cumsum()
    cum_tv   = (df["typ"] * vol).cumsum()
    df["vwap"]  = cum_tv / cum_vol.replace(0, np.nan)

    # Volume-weighted variance (accumulated)
    sq_dev     = vol * (df["typ"] - df["vwap"]) ** 2
    cum_sq     = sq_dev.cumsum()
    vw_var     = cum_sq / cum_vol.replace(0, np.nan)
    df["sigma"] = np.sqrt(vw_var.clip(lower=0.0))

    for k in k_values:
        df[f"upper_{k}"] = df["vwap"] + k * df["sigma"]
        df[f"lower_{k}"] = df["vwap"] - k * df["sigma"]

    return df


# ── Per-setup record ──────────────────────────────────────────────────────────

class SetupRecord:
    """One k-σ extension setup detected in a session."""

    __slots__ = (
        "date_et", "k", "direction",
        "entry", "stop", "stop_pts", "stop_usd", "skip_stop_cap",
        "vwap_at_entry", "sigma_at_entry",
        "win_vwap", "win_inner_band",
        "vol_exhaust", "time_bucket",
        "entry_ts",
    )

    def __init__(
        self,
        date_et,
        k: float,
        direction: str,        # "SHORT" | "LONG"
        entry: float,
        stop: float,
        stop_pts: float,
        stop_usd: float,
        skip_stop_cap: bool,
        vwap_at_entry: float,
        sigma_at_entry: float,
        vol_exhaust: bool,
        time_bucket: str,      # "early" | "midday" | "late"
        entry_ts,
    ):
        self.date_et       = date_et
        self.k             = k
        self.direction     = direction
        self.entry         = entry
        self.stop          = stop
        self.stop_pts      = stop_pts
        self.stop_usd      = stop_usd
        self.skip_stop_cap = skip_stop_cap
        self.vwap_at_entry = vwap_at_entry
        self.sigma_at_entry = sigma_at_entry
        self.vol_exhaust   = vol_exhaust
        self.time_bucket   = time_bucket
        self.entry_ts      = entry_ts
        # outcomes (filled by simulate_setup)
        self.win_vwap       = False
        self.win_inner_band = False


def _time_bucket(ts) -> str:
    """Classify bar timestamp into session time bucket (ET)."""
    t = ts.time()
    if t < time(11, 0):
        return "early"
    if t >= time(14, 0):
        return "late"
    return "midday"


def _detect_vol_exhaust(bars_before: pd.DataFrame) -> bool:
    """True if the last bar in bars_before has volume > VOL_EXHAUST_MULT × 20-bar avg."""
    if len(bars_before) < 21:
        return False
    recent = bars_before["volume"].astype(float).values
    avg_20 = recent[-21:-1].mean()
    if avg_20 == 0:
        return False
    return float(recent[-1]) > VOL_EXHAUST_MULT * avg_20


# ── Session processor ─────────────────────────────────────────────────────────

def process_session(
    date_et,
    sess_df: pd.DataFrame,
    k_values: list[float],
) -> list[SetupRecord]:
    """Detect k-σ extensions in a single RTH session and return all setups (one per k)."""
    setups: list[SetupRecord] = []

    # Slice RTH window (09:30 to 15:55 ET)
    rth_df = sess_df.between_time(
        SESSION_OPEN_ET.strftime("%H:%M"), HARD_CLOSE_ET.strftime("%H:%M"),
        inclusive="both",
    )
    if len(rth_df) < MIN_BARS_FOR_SIGMA + 5:
        return setups

    # Compute VWAP + σ bands across full RTH window
    vwap_df = compute_session_vwap_bands(rth_df, k_values)

    # ATR from bars preceding the session (for stop buffer)
    # Use all bars from the session up to each detection bar (computed per-bar below)

    # Detection window: from DETECT_START_ET to DETECT_END_ET
    detect_df = vwap_df.between_time(
        DETECT_START_ET.strftime("%H:%M"), DETECT_END_ET.strftime("%H:%M"),
        inclusive="both",
    )
    if detect_df.empty:
        return setups

    # Need cumulative bar count from session open to enforce MIN_BARS_FOR_SIGMA
    # map each bar in detect_df to its position in vwap_df
    session_positions = {ts: i for i, ts in enumerate(vwap_df.index)}

    for k in k_values:
        upper_col = f"upper_{k}"
        lower_col = f"lower_{k}"

        for ts, row in detect_df.iterrows():
            pos = session_positions.get(ts, -1)
            if pos < MIN_BARS_FOR_SIGMA:
                continue

            close = float(row["close"])
            bar_high = float(row["high"])
            bar_low  = float(row["low"])
            vwap_now = float(row["vwap"])
            sigma_now = float(row["sigma"])

            if sigma_now <= 0 or math.isnan(sigma_now):
                continue

            upper = float(row[upper_col])
            lower = float(row[lower_col])

            if math.isnan(upper) or math.isnan(lower):
                continue

            direction: Optional[str] = None
            if close > upper:
                direction = "SHORT"
            elif close < lower:
                direction = "LONG"

            if direction is None:
                continue

            # Compute ATR from bars UP TO (not including) this detection bar
            bars_before_pos = vwap_df.iloc[:pos]
            atr_val = calc_atr(bars_before_pos) if len(bars_before_pos) >= 20 else 10.0
            buffer_pts = STOP_ATR_BUFFER * atr_val

            if direction == "SHORT":
                # Stop above the extension bar's high
                stop = bar_high + buffer_pts
                stop_pts = stop - close   # stop > close since we enter at close
            else:  # LONG
                # Stop below the extension bar's low
                stop = bar_low - buffer_pts
                stop_pts = close - stop   # close > stop since we enter at close

            stop_pts = max(stop_pts, TICK_SIZE)   # floor at 1 tick
            stop_usd = stop_pts * POINT_VALUE

            skip_stop_cap = stop_usd > STOP_CAP_USD

            vol_exhaust = _detect_vol_exhaust(vwap_df.iloc[:pos + 1])
            tbucket = _time_bucket(ts)

            rec = SetupRecord(
                date_et=date_et,
                k=k,
                direction=direction,
                entry=close,
                stop=stop,
                stop_pts=stop_pts,
                stop_usd=stop_usd,
                skip_stop_cap=skip_stop_cap,
                vwap_at_entry=vwap_now,
                sigma_at_entry=sigma_now,
                vol_exhaust=vol_exhaust,
                time_bucket=tbucket,
                entry_ts=ts,
            )

            # ── Walk post-entry bars ───────────────────────────────────────────
            post_df = vwap_df.loc[vwap_df.index > ts]
            walk_df = post_df.between_time(
                "00:00", HARD_CLOSE_ET.strftime("%H:%M"), inclusive="left"
            )

            for w_ts, w_row in walk_df.iterrows():
                w_high  = float(w_row["high"])
                w_low   = float(w_row["low"])
                w_vwap  = float(w_row["vwap"])
                w_sigma = float(w_row["sigma"])

                if math.isnan(w_vwap) or math.isnan(w_sigma) or w_sigma <= 0:
                    continue

                inner_band_target = (
                    w_vwap + 1.0 * w_sigma if direction == "SHORT"
                    else w_vwap - 1.0 * w_sigma
                )

                if direction == "SHORT":
                    # Stop: bar high ≥ stop → STOPPED
                    if w_high >= rec.stop:
                        break  # stopped out — neither win flag set
                    # Win: bar low ≤ VWAP → reached mean
                    if w_low <= w_vwap:
                        rec.win_vwap = True
                        rec.win_inner_band = True  # inner is easier than VWAP; must also be true
                        break
                    # Inner band: bar low ≤ +1σ → partial reversion
                    if not rec.win_inner_band and w_low <= inner_band_target:
                        rec.win_inner_band = True
                        # do NOT break — continue to check VWAP
                else:  # LONG
                    if w_low <= rec.stop:
                        break
                    if w_high >= w_vwap:
                        rec.win_vwap = True
                        rec.win_inner_band = True
                        break
                    if not rec.win_inner_band and w_high >= inner_band_target:
                        rec.win_inner_band = True

            setups.append(rec)
            break  # ONE setup per k per session (first detection)

    return setups


# ── Phase 2: σ-band stop architecture ────────────────────────────────────────

class SigmaSetup:
    """One σ-band stop setup: entry at k_entry × σ, stop at k_stop × σ."""

    __slots__ = (
        "date_et", "k_entry", "k_stop", "direction",
        "entry", "stop", "stop_pts", "stop_usd", "skip_stop_cap",
        "vwap_at_entry", "sigma_at_entry",
        "win_vwap", "win_inner_band",
        "vol_exhaust", "time_bucket",
    )

    def __init__(self, date_et, k_entry, k_stop, direction, entry, stop, stop_pts,
                 stop_usd, skip_stop_cap, vwap_at_entry, sigma_at_entry,
                 vol_exhaust, time_bucket):
        self.date_et        = date_et
        self.k_entry        = k_entry
        self.k_stop         = k_stop
        self.direction      = direction
        self.entry          = entry
        self.stop           = stop
        self.stop_pts       = stop_pts
        self.stop_usd       = stop_usd
        self.skip_stop_cap  = skip_stop_cap
        self.vwap_at_entry  = vwap_at_entry
        self.sigma_at_entry = sigma_at_entry
        self.vol_exhaust    = vol_exhaust
        self.time_bucket    = time_bucket
        self.win_vwap       = False
        self.win_inner_band = False


def process_session_sigma_stop(
    date_et,
    sess_df: pd.DataFrame,
    sigma_stop_grid: list[tuple[float, float]],
) -> list[SigmaSetup]:
    """Detect σ-band extension setups and simulate with σ-band stop architecture.

    For each (k_entry, k_stop) pair, find first extension and walk bars.
    Stop = VWAP_at_entry + k_stop × σ_at_entry  (fixed at entry time, not updated).
    Target = running VWAP (updates each bar).
    Inner band target = VWAP + k_inner × σ  where k_inner is halfway between
    k_entry and zero (i.e., going back one σ from entry).
    """
    setups: list[SigmaSetup] = []

    # All unique k_entry values needed
    all_k_entry = sorted(set(p[0] for p in sigma_stop_grid))
    all_k = sorted(set(all_k_entry + [p[1] for p in sigma_stop_grid]))

    rth_df = sess_df.between_time(
        SESSION_OPEN_ET.strftime("%H:%M"), HARD_CLOSE_ET.strftime("%H:%M"),
        inclusive="both",
    )
    if len(rth_df) < MIN_BARS_FOR_SIGMA + 5:
        return setups

    vwap_df = compute_session_vwap_bands(rth_df, all_k)

    detect_df = vwap_df.between_time(
        DETECT_START_ET.strftime("%H:%M"), DETECT_END_ET.strftime("%H:%M"),
        inclusive="both",
    )
    if detect_df.empty:
        return setups

    session_positions = {ts: i for i, ts in enumerate(vwap_df.index)}

    # For each (k_entry, k_stop) pair, find first detection and walk
    for k_entry, k_stop in sigma_stop_grid:
        upper_entry_col = f"upper_{k_entry}"
        lower_entry_col = f"lower_{k_entry}"
        upper_stop_col  = f"upper_{k_stop}"
        lower_stop_col  = f"lower_{k_stop}"

        for ts, row in detect_df.iterrows():
            pos = session_positions.get(ts, -1)
            if pos < MIN_BARS_FOR_SIGMA:
                continue

            close = float(row["close"])
            vwap_now  = float(row["vwap"])
            sigma_now = float(row["sigma"])

            if sigma_now <= 0 or math.isnan(sigma_now):
                continue

            upper_entry = float(row[upper_entry_col]) if upper_entry_col in vwap_df.columns else float("nan")
            lower_entry = float(row[lower_entry_col]) if lower_entry_col in vwap_df.columns else float("nan")
            upper_stop  = float(row[upper_stop_col])  if upper_stop_col  in vwap_df.columns else float("nan")
            lower_stop  = float(row[lower_stop_col])  if lower_stop_col  in vwap_df.columns else float("nan")

            if any(math.isnan(v) for v in [upper_entry, lower_entry, upper_stop, lower_stop]):
                continue

            direction: Optional[str] = None
            stop_price: float = 0.0
            if close > upper_entry:
                direction = "SHORT"
                stop_price = upper_stop   # above entry, so stop > entry
            elif close < lower_entry:
                direction = "LONG"
                stop_price = lower_stop   # below entry, so stop < entry

            if direction is None:
                continue

            if direction == "SHORT":
                stop_pts = stop_price - close
            else:
                stop_pts = close - stop_price

            stop_pts = max(stop_pts, TICK_SIZE)
            stop_usd = stop_pts * POINT_VALUE

            skip_stop_cap = stop_usd > STOP_CAP_USD

            vol_exhaust = _detect_vol_exhaust(vwap_df.iloc[:pos + 1])
            tbucket = _time_bucket(ts)

            setup = SigmaSetup(
                date_et=date_et,
                k_entry=k_entry,
                k_stop=k_stop,
                direction=direction,
                entry=close,
                stop=stop_price,
                stop_pts=stop_pts,
                stop_usd=stop_usd,
                skip_stop_cap=skip_stop_cap,
                vwap_at_entry=vwap_now,
                sigma_at_entry=sigma_now,
                vol_exhaust=vol_exhaust,
                time_bucket=tbucket,
            )

            # Walk post-entry bars
            post_df = vwap_df.loc[vwap_df.index > ts]
            walk_df = post_df.between_time(
                "00:00", HARD_CLOSE_ET.strftime("%H:%M"), inclusive="left"
            )

            for w_ts, w_row in walk_df.iterrows():
                w_high  = float(w_row["high"])
                w_low   = float(w_row["low"])
                w_vwap  = float(w_row["vwap"])
                w_sigma = float(w_row["sigma"])

                if math.isnan(w_vwap) or math.isnan(w_sigma) or w_sigma <= 0:
                    continue

                # Inner band target: k_entry - 1σ step toward VWAP
                inner_k = max(k_entry - 1.0, 0.0)
                if direction == "SHORT":
                    inner_target = w_vwap + inner_k * w_sigma
                else:
                    inner_target = w_vwap - inner_k * w_sigma

                if direction == "SHORT":
                    if w_high >= setup.stop:
                        break
                    if w_low <= w_vwap:
                        setup.win_vwap = True
                        setup.win_inner_band = True
                        break
                    if not setup.win_inner_band and inner_k > 0 and w_low <= inner_target:
                        setup.win_inner_band = True
                else:
                    if w_low <= setup.stop:
                        break
                    if w_high >= w_vwap:
                        setup.win_vwap = True
                        setup.win_inner_band = True
                        break
                    if not setup.win_inner_band and inner_k > 0 and w_high >= inner_target:
                        setup.win_inner_band = True

            setups.append(setup)
            break  # one setup per (k_entry, k_stop) per session

    return setups


# ── Aggregation + reporting ───────────────────────────────────────────────────

def _cell_stats(records: list[SetupRecord], skip_stop_cap: bool = True) -> dict:
    """Aggregate tradeable reversion stats for a slice of records."""
    if skip_stop_cap:
        tradeable = [r for r in records if not r.skip_stop_cap]
    else:
        tradeable = records

    n = len(tradeable)
    if n == 0:
        return {"n": 0, "wr_vwap": float("nan"), "wr_inner": float("nan"),
                "stop_med": float("nan"), "stop_p75": float("nan")}

    wins_vwap  = sum(1 for r in tradeable if r.win_vwap)
    wins_inner = sum(1 for r in tradeable if r.win_inner_band)
    stops_usd  = sorted(r.stop_usd for r in tradeable)

    return {
        "n": n,
        "wr_vwap":  wins_vwap / n,
        "wr_inner": wins_inner / n,
        "stop_med": float(np.median(stops_usd)),
        "stop_p75": float(np.percentile(stops_usd, 75)),
    }


def main() -> None:
    print("=" * 72)
    print("VWAPR Gate 0: VWAP-Deviation Reversion Rate + Stop Geometry Study")
    print(f"In-sample: {START_UTC.date()} → {END_UTC.date()}")
    print(f"Primary spec: ±2.0σ, target=VWAP, all-day RTH, no vol/time filter")
    print("=" * 72)

    # ── Load bars ─────────────────────────────────────────────────────────────
    print("\nLoading 1-min bars…", end=" ", flush=True)
    df = load_bars_et([CSV_2025, CSV_2026], START_UTC, END_UTC)
    if df.empty:
        print("ERROR: no bars loaded — check CSV paths")
        sys.exit(1)
    print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")

    # ── Session iteration ─────────────────────────────────────────────────────
    df["_date"] = df.index.date
    sessions = list(df.groupby("_date"))
    total_sessions = 0
    trading_days   = 0

    all_setups: list[SetupRecord] = []

    print(f"\nScanning {len(sessions)} calendar days…", end=" ", flush=True)
    for date_et, sess_df in sessions:
        sess_df = sess_df.drop(columns=["_date"])
        total_sessions += 1

        # Skip weekends and very sparse days (holidays etc.)
        if date_et.weekday() >= 5:
            continue
        rth_check = sess_df.between_time("09:30", "15:55", inclusive="both")
        if len(rth_check) < 30:
            continue
        trading_days += 1

        setups = process_session(date_et, sess_df, K_VALUES)
        all_setups.extend(setups)

    print(f"done.\n  {trading_days} trading days, {len(all_setups)} raw setups across all k-values.")

    if not all_setups:
        print("\nNo setups found. Check detection window or data.")
        sys.exit(1)

    # ── Grid report ───────────────────────────────────────────────────────────
    print()
    print("─" * 72)
    print("GRID: k × target  (tradeable = stop ≤ $150/contract; win = target before stop)")
    print("─" * 72)
    print(f"{'k':>6}  {'target':>10}  {'N':>5}  {'WR':>6}  {'stop_med $':>10}  {'stop_p75 $':>10}  {'freq/day':>9}")
    print(f"{'─'*6}  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*9}")

    for k in K_VALUES:
        recs_k = [r for r in all_setups if r.k == k]
        stats_vwap  = _cell_stats(recs_k, skip_stop_cap=True)
        stats_inner = _cell_stats(recs_k, skip_stop_cap=True)

        # Override inner stats with inner-band win flag
        if stats_inner["n"] > 0:
            tradeable = [r for r in recs_k if not r.skip_stop_cap]
            wins_inner = sum(1 for r in tradeable if r.win_inner_band)
            stats_inner["wr_inner"] = wins_inner / stats_inner["n"] if stats_inner["n"] > 0 else float("nan")

        # Use total recs_k for denominator of "setups per session" (raw freq)
        # but show only tradeable N in WR columns
        n_raw = len(recs_k)
        freq  = n_raw / trading_days if trading_days > 0 else 0.0

        wr_v  = stats_vwap["wr_vwap"]
        wr_i  = stats_inner["wr_inner"]
        s_med = stats_vwap["stop_med"]
        s_p75 = stats_vwap["stop_p75"]
        n_t   = stats_vwap["n"]

        def _pct(v):
            return f"{v*100:.1f}%" if not math.isnan(v) else "  n/a"

        def _usd(v):
            return f"${v:7.1f}" if not math.isnan(v) else "      n/a"

        print(f"  k={k:.1f}  {'VWAP':>10}  {n_t:>5}  {_pct(wr_v):>6}  {_usd(s_med):>10}  {_usd(s_p75):>10}  {freq:>7.2f}/d")
        # Inner band row
        tradeable_k = [r for r in recs_k if not r.skip_stop_cap]
        wins_inner_k = sum(1 for r in tradeable_k if r.win_inner_band)
        wr_inner_k = wins_inner_k / len(tradeable_k) if tradeable_k else float("nan")
        print(f"  k={k:.1f}  {'±1σ band':>10}  {n_t:>5}  {_pct(wr_inner_k):>6}  {_usd(s_med):>10}  {_usd(s_p75):>10}  {freq:>7.2f}/d")

    # ── Sensitivity: vol-exhaustion filter ────────────────────────────────────
    print()
    print("─" * 72)
    print("SENSITIVITY: k=2.0, target=VWAP — disclosed filters (NOT the primary spec)")
    print("─" * 72)

    recs_2 = [r for r in all_setups if r.k == 2.0]
    tradeable_2 = [r for r in recs_2 if not r.skip_stop_cap]

    for label, subset in [
        ("all-day, no vol filter  [PRIMARY]", tradeable_2),
        ("vol-exhaust only",  [r for r in tradeable_2 if r.vol_exhaust]),
        ("no vol-exhaust",    [r for r in tradeable_2 if not r.vol_exhaust]),
        ("early (09:45-11:00)",  [r for r in tradeable_2 if r.time_bucket == "early"]),
        ("midday (11:00-14:00)", [r for r in tradeable_2 if r.time_bucket == "midday"]),
        ("late (14:00-14:30)",   [r for r in tradeable_2 if r.time_bucket == "late"]),
    ]:
        # Frequency denominator: raw setups with that filter (incl. stop-cap-skipped)
        raw_n = len([r for r in recs_2 if (
            (label == "all-day, no vol filter  [PRIMARY]") or
            (label.startswith("vol-exhaust only") and r.vol_exhaust) or
            (label.startswith("no vol-exhaust") and not r.vol_exhaust) or
            (label.startswith("early") and r.time_bucket == "early") or
            (label.startswith("midday") and r.time_bucket == "midday") or
            (label.startswith("late") and r.time_bucket == "late")
        )])
        freq_s = raw_n / trading_days if trading_days > 0 else 0.0
        n_t2 = len(subset)
        if n_t2 == 0:
            print(f"  {label:<36}  N=     0  WR=   n/a  stop_med=      n/a  freq={freq_s:5.2f}/d")
            continue
        wins_v = sum(1 for r in subset if r.win_vwap)
        wr_v   = wins_v / n_t2
        stops_usd = [r.stop_usd for r in subset]
        s_med = np.median(stops_usd)
        s_p75 = np.percentile(stops_usd, 75)
        # Gate markers
        wr_flag   = " ✅" if wr_v >= GATE0_WIN_RATE_MIN else " ❌"
        stop_flag = " ✅" if s_med <= GATE0_STOP_USD_MAX else " ❌"
        freq_flag = " ✅" if freq_s >= GATE0_FREQ_MIN else " ❌"
        print(
            f"  {label:<36}  N={n_t2:>5}  WR={wr_v*100:5.1f}%{wr_flag}  "
            f"stop_med=${s_med:6.1f}{stop_flag}  freq={freq_s:5.2f}/d{freq_flag}"
        )

    # ── By-month breakdown (primary spec) ────────────────────────────────────
    print()
    print("─" * 72)
    print("BY-MONTH (primary spec: k=2.0, VWAP target, tradeable only)")
    print("─" * 72)
    print(f"  {'Month':<10}  {'N':>4}  {'WR':>6}  {'stop_med':>9}  {'Freq/d':>7}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*7}")

    monthly_days: dict[str, int] = defaultdict(int)
    monthly_setups: dict[str, list] = defaultdict(list)

    for date_et, _ in sessions:
        if date_et.weekday() >= 5:
            continue
        month_key = f"{date_et.year}-{date_et.month:02d}"
        # Count only days with enough RTH bars (approx: count all weekdays)
        monthly_days[month_key] += 1

    for r in tradeable_2:
        month_key = f"{r.date_et.year}-{r.date_et.month:02d}"
        monthly_setups[month_key].append(r)

    for mk in sorted(monthly_days.keys()):
        days   = monthly_days[mk]
        recs   = monthly_setups.get(mk, [])
        n      = len(recs)
        if n == 0:
            print(f"  {mk:<10}  {n:>4}   n/a    n/a       {0.0:>6.2f}/d")
            continue
        wins   = sum(1 for r in recs if r.win_vwap)
        wr     = wins / n
        stops  = [r.stop_usd for r in recs]
        s_med  = np.median(stops)
        freq   = n / days if days > 0 else 0.0
        print(f"  {mk:<10}  {n:>4}  {wr*100:5.1f}%  ${s_med:7.1f}  {freq:6.2f}/d")

    # ── Gate 0 verdict ────────────────────────────────────────────────────────
    # Primary spec: k=2.0, VWAP, all-day, no vol filter, tradeable (stop ≤ $150)
    primary = tradeable_2
    n_pri  = len(primary)
    freq_pri = len([r for r in recs_2]) / trading_days if trading_days > 0 else 0.0

    wr_pri    = sum(1 for r in primary if r.win_vwap) / n_pri if n_pri > 0 else 0.0
    stops_pri = [r.stop_usd for r in primary]
    stop_med_pri = float(np.median(stops_pri)) if stops_pri else float("nan")

    print()
    print("=" * 72)
    print("GATE 0 VERDICT  (primary spec: k=2.0 σ, target=VWAP, all-day, no filter)")
    print("=" * 72)

    def _verdict(label: str, value, threshold, pass_fn) -> str:
        p = pass_fn(value, threshold)
        flag = "✅ PASS" if p else "❌ FAIL"
        return f"  {flag}  {label}"

    wr_pass   = wr_pri    >= GATE0_WIN_RATE_MIN
    stop_pass = stop_med_pri <= GATE0_STOP_USD_MAX
    freq_pass = freq_pri  >= GATE0_FREQ_MIN

    print(f"  {'✅ PASS' if wr_pass else '❌ FAIL'}  Win rate ≥ {GATE0_WIN_RATE_MIN*100:.0f}%"
          f"         [measured: {wr_pri*100:.1f}%  (N={n_pri})]")
    print(f"  {'✅ PASS' if stop_pass else '❌ FAIL'}  Median stop ≤ ${GATE0_STOP_USD_MAX:.0f}/contract"
          f"  [measured: ${stop_med_pri:.1f}]")
    print(f"  {'✅ PASS' if freq_pass else '❌ FAIL'}  Frequency ≥ {GATE0_FREQ_MIN:.1f} setups/day"
          f"  [measured: {freq_pri:.2f}/day  (raw setups {len(recs_2)}, {trading_days} days)]")

    all_pass = wr_pass and stop_pass and freq_pass
    print()
    if all_pass:
        print("  ✅ GATE 0 PASS — all primary criteria met.")
        print("     → Proceed to Stage 1: write vwapr_core.py, vwapr_config.yaml,")
        print("       prereg_vwapr_seal.py, commit pre-registration, then backtest.")
    else:
        failing = []
        if not wr_pass:
            failing.append(f"Win rate {wr_pri*100:.1f}% < {GATE0_WIN_RATE_MIN*100:.0f}%")
        if not stop_pass:
            failing.append(f"Median stop ${stop_med_pri:.1f} > ${GATE0_STOP_USD_MAX:.0f}")
        if not freq_pass:
            failing.append(f"Frequency {freq_pri:.2f}/day < {GATE0_FREQ_MIN:.1f}")
        print(f"  ❌ GATE 0 FAIL — failing criteria:")
        for f in failing:
            print(f"     • {f}")
        print()
        print("  NOTE: Check sensitivity table above for research-justified filters.")
        print("  A filter is acceptable ONLY if it has a structural justification")
        print("  (timing/vol-exhaustion, not fitted to max PF). If no filter saves")
        print("  the primary spec → VWAP is not viable; do not pre-register.")
    print("=" * 72)

    # ── Skip-rate summary ─────────────────────────────────────────────────────
    skipped_2 = [r for r in recs_2 if r.skip_stop_cap]
    print()
    print(f"  k=2.0 raw setups: {len(recs_2)}  |  "
          f"tradeable (stop ≤ ${STOP_CAP_USD:.0f}): {len(tradeable_2)}  |  "
          f"skipped (stop > cap): {len(skipped_2)}  ({len(skipped_2)/max(len(recs_2),1)*100:.0f}%)")

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2: σ-BAND STOP ARCHITECTURE
    # (disclosed redesign: Phase 1 bar-extreme FAILED — 16.1% WR)
    # Stop = VWAP_at_entry + k_stop × σ_at_entry  (σ-native stop)
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 72)
    print("PHASE 2: σ-BAND STOP ARCHITECTURE")
    print("(Phase 1 bar-extreme stop FAILED: 16.1% WR; disclosed redesign)")
    print("Stop = VWAP + k_stop × σ at entry time  (σ-native institutional stop)")
    print("=" * 72)

    all_sigma_setups: list[SigmaSetup] = []

    print(f"\nRe-scanning {len(sessions)} days with σ-band stop grid…", end=" ", flush=True)
    for date_et, sess_df in sessions:
        sess_df = sess_df.drop(columns=["_date"], errors="ignore")
        if date_et.weekday() >= 5:
            continue
        rth_check = sess_df.between_time("09:30", "15:55", inclusive="both")
        if len(rth_check) < 30:
            continue
        s_setups = process_session_sigma_stop(date_et, sess_df, SIGMA_STOP_GRID)
        all_sigma_setups.extend(s_setups)

    print(f"done.  {len(all_sigma_setups)} total σ-band setups across all (k_entry, k_stop) pairs.")

    if not all_sigma_setups:
        print("No σ-band setups found.")
    else:
        # ── σ-band grid report ────────────────────────────────────────────────
        print()
        print("─" * 72)
        print("σ-BAND STOP GRID  (tradeable = stop ≤ $150; win = VWAP before stop)")
        print("─" * 72)
        print(f"  {'entry_k':>7}  {'stop_k':>6}  {'N':>5}  {'WR_vwap':>8}  {'stop_med $':>10}  {'stop_p75 $':>10}  {'freq/d':>7}")
        print(f"  {'─'*7}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*7}")

        for k_e, k_s in SIGMA_STOP_GRID:
            cell = [r for r in all_sigma_setups if r.k_entry == k_e and r.k_stop == k_s]
            tradeable_c = [r for r in cell if not r.skip_stop_cap]
            n_raw_c = len(cell)
            n_t_c   = len(tradeable_c)
            freq_c  = n_raw_c / trading_days if trading_days > 0 else 0.0

            if n_t_c == 0:
                print(f"  {k_e:>7.1f}  {k_s:>6.1f}  {0:>5}     n/a        n/a         n/a      {freq_c:6.2f}/d")
                continue

            wins_v = sum(1 for r in tradeable_c if r.win_vwap)
            wr_v   = wins_v / n_t_c
            stops_c = [r.stop_usd for r in tradeable_c]
            s_med_c = np.median(stops_c)
            s_p75_c = np.percentile(stops_c, 75)

            # Gate markers
            wr_flag   = "✅" if wr_v >= GATE0_WIN_RATE_MIN else "❌"
            stop_flag = "✅" if s_med_c <= GATE0_STOP_USD_MAX else "❌"
            freq_flag = "✅" if freq_c >= GATE0_FREQ_MIN else "❌"

            print(
                f"  {k_e:>7.1f}  {k_s:>6.1f}  {n_t_c:>5}  "
                f"{wr_v*100:6.1f}% {wr_flag}  "
                f"${s_med_c:7.1f} {stop_flag}  "
                f"${s_p75_c:7.1f}  "
                f"{freq_c:6.2f}/d {freq_flag}"
            )

        # ── Primary σ-band spec verdict (k_entry=2.0, k_stop=3.0) ─────────────
        primary_pair = (2.0, 3.0)
        primary_cell = [r for r in all_sigma_setups
                        if r.k_entry == primary_pair[0] and r.k_stop == primary_pair[1]]
        primary_tradeable = [r for r in primary_cell if not r.skip_stop_cap]

        n_p2  = len(primary_tradeable)
        freq_p2 = len(primary_cell) / trading_days if trading_days > 0 else 0.0
        wr_p2   = sum(1 for r in primary_tradeable if r.win_vwap) / n_p2 if n_p2 > 0 else 0.0
        stops_p2 = [r.stop_usd for r in primary_tradeable]
        stop_med_p2 = float(np.median(stops_p2)) if stops_p2 else float("nan")

        wr_pass2   = wr_p2       >= GATE0_WIN_RATE_MIN
        stop_pass2 = stop_med_p2 <= GATE0_STOP_USD_MAX
        freq_pass2 = freq_p2     >= GATE0_FREQ_MIN

        print()
        print("─" * 72)
        print("GATE 0 VERDICT — σ-band primary spec: k_entry=2.0, k_stop=3.0, VWAP, all-day")
        print("─" * 72)
        print(f"  {'✅ PASS' if wr_pass2 else '❌ FAIL'}  Win rate ≥ {GATE0_WIN_RATE_MIN*100:.0f}%"
              f"         [measured: {wr_p2*100:.1f}%  (N={n_p2})]")
        print(f"  {'✅ PASS' if stop_pass2 else '❌ FAIL'}  Median stop ≤ ${GATE0_STOP_USD_MAX:.0f}/contract"
              f"  [measured: ${stop_med_p2:.1f}]")
        print(f"  {'✅ PASS' if freq_pass2 else '❌ FAIL'}  Frequency ≥ {GATE0_FREQ_MIN:.1f} setups/day"
              f"  [measured: {freq_p2:.2f}/day  ({len(primary_cell)} raw setups, {trading_days} days)]")

        all_pass2 = wr_pass2 and stop_pass2 and freq_pass2
        print()
        if all_pass2:
            print("  ✅ GATE 0 PASS (σ-band stop redesign)")
            print("     → Proceed to Stage 1: write vwapr_core.py, vwapr_config.yaml,")
            print("       prereg_vwapr_seal.py, commit pre-registration, then backtest.")
            print("     → Pre-reg MUST disclose: Phase 1 bar-extreme FAILED (16.1% WR);")
            print("       σ-band stop chosen as structural redesign on 2026-06-08.")
        else:
            failing2 = []
            if not wr_pass2:
                failing2.append(f"Win rate {wr_p2*100:.1f}% < {GATE0_WIN_RATE_MIN*100:.0f}%")
            if not stop_pass2:
                failing2.append(f"Median stop ${stop_med_p2:.1f} > ${GATE0_STOP_USD_MAX:.0f}")
            if not freq_pass2:
                failing2.append(f"Frequency {freq_p2:.2f}/day < {GATE0_FREQ_MIN:.1f}")
            print("  ❌ GATE 0 FAIL (σ-band redesign also failed)")
            for f2 in failing2:
                print(f"     • {f2}")
            print()
            print("  VWAP mean reversion is NOT viable at Topstep combine limits.")
            print("  Do not pre-register. Update memory and stop.")

        # ── By-month for σ-band primary ────────────────────────────────────────
        if n_p2 > 0:
            print()
            print("─" * 72)
            print("BY-MONTH  (σ-band primary: k_entry=2.0, k_stop=3.0, VWAP, tradeable)")
            print("─" * 72)
            print(f"  {'Month':<10}  {'N':>4}  {'WR':>6}  {'stop_med':>9}  {'Freq/d':>7}")

            monthly_sigma: dict[str, list] = defaultdict(list)
            for r in primary_tradeable:
                mk = f"{r.date_et.year}-{r.date_et.month:02d}"
                monthly_sigma[mk].append(r)

            for mk in sorted(monthly_days.keys()):
                days  = monthly_days[mk]
                recs_m = monthly_sigma.get(mk, [])
                n_m    = len(recs_m)
                if n_m == 0:
                    print(f"  {mk:<10}  {n_m:>4}   n/a    n/a       {0.0:>6.2f}/d")
                    continue
                wins_m = sum(1 for r in recs_m if r.win_vwap)
                wr_m   = wins_m / n_m
                s_m    = np.median([r.stop_usd for r in recs_m])
                fr_m   = n_m / days if days > 0 else 0.0
                print(f"  {mk:<10}  {n_m:>4}  {wr_m*100:5.1f}%  ${s_m:7.1f}  {fr_m:6.2f}/d")

    print()
    print("=" * 72)
    print("END OF STUDY")
    print("=" * 72)


if __name__ == "__main__":
    main()
