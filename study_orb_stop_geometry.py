"""study_orb_stop_geometry.py — ORBM-3 Phase A: Stop-Geometry Diagnostic.

Answers the load-bearing question before pre-registering ORBM-3:
  What stop architecture (boundary-buffered vs. pure-ATR) at what k-multiple
  survives the ORB retest noise band while keeping stops ≤ $150/contract?

Two measurements:

  1. MAE Distribution (structural baseline):
     Among breakouts that reach the 1.5R geometric continuation target
     (entry + 1.5 × 0.25×ORB_size) before the ORB boundary is touched again,
     how far do they retrace toward the boundary? The 75th-percentile MAE
     (in ATR multiples) is the *minimum structurally-justified k* — placing
     the stop further than the 75th-pct MAE lets 75% of winning trades breathe.

  2. Grid Scan (two architectures × two entries × four k values = 16 cells):
     Arch A (boundary-buffered): stop = broken_boundary ∓ k × ATR
     Arch B (pure-ATR from entry): stop = entry ∓ k × ATR
     Entry 1 (chase):    enter at extension-bar close
     Entry 2 (pullback): enter at first retest of broken boundary (±tolerance)
     k ∈ {0.25, 0.5, 0.75, 1.0}
     Per cell: median stop $/contract, skip-rate (>$150), NSR (stop before TP),
     implied WR & PF, and frequency (traded setups/ORB-session-day).

Gate A verdict: a cell passes if, on in-sample data:
  • stop ≤ $150/contract on ≥ 60% of setups
  • NSR ≤ 35%  (implied WR ≥ 65%)
  • frequency ≥ 1.0 setup/day
  • k chosen ≥ 75th-pct MAE in ATR multiples (structural, not PF-optimized)

Extension threshold: 0.25 × ORB_size (ORBM-2 baseline; frequency already measured).
In-sample: 2025-01-01 → 2026-02-28 UTC.

Usage:
    .venv/bin/python study_orb_stop_geometry.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.research.sorm_core import (
    POINT_VALUE_USD,
    TICK_SIZE,
    Direction,
    ExtensionEvent,
    OpeningRange,
    SORMConfig,
    build_opening_range,
    detect_extension,
    load_bars_et,
)
from src.research.strategy_core import calc_atr

# ── Constants ─────────────────────────────────────────────────────────────────

UTC = timezone.utc
IN_SAMPLE_START = datetime(2025, 1, 1, tzinfo=UTC)
IN_SAMPLE_END   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

# Extension / timing config (reuse sorm_core's duck-typing)
CFG = SORMConfig(
    extension_threshold=0.25,   # ORBM-2 baseline
    orb_min_size_points=5.0,    # same as ORBM-2
)
HARD_CLOSE_STR = CFG.hard_close_et.strftime("%H:%M")

# Grid parameters
K_VALUES:       list[float] = [0.25, 0.50, 0.75, 1.00]
STOP_CAP_USD:   float = 150.0    # skip if stop > this
STOP_CAP_FRAC:  float = 0.60     # gate: ≥60% of setups must pass cap
TP_MULTIPLE:    float = 1.50     # 1.5R target

# Pullback-entry detection
PULLBACK_TOL_FRAC:  float = 0.10   # retest within 0.10 × ORB_size of boundary
PULLBACK_TIMEOUT:   int   = 20     # give up after this many bars

# Gate A thresholds
GATE_NSR_MAX:       float = 0.35   # NSR ≤ 35% (WR ≥ 65%)
GATE_FREQ_MIN:      float = 1.00   # ≥ 1.0 setups/ORB-session-day


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class SessionRecord:
    date_et: date
    orb: OpeningRange
    ext: ExtensionEvent
    atr: float                          # 20-bar ATR at extension time (M1)
    post_ext: pd.DataFrame              # bars after extension detection bar
    pullback_price: Optional[float]     # close of pullback bar (None if timeout)
    pullback_ts: Optional[datetime]


@dataclass
class CellResult:
    arch: str    # "A" or "B"
    entry: int   # 1 or 2
    k: float
    stop_usd_all: list[float] = field(default_factory=list)  # incl. skipped
    n_traded: int = 0
    n_tp:     int = 0
    n_sl:     int = 0
    n_ts:     int = 0    # time-stop (11:30 hard close)

    @property
    def n_total_ext(self) -> int:
        return len(self.stop_usd_all)

    @property
    def skip_rate(self) -> float:
        if not self.stop_usd_all:
            return 0.0
        return sum(1 for s in self.stop_usd_all if s > STOP_CAP_USD) / len(self.stop_usd_all)

    @property
    def frac_under_cap(self) -> float:
        return 1.0 - self.skip_rate

    @property
    def nsr(self) -> float:
        return self.n_sl / self.n_traded if self.n_traded > 0 else 1.0

    @property
    def win_rate(self) -> float:
        return self.n_tp / self.n_traded if self.n_traded > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        """Theoretical PF assuming TP at 1.5R and SL at 1R (time-stops excluded)."""
        n_decisive = self.n_tp + self.n_sl
        if n_decisive == 0:
            return 0.0
        gross_profit = self.n_tp * TP_MULTIPLE   # in R units
        gross_loss   = self.n_sl * 1.0
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def median_stop_usd(self) -> float:
        traded_stops = [s for s in self.stop_usd_all if s <= STOP_CAP_USD]
        return float(np.median(traded_stops)) if traded_stops else 0.0


# ── Session data collection ───────────────────────────────────────────────────

def _detect_pullback(
    post_ext: pd.DataFrame,
    orb: OpeningRange,
    ext: ExtensionEvent,
) -> tuple[Optional[float], Optional[datetime]]:
    """Find first post-extension bar that retests the broken ORB boundary.

    Retest = close within PULLBACK_TOL_FRAC × ORB_size of the broken boundary.
    """
    tol = PULLBACK_TOL_FRAC * orb.size
    bars = post_ext.between_time("00:00", HARD_CLOSE_STR, inclusive="left")

    for i, (ts, row) in enumerate(bars.iterrows()):
        if i >= PULLBACK_TIMEOUT:
            break
        close = float(row["close"])
        if ext.direction == Direction.BEARISH:  # upward ext → LONG; retest ORB_high
            if close <= orb.high + tol:
                return close, ts
        else:                                    # downward ext → SHORT; retest ORB_low
            if close >= orb.low - tol:
                return close, ts
    return None, None


def collect_sessions(df: pd.DataFrame) -> tuple[list[SessionRecord], int, int]:
    """Walk all in-sample session days and collect structured records.

    Returns:
        (records, n_sessions, n_orb_sessions)
    """
    records: list[SessionRecord] = []
    n_sessions = 0
    n_orb = 0

    df["_date"] = df.index.date
    for date_et, sess_df in df.groupby("_date"):
        if date_et.weekday() >= 5:
            continue
        sess_df = sess_df.drop(columns=["_date"])
        n_sessions += 1

        orb = build_opening_range(sess_df, CFG)
        if orb is None:
            continue
        n_orb += 1

        ext = detect_extension(sess_df, orb, CFG)
        if ext is None:
            continue

        # ATR: use M1 bars up to and including extension detection bar
        bars_to_ext = sess_df[sess_df.index <= ext.detection_bar_ts]
        atr = calc_atr(bars_to_ext)

        # Post-extension bars (within trading hours up to 11:30)
        post_ext = sess_df.loc[sess_df.index > ext.detection_bar_ts]
        post_ext = post_ext.between_time("00:00", HARD_CLOSE_STR, inclusive="left")

        # Pullback entry detection
        pb_price, pb_ts = _detect_pullback(post_ext, orb, ext)

        records.append(SessionRecord(
            date_et=date_et,
            orb=orb,
            ext=ext,
            atr=atr,
            post_ext=post_ext,
            pullback_price=pb_price,
            pullback_ts=pb_ts,
        ))

    return records, n_sessions, n_orb


# ── MAE distribution study ────────────────────────────────────────────────────

def measure_mae_distribution(records: list[SessionRecord]) -> dict:
    """Measure worst-pullback (MAE) among breakouts that reach the 1.5R target.

    Reference entry:   extension_close (Entry 1 / chase)
    Reference TP:      entry + 1.5 × (0.25 × ORB_size)   [geometric, arch-independent]
    Disqualifier:      price touches broken ORB boundary before TP (trade "fails")

    MAE is the worst adverse excursion from entry toward the boundary,
    for trades that ultimately win this structural contest.

    Returns dict with raw and ATR-relative MAE percentile distributions.
    """
    ref_tp_frac = TP_MULTIPLE * 0.25   # = 0.375× ORB_size beyond boundary

    mae_pts: list[float] = []
    mae_atr: list[float] = []
    n_wins = 0
    n_losses = 0
    n_timout = 0

    for rec in records:
        orb = rec.orb
        ext = rec.ext
        entry = ext.extension_close

        if ext.direction == Direction.BEARISH:  # LONG
            tp_level   = entry + ref_tp_frac * orb.size
            boundary   = orb.high
            disqualify = lambda lo, _: lo < boundary       # touched back into range
            favorable  = lambda _, hi: hi >= tp_level
            mae_fn     = lambda lo, _: max(0.0, entry - lo)
        else:                                   # SHORT
            tp_level   = entry - ref_tp_frac * orb.size
            boundary   = orb.low
            disqualify = lambda _, hi: hi > boundary
            favorable  = lambda lo, _: lo <= tp_level
            mae_fn     = lambda _, hi: max(0.0, hi - entry)

        running_mae = 0.0
        outcome = "timeout"

        for _, row in rec.post_ext.iterrows():
            bar_hi = float(row["high"])
            bar_lo = float(row["low"])

            # Update running MAE (worst pullback toward boundary so far)
            running_mae = max(running_mae, mae_fn(bar_lo, bar_hi))

            # Check disqualifier first (touched back to boundary → loss)
            if disqualify(bar_lo, bar_hi):
                outcome = "loss"
                break
            # Check target
            if favorable(bar_lo, bar_hi):
                outcome = "win"
                break

        if outcome == "win":
            n_wins += 1
            mae_pts.append(running_mae)
            if rec.atr > 0:
                mae_atr.append(running_mae / rec.atr)
        elif outcome == "loss":
            n_losses += 1
        else:
            n_timout += 1

    result = {
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_timeout": n_timout,
        "total": len(records),
        "win_rate": n_wins / len(records) if records else 0.0,
    }
    if mae_pts:
        pcts = [25, 50, 75, 90, 95]
        result["mae_pts"] = {p: float(np.percentile(mae_pts, p)) for p in pcts}
        result["mae_atr"] = {p: float(np.percentile(mae_atr, p)) for p in pcts} if mae_atr else {}
        result["mae_pts_raw"] = mae_pts
        result["mae_atr_raw"] = mae_atr
    return result


# ── Grid simulation ───────────────────────────────────────────────────────────

def _compute_stop(
    arch: str,
    entry: float,
    boundary: float,
    k: float,
    atr: float,
    direction: Direction,
) -> float:
    """Compute stop level for the given architecture."""
    if arch == "A":
        # Boundary-buffered: k×ATR below (LONG) or above (SHORT) the ORB boundary
        if direction == Direction.BEARISH:  # LONG
            return boundary - k * atr
        else:                               # SHORT
            return boundary + k * atr
    else:
        # Pure-ATR from entry: k×ATR adverse from entry
        if direction == Direction.BEARISH:  # LONG
            return entry - k * atr
        else:                               # SHORT
            return entry + k * atr


def simulate_grid(records: list[SessionRecord]) -> dict[tuple, CellResult]:
    """Simulate all 16 grid cells across all session records."""
    cells: dict[tuple, CellResult] = {}
    for arch in ["A", "B"]:
        for entry_mode in [1, 2]:
            for k in K_VALUES:
                cells[(arch, entry_mode, k)] = CellResult(arch=arch, entry=entry_mode, k=k)

    for rec in records:
        orb  = rec.orb
        ext  = rec.ext

        if ext.direction == Direction.BEARISH:  # LONG
            boundary = orb.high
        else:                                    # SHORT
            boundary = orb.low

        for arch in ["A", "B"]:
            for entry_mode in [1, 2]:
                for k in K_VALUES:
                    cell = cells[(arch, entry_mode, k)]

                    # ── Entry price ───────────────────────────────────────
                    if entry_mode == 1:
                        entry_price = ext.extension_close
                        post = rec.post_ext
                    else:
                        if rec.pullback_price is None:
                            continue   # no pullback in this session
                        entry_price = rec.pullback_price
                        # Post-entry bars start after the pullback bar
                        post = rec.post_ext.loc[rec.post_ext.index > rec.pullback_ts]

                    # ── Stop calculation ──────────────────────────────────
                    stop_level = _compute_stop(arch, entry_price, boundary, k, rec.atr,
                                               ext.direction)

                    if ext.direction == Direction.BEARISH:  # LONG
                        stop_pts = max(0.0, entry_price - stop_level)
                    else:                                    # SHORT
                        stop_pts = max(0.0, stop_level - entry_price)

                    stop_usd = stop_pts * POINT_VALUE_USD
                    cell.stop_usd_all.append(stop_usd)

                    if stop_usd > STOP_CAP_USD or stop_pts < TICK_SIZE:
                        continue   # skip this trade

                    # ── TP level ──────────────────────────────────────────
                    if ext.direction == Direction.BEARISH:  # LONG
                        tp_level = entry_price + TP_MULTIPLE * stop_pts
                    else:
                        tp_level = entry_price - TP_MULTIPLE * stop_pts

                    # ── Walk bars ─────────────────────────────────────────
                    cell.n_traded += 1
                    outcome = "ts"   # default: time-stop at 11:30

                    for _, row in post.iterrows():
                        bar_hi = float(row["high"])
                        bar_lo = float(row["low"])

                        if ext.direction == Direction.BEARISH:  # LONG
                            if bar_lo <= stop_level:
                                outcome = "sl"
                                break
                            if bar_hi >= tp_level:
                                outcome = "tp"
                                break
                        else:                                    # SHORT
                            if bar_hi >= stop_level:
                                outcome = "sl"
                                break
                            if bar_lo <= tp_level:
                                outcome = "tp"
                                break

                    if outcome == "sl":
                        cell.n_sl += 1
                    elif outcome == "tp":
                        cell.n_tp += 1
                    else:
                        cell.n_ts += 1

    return cells


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(
    records: list[SessionRecord],
    n_sessions: int,
    n_orb: int,
    mae: dict,
    cells: dict[tuple, CellResult],
) -> None:
    n_ext = len(records)
    n_pb  = sum(1 for r in records if r.pullback_price is not None)

    print("=" * 76)
    print("ORBM-3 Phase A: Stop-Geometry Diagnostic")
    print(f"In-sample: {IN_SAMPLE_START.date()} → {IN_SAMPLE_END.date()}")
    print("=" * 76)

    print()
    print("─" * 76)
    print("FUNNEL")
    print("─" * 76)
    print(f"  Session days:                {n_sessions:>5}")
    print(f"  Sessions with valid ORB:     {n_orb:>5}  ({n_orb/n_sessions*100:.0f}%)")
    print(f"  Sessions with extension:     {n_ext:>5}  ({n_ext/n_orb*100:.0f}% of ORB sessions)")
    print(f"  Sessions with pullback (E2): {n_pb:>5}  ({n_pb/n_ext*100:.0f}% of extensions)")

    # ── MAE Study ──────────────────────────────────────────────────────────
    print()
    print("─" * 76)
    print("MAE DISTRIBUTION (breakouts that reach 1.5R geometric target)")
    print("  Reference entry: extension-bar close (Entry 1)")
    print("  Reference TP:    entry + 1.5 × 0.25 × ORB_size (geometric)")
    print("  Disqualifier:    price touches back through ORB boundary before TP")
    print("─" * 76)
    print(f"  Wins  (reached TP before boundary retest):  {mae['n_wins']:>4}  ({mae['win_rate']*100:.1f}% of extensions)")
    print(f"  Losses (boundary touched before TP):         {mae['n_losses']:>4}")
    print(f"  Timeout (no resolution before 11:30):        {mae['n_timeout']:>4}")
    print()

    if "mae_pts" in mae:
        print("  Max Adverse Excursion (MAE) among WINNING breakouts:")
        print(f"  {'Pct':>5}  {'pts':>7}  {'ATR mult':>9}  {'interpretation'}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*9}  {'─'*35}")
        for p in [25, 50, 75, 90, 95]:
            pts_v = mae["mae_pts"][p]
            atr_v = mae["mae_atr"].get(p, float("nan"))
            note = ""
            if p == 75:
                note = " ← GATE A: minimum structural k"
            elif p == 50:
                note = " ← median winner's worst retest"
            print(f"  {p:>5}  {pts_v:>7.2f}  {atr_v:>9.3f}  {note}")
        p75_atr = mae["mae_atr"].get(75, float("nan"))
        print()
        print(f"  ▶ 75th-pct MAE = {p75_atr:.3f} × ATR")
        print(f"    A stop placed at k ≥ {p75_atr:.2f}× ATR beyond entry/boundary survives")
        print(f"    75% of winning trades' retests without being stopped out.")
    else:
        print("  No winning breakouts found — cannot compute MAE distribution.")

    # ── Grid ───────────────────────────────────────────────────────────────
    p75_atr_gate = mae["mae_atr"].get(75, float("nan")) if "mae_atr" in mae else float("nan")

    print()
    print("─" * 76)
    print("GRID SCAN  (16 cells: 2 arch × 2 entries × 4 k values)")
    print("─" * 76)
    print(f"  {'Cell':<22}  {'Med$stop':>8}  {'≤$150':>6}  {'NSR':>6}  {'WR':>6}  {'PF':>5}  {'Freq':>6}  {'Gate'}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*6}")

    gate_passes: list[tuple] = []

    for arch in ["A", "B"]:
        for entry_mode in [1, 2]:
            for k in K_VALUES:
                c = cells[(arch, entry_mode, k)]
                entry_label = f"E{entry_mode}"
                n_traded_for_freq = c.n_traded
                freq = n_traded_for_freq / n_orb if n_orb > 0 else 0.0

                # Gate checks
                ok_stop   = c.frac_under_cap >= STOP_CAP_FRAC
                ok_nsr    = c.nsr <= GATE_NSR_MAX
                ok_freq   = freq >= GATE_FREQ_MIN
                ok_struct = k >= p75_atr_gate if not np.isnan(p75_atr_gate) else False
                all_ok    = ok_stop and ok_nsr and ok_freq and ok_struct

                gate_str = "✅ PASS" if all_ok else (
                    "⚠️ " +
                    ("stop " if not ok_stop else "") +
                    ("NSR " if not ok_nsr else "") +
                    ("freq " if not ok_freq else "") +
                    ("k<MAE" if not ok_struct else "")
                ).strip()

                label = f"Arch{arch} {entry_label} k={k:.2f}"
                print(f"  {label:<22}  "
                      f"${c.median_stop_usd:>7.0f}  "
                      f"{c.frac_under_cap*100:>5.0f}%  "
                      f"{c.nsr*100:>5.0f}%  "
                      f"{c.win_rate*100:>5.0f}%  "
                      f"{c.profit_factor:>5.2f}  "
                      f"{freq:>5.2f}/d  "
                      f"{gate_str}")

                if all_ok:
                    gate_passes.append((arch, entry_mode, k, c, freq))

    # ── Gate A Verdict ─────────────────────────────────────────────────────
    print()
    print("=" * 76)
    print("GATE A VERDICT")
    print("=" * 76)
    print(f"  Criteria (all required):")
    print(f"    ① Stop ≤ $150 on ≥ {STOP_CAP_FRAC*100:.0f}% of setups")
    print(f"    ② NSR ≤ {GATE_NSR_MAX*100:.0f}%  (WR ≥ {(1-GATE_NSR_MAX)*100:.0f}%)")
    print(f"    ③ Frequency ≥ {GATE_FREQ_MIN:.1f} setup/ORB-session-day")
    if not np.isnan(p75_atr_gate):
        print(f"    ④ k ≥ {p75_atr_gate:.3f} × ATR  (75th-pct MAE — structural justification)")
    else:
        print(f"    ④ k ≥ 75th-pct MAE in ATR multiples (could not compute — no winning breakouts)")
    print()

    if gate_passes:
        print(f"  ✅ GATE A PASS — {len(gate_passes)} cell(s) meet all criteria")
        print()
        print("  Passing cells (candidates for ORBM-3 pre-registration):")
        print(f"  {'Cell':<22}  {'Med$stop':>8}  {'NSR':>6}  {'WR':>6}  {'PF':>5}  {'Freq':>6}")
        for arch, entry_mode, k, c, freq in gate_passes:
            label = f"Arch{arch} E{entry_mode} k={k:.2f}"
            print(f"  {label:<22}  "
                  f"${c.median_stop_usd:>7.0f}  "
                  f"{c.nsr*100:>5.0f}%  "
                  f"{c.win_rate*100:>5.0f}%  "
                  f"{c.profit_factor:>5.2f}  "
                  f"{freq:>5.2f}/d")
        print()
        # Recommend: pick the passing cell with highest PF among those with best freq
        best = max(gate_passes, key=lambda x: (x[3].profit_factor, x[4]))
        arch_b, em_b, k_b, c_b, freq_b = best
        print(f"  RECOMMENDED FOR ORBM-3:")
        print(f"    Architecture:  {'A (boundary − k×ATR)' if arch_b == 'A' else 'B (entry − k×ATR)'}")
        print(f"    Entry mode:    {'E1 (chase: extension-bar close)' if em_b == 1 else 'E2 (pullback: retest of boundary)'}")
        print(f"    k:             {k_b}")
        print(f"    Median stop:   ${c_b.median_stop_usd:.0f}/contract")
        print(f"    NSR:           {c_b.nsr*100:.0f}%")
        print(f"    WR:            {c_b.win_rate*100:.0f}%")
        print(f"    PF (est):      {c_b.profit_factor:.2f}")
        print(f"    Frequency:     {freq_b:.2f}/day")
        print()
        print("  → Proceed to Phase B: write orbm3_config.yaml, orbm3_core.py,")
        print("    prereg_orbm3_seal.py, commit, then backtest_orbm3_combine.py.")
    else:
        print("  ❌ GATE A FAIL — no cell meets all four criteria")
        print()
        print("  The ORB framework is structurally incompatible with Topstep combine")
        print("  risk limits at any tested stop architecture.")
        print()
        print("  → Pivot to VWAP-deviation mean reversion (new pre-registration).")
        print("    Do NOT build ORBM-3. The ORB work is done.")

    print("=" * 76)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading bars…", end=" ", flush=True)
    df = load_bars_et([CSV_2025, CSV_2026], IN_SAMPLE_START, IN_SAMPLE_END)
    if df.empty:
        print("ERROR: no bars loaded — check CSV paths")
        sys.exit(1)
    print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")

    print("Building session records…", end=" ", flush=True)
    records, n_sessions, n_orb = collect_sessions(df)
    print(f"{len(records)} extension sessions")

    print("Running MAE study…", end=" ", flush=True)
    mae = measure_mae_distribution(records)
    print(f"{mae['n_wins']} winners")

    print("Running grid simulation (16 cells)…", end=" ", flush=True)
    cells = simulate_grid(records)
    print("done")
    print()

    print_report(records, n_sessions, n_orb, mae, cells)


if __name__ == "__main__":
    main()
