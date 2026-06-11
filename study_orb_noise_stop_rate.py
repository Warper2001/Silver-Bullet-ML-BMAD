"""study_orb_noise_stop_rate.py — ORBM-1 v2 diagnostic: noise-stop rate.

Measures the critical unknown before pre-registering ORBM-1 v2:
  What fraction of tradeable ORB extensions see the stop hit FIRST —
  before reaching 1R TP or the 11:30 ET hard close?

Strategy under test (ORBM-1 v2):
  Entry:  Extension close (0.5×ORB_size beyond boundary), in direction of extension
  Stop:   ORB boundary ± 1 tick (0.5×ORB_size from entry); skip if > $150/contract (75 pts)
  TP:     1R = stop_distance pts in the continuation direction
  Close:  11:30 ET hard close

Outcome classification per extension:
  WIN         — 1R target hit before stop or 11:30
  STOP_LOSS   — stop hit before 1R or 11:30
  TIME_POS    — reached 11:30 with unrealized gain (close > entry)
  TIME_NEG    — reached 11:30 with unrealized loss (close < entry)
  TIME_EVEN   — reached 11:30 flat

Decision rule:
  NSR (noise-stop rate) = STOP_LOSS / tradeable_extensions
  NSR ≤ 20%: proceed to pre-register ORBM-1 v2
  NSR > 20%: re-evaluate entry timing or abandon MNQ ORB breakout approach

In-sample: 2025-01-01 → 2026-02-28

Usage:
    .venv/bin/python study_orb_noise_stop_rate.py
"""

import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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

UTC = timezone.utc
IN_SAMPLE_START = datetime(2025, 1, 1, tzinfo=UTC)
IN_SAMPLE_END   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

CFG = SORMConfig()

# ORBM-1 v2 risk parameters
STOP_CAP_PTS = 75.0        # skip if stop > 75 pts ($150/contract)
STOP_SMALL_PTS = 50.0      # stop < 50 pts → 2 contracts
STOP_CAP_USD = STOP_CAP_PTS * POINT_VALUE_USD  # $150/contract

WIN        = "WIN"
STOP_LOSS  = "STOP_LOSS"
TIME_POS   = "TIME_POS"
TIME_NEG   = "TIME_NEG"
TIME_EVEN  = "TIME_EVEN"


def simulate_v2_trade(post_df, entry: float, stop: float, tp: float,
                      direction: Direction, hard_close_str: str) -> tuple[str, float]:
    """Bar-by-bar simulation of ORBM-1 v2.

    Returns (outcome, pnl_pts) where pnl_pts is the trade P&L in index points
    (before × contracts × POINT_VALUE_USD).
    """
    post_close = post_df.between_time("00:00", hard_close_str, inclusive="left")

    for _, row in post_close.iterrows():
        bar_high  = float(row["high"])
        bar_low   = float(row["low"])
        bar_close = float(row["close"])

        if direction == Direction.BEARISH:  # upward extension → LONG
            # Check stop first (conservative — assumes stop hit before TP on same bar)
            if bar_low <= stop:
                pnl_pts = stop - entry  # negative
                return STOP_LOSS, pnl_pts
            if bar_high >= tp:
                pnl_pts = tp - entry   # positive = 1R
                return WIN, pnl_pts
        else:  # downward extension → SHORT
            if bar_high >= stop:
                pnl_pts = entry - stop  # negative
                return STOP_LOSS, pnl_pts
            if bar_low <= tp:
                pnl_pts = entry - tp   # positive = 1R
                return WIN, pnl_pts

    # Reached 11:30 ET without hitting stop or TP
    last_close = float(post_close.iloc[-1]["close"]) if not post_close.empty else entry
    if direction == Direction.BEARISH:
        pnl_pts = last_close - entry
    else:
        pnl_pts = entry - last_close

    if pnl_pts > 0.1:
        return TIME_POS, pnl_pts
    elif pnl_pts < -0.1:
        return TIME_NEG, pnl_pts
    else:
        return TIME_EVEN, 0.0


def main() -> None:
    print("=" * 70)
    print("ORBM-1 v2 Noise-Stop Rate Diagnostic")
    print("Stop: ORB boundary (0.5×ORB_size), capped at 75 pts ($150/contract)")
    print("TP:   1R = stop_distance pts in continuation direction")
    print(f"In-sample: {IN_SAMPLE_START.date()} → {IN_SAMPLE_END.date()}")
    print("=" * 70)

    print("\nLoading bars…", end=" ", flush=True)
    df = load_bars_et([CSV_2025, CSV_2026], IN_SAMPLE_START, IN_SAMPLE_END)
    if df.empty:
        print("ERROR: no bars loaded")
        sys.exit(1)
    df["_date"] = df.index.date
    print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")

    hard_close_str = CFG.hard_close_et.strftime("%H:%M")
    sessions = df.groupby("_date")

    records: list[dict] = []
    monthly: dict[str, list[dict]] = defaultdict(list)

    n_total = 0
    n_orb = 0
    n_ext = 0
    n_skipped_cap = 0

    for date_et, sess_df in sessions:
        if date_et.weekday() >= 5:
            continue
        sess_df = sess_df.drop(columns=["_date"])
        n_total += 1

        orb = build_opening_range(sess_df, CFG)
        if orb is None:
            continue
        n_orb += 1

        ext = detect_extension(sess_df, orb, CFG)
        if ext is None:
            continue
        n_ext += 1

        entry = ext.extension_close

        # Stop: ORB boundary ± 1 tick
        if ext.direction == Direction.BEARISH:  # upward → LONG
            stop  = orb.high - TICK_SIZE   # just below ORB_high
            tp    = entry + (entry - stop) # 1R above entry
        else:                               # downward → SHORT
            stop  = orb.low + TICK_SIZE    # just above ORB_low
            tp    = entry - (stop - entry) # 1R below entry

        stop_pts = abs(entry - stop)
        stop_usd = stop_pts * POINT_VALUE_USD

        if stop_pts > STOP_CAP_PTS:
            n_skipped_cap += 1
            continue  # stop too wide

        # Contracts
        contracts = 2 if stop_pts < STOP_SMALL_PTS else 1

        post_df = sess_df.loc[sess_df.index > ext.detection_bar_ts]
        outcome, pnl_pts = simulate_v2_trade(
            post_df, entry, stop, tp, ext.direction, hard_close_str
        )

        pnl_usd = pnl_pts * POINT_VALUE_USD * contracts

        rec = {
            "date":       date_et,
            "direction":  ext.direction.value,
            "entry":      entry,
            "stop":       stop,
            "tp":         tp,
            "stop_pts":   stop_pts,
            "stop_usd":   stop_usd,
            "contracts":  contracts,
            "outcome":    outcome,
            "pnl_pts":    pnl_pts,
            "pnl_usd":    pnl_usd,
            "orb_size":   orb.size,
        }
        records.append(rec)
        ym = f"{date_et.year}-{date_et.month:02d}"
        monthly[ym].append(rec)

    n_trade = len(records)
    if n_trade == 0:
        print("ERROR: no tradeable extensions found")
        sys.exit(1)

    # ── Outcome counts ────────────────────────────────────────────────────────
    outcome_counts = {WIN: 0, STOP_LOSS: 0, TIME_POS: 0, TIME_NEG: 0, TIME_EVEN: 0}
    for r in records:
        outcome_counts[r["outcome"]] += 1

    n_win   = outcome_counts[WIN]
    n_sl    = outcome_counts[STOP_LOSS]
    n_tp    = outcome_counts[TIME_POS]
    n_tn    = outcome_counts[TIME_NEG]
    n_te    = outcome_counts[TIME_EVEN]

    nsr     = n_sl / n_trade
    win_r   = n_win / n_trade
    # "True" directional hit: WIN + TIME_POS (close positive, regardless of TP)
    dir_hit = (n_win + n_tp) / n_trade

    total_pnl    = sum(r["pnl_usd"] for r in records)
    pnl_per_day  = total_pnl / n_trade
    wins_pnl     = [r["pnl_usd"] for r in records if r["outcome"] == WIN]
    loss_pnl     = [r["pnl_usd"] for r in records if r["outcome"] == STOP_LOSS]
    avg_win_usd  = np.mean(wins_pnl) if wins_pnl else 0.0
    avg_loss_usd = np.mean(loss_pnl) if loss_pnl else 0.0

    gross_profit = sum(r["pnl_usd"] for r in records if r["pnl_usd"] > 0)
    gross_loss   = sum(-r["pnl_usd"] for r in records if r["pnl_usd"] < 0)
    pf           = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # ── Print ─────────────────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("PIPELINE SUMMARY")
    print("─" * 70)
    print(f"  Sessions:             {n_total}")
    print(f"  Valid ORB sessions:   {n_orb}")
    print(f"  Extensions detected:  {n_ext}")
    print(f"  Skipped (stop cap):   {n_skipped_cap}  (stop > 75 pts / $150/contract)")
    print(f"  Tradeable:            {n_trade}  ({n_trade/n_ext*100:.0f}% of extensions)")

    print()
    print("─" * 70)
    print("OUTCOME BREAKDOWN  (bar-by-bar simulation)")
    print("─" * 70)
    for label, key, desc in [
        ("WIN (1R hit first)",        WIN,       "TP reached before stop or 11:30"),
        ("STOP_LOSS (stop hit first)", STOP_LOSS, "stop triggered before TP or 11:30"),
        ("TIME_POS (11:30, profit)",   TIME_POS,  "11:30 close with unrealized gain"),
        ("TIME_NEG (11:30, loss)",     TIME_NEG,  "11:30 close with unrealized loss"),
        ("TIME_EVEN",                  TIME_EVEN, "11:30 close near entry"),
    ]:
        n = outcome_counts[key]
        pct = n / n_trade * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<30} {n:>4} ({pct:5.1f}%)  {bar}")

    print()
    print("─" * 70)
    print("KEY METRICS")
    print("─" * 70)
    print(f"  Noise-stop rate (NSR):         {nsr*100:.1f}%  — fraction stopped before TP or 11:30")
    print(f"  1R win rate:                   {win_r*100:.1f}%")
    print(f"  Directional hit rate:          {dir_hit*100:.1f}%  (WIN + TIME_POS)")
    print(f"  Avg win (USD):                 ${avg_win_usd:.0f}")
    print(f"  Avg loss (USD):                ${avg_loss_usd:.0f}")
    print(f"  Profit factor:                 {pf:.2f}")
    print(f"  Total P&L ({n_trade} trades):        ${total_pnl:,.0f}")
    print(f"  P&L per trade:                 ${pnl_per_day:.0f}")
    print(f"  Implied 60-day P&L:            ${pnl_per_day * (n_trade / 14 * 3):.0f}  (at {n_trade/14*3:.0f} trades/60 days)")

    # ── Gate verdict ──────────────────────────────────────────────────────────
    print()
    print("─" * 70)
    if nsr <= 0.15:
        verdict = f"✅ NSR PASS (STRONG)  — {nsr*100:.1f}% ≤ 15%; proceed to pre-register ORBM-1 v2"
    elif nsr <= 0.20:
        verdict = f"✅ NSR PASS (MARGINAL) — {nsr*100:.1f}% ≤ 20%; proceed with caution"
    elif nsr <= 0.35:
        verdict = f"⚠️  NSR ELEVATED — {nsr*100:.1f}%; re-evaluate entry timing before pre-registering"
    else:
        verdict = f"🔴 NSR FAIL — {nsr*100:.1f}% > 35%; stop placement too tight for MNQ noise floor"
    print(f"  {verdict}")
    print("─" * 70)

    # ── Direction breakdown ───────────────────────────────────────────────────
    bear = [r for r in records if r["direction"] == "BEARISH"]
    bull = [r for r in records if r["direction"] == "BULLISH"]
    print()
    print("DIRECTION BREAKDOWN")
    print("─" * 70)
    for label, recs in [("Upward ext (LONG)", bear), ("Downward ext (SHORT)", bull)]:
        if not recs:
            continue
        n  = len(recs)
        sl = sum(1 for r in recs if r["outcome"] == STOP_LOSS)
        w  = sum(1 for r in recs if r["outcome"] == WIN)
        tp_pos = sum(1 for r in recs if r["outcome"] == TIME_POS)
        pl = sum(r["pnl_usd"] for r in recs)
        print(f"  {label}: N={n}  WR={w/n*100:.0f}%  NSR={sl/n*100:.0f}%"
              f"  TIME_POS={tp_pos/n*100:.0f}%  TotalP&L=${pl:,.0f}")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    print()
    print("BY-MONTH: NSR, WR, P&L")
    print(f"  {'Month':<10}  {'N':>4}  {'NSR':>6}  {'WR(1R)':>8}  {'PF':>5}  {'P&L':>8}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*8}")
    for ym in sorted(monthly):
        recs = monthly[ym]
        n = len(recs)
        sl = sum(1 for r in recs if r["outcome"] == STOP_LOSS)
        w  = sum(1 for r in recs if r["outcome"] == WIN)
        gp = sum(r["pnl_usd"] for r in recs if r["pnl_usd"] > 0)
        gl = sum(-r["pnl_usd"] for r in recs if r["pnl_usd"] < 0)
        mpf = gp / gl if gl > 0 else float("inf")
        pl = sum(r["pnl_usd"] for r in recs)
        mpf_str = f"{mpf:.2f}" if mpf != float("inf") else "∞"
        print(f"  {ym:<10}  {n:>4}  {sl/n*100:>5.0f}%  {w/n*100:>7.0f}%  {mpf_str:>5}  ${pl:>7,.0f}")

    # ── Stop size distribution ────────────────────────────────────────────────
    stops = [r["stop_pts"] for r in records]
    print()
    print("STOP SIZE DISTRIBUTION (tradeable, in index points)")
    print(f"  Min:    {min(stops):.1f} pts  (${min(stops)*POINT_VALUE_USD:.0f}/contract)")
    print(f"  Median: {np.median(stops):.1f} pts  (${np.median(stops)*POINT_VALUE_USD:.0f}/contract)")
    print(f"  Mean:   {np.mean(stops):.1f} pts  (${np.mean(stops)*POINT_VALUE_USD:.0f}/contract)")
    print(f"  Max:    {max(stops):.1f} pts  (${max(stops)*POINT_VALUE_USD:.0f}/contract)")
    n2c = sum(1 for r in records if r["contracts"] == 2)
    n1c = sum(1 for r in records if r["contracts"] == 1)
    print(f"  2-contract trades (stop < 50 pts): {n2c} ({n2c/n_trade*100:.0f}%)")
    print(f"  1-contract trades (50–75 pts):     {n1c} ({n1c/n_trade*100:.0f}%)")

    print()
    print("=" * 70)
    print(f"PRE-REGISTRATION DECISION GATE")
    print(f"  NSR = {nsr*100:.1f}%")
    if nsr <= 0.20:
        print("  → PROCEED: pre-register ORBM-1 v2 with ORB boundary stop + 1R TP")
        print("  → Disclose Phase A contamination in the prereg document")
        print("  → OOS holdout (≥ 2026-03-01) is the only verdict that matters")
    else:
        print("  → STOP: NSR too high; reconsider entry timing (wait for ORB boundary")
        print("    retest rather than entering at extension close) before pre-registering")
    print("=" * 70)


if __name__ == "__main__":
    main()
