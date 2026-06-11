"""study_orb_continuation_target.py — ORBM-1 Gate A / Study 2: TP Achievability.

Answers Mary's controlling question before pre-registering ORBM-1:
    Of the 182 ORB extensions, what fraction reach a 2R or 3R continuation target
    before 11:30 ET — or before stop is hit?

Where R = stop distance = (entry_price − opposite_orb_boundary) for a long
breakout, or (opposite_boundary − entry_price) for a short breakout.

For ORBM-1 (breakout momentum):
  Upward extension (close above ORB_high + 0.5×ORB_size) → LONG entry
    entry  = extension bar close
    stop   = ORB_low + 1 tick   (opposite boundary)
    target = entry + N×R        (continuation above entry)

  Downward extension (close below ORB_low − 0.5×ORB_size) → SHORT entry
    entry  = extension bar close
    stop   = ORB_high − 1 tick  (opposite boundary)
    target = entry − N×R        (continuation below entry)

Also reports:
  • Close-positive-by-11:30 rate (Mary's ~52% to confirm)
  • Stop-size distribution (what % pass the $200/contract cap)
  • MFE distribution in R multiples

Gate A verdict (Study 2):
  ≥ 40% reach 2R before stop/close → strong TP achievability → PASS (target)
  30–39% reach 2R                  → minimum pass
  < 30% reach 2R                   → combine profit target not achievable → FAIL

In-sample: 2025-01-01 → 2026-02-28 UTC

Usage:
    .venv/bin/python study_orb_continuation_target.py
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

STOP_SKIP_USD = 200.0   # per contract
STOP_SMALL_USD = 100.0  # per contract

# TP multiple targets to evaluate
TP_MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def simulate_orbm_trade(post_df, entry: float, stop: float, direction: Direction, hard_close_str: str):
    """Simulate ORBM-1 trade: walk bars and find MFE and which R multiple first hit.

    Returns:
        (close_positive: bool, mfe_r: float, best_r_hit: float, stopped_out: bool)
        mfe_r: max favorable excursion in R units before exit
        best_r_hit: highest TP multiple reached before stop/close (0 if stopped)
        stopped_out: True if stop was hit before 11:30 close
    """
    if direction == Direction.BEARISH:  # upward extension → LONG
        r_pts = entry - stop            # positive: how far stop is below entry
    else:                               # downward extension → SHORT
        r_pts = stop - entry            # positive: how far stop is above entry

    if r_pts <= 0:
        return False, 0.0, 0.0, False

    post_close = post_df.between_time("00:00", hard_close_str, inclusive="left")

    mfe_r = 0.0
    best_r_hit = 0.0
    stopped = False
    final_close = None

    for _, row in post_close.iterrows():
        bar_high = float(row["high"])
        bar_low  = float(row["low"])
        bar_close = float(row["close"])
        final_close = bar_close

        if direction == Direction.BEARISH:  # LONG
            # Favorable move: bar's high above entry
            fav_pts = bar_high - entry
            # Stop check: bar's low hits or crosses stop
            if bar_low <= stop:
                stopped = True
                break
        else:  # SHORT
            # Favorable move: bar's low below entry
            fav_pts = entry - bar_low
            # Stop check: bar's high hits or crosses stop
            if bar_high >= stop:
                stopped = True
                break

        if fav_pts > 0:
            fav_r = fav_pts / r_pts
            if fav_r > mfe_r:
                mfe_r = fav_r
            for mult in TP_MULTIPLES:
                if fav_r >= mult and mult > best_r_hit:
                    best_r_hit = mult

    # Close-positive: did we end above entry (long) / below entry (short)?
    if final_close is not None and not stopped:
        if direction == Direction.BEARISH:
            close_positive = final_close > entry
        else:
            close_positive = final_close < entry
    else:
        close_positive = False

    return close_positive, mfe_r, best_r_hit, stopped


def main() -> None:
    print("=" * 70)
    print("ORBM-1 Gate A / Study 2: TP Achievability (Max-Fav Excursion)")
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

    # Track all extension outcomes
    extensions_all: list[dict] = []
    monthly: dict[str, list[dict]] = defaultdict(list)

    sessions = df.groupby("_date")
    n_sessions = 0
    n_orb = 0

    for date_et, sess_df in sessions:
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

        # ORBM-1 entry: in the direction of the extension
        entry = ext.extension_close

        # Stop: opposite ORB boundary
        if ext.direction == Direction.BEARISH:  # upward extension → LONG
            stop = orb.low + TICK_SIZE
        else:                                   # downward extension → SHORT
            stop = orb.high - TICK_SIZE

        # Stop distance in USD per contract
        if ext.direction == Direction.BEARISH:
            stop_pts = entry - stop
        else:
            stop_pts = stop - entry

        stop_usd = max(stop_pts, 0) * POINT_VALUE_USD

        # Contracts based on stop size
        if stop_usd > STOP_SKIP_USD:
            contracts = 0   # skip
        elif stop_usd < STOP_SMALL_USD:
            contracts = 2
        else:
            contracts = 1

        post_df = sess_df.loc[sess_df.index > ext.detection_bar_ts]
        close_pos, mfe_r, best_r, stopped = simulate_orbm_trade(
            post_df, entry, stop, ext.direction, hard_close_str
        )

        rec = {
            "date": date_et,
            "direction": ext.direction.value,
            "entry": entry,
            "stop": stop,
            "stop_pts": stop_pts,
            "stop_usd": stop_usd,
            "contracts": contracts,
            "close_positive": close_pos,
            "mfe_r": mfe_r,
            "best_r_hit": best_r,
            "stopped_out": stopped,
            "orb_size": orb.size,
        }
        extensions_all.append(rec)
        ym = f"{date_et.year}-{date_et.month:02d}"
        monthly[ym].append(rec)

    n_ext = len(extensions_all)
    if n_ext == 0:
        print("ERROR: no extensions found")
        sys.exit(1)

    # ── Filter sets ──────────────────────────────────────────────────────────
    tradeable   = [r for r in extensions_all if r["contracts"] > 0]
    non_skipped = tradeable  # same as tradeable
    n_trade     = len(tradeable)

    # ── Metrics ──────────────────────────────────────────────────────────────
    def pct_reach(recs, mult):
        if not recs:
            return 0.0
        return sum(1 for r in recs if r["best_r_hit"] >= mult) / len(recs)

    def pct_closed_pos(recs):
        if not recs:
            return 0.0
        return sum(1 for r in recs if r["close_positive"]) / len(recs)

    def pct_stopped(recs):
        if not recs:
            return 0.0
        return sum(1 for r in recs if r["stopped_out"]) / len(recs)

    mfe_r_all   = [r["mfe_r"] for r in extensions_all]
    mfe_r_trade = [r["mfe_r"] for r in tradeable]
    stops_usd   = [r["stop_usd"] for r in extensions_all]

    print()
    print("─" * 70)
    print("STOP SIZE DISTRIBUTION (all 182 extensions)")
    print("─" * 70)
    print(f"  Median stop:     {np.median(stops_usd):.0f} pts × $2 = ${np.median(stops_usd):.0f}/contract")
    print(f"  Mean stop:       {np.mean(stops_usd):.0f} pts × $2 = ${np.mean(stops_usd):.0f}/contract")
    print(f"  Skipped (>$200): {sum(1 for r in extensions_all if r['contracts']==0):>4} / {n_ext}  ({sum(1 for r in extensions_all if r['contracts']==0)/n_ext*100:.0f}%)")
    print(f"  1-contract:      {sum(1 for r in extensions_all if r['contracts']==1):>4} / {n_ext}")
    print(f"  2-contract:      {sum(1 for r in extensions_all if r['contracts']==2):>4} / {n_ext}")
    print(f"  Tradeable total: {n_trade:>4} / {n_ext}")

    print()
    print("─" * 70)
    print("TP ACHIEVABILITY — ALL EXTENSIONS (unfiltered, N={})".format(n_ext))
    print("─" * 70)
    print(f"  Close positive by {hard_close_str} ET:  {pct_closed_pos(extensions_all)*100:.1f}%")
    print(f"  Stopped out before {hard_close_str} ET: {pct_stopped(extensions_all)*100:.1f}%")
    print(f"  Median MFE:                          {np.median(mfe_r_all):.2f}R")
    print(f"  Mean MFE:                            {np.mean(mfe_r_all):.2f}R")
    print()
    print("  Fraction reaching each TP multiple:")
    for mult in TP_MULTIPLES:
        r = pct_reach(extensions_all, mult)
        bar = "█" * int(r * 40)
        marker = " ← Mary's 30% bar" if abs(mult - 2.0) < 0.01 else (" ← 3R target" if abs(mult - 3.0) < 0.01 else "")
        print(f"    {mult:.1f}R: {r*100:5.1f}%  {bar}{marker}")

    print()
    print("─" * 70)
    print("TP ACHIEVABILITY — TRADEABLE ONLY (stop ≤ $200/contract, N={})".format(n_trade))
    print("─" * 70)
    if n_trade > 0:
        print(f"  Close positive by {hard_close_str} ET:  {pct_closed_pos(tradeable)*100:.1f}%")
        print(f"  Stopped out before {hard_close_str} ET: {pct_stopped(tradeable)*100:.1f}%")
        print(f"  Median MFE:                          {np.median(mfe_r_trade):.2f}R")
        print(f"  Mean MFE:                            {np.mean(mfe_r_trade):.2f}R")
        print()
        print("  Fraction reaching each TP multiple:")
        for mult in TP_MULTIPLES:
            r = pct_reach(tradeable, mult)
            bar = "█" * int(r * 40)
            marker = " ← Mary's 30% bar" if abs(mult - 2.0) < 0.01 else (" ← 3R target" if abs(mult - 3.0) < 0.01 else "")
            print(f"    {mult:.1f}R: {r*100:5.1f}%  {bar}{marker}")

    # ── Direction split ──────────────────────────────────────────────────────
    bear_ext = [r for r in tradeable if r["direction"] == "BEARISH"]
    bull_ext = [r for r in tradeable if r["direction"] == "BULLISH"]

    print()
    print("─" * 70)
    print("DIRECTION BREAKDOWN (tradeable only)")
    print("─" * 70)
    for label, recs in [("Upward extensions (LONG)", bear_ext),
                         ("Downward extensions (SHORT)", bull_ext)]:
        n = len(recs)
        if n == 0:
            continue
        print(f"  {label}: N={n}")
        print(f"    Close positive: {pct_closed_pos(recs)*100:.1f}%  |  "
              f"Stopped: {pct_stopped(recs)*100:.1f}%  |  "
              f"2R: {pct_reach(recs, 2.0)*100:.1f}%  |  "
              f"3R: {pct_reach(recs, 3.0)*100:.1f}%")

    # ── Gate A verdict ───────────────────────────────────────────────────────
    rate_2r_trade = pct_reach(tradeable, 2.0) if n_trade > 0 else 0.0
    rate_3r_trade = pct_reach(tradeable, 3.0) if n_trade > 0 else 0.0

    print()
    print("─" * 70)
    if rate_2r_trade >= 0.40:
        verdict = f"✅ GATE A STUDY 2 PASS (TARGET) — {rate_2r_trade*100:.1f}% reach 2R ≥ 40%"
    elif rate_2r_trade >= 0.30:
        verdict = f"✅ GATE A STUDY 2 PASS (MINIMUM) — {rate_2r_trade*100:.1f}% reach 2R ≥ 30%"
    else:
        verdict = f"🔴 GATE A STUDY 2 FAIL — {rate_2r_trade*100:.1f}% reach 2R < 30%; TP target not achievable"
    print(f"  {verdict}")
    print(f"  (3R achievability: {rate_3r_trade*100:.1f}%)")
    print("─" * 70)

    # ── Monthly breakdown ────────────────────────────────────────────────────
    print()
    print("BY-MONTH: tradeable extensions, close-positive rate, 2R rate")
    print(f"  {'Month':<10}  {'N':>4}  {'Close+':>8}  {'2R hit':>8}  {'3R hit':>8}")
    print(f"  {'─'*10}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}")
    for ym in sorted(monthly):
        recs_t = [r for r in monthly[ym] if r["contracts"] > 0]
        if not recs_t:
            continue
        cp = pct_closed_pos(recs_t)
        r2 = pct_reach(recs_t, 2.0)
        r3 = pct_reach(recs_t, 3.0)
        print(f"  {ym:<10}  {len(recs_t):>4}  {cp*100:>7.0f}%  {r2*100:>7.0f}%  {r3*100:>7.0f}%")

    print()
    print("=" * 70)
    print("Summary for pre-registration decision:")
    print(f"  Tradeable extensions:  {n_trade} / {n_ext} total ({n_trade/n_ext*100:.0f}%)")
    print(f"  Close positive by 11:30: {pct_closed_pos(tradeable)*100:.1f}%")
    print(f"  Reach 2R before stop:  {rate_2r_trade*100:.1f}%  [Mary minimum: 30%, target: 40%]")
    print(f"  Reach 3R before stop:  {rate_3r_trade*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
