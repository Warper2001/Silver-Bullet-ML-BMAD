"""study_orb_control_window.py — ORBM-1 Gate A / Study 1: ORB vs. Control Window.

Dr. Quinn's structural test: is the 74% continuation rate ORB-specific, or is it
generic intraday momentum that appears in any time window?

Method:
  Run the identical extension/continuation logic against TWO windows:
    ORB window:     build 09:30–09:44 ET, extensions 09:45–10:44 ET, close 11:30
    Control window: build 12:00–12:14 ET, extensions 12:15–13:14 ET, close 14:00

  The "continuation rate" = % of extensions that do NOT revert to their window's
  mid before hard close. This is exactly the Gate 0 measurement logic.

  ORB-specific excess = ORB_rate − Control_rate.

Gate A verdict (Study 1):
  Excess ≥ 10 ppt  → ORB context is doing meaningful work → PASS
  Excess 5–9 ppt   → marginal — report and let user decide
  Excess < 5 ppt   → generic momentum; ORB window adds nothing → FAIL

In-sample: 2025-01-01 → 2026-02-28 UTC
Pre-registration for SORM: ffb60f5 (exploratory phase — new prereg before backtest)

Usage:
    .venv/bin/python study_orb_control_window.py
"""

import sys
from collections import defaultdict
from datetime import datetime, time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.research.sorm_core import (
    SORMConfig,
    build_opening_range,
    check_reversion_to_mid,
    detect_extension,
    load_bars_et,
)

UTC = timezone.utc
IN_SAMPLE_START = datetime(2025, 1, 1, tzinfo=UTC)
IN_SAMPLE_END   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

# ── Two configurations ────────────────────────────────────────────────────────

ORB_CFG = SORMConfig(
    orb_start_et=time(9, 30),
    orb_end_et=time(9, 45),        # [09:30, 09:45) — 15 bars
    extension_start_et=time(9, 45),
    extension_end_et=time(10, 45), # [09:45, 10:45) — 60 bars
    hard_close_et=time(11, 30),
    extension_threshold=0.5,
    orb_min_size_points=2.0,
)

CTRL_CFG = SORMConfig(
    orb_start_et=time(12, 0),
    orb_end_et=time(12, 15),       # [12:00, 12:15) — 15 bars
    extension_start_et=time(12, 15),
    extension_end_et=time(13, 15), # [12:15, 13:15) — 60 bars
    hard_close_et=time(14, 0),
    extension_threshold=0.5,
    orb_min_size_points=2.0,
)


def run_window(df, cfg: SORMConfig, label: str) -> dict:
    """Run the continuation study for one window configuration."""
    print(f"\n  Running {label} window…", end=" ", flush=True)

    sessions = df.groupby(df.index.date)
    counts = {
        "sessions": 0,
        "with_range": 0,
        "with_extension": 0,
        "continued": 0,
    }
    monthly: dict[str, dict] = defaultdict(lambda: {"ext": 0, "cont": 0})

    hard_close_str = cfg.hard_close_et.strftime("%H:%M")

    for date_et, sess_df in sessions:
        if date_et.weekday() >= 5:
            continue
        sess_df = sess_df.copy()
        counts["sessions"] += 1

        orb = build_opening_range(sess_df, cfg)
        if orb is None:
            continue
        counts["with_range"] += 1

        ext = detect_extension(sess_df, orb, cfg)
        if ext is None:
            continue
        counts["with_extension"] += 1

        month_key = f"{date_et.year}-{date_et.month:02d}"
        monthly[month_key]["ext"] += 1

        # Post-extension bars up to hard_close
        post_df = sess_df.loc[sess_df.index > ext.detection_bar_ts]
        post_df = post_df.between_time("00:00", hard_close_str, inclusive="left")

        reverted = check_reversion_to_mid(post_df, orb, ext)
        if not reverted:
            counts["continued"] += 1
            monthly[month_key]["cont"] += 1

    n_ext  = counts["with_extension"]
    n_cont = counts["continued"]
    rate   = n_cont / n_ext if n_ext > 0 else 0.0

    print(f"{n_ext} extensions, {n_cont} continued ({rate*100:.1f}%)")
    return {"label": label, "counts": counts, "monthly": dict(monthly), "rate": rate}


def main() -> None:
    print("=" * 70)
    print("ORBM-1 Gate A / Study 1: ORB vs. Control Window Continuation Rate")
    print(f"In-sample: {IN_SAMPLE_START.date()} → {IN_SAMPLE_END.date()}")
    print("=" * 70)

    print("\nLoading bars…", end=" ", flush=True)
    df = load_bars_et([CSV_2025, CSV_2026], IN_SAMPLE_START, IN_SAMPLE_END)
    if df.empty:
        print("ERROR: no bars loaded")
        sys.exit(1)
    df["_date"] = df.index.date
    print(f"{len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")

    orb_result  = run_window(df, ORB_CFG,  "ORB (09:30–10:45)")
    ctrl_result = run_window(df, CTRL_CFG, "Control (12:00–13:15)")

    orb_rate  = orb_result["rate"]
    ctrl_rate = ctrl_result["rate"]
    excess    = orb_rate - ctrl_rate

    print()
    print("─" * 70)
    print("RESULTS")
    print("─" * 70)

    orb_n  = orb_result["counts"]["with_extension"]
    ctrl_n = ctrl_result["counts"]["with_extension"]

    print(f"  ORB window continuation:     {orb_rate*100:.1f}%  (N={orb_n})")
    print(f"  Control window continuation: {ctrl_rate*100:.1f}%  (N={ctrl_n})")
    print(f"  ORB-specific excess:         {excess*100:+.1f} ppt")

    print()
    if excess >= 0.10:
        verdict = f"✅ GATE A STUDY 1 PASS — excess {excess*100:.1f} ppt ≥ 10 ppt; ORB window adds specific value"
    elif excess >= 0.05:
        verdict = f"⚠️  GATE A STUDY 1 MARGINAL — excess {excess*100:.1f} ppt (5–10 ppt); consider implications"
    else:
        verdict = f"🔴 GATE A STUDY 1 FAIL — excess {excess*100:.1f} ppt < 5 ppt; generic momentum only"
    print(f"  {verdict}")

    # Monthly breakdown side-by-side
    print()
    print("BY-MONTH: ORB vs. Control continuation rates")
    print(f"  {'Month':<10}  {'ORB N':>6}  {'ORB %':>7}  {'Ctrl N':>7}  {'Ctrl %':>8}  {'Excess':>8}")
    print(f"  {'─'*10}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")

    orb_m  = orb_result["monthly"]
    ctrl_m = ctrl_result["monthly"]
    all_months = sorted(set(orb_m) | set(ctrl_m))

    for ym in all_months:
        om = orb_m.get(ym, {"ext": 0, "cont": 0})
        cm = ctrl_m.get(ym, {"ext": 0, "cont": 0})
        or_  = om["cont"] / om["ext"] if om["ext"] > 0 else float("nan")
        cr_  = cm["cont"] / cm["ext"] if cm["ext"] > 0 else float("nan")
        ex_  = or_ - cr_ if (om["ext"] > 0 and cm["ext"] > 0) else float("nan")
        or_str  = f"{or_*100:.0f}%"  if om["ext"] > 0 else "—"
        cr_str  = f"{cr_*100:.0f}%"  if cm["ext"] > 0 else "—"
        ex_str  = f"{ex_*100:+.0f} ppt" if not (ex_ != ex_) else "—"
        print(f"  {ym:<10}  {om['ext']:>6}  {or_str:>7}  {cm['ext']:>7}  {cr_str:>8}  {ex_str:>8}")

    print()
    print("=" * 70)
    print("Interpretation guide:")
    print("  ORB excess ≥ 10 ppt → ORB-specific edge, proceed to Study 2")
    print("  ORB excess 5–10 ppt → marginal, review monthly stability")
    print("  ORB excess < 5 ppt  → generic momentum, ORBM-1 loses its structural argument")
    print("=" * 70)


if __name__ == "__main__":
    main()
