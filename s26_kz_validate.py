"""
S26 Kill-Zone Filter Validation — Pre-Registration Baseline
===========================================================
Applies the S26 pre-committed filter to the S23 in-sample labeled trade dataset
(2025 pre-cutoff data) to establish the baseline N / PF / Sharpe that appear
in the S26 pre-registration document.

S26 filter definition (exact, locked):
  hour_et in [10, 11, 14]   →  10:00-12:00 ET (NY AM) + 14:00-15:00 ET (NY PM)
  dow_et != 0                →  exclude Monday (0=Mon, 1=Tue already blocked in S25)

No holdout data is accessed. This is purely an in-sample characterization of the
filter rule before any live trades are observed.

Run:
  .venv/bin/python s26_kz_validate.py
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH  = Path("data/ml_training/s23_meta_labels_2025.csv")
REPORT_DIR  = Path("data/reports")
CONTRACTS   = 5
MNQ_DOLLAR  = 2.0

# ── S26 filter (pre-committed, identical to preregistration doc) ──────────────

KZ_HOURS = {10, 11, 14}   # 10:00-12:00 ET (NY AM extended) + 14:00-15:00 ET (NY PM)
BLOCKED_DOW = {0}          # Monday (0). Tuesday (1) already blocked by S25 system.

def is_s26_eligible(hour_et: int, dow_et: int) -> bool:
    return hour_et in KZ_HOURS and dow_et not in BLOCKED_DOW


# ── Statistics helpers ─────────────────────────────────────────────────────────

def profit_factor(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    return float(gp / gl) if gl > 0 else float("inf")

def annualized_sharpe(pnl: pd.Series) -> float:
    if len(pnl) < 2:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(len(pnl)))

def stats_block(label: str, sub: pd.DataFrame, total_n: int) -> str:
    n = len(sub)
    if n == 0:
        return f"  {label:<40s}  N=0\n"
    pf    = profit_factor(sub["pnl_1x"])
    wr_tp = (sub["label"] == 1).mean() * 100
    wr_pnl= (sub["pnl_1x"] > 0).mean() * 100
    avg   = sub["pnl_1x"].mean()
    total = sub["pnl_1x"].sum()
    sh    = annualized_sharpe(sub["pnl_1x"])
    pct   = n / total_n * 100
    exits = sub["exit_type"].value_counts().to_dict()
    ann_pnl_5c = total * CONTRACTS
    return (
        f"  {label:<40s}  N={n:>3d} ({pct:.0f}% of all)  "
        f"PF={pf:>6.4f}  WR_TP={wr_tp:>5.1f}%  WR_pnl={wr_pnl:>5.1f}%  "
        f"AvgPnL={avg:>+8.2f}  AnnPnL_1x={total:>+8.0f}  "
        f"AnnPnL_5c=${ann_pnl_5c:>+8.0f}  Sharpe≈{sh:>5.2f}  Exits={exits}\n"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}. Run s23_meta_label_features.py first.")

    df = pd.read_csv(INPUT_PATH, parse_dates=["entry_ts", "exit_ts"])
    df = df.sort_values("entry_ts").reset_index(drop=True)
    n_total = len(df)

    # Apply S26 filter
    s26_mask = df.apply(lambda r: is_s26_eligible(int(r["hour_et"]), int(r["dow_et"])), axis=1)
    s26  = df[s26_mask]
    excl = df[~s26_mask]

    lines = []
    lines.append("=" * 100)
    lines.append("S26 Kill-Zone Filter — Pre-Registration Baseline Validation")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source:    {INPUT_PATH}  ({n_total} in-sample trades, 2025 pre-cutoff)")
    lines.append("=" * 100)
    lines.append("")

    lines.append("S26 FILTER DEFINITION (pre-committed):")
    lines.append("  Kill zones:   hour_et in {10, 11, 14}  →  10:00-12:00 ET + 14:00-15:00 ET")
    lines.append("  Blocked days: dow_et == 0 (Monday)   [Tuesday already blocked by S25 system]")
    lines.append("")

    lines.append("OVERALL BASELINE (all in-sample trades, no filter):")
    lines.append(stats_block("All trades (S25 equivalent)", df, n_total).rstrip())
    lines.append("")

    lines.append("S26 SUBGROUP:")
    lines.append(stats_block("S26 eligible  (KZ + not Mon)", s26, n_total).rstrip())
    lines.append(stats_block("S26 excluded  (non-KZ or Mon)", excl, n_total).rstrip())
    lines.append("")

    lines.append("KILL-ZONE BREAKDOWN (within S26 eligible):")
    for h in sorted(KZ_HOURS):
        sub = s26[s26["hour_et"] == h]
        lines.append(stats_block(f"  hour_et={h:02d}:00-{h+1:02d}:00 ET", sub, n_total).rstrip())
    lines.append("")

    lines.append("DAY-OF-WEEK BREAKDOWN (S26 eligible trades only):")
    dow_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri"}
    for d in sorted(s26["dow_et"].unique()):
        sub = s26[s26["dow_et"] == d]
        lines.append(stats_block(f"  {dow_map.get(int(d), str(d))}", sub, n_total).rstrip())
    lines.append("")

    # Key numbers for pre-registration doc
    s26_pf = profit_factor(s26["pnl_1x"])
    s26_sh = annualized_sharpe(s26["pnl_1x"])
    s26_wr_tp  = (s26["label"] == 1).mean() * 100
    s26_wr_pnl = (s26["pnl_1x"] > 0).mean() * 100
    s26_ann_1x = s26["pnl_1x"].sum()
    s26_ann_5c = s26_ann_1x * CONTRACTS
    s26_trades_per_day = len(s26) / 252.0

    lines.append("=" * 100)
    lines.append("PRE-REGISTRATION REFERENCE NUMBERS (copy into preregistration_s26_kz_filtered_live.md)")
    lines.append("=" * 100)
    lines.append(f"  In-sample year:              2025 (pre-cutoff, {n_total} total trades)")
    lines.append(f"  S26 eligible trades (N):     {len(s26)}")
    lines.append(f"  Profit Factor:               {s26_pf:.4f}")
    lines.append(f"  Win Rate (TP hit):           {s26_wr_tp:.1f}%")
    lines.append(f"  Win Rate (pnl > 0):          {s26_wr_pnl:.1f}%")
    lines.append(f"  Annualized Sharpe (proxy):   {s26_sh:.2f}")
    lines.append(f"  Annual P&L (1x, raw):        ${s26_ann_1x:+.0f}")
    lines.append(f"  Annual P&L (5 contracts):    ${s26_ann_5c:+.0f}")
    lines.append(f"  Avg trades/day (filtered):   {s26_trades_per_day:.2f}")
    lines.append(f"  Avg trades/month (filtered): {s26_trades_per_day * 21:.1f}")
    lines.append(f"  Exit breakdown:              {s26['exit_type'].value_counts().to_dict()}")
    lines.append("")
    lines.append("  S25 equivalent (unfiltered):  N=109, PF=1.1656, Sharpe≈0.37")
    lines.append(f"  S26 PF lift vs S25:          +{s26_pf - 1.1656:.4f} PF points")
    lines.append(f"  S26 Sharpe lift vs S25:      +{s26_sh - 0.37:.2f}")
    lines.append("")
    lines.append("  Expected live frequency at filtered rate:")
    live_per_day = s26_trades_per_day * (3/5)  # conservative: live fires ~60% of backtest rate
    lines.append(f"    Conservative (60% of backtest): {live_per_day:.2f}/day → N=20 in ~{int(20/live_per_day)+1} days")
    lines.append(f"    Backtest rate:                  {s26_trades_per_day:.2f}/day → N=20 in ~{int(20/s26_trades_per_day)+1} days")
    lines.append("=" * 100)

    report = "\n".join(lines)
    print(report)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORT_DIR / f"s26_kz_validate_{ts}.txt"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    main()
