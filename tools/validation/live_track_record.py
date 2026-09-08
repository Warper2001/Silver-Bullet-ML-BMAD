"""How long until a LIVE track record could tell you anything?

WHY THIS EXISTS
---------------
A backtest Sharpe is selected -- it survived a search, so it needs deflating.
A *live, prospective* record is not: nobody chose it after seeing it, so N=1 and
no multiple-testing correction applies. The question changes from "is this real?"
to "is the record long enough to distinguish from zero yet?"

Minimum Track Record Length answers exactly that:

    MinTRL = 1 + (1 - g3*SR + (g4-1)/4 * SR^2) * (Z^-1(a) / (SR - SR*))^2

giving the number of observations needed before a Sharpe of this size, with
these fat tails and this skew, is significantly above zero.

READ-ONLY, AND IT MATTERS
-------------------------
data/trades.db is a live trading control. This tool opens it with
`mode=ro` and only ever SELECTs.

BACKFILL CONTAMINATION
----------------------
The ledger mixes real prospective trades (`write_mode='realtime'`) with backtest
replays written into it (`write_mode='backfilled'`). Summing pnl over the whole
table gives a number that is mostly backtest. This tool excludes backfilled rows
from the live statistics and prints what it excluded, because that exclusion is
the single most important step in the calculation.

USAGE
-----
    .venv-research/bin/python tools/validation/live_track_record.py
    .venv-research/bin/python tools/validation/live_track_record.py \
        --db /root/Silver-Bullet-ML-BMAD/data/trades.db --min-trades 10
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deflated_sharpe import (  # noqa: E402
    min_track_record_length,
    probabilistic_sharpe,
    sharpe_moments,
)

DEFAULT_DB = "/root/Silver-Bullet-ML-BMAD/data/trades.db"


def load(db: str) -> pd.DataFrame:
    if not Path(db).exists():
        sys.exit(f"ERROR: {db} not found.")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        df = pd.read_sql(
            "SELECT trader_id, timestamp, pnl, write_mode, execution_mode FROM trades", con)
    finally:
        con.close()
    # format="ISO8601" is REQUIRED, not cosmetic. This ledger mixes
    # '2026-06-11T13:30:00+00:00' with '2026-06-24T18:00:03.542998+00:00', and
    # pandas >= 2 infers a single format from the FIRST row, then coerces every
    # row that does not match it to NaT. With errors="coerce" that is silent:
    # it discarded 23 of trader-mim-nb's 24 rows before this was caught.
    raw_n = len(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce",
                                     utc=True, format="ISO8601")
    bad = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp", "pnl"])
    if bad:
        print(f"WARNING: {bad} of {raw_n} rows have an unparseable timestamp "
              f"and were dropped. Investigate before trusting these numbers.\n")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--min-trades", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.95)
    args = ap.parse_args()

    df = load(args.db)
    print(f"Ledger: {args.db}  ({len(df)} rows with a usable timestamp and pnl)\n")

    # ---- backfill exclusion, reported loudly --------------------------------
    is_backfill = df["write_mode"].eq("backfilled")
    bf = df[is_backfill]
    print("=" * 78)
    print("EXCLUDED: backfilled rows (backtest replays written into the live ledger)")
    print("=" * 78)
    if bf.empty:
        print("  none")
    else:
        g = bf.groupby("trader_id")["pnl"].agg(["count", "sum"]).sort_values("count", ascending=False)
        for tid, r in g.iterrows():
            print(f"  {tid:<22s} {int(r['count']):5d} rows   pnl {r['sum']:+12,.2f}  <- NOT live P&L")
        print(f"  {'TOTAL EXCLUDED':<22s} {len(bf):5d} rows   pnl {bf['pnl'].sum():+12,.2f}")
    rest = df[~is_backfill].copy()

    # A NULL write_mode means one of two very different things, and the date
    # separates them cleanly: recent rows written after the column stopped being
    # populated (genuinely live), versus legacy rows that predate the column
    # entirely (old paper/backtest output, no symbol, no execution_mode).
    # Rule: a NULL row counts as prospective only if it falls at or after that
    # trader's FIRST explicitly-'realtime' row. Without this, trader-yank's 8
    # rows from May-June 2025 get counted as live trades.
    first_rt = (rest[rest["write_mode"].eq("realtime")]
                .groupby("trader_id")["timestamp"].min())
    is_null = rest["write_mode"].isna()
    era = rest["trader_id"].map(first_rt)
    legacy = is_null & (era.isna() | (rest["timestamp"] < era))
    lg = rest[legacy]
    print("\n" + "=" * 78)
    print("EXCLUDED: NULL-write_mode rows that PREDATE the trader's realtime era")
    print("=" * 78)
    if lg.empty:
        print("  none")
    else:
        for tid, gg in lg.groupby("trader_id"):
            print(f"  {tid:<22s} {len(gg):5d} rows   pnl {gg['pnl'].sum():+12,.2f}   "
                  f"{str(gg['timestamp'].min())[:10]} -> {str(gg['timestamp'].max())[:10]}"
                  f"   (legacy, pre-column)")
    live = rest[~legacy].copy()
    kept_null = int((live["write_mode"].isna()).sum())
    print(f"\nProspective rows: {len(live)}  "
          f"({kept_null} of them NULL-mode but inside the realtime era, so kept)\n")

    # ---- per-strategy prospective statistics --------------------------------
    rows = []
    for tid, g in live.groupby("trader_id"):
        daily = g.set_index("timestamp")["pnl"].groupby(lambda t: t.date()).sum().sort_index()
        if len(daily) < 2:
            continue
        # zero-fill non-trading business days inside the strategy's own window
        full = pd.date_range(daily.index.min(), daily.index.max(), freq="B").date
        daily = daily.reindex(full).fillna(0.0)
        r = daily.to_numpy(float)
        sr, g3, g4, T = sharpe_moments(r)
        if not np.isfinite(sr):
            continue
        mtrl = min_track_record_length(sr, g3, g4, 0.0, args.alpha)
        modes = ",".join(sorted(set(g["execution_mode"].dropna().astype(str)))) or "unset"
        rows.append({
            "trader": tid, "mode": modes, "trades": len(g), "days": T,
            "total_pnl": g["pnl"].sum(),
            "sr_daily": sr, "sr_ann": sr * np.sqrt(252),
            "skew": g3, "kurt": g4,
            "psr_vs_zero": probabilistic_sharpe(sr, g3, g4, T, 0.0),
            # sensitivity: PSR with normality imposed instead of the ESTIMATED
            # skew/kurtosis. On a short record those moments are themselves
            # badly estimated, and PSR treats them as known. A verdict that
            # holds under one and not the other rests on 3rd/4th moments
            # measured from too few observations to believe.
            "psr_normal": probabilistic_sharpe(sr, 0.0, 3.0, T, 0.0),
            "min_trl_days": mtrl,
            "days_short": (mtrl - T) if np.isfinite(mtrl) else np.inf,
            "moments_ok": T >= 100,
        })

    if not rows:
        sys.exit("No strategy has enough prospective history to evaluate.")
    out = pd.DataFrame(rows).sort_values("trades", ascending=False)
    keep = out[out["trades"] >= args.min_trades]
    skipped = out[out["trades"] < args.min_trades]

    print("=" * 78)
    print(f"PROSPECTIVE (live/paper) RECORDS -- N=1, no selection, so NO deflation applies")
    print("=" * 78)
    pd.set_option("display.width", 200)
    cols = ["trader", "mode", "trades", "days", "total_pnl", "sr_ann", "skew", "kurt",
            "psr_vs_zero", "psr_normal", "min_trl_days", "days_short", "moments_ok"]
    print(keep[cols].to_string(index=False, float_format=lambda v: f"{v:12.3f}"))
    if not skipped.empty:
        print(f"\n(skipped, fewer than {args.min_trades} trades: "
              f"{', '.join(f'{r.trader}={r.trades}' for r in skipped.itertuples())})")

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("  psr_vs_zero   P(true Sharpe > 0) given this record's length, skew and tails.")
    print("                0.95 is the usual bar. 0.50 means a coin flip.")
    print("  min_trl_days  business days of record needed to clear that bar at "
          f"{args.alpha:.0%}.")
    print("  psr_normal    the same, with normality imposed instead of the ESTIMATED")
    print("                skew/kurtosis -- a sensitivity check.")
    print("  moments_ok    False when the record has < 100 days, in which case the")
    print("                skew and kurtosis PSR relies on are themselves guesses and")
    print("                any verdict that moves between psr_vs_zero and psr_normal")
    print("                is resting on them.")
    print("  days_short    how many more days are needed. inf = the Sharpe is <= 0,")
    print("                so no amount of further record will establish it.\n")
    for r in keep.itertuples():
        if r.psr_vs_zero >= args.alpha and r.psr_normal >= args.alpha and r.moments_ok:
            print(f"  {r.trader:<22s} ESTABLISHED (PSR {r.psr_vs_zero:.3f}) on {r.days} days")
        elif r.psr_vs_zero >= args.alpha:
            why = ("record < 100 days, so its skew/kurtosis are unreliable"
                   if not r.moments_ok else "")
            if r.psr_normal < args.alpha:
                why = (why + "; " if why else "") + \
                      f"verdict flips under normality (PSR {r.psr_normal:.3f})"
            print(f"  {r.trader:<22s} PROVISIONAL -- PSR {r.psr_vs_zero:.3f} clears the bar "
                  f"but {why}")
        elif np.isfinite(r.days_short):
            print(f"  {r.trader:<22s} NOT YET -- PSR {r.psr_vs_zero:.3f}; needs "
                  f"{r.min_trl_days:,.0f} days, has {r.days} "
                  f"({r.days_short:,.0f} more, ~{r.days_short / 252:.1f} yr)")
        else:
            print(f"  {r.trader:<22s} NEGATIVE Sharpe -- no further record establishes it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
