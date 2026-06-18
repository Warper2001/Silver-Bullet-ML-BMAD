#!/usr/bin/env python3
"""Aggregate the shadow-parity log into the ProjectX data-feed cutover gate.

Reads a shadow-parity CSV (written live by the bots while TradeStation stays the
signal source) and reports, per ET session + overall: OHLC median |Δ| & %-within-tick,
volume mean |Δ|, coverage gaps (ts_only / px_only), fetch-error rate, and fetch latency
p50/p95. Ports the verdict + contract-mismatch logic from tools/bar_parity_probe.py.

This is the read-only evidence summary that gates Stage-2 cutover (per the plan):
OHLC median ≤ 1 tick AND ≥95% within tick AND no sustained coverage gaps AND low
fetch-error rate, across N≥10 live sessions, per bot.

Usage:
    .venv/bin/python tools/analyze_shadow_parity.py [--csv logs/yank_shadow_parity.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
TICK = 0.25
BASE = Path(__file__).resolve().parent.parent


def _verdict_line(g: pd.DataFrame) -> tuple[str, dict]:
    both = g[g["coverage"] == "both"]
    n_both = len(both)
    m = {
        "minutes": len(g),
        "both": n_both,
        "ts_only": int((g["coverage"] == "ts_only").sum()),
        "px_only": int((g["coverage"] == "px_only").sum()),
        "err": int((g["px_error"].fillna("") != "").sum()),
        "lat_p50": float(np.nanpercentile(g["px_fetch_ms"], 50)) if len(g) else float("nan"),
        "lat_p95": float(np.nanpercentile(g["px_fetch_ms"], 95)) if len(g) else float("nan"),
    }
    if n_both:
        ohlc = pd.concat([both[c].abs() for c in ("d_open", "d_high", "d_low", "d_close")])
        m["ohlc_median"] = float(ohlc.median())
        m["within_tick"] = float((ohlc <= TICK).mean())
        m["ohlc_max"] = float(ohlc.max())
        m["vol_mean"] = float(both["d_vol"].abs().mean())
        m["close_signed_mean"] = float(both["d_close"].mean())
        m["close_signed_std"] = float(both["d_close"].std() or 0.0)
    else:
        m.update(ohlc_median=float("nan"), within_tick=float("nan"), ohlc_max=float("nan"),
                 vol_mean=float("nan"), close_signed_mean=float("nan"), close_signed_std=float("nan"))
    return "", m


def classify(df: pd.DataFrame) -> tuple[str, str]:
    """Machine-readable status for alerting:
      PASS         — quality holds AND ≥10 sessions (ready to pre-register cutover)
      PROBLEM      — parity degraded (median > tick / <95% within tick / coverage gaps /
                     fetch errors / contract mismatch) — worth a look
      ACCUMULATING — quality holds, still <10 sessions (no action; silent in daily alert)
      NODATA       — no matched bars yet
    Returns (status, one_line_reason)."""
    _, A = _verdict_line(df)
    n_sessions = df["session"].nunique()
    if not A["both"]:
        return "NODATA", "no matched (both-feed) minutes yet"
    gap_frac = (A["ts_only"] + A["px_only"]) / max(A["minutes"], 1)
    err_frac = A["err"] / max(A["minutes"], 1)
    if A["ohlc_median"] > 20 and abs(A["close_signed_mean"]) > 20 \
            and A["close_signed_std"] < abs(A["close_signed_mean"]) * 0.5:
        return "PROBLEM", f"contract mismatch (close gap ≈ {A['close_signed_mean']:+.1f}pt)"
    if A["ohlc_median"] > TICK:
        return "PROBLEM", f"OHLC median {A['ohlc_median']:.3f} > {TICK} tick"
    if A["within_tick"] < 0.95:
        return "PROBLEM", f"only {A['within_tick']*100:.1f}% within 1 tick (<95%)"
    if gap_frac > 0.01:
        return "PROBLEM", f"coverage gaps {gap_frac*100:.1f}% (ts_only {A['ts_only']}, px_only {A['px_only']})"
    if err_frac > 0.02:
        return "PROBLEM", f"fetch-error rate {err_frac*100:.1f}%"
    if n_sessions >= 10:
        return "PASS", f"{n_sessions} sessions, median {A['ohlc_median']:.3f}, {A['within_tick']*100:.1f}% within tick"
    return "ACCUMULATING", f"{n_sessions}/10 sessions, parity clean"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=BASE / "logs" / "yank_shadow_parity.csv")
    ap.add_argument("--status", action="store_true",
                    help="print just a machine-readable status line (PASS/PROBLEM/ACCUMULATING/NODATA) and exit")
    args = ap.parse_args()
    if not args.csv.exists():
        if args.status:
            print("NODATA | no shadow log yet")
        else:
            print(f"No shadow-parity log yet at {args.csv} — nothing to analyze.")
        return 0

    df = pd.read_csv(args.csv)
    if df.empty:
        print("NODATA | empty log" if args.status else f"{args.csv} is empty.")
        return 0
    for c in ("d_open", "d_high", "d_low", "d_close", "d_vol", "ohlc_max_abs", "px_fetch_ms"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["session"] = (pd.to_datetime(df["minute"], utc=True).dt.tz_convert(ET).dt.date)

    if args.status:
        st, reason = classify(df)
        print(f"{st} | {reason}")
        return 0

    print("=" * 90)
    print(f"Shadow parity — {args.csv}  ({len(df)} minutes, {df['session'].nunique()} sessions)")
    print("=" * 90)
    hdr = f"{'session':12s} {'min':>5s} {'both':>5s} {'tsOnly':>6s} {'pxOnly':>6s} {'medΔ':>6s} {'%tick':>6s} {'maxΔ':>6s} {'volΔ':>6s} {'err':>4s} {'lat_p95':>8s}"
    print(hdr)
    for sess, g in df.groupby("session"):
        _, m = _verdict_line(g)
        med = m["ohlc_median"]; wt = m["within_tick"]
        print(f"{str(sess):12s} {m['minutes']:5d} {m['both']:5d} {m['ts_only']:6d} {m['px_only']:6d} "
              f"{med:6.3f} {wt*100:5.1f}% {m['ohlc_max']:6.2f} {m['vol_mean']:6.1f} {m['err']:4d} {m['lat_p95']:7.0f}ms")

    print("-" * 90)
    _, A = _verdict_line(df)
    print("OVERALL:")
    print(f"  OHLC abs-diff: median {A['ohlc_median']:.4f} | {A['within_tick']*100:.1f}% within 1 tick | max {A['ohlc_max']:.2f}")
    print(f"  Volume mean |Δ|: {A['vol_mean']:.1f} contracts   Coverage gaps: ts_only {A['ts_only']}, px_only {A['px_only']}")
    print(f"  Fetch errors: {A['err']}/{A['minutes']}   latency p50 {A['lat_p50']:.0f}ms / p95 {A['lat_p95']:.0f}ms")

    # contract-mismatch guard (a large near-constant signed close gap = wrong front month)
    if A["both"] and A["ohlc_median"] > 20 and abs(A["close_signed_mean"]) > 20 \
            and A["close_signed_std"] < abs(A["close_signed_mean"]) * 0.5:
        print(f"\n  ⛔ LIKELY CONTRACT MISMATCH — close gap ≈ {A['close_signed_mean']:+.1f}pt, near-constant.")
        print("     The PX contract differs from the TS symbol (roll mismatch); not a feed divergence.")
        return 0

    n_sessions = df["session"].nunique()
    gate = (A["both"] and A["ohlc_median"] <= TICK and A["within_tick"] >= 0.95
            and A["ts_only"] == 0 and A["px_only"] == 0 and A["err"] == 0 and n_sessions >= 10)
    print("\n  GATE (median ≤ tick, ≥95% within tick, no gaps, no errors, ≥10 sessions): "
          + ("PASS — ready to pre-register the cutover." if gate else "NOT YET MET (keep accumulating / investigate flags above)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
