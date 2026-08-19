#!/usr/bin/env python3
"""§5 acceptance gate for the YANK gap-ceiling denomination prereg.

Runs G1–G4 from ``_bmad-output/preregistration_yank_gap_ceiling_denomination.md``
(sealed 2026-08-07, PR #37) — never executed before now. Read-only: never trades,
never writes to strategy_config.yaml, never touches the live bot.

The change under test: replace the absolute ``max_gap_dollars=$60`` FVG ceiling with
an ATR-relative ceiling ``max_gap_atr_ratio=0.426`` (derived, not asserted, per §3 of
the seal — median of cap_pts/H1_ATR over 2025-01-01..2026-02-28, pre-holdout).

    OLD ceiling: gap_pts * POINT_VALUE_USD <= max_gap_dollars        (60.0, absolute)
    NEW ceiling: gap_pts <= max_gap_atr_ratio * H1_ATR               (0.426, relative)

Both variants keep the SAME floor (min_gap_atr_ratio * H1_ATR) and the SAME 1-min
ATR gate (atr_threshold * calc_atr(trailing 20 1-min bars)) — only the ceiling
differs. This scans every raw 3-bar FVG shape in the derivation window and asks: for
each candidate, does OLD accept/reject agree with NEW?

Gates (§5, pre-committed):
    G1  accept/reject agreement on the non-binding sub-period (H1_ATR <= 120)  >= 97%
    G2  empty-window H1 bars under the new ceiling, whole derivation window    == 0
    G3  median accepted gap_pts shift on the non-binding sub-period           <= 10%
    G4  fvg_feasibility.py reports level=0 over the last 30 days of live bars, new cfg

Usage:
    .venv/bin/python tools/yank_gap_ceiling_g5_gate.py
    .venv/bin/python tools/yank_gap_ceiling_g5_gate.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.research.strategy_core import POINT_VALUE_USD, calc_atr, resample_to_h1  # noqa: E402
from src.research.config_loader import load_strategy_config  # noqa: E402

# §3: derivation window is strictly pre-holdout (holdout opens 2026-03-01).
DERIV_START = pd.Timestamp("2025-01-01", tz="UTC")
DERIV_END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")

# §3: the value this gate tests. Fixed by the seal's derivation procedure — not swept.
MAX_GAP_ATR_RATIO = 0.426

LIVE_CHECKOUT = Path("/root/Silver-Bullet-ML-BMAD")
BAR_SOURCES = [
    LIVE_CHECKOUT / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2025.csv",
    BASE / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2026_ytd.csv",
]
FEASIBILITY_BARS = LIVE_CHECKOUT / "logs" / "yank_shadow_parity.csv"
CONFIG_PATH = BASE / "strategy_config.yaml"


def load_1m_bars() -> pd.DataFrame:
    frames = []
    for p in BAR_SOURCES:
        if not p.exists():
            raise SystemExit(f"missing derivation-window source: {p}")
        df = pd.read_csv(p)
        lower = {c.lower(): c for c in df.columns}
        tcol = next((lower[k] for k in ("timestamp", "datetime", "date", "time") if k in lower), None)
        if tcol is None:
            raise SystemExit(f"{p}: no timestamp-like column")
        df["timestamp"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        out = pd.DataFrame({"timestamp": df["timestamp"]})
        for c in ("open", "high", "low", "close", "volume"):
            out[c] = df[lower[c]] if c in lower else 0
        frames.append(out)
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.drop_duplicates("timestamp").sort_values("timestamp")
    bars = bars[(bars["timestamp"] >= DERIV_START) & (bars["timestamp"] <= DERIV_END)]
    bars = bars.set_index("timestamp")
    bars[["open", "high", "low", "close"]] = bars[["open", "high", "low", "close"]].astype("float64")
    bars.index.name = "timestamp"
    return bars


def h1_atr_lookup(bars_1m: pd.DataFrame) -> pd.Series:
    """H1 ATR indexed by H1 bin-start, using the live calc_atr verbatim (see
    tools/fvg_feasibility.py::h1_atr_series — same construction, reused here rather
    than imported so this script has no runtime dependency beyond strategy_core)."""
    h1 = resample_to_h1(bars_1m)
    vals, idx = [], []
    for i in range(20, len(h1) + 1):
        vals.append(calc_atr(h1.iloc[i - 20:i]))
        idx.append(h1.index[i - 1])
    return pd.Series(vals, index=pd.DatetimeIndex(idx, name="timestamp"), name="h1_atr")


def scan_candidates(bars: pd.DataFrame, h1_atr: pd.Series, cfg) -> pd.DataFrame:
    """Every raw 3-bar FVG shape (before ceiling/floor filters), with the
    last-COMPLETED H1 ATR at that instant (no look-ahead: bin = t.floor('1h') - 1h,
    since the current hour's bar is still open at t)."""
    o, h, l, c = (bars["open"].to_numpy(), bars["high"].to_numpy(),
                  bars["low"].to_numpy(), bars["close"].to_numpy())
    ts = bars.index
    n = len(bars)
    rows = []
    for i in range(2, n):
        c1_low, c1_high = l[i - 2], h[i - 2]
        c2_open, c2_close = o[i - 1], c[i - 1]
        c3_low, c3_high = l[i], h[i]

        bullish = c3_low > c1_high and c2_close > c2_open
        bearish = c1_low > c3_high and c2_close < c2_open
        if not (bullish or bearish):
            continue
        top, bot = (c3_low, c1_high) if bullish else (c1_low, c3_high)
        if top <= bot:
            continue
        gap_pts = top - bot

        window_start = max(0, i - 19)
        atr1m = calc_atr(bars.iloc[window_start:i + 1])

        completed_h1_bin = ts[i].floor("1h") - pd.Timedelta(hours=1)
        atr_h1 = h1_atr.get(completed_h1_bin, float("nan"))

        rows.append((ts[i], gap_pts, atr1m, atr_h1))

    return pd.DataFrame(rows, columns=["timestamp", "gap_pts", "atr1m", "h1_atr"]).set_index("timestamp")


def accept(gap_pts, atr1m, h1_atr, cfg, *, use_new_ceiling: bool) -> bool:
    if gap_pts < cfg.atr_threshold * atr1m:
        return False
    have_h1 = h1_atr == h1_atr and h1_atr > 0  # not NaN and positive
    if use_new_ceiling and have_h1:
        if gap_pts > MAX_GAP_ATR_RATIO * h1_atr:
            return False
    else:
        if gap_pts * POINT_VALUE_USD > cfg.max_gap_dollars:
            return False
    if have_h1 and gap_pts < cfg.min_gap_atr_ratio * h1_atr:
        return False
    return True


def run_gates(cand: pd.DataFrame, cfg) -> dict:
    cand = cand.copy()
    cand["accept_old"] = [accept(g, a1, ah, cfg, use_new_ceiling=False)
                          for g, a1, ah in zip(cand.gap_pts, cand.atr1m, cand.h1_atr)]
    cand["accept_new"] = [accept(g, a1, ah, cfg, use_new_ceiling=True)
                          for g, a1, ah in zip(cand.gap_pts, cand.atr1m, cand.h1_atr)]

    have_h1 = cand["h1_atr"].notna() & (cand["h1_atr"] > 0)
    non_binding = cand[have_h1 & (cand["h1_atr"] <= 120)]

    # G1
    n_nb = len(non_binding)
    agree = int((non_binding["accept_old"] == non_binding["accept_new"]).sum()) if n_nb else 0
    g1_pct = 100.0 * agree / n_nb if n_nb else float("nan")

    # G3 — median accepted gap, non-binding sub-period, old vs new
    old_med = non_binding.loc[non_binding["accept_old"], "gap_pts"].median()
    new_med = non_binding.loc[non_binding["accept_new"], "gap_pts"].median()
    g3_shift_pct = (abs(new_med - old_med) / old_med * 100.0) if old_med and old_med == old_med else float("nan")

    return {
        "n_candidates_total": int(len(cand)),
        "n_candidates_non_binding": n_nb,
        "g1_agreement_pct": round(g1_pct, 3) if g1_pct == g1_pct else None,
        "g1_pass": bool(n_nb and g1_pct >= 97.0),
        "g3_old_median_gap_pts": round(float(old_med), 3) if old_med == old_med else None,
        "g3_new_median_gap_pts": round(float(new_med), 3) if new_med == new_med else None,
        "g3_shift_pct": round(g3_shift_pct, 3) if g3_shift_pct == g3_shift_pct else None,
        "g3_pass": bool(g3_shift_pct == g3_shift_pct and g3_shift_pct <= 10.0),
    }


def run_g2(h1_atr: pd.Series, cfg) -> dict:
    floor_pts = cfg.min_gap_atr_ratio * h1_atr
    cap_new = MAX_GAP_ATR_RATIO * h1_atr
    empty = floor_pts > cap_new
    return {
        "h1_bars_checked": int(len(h1_atr)),
        "g2_empty_count": int(empty.sum()),
        "g2_pass": bool(int(empty.sum()) == 0),
    }


def run_g4(cfg) -> dict:
    if not FEASIBILITY_BARS.exists():
        return {"g4_pass": None, "g4_note": f"missing {FEASIBILITY_BARS}"}
    from tools.fvg_feasibility import load_bars, h1_atr_series  # noqa

    bars = load_bars(FEASIBILITY_BARS)
    atr = h1_atr_series(bars)
    if atr.empty:
        return {"g4_pass": None, "g4_note": "not enough H1 bars"}
    cutoff = atr.index.max() - pd.Timedelta(days=30)
    recent = atr[atr.index >= cutoff]
    floor_pts = cfg.min_gap_atr_ratio * recent
    cap_new = MAX_GAP_ATR_RATIO * recent
    empty = floor_pts > cap_new
    pct_empty = float(empty.mean() * 100.0) if len(recent) else float("nan")
    return {
        "g4_h1_bars_last_30d": int(len(recent)),
        "g4_pct_empty": round(pct_empty, 3) if pct_empty == pct_empty else None,
        "g4_pass": bool(len(recent) and int(empty.sum()) == 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = load_strategy_config(CONFIG_PATH) if CONFIG_PATH.exists() else None
    if cfg is None:
        raise SystemExit(f"missing {CONFIG_PATH}")

    print("Loading derivation-window 1-min bars (2025-01-01..2026-02-28)...", file=sys.stderr)
    bars = load_1m_bars()
    print(f"  {len(bars)} bars, {bars.index.min()} .. {bars.index.max()}", file=sys.stderr)

    h1_atr = h1_atr_lookup(bars)
    print(f"  {len(h1_atr)} H1 ATR points", file=sys.stderr)

    print("Scanning 3-bar FVG candidates...", file=sys.stderr)
    cand = scan_candidates(bars, h1_atr, cfg)
    print(f"  {len(cand)} raw candidates", file=sys.stderr)

    g1g3 = run_gates(cand, cfg)
    g2 = run_g2(h1_atr, cfg)
    g4 = run_g4(cfg)

    result = {
        "derivation_window": [str(DERIV_START.date()), str(DERIV_END.date())],
        "max_gap_atr_ratio_under_test": MAX_GAP_ATR_RATIO,
        "live_config": {
            "atr_threshold": cfg.atr_threshold,
            "max_gap_dollars": cfg.max_gap_dollars,
            "min_gap_atr_ratio": cfg.min_gap_atr_ratio,
        },
        **g1g3,
        **g2,
        **g4,
    }
    result["all_pass"] = bool(result["g1_pass"] and result["g2_pass"] and result["g3_pass"]
                              and result.get("g4_pass"))

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print("=== §5 acceptance gate — YANK gap-ceiling denomination (max_gap_atr_ratio=0.426) ===")
        print(f"derivation window: {result['derivation_window'][0]} .. {result['derivation_window'][1]}")
        print(f"candidates scanned: {result['n_candidates_total']} total, "
              f"{result['n_candidates_non_binding']} in the non-binding sub-period (H1_ATR<=120)")
        print()
        print(f"G1 (agreement >= 97%)         : {result['g1_agreement_pct']}%  "
              f"-> {'PASS' if result['g1_pass'] else 'FAIL'}")
        print(f"G2 (empty H1 bars == 0)        : {result['g2_empty_count']} / {result['h1_bars_checked']}  "
              f"-> {'PASS' if result['g2_pass'] else 'FAIL'}")
        print(f"G3 (median gap shift <= 10%)   : old={result['g3_old_median_gap_pts']}pts "
              f"new={result['g3_new_median_gap_pts']}pts shift={result['g3_shift_pct']}%  "
              f"-> {'PASS' if result['g3_pass'] else 'FAIL'}")
        g4_line = (f"{result.get('g4_pct_empty')}% empty over {result.get('g4_h1_bars_last_30d')} H1 bars"
                  if result.get('g4_pass') is not None else result.get('g4_note', 'could not run'))
        print(f"G4 (fvg_feasibility level=0)   : {g4_line}  "
              f"-> {'PASS' if result.get('g4_pass') else ('N/A' if result.get('g4_pass') is None else 'FAIL')}")
        print()
        print(f"VERDICT: {'ALL GATES PASS — H1 CONFIRMED (Response A authorized by evidence)' if result['all_pass'] else 'AT LEAST ONE GATE FAILED — H1 REJECTED, pre-committed outcome is Response B (leave the gate alone)'}")

    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
