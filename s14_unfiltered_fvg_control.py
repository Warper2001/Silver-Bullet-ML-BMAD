"""
S14 Unfiltered FVG Control — Program C Phase 2 Pre-Decision Test
Pre-registration: _bmad-output/preregistration_s14_unfiltered_fvg.md
Access token: --preregistration dbfa46f

Tests whether removing the H1 sweep requirement produces a better result on
the sealed holdout than the filtered strategy (S12 real PF 1.2154, 96 trades).

Decision tree (from pre-registration):
  S14 PF ≤ 1.1350 (S12 p90 random)   → no_unfiltered_edge
  S14 PF > 1.1350 AND ≤ 1.2154       → edge_exists_stay_filtered
  S14 PF > 1.2154 AND trades ≥ 80    → high_frequency_pivot_approved

Usage:
    .venv/bin/python s14_unfiltered_fvg_control.py --preregistration dbfa46f
"""

import argparse
import csv
import fnmatch
import re
import subprocess
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── Frozen parameters (preregistration_s14_unfiltered_fvg.md) ───────────────
SL_MULT           = 5.0
TP_MULT           = 6.0
MAX_HOLD          = 60
MAX_PENDING_BARS  = 240
ATR_PERIOD        = 20
VOL_LOOKBACK      = 120
VOL_THRESH        = 0.75
ATR_THRESHOLD     = 0.5
MIN_GAP_ATR_RATIO = 0.15
MAX_GAP_DOLLARS   = 60.0
MNQ_DOLLAR        = 2.0
ENTRY_PCT         = 0.5
MNQ_TICK          = 0.25

# ── S12 reference values (pre-committed; do not change after observing S14 results)
S12_P90_RANDOM  = 1.1350   # 90th-pct of 50-seed random baseline
S12_REAL_PF     = 1.2154   # filtered strategy PF on same holdout
S12_REAL_TRADES = 96       # filtered strategy trade count
HF_MIN_TRADES   = 80       # minimum trades for high_frequency_pivot_approved

HOLDOUT_CSV      = Path("data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv")
HOLDOUT_CUTOFF   = datetime(2026, 3, 1, tzinfo=timezone.utc)
ACCESS_LOG_PATH  = Path("data/sealed_holdout/ACCESS_LOG.md")
REPORTS_DIR      = Path("data/reports")
_PREREG_PATTERNS = (
    "_bmad-output/preregistration*.md",
    "data/sealed_holdout/PREREGISTRATION*.md",
)
ET_TZ = pytz.timezone("US/Eastern")


# ── Access log gate (verbatim from s12_random_entry_control.py) ──────────────

def _exit_access_denied(msg: str) -> None:
    print(f"\n{'='*70}", file=sys.stderr)
    print("SEALED HOLDOUT ACCESS DENIED", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(msg, file=sys.stderr)
    print(f"\nSee {ACCESS_LOG_PATH} for the full access protocol.", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    sys.exit(1)


def verify_preregistration(sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{4,40}", sha, re.IGNORECASE):
        _exit_access_denied(
            f"'{sha}' is not a valid git SHA (expected 4-40 hex characters).\n"
            "Commit your pre-registration document first, then supply that commit's SHA."
        )
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", sha],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        _exit_access_denied("'git' is not on PATH — cannot verify pre-registration.")
    if result.returncode != 0 or result.stdout.strip() != "commit":
        _exit_access_denied(
            f"SHA '{sha}' was not found in the local git repo (or is not a commit).\n"
            "Commit your pre-registration document first, then supply that commit's SHA."
        )
    files_result = subprocess.run(
        ["git", "show", sha, "--name-only", "--format="],
        capture_output=True, text=True, check=False,
    )
    file_lines = [ln.strip() for ln in files_result.stdout.splitlines() if ln.strip()]
    found = any(
        fnmatch.fnmatch(f, pattern)
        for f in file_lines
        for pattern in _PREREG_PATTERNS
    )
    if not found:
        _exit_access_denied(
            f"Commit {sha[:8]} contains no pre-registration document.\n"
            "Expected a file matching one of:\n"
            + "\n".join(f"  {p}" for p in _PREREG_PATTERNS)
            + "\n\nFiles in that commit:\n"
            + ("\n".join(f"  {f}" for f in file_lines[:20]) or "  (none)")
        )


def append_access_log(sha: str, argv: list) -> None:
    if not ACCESS_LOG_PATH.exists():
        _exit_access_denied(
            f"{ACCESS_LOG_PATH} does not exist.\n"
            "Run Program C Phase 0.4 to establish the sealed holdout first."
        )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    argv_str = " ".join(argv[1:]).replace("|", "\\|")
    log_row  = f"| {date_str} | {sha} | s14 script | `{argv_str}` | pending |\n"
    lines = ACCESS_LOG_PATH.read_text().splitlines(keepends=True)
    last_table_idx = max(
        (i for i, ln in enumerate(lines) if ln.startswith("|")), default=None
    )
    if last_table_idx is not None:
        lines.insert(last_table_idx + 1, log_row)
    else:
        lines.append(log_row)
    ACCESS_LOG_PATH.write_text("".join(lines))
    print(f"Access recorded → {ACCESS_LOG_PATH}  (SHA {sha[:12]})")
    print("  ACTION REQUIRED: commit ACCESS_LOG.md to make this record permanent.")


# ── Bar types and loading (verbatim from s12) ─────────────────────────────────

Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close"])


def load_holdout_bars() -> list:
    if not HOLDOUT_CSV.exists():
        print(f"ERROR: holdout CSV not found: {HOLDOUT_CSV}", file=sys.stderr)
        sys.exit(1)
    bars = []
    with open(HOLDOUT_CSV) as f:
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


# ── ATR, H1, vol regime (verbatim from s12) ───────────────────────────────────

def calc_atr(bars: list, end_idx: int, period: int = ATR_PERIOD) -> float:
    if end_idx < period + 1:
        return 10.0
    trs = []
    for i in range(end_idx - period, end_idx):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs))


def build_h1_df(bars: list, up_to_idx: int) -> pd.DataFrame:
    """Resample last 3000 bars up to up_to_idx to 1-hour OHLC; return completed bars."""
    if up_to_idx < 2:
        return pd.DataFrame()
    cap = bars[max(0, up_to_idx - 3000):up_to_idx]
    df = pd.DataFrame({
        "timestamp": [b.timestamp for b in cap],
        "open":  [b.open  for b in cap],
        "high":  [b.high  for b in cap],
        "low":   [b.low   for b in cap],
        "close": [b.close for b in cap],
    }).set_index("timestamp")
    h1 = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if len(h1) < 2:
        return pd.DataFrame()
    return h1.iloc[:-1].reset_index()


def calc_h1_atr(h1_df: pd.DataFrame) -> float:
    if len(h1_df) < ATR_PERIOD + 1:
        return 0.0
    h = h1_df["high"].values
    l = h1_df["low"].values
    c = h1_df["close"].values
    trs = [
        max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        for i in range(len(h1_df) - ATR_PERIOD, len(h1_df))
    ]
    return float(np.mean(trs))


class VolRegimeTracker:
    def __init__(self):
        self._history: list = []
        self.current_h1_atr: float = 0.0

    def update(self, h1_df: pd.DataFrame) -> bool:
        if len(h1_df) < 2:
            return False
        atr = calc_h1_atr(h1_df)
        self.current_h1_atr = atr
        if atr <= 0:
            return False
        self._history.append(atr)
        if len(self._history) > VOL_LOOKBACK:
            self._history.pop(0)
        if len(self._history) < 20:
            return False
        pct_rank = sum(v < atr for v in self._history) / len(self._history)
        return pct_rank > VOL_THRESH


def is_market_open(ts: datetime) -> bool:
    wd, h = ts.weekday(), ts.hour
    if wd == 5: return False
    if wd == 6: return h >= 23
    if wd == 4: return h < 22
    return h != 22


def is_tuesday_et(ts: datetime) -> bool:
    return ts.astimezone(ET_TZ).weekday() == 1


def profit_factor(pnl: list) -> float:
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    return gp / gl if gl > 0 else float("inf")


def snap_tick(price: float) -> float:
    return round(round(price / MNQ_TICK) * MNQ_TICK, 4)


# ── S14 strategy: FVG only, no H1 sweep required ─────────────────────────────

def run_unfiltered_fvg(bars: list):
    """
    Bearish FVG strategy with H1 sweep requirement REMOVED.
    H1 structure still computed for vol regime gate.
    Returns (pf: float | None, n_trades: int, pnl: list).
    """
    vol_tracker = VolRegimeTracker()
    last_h1_boundary = None
    regime_high = False

    pending      = False
    active       = False
    entry_price  = sl_price = tp_price = 0.0
    bars_pending = bars_held = 0
    pnl          = []
    n_trades     = 0

    for i, bar in enumerate(bars):
        h1_boundary = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        if h1_boundary != last_h1_boundary:
            last_h1_boundary = h1_boundary
            h1_df       = build_h1_df(bars, i)
            regime_high = vol_tracker.update(h1_df)
            # NOTE: No sweep detection — this is the only difference from S12

        # Advance active trade
        if active:
            bars_held += 1
            if bar.high >= sl_price:
                pnl.append((entry_price - sl_price) * MNQ_DOLLAR)
                active = False; n_trades += 1
            elif bar.low <= tp_price:
                pnl.append((entry_price - tp_price) * MNQ_DOLLAR)
                active = False; n_trades += 1
            elif bars_held >= MAX_HOLD:
                pnl.append((entry_price - bar.close) * MNQ_DOLLAR)
                active = False; n_trades += 1
            continue

        # Advance pending limit order
        if pending:
            bars_pending += 1
            if bar.high >= entry_price:
                active = True; pending = False; bars_held = 0
            elif bars_pending >= MAX_PENDING_BARS:
                pending = False
            continue

        # Gate: look for new signal
        if i < 2:
            continue
        if not is_market_open(bar.timestamp):
            continue
        if is_tuesday_et(bar.timestamp):
            continue
        if regime_high:
            continue
        # ← sweep_active check deliberately absent (S14 pre-registration §What Changes)

        # Bearish 3-bar FVG
        c1, c2, c3 = bars[i-2], bars[i-1], bar
        if not (c1.low > c3.high and c2.close < c2.open):
            continue
        gap = c1.low - c3.high
        if gap <= 0:
            continue
        if gap < ATR_THRESHOLD * calc_atr(bars, i + 1):
            continue
        if gap * MNQ_DOLLAR > MAX_GAP_DOLLARS:
            continue
        h1_atr = vol_tracker.current_h1_atr
        if h1_atr > 0 and gap < MIN_GAP_ATR_RATIO * h1_atr:
            continue

        entry_price  = snap_tick(c3.high + gap * ENTRY_PCT)
        sl_price     = snap_tick(entry_price + gap * SL_MULT)
        tp_price     = snap_tick(entry_price - gap * TP_MULT)
        pending      = True
        bars_pending = 0

    pf = profit_factor(pnl) if pnl else None
    return pf, n_trades, pnl


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(pf, n_trades, pnl, sha: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("S14 Unfiltered FVG Control — Program C Phase 2 Pre-Decision Test")
    lines.append(f"Pre-registration SHA: {sha}")
    lines.append(f"Pre-registration doc: _bmad-output/preregistration_s14_unfiltered_fvg.md")
    lines.append(f"Holdout: {HOLDOUT_CSV}")
    lines.append("=" * 70)
    lines.append("")

    # Results table
    lines.append("── S14 Results ─────────────────────────────────────────────────────")
    pf_str = f"{pf:.4f}" if pf is not None and not (isinstance(pf, float) and np.isinf(pf)) else ("inf" if pf else "N/A")
    lines.append(f"  S14 unfiltered FVG PF   : {pf_str}")
    lines.append(f"  S14 trade count         : {n_trades}")
    if pnl:
        net = sum(pnl)
        wr  = sum(1 for p in pnl if p > 0) / len(pnl) * 100
        lines.append(f"  Win rate                : {wr:.1f}%")
        lines.append(f"  Net P&L (1 contract)    : ${net:,.2f}")
    lines.append("")

    # Comparison to S12 reference values
    lines.append("── Comparison to S12 Reference (pre-committed) ─────────────────────")
    lines.append(f"  S12 p90 random baseline : {S12_P90_RANDOM:.4f}")
    lines.append(f"  S12 filtered PF         : {S12_REAL_PF:.4f}  ({S12_REAL_TRADES} trades)")
    lines.append(f"  HF min trades threshold : {HF_MIN_TRADES}")
    lines.append("")

    # Verdict
    lines.append("── S14 Verdict (pre-committed decision rule) ────────────────────────")
    if pf is None:
        verdict = "NO_TRADES"
        lines.append(f"  Verdict: {verdict}")
        lines.append("  S14 produced 0 trades — cannot evaluate. Check filters.")
    elif pf <= S12_P90_RANDOM:
        verdict = "no_unfiltered_edge"
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  S14 PF {pf_str} ≤ S12 p90 random {S12_P90_RANDOM:.4f}.")
        lines.append("  The FVG pattern has no edge without the H1 sweep.")
        lines.append("  Direction: Proceed to Phase 2 ML on the filtered 191-trade set.")
    elif pf <= S12_REAL_PF:
        verdict = "edge_exists_stay_filtered"
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  S14 PF {pf_str} > S12 p90 {S12_P90_RANDOM:.4f} BUT ≤ S12 filtered {S12_REAL_PF:.4f}.")
        lines.append("  Unfiltered FVG beats random but the filtered strategy is superior per-trade.")
        lines.append("  Direction: Proceed to Phase 2 ML — the sweep adds value.")
    elif n_trades < HF_MIN_TRADES:
        verdict = "edge_exists_stay_filtered"
        lines.append(f"  Verdict: {verdict}  (PF qualifies but trade count {n_trades} < {HF_MIN_TRADES})")
        lines.append(f"  S14 PF {pf_str} > S12 filtered {S12_REAL_PF:.4f} but only {n_trades} trades.")
        lines.append("  Insufficient sample for high-frequency pivot claim.")
        lines.append("  Direction: Proceed to Phase 2 ML.")
    else:
        verdict = "high_frequency_pivot_approved"
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  S14 PF {pf_str} > S12 filtered {S12_REAL_PF:.4f} with {n_trades} trades ≥ {HF_MIN_TRADES}.")
        lines.append("  Unfiltered FVG outperforms filtered on holdout with adequate sample.")
        lines.append("  Direction: Design a new program around the unfiltered FVG base.")
        lines.append("  Do NOT apply Phase 2 ML to the filtered set — pivot instead.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines), verdict


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S14 Unfiltered FVG Control — Program C Phase 2 Pre-Decision Test"
    )
    parser.add_argument("--preregistration", metavar="SHA",
                        help="Git SHA of the S14 pre-registration commit (required)")
    args = parser.parse_args()

    if not args.preregistration:
        print("\n" + "=" * 70, file=sys.stderr)
        print("SEALED HOLDOUT ACCESS DENIED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(
            "This script requires a --preregistration <SHA> argument.\n"
            "Supply the SHA of a git commit containing a pre-registration\n"
            "document before this script may access the sealed holdout.\n\n"
            "See ACCESS_LOG.md for the full access protocol.",
            file=sys.stderr,
        )
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    sha = args.preregistration.strip().lower()
    verify_preregistration(sha)
    append_access_log(sha, sys.argv)

    print(f"\nLoading holdout bars from {HOLDOUT_CSV} …")
    bars = load_holdout_bars()
    print(f"Loaded {len(bars):,} bars  ({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})")
    print("Running unfiltered FVG strategy (H1 sweep disabled) …\n")

    pf, n_trades, pnl = run_unfiltered_fvg(bars)

    report, verdict = build_report(pf, n_trades, pnl, sha)
    print(report)

    # Update ACCESS_LOG with final result
    log_text = ACCESS_LOG_PATH.read_text()
    pf_display = f"{pf:.4f}" if pf is not None else "N/A"
    updated  = log_text.replace(
        "| pending |",
        f"| {verdict}: PF {pf_display} ({n_trades} trades) |",
        1,
    )
    ACCESS_LOG_PATH.write_text(updated)

    # Write report file
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"s14_{stamp}.txt"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")


if __name__ == "__main__":
    main()
