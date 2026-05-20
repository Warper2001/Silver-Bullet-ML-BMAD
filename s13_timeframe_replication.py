"""
S13 Timeframe Replication — Program C Phase 1 Falsification Test
Pre-registration: _bmad-output/preregistration_phase1.md
Access token: --preregistration 910e95c

Tests whether the H1-sweep + FVG pattern persists across 1-min, 5-min, and 15-min
resolutions. Runs only when S12 returned patterns_survive.
Verdict: best_TF_PF >= 1.1 → design_phase2_ml_test; < 1.1 → PIVOT.

Usage:
    .venv/bin/python s13_timeframe_replication.py --preregistration 910e95c
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

# ── Frozen parameters (preregistration_phase1.md §Frozen Parameters) ──────────
SL_MULT           = 5.0
TP_MULT           = 6.0
MAX_HOLD          = 60        # bars at each resolution (scales with bar size)
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
BEARISH_ONLY      = True

HOLDOUT_CSV      = Path("data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv")
HOLDOUT_CUTOFF   = datetime(2026, 3, 1, tzinfo=timezone.utc)
ACCESS_LOG_PATH  = Path("data/sealed_holdout/ACCESS_LOG.md")
REPORTS_DIR      = Path("data/reports")
_PREREG_PATTERNS = (
    "_bmad-output/preregistration*.md",
    "data/sealed_holdout/PREREGISTRATION*.md",
)
ET_TZ = pytz.timezone("US/Eastern")


# ── Access log gate (copied from s12_random_entry_control.py; accessor: s13) ──

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
    log_row  = f"| {date_str} | {sha} | s13 script | `{argv_str}` | pending |\n"
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
    print(f"  ACTION REQUIRED: commit {ACCESS_LOG_PATH} to git to make this record permanent.")


# ── Bar types and loading ─────────────────────────────────────────────────────

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


def resample_bars(bars: list, tf_min: int) -> list:
    """Resample 1-min bars to tf_min-minute OHLCV bars."""
    df = pd.DataFrame({
        "timestamp": [b.timestamp for b in bars],
        "open":  [b.open  for b in bars],
        "high":  [b.high  for b in bars],
        "low":   [b.low   for b in bars],
        "close": [b.close for b in bars],
    }).set_index("timestamp")
    resampled = df.resample(f"{tf_min}min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if resampled.empty:
        raise ValueError(f"Resampling to {tf_min}-min produced an empty DataFrame.")
    result = []
    for ts, row in resampled.iterrows():
        ts_dt = ts.to_pydatetime()
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        result.append(Bar(
            timestamp=ts_dt,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        ))
    return result


# ── ATR (20-bar SMA of True Range) ───────────────────────────────────────────

def calc_atr(bars: list, end_idx: int, period: int = ATR_PERIOD) -> float:
    if end_idx < period + 1:
        return 10.0
    trs = []
    for i in range(end_idx - period, end_idx):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs))


# ── H1 resampling ─────────────────────────────────────────────────────────────

def build_h1_df(bars: list, up_to_idx: int) -> pd.DataFrame:
    """Resample bars[:up_to_idx] to 1-hour OHLC; return completed bars (iloc[:-1])."""
    if up_to_idx < 2:
        return pd.DataFrame()
    df = pd.DataFrame({
        "timestamp": [b.timestamp for b in bars[:up_to_idx]],
        "open":  [b.open  for b in bars[:up_to_idx]],
        "high":  [b.high  for b in bars[:up_to_idx]],
        "low":   [b.low   for b in bars[:up_to_idx]],
        "close": [b.close for b in bars[:up_to_idx]],
    }).set_index("timestamp")
    h1 = df.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if len(h1) < 2:
        return pd.DataFrame()
    return h1.iloc[:-1].reset_index()


# ── H1 ATR ───────────────────────────────────────────────────────────────────

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


# ── Volatility regime tracker ─────────────────────────────────────────────────

class VolRegimeTracker:
    def __init__(self):
        self._history: list = []
        self.current_h1_atr: float = 0.0

    def update(self, h1_df: pd.DataFrame) -> bool:
        """Return True (high regime = block trading) after updating with new H1 bars."""
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


# ── Market filters ────────────────────────────────────────────────────────────

def is_market_open(ts: datetime) -> bool:
    wd, h = ts.weekday(), ts.hour
    if wd == 5: return False
    if wd == 6: return h >= 23
    if wd == 4: return h < 22
    return h != 22


def is_tuesday_et(ts: datetime) -> bool:
    return ts.astimezone(ET_TZ).weekday() == 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def profit_factor(pnl: list) -> float:
    gp = sum(p for p in pnl if p > 0)
    gl = abs(sum(p for p in pnl if p < 0))
    return gp / gl if gl > 0 else float("inf")


def snap_tick(price: float) -> float:
    return round(round(price / MNQ_TICK) * MNQ_TICK, 4)


# ── H1 bearish sweep detection ────────────────────────────────────────────────

def detect_bearish_sweep(h1_df: pd.DataFrame):
    """Returns (detected: bool, sweep_ts: datetime | None) for the last completed H1 bar."""
    if len(h1_df) < 5:
        return False, None
    highs  = h1_df["high"].values
    closes = h1_df["close"].values
    ts_col = h1_df["timestamp"]
    n = len(h1_df)
    last_idx = n - 1
    last_high  = highs[last_idx]
    last_close = closes[last_idx]
    last_ts    = pd.Timestamp(ts_col.iloc[last_idx]).to_pydatetime()
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    cutoff_ts = last_ts - timedelta(hours=2)

    for i in range(2, last_idx - 1):
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2]
                and highs[i] > highs[i+1] and highs[i] > highs[i+2]):
            swing_ts = pd.Timestamp(ts_col.iloc[i]).to_pydatetime()
            if swing_ts.tzinfo is None:
                swing_ts = swing_ts.replace(tzinfo=timezone.utc)
            if swing_ts < cutoff_ts and last_high > highs[i] and last_close < highs[i]:
                return True, last_ts
    return False, None


# ── Strategy simulation (resolution-agnostic) ─────────────────────────────────

def run_strategy(bars: list):
    """
    Bearish-only Tier2 H1-sweep + FVG strategy. Resolution-agnostic:
    bars can be 1-min, 5-min, or 15-min. MAX_HOLD is in bars of the
    supplied resolution. H1 boundary detection uses hour-aligned timestamps,
    which works correctly at any sub-hourly resolution.
    Returns (pf: float | None, n_trades: int).
    """
    vol_tracker = VolRegimeTracker()
    last_h1_boundary = None
    sweep_active  = False
    sweep_expires = datetime.min.replace(tzinfo=timezone.utc)
    regime_high   = False

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
            detected, sweep_ts = detect_bearish_sweep(h1_df)
            if detected:
                sweep_active  = True
                sweep_expires = sweep_ts + timedelta(hours=6)

        if sweep_active and bar.timestamp >= sweep_expires:
            sweep_active = False

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
        if not sweep_active:
            continue

        # Bearish 3-bar FVG on last 3 bars
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
    return pf, n_trades


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(results: dict, sha: str) -> str:
    """
    results = {1: (pf, n_trades), 5: (pf, n_trades), 15: (pf, n_trades)}
    pf may be None if a resolution produced 0 trades.
    """
    lines = []
    sep = "=" * 72
    lines += [
        sep,
        "S13 TIMEFRAME REPLICATION — Program C Phase 1 Falsification Test",
        f"Pre-registration SHA : {sha}",
        f"Holdout file         : {HOLDOUT_CSV}",
        f"S12 prerequisite     : patterns_survive (real PF 1.2154 > p90 1.1350)",
        sep,
        "",
        f"{'TF (min)':>10}  {'PF':>9}  {'Trades':>7}  {'MAX_HOLD':>10}",
        "-" * 46,
    ]

    valid_pfs = []
    for tf in (1, 5, 15):
        pf_val, n = results[tf]
        pf_str = f"{pf_val:.4f}" if pf_val is not None else "N/A"
        hold_note = f"{MAX_HOLD} bars = {MAX_HOLD * tf} min"
        lines.append(f"{tf:>10}  {pf_str:>9}  {n:>7}  {hold_note:>10}")
        if pf_val is not None:
            valid_pfs.append(pf_val)

    lines.append("")

    if not valid_pfs:
        best_tf_pf = None
        best_note  = "N/A — no resolution produced trades"
    else:
        best_tf_pf = max(valid_pfs)
        best_tf    = next(tf for tf in (1, 5, 15) if results[tf][0] == best_tf_pf)
        best_note  = f"{best_tf_pf:.4f}  (best at {best_tf}-min)"

    lines += [
        f"best_TF_PF : {best_note}",
        "",
        "─" * 72,
        "S13 VERDICT (pre-committed decision rule — preregistration_phase1.md §S13):",
    ]

    if best_tf_pf is None:
        verdict = "PIVOT — no resolution produced trades; pattern cannot be assessed"
    elif best_tf_pf >= 1.1:
        verdict = f"design_phase2_ml_test — best_TF_PF {best_tf_pf:.4f} ≥ 1.1"
    else:
        verdict = f"PIVOT — best_TF_PF {best_tf_pf:.4f} < 1.1; pattern does not generalise across timeframes"

    lines += [f"  {verdict}", "─" * 72, ""]

    lines.append("ACTION REQUIRED:")
    lines.append("  1. Commit data/sealed_holdout/ACCESS_LOG.md to git (make record permanent).")
    lines.append("  2. Update ACCESS_LOG.md 'pending' → verdict above.")
    if "design_phase2_ml_test" in verdict:
        lines.append("  3. Proceed to Phase 2 ML hypothesis design.")
    else:
        lines.append("  3. PIVOT verdict — execute pivot menu from preregistration_phase1.md §Pivot Menu.")
        lines.append("     No new development on this strategy family until pivot decision is committed.")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S13 Timeframe Replication — Program C Phase 1 (requires --preregistration)"
    )
    parser.add_argument(
        "--preregistration",
        metavar="SHA",
        help="SHA of the pre-registration commit (required for holdout data ≥ 2026-03-01)",
    )
    args = parser.parse_args()

    # GATE fires BEFORE loading any holdout bars — HOLDOUT_CSV is always post-cutoff data.
    if not args.preregistration:
        _exit_access_denied(
            "This script loads the sealed holdout (≥ 2026-03-01) and requires a\n"
            "pre-registration commit SHA before any data is read.\n\n"
            "Usage:\n"
            "  .venv/bin/python s13_timeframe_replication.py --preregistration <git-sha>\n\n"
            "The SHA must point to a commit containing a pre-registration document.\n"
            "See ACCESS_LOG.md for the full access protocol."
        )
    verify_preregistration(args.preregistration)
    append_access_log(args.preregistration, sys.argv)

    print(f"\nLoading holdout bars from {HOLDOUT_CSV} ...")
    bars_1m = load_holdout_bars()
    print(f"  {len(bars_1m)} 1-min bars loaded.")

    print("Resampling to 5-min and 15-min ...")
    bars_5m  = resample_bars(bars_1m, 5)
    bars_15m = resample_bars(bars_1m, 15)
    print(f"  5-min: {len(bars_5m)} bars  |  15-min: {len(bars_15m)} bars")

    results = {}
    for label, bars in ((1, bars_1m), (5, bars_5m), (15, bars_15m)):
        print(f"\nRunning strategy on {label}-min bars ({len(bars)} bars) ...")
        pf, n = run_strategy(bars)
        results[label] = (pf, n)
        pf_str = f"{pf:.4f}" if pf is not None else "N/A"
        print(f"  {label}-min: PF={pf_str}, trades={n}")

    report = build_report(results, args.preregistration)
    print("\n" + report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_tag  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"s13_{ts_tag}.txt"
    out_path.write_text(report)
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
