"""
S12 Random-Entry Control — Program C Phase 1 Falsification Test
Pre-registration: _bmad-output/preregistration_phase1.md
Access token: --preregistration 910e95c

Tests whether the real Tier2 strategy's PF on the sealed holdout exceeds
the 90th percentile of a 50-seed random-entry distribution under identical
time filters, position sizing, and exit logic.

Usage:
    .venv/bin/python s12_random_entry_control.py --preregistration 910e95c
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
N_SEEDS           = 50

HOLDOUT_CSV      = Path("data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv")
HOLDOUT_CUTOFF   = datetime(2026, 3, 1, tzinfo=timezone.utc)
ACCESS_LOG_PATH  = Path("data/sealed_holdout/ACCESS_LOG.md")
REPORTS_DIR      = Path("data/reports")
_PREREG_PATTERNS = (
    "_bmad-output/preregistration*.md",
    "data/sealed_holdout/PREREGISTRATION*.md",
)
ET_TZ = pytz.timezone("US/Eastern")


# ── Access log gate (copied from backtest_tier2_1year_validation.py:66-141) ───

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
    log_row  = f"| {date_str} | {sha} | s12 script | `{argv_str}` | pending |\n"
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


# ── Real strategy simulation ──────────────────────────────────────────────────

def run_real_strategy(bars: list):
    """Bearish-only Tier2 H1-sweep + FVG strategy on holdout. Returns (pf, n_trades)."""
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


# ── Filter pre-computation (shared across all 50 seeds) ──────────────────────

def precompute_time_filter(bars: list):
    """
    Compute vol_regime and time-filter mask once; all seeds share this state.
    Returns (vol_high_per_bar: list[bool], time_ok_per_bar: list[bool]).
    """
    vol_tracker = VolRegimeTracker()
    last_h1_boundary = None
    regime_high = False
    vol_high = []
    time_ok  = []

    for i, bar in enumerate(bars):
        h1_boundary = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        if h1_boundary != last_h1_boundary:
            last_h1_boundary = h1_boundary
            h1_df       = build_h1_df(bars, i)
            regime_high = vol_tracker.update(h1_df)
        vol_high.append(regime_high)
        time_ok.append(is_market_open(bar.timestamp) and not is_tuesday_et(bar.timestamp))

    return vol_high, time_ok


# ── Random seed simulation ────────────────────────────────────────────────────

def run_random_seed(bars: list, seed: int, vol_high: list, time_ok: list):
    """
    Coin-flip entries at bars passing time filters. Market order (no pending period).
    SL = 5×ATR₂₀, TP = 6×ATR₂₀. Returns (pf, n_trades).
    """
    rng      = np.random.default_rng(seed)
    active   = False
    entry    = sl = tp = 0.0
    direction= "SHORT"
    bars_held= 0
    pnl      = []
    n_trades = 0

    for i, bar in enumerate(bars):
        if active:
            bars_held += 1
            if direction == "SHORT":
                if bar.high >= sl:
                    pnl.append((entry - sl) * MNQ_DOLLAR); active = False; n_trades += 1
                elif bar.low <= tp:
                    pnl.append((entry - tp) * MNQ_DOLLAR); active = False; n_trades += 1
                elif bars_held >= MAX_HOLD:
                    pnl.append((entry - bar.close) * MNQ_DOLLAR); active = False; n_trades += 1
            else:
                if bar.low <= sl:
                    pnl.append((sl - entry) * MNQ_DOLLAR); active = False; n_trades += 1
                elif bar.high >= tp:
                    pnl.append((tp - entry) * MNQ_DOLLAR); active = False; n_trades += 1
                elif bars_held >= MAX_HOLD:
                    pnl.append((bar.close - entry) * MNQ_DOLLAR); active = False; n_trades += 1
            continue

        if not time_ok[i] or vol_high[i]:
            continue

        direction = "SHORT" if rng.integers(2) == 0 else "LONG"
        entry     = bar.close
        atr       = calc_atr(bars, i + 1)
        if direction == "SHORT":
            sl = entry + SL_MULT * atr
            tp = entry - TP_MULT * atr
        else:
            sl = entry - SL_MULT * atr
            tp = entry + TP_MULT * atr
        active    = True
        bars_held = 0

    pf = profit_factor(pnl) if pnl else None
    return pf, n_trades


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(real_pf, real_n, seed_results: list, sha: str) -> str:
    lines = []
    sep = "=" * 72
    lines += [
        sep,
        "S12 RANDOM-ENTRY CONTROL — Program C Phase 1 Falsification Test",
        f"Pre-registration SHA : {sha}",
        f"Holdout file         : {HOLDOUT_CSV}",
        f"Seeds                : {N_SEEDS}  (seeds 0–{N_SEEDS-1})",
        sep,
        "",
        f"{'Seed':>5}  {'PF':>9}  {'Trades':>7}",
        "-" * 30,
    ]
    finite_pfs = []
    for seed, (pf_val, n) in enumerate(seed_results):
        if pf_val is None:
            lines.append(f"{seed:>5}  {'N/A':>9}  {n:>7}")
        else:
            finite_pfs.append(pf_val)
            pf_str = f"inf" if pf_val == float("inf") else f"{pf_val:.4f}"
            lines.append(f"{seed:>5}  {pf_str:>9}  {n:>7}")

    lines.append("")
    if finite_pfs:
        # Convert inf→NaN so nanpercentile/nanmedian exclude all-winner seeds
        # (inf seeds are valid but would make p90=inf if they're the majority)
        pfs_arr = np.where(np.isinf(finite_pfs), np.nan, finite_pfs)
        n_inf = int(np.sum(np.isinf(finite_pfs)))
        med = float(np.nanmedian(pfs_arr))
        p90 = float(np.nanpercentile(pfs_arr, 90))
        p10 = float(np.nanpercentile(pfs_arr, 10))
        finite_noninf = [p for p in finite_pfs if p != float("inf")]
        min_pf = min(finite_noninf) if finite_noninf else float("inf")
        max_pf = max(finite_noninf) if finite_noninf else float("inf")
        inf_note = f"  ({n_inf} seeds had inf PF — excluded from pct stats, treated as above p90)" if n_inf else ""
        lines += [
            f"Random-entry distribution ({len(finite_pfs)} seeds with ≥1 trade):{inf_note}",
            f"  10th pct : {p10:.4f}",
            f"  Median   : {med:.4f}  ← lower S12 threshold",
            f"  90th pct : {p90:.4f}  ← upper S12 threshold (must beat to claim edge)",
            f"  Min / Max: {min_pf:.4f} / {max_pf:.4f}",
        ]
    else:
        lines.append("WARNING: All 50 seeds produced 0 trades — cannot compute distribution.")
        med = p90 = None

    real_str = f"{real_pf:.4f}" if real_pf is not None else "N/A (0 trades)"
    lines += ["", f"Real strategy PF     : {real_str}  ({real_n} trades)"]

    # S12 verdict (pre-committed decision rule from preregistration_phase1.md)
    lines += ["", "─" * 72, "S12 VERDICT (pre-committed decision rule):"]
    if real_pf is None:
        verdict = "PIVOT — real strategy produced 0 trades on holdout"
    elif med is None:
        verdict = "INCONCLUSIVE — no finite random-entry PFs to compare against"
    elif real_pf > p90:
        verdict = f"patterns_survive — real PF {real_pf:.4f} > 90th-pct {p90:.4f}"
    elif real_pf > med:
        verdict = (f"PIVOT (ambiguous) — real PF {real_pf:.4f} is between "
                   f"median {med:.4f} and 90th-pct {p90:.4f}; ambiguous = PIVOT per pre-reg")
    else:
        verdict = f"PIVOT — real PF {real_pf:.4f} ≤ median {med:.4f} of random-entry distribution"
    lines += [f"  {verdict}", "─" * 72, ""]

    lines.append("ACTION REQUIRED:")
    lines.append("  1. Commit data/sealed_holdout/ACCESS_LOG.md (make access record permanent).")
    lines.append("  2. Update ACCESS_LOG.md 'pending' → verdict above.")
    if "patterns_survive" in verdict:
        lines.append("  3. Proceed to S13 timeframe replication with the same --preregistration SHA.")
    else:
        lines.append("  3. PIVOT verdict — execute pivot menu from preregistration_phase1.md §Pivot Menu.")
        lines.append("     No new development on this strategy family until pivot decision is committed.")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S12 Random-Entry Control — Program C Phase 1 Falsification Test"
    )
    parser.add_argument(
        "--preregistration",
        metavar="GIT_SHA",
        help="SHA of the pre-registration commit (required for holdout data ≥ 2026-03-01)",
    )
    args = parser.parse_args()

    # Gate fires BEFORE loading any holdout bars — HOLDOUT_CSV is always post-cutoff data.
    if not args.preregistration:
        _exit_access_denied(
            "This script loads the sealed holdout (≥ 2026-03-01) and requires a\n"
            "pre-registration commit SHA before any data is read.\n\n"
            "Usage:\n"
            "  .venv/bin/python s12_random_entry_control.py --preregistration <git-sha>\n\n"
            "The SHA must point to a commit containing a pre-registration document.\n"
            "See ACCESS_LOG.md for the full access protocol."
        )
    sys.stdout.flush()
    verify_preregistration(args.preregistration)
    append_access_log(args.preregistration, sys.argv)

    print()
    print("Loading holdout bars …")
    bars = load_holdout_bars()
    print(f"  Loaded {len(bars):,} bars  "
          f"({bars[0].timestamp.date()} → {bars[-1].timestamp.date()})")

    print()
    print("Running real Tier2 strategy on holdout …")
    real_pf, real_n = run_real_strategy(bars)
    real_str = f"{real_pf:.4f}" if real_pf is not None else "N/A"
    print(f"  Real strategy: PF={real_str}  trades={real_n}")

    print()
    print(f"Pre-computing time/vol-regime filter mask …")
    vol_high, time_ok = precompute_time_filter(bars)
    open_bars = sum(1 for ok, vh in zip(time_ok, vol_high) if ok and not vh)
    print(f"  {open_bars:,} bars pass time + vol-regime filters")

    print(f"Running {N_SEEDS} random-entry seeds …")
    seed_results = []
    for seed in range(N_SEEDS):
        pf_val, n = run_random_seed(bars, seed, vol_high, time_ok)
        seed_results.append((pf_val, n))
        if (seed + 1) % 10 == 0:
            print(f"  … {seed + 1}/{N_SEEDS} seeds done")

    report = build_report(real_pf, real_n, seed_results, args.preregistration or "none")
    print()
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"s12_{ts_str}.txt"
    report_path.write_text(report)
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
