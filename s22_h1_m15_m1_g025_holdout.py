"""
S22 H1·M15·M1·g0.25 Holdout Test — Program C Phase 2b
Pre-registration: _bmad-output/preregistration_h1_m15_m1_g025.md

Architecture: H1 liquidity sweep → M15 CHoCH confirmation → M1 FVG entry
Gap filter: MIN_GAP_ATR_RATIO = 0.25

S21 finding: M15 CHoCH is load-bearing for H1+M1 edge.
Without confirm, H1+M1 has PF<1.0 at all gap levels on 2025 data.

Decision rule (pre-committed):
  N < 10                          → insufficient_sample
  N ≥ 10 AND PF ≤ 1.1350         → no_edge
  N ≥ 10 AND PF > 1.1350 ≤ 1.1656 → edge_confirmed
  N ≥ 10 AND PF > 1.1656         → edge_exceeds_insample

Usage:
    .venv/bin/python s22_h1_m15_m1_g025_holdout.py --preregistration <SHA>
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

# ── Frozen parameters (pre-registration: preregistration_h1_m15_m1_g025.md) ──
SL_MULT           = 5.0
TP_MULT           = 6.0
ATR_PERIOD        = 20
VOL_LOOKBACK      = 120
VOL_THRESH        = 0.75
ATR_THRESHOLD     = 0.5
MIN_GAP_ATR_RATIO = 0.25
MAX_GAP_DOLLARS   = 60.0
MNQ_DOLLAR        = 2.0
ENTRY_PCT         = 0.5
MNQ_TICK          = 0.25
BEARISH_ONLY      = True
H1_BAR_CAP        = 3000
CHOCH_ATR_MULT    = 0.3
SWING_RADIUS      = 2

SWEEP_TF_HOURS  = 1    # H1
CONFIRM_TF_MIN  = 15   # M15 CHoCH
ENTRY_TF_MIN    = 1    # M1 FVG
MAX_HOLD_MIN    = 60   # 60 M1 bars
MAX_PENDING_MIN = 240  # 240 M1 bars

# S22 decision thresholds (pre-committed)
S12_P90_RANDOM = 1.1350
INSAMPLE_PF    = 1.1656   # H1·M15·M1·g0.25 in-sample PF (N=109, 2025)
MIN_N          = 10

HOLDOUT_CSV     = Path("data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv")
ACCESS_LOG_PATH = Path("data/sealed_holdout/ACCESS_LOG.md")
REPORTS_DIR     = Path("data/reports")
_PREREG_PATTERNS = (
    "_bmad-output/preregistration*.md",
    "data/sealed_holdout/PREREGISTRATION*.md",
)
ET_TZ = pytz.timezone("US/Eastern")


# ── Access gate ───────────────────────────────────────────────────────────────

def _exit_denied(msg: str) -> None:
    print(f"\n{'='*70}", file=sys.stderr)
    print("SEALED HOLDOUT ACCESS DENIED", file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)
    print(msg, file=sys.stderr)
    print(f"\nSee {ACCESS_LOG_PATH} for the full access protocol.", file=sys.stderr)
    print(f"{'='*70}\n", file=sys.stderr)
    sys.exit(1)


def verify_preregistration(sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{4,40}", sha, re.IGNORECASE):
        _exit_denied(
            f"'{sha}' is not a valid git SHA.\n"
            "Commit your pre-registration document first."
        )
    result = subprocess.run(
        ["git", "cat-file", "-t", sha],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or result.stdout.strip() != "commit":
        _exit_denied(
            f"SHA '{sha}' not found in local git repo (or is not a commit).\n"
            "Commit your pre-registration document first."
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
        _exit_denied(
            f"Commit {sha[:8]} contains no pre-registration document.\n"
            "Expected a file matching one of:\n"
            + "\n".join(f"  {p}" for p in _PREREG_PATTERNS)
            + "\n\nFiles in that commit:\n"
            + ("\n".join(f"  {f}" for f in file_lines[:20]) or "  (none)")
        )


def append_access_log(sha: str, argv: list) -> None:
    if not ACCESS_LOG_PATH.exists():
        _exit_denied(f"{ACCESS_LOG_PATH} does not exist.")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    argv_str = " ".join(argv[1:]).replace("|", "\\|")
    log_row  = f"| {date_str} | {sha} | s22 script | `{argv_str}` | pending |\n"
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


# ── Types and loading ─────────────────────────────────────────────────────────

Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close"])


def load_bars(csv_path: Path) -> list:
    if not csv_path.exists():
        print(f"ERROR: not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    bars = []
    with open(csv_path) as f:
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


# ── Precomputed TF indices ────────────────────────────────────────────────────

def build_completed_idx(bars_1min: list, bars_tf: list, tf_min: int) -> list:
    result = [-1] * len(bars_1min)
    j = 0
    n_tf = len(bars_tf)
    tf_delta = timedelta(minutes=tf_min)
    for i, b in enumerate(bars_1min):
        while j < n_tf and bars_tf[j].timestamp + tf_delta <= b.timestamp:
            j += 1
        result[i] = j - 1
    return result


# ── Utility ───────────────────────────────────────────────────────────────────

def snap_tick(price: float) -> float:
    return round(round(price / MNQ_TICK) * MNQ_TICK, 4)


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


def calc_atr(bars: list, end_idx: int, period: int = ATR_PERIOD) -> float:
    if end_idx < period + 1:
        return 10.0
    trs = []
    for i in range(end_idx - period, end_idx):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs))


# ── Vol regime ────────────────────────────────────────────────────────────────

class VolRegimeTracker:
    def __init__(self):
        self._atr_history: list = []
        self.current_h1_atr: float = 0.0
        self._last_fed: int = -1

    def feed_up_to(self, h1_bars: list, up_to: int) -> None:
        for j in range(self._last_fed + 1, up_to + 1):
            if j < ATR_PERIOD + 1:
                continue
            trs = []
            for k in range(j - ATR_PERIOD, j):
                h  = h1_bars[k].high; l = h1_bars[k].low; pc = h1_bars[k-1].close
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            self.current_h1_atr = float(np.mean(trs))
            self._atr_history.append(self.current_h1_atr)
            if len(self._atr_history) > VOL_LOOKBACK:
                self._atr_history.pop(0)
        self._last_fed = max(self._last_fed, up_to)

    def is_high(self) -> bool:
        if len(self._atr_history) < 20 or self.current_h1_atr <= 0:
            return False
        pct = sum(v < self.current_h1_atr for v in self._atr_history) / len(self._atr_history)
        return pct > VOL_THRESH


# ── Sweep and CHoCH ───────────────────────────────────────────────────────────

def detect_bearish_sweep(htf_bars: list, up_to_idx: int):
    n = up_to_idx + 1
    if n < 5:
        return False, None
    start = max(0, up_to_idx - H1_BAR_CAP)
    bars  = htf_bars[start:up_to_idx + 1]
    n     = len(bars)
    if n < 5:
        return False, None
    last_bar   = bars[-1]
    last_high  = last_bar.high
    last_close = last_bar.close
    last_ts    = last_bar.timestamp
    cutoff_ts  = last_ts - timedelta(hours=2)
    for i in range(2, n - 3):
        h = bars[i].high
        if not (h > bars[i-1].high and h > bars[i-2].high
                and h > bars[i+1].high and h > bars[i+2].high):
            continue
        if bars[i].timestamp >= cutoff_ts:
            continue
        if last_high > h and last_close < h:
            return True, last_ts
    return False, None


def get_latest_swing_low(bars: list, up_to_idx: int) -> float | None:
    r = SWING_RADIUS
    for i in range(up_to_idx - r, r - 1, -1):
        lo = bars[i].low
        if all(bars[i + k].low >= lo for k in range(-r, r + 1) if k != 0):
            return lo
    return None


def choch_fired(bars_m15: list, conf_idx: int, atr_m15: float) -> bool:
    r = SWING_RADIUS
    if conf_idx < r * 2 + 1:
        return False
    swing_low = get_latest_swing_low(bars_m15, conf_idx - 1)
    if swing_low is None:
        return False
    return bars_m15[conf_idx].close < swing_low - CHOCH_ATR_MULT * atr_m15


# ── Core strategy ─────────────────────────────────────────────────────────────

def run_strategy(bars_1min: list):
    """H1·M15·M1·g0.25 cascade. Returns (pnl, n_trades)."""
    bars_m15 = resample_bars(bars_1min, CONFIRM_TF_MIN)
    bars_h1  = resample_bars(bars_1min, 60)

    m15_idx = build_completed_idx(bars_1min, bars_m15, CONFIRM_TF_MIN)
    h1_idx  = build_completed_idx(bars_1min, bars_h1,  60)

    vol_tracker  = VolRegimeTracker()
    last_h1_vol  = -1
    last_htf_idx = -1

    sweep_active     = False
    sweep_expires_ts = datetime.min.replace(tzinfo=timezone.utc)
    confirm_done     = False
    last_m15_conf    = -1

    pending      = False
    active       = False
    entry_price  = sl_price = tp_price = 0.0
    pending_bars = active_bars = 0

    pnl      = []
    n_trades = 0

    for i, bar in enumerate(bars_1min):
        cur_ts = bar.timestamp

        cur_h1 = h1_idx[i]
        if cur_h1 > last_h1_vol and cur_h1 >= 0:
            last_h1_vol = cur_h1
            vol_tracker.feed_up_to(bars_h1, cur_h1)

        regime_high = vol_tracker.is_high()

        # Advance active trade
        if active:
            active_bars += 1
            if bar.high >= sl_price:
                pnl.append((entry_price - sl_price) * MNQ_DOLLAR)
                n_trades += 1; active = False; active_bars = 0
            elif bar.low <= tp_price:
                pnl.append((entry_price - tp_price) * MNQ_DOLLAR)
                n_trades += 1; active = False; active_bars = 0
            elif active_bars >= MAX_HOLD_MIN:
                pnl.append((entry_price - bar.close) * MNQ_DOLLAR)
                n_trades += 1; active = False; active_bars = 0
            if active:
                continue

        # Advance pending
        if pending:
            pending_bars += 1
            if bar.high >= entry_price:
                active = True; pending = False; pending_bars = 0; active_bars = 0
                continue
            elif pending_bars >= MAX_PENDING_MIN:
                pending = False; pending_bars = 0
            else:
                continue

        if not is_market_open(cur_ts):
            continue
        if is_tuesday_et(cur_ts):
            continue
        if regime_high:
            continue

        # Expire sweep
        if sweep_active and cur_ts >= sweep_expires_ts:
            sweep_active  = False
            confirm_done  = False
            last_m15_conf = -1
            last_htf_idx  = -1

        # Detect H1 sweep
        if not sweep_active:
            if cur_h1 >= 4 and cur_h1 > last_htf_idx:
                last_htf_idx = cur_h1
                detected, sweep_ts = detect_bearish_sweep(bars_h1, cur_h1)
                if detected:
                    sweep_active     = True
                    sweep_expires_ts = sweep_ts + timedelta(hours=SWEEP_TF_HOURS * 6)
                    confirm_done     = False
                    last_m15_conf    = -1

        if not sweep_active:
            continue

        # M15 CHoCH confirmation
        if not confirm_done:
            cur_m15 = m15_idx[i]
            if cur_m15 >= SWING_RADIUS * 2 + 1 and cur_m15 > last_m15_conf:
                last_m15_conf = cur_m15
                atr_m15 = calc_atr(bars_m15, cur_m15 + 1)
                if choch_fired(bars_m15, cur_m15, atr_m15):
                    confirm_done = True

        if not confirm_done:
            continue

        # M1 FVG detection on current bar
        if i < 2:
            continue

        c1 = bars_1min[i - 2]
        c2 = bars_1min[i - 1]
        c3 = bars_1min[i]

        if not (c1.low > c3.high and c2.close < c2.open):
            continue
        gap = c1.low - c3.high
        if gap <= 0:
            continue
        m1_atr = calc_atr(bars_1min, i + 1)
        if gap < ATR_THRESHOLD * m1_atr:
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
        pending_bars = 0

    return pnl, n_trades


# ── Report and verdict ────────────────────────────────────────────────────────

def apply_verdict(pf, n_trades):
    if n_trades < MIN_N:
        return "insufficient_sample"
    if pf is None:
        return "insufficient_sample"
    if pf <= S12_P90_RANDOM:
        return "no_edge"
    if pf <= INSAMPLE_PF:
        return "edge_confirmed"
    return "edge_exceeds_insample"


def build_report(pf, n_trades, pnl, verdict, sha: str) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("S22 H1·M15·M1·g0.25 Holdout Test — Program C Phase 2b")
    lines.append(f"Pre-registration SHA: {sha}")
    lines.append(f"Pre-registration doc: _bmad-output/preregistration_h1_m15_m1_g025.md")
    lines.append(f"Holdout: {HOLDOUT_CSV}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("── Architecture (frozen) ────────────────────────────────────────────")
    lines.append("  Sweep:   H1 (1-hour) liquidity sweep, expires 6h after sweep bar")
    lines.append("  Confirm: M15 CHoCH (close < swing_low − 0.3×M15_ATR)")
    lines.append("  Entry:   M1 bearish FVG, midpoint entry")
    lines.append(f"  MIN_GAP_ATR_RATIO = {MIN_GAP_ATR_RATIO}")
    lines.append(f"  MAX_HOLD = {MAX_HOLD_MIN} M1 bars  MAX_PENDING = {MAX_PENDING_MIN} M1 bars")
    lines.append("")

    lines.append("── S22 Results ──────────────────────────────────────────────────────")
    pf_str = (f"{pf:.4f}" if pf is not None and not (isinstance(pf, float) and np.isinf(pf))
              else ("inf" if pf else "N/A"))
    lines.append(f"  PF              : {pf_str}")
    lines.append(f"  Trade count     : {n_trades}")
    if pnl:
        net = sum(pnl)
        wr  = sum(1 for p in pnl if p > 0) / len(pnl) * 100
        lines.append(f"  Win rate        : {wr:.1f}%")
        lines.append(f"  Net P&L (1x)    : ${net:,.2f}")
        avg_w = np.mean([p for p in pnl if p > 0]) if any(p > 0 for p in pnl) else 0
        avg_l = np.mean([p for p in pnl if p < 0]) if any(p < 0 for p in pnl) else 0
        lines.append(f"  Avg win         : ${avg_w:,.2f}")
        lines.append(f"  Avg loss        : ${avg_l:,.2f}")
    lines.append("")

    lines.append("── Comparison (pre-committed) ───────────────────────────────────────")
    lines.append(f"  S12 p90 random baseline    : {S12_P90_RANDOM:.4f}")
    lines.append(f"  S22 in-sample PF (2025)    : {INSAMPLE_PF:.4f}  (N=109, g0.25)")
    lines.append(f"  Minimum N for verdict      : {MIN_N}")
    lines.append("")

    lines.append("── S22 Verdict (pre-committed decision rule) ────────────────────────")
    if verdict == "insufficient_sample":
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  N={n_trades} < {MIN_N} minimum. M15 CHoCH + g0.25 too restrictive")
        lines.append("  for this holdout window. Re-evaluate architecture.")
    elif verdict == "no_edge":
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  PF {pf_str} ≤ S12 p90 random {S12_P90_RANDOM:.4f}.")
        lines.append("  H1·M15·M1 has no holdout edge at g0.25. PIVOT.")
    elif verdict == "edge_confirmed":
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  PF {pf_str} > S12 p90 {S12_P90_RANDOM:.4f}.")
        lines.append("  Holdout edge confirmed. Proceed to Phase 2 ML meta-labeling.")
        lines.append("  Architecture: H1·M15·M1·g0.25 is the new baseline.")
    else:
        lines.append(f"  Verdict: {verdict}")
        lines.append(f"  PF {pf_str} > in-sample PF {INSAMPLE_PF:.4f}.")
        lines.append("  Unusually strong holdout result. Proceed to Phase 2 ML with")
        lines.append("  high confidence.")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="S22 H1·M15·M1·g0.25 Holdout Test")
    parser.add_argument("--preregistration", metavar="SHA",
                        help="Git SHA of the S22 pre-registration commit (required)")
    args = parser.parse_args()

    if not args.preregistration:
        print("\n" + "=" * 70, file=sys.stderr)
        print("SEALED HOLDOUT ACCESS DENIED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(
            "This script requires --preregistration <SHA>.\n"
            "Commit _bmad-output/preregistration_h1_m15_m1_g025.md first,\n"
            "then supply that commit SHA.\n\nSee ACCESS_LOG.md for the protocol.",
            file=sys.stderr,
        )
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    sha = args.preregistration.strip().lower()
    verify_preregistration(sha)
    append_access_log(sha, sys.argv)

    print(f"\nLoading holdout bars from {HOLDOUT_CSV} …")
    bars_1min = load_bars(HOLDOUT_CSV)
    print(f"Loaded {len(bars_1min):,} bars  "
          f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})")
    print("Running H1·M15·M1·g0.25 strategy …\n")

    pnl, n_trades = run_strategy(bars_1min)
    pf = profit_factor(pnl) if pnl else None
    verdict = apply_verdict(pf, n_trades)
    report = build_report(pf, n_trades, pnl, verdict, sha)
    print(report)

    pf_display = f"{pf:.4f}" if pf is not None else "N/A"
    log_text = ACCESS_LOG_PATH.read_text()
    updated  = log_text.replace(
        "| pending |",
        f"| {verdict}: PF {pf_display} ({n_trades} trades) |",
        1,
    )
    ACCESS_LOG_PATH.write_text(updated)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"s22_{stamp}.txt"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")
    print("ACTION REQUIRED: git commit ACCESS_LOG.md to make the access record permanent.")


if __name__ == "__main__":
    main()
