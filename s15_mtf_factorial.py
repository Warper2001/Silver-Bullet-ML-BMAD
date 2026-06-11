"""
S15 Multi-Timeframe Factorial DOE — Program C Phase 2a (Round 1)

Tests 12 TF cascade combinations on pre-cutoff 2025 data only.
No holdout access. No pre-registration required.

2×3×2 factorial: sweep_tf (H1/H4) × confirm_tf (none/M5/M15) × entry_tf (M1/M15)

Usage:
    .venv/bin/python s15_mtf_factorial.py
"""

import csv
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── Frozen parameters (identical to Phase 1 / S14) ───────────────────────────
SL_MULT           = 5.0
TP_MULT           = 6.0
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
H1_BAR_CAP        = 3000

# Time-stop and pending limits in MINUTES (per entry TF):
#   M1  entry: 60 bars  × 1 min  = 60 min hold, 240 bars × 1 min = 240 min pending
#   M15 entry: 12 bars  × 15 min = 180 min hold, 16 bars × 15 min = 240 min pending
MAX_HOLD_MIN    = {1: 60,  15: 180}
MAX_PENDING_MIN = {1: 240, 15: 240}

# CHoCH break threshold: close must be this many ATR units below swing low
CHOCH_ATR_MULT  = 0.3
SWING_RADIUS    = 2   # bars each side for swing detection

# Bonferroni threshold for 12 combinations at alpha=0.05 (large-sample approx)
BONFERRONI_T_CRIT = 2.69

DATA_CSV    = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
REPORTS_DIR = Path("data/reports")
ET_TZ       = pytz.timezone("US/Eastern")

# ── Types ─────────────────────────────────────────────────────────────────────

Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close"])

CascadeConfig = namedtuple("CascadeConfig", [
    "sweep_tf_hours",   # 1=H1, 4=H4
    "confirm_tf_min",   # 0=none, 5=M5, 15=M15
    "entry_tf_min",     # 1=M1, 15=M15
    "name",
])

COMBINATIONS = [
    # H1 sweep variants
    CascadeConfig(1, 0,  1,  "H1·NoConf·M1"),    # current system baseline
    CascadeConfig(1, 15, 1,  "H1·M15·M1"),
    CascadeConfig(1, 5,  1,  "H1·M5·M1"),
    CascadeConfig(1, 0,  15, "H1·NoConf·M15"),
    CascadeConfig(1, 15, 15, "H1·M15·M15"),
    CascadeConfig(1, 5,  15, "H1·M5·M15"),
    # H4 sweep variants
    CascadeConfig(4, 0,  1,  "H4·NoConf·M1"),
    CascadeConfig(4, 15, 1,  "H4·M15·M1"),
    CascadeConfig(4, 5,  1,  "H4·M5·M1"),
    CascadeConfig(4, 0,  15, "H4·NoConf·M15"),
    CascadeConfig(4, 15, 15, "H4·M15·M15"),
    CascadeConfig(4, 5,  15, "H4·M5·M15"),
]


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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bars(csv_path: Path) -> list:
    if not csv_path.exists():
        print(f"ERROR: data file not found: {csv_path}", file=sys.stderr)
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


# ── Precomputed TF index arrays ───────────────────────────────────────────────

class PrecomputedTFs:
    """Pre-resample all TFs once; build 'last completed bar index' arrays."""

    def __init__(self, bars_1min: list):
        self.bars_m5  = resample_bars(bars_1min, 5)
        self.bars_m15 = resample_bars(bars_1min, 15)
        self.bars_h1  = resample_bars(bars_1min, 60)
        self.bars_h4  = resample_bars(bars_1min, 240)
        print(f"  M5:  {len(self.bars_m5):,} bars  "
              f"M15: {len(self.bars_m15):,} bars  "
              f"H1: {len(self.bars_h1):,} bars  "
              f"H4: {len(self.bars_h4):,} bars")

        self.m5_idx  = self._build(bars_1min, self.bars_m5,  5)
        self.m15_idx = self._build(bars_1min, self.bars_m15, 15)
        self.h1_idx  = self._build(bars_1min, self.bars_h1,  60)
        self.h4_idx  = self._build(bars_1min, self.bars_h4,  240)

    @staticmethod
    def _build(bars_1min: list, bars_tf: list, tf_min: int) -> list:
        """Return array: result[i] = index of last COMPLETED tf bar at 1-min bar i (-1 if none).
        A bar starting at T is complete when current_ts >= T + tf_min.
        """
        result = [-1] * len(bars_1min)
        j = 0
        n_tf = len(bars_tf)
        tf_delta = timedelta(minutes=tf_min)
        for i, b in enumerate(bars_1min):
            while j < n_tf and bars_tf[j].timestamp + tf_delta <= b.timestamp:
                j += 1
            result[i] = j - 1
        return result


# ── Vol regime tracker (H1-bar-based) ────────────────────────────────────────

class VolRegimeTracker:
    """Tracks H1 ATR percentile using pre-resampled H1 bars."""

    def __init__(self):
        self._atr_history: list = []
        self.current_h1_atr: float = 0.0
        self._last_fed_idx: int = -1

    def feed_up_to(self, h1_bars: list, h1_up_to_idx: int) -> None:
        """Feed any new H1 bars up to (and including) h1_up_to_idx."""
        for j in range(self._last_fed_idx + 1, h1_up_to_idx + 1):
            if j < ATR_PERIOD + 1:
                continue
            trs = []
            for k in range(j - ATR_PERIOD, j):
                h  = h1_bars[k].high
                l  = h1_bars[k].low
                pc = h1_bars[k - 1].close
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            atr = float(np.mean(trs))
            self.current_h1_atr = atr
            self._atr_history.append(atr)
            if len(self._atr_history) > VOL_LOOKBACK:
                self._atr_history.pop(0)
        self._last_fed_idx = max(self._last_fed_idx, h1_up_to_idx)

    def is_high(self) -> bool:
        if len(self._atr_history) < 20 or self.current_h1_atr <= 0:
            return False
        pct = sum(v < self.current_h1_atr for v in self._atr_history) / len(self._atr_history)
        return pct > VOL_THRESH


# ── Bearish sweep detection (generalized for H1 or H4) ───────────────────────

def detect_bearish_sweep(htf_bars: list, up_to_idx: int):
    """
    Check if the most recently completed HTF bar (up_to_idx) swept above a prior
    swing high and closed back below it.

    Returns (detected: bool, sweep_ts: datetime | None).
    """
    n = up_to_idx + 1
    if n < 5:
        return False, None

    # Apply H1_BAR_CAP to avoid all-time-high swing creep
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

    # Scan for swing highs (2-bar symmetric radius) older than cutoff
    for i in range(2, n - 3):
        h = bars[i].high
        if not (h > bars[i-1].high and h > bars[i-2].high
                and h > bars[i+1].high and h > bars[i+2].high):
            continue
        ts_i = bars[i].timestamp
        if ts_i >= cutoff_ts:
            continue
        if last_high > h and last_close < h:
            return True, last_ts

    return False, None


# ── CHoCH (swing low detection for confirm TF) ───────────────────────────────

def get_latest_swing_low(bars: list, up_to_idx: int, radius: int = SWING_RADIUS):
    """
    Return the price of the most recent confirmed swing low on the given bar list,
    looking back from up_to_idx. A swing at index j is confirmed after j+radius bars.

    Returns float or None.
    """
    # The most recent possible confirmed swing is at up_to_idx - radius
    for i in range(up_to_idx - radius, radius - 1, -1):
        lo = bars[i].low
        if all(bars[i + k].low >= lo for k in range(-radius, radius + 1) if k != 0):
            return lo
    return None


def choch_fired(bars_confirm: list, conf_idx: int, atr_confirm: float) -> bool:
    """
    Returns True if the bar at conf_idx closes below the most recent swing low
    by at least CHOCH_ATR_MULT × atr_confirm.
    """
    if conf_idx < SWING_RADIUS * 2 + 1:
        return False
    swing_low = get_latest_swing_low(bars_confirm, conf_idx - 1)
    if swing_low is None:
        return False
    threshold = swing_low - CHOCH_ATR_MULT * atr_confirm
    return bars_confirm[conf_idx].close < threshold


# ── Core cascade simulation ───────────────────────────────────────────────────

def run_cascade(bars_1min: list, precomp: PrecomputedTFs, cfg: CascadeConfig):
    """
    Run one cascade combination on 1-min bars.
    Returns (pnl: list[float], n_trades: int).
    """
    # Select HTF and confirm TF arrays
    if cfg.sweep_tf_hours == 1:
        htf_bars    = precomp.bars_h1
        htf_idx_arr = precomp.h1_idx
    else:
        htf_bars    = precomp.bars_h4
        htf_idx_arr = precomp.h4_idx

    if cfg.confirm_tf_min == 5:
        conf_bars    = precomp.bars_m5
        conf_idx_arr = precomp.m5_idx
    elif cfg.confirm_tf_min == 15:
        conf_bars    = precomp.bars_m15
        conf_idx_arr = precomp.m15_idx
    else:
        conf_bars    = None
        conf_idx_arr = None

    if cfg.entry_tf_min == 15:
        ent_bars    = precomp.bars_m15
        ent_idx_arr = precomp.m15_idx
    else:
        ent_bars    = bars_1min
        ent_idx_arr = None  # entry TF == 1-min, use i directly

    max_hold_min = MAX_HOLD_MIN[cfg.entry_tf_min]
    max_pend_min = MAX_PENDING_MIN[cfg.entry_tf_min]
    sweep_expire_hours = cfg.sweep_tf_hours * 6

    vol_tracker = VolRegimeTracker()
    last_h1_idx = -1

    # Sweep state
    sweep_active     = False
    sweep_expires_ts = datetime.min.replace(tzinfo=timezone.utc)
    confirm_done     = False
    last_htf_idx     = -1
    last_conf_idx    = -1

    # FVG scan tracking
    last_scanned_ent_idx = -1

    # Trade state
    pending      = False
    active       = False
    entry_price  = sl_price = tp_price = 0.0
    pending_ts   = entry_ts = datetime.min.replace(tzinfo=timezone.utc)

    pnl      = []
    n_trades = 0

    for i, bar in enumerate(bars_1min):
        cur_ts = bar.timestamp

        # ── Update vol regime on new H1 bar ──────────────────────────────────
        cur_h1_idx = precomp.h1_idx[i]
        if cur_h1_idx > last_h1_idx and cur_h1_idx >= 0:
            last_h1_idx = cur_h1_idx
            vol_tracker.feed_up_to(precomp.bars_h1, cur_h1_idx)

        regime_high = vol_tracker.is_high()

        # ── Advance active trade (check on every 1-min bar for realistic exits) ─
        if active:
            elapsed = (cur_ts - entry_ts).total_seconds() / 60
            if bar.high >= sl_price:
                pnl.append((entry_price - sl_price) * MNQ_DOLLAR)
                n_trades += 1
                active = False
            elif bar.low <= tp_price:
                pnl.append((entry_price - tp_price) * MNQ_DOLLAR)
                n_trades += 1
                active = False
            elif elapsed >= max_hold_min:
                pnl.append((entry_price - bar.close) * MNQ_DOLLAR)
                n_trades += 1
                active = False
            if active:
                continue

        # ── Advance pending limit order ───────────────────────────────────────
        if pending:
            elapsed = (cur_ts - pending_ts).total_seconds() / 60
            if bar.high >= entry_price:
                active   = True
                pending  = False
                entry_ts = cur_ts
                continue
            elif elapsed >= max_pend_min:
                pending = False
                # fall through — may still be in SWEEP_ACTIVE
            else:
                continue

        # ── Basic per-bar filters ─────────────────────────────────────────────
        if not is_market_open(cur_ts):
            continue
        if is_tuesday_et(cur_ts):
            continue
        if regime_high:
            continue

        # ── Expire sweep ──────────────────────────────────────────────────────
        if sweep_active and cur_ts >= sweep_expires_ts:
            sweep_active  = False
            confirm_done  = False
            last_conf_idx = -1
            last_scanned_ent_idx = -1

        # ── Detect new sweep (only from IDLE state) ───────────────────────────
        if not sweep_active:
            cur_htf_idx = htf_idx_arr[i]
            if cur_htf_idx >= 4 and cur_htf_idx > last_htf_idx:
                last_htf_idx = cur_htf_idx
                detected, sweep_ts = detect_bearish_sweep(htf_bars, cur_htf_idx)
                if detected:
                    sweep_active     = True
                    sweep_expires_ts = sweep_ts + timedelta(hours=sweep_expire_hours)
                    confirm_done     = (cfg.confirm_tf_min == 0)
                    last_conf_idx    = -1
                    last_scanned_ent_idx = -1

        if not sweep_active:
            continue

        # ── CHoCH confirmation ────────────────────────────────────────────────
        if not confirm_done and cfg.confirm_tf_min > 0:
            cur_conf_idx = conf_idx_arr[i]
            if cur_conf_idx >= SWING_RADIUS * 2 + 1 and cur_conf_idx > last_conf_idx:
                last_conf_idx = cur_conf_idx
                conf_atr = calc_atr(conf_bars, cur_conf_idx + 1)
                if choch_fired(conf_bars, cur_conf_idx, conf_atr):
                    confirm_done = True

        if not confirm_done:
            continue

        # ── FVG detection on entry TF ─────────────────────────────────────────
        if cfg.entry_tf_min == 1:
            cur_ent_idx = i
        else:
            cur_ent_idx = ent_idx_arr[i]

        if cur_ent_idx < 2:
            continue
        if cur_ent_idx <= last_scanned_ent_idx:
            continue
        last_scanned_ent_idx = cur_ent_idx

        c1 = ent_bars[cur_ent_idx - 2]
        c2 = ent_bars[cur_ent_idx - 1]
        c3 = ent_bars[cur_ent_idx]

        if not (c1.low > c3.high and c2.close < c2.open):
            continue
        gap = c1.low - c3.high
        if gap <= 0:
            continue
        entry_atr = calc_atr(ent_bars, cur_ent_idx + 1)
        if gap < ATR_THRESHOLD * entry_atr:
            continue
        if gap * MNQ_DOLLAR > MAX_GAP_DOLLARS:
            continue
        h1_atr = vol_tracker.current_h1_atr
        if h1_atr > 0 and gap < MIN_GAP_ATR_RATIO * h1_atr:
            continue

        entry_price = snap_tick(c3.high + gap * ENTRY_PCT)
        sl_price    = snap_tick(entry_price + gap * SL_MULT)
        tp_price    = snap_tick(entry_price - gap * TP_MULT)
        pending     = True
        pending_ts  = cur_ts

    return pnl, n_trades


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(pnl: list, n_trades: int, bars_1min: list) -> dict:
    if not pnl:
        return {
            "pf": None, "n": 0, "wr": 0.0,
            "sharpe": None, "dsr": False, "t": None,
        }

    pf = profit_factor(pnl)
    n  = len(pnl)
    wr = sum(1 for p in pnl if p > 0) / n * 100

    # Annualized Sharpe from per-trade P&L (approximation)
    mean_pnl = np.mean(pnl)
    std_pnl  = np.std(pnl, ddof=1)

    # Estimate trades-per-year scaling: assume 252 trading days, n_trades in ~250 days
    # Use per-trade Sharpe scaled by sqrt(expected_annual_trades)
    trading_days = (bars_1min[-1].timestamp - bars_1min[0].timestamp).days * (252 / 365)
    trades_per_day = n / max(trading_days, 1)
    annual_trades  = trades_per_day * 252
    sharpe = (mean_pnl / std_pnl * np.sqrt(annual_trades)) if std_pnl > 0 else None

    # DSR flag: t-stat using per-trade P&L (Bonferroni for 12 tests)
    t_stat = (mean_pnl / (std_pnl / np.sqrt(n))) if std_pnl > 0 else None
    dsr    = (t_stat is not None and t_stat > BONFERRONI_T_CRIT)

    return {
        "pf": pf, "n": n_trades, "wr": wr,
        "sharpe": sharpe, "dsr": dsr, "t": t_stat,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(results: list, bars_1min: list) -> str:
    lines = []
    lines.append("=" * 75)
    lines.append("S15 Multi-Timeframe Factorial DOE — Program C Phase 2a (Round 1)")
    lines.append(f"Data: {DATA_CSV}")
    lines.append(f"Bars: {len(bars_1min):,}  "
                 f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})")
    lines.append(f"Pre-cutoff only — NO holdout access")
    lines.append("=" * 75)
    lines.append("")

    # Results table
    hdr = f"{'Combination':<22} {'PF':>6}  {'N':>4}  {'WR%':>5}  {'Sharpe':>6}  {'t':>5}  DSR"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for cfg, stats in results:
        pf_s     = f"{stats['pf']:.4f}" if stats['pf'] is not None else " N/A "
        wr_s     = f"{stats['wr']:.1f}" if stats['n'] > 0 else "  N/A"
        sh_s     = f"{stats['sharpe']:.2f}" if stats['sharpe'] is not None else "   N/A"
        t_s      = f"{stats['t']:.2f}" if stats['t'] is not None else "  N/A"
        dsr_s    = " *" if stats['dsr'] else "  "
        base_tag = " (base)" if cfg.confirm_tf_min == 0 and cfg.sweep_tf_hours == 1 and cfg.entry_tf_min == 1 else ""
        lines.append(
            f"{cfg.name + base_tag:<22} {pf_s:>6}  {stats['n']:>4}  {wr_s:>5}  "
            f"{sh_s:>6}  {t_s:>5}  {dsr_s}"
        )

    lines.append("")
    lines.append(f"DSR * = t-stat > {BONFERRONI_T_CRIT} (Bonferroni-corrected, alpha/12, large-N approx)")
    lines.append("")

    # Main effects
    lines.append("── Main Effects ─────────────────────────────────────────────────────────")

    def avg_pf(subset):
        pfs = [s["pf"] for _, s in subset if s["pf"] is not None and not (isinstance(s["pf"], float) and np.isinf(s["pf"]))]
        return float(np.mean(pfs)) if pfs else None

    r = results
    h1_rows   = [(c, s) for c, s in r if c.sweep_tf_hours == 1]
    h4_rows   = [(c, s) for c, s in r if c.sweep_tf_hours == 4]
    none_rows = [(c, s) for c, s in r if c.confirm_tf_min == 0]
    m5_rows   = [(c, s) for c, s in r if c.confirm_tf_min == 5]
    m15c_rows = [(c, s) for c, s in r if c.confirm_tf_min == 15]
    m1e_rows  = [(c, s) for c, s in r if c.entry_tf_min == 1]
    m15e_rows = [(c, s) for c, s in r if c.entry_tf_min == 15]

    def pf_str(v): return f"{v:.4f}" if v is not None else "N/A"

    h1_pf  = avg_pf(h1_rows);   h4_pf   = avg_pf(h4_rows)
    no_pf  = avg_pf(none_rows); m5_pf   = avg_pf(m5_rows);  m15c_pf = avg_pf(m15c_rows)
    m1_pf  = avg_pf(m1e_rows);  m15e_pf = avg_pf(m15e_rows)

    sweep_spread   = abs(h1_pf - h4_pf)   if None not in (h1_pf, h4_pf)   else 0
    confirm_spread = max(no_pf or 0, m5_pf or 0, m15c_pf or 0) - min(no_pf or 0, m5_pf or 0, m15c_pf or 0)
    entry_spread   = abs(m1_pf - m15e_pf) if None not in (m1_pf, m15e_pf) else 0

    lines.append(f"  Sweep TF:   H1={pf_str(h1_pf)}  H4={pf_str(h4_pf)}  (spread {sweep_spread:.4f})")
    lines.append(f"  Confirm TF: none={pf_str(no_pf)}  M5={pf_str(m5_pf)}  M15={pf_str(m15c_pf)}  (spread {confirm_spread:.4f})")
    lines.append(f"  Entry TF:   M1={pf_str(m1_pf)}  M15={pf_str(m15e_pf)}  (spread {entry_spread:.4f})")
    lines.append("")

    spreads = {
        "Sweep TF":   sweep_spread,
        "Confirm TF": confirm_spread,
        "Entry TF":   entry_spread,
    }
    dominant = max(spreads, key=lambda k: spreads[k])
    lines.append(f"  Dominant factor: {dominant} (largest avg-PF spread = {spreads[dominant]:.4f})")
    lines.append(f"  Round 2 recommendation: Expand {dominant} with additional levels/settings.")
    lines.append("")
    lines.append("=" * 75)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nS15 Multi-Timeframe Factorial DOE — Program C Phase 2a")
    print("Running on PRE-CUTOFF data only — no holdout access\n")

    print(f"Loading: {DATA_CSV}")
    bars_1min = load_bars(DATA_CSV)
    print(f"Loaded {len(bars_1min):,} bars  "
          f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})\n")

    print("Pre-resampling to M5, M15, H1, H4...")
    precomp = PrecomputedTFs(bars_1min)
    print("Building completion index arrays... done\n")

    results = []
    for k, cfg in enumerate(COMBINATIONS, 1):
        print(f"[{k:2d}/12] {cfg.name:<22}", end="", flush=True)
        pnl, n_trades = run_cascade(bars_1min, precomp, cfg)
        stats = compute_stats(pnl, n_trades, bars_1min)
        results.append((cfg, stats))
        pf_display = f"{stats['pf']:.4f}" if stats['pf'] is not None else "N/A"
        print(f"  PF={pf_display}  N={n_trades}  "
              f"WR={stats['wr']:.1f}%  "
              f"{'DSR*' if stats['dsr'] else ''}")

    print()
    report = build_report(results, bars_1min)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"s15_factorial_{stamp}.txt"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")
    print("Round 1 complete. See report for Round 2 recommendations.")


if __name__ == "__main__":
    main()
