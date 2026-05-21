"""
S16 Multi-Timeframe Factorial DOE — Program C Phase 2a (Round 2)

Deepens the two dominant factors from Round 1:
  - Entry TF: adds M5 (5-min) and M30 (30-min) between M1 and M15
  - Sweep TF: retains H4 focus (won by 0.0717 avg-PF over H1)
  - Confirm TF: tests tighter CHoCH (1-bar swing radius vs default 2-bar)
  - FVG quality: tests tighter MIN_GAP_ATR_RATIO=0.25 vs default 0.15

Round 1 winners (reference, re-run for consistency check):
  H4·M15·M15: PF=1.1155  H4·M5·M15: PF=1.1102  H4·NoConf·M15: PF=1.0753

Usage:
    .venv/bin/python s16_mtf_factorial_r2.py
"""

import csv
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── Frozen parameters (identical to Phase 1 / S14 / S15) ─────────────────────
SL_MULT           = 5.0
TP_MULT           = 6.0
ATR_PERIOD        = 20
VOL_LOOKBACK      = 120
VOL_THRESH        = 0.75
ATR_THRESHOLD     = 0.5
MIN_GAP_ATR_RATIO = 0.15      # default; "tight" variant uses 0.25
MAX_GAP_DOLLARS   = 60.0
MNQ_DOLLAR        = 2.0
ENTRY_PCT         = 0.5
MNQ_TICK          = 0.25
BEARISH_ONLY      = True
H1_BAR_CAP        = 3000
CHOCH_ATR_MULT    = 0.3
SWING_RADIUS      = 2          # default; "r1" variant uses 1
MIN_GAP_ATR_TIGHT = 0.25       # tight-FVG variant override

# Time-stop and pending limits in MINUTES per entry TF:
#   M1:  60 bars × 1 min  = 60 min hold,  240 bars × 1 min  = 240 min pending
#   M5:  24 bars × 5 min  = 120 min hold, 48 bars  × 5 min  = 240 min pending
#   M15: 12 bars × 15 min = 180 min hold, 16 bars  × 15 min = 240 min pending
#   M30:  6 bars × 30 min = 180 min hold,  8 bars  × 30 min = 240 min pending
MAX_HOLD_MIN    = {1: 60, 5: 120, 15: 180, 30: 180}
MAX_PENDING_MIN = {1: 240, 5: 240, 15: 240, 30: 240}

# Bonferroni for 14 combinations at alpha=0.05 (large-sample approx)
BONFERRONI_T_CRIT = 2.73

DATA_CSV    = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
REPORTS_DIR = Path("data/reports")
ET_TZ       = pytz.timezone("US/Eastern")

# ── Types ─────────────────────────────────────────────────────────────────────

Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close"])

CascadeConfig = namedtuple("CascadeConfig", [
    "sweep_tf_hours",   # 1=H1, 4=H4
    "confirm_tf_min",   # 0=none, 5=M5, 15=M15
    "entry_tf_min",     # 1=M1, 5=M5, 15=M15, 30=M30
    "name",
    "variant",          # "" | "r1" (swing_radius=1) | "tight" (MIN_GAP_ATR_RATIO=0.25)
])

COMBINATIONS = [
    # ── Re-run R1 winners as consistency references ───────────────────────────
    CascadeConfig(4, 0,  15, "H4·NoConf·M15",      ""),   # R1: 1.0753
    CascadeConfig(4, 5,  15, "H4·M5·M15",           ""),   # R1: 1.1102
    CascadeConfig(4, 15, 15, "H4·M15·M15",          ""),   # R1: 1.1155
    CascadeConfig(4, 5,  1,  "H4·M5·M1",            ""),   # R1: 1.0400 (hi-freq)

    # ── Tighter CHoCH: 1-bar swing radius instead of default 2-bar ──────────
    CascadeConfig(4, 5,  15, "H4·M5r1·M15",         "r1"),
    CascadeConfig(4, 15, 15, "H4·M15r1·M15",        "r1"),

    # ── New entry TF: M5 (between M1 and M15) ────────────────────────────────
    CascadeConfig(4, 0,  5,  "H4·NoConf·M5",        ""),
    CascadeConfig(4, 5,  5,  "H4·M5conf·M5",        ""),
    CascadeConfig(4, 15, 5,  "H4·M15·M5",           ""),
    CascadeConfig(1, 0,  5,  "H1·NoConf·M5",        ""),
    CascadeConfig(1, 5,  5,  "H1·M5conf·M5",        ""),

    # ── Tighter FVG quality gate ──────────────────────────────────────────────
    CascadeConfig(4, 0,  15, "H4·NoConf·M15·tight", "tight"),

    # ── New entry TF: M30 (between M15 and H1) ───────────────────────────────
    CascadeConfig(4, 0,  30, "H4·NoConf·M30",       ""),
    CascadeConfig(4, 15, 30, "H4·M15·M30",          ""),
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
    def __init__(self, bars_1min: list):
        self.bars_m5  = resample_bars(bars_1min, 5)
        self.bars_m15 = resample_bars(bars_1min, 15)
        self.bars_m30 = resample_bars(bars_1min, 30)
        self.bars_h1  = resample_bars(bars_1min, 60)
        self.bars_h4  = resample_bars(bars_1min, 240)
        print(f"  M5: {len(self.bars_m5):,}  M15: {len(self.bars_m15):,}  "
              f"M30: {len(self.bars_m30):,}  H1: {len(self.bars_h1):,}  "
              f"H4: {len(self.bars_h4):,}")

        self.m5_idx  = self._build(bars_1min, self.bars_m5,  5)
        self.m15_idx = self._build(bars_1min, self.bars_m15, 15)
        self.m30_idx = self._build(bars_1min, self.bars_m30, 30)
        self.h1_idx  = self._build(bars_1min, self.bars_h1,  60)
        self.h4_idx  = self._build(bars_1min, self.bars_h4,  240)

    @staticmethod
    def _build(bars_1min: list, bars_tf: list, tf_min: int) -> list:
        """result[i] = index of last COMPLETED tf bar when processing 1-min bar i."""
        result = [-1] * len(bars_1min)
        j = 0
        n_tf = len(bars_tf)
        tf_delta = timedelta(minutes=tf_min)
        for i, b in enumerate(bars_1min):
            while j < n_tf and bars_tf[j].timestamp + tf_delta <= b.timestamp:
                j += 1
            result[i] = j - 1
        return result


# ── Vol regime tracker ────────────────────────────────────────────────────────

class VolRegimeTracker:
    def __init__(self):
        self._atr_history: list = []
        self.current_h1_atr: float = 0.0
        self._last_fed_idx: int = -1

    def feed_up_to(self, h1_bars: list, h1_up_to_idx: int) -> None:
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


# ── Sweep detection ───────────────────────────────────────────────────────────

def detect_bearish_sweep(htf_bars: list, up_to_idx: int):
    """Returns (detected, sweep_ts). Looks for last bar sweeping a prior swing high."""
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


# ── CHoCH helpers ─────────────────────────────────────────────────────────────

def get_latest_swing_low(bars: list, up_to_idx: int, radius: int) -> float | None:
    for i in range(up_to_idx - radius, radius - 1, -1):
        lo = bars[i].low
        if all(bars[i + k].low >= lo for k in range(-radius, radius + 1) if k != 0):
            return lo
    return None


def choch_fired(bars_confirm: list, conf_idx: int, atr_confirm: float, radius: int) -> bool:
    if conf_idx < radius * 2 + 1:
        return False
    swing_low = get_latest_swing_low(bars_confirm, conf_idx - 1, radius)
    if swing_low is None:
        return False
    return bars_confirm[conf_idx].close < swing_low - CHOCH_ATR_MULT * atr_confirm


# ── Core cascade simulation ───────────────────────────────────────────────────

def run_cascade(bars_1min: list, precomp: PrecomputedTFs, cfg: CascadeConfig):
    """Run one cascade combination. Returns (pnl, n_trades)."""
    # Variant overrides
    swing_radius   = 1 if cfg.variant == "r1"    else SWING_RADIUS
    min_gap_ratio  = MIN_GAP_ATR_TIGHT if cfg.variant == "tight" else MIN_GAP_ATR_RATIO

    # HTF selection
    if cfg.sweep_tf_hours == 1:
        htf_bars, htf_idx_arr = precomp.bars_h1, precomp.h1_idx
    else:
        htf_bars, htf_idx_arr = precomp.bars_h4, precomp.h4_idx

    # Confirm TF selection
    if cfg.confirm_tf_min == 5:
        conf_bars, conf_idx_arr = precomp.bars_m5, precomp.m5_idx
    elif cfg.confirm_tf_min == 15:
        conf_bars, conf_idx_arr = precomp.bars_m15, precomp.m15_idx
    else:
        conf_bars = conf_idx_arr = None

    # Entry TF selection
    if cfg.entry_tf_min == 5:
        ent_bars, ent_idx_arr = precomp.bars_m5, precomp.m5_idx
    elif cfg.entry_tf_min == 15:
        ent_bars, ent_idx_arr = precomp.bars_m15, precomp.m15_idx
    elif cfg.entry_tf_min == 30:
        ent_bars, ent_idx_arr = precomp.bars_m30, precomp.m30_idx
    else:
        ent_bars, ent_idx_arr = bars_1min, None

    max_hold_min     = MAX_HOLD_MIN[cfg.entry_tf_min]
    max_pend_min     = MAX_PENDING_MIN[cfg.entry_tf_min]
    sweep_expire_hrs = cfg.sweep_tf_hours * 6

    vol_tracker = VolRegimeTracker()
    last_h1_idx = -1

    sweep_active     = False
    sweep_expires_ts = datetime.min.replace(tzinfo=timezone.utc)
    confirm_done     = False
    last_htf_idx     = -1
    last_conf_idx    = -1
    last_scanned_ent_idx = -1

    pending      = False
    active       = False
    entry_price  = sl_price = tp_price = 0.0
    pending_ts   = entry_ts = datetime.min.replace(tzinfo=timezone.utc)

    pnl      = []
    n_trades = 0

    for i, bar in enumerate(bars_1min):
        cur_ts = bar.timestamp

        # Update vol regime on new H1 bar
        cur_h1_idx = precomp.h1_idx[i]
        if cur_h1_idx > last_h1_idx and cur_h1_idx >= 0:
            last_h1_idx = cur_h1_idx
            vol_tracker.feed_up_to(precomp.bars_h1, cur_h1_idx)

        regime_high = vol_tracker.is_high()

        # Advance active trade
        if active:
            elapsed = (cur_ts - entry_ts).total_seconds() / 60
            if bar.high >= sl_price:
                pnl.append((entry_price - sl_price) * MNQ_DOLLAR)
                n_trades += 1; active = False
            elif bar.low <= tp_price:
                pnl.append((entry_price - tp_price) * MNQ_DOLLAR)
                n_trades += 1; active = False
            elif elapsed >= max_hold_min:
                pnl.append((entry_price - bar.close) * MNQ_DOLLAR)
                n_trades += 1; active = False
            if active:
                continue

        # Advance pending limit order
        if pending:
            elapsed = (cur_ts - pending_ts).total_seconds() / 60
            if bar.high >= entry_price:
                active = True; pending = False; entry_ts = cur_ts
                continue
            elif elapsed >= max_pend_min:
                pending = False
            else:
                continue

        # Per-bar gates
        if not is_market_open(cur_ts):
            continue
        if is_tuesday_et(cur_ts):
            continue
        if regime_high:
            continue

        # Expire sweep
        if sweep_active and cur_ts >= sweep_expires_ts:
            sweep_active = False
            confirm_done = False
            last_conf_idx = -1
            last_scanned_ent_idx = -1

        # Detect new sweep
        if not sweep_active:
            cur_htf_idx = htf_idx_arr[i]
            if cur_htf_idx >= 4 and cur_htf_idx > last_htf_idx:
                last_htf_idx = cur_htf_idx
                detected, sweep_ts = detect_bearish_sweep(htf_bars, cur_htf_idx)
                if detected:
                    sweep_active     = True
                    sweep_expires_ts = sweep_ts + timedelta(hours=sweep_expire_hrs)
                    confirm_done     = (cfg.confirm_tf_min == 0)
                    last_conf_idx    = -1
                    last_scanned_ent_idx = -1

        if not sweep_active:
            continue

        # CHoCH confirmation
        if not confirm_done and cfg.confirm_tf_min > 0:
            cur_conf_idx = conf_idx_arr[i]
            if cur_conf_idx >= swing_radius * 2 + 1 and cur_conf_idx > last_conf_idx:
                last_conf_idx = cur_conf_idx
                conf_atr = calc_atr(conf_bars, cur_conf_idx + 1)
                if choch_fired(conf_bars, cur_conf_idx, conf_atr, swing_radius):
                    confirm_done = True

        if not confirm_done:
            continue

        # FVG detection on entry TF
        cur_ent_idx = i if ent_idx_arr is None else ent_idx_arr[i]
        if cur_ent_idx < 2 or cur_ent_idx <= last_scanned_ent_idx:
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
        if h1_atr > 0 and gap < min_gap_ratio * h1_atr:
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
        return {"pf": None, "n": 0, "wr": 0.0, "sharpe": None, "dsr": False, "t": None}
    pf = profit_factor(pnl)
    n  = len(pnl)
    wr = sum(1 for p in pnl if p > 0) / n * 100
    mean_pnl = np.mean(pnl)
    std_pnl  = np.std(pnl, ddof=1)
    trading_days   = (bars_1min[-1].timestamp - bars_1min[0].timestamp).days * (252 / 365)
    trades_per_day = n / max(trading_days, 1)
    annual_trades  = trades_per_day * 252
    sharpe = (mean_pnl / std_pnl * np.sqrt(annual_trades)) if std_pnl > 0 else None
    t_stat = (mean_pnl / (std_pnl / np.sqrt(n))) if std_pnl > 0 else None
    dsr    = (t_stat is not None and t_stat > BONFERRONI_T_CRIT)
    return {"pf": pf, "n": n_trades, "wr": wr, "sharpe": sharpe, "dsr": dsr, "t": t_stat}


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(results: list, bars_1min: list) -> str:
    n_combos = len(results)
    lines = []
    lines.append("=" * 78)
    lines.append("S16 Multi-Timeframe Factorial DOE — Program C Phase 2a (Round 2)")
    lines.append(f"Data: {DATA_CSV}")
    lines.append(f"Bars: {len(bars_1min):,}  "
                 f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})")
    lines.append(f"Pre-cutoff only — NO holdout access  |  {n_combos} combinations")
    lines.append("=" * 78)
    lines.append("")

    hdr = f"{'Combination':<24} {'PF':>6}  {'N':>4}  {'WR%':>5}  {'Sharpe':>6}  {'t':>5}  DSR  Note"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    R1_REF = {  # R1 results for delta display
        "H4·NoConf·M15": 1.0753, "H4·M5·M15": 1.1102,
        "H4·M15·M15": 1.1155,   "H4·M5·M1": 1.0400,
    }
    for cfg, stats in results:
        pf_s  = f"{stats['pf']:.4f}" if stats['pf'] is not None else "  N/A "
        wr_s  = f"{stats['wr']:.1f}" if stats['n'] > 0 else "  N/A"
        sh_s  = f"{stats['sharpe']:.2f}" if stats['sharpe'] is not None else "   N/A"
        t_s   = f"{stats['t']:.2f}" if stats['t'] is not None else "  N/A"
        dsr_s = " * " if stats['dsr'] else "   "
        note = ""
        if cfg.name in R1_REF and stats['pf'] is not None:
            delta = stats['pf'] - R1_REF[cfg.name]
            note = f"R1Δ{delta:+.4f}"
        elif cfg.variant == "r1":
            note = "radius=1"
        elif cfg.variant == "tight":
            note = "gap≥0.25"
        lines.append(
            f"{cfg.name:<24} {pf_s:>6}  {stats['n']:>4}  {wr_s:>5}  "
            f"{sh_s:>6}  {t_s:>5}  {dsr_s}  {note}"
        )

    lines.append("")
    lines.append(f"DSR * = t-stat > {BONFERRONI_T_CRIT} (Bonferroni alpha/{n_combos})")
    lines.append("")

    # Main effects (Round 2 perspective)
    lines.append("── Main Effects (Round 2) ───────────────────────────────────────────────")

    def avg_pf(subset):
        pfs = [s["pf"] for _, s in subset
               if s["pf"] is not None and not np.isinf(s["pf"])]
        return float(np.mean(pfs)) if pfs else None

    r = results
    h1_rows   = [(c, s) for c, s in r if c.sweep_tf_hours == 1]
    h4_rows   = [(c, s) for c, s in r if c.sweep_tf_hours == 4]
    none_rows = [(c, s) for c, s in r if c.confirm_tf_min == 0 and c.variant != "tight"]
    m5c_rows  = [(c, s) for c, s in r if c.confirm_tf_min == 5  and c.variant != "r1"]
    m5r_rows  = [(c, s) for c, s in r if c.confirm_tf_min == 5  and c.variant == "r1"]
    m15c_rows = [(c, s) for c, s in r if c.confirm_tf_min == 15 and c.variant != "r1"]
    m15r_rows = [(c, s) for c, s in r if c.confirm_tf_min == 15 and c.variant == "r1"]
    m1e_rows  = [(c, s) for c, s in r if c.entry_tf_min == 1]
    m5e_rows  = [(c, s) for c, s in r if c.entry_tf_min == 5]
    m15e_rows = [(c, s) for c, s in r if c.entry_tf_min == 15 and c.variant != "tight"]
    m30e_rows = [(c, s) for c, s in r if c.entry_tf_min == 30]

    def pfs(v): return f"{v:.4f}" if v is not None else "N/A"

    h1_pf   = avg_pf(h1_rows);   h4_pf   = avg_pf(h4_rows)
    no_pf   = avg_pf(none_rows); m5c_pf  = avg_pf(m5c_rows)
    m15c_pf = avg_pf(m15c_rows); m5r_pf  = avg_pf(m5r_rows);  m15r_pf = avg_pf(m15r_rows)
    m1e_pf  = avg_pf(m1e_rows);  m5e_pf  = avg_pf(m5e_rows)
    m15e_pf = avg_pf(m15e_rows); m30e_pf = avg_pf(m30e_rows)

    def spread(*vals):
        vs = [v for v in vals if v is not None]
        return max(vs) - min(vs) if len(vs) >= 2 else 0

    sweep_sp   = spread(h1_pf, h4_pf)
    conf_sp    = spread(no_pf, m5c_pf, m15c_pf)
    entry_sp   = spread(m1e_pf, m5e_pf, m15e_pf, m30e_pf)

    lines.append(f"  Sweep TF:    H1={pfs(h1_pf)}  H4={pfs(h4_pf)}  (spread {sweep_sp:.4f})")
    lines.append(f"  Confirm TF:  none={pfs(no_pf)}  M5={pfs(m5c_pf)}  M15={pfs(m15c_pf)}  (spread {conf_sp:.4f})")
    lines.append(f"               M5(r=1)={pfs(m5r_pf)}  M15(r=1)={pfs(m15r_pf)}  ← radius-1 variants")
    lines.append(f"  Entry TF:    M1={pfs(m1e_pf)}  M5={pfs(m5e_pf)}  M15={pfs(m15e_pf)}  M30={pfs(m30e_pf)}  (spread {entry_sp:.4f})")
    lines.append("")

    # Rank by PF
    ranked = sorted(
        [(cfg, s) for cfg, s in results if s["pf"] is not None and not np.isinf(s["pf"])],
        key=lambda x: x[1]["pf"], reverse=True
    )
    lines.append("── Top 5 by PF ──────────────────────────────────────────────────────────")
    for cfg, s in ranked[:5]:
        lines.append(f"  {cfg.name:<24}  PF={s['pf']:.4f}  N={s['n']}")
    lines.append("")

    spreads = {"Sweep TF": sweep_sp, "Confirm TF": conf_sp, "Entry TF": entry_sp}
    dominant = max(spreads, key=lambda k: spreads[k])
    lines.append(f"  Dominant factor: {dominant} (spread {spreads[dominant]:.4f})")
    lines.append(f"  Round 3 recommendation: Expand {dominant} or pre-register top combo for holdout.")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nS16 Multi-Timeframe Factorial DOE — Program C Phase 2a (Round 2)")
    print("Running on PRE-CUTOFF data only — no holdout access\n")

    print(f"Loading: {DATA_CSV}")
    bars_1min = load_bars(DATA_CSV)
    print(f"Loaded {len(bars_1min):,} bars  "
          f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})\n")

    print("Pre-resampling to M5, M15, M30, H1, H4...")
    precomp = PrecomputedTFs(bars_1min)
    print("Building completion index arrays... done\n")

    results = []
    n = len(COMBINATIONS)
    for k, cfg in enumerate(COMBINATIONS, 1):
        print(f"[{k:2d}/{n}] {cfg.name:<24}", end="", flush=True)
        if cfg.variant:
            print(f" [{cfg.variant}]", end="", flush=True)
        pnl, n_trades = run_cascade(bars_1min, precomp, cfg)
        stats = compute_stats(pnl, n_trades, bars_1min)
        results.append((cfg, stats))
        pf_s = f"{stats['pf']:.4f}" if stats['pf'] is not None else "N/A"
        print(f"  PF={pf_s}  N={n_trades}  WR={stats['wr']:.1f}%  "
              f"{'DSR*' if stats['dsr'] else ''}")

    print()
    report = build_report(results, bars_1min)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"s16_factorial_r2_{stamp}.txt"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")
    print("Round 2 complete. See report for Round 3 / pre-registration recommendation.")


if __name__ == "__main__":
    main()
