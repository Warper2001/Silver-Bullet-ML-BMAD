"""
S20 H1-Sweep Frequency DOE — Program C Phase 2b (PIVOT)

PIVOT from H4 sweep: S18 and S19 returned insufficient_sample (N=2, N=5)
on the 2.5-month sealed holdout. Per the pre-committed S19 decision rule,
the H4 cascade trades too infrequently for the holdout window.

This screens H1 sweep + M15 entry variants on pre-cutoff data to identify
a candidate that generates ≥40 trades/year (→ ≥10 on holdout).

Architecture: H1 sweep (6h window) + optional M15 CHoCH + M15 FVG entry.
Fixed: M15 entry (Phase 2a finding). Vary: confirm ∈ {none, M15}, gap ∈ {0.15, 0.20, 0.25}.

Usage:
    .venv/bin/python s20_h1_m15_frequency_doe.py
"""

import csv
import math
import sys
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── Frozen parameters (identical to Phase 1 / S14–S19) ───────────────────────
SL_MULT           = 5.0
TP_MULT           = 6.0
ATR_PERIOD        = 20
VOL_LOOKBACK      = 120
VOL_THRESH        = 0.75
ATR_THRESHOLD     = 0.5
MAX_GAP_DOLLARS   = 60.0
MNQ_DOLLAR        = 2.0
ENTRY_PCT         = 0.5
MNQ_TICK          = 0.25
BEARISH_ONLY      = True
H1_BAR_CAP        = 3000
CHOCH_ATR_MULT    = 0.3
SWING_RADIUS      = 2

# Fixed for all S20 combinations
SWEEP_TF_HOURS  = 1    # H1 sweep (PIVOT from H4)
ENTRY_TF_MIN    = 15   # M15 entry (Phase 2a finding)
MAX_HOLD_MIN    = 180  # 12 × 15 min
MAX_PENDING_MIN = 240  # 16 × 15 min

# Bonferroni for 6 combinations at alpha=0.05
BONFERRONI_T_CRIT   = 2.64
MIN_N_FOR_CANDIDATE = 40   # need ≥40/yr → ≥10 on 2.5-month holdout

DATA_CSV    = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
REPORTS_DIR = Path("data/reports")
ET_TZ       = pytz.timezone("US/Eastern")

# ── Types ─────────────────────────────────────────────────────────────────────

Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close"])

CascadeConfig = namedtuple("CascadeConfig", [
    "sweep_tf_hours",  # fixed: 1
    "confirm_tf_min",  # 0=none, 15=M15
    "entry_tf_min",    # fixed: 15
    "gap_ratio",       # MIN_GAP_ATR_RATIO: 0.15 | 0.20 | 0.25
    "name",
])

COMBINATIONS = [
    # H1·NoConf·M15 — gap ratio sweep
    CascadeConfig(1, 0,  15, 0.15, "H1·NoConf·M15·g0.15"),   # Phase 1 w/ M15 entry
    CascadeConfig(1, 0,  15, 0.20, "H1·NoConf·M15·g0.20"),
    CascadeConfig(1, 0,  15, 0.25, "H1·NoConf·M15·g0.25"),
    # H1·M15·M15 — gap ratio sweep
    CascadeConfig(1, 15, 15, 0.15, "H1·M15·M15·g0.15"),
    CascadeConfig(1, 15, 15, 0.20, "H1·M15·M15·g0.20"),
    CascadeConfig(1, 15, 15, 0.25, "H1·M15·M15·g0.25"),
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


# ── Precomputed TF indices ────────────────────────────────────────────────────

class PrecomputedTFs:
    def __init__(self, bars_1min: list):
        self.bars_m15 = resample_bars(bars_1min, 15)
        self.bars_h1  = resample_bars(bars_1min, 60)
        print(f"  M15: {len(self.bars_m15):,}  H1: {len(self.bars_h1):,}")
        self.m15_idx = self._build(bars_1min, self.bars_m15, 15)
        self.h1_idx  = self._build(bars_1min, self.bars_h1,  60)

    @staticmethod
    def _build(bars_1min: list, bars_tf: list, tf_min: int) -> list:
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
            self.current_h1_atr = float(np.mean(trs))
            self._atr_history.append(self.current_h1_atr)
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

def get_latest_swing_low(bars: list, up_to_idx: int) -> float | None:
    r = SWING_RADIUS
    for i in range(up_to_idx - r, r - 1, -1):
        lo = bars[i].low
        if all(bars[i + k].low >= lo for k in range(-r, r + 1) if k != 0):
            return lo
    return None


def choch_fired(bars_confirm: list, conf_idx: int, atr_confirm: float) -> bool:
    r = SWING_RADIUS
    if conf_idx < r * 2 + 1:
        return False
    swing_low = get_latest_swing_low(bars_confirm, conf_idx - 1)
    if swing_low is None:
        return False
    return bars_confirm[conf_idx].close < swing_low - CHOCH_ATR_MULT * atr_confirm


# ── Core simulation ───────────────────────────────────────────────────────────

def run_cascade(bars_1min: list, precomp: PrecomputedTFs, cfg: CascadeConfig):
    """Run H1·M15-entry cascade with specified gap ratio. Returns (pnl, n_trades)."""
    min_gap_ratio = cfg.gap_ratio

    if cfg.confirm_tf_min == 15:
        conf_bars, conf_idx_arr = precomp.bars_m15, precomp.m15_idx
    else:
        conf_bars = conf_idx_arr = None

    vol_tracker    = VolRegimeTracker()
    last_h1_vol    = -1   # last H1 bar fed to vol regime
    last_htf_idx   = -1   # last H1 bar checked for sweep

    sweep_active     = False
    sweep_expires_ts = datetime.min.replace(tzinfo=timezone.utc)
    confirm_done     = False
    last_conf_idx    = -1
    last_scanned_m15 = -1

    pending     = False
    active      = False
    entry_price = sl_price = tp_price = 0.0
    pending_ts  = entry_ts = datetime.min.replace(tzinfo=timezone.utc)

    pnl      = []
    n_trades = 0

    for i, bar in enumerate(bars_1min):
        cur_ts = bar.timestamp

        # Update vol regime on each new completed H1 bar
        cur_h1_idx = precomp.h1_idx[i]
        if cur_h1_idx > last_h1_vol and cur_h1_idx >= 0:
            last_h1_vol = cur_h1_idx
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
            elif elapsed >= MAX_HOLD_MIN:
                pnl.append((entry_price - bar.close) * MNQ_DOLLAR)
                n_trades += 1; active = False
            if active:
                continue

        # Advance pending
        if pending:
            elapsed = (cur_ts - pending_ts).total_seconds() / 60
            if bar.high >= entry_price:
                active = True; pending = False; entry_ts = cur_ts
                continue
            elif elapsed >= MAX_PENDING_MIN:
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
            sweep_active     = False
            confirm_done     = False
            last_conf_idx    = -1
            last_scanned_m15 = -1
            last_htf_idx     = -1

        # Detect new H1 sweep (cur_h1_idx already computed above)
        if not sweep_active:
            if cur_h1_idx >= 4 and cur_h1_idx > last_htf_idx:
                last_htf_idx = cur_h1_idx
                detected, sweep_ts = detect_bearish_sweep(precomp.bars_h1, cur_h1_idx)
                if detected:
                    sweep_active     = True
                    sweep_expires_ts = sweep_ts + timedelta(hours=SWEEP_TF_HOURS * 6)
                    confirm_done     = (cfg.confirm_tf_min == 0)
                    last_conf_idx    = -1
                    last_scanned_m15 = -1

        if not sweep_active:
            continue

        # CHoCH confirmation
        if not confirm_done and cfg.confirm_tf_min > 0:
            cur_conf_idx = conf_idx_arr[i]
            if cur_conf_idx >= SWING_RADIUS * 2 + 1 and cur_conf_idx > last_conf_idx:
                last_conf_idx = cur_conf_idx
                conf_atr = calc_atr(conf_bars, cur_conf_idx + 1)
                if choch_fired(conf_bars, cur_conf_idx, conf_atr):
                    confirm_done = True

        if not confirm_done:
            continue

        # FVG detection on M15 entry TF
        cur_m15_idx = precomp.m15_idx[i]
        if cur_m15_idx < 2 or cur_m15_idx <= last_scanned_m15:
            continue
        last_scanned_m15 = cur_m15_idx

        c1 = precomp.bars_m15[cur_m15_idx - 2]
        c2 = precomp.bars_m15[cur_m15_idx - 1]
        c3 = precomp.bars_m15[cur_m15_idx]

        if not (c1.low > c3.high and c2.close < c2.open):
            continue
        gap = c1.low - c3.high
        if gap <= 0:
            continue
        entry_atr = calc_atr(precomp.bars_m15, cur_m15_idx + 1)
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
        return {
            "pf": None, "n": 0, "wr": 0.0,
            "sharpe": None, "t": None, "dsr": False, "ir": None,
        }
    pf  = profit_factor(pnl)
    n   = len(pnl)
    wr  = sum(1 for p in pnl if p > 0) / n * 100
    mn  = np.mean(pnl)
    std = np.std(pnl, ddof=1)
    trading_days  = (bars_1min[-1].timestamp - bars_1min[0].timestamp).days * (252 / 365)
    annual_trades = (n / max(trading_days, 1)) * 252
    sharpe = (mn / std * math.sqrt(annual_trades)) if std > 0 else None
    t_stat = (mn / (std / math.sqrt(n))) if std > 0 else None
    dsr    = (t_stat is not None and t_stat > BONFERRONI_T_CRIT)
    ir     = pf * math.sqrt(n_trades) if pf is not None and not math.isinf(pf) else None
    return {
        "pf": pf, "n": n_trades, "wr": wr,
        "sharpe": sharpe, "t": t_stat, "dsr": dsr, "ir": ir,
    }


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(results: list, bars_1min: list) -> str:
    lines = []
    W = 80
    lines.append("=" * W)
    lines.append("S20 H1-Sweep Frequency DOE — Program C Phase 2b (PIVOT)")
    lines.append(f"Data: {DATA_CSV}")
    lines.append(f"Bars: {len(bars_1min):,}  "
                 f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})")
    lines.append(f"Architecture: H1 sweep + M15 entry (fixed)  |  6 combinations")
    lines.append(f"Pre-cutoff only — NO holdout access")
    lines.append("=" * W)
    lines.append("")

    # Main results table
    hdr = (f"{'Combination':<26} {'PF':>6}  {'N':>4}  {'WR%':>5}  "
           f"{'Sharpe':>6}  {'t':>5}  {'IR':>5}  DSR")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for cfg, stats in results:
        pf_s  = f"{stats['pf']:.4f}" if stats['pf'] is not None else "   N/A"
        wr_s  = f"{stats['wr']:.1f}"  if stats['n'] > 0          else "  N/A"
        sh_s  = f"{stats['sharpe']:.2f}" if stats['sharpe'] is not None else "   N/A"
        t_s   = f"{stats['t']:.2f}"  if stats['t'] is not None   else "  N/A"
        ir_s  = f"{stats['ir']:.2f}" if stats['ir'] is not None  else "  N/A"
        dsr_s = " *" if stats['dsr'] else "  "
        lines.append(
            f"{cfg.name:<26} {pf_s:>6}  {stats['n']:>4}  {wr_s:>5}  "
            f"{sh_s:>6}  {t_s:>5}  {ir_s:>5}  {dsr_s}"
        )

    lines.append("")
    lines.append(f"DSR * = t > {BONFERRONI_T_CRIT} (Bonferroni alpha/6)  "
                 f"|  IR = PF × √N (information-ratio proxy)")
    lines.append("")

    # ── Dose-response tables ──────────────────────────────────────────────────
    lines.append("── Dose-Response: PF and N by gap_ratio ────────────────────────────────")
    lines.append("")

    groups = [
        ("H1·NoConf·M15",  0),
        ("H1·M15·M15",    15),
    ]

    for group_label, conf_min in groups:
        group = [(c, s) for c, s in results if c.confirm_tf_min == conf_min]
        if not group:
            continue
        lines.append(f"  {group_label}:")
        lines.append(f"  {'gap_ratio':>9}  {'PF':>6}  {'N':>4}  {'IR':>5}  "
                     f"{'~holdout N':>10}  {'trend':>6}")
        prev_pf = None
        for cfg, stats in sorted(group, key=lambda x: x[0].gap_ratio):
            pf_v = stats['pf']
            ir_v = stats['ir']
            pf_s = f"{pf_v:.4f}" if pf_v is not None else "  N/A "
            ir_s = f"{ir_v:.2f}" if ir_v is not None else "  N/A"
            holdout_n = f"~{stats['n'] * 0.26:.1f}" if stats['n'] > 0 else "  N/A"
            if prev_pf is not None and pf_v is not None:
                trend = "↑" if pf_v > prev_pf else ("↓" if pf_v < prev_pf else "→")
            else:
                trend = "  "
            lines.append(f"  {cfg.gap_ratio:>9.2f}  {pf_s:>6}  {stats['n']:>4}  "
                         f"{ir_s:>5}  {holdout_n:>10}  {trend:>6}")
            prev_pf = pf_v
        lines.append("")

    # ── Pre-registration candidate ────────────────────────────────────────────
    lines.append("── Pre-Registration Candidate ──────────────────────────────────────────")
    eligible = [
        (cfg, stats) for cfg, stats in results
        if stats['n'] >= MIN_N_FOR_CANDIDATE
        and stats['ir'] is not None
        and not math.isinf(stats['ir'])
    ]

    if eligible:
        best_cfg, best_stats = max(eligible, key=lambda x: x[1]['ir'])
        slug = best_cfg.name.lower().replace("·", "-").replace(".", "p")
        holdout_n_est = best_stats['n'] * 0.26
        lines.append("")
        lines.append(f"  PRE-REGISTRATION CANDIDATE: {best_cfg.name}")
        lines.append(f"    PF            = {best_stats['pf']:.4f}")
        lines.append(f"    N (in-sample) = {best_stats['n']}")
        lines.append(f"    IR            = {best_stats['ir']:.2f}  "
                     f"(highest PF×√N with N≥{MIN_N_FOR_CANDIDATE})")
        lines.append(f"    WR            = {best_stats['wr']:.1f}%")
        lines.append(f"    DSR           = {'YES *' if best_stats['dsr'] else 'no (screening only)'}")
        lines.append(f"    Expected holdout N ≈ {holdout_n_est:.1f} trades "
                     f"({'PASS ≥10' if holdout_n_est >= 10 else 'WARNING <10'})")
        lines.append("")
        lines.append(f"    Suggested holdout hypothesis: PF > 1.1350 (S12 p90 random baseline)")
        lines.append(f"    Suggested pre-registration:   _bmad-output/preregistration_{slug}.md")
        lines.append("")
        runners = sorted(eligible, key=lambda x: x[1]['ir'], reverse=True)[1:4]
        if runners:
            lines.append("  Runner-up candidates (N ≥ 40, sorted by IR):")
            for cfg2, s2 in runners:
                hn = s2['n'] * 0.26
                lines.append(
                    f"    {cfg2.name:<26}  PF={s2['pf']:.4f}  N={s2['n']}  "
                    f"IR={s2['ir']:.2f}  ~holdout N≈{hn:.1f}"
                )
    else:
        lines.append("")
        lines.append(f"  WARNING: No eligible candidates (all combos have N < {MIN_N_FOR_CANDIDATE}).")
        lines.append("  Consider g0.15 with H1 sweep or a different architecture.")

    lines.append("")
    lines.append("=" * W)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nS20 H1-Sweep Frequency DOE — Program C Phase 2b (PIVOT)")
    print("Running on PRE-CUTOFF data only — no holdout access\n")

    print(f"Loading: {DATA_CSV}")
    bars_1min = load_bars(DATA_CSV)
    print(f"Loaded {len(bars_1min):,} bars  "
          f"({bars_1min[0].timestamp.date()} → {bars_1min[-1].timestamp.date()})\n")

    print("Pre-resampling to M5, M15, H1...")
    precomp = PrecomputedTFs(bars_1min)
    print("Building completion index arrays... done\n")

    results = []
    n_total = len(COMBINATIONS)
    for k, cfg in enumerate(COMBINATIONS, 1):
        print(f"[{k}/{n_total}] {cfg.name:<26}", end="", flush=True)
        pnl, n_trades = run_cascade(bars_1min, precomp, cfg)
        stats = compute_stats(pnl, n_trades, bars_1min)
        results.append((cfg, stats))
        pf_s = f"{stats['pf']:.4f}" if stats['pf'] is not None else "N/A"
        ir_s = f"{stats['ir']:.2f}" if stats['ir'] is not None else "N/A"
        print(f"  PF={pf_s}  N={n_trades}  IR={ir_s}  "
              f"{'DSR*' if stats['dsr'] else ''}")

    print()
    report = build_report(results, bars_1min)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"s20_h1m15_doe_{stamp}.txt"
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")
    print("S20 complete. Review pre-registration candidate above.")


if __name__ == "__main__":
    main()
