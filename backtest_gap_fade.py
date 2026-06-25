"""
backtest_gap_fade.py — GAP-1 Panic-Open Mean-Reversion Fade on MNQ

Pre-registration: _bmad-output/preregistration_gap_fade_panic_open.md

Hypothesis: Large overnight gaps (>=0.5% of prior RTH close) on MNQ mean-revert
during the first 3.5 RTH hours. Fade the gap at RTH open, exit at prior RTH close
(full fill), OR 2x gap SL, OR 13:00 ET time-stop. Exclude Fridays.

Usage:
    .venv/bin/python backtest_gap_fade.py

Output:
    data/reports/gap_fade_<timestamp>.csv  — one row per trade
    Console summary with Gate 0 verdict
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── FROZEN PARAMETERS (pre-registered; do NOT change) ─────────────────────────
GAP_MIN_PCT     = 0.005    # 0.5% minimum gap to trigger a trade
STOP_MULT       = 2.0      # stop = entry +/- STOP_MULT x gap_abs beyond open
TIME_STOP_HOUR  = 13       # close at market at open of 13:00 ET bar
EXCLUDE_DOW     = {4}      # 4 = Friday (0=Mon ... 4=Fri)
MIN_RTH_BARS    = 300      # minimum bars in prior session to compute prior close
MNQ_PV          = 2.0      # $2 per point per MNQ contract
CONTRACTS       = 1

# ── Gate 0 thresholds ─────────────────────────────────────────────────────────
GATE_N_MIN      = 60
GATE_PF_STRONG  = 1.40
GATE_PF_WEAK    = 1.10
GATE_WR_MIN     = 0.55
GATE_MAX_CON_L  = 10
GATE_WORST_MO   = -600.0   # worst single month P&L floor (gross, 1ct)

# ── Data paths ─────────────────────────────────────────────────────────────────
CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
REPORTS  = Path("data/reports")

ET_TZ = pytz.timezone("US/Eastern")
RTH_START = (9, 30)   # inclusive
RTH_END   = (16, 0)   # exclusive (last bar = 15:59)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    dfs = []
    for path in [CSV_2025, CSV_2026]:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        ts = df["timestamp"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df["timestamp"] = ts.dt.tz_convert(ET_TZ)
        dfs.append(df)
    out = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    return out.set_index("timestamp")


def is_rth(ts: pd.Timestamp) -> bool:
    h, m = ts.hour, ts.minute
    after_open = (h == RTH_START[0] and m >= RTH_START[1]) or h > RTH_START[0]
    before_close = h < RTH_END[0] or (h == RTH_END[0] and m < RTH_END[1])
    return after_open and before_close


# ── Session-level aggregation ──────────────────────────────────────────────────
def build_session_map(df: pd.DataFrame) -> dict:
    """Return {date: (prior_rth_close, rth_open_price, rth_open_dow)}."""
    rth = df[df.index.map(is_rth)].copy()
    rth["date_et"] = rth.index.date

    by_date = rth.groupby("date_et")
    rth_close = by_date["close"].last()
    rth_open  = by_date["open"].first()
    rth_bars  = by_date["close"].count()
    rth_dow   = by_date.apply(lambda g: g.index[0].weekday())

    sessions = {}
    dates = sorted(rth_close.index)
    for i in range(1, len(dates)):
        today     = dates[i]
        yesterday = dates[i - 1]
        if rth_bars[yesterday] < MIN_RTH_BARS:
            continue
        sessions[today] = {
            "prior_close": rth_close[yesterday],
            "rth_open":    rth_open[today],
            "dow":         rth_dow[today],
        }
    return sessions


# ── Per-day trade simulation ───────────────────────────────────────────────────
def simulate_day(day_bars: pd.DataFrame, direction: int,
                 entry: float, target: float, stop: float) -> tuple[str, float]:
    """Simulate one trade bar-by-bar after the RTH open bar.

    direction: +1 = long (gap down), -1 = short (gap up)
    Returns (outcome, pnl_points).
    """
    for ts, bar in day_bars.iterrows():
        # Time stop: close at open of 13:00 ET bar
        if ts.hour >= TIME_STOP_HOUR:
            return "time", direction * (bar["open"] - entry)

        if direction == -1:   # short
            if bar["low"]  <= target: return "fill", direction * (target - entry)
            if bar["high"] >= stop:   return "stop", direction * (stop   - entry)
        else:                         # long
            if bar["high"] >= target: return "fill", direction * (target - entry)
            if bar["low"]  <= stop:   return "stop", direction * (stop   - entry)

    # EOD without exit (should not happen given 13:00 time stop)
    return "eod", direction * (day_bars["close"].iloc[-1] - entry)


# ── Main backtest loop ─────────────────────────────────────────────────────────
def run(df: pd.DataFrame, sessions: dict) -> list[dict]:
    rth = df[df.index.map(is_rth)].copy()
    rth["date_et"] = rth.index.date

    trades = []
    for date_et, sess in sorted(sessions.items()):
        if sess["dow"] in EXCLUDE_DOW:
            continue

        prior_close = sess["prior_close"]
        rth_open    = sess["rth_open"]
        gap         = rth_open - prior_close
        gap_abs     = abs(gap)
        gap_pct     = gap_abs / prior_close

        if gap_pct < GAP_MIN_PCT:
            continue

        direction = -1 if gap > 0 else 1    # fade the gap
        entry     = rth_open
        target    = prior_close
        stop      = entry + direction * (-STOP_MULT * gap_abs)
        # cleaner sign logic:
        stop = (entry + STOP_MULT * gap_abs) if direction == -1 else (entry - STOP_MULT * gap_abs)

        day_bars = rth[rth["date_et"] == date_et]
        # Skip the opening bar itself (entry is at open of that bar)
        if len(day_bars) < 2:
            continue
        sim_bars = day_bars.iloc[1:]

        outcome, pnl_pts = simulate_day(sim_bars, direction, entry, target, stop)
        pnl_usd = pnl_pts * MNQ_PV * CONTRACTS

        trades.append({
            "date":        str(date_et),
            "dow":         ["Mon","Tue","Wed","Thu","Fri"][sess["dow"]],
            "gap_pct":     round(gap_pct * 100, 3),
            "gap_abs_pts": round(gap_abs, 2),
            "direction":   "short" if direction == -1 else "long",
            "entry":       round(entry, 2),
            "target":      round(target, 2),
            "stop":        round(stop, 2),
            "outcome":     outcome,
            "pnl_pts":     round(pnl_pts, 2),
            "pnl_usd":     round(pnl_usd, 2),
        })

    return trades


# ── Reporting ──────────────────────────────────────────────────────────────────
def report(trades: list[dict], label: str = "IS 2025") -> dict:
    if not trades:
        print("No trades generated.")
        return {}

    pnls_pts = np.array([t["pnl_pts"] for t in trades])
    pnls_usd = np.array([t["pnl_usd"] for t in trades])
    wins = pnls_pts > 0
    losses = pnls_pts < 0

    gross_w = pnls_usd[wins].sum()
    gross_l = abs(pnls_usd[losses].sum())
    pf = gross_w / gross_l if gross_l > 0 else float("inf")
    wr = wins.mean()

    # Max consecutive losses
    mc = cur = 0
    for w in wins:
        if not w:
            cur += 1; mc = max(mc, cur)
        else:
            cur = 0

    # Monthly P&L
    t = pd.DataFrame(trades)
    t["month"] = pd.to_datetime(t["date"]).dt.to_period("M")
    monthly = t.groupby("month")["pnl_usd"].sum()
    worst_mo = monthly.min()
    pos_months = (monthly > 0).sum()

    print(f"\n{'='*60}")
    print(f"GAP-1 PANIC-OPEN MEAN-REVERSION FADE — {label}")
    print(f"{'='*60}")
    print(f"N trades          : {len(trades)}")
    print(f"Win rate          : {wr*100:.1f}%  ({wins.sum()}W / {losses.sum()}L)")
    print(f"Gross PF          : {pf:.3f}")
    print(f"Total P&L (gross) : {pnls_pts.sum():.1f} pts  (${pnls_usd.sum():.0f})")
    print(f"Avg winner        : {pnls_pts[wins].mean():.1f} pts  (${pnls_usd[wins].mean():.0f})")
    print(f"Avg loser         : {pnls_pts[losses].mean():.1f} pts  (${pnls_usd[losses].mean():.0f})")
    print(f"Max consec losses : {mc}")
    print(f"Worst month       : ${worst_mo:.0f}")
    print(f"Positive months   : {pos_months}/{len(monthly)}")

    outcomes = t["outcome"].value_counts().to_dict()
    print(f"Exit breakdown    : {outcomes}")

    print(f"\nMonthly P&L:")
    for mo, pnl in monthly.items():
        sign = "+" if pnl >= 0 else ""
        print(f"  {mo}: {sign}${pnl:.0f}")

    # Gate 0 verdict
    n = len(trades)
    print(f"\n{'='*60}")
    print(f"GATE 0 DECISION")
    print(f"{'='*60}")
    checks = {
        f"N >= {GATE_N_MIN}": n >= GATE_N_MIN,
        f"WR >= {GATE_WR_MIN*100:.0f}%": wr >= GATE_WR_MIN,
        f"Max consec losses <= {GATE_MAX_CON_L}": mc <= GATE_MAX_CON_L,
        f"Worst month >= -${abs(GATE_WORST_MO):.0f}": worst_mo >= GATE_WORST_MO,
    }
    for check, passed in checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")

    all_secondary = all(checks.values())
    if n < GATE_N_MIN:
        verdict = "INSUFFICIENT_SAMPLE"
    elif pf <= GATE_PF_WEAK:
        verdict = "NO_EDGE"
    elif pf <= GATE_PF_STRONG:
        verdict = "WEAK_EDGE" if all_secondary else "WEAK_EDGE (secondary fail)"
    else:
        verdict = "STRONG_EDGE" if all_secondary else "STRONG_EDGE (secondary fail)"

    print(f"\n  PF = {pf:.3f}")
    print(f"  VERDICT: {verdict}")

    if "EDGE" in verdict:
        if "STRONG" in verdict:
            print("  → Proceed to live paper at 1ct. Decision after N>=30 live trades + 30 days.")
        else:
            print("  → Proceed to live paper at 1ct. Decision after N>=40 live trades + 30 days.")

    return {"n": n, "pf": pf, "wr": wr, "mc": mc, "worst_mo": worst_mo, "verdict": verdict}


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    sessions = build_session_map(df)
    print(f"Sessions with valid prior close: {len(sessions)}")

    trades = run(df, sessions)
    metrics = report(trades, "IS 2025+2026YTD (combined)")

    if not trades:
        raise SystemExit(0)

    # Write trade CSV
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS / f"gap_fade_{stamp}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)
    print(f"\nTrade log: {out}")
