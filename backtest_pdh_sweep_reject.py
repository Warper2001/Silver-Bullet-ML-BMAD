"""
backtest_pdh_sweep_reject.py — PDH-S1 Prior-Day High Sweep-and-Reject (SHORT-ONLY)

Pre-registration: _bmad-output/preregistration_pdh_sweep_reject_v1.md

Hypothesis: When MNQ sweeps the prior RTH session's High (PDH) in the first 120 min
of RTH and closes back below PDH within 5 bars, this signals a liquidity grab and
price should revert SHORT with PF >= 1.10 at 2:1 R:R.

DISCLOSURE: REJECT_BARS=5 was selected after observing IS results. All other
parameters are first-principles. See pre-registration for full disclosure.

Usage:
    .venv/bin/python backtest_pdh_sweep_reject.py            # IS only (Gate 0)
    .venv/bin/python backtest_pdh_sweep_reject.py --oos      # IS + OOS (requires Gate 0 PASS)
    .venv/bin/python backtest_pdh_sweep_reject.py --oos --log-access  # also log OOS access

Output:
    data/reports/pdh_sweep_IS_<timestamp>.csv
    data/reports/pdh_sweep_OOS_<timestamp>.csv  (if --oos)
"""
import argparse
import csv
import datetime as dt
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# ── FROZEN PARAMETERS (pre-registered; do NOT change) ─────────────────────────
SWEEP_WINDOW_MINS  = 120      # first 120 min of RTH (09:30-11:30 ET)
REJECT_BARS        = 5        # DISCLOSED: IS-derived. Bars after sweep to confirm rejection.
SL_BUFFER_PTS      = 20.0     # pts above sweep high for stop loss
TP_R_MULT          = 2.0      # take profit = 2x risk below entry
TIME_STOP_BARS     = 60       # bars from fill; close at market if not hit TP/SL
GAP_SKIP_PTS       = 200.0    # skip if RTH open > PDH + 200 pts (gap day)
MIN_RTH_BARS       = 300      # minimum bars per session to count as valid
COMMISSION_RT      = 4.00     # $ round-trip per contract
POINT_VALUE        = 2.00     # MNQ $/point
CONTRACTS          = 1
RISK_CAP_USD       = 400.0    # per-trade risk cap for combine safety

# ── Gate 0 thresholds ─────────────────────────────────────────────────────────
GATE0_PF_MIN = 1.10
GATE0_N_MIN  = 40
GATE0_WR_MIN = 0.35

# ── OOS early stop ─────────────────────────────────────────────────────────────
OOS_EARLY_STOP_N  = 10
OOS_EARLY_STOP_PF = 0.80

# ── Data paths (resolved from repo root, works from worktree) ─────────────────
_REPO        = Path(__file__).resolve().parents[3]
CSV_2025     = _REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
CSV_2026     = _REPO / "data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv"
REPORTS      = _REPO / "data/reports"
ACCESS_LOG   = _REPO / "data/sealed_holdout/ACCESS_LOG.md"

ET_TZ = pytz.timezone("US/Eastern")
RTH_START_MINS = 9 * 60 + 30    # 570
RTH_END_MINS   = 16 * 60         # 960


# ── Data loading ───────────────────────────────────────────────────────────────
def load_et(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    ts = df["timestamp"]
    ts = ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    df["timestamp"] = ts.dt.tz_convert(ET_TZ)
    df["ts_et"]       = df["timestamp"]
    df["hour_et"]     = df["ts_et"].dt.hour
    df["minute_et"]   = df["ts_et"].dt.minute
    df["date_et"]     = df["ts_et"].dt.date
    df["bar_mins_et"] = df["hour_et"] * 60 + df["minute_et"]
    df["is_rth"]      = (
        (df["bar_mins_et"] >= RTH_START_MINS) &
        (df["bar_mins_et"] < RTH_END_MINS)
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def build_pdh_map(rth_df: pd.DataFrame) -> dict:
    """Return {date: pdh} using prior calendar day's RTH high."""
    daily = (
        rth_df.groupby("date_et")
        .agg(rth_high=("high", "max"), n_bars=("high", "count"))
        .reset_index()
        .sort_values("date_et")
        .reset_index(drop=True)
    )
    pdh_map = {}
    for i in range(1, len(daily)):
        prior = daily.iloc[i - 1]
        if prior["n_bars"] >= MIN_RTH_BARS:
            pdh_map[daily.iloc[i]["date_et"]] = prior["rth_high"]
    return pdh_map


# ── Backtest simulation ────────────────────────────────────────────────────────
def run_backtest(rth_df: pd.DataFrame, pdh_map: dict, label: str) -> list[dict]:
    trades = []
    skipped_no_pdh = skipped_gap = 0

    for date_et in sorted(rth_df["date_et"].unique()):
        if date_et not in pdh_map:
            skipped_no_pdh += 1
            continue

        pdh = pdh_map[date_et]
        day_bars = rth_df[rth_df["date_et"] == date_et].reset_index(drop=True)

        if len(day_bars) < 60:
            continue

        rth_open = day_bars.iloc[0]["open"]
        if rth_open > pdh + GAP_SKIP_PTS:
            skipped_gap += 1
            continue

        in_watch   = False
        sweep_high = None
        sweep_idx  = None
        traded     = False

        for i, bar in day_bars.iterrows():
            if traded:
                break

            bm = bar["bar_mins_et"]

            # Phase 1: look for upside sweep within the kill-zone window
            # Always continue after Phase 1 processing — if we just set in_watch,
            # we skip reject-check on the sweep bar itself (next bar starts Phase 2)
            if not in_watch and bm < RTH_START_MINS + SWEEP_WINDOW_MINS:
                if bar["high"] > pdh:
                    in_watch   = True
                    sweep_high = bar["high"]
                    sweep_idx  = i
                continue

            # Phase 2: in sweep watch — look for rejection close within REJECT_BARS
            if in_watch:
                bars_since = i - sweep_idx
                if bars_since > REJECT_BARS:
                    in_watch = False
                    sweep_high = None
                    sweep_idx  = None
                    continue

                if bar["close"] < pdh:  # REJECTION CONFIRMED
                    entry  = bar["close"]
                    sl     = sweep_high + SL_BUFFER_PTS
                    risk   = sl - entry
                    if risk <= 0:
                        in_watch = False
                        continue

                    # Per-trade risk cap (combine safety)
                    risk_usd = risk * POINT_VALUE * CONTRACTS
                    if risk_usd > RISK_CAP_USD:
                        in_watch = False
                        continue

                    tp = entry - TP_R_MULT * risk

                    # Simulate exit from next bar onward
                    exit_price  = None
                    exit_reason = None
                    j_limit = min(i + 1 + TIME_STOP_BARS, len(day_bars))
                    for j in range(i + 1, j_limit):
                        fb = day_bars.iloc[j]
                        if fb["high"] >= sl:
                            exit_price  = sl
                            exit_reason = "SL"
                            break
                        if fb["low"] <= tp:
                            exit_price  = tp
                            exit_reason = "TP"
                            break
                    if exit_price is None:
                        j = min(i + TIME_STOP_BARS, len(day_bars) - 1)
                        exit_price  = day_bars.iloc[j]["close"]
                        exit_reason = "TIME"

                    pnl_pts = entry - exit_price        # positive = profit for short
                    pnl_usd = pnl_pts * POINT_VALUE * CONTRACTS - COMMISSION_RT

                    trades.append({
                        "window":      label,
                        "date":        str(date_et),
                        "direction":   "SHORT",
                        "pdh":         round(pdh, 2),
                        "sweep_high":  round(sweep_high, 2),
                        "entry":       round(entry, 2),
                        "sl":          round(sl, 2),
                        "tp":          round(tp, 2),
                        "exit":        round(exit_price, 2),
                        "exit_reason": exit_reason,
                        "risk_pts":    round(risk, 2),
                        "risk_usd":    round(risk_usd, 2),
                        "pnl_pts":     round(pnl_pts, 2),
                        "pnl_usd":     round(pnl_usd, 2),
                        "win":         pnl_pts > 0,
                    })
                    traded   = True
                    in_watch = False

    print(f"  Skipped (no prior PDH): {skipped_no_pdh}  |  Skipped (gap day): {skipped_gap}")
    return trades


# ── Reporting ──────────────────────────────────────────────────────────────────
def print_report(trades: list[dict], label: str) -> dict:
    if not trades:
        print(f"\n[{label}] NO TRADES GENERATED")
        return {"n": 0, "pf": 0.0, "wr": 0.0, "net": 0.0}

    t = pd.DataFrame(trades)
    n   = len(t)
    wr  = t["win"].mean()
    gw  = t[t["win"]]["pnl_usd"].sum()
    gl  = abs(t[~t["win"]]["pnl_usd"].sum())
    pf  = gw / gl if gl > 0 else float("inf")
    net = t["pnl_usd"].sum()

    # Max consecutive losses
    mc = cur = 0
    for w in t["win"]:
        if not w:
            cur += 1; mc = max(mc, cur)
        else:
            cur = 0

    # Monthly
    t["month"] = pd.to_datetime(t["date"]).dt.to_period("M")
    monthly     = t.groupby("month").agg(n=("win","count"), wr=("win","mean"), net=("pnl_usd","sum"))
    pos_months  = (monthly["net"] > 0).sum()

    print(f"\n{'='*60}")
    print(f"PDH-S1 SWEEP-REJECT (SHORT-ONLY) — {label}")
    print(f"{'='*60}")
    print(f"N trades          : {n}")
    print(f"Win rate          : {wr*100:.1f}%  ({t['win'].sum():.0f}W / {(~t['win']).sum():.0f}L)")
    print(f"Profit factor     : {pf:.4f}")
    print(f"Net P&L (1ct)     : ${net:.2f}")
    print(f"Avg win           : {t[t['win']]['pnl_pts'].mean():.2f} pts  (${t[t['win']]['pnl_usd'].mean():.0f})")
    print(f"Avg loss          : {t[~t['win']]['pnl_pts'].mean():.2f} pts  (${t[~t['win']]['pnl_usd'].mean():.0f})")
    print(f"Avg risk          : {t['risk_pts'].mean():.2f} pts  (${t['risk_usd'].mean():.0f})")
    print(f"Max risk          : {t['risk_pts'].max():.2f} pts  (${t['risk_usd'].max():.0f})")
    print(f"Max consec losses : {mc}")
    print(f"Exit breakdown    : {dict(t['exit_reason'].value_counts())}")
    print(f"Positive months   : {pos_months}/{len(monthly)}")

    print(f"\nMonthly breakdown:")
    for idx, row in monthly.iterrows():
        sign = "+" if row["net"] >= 0 else ""
        print(f"  {idx}: N={int(row['n']):3d}, WR={row['wr']*100:5.1f}%, Net={sign}${row['net']:7.2f}")

    return {"n": n, "pf": pf, "wr": wr, "net": net, "mc": mc}


def gate0_check(m: dict) -> bool:
    print(f"\n{'='*60}")
    print("GATE 0 CHECK (IS 2025)")
    print(f"{'='*60}")
    p_pf = m["pf"] >= GATE0_PF_MIN
    p_n  = m["n"]  >= GATE0_N_MIN
    p_wr = m["wr"] >= GATE0_WR_MIN
    print(f"  PF >= {GATE0_PF_MIN}: {m['pf']:.4f}  {'PASS' if p_pf else 'FAIL'}")
    print(f"  N  >= {GATE0_N_MIN}:  {m['n']}       {'PASS' if p_n else 'FAIL'}")
    print(f"  WR >= {GATE0_WR_MIN*100:.0f}%: {m['wr']*100:.1f}%    {'PASS' if p_wr else 'FAIL'}")
    passed = p_pf and p_n and p_wr
    print(f"\n  GATE 0: {'PASS — proceed to OOS' if passed else 'FAIL — do NOT access OOS'}")
    return passed


def oos_verdict(m: dict) -> None:
    print(f"\n{'='*60}")
    print("OOS VERDICT (2026 YTD)")
    print(f"{'='*60}")
    p_pf = m["pf"] >= GATE0_PF_MIN
    p_n  = m["n"]  >= 20
    p_net = m["net"] > 0
    print(f"  PF >= 1.10: {m['pf']:.4f}  {'PASS' if p_pf else 'FAIL'}")
    print(f"  N  >= 20:   {m['n']}       {'PASS' if p_n else 'INCONCLUSIVE'}")
    print(f"  Net > 0:    ${m['net']:.2f}  {'PASS' if p_net else 'FAIL'}")

    if m["n"] < OOS_EARLY_STOP_N:
        result = "INCONCLUSIVE (insufficient trades)"
    elif m["n"] >= OOS_EARLY_STOP_N and m["pf"] < OOS_EARLY_STOP_PF:
        result = "FAIL (early stop triggered)"
    elif p_pf and p_n and p_net:
        result = "PASS — proceed to combine deployment planning"
    elif not p_n:
        result = "INCONCLUSIVE — resume at N=20"
    else:
        result = "FAIL — strategy is IS-regime-dependent, archive"

    print(f"\n  OOS RESULT: {result}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oos",        action="store_true", help="Run OOS after IS Gate 0 pass")
    parser.add_argument("--log-access", action="store_true", help="Log OOS access to ACCESS_LOG.md")
    args = parser.parse_args()

    # ── IS: 2025 ──────────────────────────────────────────────────────────────
    print(f"Loading IS data: {CSV_2025}")
    df25    = load_et(CSV_2025)
    rth25   = df25[df25["is_rth"]].copy().reset_index(drop=True)
    pdh_map = build_pdh_map(rth25)

    print(f"\nRunning IS backtest (PDH map: {len(pdh_map)} days)...")
    is_trades  = run_backtest(rth25, pdh_map, "IS 2025")
    is_metrics = print_report(is_trades, "IN-SAMPLE 2025")
    gate_pass  = gate0_check(is_metrics)

    if is_trades:
        REPORTS.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPORTS / f"pdh_sweep_IS_{stamp}.csv"
        pd.DataFrame(is_trades).to_csv(out, index=False)
        print(f"\nIS trade log: {out}")

    if not gate_pass:
        print("\nGate 0 FAIL — OOS access not authorized. Stopping.")
        return

    if not args.oos:
        print("\n(Pass --oos to run out-of-sample 2026 backtest)")
        return

    # ── OOS: 2026 ─────────────────────────────────────────────────────────────
    print(f"\nLoading OOS data: {CSV_2026}")
    df26  = load_et(CSV_2026)

    # Build PDH map using combined data (2025 close needed for Jan 2026 first day)
    df_all  = pd.concat([df25, df26], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df_all["bar_mins_et"] = df_all["timestamp"].dt.hour * 60 + df_all["timestamp"].dt.minute
    df_all["is_rth"]      = (
        (df_all["bar_mins_et"] >= RTH_START_MINS) &
        (df_all["bar_mins_et"] < RTH_END_MINS)
    )
    df_all["date_et"] = df_all["timestamp"].dt.date
    rth_all   = df_all[df_all["is_rth"]].copy().reset_index(drop=True)
    pdh_all   = build_pdh_map(rth_all)

    cutoff = dt.date(2026, 1, 1)
    pdh_oos  = {d: v for d, v in pdh_all.items() if d >= cutoff}
    rth26    = df26[df26["is_rth"]].copy().reset_index(drop=True)

    print(f"\nRunning OOS backtest ({len(pdh_oos)} qualifying days)...")
    oos_trades  = run_backtest(rth26, pdh_oos, "OOS 2026")
    oos_metrics = print_report(oos_trades, "OUT-OF-SAMPLE 2026")

    # Check early stop
    if oos_metrics["n"] >= OOS_EARLY_STOP_N and oos_metrics["pf"] < OOS_EARLY_STOP_PF:
        print(f"\n⚠ EARLY STOP TRIGGERED: N={oos_metrics['n']}, PF={oos_metrics['pf']:.3f} < {OOS_EARLY_STOP_PF}")

    oos_verdict(oos_metrics)

    if oos_trades:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = REPORTS / f"pdh_sweep_OOS_{stamp}.csv"
        pd.DataFrame(oos_trades).to_csv(out, index=False)
        print(f"\nOOS trade log: {out}")

    if args.log_access and ACCESS_LOG.exists():
        with open(ACCESS_LOG, "a") as f:
            f.write(f"\n## PDH-S1 OOS Access — {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            f.write(f"Pre-reg: _bmad-output/preregistration_pdh_sweep_reject_v1.md\n")
            f.write(f"Gate 0 IS PF: {is_metrics['pf']:.4f} (PASS)\n")
            f.write(f"OOS: N={oos_metrics['n']}, PF={oos_metrics['pf']:.4f}, Net=${oos_metrics['net']:.2f}\n")


if __name__ == "__main__":
    main()
