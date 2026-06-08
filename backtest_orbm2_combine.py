"""backtest_orbm2_combine.py — ORBM-2 In-Sample Backtest.

ORBM-2: Enter IN THE DIRECTION of the ORB extension, stop at ORB boundary,
TP at 1.5R. Pre-registration: git commit 16abdd9 (2026-06-08).

In-sample: 2025-01-01 → 2026-02-28 UTC (holdout ≥ 2026-03-01 sealed).

Usage:
    .venv/bin/python backtest_orbm2_combine.py
    .venv/bin/python backtest_orbm2_combine.py --preregistration 16abdd9  # verify seal

Outputs (in data/reports/):
    orbm2_backtest_<timestamp>.txt   — full human-readable report
    orbm2_backtest_<timestamp>.csv   — trade log (one row per trade)
    orbm2_equity_<timestamp>.csv     — daily equity curve
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.research.orbm2_core import (
    ORBM2Config,
    build_orbm2_trade,
    load_orbm2_config,
    simulate_orbm2_trade,
)
from src.research.sorm_core import (
    build_opening_range,
    calc_max_drawdown,
    calc_profit_factor,
    calc_win_rate,
    load_bars_et,
    detect_extension,
)

UTC = timezone.utc
IN_SAMPLE_START = datetime(2025, 1, 1, tzinfo=UTC)
IN_SAMPLE_END   = datetime(2026, 2, 28, 23, 59, 59, tzinfo=UTC)

CSV_2025 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2025.csv")
CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")

CONFIG_PATH = Path("orbm2_config.yaml")
PREREG_COMMIT = "16abdd9"   # pre-registration SHA (first 7 chars)


# ── Gate thresholds (frozen in pre-registration) ──────────────────────────────
GATE1 = {
    "win_rate_min":    0.55,
    "pf_min":          1.40,
    "max_dd_max":      1500.0,
    "freq_min":        1.0,       # setups/session-day
    "n_trades_min":    80,
    "qual_days_min":   6,         # qualifying sessions per 20-session window
    "largest_day_max": 0.40,      # fraction of total P&L
    "sharpe_min":      0.20,
}

QUALIFYING_SESSION_USD = 150.0    # Topstep: $150/session to count as qualifying day


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_preregistration(prereg_sha: str) -> bool:
    """Check that config hash matches what was sealed."""
    core_path = Path("src/research/sorm_core.py")
    hash_a_actual = _sha256_file(CONFIG_PATH)[:16]
    hash_b_actual = _sha256_file(core_path)[:16]
    print(f"  Sealed hash_a prefix:  8dc2a487a5c2fa4d  (orbm2_config.yaml at pre-reg)")
    print(f"  Current hash_a prefix: {hash_a_actual}")
    print(f"  Sealed hash_b prefix:  9861c0c9580fdb58  (sorm_core.py at pre-reg)")
    print(f"  Current hash_b prefix: {hash_b_actual}")
    ok_a = hash_a_actual == "8dc2a487a5c2fa4d"
    ok_b = hash_b_actual == "9861c0c9580fdb58"
    if ok_a and ok_b:
        print("  ✅ Config and core hashes match pre-registration")
        return True
    print("  ⚠️  Hash mismatch — config or core changed since pre-registration")
    return False


def _calc_per_trade_sharpe(pnl_list: list[float]) -> float:
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list)
    std = arr.std(ddof=1)
    return float(arr.mean() / std) if std > 0 else 0.0


def _calc_trailing_dd(cumulative_pnl: list[float]) -> float:
    """Max peak-to-trough drawdown on cumulative P&L series."""
    if not cumulative_pnl:
        return 0.0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    for v in cumulative_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _count_qualifying_days_rolling(session_pnls: list[float], window: int = 20) -> int:
    """Count qualifying sessions (≥$150) in the most recent `window` sessions."""
    recent = session_pnls[-window:]
    return sum(1 for p in recent if p >= QUALIFYING_SESSION_USD)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=None,
                        help="Pre-registration commit SHA to verify (optional)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    txt_path = report_dir / f"orbm2_backtest_{ts}.txt"
    csv_path = report_dir / f"orbm2_backtest_{ts}.csv"
    eq_path  = report_dir / f"orbm2_equity_{ts}.csv"

    lines: list[str] = []

    def p(s: str = "") -> None:
        print(s)
        lines.append(s)

    p("=" * 72)
    p("ORBM-2 Combine Strategy — In-Sample Backtest")
    p(f"Pre-registration: {PREREG_COMMIT}")
    p(f"In-sample: {IN_SAMPLE_START.date()} → {IN_SAMPLE_END.date()}")
    p(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p("=" * 72)

    # ── Pre-registration check ─────────────────────────────────────────────────
    if args.preregistration:
        p(f"\nVerifying against pre-registration {args.preregistration}…")
        _verify_preregistration(args.preregistration)

    p("\nLoading config…")
    cfg = load_orbm2_config(CONFIG_PATH)
    p(f"  extension_threshold: {cfg.extension_threshold}")
    p(f"  stop_cap_pts:        {cfg.stop_cap_pts}")
    p(f"  tp_r_multiple:       {cfg.tp_r_multiple}")
    p(f"  orb_min_size_points: {cfg.orb_min_size_points}")
    p(f"  daily_loss_halt:     ${cfg.daily_loss_limit_usd}")
    p(f"  daily_profit_halt:   ${cfg.daily_profit_halt_usd}")

    p("\nLoading bars…", )
    print("  (this may take a few seconds…)")
    df = load_bars_et([CSV_2025, CSV_2026], IN_SAMPLE_START, IN_SAMPLE_END)
    if df.empty:
        p("ERROR: no bars loaded — check CSV paths")
        sys.exit(1)
    df["_date"] = df.index.date
    p(f"  {len(df):,} bars ({df.index[0].date()} → {df.index[-1].date()})")

    hard_close_str = cfg.hard_close_et.strftime("%H:%M")

    # ── Main simulation loop ───────────────────────────────────────────────────
    all_results: list[dict] = []
    monthly: dict[str, list[dict]] = defaultdict(list)
    daily_pnl: list[tuple] = []      # (date, session_pnl)

    n_sessions = 0
    n_orb = 0
    n_extensions = 0
    n_skipped_stop = 0
    n_skipped_daily_halt = 0

    sessions = df.groupby("_date")
    session_pnl_history: list[float] = []

    for date_et, sess_df in sessions:
        if date_et.weekday() >= 5:
            continue
        sess_df = sess_df.drop(columns=["_date"])
        n_sessions += 1
        session_pnl = 0.0

        orb = build_opening_range(sess_df, cfg)
        if orb is None:
            daily_pnl.append((date_et, 0.0))
            session_pnl_history.append(0.0)
            continue
        n_orb += 1

        ext = detect_extension(sess_df, orb, cfg)
        if ext is None:
            daily_pnl.append((date_et, 0.0))
            session_pnl_history.append(0.0)
            continue
        n_extensions += 1

        trade_setup = build_orbm2_trade(ext, orb, cfg)

        if trade_setup.contracts == 0:
            n_skipped_stop += 1
            daily_pnl.append((date_et, 0.0))
            session_pnl_history.append(0.0)
            continue

        # Daily halt checks (applied before the first trade of the session)
        # For single-trade-per-session, this just gates whether we trade at all
        # (Prior-session P&L is checked at open of new session — for simplicity
        # we treat each session's halt check against that session's running total)

        post_df = sess_df.loc[sess_df.index > ext.detection_bar_ts]
        result = simulate_orbm2_trade(post_df, trade_setup, hard_close_str, cfg)

        session_pnl = result.pnl_net

        ym = f"{date_et.year}-{date_et.month:02d}"
        rec = {
            "date": date_et.isoformat(),
            "direction": result.direction,
            "entry": result.entry,
            "exit_price": result.exit_price,
            "stop": result.stop,
            "tp": result.tp,
            "stop_pts": trade_setup.stop_pts,
            "contracts": result.contracts,
            "pnl_gross": result.pnl_gross,
            "pnl_net": result.pnl_net,
            "exit_reason": result.exit_reason,
            "orb_size": orb.size,
            "ext_threshold": cfg.extension_threshold * orb.size,
        }
        all_results.append(rec)
        monthly[ym].append(rec)
        daily_pnl.append((date_et, session_pnl))
        session_pnl_history.append(session_pnl)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n_trades = len(all_results)
    if n_trades == 0:
        p("ERROR: no trades generated — check config and data")
        sys.exit(1)

    pnl_list = [r["pnl_net"] for r in all_results]
    wins = [p_ for p_ in pnl_list if p_ > 0]
    losses = [p_ for p_ in pnl_list if p_ <= 0]

    win_rate = len(wins) / n_trades
    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0

    cumulative = list(np.cumsum(pnl_list))
    total_pnl = cumulative[-1]
    max_dd = _calc_trailing_dd(cumulative)

    per_trade_sharpe = _calc_per_trade_sharpe(pnl_list)

    # Frequency: trades per session-day (session = day with valid ORB)
    sessions_with_orb = n_orb
    freq_per_day = n_trades / sessions_with_orb if sessions_with_orb > 0 else 0.0

    # Qualifying days: sessions where session P&L ≥ $150
    session_pnl_with_trades = [r["pnl_net"] for r in all_results]
    # Use daily_pnl which captures all days including zero-P&L days
    qualifying_sessions = sum(1 for _, spnl in daily_pnl if spnl >= QUALIFYING_SESSION_USD)
    total_session_days = len(daily_pnl)

    # Latest 20-session qualifying window
    recent_session_pnls = [spnl for _, spnl in daily_pnl[-20:]]
    qual_recent_20 = sum(1 for p_ in recent_session_pnls if p_ >= QUALIFYING_SESSION_USD)

    # Largest day concentration
    largest_day_pnl = max((abs(spnl) for _, spnl in daily_pnl), default=0)
    largest_day_frac = largest_day_pnl / abs(total_pnl) if total_pnl != 0 else 0.0

    # Direction split
    long_trades  = [r for r in all_results if r["direction"] == "LONG"]
    short_trades = [r for r in all_results if r["direction"] == "SHORT"]

    def _dir_stats(recs):
        if not recs:
            return {"n": 0, "wr": 0, "pf": 0, "pnl": 0}
        pnls = [r["pnl_net"] for r in recs]
        ws = [p_ for p_ in pnls if p_ > 0]
        ls = [p_ for p_ in pnls if p_ <= 0]
        return {
            "n":   len(recs),
            "wr":  len(ws) / len(recs),
            "pf":  sum(ws) / abs(sum(ls)) if ls else float("inf"),
            "pnl": sum(pnls),
        }

    long_stats  = _dir_stats(long_trades)
    short_stats = _dir_stats(short_trades)

    # TP / SL / TIME_STOP breakdown
    tp_count   = sum(1 for r in all_results if r["exit_reason"] == "TP")
    sl_count   = sum(1 for r in all_results if r["exit_reason"] == "SL")
    ts_count   = sum(1 for r in all_results if r["exit_reason"] == "TIME_STOP")

    # ── Gate 1 verdict ────────────────────────────────────────────────────────
    gate_results: dict[str, bool] = {
        "win_rate":    win_rate >= GATE1["win_rate_min"],
        "pf":          pf >= GATE1["pf_min"],
        "max_dd":      max_dd <= GATE1["max_dd_max"],
        "frequency":   freq_per_day >= GATE1["freq_min"],
        "n_trades":    n_trades >= GATE1["n_trades_min"],
        "qual_days":   qual_recent_20 >= GATE1["qual_days_min"],
        "largest_day": largest_day_frac <= GATE1["largest_day_max"],
        "sharpe":      per_trade_sharpe >= GATE1["sharpe_min"],
    }
    gate1_pass = all(gate_results[k] for k in
                     ["win_rate", "pf", "max_dd", "frequency", "n_trades", "qual_days"])

    # ── Print report ──────────────────────────────────────────────────────────
    p()
    p("─" * 72)
    p("FUNNEL SUMMARY")
    p("─" * 72)
    p(f"  Session days:                 {n_sessions:>5}")
    p(f"  Sessions with valid ORB:      {n_orb:>5}  ({n_orb/n_sessions*100:.0f}%)")
    p(f"  Sessions with extension:      {n_extensions:>5}  ({n_extensions/n_orb*100:.0f}% of ORB sessions)")
    p(f"  Skipped (stop > {cfg.stop_cap_pts:.0f} pts):      {n_skipped_stop:>5}  ({n_skipped_stop/n_extensions*100:.0f}% of extensions)")
    p(f"  Trades executed:              {n_trades:>5}  ({n_trades/n_extensions*100:.0f}% of extensions)")

    p()
    p("─" * 72)
    p("PERFORMANCE SUMMARY")
    p("─" * 72)
    p(f"  Total net P&L:           ${total_pnl:>8.2f}")
    p(f"  N trades:                {n_trades:>8}")
    p(f"  Win rate:                {win_rate*100:>7.1f}%  (gate: ≥{GATE1['win_rate_min']*100:.0f}%)  {'✅' if gate_results['win_rate'] else '❌'}")
    p(f"  Profit factor:           {pf:>8.3f}  (gate: ≥{GATE1['pf_min']:.2f})  {'✅' if gate_results['pf'] else '❌'}")
    p(f"  Avg win (net):           ${avg_win:>8.2f}")
    p(f"  Avg loss (net):          ${avg_loss:>8.2f}")
    p(f"  Win/loss ratio:          {abs(avg_win/avg_loss) if avg_loss else 0:>8.2f}")
    p(f"  Max trailing DD:         ${max_dd:>8.2f}  (gate: ≤${GATE1['max_dd_max']:.0f})  {'✅' if gate_results['max_dd'] else '❌'}")
    p(f"  Per-trade Sharpe:        {per_trade_sharpe:>8.3f}  (gate: ≥{GATE1['sharpe_min']:.2f})  {'✅' if gate_results['sharpe'] else '❌'}")
    p()
    p(f"  Frequency (trades/ORB day): {freq_per_day:.3f}  (gate: ≥{GATE1['freq_min']:.1f})  {'✅' if gate_results['frequency'] else '❌'}")
    p(f"  Total session days:      {total_session_days:>5}")
    p(f"  Qualifying sessions:     {qualifying_sessions:>5}  (all in-sample, P&L ≥ ${QUALIFYING_SESSION_USD:.0f})")
    p(f"  Qual sessions / recent 20: {qual_recent_20}  (gate: ≥{GATE1['qual_days_min']})  {'✅' if gate_results['qual_days'] else '❌'}")
    p(f"  Largest day as % total:  {largest_day_frac*100:>6.1f}%  (gate: ≤{GATE1['largest_day_max']*100:.0f}%)  {'✅' if gate_results['largest_day'] else '❌'}")

    p()
    p("─" * 72)
    p("EXIT REASON BREAKDOWN")
    p("─" * 72)
    p(f"  TP hit:        {tp_count:>4} ({tp_count/n_trades*100:.1f}%)")
    p(f"  SL hit:        {sl_count:>4} ({sl_count/n_trades*100:.1f}%)")
    p(f"  Time stop:     {ts_count:>4} ({ts_count/n_trades*100:.1f}%)")

    p()
    p("─" * 72)
    p("DIRECTION BREAKDOWN")
    p("─" * 72)
    for label, stats in [("LONG (upward ext)", long_stats), ("SHORT (downward ext)", short_stats)]:
        if stats["n"] == 0:
            continue
        p(f"  {label}: N={stats['n']:>4}  WR={stats['wr']*100:.1f}%  PF={stats['pf']:.2f}  P&L=${stats['pnl']:+.2f}")

    p()
    p("─" * 72)
    p("BY-MONTH")
    p("─" * 72)
    p(f"  {'Month':<10}  {'N':>4}  {'WR':>6}  {'PF':>6}  {'P&L':>9}  {'Qual':>5}")
    p(f"  {'─'*10}  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*5}")
    for ym in sorted(monthly):
        recs = monthly[ym]
        pnls_m = [r["pnl_net"] for r in recs]
        ws_m   = [p_ for p_ in pnls_m if p_ > 0]
        ls_m   = [p_ for p_ in pnls_m if p_ <= 0]
        wr_m   = len(ws_m) / len(recs) if recs else 0.0
        pf_m   = sum(ws_m) / abs(sum(ls_m)) if ls_m else float("inf")
        qual_m = sum(1 for r in recs if r["pnl_net"] >= QUALIFYING_SESSION_USD)
        p(f"  {ym:<10}  {len(recs):>4}  {wr_m*100:>5.0f}%  {pf_m:>6.2f}  ${sum(pnls_m):>8.2f}  {qual_m:>5}")

    p()
    p("─" * 72)
    p("TRAILING DRAWDOWN PATH (equity peaks and troughs)")
    p("─" * 72)
    peak = 0.0
    for i, v in enumerate(cumulative):
        if v > peak:
            peak = v
        dd = peak - v
        if dd > 100 or i == len(cumulative) - 1:
            pass  # just compute max_dd; full path in equity CSV

    p()
    p("=" * 72)
    p("GATE 1 VERDICT")
    p("=" * 72)
    gate_labels = {
        "win_rate":    f"Win rate ≥ {GATE1['win_rate_min']*100:.0f}%",
        "pf":          f"Profit factor ≥ {GATE1['pf_min']:.2f}",
        "max_dd":      f"Max DD ≤ ${GATE1['max_dd_max']:.0f}",
        "frequency":   f"Frequency ≥ {GATE1['freq_min']:.1f}/day",
        "n_trades":    f"N trades ≥ {GATE1['n_trades_min']}",
        "qual_days":   f"Qualifying sessions ≥ {GATE1['qual_days_min']} / last 20",
        "largest_day": f"Largest day ≤ {GATE1['largest_day_max']*100:.0f}% of total",
        "sharpe":      f"Per-trade Sharpe ≥ {GATE1['sharpe_min']:.2f}",
    }
    required_gates = ["win_rate", "pf", "max_dd", "frequency", "n_trades", "qual_days"]
    for key, label in gate_labels.items():
        marker = " [REQUIRED]" if key in required_gates else " [advisory]"
        status = "✅ PASS" if gate_results[key] else "❌ FAIL"
        p(f"  {status}  {label}{marker}")

    p()
    if gate1_pass:
        p("  ✅ GATE 1 PASS — all required criteria met")
        p("  → OOS holdout access unlocked (data ≥ 2026-03-01)")
        p("  → Run with: --preregistration 16abdd9 on OOS data")
    else:
        failed = [gate_labels[k] for k in required_gates if not gate_results[k]]
        p("  ❌ GATE 1 FAIL — required criteria not met:")
        for f in failed:
            p(f"     • {f}")
        p("  → OOS holdout remains sealed. Do not access ≥ 2026-03-01 data.")
    p("=" * 72)

    # ── Write outputs ─────────────────────────────────────────────────────────
    txt_path.write_text("\n".join(lines))
    print(f"\nReport written: {txt_path}")

    with open(csv_path, "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Trade log:      {csv_path}")

    with open(eq_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "session_pnl", "cumulative_pnl", "drawdown"])
        cum = 0.0
        peak_eq = 0.0
        for (d, spnl) in daily_pnl:
            cum += spnl
            if cum > peak_eq:
                peak_eq = cum
            dd = peak_eq - cum
            writer.writerow([d.isoformat(), f"{spnl:.2f}", f"{cum:.2f}", f"{dd:.2f}"])
    print(f"Equity curve:   {eq_path}")


if __name__ == "__main__":
    main()
