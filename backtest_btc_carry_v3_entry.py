#!/usr/bin/env python3
"""
BTC Carry — v3 ENTRY-CONFIRMATION Backtest

Tests whether adding an entry-confirmation window (symmetric in spirit to the
hardened v3 exit) reduces transient-spike round-trips without materially
sacrificing return. The v3 LIVE exit is held CONSTANT across all arms; only the
ENTRY rule varies.

Arms (entry rule; exit identical = v3 live: 3-of-5 neg window OR 12-period
below-hurdle):
  H0  baseline  : enter on a SINGLE reading  ann > 10%      (current live logic)
  H1  primary   : enter when ann > 10% for >= 3 consecutive 8h periods (24h confirm)
  H2  control   : enter on a SINGLE reading  ann > 15%      (level lever)

Key differentiator metric: COST-LOSS ROUND-TRIPS = round-trips whose total net
P&L (carry collected minus 2x transition cost) is < 0. The entry-asymmetry
hypothesis predicts H1 strictly reduces these vs H0.

Pre-registration: _bmad-output/preregistration_btc_carry_entry_confirmation.md
                  (sealed in git BEFORE this script is run).
Funding data: data/kraken/PF_XBTUSD_funding_rate.csv (8h, Nov 2024 -> May 2026).
"""
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.research.strategy_core import calc_max_drawdown_pct, calc_sharpe

FUNDING_CSV = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
REPORTS_DIR = Path("data/reports")

# Frozen, shared across all arms (match live executor btc_carry_executor.py)
COST_BPS                  = 15.0     # per transition (round trip = 2x)
NEG_THRESHOLD             = -0.0001  # -0.01%/8h counts as a negative payment
NEG_WINDOW_SIZE           = 5        # v3 live exit: rolling window length
NEG_WINDOW_MIN_NEG        = 3        # v3 live exit: exit if >=3 of last 5 negative
BELOW_HURDLE_EXIT_PERIODS = 12       # v3 live exit: 12 consecutive below-hurdle
PERIODS_PER_YEAR          = 3 * 365
COST_FRAC                 = COST_BPS / 10_000.0

# Per-arm breakeven hold (days) to recoup a 30bps round trip at the arm's hurdle:
#   d_BE = 0.003 * 365 / hurdle_frac   (reported for context, not a gate)


def simulate(funding: pd.DataFrame, entry_level_pct: float,
             entry_confirm: int) -> pd.DataFrame:
    """v3 live exit held constant; entry = `entry_confirm` consecutive readings
    with annualized funding > entry_level_pct (entry_confirm=1 -> single reading)."""
    hurdle_exit = 10.0 / 100.0          # exit hurdle stays at the live 10% (unchanged)
    entry_level = entry_level_pct / 100.0
    cost = COST_FRAC

    f = funding.copy()
    f["ann_rate"] = f["funding_rate"] * 3 * 365

    above_entry_count = 0
    window = deque()
    below_hurdle_count = 0
    in_carry = False
    positions = []

    for _, row in f.iterrows():
        rate = row["funding_rate"]
        ann = row["ann_rate"]

        if not in_carry:
            window.clear()
            below_hurdle_count = 0
            # entry confirmation counter
            if ann > entry_level:
                above_entry_count += 1
            else:
                above_entry_count = 0
            new_pos = 1 if above_entry_count >= entry_confirm else 0
        else:
            above_entry_count = 0
            window.append(rate)
            if len(window) > NEG_WINDOW_SIZE:
                window.popleft()
            neg_in_window = sum(r < NEG_THRESHOLD for r in window)
            if ann < hurdle_exit:
                below_hurdle_count += 1
            else:
                below_hurdle_count = 0
            if neg_in_window >= NEG_WINDOW_MIN_NEG:
                new_pos = 0
            elif below_hurdle_count >= BELOW_HURDLE_EXIT_PERIODS:
                new_pos = 0
            else:
                new_pos = 1

        positions.append(new_pos)
        in_carry = bool(new_pos)

    f["position"] = positions
    f["pos_change"] = f["position"].diff().abs().fillna(f["position"].abs())
    f["carry_pnl"] = f["position"].shift(1).fillna(0) * f["funding_rate"]
    f["cost"] = f["pos_change"] * cost
    f["net_pnl"] = f["carry_pnl"] - f["cost"]
    f["equity"] = (1 + f["net_pnl"]).cumprod()
    return f


def per_trade_pnl(f: pd.DataFrame) -> list[float]:
    """Segment into round-trips; return each trip's total net P&L (fraction)."""
    pos = f["position"].tolist()
    net = f["net_pnl"].tolist()
    trips = []
    cur = None
    for i, p in enumerate(pos):
        if p == 1 and cur is None:
            cur = 0.0
        if cur is not None:
            cur += net[i]          # includes entry cost (this bar) + carry + exit cost
        if cur is not None and p == 0 and (i > 0 and pos[i - 1] == 1):
            trips.append(cur)
            cur = None
    if cur is not None:
        trips.append(cur)          # open trip at series end
    return trips


def score(f: pd.DataFrame, entry_level_pct: float, label: str) -> dict:
    net = f["net_pnl"].dropna()
    equity = f["equity"].dropna().tolist()
    n = len(net)
    n_trades = int(f["pos_change"].sum() / 2)
    trips = per_trade_pnl(f)
    cost_loss = sum(1 for t in trips if t < 0)
    daily_net = net.values.reshape(-1, 3).mean(axis=1) if len(net) >= 3 else net.values
    d_be = 0.003 * 365 / (10.0 / 100.0) if entry_level_pct <= 10 else 0.003 * 365 / (entry_level_pct / 100.0)
    return {
        "label": label,
        "n_trades": n_trades,
        "n_trips_segmented": len(trips),
        "cost_loss_trips": cost_loss,
        "pct_time_carry": int(f["position"].sum()) / n,
        "avg_ann_rate": float(f.loc[f["position"] == 1, "ann_rate"].mean()) if f["position"].sum() else 0.0,
        "ann_return": float((equity[-1] ** (PERIODS_PER_YEAR / n)) - 1) if equity[-1] > 0 else -1.0,
        "sharpe": calc_sharpe(daily_net.tolist()),
        "max_dd": calc_max_drawdown_pct(equity),
        "d_be_days": d_be,
    }


def verdict(r: dict) -> str:
    y = r["ann_return"] * 100
    dd = r["max_dd"]
    if y > 10.0 and dd < 0.05:
        return f"PASS (yield {y:+.1f}% > 10%, maxdd {dd*100:.2f}% < 5%)"
    if y < 5.0 or dd > 0.10:
        return f"FAIL (yield {y:+.1f}%, maxdd {dd*100:.2f}%)"
    return f"AMBIGUOUS (yield {y:+.1f}%, maxdd {dd*100:.2f}%)"


def main() -> None:
    if not FUNDING_CSV.exists():
        sys.exit(f"ERROR: {FUNDING_CSV} not found")
    df = pd.read_csv(FUNDING_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df["funding_rate"] = df["funding_rate"].astype(float)
    print(f"Loaded {len(df)} 8h rows ({df.index[0].date()} → {df.index[-1].date()})")

    arms = [
        ("H0  single >10%      ", 10.0, 1),
        ("H1  >=3 consec >10%  ", 10.0, 3),
        ("H2  single >15%      ", 15.0, 1),
    ]
    results = []
    for label, lvl, confirm in arms:
        f = simulate(df, lvl, confirm)
        results.append(score(f, lvl, label.strip()))

    W = 84
    lines = ["=" * W, "BTC CARRY — ENTRY-CONFIRMATION (v3 exit held constant)",
             f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", "=" * W]
    hdr = f"  {'Arm':<22} {'Trades':>7} {'CostLoss':>9} {'%Carry':>7} {'AvgAnn':>7} {'AnnRet':>8} {'Sharpe':>7} {'MaxDD':>7}"
    lines.append(hdr)
    lines.append("  " + "-" * (W - 2))
    for (label, _, _), r in zip(arms, results):
        lines.append(
            f"  {label:<22} {r['n_trades']:>7} {r['cost_loss_trips']:>9} "
            f"{r['pct_time_carry']*100:>6.1f}% {r['avg_ann_rate']*100:>6.1f}% "
            f"{r['ann_return']*100:>+7.1f}% {r['sharpe']:>7.2f} {r['max_dd']*100:>6.2f}%"
        )
    lines.append("")
    for (label, _, _), r in zip(arms, results):
        lines.append(f"  {label.strip():<22} d_BE={r['d_be_days']:.1f}d  {verdict(r)}")

    # Decision rule (H1 vs H0)
    h0, h1, h2 = results
    lines.append("")
    lines.append("  DECISION RULE (adopt H1 over H0 iff BOTH):")
    ret_ok = h1["ann_return"] >= 0.90 * h0["ann_return"]
    cl_ok = h1["cost_loss_trips"] < h0["cost_loss_trips"]
    lines.append(f"    (a) H1 ann_return ({h1['ann_return']*100:+.1f}%) >= 90% of H0 ({h0['ann_return']*100:+.1f}%): {'YES' if ret_ok else 'NO'}")
    lines.append(f"    (b) H1 cost-loss trips ({h1['cost_loss_trips']}) < H0 ({h0['cost_loss_trips']}): {'YES' if cl_ok else 'NO'}")
    lines.append(f"    => {'ADOPT H1 — pre-register as live entry change' if (ret_ok and cl_ok) else 'KEEP H0 — asymmetry was benign; no live change'}")
    lines.append("=" * W)
    report = "\n".join(lines)
    print("\n" + report)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"backtest_btc_carry_v3_entry_{ts}.txt"
    out.write_text(report)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
