"""TSC-1 backtest: futures term-structure / roll-yield carry, time-series timing.

Pre-registration: _bmad-output/preregistration_term_structure_carry.md
(frozen BEFORE this script was written / run against real numbers).

Reads data/term_structure/{raw_contract_bars,contract_meta}.csv (fetched by
tools/fetch_term_structure_data.py), builds a front/next contract panel per
instrument, computes annualized roll yield, applies the frozen time-series
carry-timing rule at a weekly rebalance, sweeps the one pre-declared deadzone
knob `d` on the dev window ONLY, locks it, and evaluates once on the holdout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "term_structure"
BARS_CSV = DATA_DIR / "raw_contract_bars.csv"
META_CSV = DATA_DIR / "contract_meta.csv"
OUT_MD = Path(__file__).resolve().parents[1] / "_bmad-output" / "tsc1_backtest_results.md"

COST_PER_RT = {"MGC": 6.00, "SIL": 7.00, "MHG": 4.00, "PL": 34.00, "MNQ": 2.24}
ROLL_AHEAD_DAYS = 5          # trading days before expiry -> roll to next contract
REBALANCE = "W-FRI"          # weekly, Friday-anchored
DEV_START, DEV_END = "2021-01-01", "2025-12-31"
HOLD_START, HOLD_END = "2026-01-01", "2026-09-03"
D_GRID = [0.0, 3.0, 5.0, 8.0, 12.0]   # percent, annualized


def load_panels() -> dict[str, pd.DataFrame]:
    bars = pd.read_csv(BARS_CSV, parse_dates=["date"])
    meta = pd.read_csv(META_CSV, parse_dates=["expiration_date"])
    meta["expiration_date"] = meta["expiration_date"].dt.tz_localize(None)
    meta["point_value"] = pd.to_numeric(meta["point_value"], errors="coerce")
    bars = bars.merge(meta[["symbol", "expiration_date", "point_value"]], on="symbol", how="left")
    bars = bars.dropna(subset=["expiration_date"])
    panels = {}
    for root, g in bars.groupby("root"):
        panels[root] = g.sort_values(["date", "expiration_date"]).reset_index(drop=True)
    return panels


def build_front_next(panel: pd.DataFrame) -> pd.DataFrame:
    """For each date, pick the front (nearest unexpired) and next contract."""
    rows = []
    dates = sorted(panel["date"].unique())
    for d in dates:
        day = panel[panel["date"] == d]
        # only contracts not yet expired as of this date
        live = day[day["expiration_date"] > d].sort_values("expiration_date")
        if len(live) < 2:
            continue
        front, nxt = live.iloc[0], live.iloc[1]
        days_between = (nxt["expiration_date"] - front["expiration_date"]).days
        if days_between <= 0:
            continue
        roll_yield = np.log(front["close"] / nxt["close"]) * 365.0 / days_between
        rows.append({
            "date": d,
            "front_symbol": front["symbol"], "front_close": front["close"],
            "front_expiry": front["expiration_date"],
            "next_symbol": nxt["symbol"], "next_close": nxt["close"],
            "point_value": front["point_value"],
            "days_to_front_expiry": (front["expiration_date"] - d).days,
            "roll_yield_annualized_pct": roll_yield * 100.0,
        })
    return pd.DataFrame(rows)


def simulate(fn: pd.DataFrame, root: str, d_pct: float) -> pd.DataFrame:
    """Daily P&L in USD for one instrument given deadzone d (percent)."""
    fn = fn.sort_values("date").reset_index(drop=True)
    cost = COST_PER_RT[root]

    # weekly rebalance signal: sample roll_yield at each week's last available day,
    # forward-filled as the held position sign for every day until the next sample.
    weekly = fn.set_index("date")["roll_yield_annualized_pct"].resample(REBALANCE).last()
    sig = pd.Series(0, index=weekly.index, dtype=int)
    sig[weekly > d_pct] = 1
    sig[weekly < -d_pct] = -1
    sig = sig.reindex(fn["date"], method="ffill").fillna(0).astype(int).values
    fn = fn.copy()
    fn["position"] = sig
    fn["prev_position"] = fn["position"].shift(1).fillna(0).astype(int)

    # detect contract rolls (front_symbol changes vs prior row)
    fn["prev_front_symbol"] = fn["front_symbol"].shift(1)
    fn["is_roll"] = (fn["front_symbol"] != fn["prev_front_symbol"]) & fn["prev_front_symbol"].notna()

    fn["prev_front_close"] = fn["front_close"].shift(1)
    # price P&L: only valid (non-roll) days use same-contract price diff; on a roll
    # day there is no same-contract diff to take (the front symbol just changed), so
    # price P&L on the roll day itself is 0 by construction -- the roll cost below is
    # the only P&L event that day. This avoids a fake same-day cross-contract jump.
    price_diff = np.where(fn["is_roll"], 0.0, fn["front_close"] - fn["prev_front_close"])
    fn["price_pnl"] = fn["prev_position"] * fn["point_value"] * price_diff

    # cost events: (a) a weekly rebalance changes position, (b) a roll happens while
    # a nonzero position is being carried through it.
    position_changed = fn["position"] != fn["prev_position"]
    rebalance_cost = np.where(position_changed, cost, 0.0)
    roll_cost = np.where(fn["is_roll"] & (fn["prev_position"] != 0), cost, 0.0)
    fn["cost"] = rebalance_cost + roll_cost
    fn["pnl"] = fn["price_pnl"] - fn["cost"]
    return fn


def portfolio_stats(daily_pnl: pd.Series) -> dict:
    daily_pnl = daily_pnl.dropna()
    if daily_pnl.empty or daily_pnl.std(ddof=0) == 0:
        return {"sharpe": 0.0, "pf": float("nan"), "total_pnl": daily_pnl.sum(),
                "n_days": len(daily_pnl), "n_nonzero_days": int((daily_pnl != 0).sum())}
    sharpe = daily_pnl.mean() / daily_pnl.std(ddof=0) * np.sqrt(252)
    gains = daily_pnl[daily_pnl > 0].sum()
    losses = -daily_pnl[daily_pnl < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    return {"sharpe": sharpe, "pf": pf, "total_pnl": daily_pnl.sum(),
            "n_days": len(daily_pnl), "n_nonzero_days": int((daily_pnl != 0).sum())}


def run_universe(panels: dict[str, pd.DataFrame], d_pct: float,
                  start: str, end: str) -> tuple[pd.Series, dict[str, dict]]:
    per_instrument_pnl = {}
    per_instrument_stats = {}
    for root, panel in panels.items():
        fn = build_front_next(panel)
        if fn.empty:
            continue
        sim = simulate(fn, root, d_pct)
        sim = sim.set_index("date")
        window = sim.loc[(sim.index >= start) & (sim.index <= end), "pnl"]
        if window.empty:
            continue
        per_instrument_pnl[root] = window
        per_instrument_stats[root] = portfolio_stats(window)
    if not per_instrument_pnl:
        return pd.Series(dtype=float), {}
    combined = pd.concat(per_instrument_pnl, axis=1).fillna(0.0)
    portfolio_daily = combined.sum(axis=1)
    return portfolio_daily, per_instrument_stats


def main() -> None:
    panels = load_panels()
    print("Instruments with data:", {r: len(p) for r, p in panels.items()})

    print(f"\n=== Sweep d on DEV window {DEV_START}..{DEV_END} (5 values, pre-declared grid) ===")
    sweep_results = []
    for d_pct in D_GRID:
        port, per_inst = run_universe(panels, d_pct, DEV_START, DEV_END)
        stats = portfolio_stats(port)
        sweep_results.append((d_pct, stats))
        print(f"d={d_pct:5.1f}%  Sharpe={stats['sharpe']:+.3f}  PF={stats['pf']:.3f}  "
              f"total_pnl=${stats['total_pnl']:+.2f}  n_days={stats['n_days']}")

    best_d, best_stats = max(sweep_results, key=lambda t: t[1]["sharpe"])
    print(f"\nLOCKED d = {best_d}% (max dev Sharpe = {best_stats['sharpe']:+.3f})")

    gate0_pass = best_stats["sharpe"] > 0.5 and best_stats["pf"] > 1.15
    print(f"\nGate 0 (dev, frozen bar: Sharpe>0.5 AND PF>1.15): "
          f"{'PASS' if gate0_pass else 'FAIL'}  "
          f"(Sharpe={best_stats['sharpe']:+.3f}, PF={best_stats['pf']:.3f})")

    holdout_port, holdout_per_inst = run_universe(panels, best_d, HOLD_START, HOLD_END)
    holdout_stats = portfolio_stats(holdout_port)
    gate1_pass = None
    if gate0_pass:
        gate1_pass = holdout_stats["pf"] > 1.0 and (
            np.sign(holdout_stats["sharpe"]) == np.sign(best_stats["sharpe"]) or best_stats["sharpe"] == 0
        )
        print(f"\nGate 1 (holdout {HOLD_START}..{HOLD_END}, spent once, frozen bar: PF>1.0 AND same-sign Sharpe): "
              f"{'PASS' if gate1_pass else 'FAIL'}  "
              f"(Sharpe={holdout_stats['sharpe']:+.3f}, PF={holdout_stats['pf']:.3f}, "
              f"n_days={holdout_stats['n_days']})")
    else:
        print("\nGate 0 FAILED -> holdout is NOT spent for the decision "
              "(reported below for transparency only, per no-peeking discipline "
              "this number does not count toward a PASS).")

    # per-instrument breakdown at the locked d, dev window
    _, dev_per_inst = run_universe(panels, best_d, DEV_START, DEV_END)

    lines = []
    lines.append("# TSC-1 Backtest Results\n")
    lines.append(f"Pre-registration: `_bmad-output/preregistration_term_structure_carry.md`\n")
    lines.append(f"Run date: 2026-09-03. Data: TradeStation daily settle, "
                 f"{sum(len(p) for p in panels.values())} raw bar-rows across "
                 f"{sum(p['symbol'].nunique() for p in panels.values())} contract-months, "
                 f"5 instruments.\n")
    lines.append("## Deadzone sweep (dev window, pre-declared grid, locked on max Sharpe)\n")
    lines.append("| d (%) | Sharpe | PF | Total P&L | Days |")
    lines.append("|---|---|---|---|---|")
    for d_pct, stats in sweep_results:
        lines.append(f"| {d_pct:.1f} | {stats['sharpe']:+.3f} | {stats['pf']:.3f} | "
                      f"${stats['total_pnl']:+,.2f} | {stats['n_days']} |")
    lines.append(f"\n**Locked d = {best_d}%**\n")
    lines.append(f"## Gate 0 (dev window {DEV_START}..{DEV_END}): "
                 f"{'PASS' if gate0_pass else 'FAIL'}\n")
    lines.append(f"Sharpe={best_stats['sharpe']:+.3f}, PF={best_stats['pf']:.3f}, "
                 f"total P&L=${best_stats['total_pnl']:+,.2f}, "
                 f"trading days={best_stats['n_days']}\n")
    lines.append("### Per-instrument (dev window, locked d)\n")
    lines.append("| Instrument | Sharpe | PF | Total P&L | Nonzero-position days |")
    lines.append("|---|---|---|---|---|")
    for root, s in dev_per_inst.items():
        lines.append(f"| {root} | {s['sharpe']:+.3f} | {s['pf']:.3f} | "
                      f"${s['total_pnl']:+,.2f} | {s['n_nonzero_days']} |")
    if gate0_pass:
        lines.append(f"\n## Gate 1 (holdout {HOLD_START}..{HOLD_END}, spent once): "
                     f"{'PASS' if gate1_pass else 'FAIL'}\n")
        lines.append(f"Sharpe={holdout_stats['sharpe']:+.3f}, PF={holdout_stats['pf']:.3f}, "
                     f"total P&L=${holdout_stats['total_pnl']:+,.2f}, "
                     f"trading days={holdout_stats['n_days']}\n")
        lines.append("### Per-instrument (holdout, locked d)\n")
        lines.append("| Instrument | Sharpe | PF | Total P&L | Nonzero-position days |")
        lines.append("|---|---|---|---|---|")
        for root, s in holdout_per_inst.items():
            lines.append(f"| {root} | {s['sharpe']:+.3f} | {s['pf']:.3f} | "
                          f"${s['total_pnl']:+,.2f} | {s['n_nonzero_days']} |")
    else:
        lines.append(f"\n## Gate 1: NOT EVALUATED (Gate 0 failed -- holdout not spent per pre-registration)\n")
        lines.append(f"For transparency only (does not count toward PASS/FAIL): holdout at locked d would have "
                     f"shown Sharpe={holdout_stats['sharpe']:+.3f}, PF={holdout_stats['pf']:.3f}, "
                     f"total P&L=${holdout_stats['total_pnl']:+,.2f}.\n")

    lines.append("\n## Verdict\n")
    if gate0_pass and gate1_pass:
        lines.append("**PASS.** Both gates cleared. Per the pre-registration, this licenses "
                      "a backtest-evidence conclusion that the mechanism survives this shop's "
                      "cost structure at this scale -- not an automatic live-deployment decision "
                      "(MGC/SIL costs are still unmeasured estimates; PL stays non-combine-deployable "
                      "standalone).")
    elif gate0_pass and not gate1_pass:
        lines.append("**Gate 0 PASS, Gate 1 FAIL.** The mechanism shows a dev-window edge but did not "
                      "hold up on the (short, ~8-month) 2026 holdout. Per the pre-registration this is "
                      "terminal for TSC-1 as specified -- no re-sweep, no new deadzone grid.")
    else:
        lines.append("**FAIL at Gate 0.** The time-series carry-timing mechanism, as specified, does "
                      "not clear this shop's own cost-adjusted bar on this universe. Per the "
                      "pre-registration this is terminal for TSC-1 as specified -- no re-sweep, "
                      "no new deadzone grid, no added instruments under this seal.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
