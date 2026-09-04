"""TSMOM-1 backtest: time-series momentum on the TSC-1 universe.

Pre-registration: _bmad-output/preregistration_tsmom1_time_series_momentum.md
(frozen BEFORE this script was written / run against real numbers).

Reuses data/term_structure/{raw_contract_bars,contract_meta}.csv (fetched for
TSC-1) and TSC-1's own front-contract / roll / same-contract-diff-only P&L
convention. Builds a monthly trailing-return signal per instrument, sweeps
the one pre-declared lookback knob k on the dev window, locks it, evaluates
once on holdout, and reports two non-tradeable diagnostic baselines (§5 of
the pre-registration) alongside the primary result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "term_structure"
BARS_CSV = DATA_DIR / "raw_contract_bars.csv"
META_CSV = DATA_DIR / "contract_meta.csv"
OUT_MD = Path(__file__).resolve().parents[1] / "_bmad-output" / "tsmom1_backtest_results.md"

COST_PER_RT = {"MGC": 6.00, "SIL": 7.00, "MHG": 4.00, "PL": 34.00, "MNQ": 2.24}
ROLL_AHEAD_DAYS = 5
DEV_START, DEV_END = "2021-01-01", "2025-12-31"
HOLD_START, HOLD_END = "2026-01-01", "2026-09-03"
K_GRID = [1, 3, 6, 12]  # months, Moskowitz-Ooi-Pedersen's own robustness horizons


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


def build_front_series(panel: pd.DataFrame) -> pd.DataFrame:
    """For each date, the front (nearest unexpired) contract -- same convention as TSC-1."""
    rows = []
    dates = sorted(panel["date"].unique())
    for d in dates:
        day = panel[panel["date"] == d]
        live = day[day["expiration_date"] > d].sort_values("expiration_date")
        if live.empty:
            continue
        front = live.iloc[0]
        rows.append({"date": d, "front_symbol": front["symbol"], "front_close": front["close"],
                     "front_expiry": front["expiration_date"], "point_value": front["point_value"]})
    return pd.DataFrame(rows)


def add_same_contract_return(fn: pd.DataFrame) -> pd.DataFrame:
    """log return using only same-contract consecutive closes (TSC-1's roll convention)."""
    fn = fn.sort_values("date").reset_index(drop=True)
    fn["prev_front_symbol"] = fn["front_symbol"].shift(1)
    fn["prev_front_close"] = fn["front_close"].shift(1)
    fn["is_roll"] = (fn["front_symbol"] != fn["prev_front_symbol"]) & fn["prev_front_symbol"].notna()
    fn["log_ret"] = np.where(fn["is_roll"] | fn["prev_front_close"].isna(), 0.0,
                              np.log(fn["front_close"] / fn["prev_front_close"]))
    return fn


def simulate(fn: pd.DataFrame, root: str, k_months: int) -> pd.DataFrame:
    fn = fn.copy().set_index("date")
    cost = COST_PER_RT[root]

    # month-end trailing k-month cumulative log return, sampled at each month-end,
    # forward-filled as the held position until the next month-end.
    monthly_cum_ret = fn["log_ret"].resample("D").sum().fillna(0.0)
    trailing = monthly_cum_ret.rolling(f"{k_months * 30}D", min_periods=1).sum()
    month_end_signal = trailing.resample("ME").last()
    sig = pd.Series(0, index=month_end_signal.index, dtype=int)
    sig[month_end_signal > 0] = 1
    sig[month_end_signal < 0] = -1
    sig_daily = sig.reindex(fn.index, method="ffill").shift(1).fillna(0).astype(int)
    fn["position"] = sig_daily.values
    fn["prev_position"] = fn["position"].shift(1).fillna(0).astype(int)

    price_diff = np.where(fn["is_roll"], 0.0, fn["front_close"] - fn["prev_front_close"].fillna(fn["front_close"]))
    fn["price_pnl"] = fn["prev_position"] * fn["point_value"] * price_diff

    position_changed = fn["position"] != fn["prev_position"]
    rebalance_cost = np.where(position_changed, cost, 0.0)
    roll_cost = np.where(fn["is_roll"] & (fn["prev_position"] != 0), cost, 0.0)
    fn["cost"] = rebalance_cost + roll_cost
    fn["pnl"] = fn["price_pnl"] - fn["cost"]
    return fn.reset_index()


def simulate_always_long(fn: pd.DataFrame, root: str) -> pd.DataFrame:
    fn = fn.copy()
    cost = COST_PER_RT[root]
    fn["position"] = 1
    fn["prev_position"] = 1
    price_diff = np.where(fn["is_roll"], 0.0, fn["front_close"] - fn["prev_front_close"].fillna(fn["front_close"]))
    fn["price_pnl"] = fn["prev_position"] * fn["point_value"] * price_diff
    fn["cost"] = np.where(fn["is_roll"], cost, 0.0)
    fn["pnl"] = fn["price_pnl"] - fn["cost"]
    return fn


def simulate_mean_sign(fn: pd.DataFrame, root: str, mean_sign: int) -> pd.DataFrame:
    fn = fn.copy()
    cost = COST_PER_RT[root]
    fn["position"] = mean_sign
    fn["prev_position"] = mean_sign
    price_diff = np.where(fn["is_roll"], 0.0, fn["front_close"] - fn["prev_front_close"].fillna(fn["front_close"]))
    fn["price_pnl"] = fn["prev_position"] * fn["point_value"] * price_diff
    fn["cost"] = np.where(fn["is_roll"], cost, 0.0)
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


def run_universe(front_series: dict[str, pd.DataFrame], k_months: int,
                  start: str, end: str) -> tuple[pd.Series, dict[str, dict]]:
    per_instrument_pnl, per_instrument_stats = {}, {}
    for root, fn in front_series.items():
        sim = simulate(fn, root, k_months).set_index("date")
        window = sim.loc[(sim.index >= start) & (sim.index <= end), "pnl"]
        if window.empty:
            continue
        per_instrument_pnl[root] = window
        per_instrument_stats[root] = portfolio_stats(window)
    if not per_instrument_pnl:
        return pd.Series(dtype=float), {}
    combined = pd.concat(per_instrument_pnl, axis=1).fillna(0.0)
    return combined.sum(axis=1), per_instrument_stats


def run_baseline(front_series: dict[str, pd.DataFrame], mode: str, start: str, end: str) -> pd.Series:
    per_instrument_pnl = {}
    for root, fn in front_series.items():
        fn = add_same_contract_return(fn) if "is_roll" not in fn.columns else fn
        dev_mask = (fn["date"] >= DEV_START) & (fn["date"] <= DEV_END)
        if mode == "always_long":
            sim = simulate_always_long(fn, root)
        else:
            mean_ret = fn.loc[dev_mask, "log_ret"].mean()
            mean_sign = 1 if mean_ret > 0 else (-1 if mean_ret < 0 else 0)
            sim = simulate_mean_sign(fn, root, mean_sign)
        sim = sim.set_index("date")
        window = sim.loc[(sim.index >= start) & (sim.index <= end), "pnl"]
        if not window.empty:
            per_instrument_pnl[root] = window
    if not per_instrument_pnl:
        return pd.Series(dtype=float)
    return pd.concat(per_instrument_pnl, axis=1).fillna(0.0).sum(axis=1)


def main() -> None:
    panels = load_panels()
    front_series = {root: add_same_contract_return(build_front_series(p)) for root, p in panels.items()}
    print("Instruments with data:", {r: len(p) for r, p in front_series.items()})

    print(f"\n=== Sweep k on DEV window {DEV_START}..{DEV_END} (4 values, pre-declared grid) ===")
    sweep_results = []
    for k in K_GRID:
        port, _ = run_universe(front_series, k, DEV_START, DEV_END)
        stats = portfolio_stats(port)
        sweep_results.append((k, stats))
        print(f"k={k:2d}mo  Sharpe={stats['sharpe']:+.3f}  PF={stats['pf']:.3f}  "
              f"total_pnl=${stats['total_pnl']:+.2f}  n_days={stats['n_days']}")

    best_k, best_stats = max(sweep_results, key=lambda t: t[1]["sharpe"])
    print(f"\nLOCKED k = {best_k} months (max dev Sharpe = {best_stats['sharpe']:+.3f})")

    gate0_pass = best_stats["sharpe"] > 0.5 and best_stats["pf"] > 1.15
    print(f"\nGate 0 (dev, frozen bar: Sharpe>0.5 AND PF>1.15): "
          f"{'PASS' if gate0_pass else 'FAIL'}  "
          f"(Sharpe={best_stats['sharpe']:+.3f}, PF={best_stats['pf']:.3f})")

    holdout_port, holdout_per_inst = run_universe(front_series, best_k, HOLD_START, HOLD_END)
    holdout_stats = portfolio_stats(holdout_port)
    gate1_pass = None
    if gate0_pass:
        gate1_pass = holdout_stats["pf"] > 1.0 and (
            np.sign(holdout_stats["sharpe"]) == np.sign(best_stats["sharpe"]) or best_stats["sharpe"] == 0
        )
        print(f"\nGate 1 (holdout, spent once): {'PASS' if gate1_pass else 'FAIL'} "
              f"(Sharpe={holdout_stats['sharpe']:+.3f}, PF={holdout_stats['pf']:.3f})")
    else:
        print("\nGate 0 FAILED -> holdout NOT spent for the decision (shown below for transparency only).")

    _, dev_per_inst = run_universe(front_series, best_k, DEV_START, DEV_END)

    # sec5 mechanism-check controls, dev window, informational only
    always_long_dev = run_baseline(front_series, "always_long", DEV_START, DEV_END)
    mean_sign_dev = run_baseline(front_series, "mean_sign", DEV_START, DEV_END)
    al_stats = portfolio_stats(always_long_dev)
    ms_stats = portfolio_stats(mean_sign_dev)

    lines = ["# TSMOM-1 Backtest Results\n",
             "Pre-registration: `_bmad-output/preregistration_tsmom1_time_series_momentum.md`\n",
             f"Data: TSC-1's fetched TradeStation daily settle data, "
             f"{sum(len(p) for p in front_series.values())} front-contract daily rows, 5 instruments.\n",
             "## Lookback sweep (dev window, pre-declared grid {1,3,6,12} months, locked on max Sharpe)\n",
             "| k (months) | Sharpe | PF | Total P&L | Days |",
             "|---|---|---|---|---|"]
    for k, stats in sweep_results:
        lines.append(f"| {k} | {stats['sharpe']:+.3f} | {stats['pf']:.3f} | "
                      f"${stats['total_pnl']:+,.2f} | {stats['n_days']} |")
    lines.append(f"\n**Locked k = {best_k} months**\n")
    lines.append(f"## Gate 0 (dev window {DEV_START}..{DEV_END}): {'PASS' if gate0_pass else 'FAIL'}\n")
    lines.append(f"Sharpe={best_stats['sharpe']:+.3f}, PF={best_stats['pf']:.3f}, "
                 f"total P&L=${best_stats['total_pnl']:+,.2f}, trading days={best_stats['n_days']}\n")
    lines.append("### Per-instrument (dev window, locked k)\n")
    lines.append("| Instrument | Sharpe | PF | Total P&L | Nonzero-position days |")
    lines.append("|---|---|---|---|---|")
    for root, s in dev_per_inst.items():
        lines.append(f"| {root} | {s['sharpe']:+.3f} | {s['pf']:.3f} | "
                      f"${s['total_pnl']:+,.2f} | {s['n_nonzero_days']} |")

    if gate0_pass:
        lines.append(f"\n## Gate 1 (holdout {HOLD_START}..{HOLD_END}, spent once): "
                     f"{'PASS' if gate1_pass else 'FAIL'}\n")
        lines.append(f"Sharpe={holdout_stats['sharpe']:+.3f}, PF={holdout_stats['pf']:.3f}, "
                     f"total P&L=${holdout_stats['total_pnl']:+,.2f}, trading days={holdout_stats['n_days']}\n")
    else:
        lines.append("\n## Gate 1: NOT EVALUATED (Gate 0 failed -- holdout not spent per pre-registration)\n")
        lines.append(f"For transparency only: holdout at locked k would have shown "
                     f"Sharpe={holdout_stats['sharpe']:+.3f}, PF={holdout_stats['pf']:.3f}, "
                     f"total P&L=${holdout_stats['total_pnl']:+,.2f}.\n")

    lines.append("\n## Section 5 mechanism-check controls (dev window, non-tradeable, informational only)\n")
    lines.append("| Strategy | Sharpe | PF | Total P&L |")
    lines.append("|---|---|---|")
    lines.append(f"| Locked-k TSMOM (k={best_k}mo, primary) | {best_stats['sharpe']:+.3f} | "
                 f"{best_stats['pf']:.3f} | ${best_stats['total_pnl']:+,.2f} |")
    lines.append(f"| Always-long baseline | {al_stats['sharpe']:+.3f} | {al_stats['pf']:.3f} | "
                 f"${al_stats['total_pnl']:+,.2f} |")
    lines.append(f"| Full-sample-mean-sign baseline (look-ahead, non-tradeable) | {ms_stats['sharpe']:+.3f} | "
                 f"{ms_stats['pf']:.3f} | ${ms_stats['total_pnl']:+,.2f} |")

    lines.append("\n## Verdict\n")
    if gate0_pass and gate1_pass:
        verdict = ("**PASS.** Both gates cleared. Per the pre-registration this licenses a "
                   "backtest-evidence conclusion, not an automatic live-deployment decision "
                   "(MGC/SIL costs still unmeasured; PL non-combine-deployable standalone). "
                   "See §5 mechanism-check table above -- compare locked-k TSMOM against the "
                   "full-sample-mean-sign baseline before treating this as evidence of genuine "
                   "predictability rather than a volatility/carry artifact.")
    elif gate0_pass and not gate1_pass:
        verdict = ("**Gate 0 PASS, Gate 1 FAIL.** Dev-window edge did not hold on the short 2026 "
                   "holdout. Terminal for TSMOM-1 as specified -- no re-sweep.")
    else:
        verdict = ("**FAIL at Gate 0.** Time-series momentum, as specified, does not clear this "
                   "shop's own cost-adjusted bar on this universe. Per the pre-registration this "
                   "is terminal for TSMOM-1 as specified -- no re-sweep, no vol-scaling added, "
                   "no new instruments under this seal.")
    lines.append(verdict)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
