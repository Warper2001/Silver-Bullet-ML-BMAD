"""S-M15CONF-15m: M15 Confirmation Filter at 15m — statistical power experiment.

Pre-registration: _bmad-output/preregistration_s_m15conf_15m.md
SHA: cfe7cb3
"""

import os
import tempfile
from datetime import date

import pandas as pd

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import (
    StrategyConfig,
    calc_profit_factor,
    calc_sharpe,
)

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
PRE_REG_SHA = "cfe7cb3"

BASELINE = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}
PF_THRESHOLD = 1.3
MIN_TRADES = 15


def load_and_resample() -> pd.DataFrame:
    bars = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    else:
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")

    return (
        bars.resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def run_backtest(bars_15m: pd.DataFrame, config: StrategyConfig) -> list:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            df_out = bars_15m.reset_index()
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")
            df_out.to_csv(f, index=False)
            tmp_path = f.name
        return BacktestEngine(tmp_path, config=config).run()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def metrics(trades_list: list) -> dict:
    pnls = [t.pnl_usd for t in trades_list]
    n = len(pnls)
    if n == 0:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "sharpe": 0.0, "exit_counts": {}}
    pf = calc_profit_factor(pnls)
    wr = sum(1 for p in pnls if p > 0) / n
    daily: dict[date, float] = {}
    for t in trades_list:
        d = t.timestamp_entry.date()
        daily[d] = daily.get(d, 0.0) + t.pnl_usd
    sharpe = calc_sharpe(list(daily.values()))
    exit_counts: dict[str, int] = {}
    for t in trades_list:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    return {
        "trades": n,
        "pf": pf,
        "wr": wr,
        "sharpe": sharpe,
        "exit_counts": exit_counts,
    }


def verify_m15_confirmed(m15_trades: list) -> bool:
    for t in m15_trades:
        if not t.m15_confirmed:
            print(f"  FAIL: trade not M15-confirmed: {t.timestamp_entry}")
            return False
    return True


def main() -> None:
    print("=== S-M15CONF-15m: M15 Confirmation Filter at 15m ===")
    print(f"Pre-registration SHA: {PRE_REG_SHA}")
    print(f"Bearish-only baseline: {BASELINE['trades']} trades | PF={BASELINE['pf']:.3f}")
    print(f"Pass criterion: PF > {PF_THRESHOLD} AND N ≥ {MIN_TRADES}")

    print("\nLoading and resampling 2025 1m → 15m...")
    bars_15m = load_and_resample()
    print(f"  15m bars: {len(bars_15m):,}")

    print("\nRun B — Full window (baseline verification, m15_confirmation=False)...")
    full_trades = run_backtest(
        bars_15m, config=StrategyConfig(bearish_only=True, m15_confirmation=False)
    )
    full_m = metrics(full_trades)
    print(f"  Full window trades: {full_m['trades']} (expected ~{BASELINE['trades']})")

    print("\nRun A — M15-confirmed (m15_confirmation=True)...")
    m15_trades = run_backtest(
        bars_15m, config=StrategyConfig(bearish_only=True, m15_confirmation=True)
    )
    m15_m = metrics(m15_trades)
    print(f"  M15-confirmed trades: {m15_m['trades']}")

    print("\nM15 confirmation verification (all Run A trades must have m15_confirmed=True)...")
    m15_ok = verify_m15_confirmed(m15_trades)
    if m15_ok:
        print(f"  Verification PASS — all {len(m15_trades)} trades have m15_confirmed=True")
    else:
        print("  Verification FAIL — see above")

    print(f"\n{'Run':<24} {'N':>6} {'PF':>8} {'WR':>8} {'Sharpe':>10}")
    print("-" * 60)
    print(
        f"{'M15-confirmed (A)':<24} {m15_m['trades']:>6} {m15_m['pf']:>8.3f}"
        f" {m15_m['wr']:>8.3f} {m15_m['sharpe']:>10.3f}"
    )
    print(
        f"{'Full window (B)':<24} {full_m['trades']:>6} {full_m['pf']:>8.3f}"
        f" {full_m['wr']:>8.3f} {full_m['sharpe']:>10.3f}"
    )
    print(
        f"{'S13 baseline':<24} {BASELINE['trades']:>6} {BASELINE['pf']:>8.3f}"
        f" {BASELINE['wr']:>8.3f} {BASELINE['sharpe']:>10.3f}"
    )

    n = m15_m["trades"]
    pf = m15_m["pf"]
    pf_pass = pf > PF_THRESHOLD
    n_pass = n >= MIN_TRADES

    print(f"\nH₁ pass criteria:")
    print(f"  PF > {PF_THRESHOLD}: {'✓' if pf_pass else '✗'} ({pf:.3f})")
    print(f"  N ≥ {MIN_TRADES}: {'✓' if n_pass else '✗'} ({n})")
    print(f"  M15 confirmation verification: {'✓' if m15_ok else '✗'}")

    if pf_pass and n_pass and m15_ok:
        verdict = f"H₁ SUPPORTED — M15-confirmed 15m shows PF > {PF_THRESHOLD} with N ≥ {MIN_TRADES}"
    else:
        fails = []
        if not pf_pass:
            fails.append(f"PF {pf:.3f} ≤ {PF_THRESHOLD}")
        if not n_pass:
            fails.append(f"N {n} < {MIN_TRADES}")
        if not m15_ok:
            fails.append("M15 confirmation verification failed")
        verdict = f"H₀ SUPPORTED — fails: {'; '.join(fails)}"

    print(f"\nVERDICT: {verdict}")

    today_str = date.today().strftime("%Y%m%d")
    verdict_path = f"_bmad-output/s_m15conf_15m_verdict_{today_str}.md"

    m15_exit = m15_m.get("exit_counts", {})
    full_exit = full_m.get("exit_counts", {})

    lines = [
        f"# S-M15CONF-15m Verdict — {today_str}",
        "",
        "## Pre-Registration",
        f"Sealed at git SHA: `{PRE_REG_SHA}`",
        "Doc: `_bmad-output/preregistration_s_m15conf_15m.md`",
        "",
        "## Results",
        "",
        "| Run | N | PF | WR | Daily Sharpe |",
        "|---|---|---|---|---|",
        f"| M15-confirmed (A) | {m15_m['trades']} | {m15_m['pf']:.3f} | {m15_m['wr']:.3f} | {m15_m['sharpe']:.3f} |",
        f"| Full window (B) | {full_m['trades']} | {full_m['pf']:.3f} | {full_m['wr']:.3f} | {full_m['sharpe']:.3f} |",
        f"| S13 baseline | {BASELINE['trades']} | {BASELINE['pf']:.3f} | {BASELINE['wr']:.3f} | {BASELINE['sharpe']:.3f} |",
        "",
        "## Exit Breakdown",
        "",
        "| | TP | SL | TIME_STOP |",
        "|---|---|---|---|",
        f"| M15-confirmed | {m15_exit.get('TP', 0)} | {m15_exit.get('SL', 0)} | {m15_exit.get('TIME_STOP', 0)} |",
        f"| Full window | {full_exit.get('TP', 0)} | {full_exit.get('SL', 0)} | {full_exit.get('TIME_STOP', 0)} |",
        "",
        "## M15 Confirmation Verification",
        "",
        f"{'PASS' if m15_ok else 'FAIL'} — all {len(m15_trades)} M15-confirmed trades have `m15_confirmed=True`",
        "",
        "## Pass Criteria",
        "",
        f"- PF > {PF_THRESHOLD}: {'✓' if pf_pass else '✗'} ({pf:.3f})",
        f"- N ≥ {MIN_TRADES}: {'✓' if n_pass else '✗'} ({n})",
        f"- M15 confirmation verification: {'✓' if m15_ok else '✗'}",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
    ]

    with open(verdict_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nVerdict report written: {verdict_path}")


if __name__ == "__main__":
    main()
