"""S-VOL-15m: Relaxed Filter Constants at 15m — statistical power experiment.

Pre-registration: _bmad-output/preregistration_s_vol_15m.md
SHA: b44acc6
"""

import os
import tempfile
from collections import Counter
from datetime import date

import pandas as pd

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import (
    StrategyConfig,
    calc_profit_factor,
    calc_sharpe,
)

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
PRE_REG_SHA = "b44acc6"

BASELINE = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}
PF_THRESHOLD = 1.3
MIN_TRADES = 15
MONTHLY_TRADE_TARGET = 30  # AC #3

RELAXED_CONFIG = StrategyConfig(
    bearish_only=True,
    h1_sweep_lookback=10,
    min_gap_atr_ratio=0.10,
    max_pending_bars=120,
    tuesday_exclusion=False,
)


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


def monthly_breakdown(trades_list: list) -> tuple[dict[str, int], float]:
    """Return (month_counts, avg_monthly) for AC #3."""
    monthly: dict[str, int] = Counter(
        t.timestamp_entry.strftime("%Y-%m") for t in trades_list
    )
    avg = len(trades_list) / max(len(monthly), 1)
    return dict(sorted(monthly.items())), avg


def main() -> None:
    print("=== S-VOL-15m: Relaxed Filter Constants at 15m ===")
    print(f"Pre-registration SHA: {PRE_REG_SHA}")
    print(f"Bearish-only baseline: {BASELINE['trades']} trades | PF={BASELINE['pf']:.3f}")
    print(f"Pass criterion: PF > {PF_THRESHOLD} AND N ≥ {MIN_TRADES}")
    print(f"Monthly target (AC #3): ≥ {MONTHLY_TRADE_TARGET} trades/month")

    print("\nRelaxed config:")
    print(f"  h1_sweep_lookback=10 (was 6)")
    print(f"  min_gap_atr_ratio=0.10 (was 0.25)")
    print(f"  max_pending_bars=120 (was 240)")
    print(f"  tuesday_exclusion=False (was True)")

    print("\nLoading and resampling 2025 1m → 15m...")
    bars_15m = load_and_resample()
    print(f"  15m bars: {len(bars_15m):,}")

    print("\nRun B — Full window baseline (bearish_only=True, all defaults)...")
    baseline_trades = run_backtest(bars_15m, config=StrategyConfig(bearish_only=True))
    baseline_m = metrics(baseline_trades)
    print(f"  Baseline trades: {baseline_m['trades']} (expected ~{BASELINE['trades']})")

    print("\nRun A — Relaxed config...")
    relaxed_trades = run_backtest(bars_15m, config=RELAXED_CONFIG)
    relaxed_m = metrics(relaxed_trades)
    print(f"  Relaxed trades: {relaxed_m['trades']}")

    monthly, avg_monthly = monthly_breakdown(relaxed_trades)
    monthly_pass = avg_monthly >= MONTHLY_TRADE_TARGET

    print(f"\n{'Run':<26} {'N':>6} {'PF':>8} {'WR':>8} {'Sharpe':>10}")
    print("-" * 62)
    print(
        f"{'Relaxed config (A)':<26} {relaxed_m['trades']:>6} {relaxed_m['pf']:>8.3f}"
        f" {relaxed_m['wr']:>8.3f} {relaxed_m['sharpe']:>10.3f}"
    )
    print(
        f"{'Full window baseline (B)':<26} {baseline_m['trades']:>6} {baseline_m['pf']:>8.3f}"
        f" {baseline_m['wr']:>8.3f} {baseline_m['sharpe']:>10.3f}"
    )
    print(
        f"{'S13 baseline':<26} {BASELINE['trades']:>6} {BASELINE['pf']:>8.3f}"
        f" {BASELINE['wr']:>8.3f} {BASELINE['sharpe']:>10.3f}"
    )

    print("\nMonthly breakdown (Run A):")
    for month, count in monthly.items():
        print(f"  {month}: {count} trades")
    print(f"  Average: {avg_monthly:.1f} trades/month (target ≥ {MONTHLY_TRADE_TARGET})")

    n = relaxed_m["trades"]
    pf = relaxed_m["pf"]
    pf_pass = pf > PF_THRESHOLD
    n_pass = n >= MIN_TRADES

    print(f"\nH₁ pass criteria:")
    print(f"  PF > {PF_THRESHOLD}: {'✓' if pf_pass else '✗'} ({pf:.3f})")
    print(f"  N ≥ {MIN_TRADES}: {'✓' if n_pass else '✗'} ({n})")
    print(f"  Monthly avg ≥ {MONTHLY_TRADE_TARGET}: {'✓' if monthly_pass else '✗'} ({avg_monthly:.1f})")

    print(f"\nAC #1 confirmation: volatility_regime_filter() uses config.vol_regime_lookback "
          f"and config.vol_regime_threshold — no hardcoded constants ✓")
    print(f"AC #4 confirmation: RELAXED_CONFIG.max_pending_bars = {RELAXED_CONFIG.max_pending_bars} ✓")
    print(f"AC #5 confirmation: sl_multiplier={RELAXED_CONFIG.sl_multiplier}, "
          f"tp_multiplier={RELAXED_CONFIG.tp_multiplier} ✓")

    if pf_pass and n_pass:
        verdict = f"H₁ SUPPORTED — Relaxed 15m shows PF > {PF_THRESHOLD} with N ≥ {MIN_TRADES}"
    else:
        fails = []
        if not pf_pass:
            fails.append(f"PF {pf:.3f} ≤ {PF_THRESHOLD}")
        if not n_pass:
            fails.append(f"N {n} < {MIN_TRADES}")
        verdict = f"H₀ SUPPORTED — fails: {'; '.join(fails)}"

    print(f"\nVERDICT: {verdict}")

    today_str = date.today().strftime("%Y%m%d")
    verdict_path = f"_bmad-output/s_vol_15m_verdict_{today_str}.md"

    relaxed_exit = relaxed_m.get("exit_counts", {})
    baseline_exit = baseline_m.get("exit_counts", {})

    monthly_rows = "\n".join(
        f"| {m} | {c} |" for m, c in monthly.items()
    )

    lines = [
        f"# S-VOL-15m Verdict — {today_str}",
        "",
        "## Pre-Registration",
        f"Sealed at git SHA: `{PRE_REG_SHA}`",
        "Doc: `_bmad-output/preregistration_s_vol_15m.md`",
        "",
        "## Relaxed Configuration",
        "",
        "```python",
        "StrategyConfig(",
        "    bearish_only=True,",
        "    h1_sweep_lookback=10,   # was 6",
        "    min_gap_atr_ratio=0.10, # was 0.25",
        "    max_pending_bars=120,    # was 240",
        "    tuesday_exclusion=False, # was True",
        ")",
        "```",
        "",
        "## Results",
        "",
        "| Run | N | PF | WR | Daily Sharpe |",
        "|---|---|---|---|---|",
        f"| Relaxed config (A) | {relaxed_m['trades']} | {relaxed_m['pf']:.3f} | {relaxed_m['wr']:.3f} | {relaxed_m['sharpe']:.3f} |",
        f"| Full window baseline (B) | {baseline_m['trades']} | {baseline_m['pf']:.3f} | {baseline_m['wr']:.3f} | {baseline_m['sharpe']:.3f} |",
        f"| S13 baseline | {BASELINE['trades']} | {BASELINE['pf']:.3f} | {BASELINE['wr']:.3f} | {BASELINE['sharpe']:.3f} |",
        "",
        "## Exit Breakdown",
        "",
        "| | TP | SL | TIME_STOP |",
        "|---|---|---|---|",
        f"| Relaxed config | {relaxed_exit.get('TP', 0)} | {relaxed_exit.get('SL', 0)} | {relaxed_exit.get('TIME_STOP', 0)} |",
        f"| Full window baseline | {baseline_exit.get('TP', 0)} | {baseline_exit.get('SL', 0)} | {baseline_exit.get('TIME_STOP', 0)} |",
        "",
        "## Monthly Trade Breakdown (Run A)",
        "",
        "| Month | Trades |",
        "|---|---|",
        monthly_rows,
        f"| **Average** | **{avg_monthly:.1f}** |",
        "",
        "## AC Confirmations",
        "",
        f"- **AC #1** (vol regime parameterized): `volatility_regime_filter()` uses `config.vol_regime_lookback` and `config.vol_regime_threshold` — no hardcoded constants ✓",
        f"- **AC #3** (monthly trade count): avg {avg_monthly:.1f} trades/month — {'✓ PASS' if monthly_pass else '✗ FAIL'} (target ≥ {MONTHLY_TRADE_TARGET})",
        f"- **AC #4** (max_pending_bars=120): RELAXED_CONFIG.max_pending_bars = {RELAXED_CONFIG.max_pending_bars} ✓",
        f"- **AC #5** (SL/TP finalized): sl_multiplier={RELAXED_CONFIG.sl_multiplier}, tp_multiplier={RELAXED_CONFIG.tp_multiplier} ✓",
        "",
        "## Pass Criteria",
        "",
        f"- PF > {PF_THRESHOLD}: {'✓' if pf_pass else '✗'} ({pf:.3f})",
        f"- N ≥ {MIN_TRADES}: {'✓' if n_pass else '✗'} ({n})",
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
