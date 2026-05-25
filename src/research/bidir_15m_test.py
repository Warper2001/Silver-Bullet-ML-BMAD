"""S-BIDIR-15m: Bidirectional FVG statistical power recovery at 15m resolution.

Pre-registration: _bmad-output/preregistration_s_bidir_15m.md
SHA: 9a94d2e7e75f073a717337a198610c81641d060a
"""

import math
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
PRE_REG_SHA = "9a94d2e7e75f073a717337a198610c81641d060a"

BASELINE = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}
COUNT_THRESHOLD = math.ceil(BASELINE["trades"] * 1.5)  # 92 (pre-reg text said "≥ 91" but ceil(91.5)=92)


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


def run_backtest(bars_15m: pd.DataFrame) -> list:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            df_out = bars_15m.reset_index()
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")
            df_out.to_csv(f, index=False)
            tmp_path = f.name
        engine = BacktestEngine(
            tmp_path, config=StrategyConfig(bearish_only=False)
        )
        return engine.run()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def metrics(pnls: list[float], trades_list: list) -> dict:
    n = len(pnls)
    if n == 0:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "sharpe": 0.0}
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


def main() -> None:
    print("=== S-BIDIR-15m: Bidirectional FVG Power Recovery ===")
    print(f"Pre-registration SHA: {PRE_REG_SHA}")
    print(f"Bearish-only baseline: {BASELINE['trades']} trades | PF={BASELINE['pf']:.3f}")
    print(f"Count threshold for H1: ≥ {COUNT_THRESHOLD} trades")

    print("\nLoading and resampling 2025 1m → 15m...")
    bars_15m = load_and_resample()
    print(f"  15m bars: {len(bars_15m):,}")

    print("\nRunning BacktestEngine (bearish_only=False)...")
    trades = run_backtest(bars_15m)
    n = len(trades)
    print(f"  Total trades: {n}")

    bearish_trades = [t for t in trades if t.direction == "BEARISH"]
    bullish_trades = [t for t in trades if t.direction == "BULLISH"]

    total_m = metrics([t.pnl_usd for t in trades], trades)
    bear_m = metrics([t.pnl_usd for t in bearish_trades], bearish_trades)
    bull_m = metrics([t.pnl_usd for t in bullish_trades], bullish_trades)

    print(f"\n{'Direction':<12} {'Trades':>8} {'PF':>8} {'WR':>8} {'Sharpe':>10}")
    print("-" * 52)
    print(f"{'TOTAL':<12} {total_m['trades']:>8} {total_m['pf']:>8.3f} {total_m['wr']:>8.3f} {total_m['sharpe']:>10.3f}")
    print(f"{'BEARISH':<12} {bear_m['trades']:>8} {bear_m['pf']:>8.3f} {bear_m['wr']:>8.3f} {bear_m['sharpe']:>10.3f}")
    print(f"{'BULLISH':<12} {bull_m['trades']:>8} {bull_m['pf']:>8.3f} {bull_m['wr']:>8.3f} {bull_m['sharpe']:>10.3f}")
    print(f"{'BASELINE':<12} {BASELINE['trades']:>8} {BASELINE['pf']:>8.3f} {BASELINE['wr']:>8.3f} {BASELINE['sharpe']:>10.3f}")

    count_ratio = n / BASELINE["trades"] if BASELINE["trades"] > 0 else 0
    print(f"\nCount ratio vs baseline: {count_ratio:.2f}x (threshold ≥ 1.5x)")

    count_pass = n >= COUNT_THRESHOLD
    pf_pass = total_m["pf"] > 1.0
    bear_pf_pass = bear_m["pf"] > 1.0 if bear_m["trades"] > 0 else False
    bull_pf_pass = bull_m["pf"] > 1.0 if bull_m["trades"] > 0 else False
    consistency = bear_pf_pass and bull_pf_pass

    print(f"\nConsistency criterion:")
    print(f"  Count ≥ 1.5×: {'✓' if count_pass else '✗'} ({n} vs {COUNT_THRESHOLD})")
    print(f"  Total PF > 1.0: {'✓' if pf_pass else '✗'} ({total_m['pf']:.3f})")
    print(f"  Bearish PF > 1.0: {'✓' if bear_pf_pass else '✗'} ({bear_m['pf']:.3f})")
    print(f"  Bullish PF > 1.0: {'✓' if bull_pf_pass else '✗'} ({bull_m['pf']:.3f})")
    print(f"  Direction consistency: {'✓' if consistency else '✗'}")

    if count_pass and pf_pass and consistency:
        verdict = "H₁ SUPPORTED — bidirectional 15m recovers power with maintained edge"
    else:
        fails = []
        if not count_pass:
            fails.append(f"count {n} < {COUNT_THRESHOLD}")
        if not pf_pass:
            fails.append(f"PF {total_m['pf']:.3f} ≤ 1.0")
        if not consistency:
            if not bear_pf_pass:
                fails.append(f"bearish PF {bear_m['pf']:.3f} ≤ 1.0")
            if not bull_pf_pass:
                fails.append(f"bullish PF {bull_m['pf']:.3f} ≤ 1.0")
        verdict = f"H₀ SUPPORTED — fails: {'; '.join(fails)}"

    print(f"\nVERDICT: {verdict}")

    today_str = date.today().strftime("%Y%m%d")
    verdict_path = f"_bmad-output/s_bidir_15m_verdict_{today_str}.md"

    lines = [
        f"# S-BIDIR-15m Verdict — {today_str}",
        "",
        "## Pre-Registration",
        f"Sealed at git SHA: `{PRE_REG_SHA}`",
        "Doc: `_bmad-output/preregistration_s_bidir_15m.md`",
        "",
        "## Results",
        "",
        "| Direction | Trades | PF | WR | Daily Sharpe |",
        "|---|---|---|---|---|",
        f"| TOTAL (bidir) | {total_m['trades']} | {total_m['pf']:.3f} | {total_m['wr']:.3f} | {total_m['sharpe']:.3f} |",
        f"| BEARISH only | {bear_m['trades']} | {bear_m['pf']:.3f} | {bear_m['wr']:.3f} | {bear_m['sharpe']:.3f} |",
        f"| BULLISH only | {bull_m['trades']} | {bull_m['pf']:.3f} | {bull_m['wr']:.3f} | {bull_m['sharpe']:.3f} |",
        f"| BASELINE (bearish-only S13) | {BASELINE['trades']} | {BASELINE['pf']:.3f} | {BASELINE['wr']:.3f} | {BASELINE['sharpe']:.3f} |",
        "",
        "## Exit Breakdown",
        "",
        "| | TP | SL | TIME_STOP |",
        "|---|---|---|---|",
        f"| BEARISH | {bear_m.get('exit_counts', {}).get('TP', 0)} | {bear_m.get('exit_counts', {}).get('SL', 0)} | {bear_m.get('exit_counts', {}).get('TIME_STOP', 0)} |",
        f"| BULLISH | {bull_m.get('exit_counts', {}).get('TP', 0)} | {bull_m.get('exit_counts', {}).get('SL', 0)} | {bull_m.get('exit_counts', {}).get('TIME_STOP', 0)} |",
        "",
        "## Consistency Criterion",
        "",
        f"- Count ≥ 1.5× baseline (≥ {COUNT_THRESHOLD}): {'✓' if count_pass else '✗'} ({n} trades, {count_ratio:.2f}×)",
        f"- Total PF > 1.0: {'✓' if pf_pass else '✗'} ({total_m['pf']:.3f})",
        f"- Bearish PF > 1.0: {'✓' if bear_pf_pass else '✗'} ({bear_m['pf']:.3f})",
        f"- Bullish PF > 1.0: {'✓' if bull_pf_pass else '✗'} ({bull_m['pf']:.3f})",
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
