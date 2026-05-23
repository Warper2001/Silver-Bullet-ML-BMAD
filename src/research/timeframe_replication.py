"""S13 Timeframe Replication — run BacktestEngine on 5m and 15m resamples.

Pre-registration: _bmad-output/preregistration_s13_timeframe.md
SHA: 5fde2d254277ab5b2943d608a1e8833d5a7243e2
"""

import os
import sys
import tempfile
from datetime import date

import numpy as np
import pandas as pd

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import (
    StrategyConfig,
    calc_profit_factor,
    calc_sharpe,
)

CSV_PATH = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"

BASELINE = {
    "label": "1m (baseline)",
    "trades": 129,
    "pf": 0.937,
    "wr": 0.460,
    "sharpe": None,
}


def load_1min_bars() -> pd.DataFrame:
    bars = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    else:
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")
    return bars


def resample_bars(bars_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    return (
        bars_1m.resample(freq)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def run_on_resampled(bars: pd.DataFrame, label: str) -> dict:
    print(f"\nRunning {label} backtest ({len(bars)} bars)...")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            df_out = bars.reset_index()
            # Convert ET → UTC to match original CSV format; _load_bars() converts UTC → ET
            df_out["timestamp"] = df_out["timestamp"].dt.tz_convert("UTC")
            df_out.to_csv(f, index=False)
            tmp_path = f.name

        engine = BacktestEngine(tmp_path, config=StrategyConfig())
        trades = engine.run()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    pnls = [t.pnl_usd for t in trades]
    n = len(pnls)
    if n == 0:
        return {"label": label, "trades": 0, "pf": 0.0, "wr": 0.0, "sharpe": 0.0}

    pf = calc_profit_factor(pnls)
    wr = sum(1 for p in pnls if p > 0) / n

    # Daily Sharpe
    daily: dict[date, float] = {}
    for t in trades:
        d = t.timestamp_entry.date()
        daily[d] = daily.get(d, 0.0) + t.pnl_usd
    sharpe = calc_sharpe(list(daily.values()))

    # Exit reason breakdown
    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    print(f"  {label}: {n} trades | PF={pf:.3f} | WR={wr:.3f} | Sharpe={sharpe:.3f}")
    print(f"  Exits: {exit_counts}")

    return {
        "label": label,
        "trades": n,
        "pf": pf,
        "wr": wr,
        "sharpe": sharpe,
        "exit_counts": exit_counts,
    }


def write_verdict(results: list[dict], verdict_date: str) -> str:
    path = f"_bmad-output/s13_verdict_{verdict_date}.md"

    r5 = next(r for r in results if r["label"] == "5m")
    r15 = next(r for r in results if r["label"] == "15m")

    pf_5 = r5["pf"]
    pf_15 = r15["pf"]

    if pf_5 > 1.0 and pf_15 > 1.0:
        verdict = "PATTERNS SURVIVE (both timeframes PF > 1.0)"
        detail = "Both 5m and 15m show PF > 1.0. H₁ (alternative) is supported."
    else:
        verdict = "H₀ SUPPORTED — PATTERN IS TIMEFRAME-SPECIFIC (consistent with noise)"
        failing = []
        if pf_5 <= 1.0:
            failing.append(f"5m PF={pf_5:.3f} ≤ 1.0")
        if pf_15 <= 1.0:
            failing.append(f"15m PF={pf_15:.3f} ≤ 1.0")
        detail = f"Consistency criterion failed: {', '.join(failing)}."

    lines = [
        f"# S13 Timeframe Replication Verdict — {verdict_date}",
        "",
        "## Pre-Registration",
        "Sealed at git SHA: `5fde2d254277ab5b2943d608a1e8833d5a7243e2`",
        "Doc: `_bmad-output/preregistration_s13_timeframe.md`",
        "",
        "## Results",
        "",
        "| Timeframe | Trades | PF | WR | Daily Sharpe |",
        "|---|---|---|---|---|",
        f"| 1m (baseline) | {BASELINE['trades']} | {BASELINE['pf']:.3f} | {BASELINE['wr']:.3f} | ~0 |",
    ]

    for r in results:
        sharpe_str = f"{r['sharpe']:.3f}" if r["sharpe"] is not None else "n/a"
        lines.append(
            f"| {r['label']} | {r['trades']} | {r['pf']:.3f} | {r['wr']:.3f} | {sharpe_str} |"
        )

    lines += [
        "",
        "## Exit Reason Breakdown",
        "",
        "| Timeframe | TP | SL | TIME_STOP |",
        "|---|---|---|---|",
    ]
    for r in results:
        ec = r.get("exit_counts", {})
        lines.append(
            f"| {r['label']} | {ec.get('TP', 0)} | {ec.get('SL', 0)} | {ec.get('TIME_STOP', 0)} |"
        )

    lines += [
        "",
        "## Known Behavioral Deltas at 5m / 15m",
        "",
        "| Behavior | 1m baseline | 5m | 15m |",
        "|---|---|---|---|",
        "| FVG span | 3 min | 15 min | 45 min |",
        "| `max_hold_bars=60` | 60 min | 300 min | 900 min |",
        "| `max_pending_bars=240` | 4 hr | 20 hr | 60 hr |",
        "| H1 resample | 60 bars/H1 | 12 bars/H1 | 4 bars/H1 |",
        "| Bar ATR | true 1-min ATR | 5-min ATR (larger) | 15-min ATR (larger) |",
        "",
        "These deltas are expected and do not invalidate the test. StrategyConfig was NOT adjusted.",
        "",
        "## Consistency Criterion",
        "",
        "**Pre-committed rule:** Both 5m AND 15m must show PF > 1.0 for H₁ to be supported.",
        "If either is ≤ 1.0, H₀ is supported (pattern is timeframe-specific = consistent with noise).",
        "",
        f"**Result:** {detail}",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Combined with S12",
        "",
        "S12 verdict (random-entry control): **AMBIGUOUS = TREATED AS FAIL = PIVOT**",
        f"S13 verdict (timeframe replication): **{verdict}**",
        "",
        "See Story 6.3 for Phase 1 synthesis verdict.",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nVerdict report written: {path}")
    return path


def main() -> None:
    print("=== S13 Timeframe Replication ===")
    print(f"Pre-registration SHA: 5fde2d254277ab5b2943d608a1e8833d5a7243e2")
    print(f"Data: {CSV_PATH}")

    print("\nLoading 1-min bars...")
    bars_1m = load_1min_bars()
    print(f"  Loaded {len(bars_1m):,} bars ({bars_1m.index[0].date()} → {bars_1m.index[-1].date()})")

    print("\nResampling to 5m and 15m...")
    bars_5m = resample_bars(bars_1m, "5min")
    bars_15m = resample_bars(bars_1m, "15min")
    print(f"  5m: {len(bars_5m):,} bars")
    print(f"  15m: {len(bars_15m):,} bars")

    print(f"\n1m baseline (from Epic 1): {BASELINE['trades']} trades | PF={BASELINE['pf']:.3f} | WR={BASELINE['wr']:.3f}")

    results = []
    r5 = run_on_resampled(bars_5m, "5m")
    results.append(r5)
    r15 = run_on_resampled(bars_15m, "15m")
    results.append(r15)

    print("\n=== SUMMARY ===")
    print(f"{'Timeframe':<15} {'Trades':>8} {'PF':>8} {'WR':>8} {'Sharpe':>10}")
    print("-" * 55)
    print(f"{'1m (baseline)':<15} {BASELINE['trades']:>8} {BASELINE['pf']:>8.3f} {BASELINE['wr']:>8.3f} {'~0':>10}")
    for r in results:
        sharpe_str = f"{r['sharpe']:.3f}" if r["sharpe"] is not None else "n/a"
        print(f"{r['label']:<15} {r['trades']:>8} {r['pf']:>8.3f} {r['wr']:>8.3f} {sharpe_str:>10}")

    # Consistency criterion
    pf_5 = r5["pf"]
    pf_15 = r15["pf"]
    print(f"\nConsistency criterion (both 5m AND 15m PF > 1.0):")
    print(f"  5m PF={pf_5:.3f} {'✓ > 1.0' if pf_5 > 1.0 else '✗ ≤ 1.0'}")
    print(f"  15m PF={pf_15:.3f} {'✓ > 1.0' if pf_15 > 1.0 else '✗ ≤ 1.0'}")

    if pf_5 > 1.0 and pf_15 > 1.0:
        print("\nVERDICT: PATTERNS SURVIVE")
    else:
        print("\nVERDICT: H₀ SUPPORTED — PATTERN IS TIMEFRAME-SPECIFIC (consistent with noise)")

    today = date.today().strftime("%Y%m%d")
    write_verdict(results, today)


if __name__ == "__main__":
    main()
