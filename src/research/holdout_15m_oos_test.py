"""Phase 2 — 15m FVG+H1-sweep sealed holdout OOS test.

Pre-registration: _bmad-output/preregistration_phase2_15m.md
SHA: 5b581f4d88e5bf66216e23c4b66eb331ffb9b43b
Pass/fail threshold: PF > 1.1
"""

import os
import tempfile
from datetime import date, datetime, timezone

import pandas as pd

from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import (
    StrategyConfig,
    calc_profit_factor,
    calc_sharpe,
)

HOLDOUT_CSV = "data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv"
ACCESS_LOG = "data/sealed_holdout/ACCESS_LOG.md"
PRE_REG_SHA = "5b581f4d88e5bf66216e23c4b66eb331ffb9b43b"
PASS_THRESHOLD = 1.1

# Training-window reference (S13, 2025 full year)
TRAINING_15M = {"trades": 61, "pf": 1.179, "wr": 0.475, "sharpe": 1.373}


def load_holdout_1min() -> pd.DataFrame:
    bars = pd.read_csv(HOLDOUT_CSV, parse_dates=["timestamp"])
    if bars["timestamp"].dt.tz is None:
        bars["timestamp"] = bars["timestamp"].dt.tz_localize("UTC")
    else:
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.drop(columns=["notional"], errors="ignore")
    return bars


def resample_to_15m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        bars_1m.resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["open", "high", "low", "close"])
    )


def append_access_log(result_summary: str) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"| {today} | `{PRE_REG_SHA}` | holdout_15m_oos_test.py | "
        f"Phase 2 definitive 15m OOS test | {result_summary} |\n"
    )
    with open(ACCESS_LOG, "a") as f:
        f.write(entry)
    print(f"Access log updated: {ACCESS_LOG}")


def main() -> None:
    print("=== Phase 2 — 15m Sealed Holdout OOS Test ===")
    print(f"Pre-registration SHA: {PRE_REG_SHA}")
    print(f"Holdout: {HOLDOUT_CSV}")
    print(f"Pass/fail threshold: PF > {PASS_THRESHOLD}")

    print("\nLoading holdout 1-min bars...")
    bars_1m = load_holdout_1min()
    print(
        f"  Loaded {len(bars_1m):,} bars "
        f"({bars_1m.index[0].date()} → {bars_1m.index[-1].date()})"
    )

    print("\nResampling to 15m...")
    bars_15m = resample_to_15m(bars_1m)
    print(f"  15m bars: {len(bars_15m):,}")

    print("\nRunning BacktestEngine on 15m holdout (StrategyConfig defaults)...")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            df_out = bars_15m.reset_index()
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
        print("\n  NO TRADES — insufficient data or all filters blocked")
        append_access_log("no_trades: 0 trades")
        return

    pf = calc_profit_factor(pnls)
    wr = sum(1 for p in pnls if p > 0) / n

    daily: dict[date, float] = {}
    for t in trades:
        d = t.timestamp_entry.date()
        daily[d] = daily.get(d, 0.0) + t.pnl_usd
    sharpe = calc_sharpe(list(daily.values()))

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    passed = pf > PASS_THRESHOLD

    print(f"\n  Trades:  {n}")
    print(f"  PF:      {pf:.3f}  (threshold: > {PASS_THRESHOLD})")
    print(f"  WR:      {wr:.3f}")
    print(f"  Sharpe:  {sharpe:.3f}")
    print(f"  Exits:   {exit_counts}")

    print(f"\n  Training-window 15m reference: "
          f"{TRAINING_15M['trades']} trades | PF={TRAINING_15M['pf']:.3f} | "
          f"WR={TRAINING_15M['wr']:.3f} | Sharpe={TRAINING_15M['sharpe']:.3f}")

    verdict = "PASS — H₁ SUPPORTED" if passed else "FAIL — H₀ SUPPORTED"
    print(f"\nVERDICT: {verdict}")

    # Append to access log BEFORE writing verdict file
    result_summary = (
        f"PF={pf:.4f} ({'PASS' if passed else 'FAIL'}), "
        f"N={n}, WR={wr:.3f}, Sharpe={sharpe:.3f}"
    )
    append_access_log(result_summary)

    # Write verdict report
    today_str = date.today().strftime("%Y%m%d")
    verdict_path = f"_bmad-output/s_phase2_15m_verdict_{today_str}.md"

    time_stop_pct = exit_counts.get("TIME_STOP", 0) / n * 100 if n > 0 else 0
    train_time_stop_pct = 7 / 61 * 100  # S13 training result

    lines = [
        f"# Phase 2 — 15m Holdout OOS Test Verdict — {today_str}",
        "",
        "## Pre-Registration",
        f"Sealed at git SHA: `{PRE_REG_SHA}`",
        "Doc: `_bmad-output/preregistration_phase2_15m.md`",
        "",
        "## Holdout Data",
        f"- File: `{HOLDOUT_CSV}`",
        f"- Window: {bars_1m.index[0].date()} → {bars_1m.index[-1].date()}",
        f"- 1m bars: {len(bars_1m):,} | 15m bars after resample: {len(bars_15m):,}",
        "",
        "## Results",
        "",
        "| | Holdout (OOS) | Training (S13 2025) |",
        "|---|---|---|",
        f"| Trades | {n} | {TRAINING_15M['trades']} |",
        f"| PF | {pf:.3f} | {TRAINING_15M['pf']:.3f} |",
        f"| WR | {wr:.3f} | {TRAINING_15M['wr']:.3f} |",
        f"| Daily Sharpe | {sharpe:.3f} | {TRAINING_15M['sharpe']:.3f} |",
        f"| TIME_STOP % | {time_stop_pct:.0f}% | {train_time_stop_pct:.0f}% |",
        "",
        "## Exit Breakdown (Holdout)",
        "",
        "| TP | SL | TIME_STOP |",
        "|---|---|---|",
        f"| {exit_counts.get('TP', 0)} | {exit_counts.get('SL', 0)} | {exit_counts.get('TIME_STOP', 0)} |",
        "",
        "## Pass/Fail Threshold",
        "",
        f"**Pre-committed threshold:** PF > {PASS_THRESHOLD}",
        f"**Observed PF:** {pf:.3f}",
        f"**Result:** {'PASS ✓' if passed else 'FAIL ✗'}",
        "",
        "## Sample Size Caveat",
        "",
        f"N={n} trades over ~2.5 months. Expected ~13 based on training rate.",
        "Small sample — treat result as directional evidence, not high-confidence conclusion.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]

    if passed:
        lines += [
            "PF > 1.1 on sealed holdout. H₁ (15m edge is real and generalises) is supported.",
            "",
            "**Next step (Story 7.3):** Synthesise with Epic 6 Phase 1 context. Consider unblocking Epic 2",
            "(strategy enhancement) starting from 15m infrastructure.",
        ]
    else:
        lines += [
            f"PF={pf:.3f} ≤ 1.1 on sealed holdout. H₀ is supported.",
            "",
            "**Next step (Story 7.3):** PIVOT again. P1 (15m) is now exhausted.",
            "Select from P2-P5 per pre-committed pivot menu in `_bmad-output/phase1_verdict_20260523.md`.",
        ]

    lines += [
        "",
        "## Access Log",
        "",
        f"This run was logged in `{ACCESS_LOG}` before results were printed.",
        f"Pre-reg SHA: `{PRE_REG_SHA}`",
        "",
        "_Produced by `src/research/holdout_15m_oos_test.py` (Story 7.1/7.2)._",
    ]

    with open(verdict_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nVerdict report written: {verdict_path}")


if __name__ == "__main__":
    main()
