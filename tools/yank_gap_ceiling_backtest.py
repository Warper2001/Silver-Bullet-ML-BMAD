#!/usr/bin/env python3
"""One-shot in-sample backtest: old $60 ceiling vs new 0.426xATR ceiling.

Explicitly scoped to what the gap-ceiling seal (_bmad-output/
preregistration_yank_gap_ceiling_denomination.md) allows: the SAME derivation
window as §3/§5 (2025-01-01..2026-02-28, strictly pre-holdout). Does NOT read
data/sealed_holdout/ or any bar on/after 2026-03-01 — §7 forbids re-running the
sealed holdout, and this script enforces that as a hard date filter, not a
convention.

Uses BacktestEngine (src/research/backtest_engine.py) — pure strategy_core
replay, deterministic, no mocking. IMPORTANT CAVEAT: BacktestEngine does NOT
apply the ML meta-labeling filter (ml_threshold=0.50 live) — it produces the
RAW pre-ML trade population for both arms. That omission is applied
IDENTICALLY to both the old-ceiling and new-ceiling runs, so the *comparison*
between them is apples-to-apples, but neither number is what YANK's live bot
would actually have taken (live ML@0.50 historically cut roughly a third of
raw signals). Report accordingly — this is a delta on the raw signal
population, not a faithful P&L reproduction.

Read-only. Never writes to strategy_config.yaml or any live state.

Usage:
    .venv/bin/python tools/yank_gap_ceiling_backtest.py
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.research.backtest_engine import BacktestEngine  # noqa: E402
from src.research.config_loader import load_strategy_config  # noqa: E402
from src.research.strategy_core import calc_profit_factor  # noqa: E402

LIVE_CHECKOUT = Path("/root/Silver-Bullet-ML-BMAD")
DERIV_START = pd.Timestamp("2025-01-01", tz="UTC")
DERIV_END = pd.Timestamp("2026-02-28 23:59:59", tz="UTC")   # HARD cutoff — sealed holdout starts 03-01
MAX_GAP_ATR_RATIO = 0.426

BAR_SOURCES = [
    LIVE_CHECKOUT / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2025.csv",
    BASE / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2026_ytd.csv",
]
CONFIG_PATH = BASE / "strategy_config.yaml"


def load_and_write_temp_csv() -> Path:
    frames = []
    for p in BAR_SOURCES:
        if not p.exists():
            raise SystemExit(f"missing derivation-window source: {p}")
        df = pd.read_csv(p, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
        frames.append(df)
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.drop_duplicates("timestamp").sort_values("timestamp")

    # HARD enforcement of the seal's boundary — not a convention, a filter.
    bars = bars[(bars["timestamp"] >= DERIV_START) & (bars["timestamp"] <= DERIV_END)]
    assert bars["timestamp"].max() < pd.Timestamp("2026-03-01", tz="UTC"), \
        "refusing to proceed: bars extend into the sealed holdout"

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    bars.to_csv(tmp.name, index=False)
    tmp.close()
    return Path(tmp.name), bars["timestamp"].min(), bars["timestamp"].max(), len(bars)


def summarize(trades, label: str) -> dict:
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0}
    pnls = [t.pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    tstops = sum(1 for t in trades if t.exit_reason == "TIME_STOP")
    entries = {(t.timestamp_entry, t.direction) for t in trades}
    return {
        "label": label,
        "n": n,
        "net_pnl": sum(pnls),
        "pf": calc_profit_factor(pnls),
        "win_rate": len(wins) / n,
        "avg_win": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(losses) / len(losses)) if losses else 0.0,
        "tstop_pct": tstops / n * 100.0,
        "entries": entries,
    }


def main() -> int:
    print("Loading derivation-window bars (2025-01-01..2026-02-28, HARD-filtered)...", file=sys.stderr)
    tmp_csv, dmin, dmax, n_bars = load_and_write_temp_csv()
    print(f"  {n_bars} bars, {dmin} .. {dmax}", file=sys.stderr)

    base_cfg = load_strategy_config(CONFIG_PATH) if CONFIG_PATH.exists() else None
    if base_cfg is None:
        raise SystemExit(f"missing {CONFIG_PATH}")
    # Force the OLD arm's ceiling off explicitly (defensive — should already be 0.0)
    old_cfg = dataclasses.replace(base_cfg, max_gap_atr_ratio=0.0)
    new_cfg = dataclasses.replace(base_cfg, max_gap_atr_ratio=MAX_GAP_ATR_RATIO)

    try:
        print("Running OLD ceiling ($60 flat)...", file=sys.stderr)
        old_trades = BacktestEngine(str(tmp_csv), old_cfg).run()
        print(f"  {len(old_trades)} trades", file=sys.stderr)

        print("Running NEW ceiling (0.426 x H1_ATR)...", file=sys.stderr)
        new_trades = BacktestEngine(str(tmp_csv), new_cfg).run()
        print(f"  {len(new_trades)} trades", file=sys.stderr)
    finally:
        tmp_csv.unlink(missing_ok=True)

    old_s = summarize(old_trades, "OLD ($60 flat ceiling)")
    new_s = summarize(new_trades, "NEW (0.426 x H1_ATR)")

    new_only = new_s["entries"] - old_s["entries"]
    new_only_trades = [t for t in new_trades
                       if (t.timestamp_entry, t.direction) in new_only]
    incr = summarize(new_only_trades, "NEW-ONLY (incremental trades the old ceiling always rejected)")

    print()
    print("=== Gap-ceiling backtest — RAW pre-ML population, derivation window only ===")
    print(f"window: {dmin} .. {dmax}  (HARD stop before 2026-03-01, sealed holdout untouched)")
    print("CAVEAT: BacktestEngine does not apply the ml_threshold=0.50 live ML filter.")
    print("        These are RAW accepted-signal counts, not faithful live P&L.")
    print()
    header = f"{'':<45} {'N':>6} {'Net PnL':>12} {'PF':>7} {'WR':>7} {'AvgWin':>9} {'AvgLoss':>9} {'TSTOP%':>7}"
    print(header)
    print("-" * len(header))
    for s in (old_s, new_s, incr):
        if s["n"] == 0:
            print(f"{s['label']:<45} {0:>6}   (no trades)")
            continue
        pf_str = "inf" if s["pf"] == float("inf") else f"{s['pf']:.3f}"
        print(f"{s['label']:<45} {s['n']:>6} {s['net_pnl']:>12.2f} {pf_str:>7} "
              f"{s['win_rate']:>7.3f} {s['avg_win']:>9.2f} {s['avg_loss']:>9.2f} {s['tstop_pct']:>6.1f}%")
    print()
    print(f"Old ceiling total trades:  {old_s['n']}")
    print(f"New ceiling total trades:  {new_s['n']}  (+{new_s['n'] - old_s['n']} vs old, "
          f"+{100.0*(new_s['n']-old_s['n'])/old_s['n']:.1f}% if old_s['n']>0)" if old_s['n'] else "")
    print(f"Incremental (new-only) trades: {len(new_only_trades)}")
    print()
    print("Reminder: this is IN-SAMPLE on data the max_gap_atr_ratio VALUE was derived from")
    print("(median of cap_pts/H1_ATR over this same window). Not decision-grade. Not the")
    print("sealed 2026 holdout. Not a projection of forward performance.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
