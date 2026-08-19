#!/usr/bin/env python3
"""Gate-opening battery, re-run against TODAY's live 1-min config (2026-08-19).

Alex asked to re-run the Epic 2 BIDIR/KZ/M15CONF/VOL battery
(_bmad-output/s_{bidir,kz,m15conf,vol}_15m_verdict_20260523.md — all four H0,
baseline won every time) against the current config rather than assume the old
verdict still holds.

IMPORTANT — this is NOT a literal re-run. Two things changed since 2026-05-23:

1. Timeframe: the 2026-05-23 battery ran on 15-MINUTE bars (S13-era, the
   resolution S12/S13 pivoted to because 1-minute was ambiguous on a DIFFERENT
   question). YANK's live strategy trades 1-MINUTE FVGs. This script is the
   first native 1-minute version of this battery, not a repeat of the old one.
2. Baseline: today's live strategy_config.yaml already has
   enable_kill_zone_filter=true (09:30-11:00 ET) — the 2026-05-23 KZ test
   ADDED a kill zone to a baseline that didn't have one. Today the "open the
   gate" direction is the opposite: REMOVE it.

EXPLORATORY, not confirmatory. Unlike the 2026-05-23 battery (each variant had
its own preregistration_s_*.md sealed BEFORE running, with a committed pass
bar), this has no pre-registration and no committed pass/fail threshold — it
is being run to see whether the shape of the old verdict (every gate-loosening
variant loses to baseline) still holds under today's config, not to certify
any variant for deployment. Do not treat a "winning" variant here as
authorization to ship it — that would require its own prereg, exactly like
gap-ceiling did.

Same hard scope as the gap-ceiling backtest: pre-holdout derivation window
only (2025-01-01..2026-02-28), asserts bars never cross into 2026-03-01,
never reads data/sealed_holdout/. RAW pre-ML population (BacktestEngine does
not apply ml_threshold=0.50) — applies identically to every arm, so the
comparison is fair, but no arm's number is faithful live P&L.

Usage:
    .venv/bin/python tools/yank_gate_opening_battery.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.research.backtest_engine import BacktestEngine  # noqa: E402
from src.research.config_loader import load_strategy_config  # noqa: E402
from src.research.strategy_core import calc_profit_factor  # noqa: E402
from tools.yank_gap_ceiling_backtest import load_and_write_temp_csv, MAX_GAP_ATR_RATIO  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / "strategy_config.yaml"


def summarize(trades, label: str) -> dict:
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0, "net_pnl": 0.0, "pf": float("nan"),
                "wr": float("nan"), "tstop_pct": float("nan")}
    pnls = [t.pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    tstops = sum(1 for t in trades if t.exit_reason == "TIME_STOP")
    return {
        "label": label, "n": n, "net_pnl": sum(pnls), "pf": calc_profit_factor(pnls),
        "wr": len(wins) / n, "tstop_pct": tstops / n * 100.0,
    }


def main() -> int:
    print("Loading derivation-window bars (2025-01-01..2026-02-28, HARD-filtered)...", file=sys.stderr)
    tmp_csv, dmin, dmax, n_bars = load_and_write_temp_csv()
    print(f"  {n_bars} bars, {dmin} .. {dmax}", file=sys.stderr)

    live_cfg = load_strategy_config(CONFIG_PATH)
    baseline_cfg = dataclasses.replace(live_cfg, max_gap_atr_ratio=MAX_GAP_ATR_RATIO)

    variants = {
        "BASELINE (today's live config + new ceiling)": baseline_cfg,
        "BIDIR (bearish_only=False)": dataclasses.replace(baseline_cfg, bearish_only=False),
        "KZ (enable_kill_zone_filter=False — the OPEN direction today)":
            dataclasses.replace(baseline_cfg, enable_kill_zone_filter=False),
        "M15CONF (m15_confirmation=False)": dataclasses.replace(baseline_cfg, m15_confirmation=False),
        "VOL (h1_lookback 6->10, floor 0.25->0.10, pending 240->120, Tue back on)":
            dataclasses.replace(baseline_cfg, h1_sweep_lookback=10, min_gap_atr_ratio=0.10,
                                max_pending_bars=120, tuesday_exclusion=False),
    }

    results = []
    try:
        for label, cfg in variants.items():
            print(f"Running {label}...", file=sys.stderr)
            trades = BacktestEngine(str(tmp_csv), cfg).run()
            print(f"  {len(trades)} trades", file=sys.stderr)
            results.append(summarize(trades, label))
    finally:
        tmp_csv.unlink(missing_ok=True)

    print()
    print("=== Gate-opening battery vs today's live config — RAW pre-ML, derivation window only ===")
    print(f"window: {dmin} .. {dmax}  (HARD stop before 2026-03-01, sealed holdout untouched)")
    print("EXPLORATORY — no pre-registration, no committed pass bar. See docstring.")
    print()
    header = f"{'':<58} {'N':>5} {'Net PnL':>12} {'PF':>7} {'WR':>6} {'TSTOP%':>7}"
    print(header)
    print("-" * len(header))
    base_pf = results[0]["pf"]
    for r in results:
        pf_str = "inf" if r["pf"] == float("inf") else (f"{r['pf']:.3f}" if r["pf"] == r["pf"] else "n/a")
        wr_str = f"{r['wr']:.3f}" if r["wr"] == r["wr"] else "n/a"
        ts_str = f"{r['tstop_pct']:.1f}%" if r["tstop_pct"] == r["tstop_pct"] else "n/a"
        beats = ""
        if r["label"] != results[0]["label"] and r["pf"] == r["pf"]:
            beats = "  <- beats baseline" if r["pf"] > base_pf else ""
        print(f"{r['label']:<58} {r['n']:>5} {r['net_pnl']:>12.2f} {pf_str:>7} {wr_str:>6} {ts_str:>7}{beats}")
    print()
    print("Reminder: exploratory, in-sample on the pre-holdout window, RAW pre-ML population.")
    print("A variant beating baseline here is a lead to pre-register and test properly —")
    print("not a result to act on directly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
