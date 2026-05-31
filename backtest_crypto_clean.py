#!/usr/bin/env python3
"""
BTC backtest using the MNQ S25 trader engine (tier2_streaming_working.py).

Why: s26_crypto_streaming_working.py has two fatal bugs:
  1. Golden Flip inverts FVG direction → make_entry_decision always returns None
  2. _close_active_trade is a stub (pass) → completed_trades never populated

This script feeds BTC 1-min bars into Tier2StreamingTrader (the working MNQ code)
with BTC-calibrated config to test whether the S25 signal stack has edge on BTC.

BTC calibration vs MNQ:
  - atr_threshold: 0.25 (vs 0.5) — 1-min BTC FVGs smaller relative to ATR
  - min_gap_atr_ratio: 0.04 (vs 0.25) — H1 ATR ~700pts vs MNQ ~30pts
  - max_gap_dollars: 150 (vs 60) — BTC gap × POINT_VALUE_USD(2.0) headroom
  - max_daily_loss: -500 (vs -750) — 1 contract

Usage: .venv/bin/python backtest_crypto_clean.py
"""
import asyncio, csv, sys, logging, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytz

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).parent))

# Use the WORKING MNQ trader, not the broken crypto one
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.research.strategy_core import StrategyConfig
from src.data.models import DollarBar

ET_TZ    = pytz.timezone("America/New_York")
DATA_CSV = Path("data/kraken/PF_XBTUSD_1min.csv")


def load_data(start_year: int = 2025) -> list[dict]:
    rows = []
    with open(DATA_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["timestamp"][:4]) < start_year:
                continue
            rows.append({
                "timestamp": datetime.fromisoformat(row["timestamp"]),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": float(row["volume"]),
            })
    return rows


class MockClient:
    async def cancel_order(self, *a, **kw):              return True
    async def close_position_at_market(self, *a, **kw): return True
    async def submit_bracket_order(self, *a, **kw):     return ("BT1", "BT2", "BT3")


async def run_config(bars: list[dict], cfg: StrategyConfig) -> dict:
    t0 = time.perf_counter()

    trader = Tier2StreamingTrader(symbol="MNQM26")  # uses working close logic
    trader._strategy_config = cfg
    trader._ts_client = MockClient()

    # Silence all I/O side-effects
    trader._risk_manager._persist              = lambda: None
    trader._state_persistence.save_state       = lambda x: None
    trader._trade_logger.append_trade          = lambda x: None
    trader._log_filter_decision                = lambda *a, **kw: None
    if hasattr(trader, "ml_filter") and trader.ml_filter:
        trader.ml_filter._log_decision         = lambda *a, **kw: None
        trader.ml_filter.threshold             = cfg.ml_threshold
    # Silence equity curve writes
    if hasattr(trader, "_append_equity_curve"):
        trader._append_equity_curve            = lambda: None

    last_h1_key  = -1
    last_m15_key = -1
    n = len(bars)

    for i, raw in enumerate(bars):
        notional = max(raw["volume"] * raw["close"], 1.0)
        bar = DollarBar(
            timestamp=raw["timestamp"],
            open=raw["open"], high=raw["high"],
            low=raw["low"],  close=raw["close"],
            volume=int(raw["volume"]), notional_value=notional,
        )

        trader.dollar_bars.append(bar)
        if len(trader.dollar_bars) > 7500:
            del trader.dollar_bars[:-7500]
        trader._last_processed_timestamp = bar.timestamp

        bar_et = bar.timestamp.astimezone(ET_TZ)
        day    = bar_et.date()

        if trader._current_day != day:
            if trader._current_day is not None:
                trader._daily_ranges.append(trader._session_high - trader._session_low)
                if len(trader._daily_ranges) > 20:
                    trader._daily_ranges.pop(0)
            trader._current_day        = day
            trader._session_open_price = np.nan
            trader._session_high       = bar.high
            trader._session_low        = bar.low
            trader._risk_manager.check_and_update(bar_et, cfg.max_daily_loss)
        else:
            trader._session_high = max(trader._session_high, bar.high)
            trader._session_low  = min(trader._session_low,  bar.low)

        if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
            trader._session_open_price = bar.open

        # H1 update once per hour boundary
        h1_key = day.toordinal() * 24 + bar_et.hour
        if h1_key != last_h1_key:
            trader._update_h1_structure()
            last_h1_key = h1_key

        # M15 update once per 15-min boundary
        m15_key = h1_key * 4 + bar_et.minute // 15
        if m15_key != last_m15_key:
            trader._update_m15_choch()
            last_m15_key = m15_key

        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(bar, is_backfill=False)

    trades  = trader.completed_trades
    elapsed = time.perf_counter() - t0

    if not trades:
        return {"n": 0, "pnl": 0.0, "pf": 0.0, "wr": 0.0, "avg": 0.0, "elapsed": elapsed}

    pnls   = [t.pnl for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")
    wr     = 100 * len(wins) / len(pnls)
    return {
        "n":       len(pnls),
        "pnl":     sum(pnls),
        "pf":      pf,
        "wr":      wr,
        "avg":     sum(pnls) / len(pnls),
        "elapsed": elapsed,
    }


def make_cfg(gap_ratio: float, atr_th: float, ml: float,
             daily_loss: float = -500.0, m15: bool = True) -> StrategyConfig:
    return StrategyConfig(
        sl_multiplier=5.0, tp_multiplier=6.0, entry_pct=0.5,
        atr_threshold=atr_th,
        max_gap_dollars=150.0,    # BTC 150pt × POINT_VALUE_USD(2) = $300 cap
        max_hold_bars=60, max_pending_bars=240,
        contracts_per_trade=1,
        max_daily_loss=daily_loss,
        vol_regime_lookback=120, vol_regime_threshold=0.75,
        min_gap_atr_ratio=gap_ratio,
        ml_threshold=ml,
        bearish_only=True,         # pure bearish, no Golden Flip
        h1_sweep_lookback=6,
        tuesday_exclusion=False,   # crypto runs 24/7
        m15_confirmation=m15,
        enable_kill_zone_filter=False,
        commission_per_roundtrip=0.04,
    )


CONFIGS = [
    # (label,                          gap,   atr_th, ml,   daily, m15)
    ("S25-equiv  gap=0.04 atr=0.25",  0.04,  0.25,   0.0,  -500,  True),
    ("Looser     gap=0.03 atr=0.20",  0.03,  0.20,   0.0,  -500,  True),
    ("No CHoCH   gap=0.04 atr=0.25",  0.04,  0.25,   0.0,  -500,  False),
    ("Tighter    gap=0.05 atr=0.30",  0.05,  0.30,   0.0,  -500,  True),
    ("Wide open  gap=0.02 atr=0.15",  0.02,  0.15,   0.0,  -500,  False),
]


async def main():
    print("Loading BTC 1-min data (2025+2026)...")
    bars = load_data(start_year=2025)
    d0, d1 = bars[0]["timestamp"].date(), bars[-1]["timestamp"].date()
    print(f"  {len(bars):,} bars  ({d0} → {d1})")
    print(f"  Engine: tier2_streaming_working.py (S25 bearish-only, no Golden Flip)\n")

    hdr = f"  {'Config':<33} {'N':>6} {'PF':>6} {'WR%':>5} {'Avg$':>7} {'Total$':>9}  {'Time':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    results = []
    for label, gap, atr_th, ml, dloss, m15 in CONFIGS:
        cfg = make_cfg(gap, atr_th, ml, dloss, m15)
        r   = await run_config(bars, cfg)
        results.append((label, r))
        if r["n"] == 0:
            row = f"  {label:<33} {'0':>6} {'—':>6} {'—':>5} {'—':>7} {'—':>9}  {r['elapsed']:>4.0f}s"
        else:
            row = (f"  {label:<33} {r['n']:>6,} {r['pf']:>6.3f} {r['wr']:>5.1f} "
                   f"${r['avg']:>6.1f} ${r['pnl']:>+8.0f}  {r['elapsed']:>4.0f}s")
        print(row)

    print()
    valid = [(l, r) for l, r in results if r["n"] >= 10]
    if valid:
        best_l, best_r = max(valid, key=lambda x: x[1]["pf"])
        print(f"  Best (N≥10): {best_l}  →  PF={best_r['pf']:.3f}  N={best_r['n']}  P&L=${best_r['pnl']:+.0f}")
    else:
        print("  No configs produced ≥10 trades.")
    print()
    print("  Bearish-only: SHORT entries on bearish H1 sweep + M15 CHoCH + M1 FVG.")
    print("  1 contract, POINT_VALUE_USD=$2/pt (MNQ engine), commission $0.04/RT.")


if __name__ == "__main__":
    asyncio.run(main())
