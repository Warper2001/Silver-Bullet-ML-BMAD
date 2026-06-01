#!/usr/bin/env python3
"""
Backtest for pre-registered hypothesis btc-frrf (BTC Funding Rate Regime Filter).

Uses s26_crypto_streaming_working.py with FundingRateFilter loaded from
data/kraken/PF_XBTUSD_funding_rate.csv (mark/spot basis proxy).

Config: strategy_config_btc_frrf.yaml
Decision threshold: PF > 1.20 AND N >= 30

Usage: .venv/bin/python backtest_btc_frrf.py
"""
import asyncio, csv, os, sys, logging, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pytz

os.environ["IS_BACKTEST"] = "1"
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).parent))

from src.research.s26_crypto_streaming_working import Tier2StreamingTrader
from src.research.config_loader import load_strategy_config
from src.data.models import DollarBar

ET_TZ        = pytz.timezone("America/New_York")
PRICE_CSV    = Path("data/kraken/PF_XBTUSD_1min.csv")
FUNDING_CSV  = Path("data/kraken/PF_XBTUSD_funding_rate.csv")
CONFIG       = Path("strategy_config_btc_frrf_s27v2.yaml")


def load_price_data(start_year: int = 2025) -> list[dict]:
    rows = []
    with open(PRICE_CSV) as f:
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


async def run(bars: list[dict]) -> dict:
    cfg = load_strategy_config(CONFIG)
    t0  = time.perf_counter()

    trader = Tier2StreamingTrader(symbol="PF_XBTUSD")
    trader._strategy_config = cfg
    trader._ts_client = MockClient()

    # Load historical funding rates
    trader.funding_filter.load_historical(FUNDING_CSV)

    # Silence side effects
    trader._risk_manager._persist        = lambda: None
    trader._state_persistence.save_state = lambda x: None
    trader._trade_logger.append_trade    = lambda x: None
    trader._log_filter_decision          = lambda *a, **kw: None
    if trader.ml_filter:
        trader.ml_filter._log_decision   = lambda *a, **kw: None
        trader.ml_filter.threshold       = cfg.ml_threshold

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

        h1_key = day.toordinal() * 24 + bar_et.hour
        if h1_key != last_h1_key:
            trader._update_h1_structure()
            last_h1_key = h1_key

        m15_key = h1_key * 4 + bar_et.minute // 15
        if m15_key != last_m15_key:
            trader._update_m15_choch()
            last_m15_key = m15_key

        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(bar, is_backfill=False)

    elapsed = time.perf_counter() - t0
    trades  = trader.completed_trades

    if not trades:
        return {"n": 0, "elapsed": elapsed, "trades": []}

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
        "trades":  trades,
    }


async def main():
    print(f"Config:        {CONFIG}")
    print(f"Funding data:  {FUNDING_CSV}")
    print("Loading BTC 1-min data (2025+2026)...")
    bars = load_price_data(start_year=2025)
    d0, d1 = bars[0]["timestamp"].date(), bars[-1]["timestamp"].date()
    print(f"  {len(bars):,} bars  ({d0} → {d1})\n")

    r = await run(bars)

    if r["n"] == 0:
        print("  0 trades — check FundingRateFilter loaded, sweep detection, gap params.")
        return

    print(f"  Total trades:  {r['n']:,}")
    print(f"  PF:            {r['pf']:.3f}  (decision threshold: > 1.20)")
    print(f"  Win rate:      {r['wr']:.1f}%")
    print(f"  Avg P&L:       ${r['avg']:+.2f}")
    print(f"  Total P&L:     ${r['pnl']:+,.2f}")
    print(f"  Time:          {r['elapsed']:.0f}s\n")

    # Direction breakdown
    longs  = [t for t in r["trades"] if t.direction == "LONG"]
    shorts = [t for t in r["trades"] if t.direction == "SHORT"]
    for direction, subset in [("SHORT", shorts), ("LONG", longs)]:
        if not subset:
            print(f"  {direction}: 0 trades")
            continue
        spnls  = [t.pnl for t in subset]
        swins  = [p for p in spnls if p > 0]
        slosse = [p for p in spnls if p <= 0]
        spf    = sum(swins) / abs(sum(slosse)) if slosse else float("inf")
        print(f"  {direction}: N={len(subset)}  PF={spf:.3f}  WR={100*len(swins)/len(spnls):.1f}%  "
              f"P&L=${sum(spnls):+,.2f}")

    # Exit breakdown
    print()
    exits: dict[str, int] = {}
    for t in r["trades"]:
        key = getattr(t, "exit_type", None) or getattr(t, "exit_reason", "?")
        exits[key] = exits.get(key, 0) + 1
    print("  Exit breakdown:")
    for k, v in sorted(exits.items(), key=lambda x: -x[1]):
        print(f"    {k:<12} {v:>4}  ({100*v/r['n']:.1f}%)")

    print()
    verdict = "PASS ✅" if r["pf"] > 1.20 and r["n"] >= 30 else \
              "INSUFFICIENT N" if r["n"] < 30 else "FAIL ❌"
    print(f"  Backtest verdict: {verdict}")
    print(f"  (Informational only — live N≥30 AND 60 days decides the study)")


if __name__ == "__main__":
    asyncio.run(main())
