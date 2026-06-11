import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytz

logging.basicConfig(level=logging.WARNING, format="%(message)s")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.research.s26_crypto_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

ET_TZ = pytz.timezone("US/Eastern")

async def main():
    print("Loading Kraken data from CSV...")
    csv_path = Path("data/kraken/PF_XBTUSD_1min.csv")
    bars_data = []
    
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["timestamp"].startswith("2026"):
                continue
            # "2024-11-08T00:00:00+00:00"
            ts = datetime.fromisoformat(row["timestamp"])
            bars_data.append({
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            })

    print(f"Loaded {len(bars_data)} 1-minute bars for 2026 YTD.")
    
    trader = Tier2StreamingTrader(symbol="PF_XBTUSD")
    # Turn off risk manager persistence
    trader._risk_manager._persist = lambda: None
    trader._state_persistence.save_state = lambda x: None
    trader._trade_logger.append_trade = lambda x: None

    print("Running backtest replay...")
    for i, raw in enumerate(bars_data):
        notional = raw["volume"] * raw["close"]
        if notional <= 0: notional = 1.0
        
        bar = DollarBar(
            timestamp=raw["timestamp"],
            open=raw["open"],
            high=raw["high"],
            low=raw["low"],
            close=raw["close"],
            volume=int(raw["volume"]),
            notional_value=notional
        )
        
        trader.dollar_bars.append(bar)
        if len(trader.dollar_bars) > 7500:
            del trader.dollar_bars[:-7500]
            
        trader._last_processed_timestamp = bar.timestamp

        bar_et = bar.timestamp.astimezone(ET_TZ)
        if trader._current_day != bar_et.date():
            if trader._current_day is not None:
                trader._daily_ranges.append(trader._session_high - trader._session_low)
                if len(trader._daily_ranges) > 20:
                    trader._daily_ranges.pop(0)
            trader._current_day = bar_et.date()
            trader._session_open_price = np.nan
            trader._session_high, trader._session_low = bar.high, bar.low
            # Reset daily risk
            trader._risk_manager.check_and_update(bar_et, trader._strategy_config.max_daily_loss)
        else:
            trader._session_high = max(trader._session_high, bar.high)
            trader._session_low  = min(trader._session_low,  bar.low)
            
        if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
            trader._session_open_price = bar.open

        if bar_et.minute == 0:
            trader._update_h1_structure()
        if bar_et.minute % 15 == 0:
            trader._update_m15_choch()
        await trader._advance_active_trade(bar)
        
        await trader._detect_and_enter(bar, is_backfill=False)

    print()
    print("=" * 65)
    print("S26 CRYPTO BACKTEST TRADES (2026 YTD)")
    print("=" * 65)
    for t in trader.completed_trades:
        print(
            f"  {t.exit_type.upper():<4}  {t.direction}  "
            f"entry={t.entry_price:.2f}  exit={t.exit_price:.2f}  "
            f"P&L=${t.pnl:+.2f}  "
            f"{t.entry_time.strftime('%Y-%m-%d %H:%M')} -> {t.exit_time.strftime('%H:%M')} UTC"
        )
        
    bt_count = len(trader.completed_trades)
    bt_pnl   = sum(t.pnl for t in trader.completed_trades)
    print(f"\n  Trades filled: {bt_count}")
    print(f"  Net P&L:       ${bt_pnl:+.2f}")

if __name__ == "__main__":
    asyncio.run(main())
