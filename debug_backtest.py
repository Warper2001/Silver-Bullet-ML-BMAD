import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytz

# Set logging to INFO to see what is happening!
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
            if not row["timestamp"].startswith("2026-01-1"):  # load a slice in mid-January to ensure warmup is already complete
                continue
            ts = datetime.fromisoformat(row["timestamp"])
            bars_data.append({
                "timestamp": ts,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"])
            })

    print(f"Loaded {len(bars_data)} 1-minute bars.")
    
    trader = Tier2StreamingTrader(symbol="PF_XBTUSD")
    trader._risk_manager._persist = lambda: None
    trader._state_persistence.save_state = lambda x: None
    trader._trade_logger.append_trade = lambda x: None

    # Let's populate some initial bars first to pre-warm the buffer
    print("Running debug backtest...")
    for i, raw in enumerate(bars_data[:3000]):
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

    print(f"Completed trades: {len(trader.completed_trades)}")

if __name__ == "__main__":
    asyncio.run(main())
