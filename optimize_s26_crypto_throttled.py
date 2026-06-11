import os
os.environ["IS_BACKTEST"] = "1"

import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytz
from concurrent.futures import ProcessPoolExecutor
import itertools
import pandas as pd

logging.getLogger().setLevel(logging.CRITICAL)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.research.s26_crypto_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar
from src.research.strategy_core import StrategyConfig, calc_profit_factor, calc_max_drawdown_pct

ET_TZ = pytz.timezone("America/New_York")

print("Loading Kraken data from CSV...")
csv_path = Path("/root/Silver-Bullet-ML-BMAD/data/kraken/PF_XBTUSD_1min.csv")
global_bars_data = []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["timestamp"].startswith("2026"): continue
        ts = datetime.fromisoformat(row["timestamp"])
        global_bars_data.append({
            "timestamp": ts,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"])
        })

def run_backtest(params):
    gap, m15_conf, sl, tp, ml_thresh = params
    
    cfg = StrategyConfig(
        sl_multiplier=sl, tp_multiplier=tp, entry_pct=0.5, atr_threshold=0.5,
        max_gap_dollars=6000.0, max_hold_bars=60, max_pending_bars=240,
        contracts_per_trade=1, max_daily_loss=-500.0, vol_regime_lookback=120,
        vol_regime_threshold=0.75, min_gap_atr_ratio=gap, ml_threshold=ml_thresh,
        bearish_only=False, h1_sweep_lookback=6, tuesday_exclusion=False,
        m15_confirmation=m15_conf, enable_kill_zone_filter=False,
        commission_per_roundtrip=0.04
    )

    trader = Tier2StreamingTrader(symbol="PF_XBTUSD")
    trader._strategy_config = cfg
    
    # Mute side effects
    trader._risk_manager._persist = lambda: None
    trader._state_persistence.save_state = lambda x: None
    trader._trade_logger.append_trade = lambda x: None
    trader._log_filter_decision = lambda *args, **kwargs: None
    if hasattr(trader, "ml_filter") and trader.ml_filter:
        trader.ml_filter._log_decision = lambda *args, **kwargs: None
    
    # MOCK _ts_client calls since this is a pure backtest
    class MockClient:
        async def cancel_order(self, *args, **kwargs): return True
        async def close_position_at_market(self, *args, **kwargs): return True
        async def submit_bracket_order(self, *args, **kwargs): return ("1","2","3")
    trader._ts_client = MockClient()
    
    async def replay():
        last_h1_hour = -1
        last_m15_min = -1
        for i, raw in enumerate(global_bars_data):
            notional = raw["volume"] * raw["close"]
            if notional <= 0: notional = 1.0
            bar = DollarBar(
                timestamp=raw["timestamp"], open=raw["open"], high=raw["high"],
                low=raw["low"], close=raw["close"], volume=int(raw["volume"]),
                notional_value=notional
            )
            
            trader.dollar_bars.append(bar)
            if len(trader.dollar_bars) > 7500: del trader.dollar_bars[:-7500]
            trader._last_processed_timestamp = bar.timestamp

            bar_et = bar.timestamp.astimezone(ET_TZ)
            if trader._current_day != bar_et.date():
                if trader._current_day is not None:
                    trader._daily_ranges.append(trader._session_high - trader._session_low)
                    if len(trader._daily_ranges) > 20: trader._daily_ranges.pop(0)
                trader._current_day = bar_et.date()
                trader._session_open_price = np.nan
                trader._session_high, trader._session_low = bar.high, bar.low
                trader._risk_manager.check_and_update(bar_et, trader._strategy_config.max_daily_loss)
            else:
                trader._session_high = max(trader._session_high, bar.high)
                trader._session_low  = min(trader._session_low,  bar.low)
                
            if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
                trader._session_open_price = bar.open

            if bar_et.hour != last_h1_hour or last_h1_hour == -1:
                trader._update_h1_structure()
                last_h1_hour = bar_et.hour
                
            current_m15_bin = bar_et.minute // 15
            if current_m15_bin != last_m15_min or last_m15_min == -1:
                trader._update_m15_choch()
                last_m15_min = current_m15_bin

            await trader._advance_active_trade(bar)
            await trader._detect_and_enter(bar, is_backfill=False)
            
    asyncio.run(replay())
    
    trades = trader.completed_trades
    n_trades = len(trades)
    pnls = [t.pnl for t in trades]
    net_pnl = sum(pnls)
    pf = calc_profit_factor(pnls) if n_trades > 0 else 0.0
    wr = sum(1 for p in pnls if p > 0) / n_trades * 100 if n_trades > 0 else 0.0
    
    return {
        "gap_ratio": gap, "m15_choch": m15_conf, "sl_mult": sl, "tp_mult": tp, "ml_thresh": ml_thresh,
        "trades": n_trades, "net_pnl": round(net_pnl, 2), "pf": round(pf, 2), "wr_%": round(wr, 1)
    }

def main():
    gaps = [0.10, 0.15, 0.20]
    m15s = [True]
    sls = [3.0, 5.0]
    tps = [4.0, 6.0, 8.0]
    mls = [0.4, 0.5, 0.6]
    
    grid = list(itertools.product(gaps, m15s, sls, tps, mls))
    print(f"Running {len(grid)} combinations sequentially with ProcessPoolExecutor...")
    
    results = []
    with ProcessPoolExecutor(max_workers=1) as executor:
        for i, res in enumerate(executor.map(run_backtest, grid)):
            results.append(res)
            print(f"Progress: {i+1}/{len(grid)} - {res}")
            
    df = pd.DataFrame(results)
    df = df.sort_values("net_pnl", ascending=False)
    print("\nTop 15 Combinations by Net PnL (2026 YTD):")
    print(df.head(15).to_string(index=False))
    df.to_csv("logs/s26_crypto_optimization.csv", index=False)

if __name__ == "__main__":
    main()
