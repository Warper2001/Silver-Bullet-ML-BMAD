import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytz
import pandas as pd

logging.getLogger().setLevel(logging.CRITICAL)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.research.s26_crypto_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar
from src.research.strategy_core import StrategyConfig, calc_profit_factor

ET_TZ = pytz.timezone("US/Eastern")

print("Loading Kraken data from CSV...")
csv_path = Path("data/kraken/PF_XBTUSD_1min.csv")
bars_data = []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not row["timestamp"].startswith("2026"): continue
        ts = datetime.fromisoformat(row["timestamp"])
        bars_data.append({
            "timestamp": ts,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"])
        })

print(f"Loaded {len(bars_data)} bars. Running optimization grid...")

def run_backtest(params):
    gap, m15_conf, sl, tp = params
    
    cfg = StrategyConfig(
        sl_multiplier=sl, tp_multiplier=tp, entry_pct=0.5, atr_threshold=0.5,
        max_gap_dollars=6000.0, max_hold_bars=60, max_pending_bars=240,
        contracts_per_trade=1, max_daily_loss=-500.0, vol_regime_lookback=120,
        vol_regime_threshold=0.75, min_gap_atr_ratio=gap, ml_threshold=0.0,
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
    
    # MOCK _ts_client calls
    class MockClient:
        async def cancel_order(self, *args, **kwargs): return True
        async def close_position_at_market(self, *args, **kwargs): return True
        async def submit_bracket_order(self, *args, **kwargs): return ("1","2","3")
    trader._ts_client = MockClient()
    
    async def replay():
        for i, raw in enumerate(bars_data):
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

            trader._update_h1_structure()
            trader._update_m15_choch()
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
        "gap_ratio": gap, "m15_choch": m15_conf, "sl_mult": sl, "tp_mult": tp,
        "trades": n_trades, "net_pnl": round(net_pnl, 2), "pf": round(pf, 2), "wr_%": round(wr, 1)
    }

def main():
    gaps = [0.10, 0.15, 0.20, 0.25, 0.35]
    m15s = [True, False]
    sls = [3.0, 5.0]
    tps = [4.0, 6.0, 8.0]
    
    grid = list(itertools.product(gaps, m15s, sls, tps))
    print(f"Running {len(grid)} combinations sequentially...")
    
    results = []
    for i, params in enumerate(grid):
        res = run_backtest(params)
        results.append(res)
        print(f"[{i+1}/{len(grid)}] gap={res['gap_ratio']} m15={res['m15_choch']} SL={res['sl_mult']}x TP={res['tp_mult']}x | Trades={res['trades']} PnL=${res['net_pnl']} PF={res['pf']} WR={res['wr_%']}%")
            
    df = pd.DataFrame(results)
    df = df.sort_values("net_pnl", ascending=False)
    print("\n" + "="*80)
    print("TOP 15 COMBINATIONS BY NET P&L (2026 YTD)")
    print("="*80)
    print(df.head(15).to_string(index=False))
    df.to_csv("logs/s26_crypto_optimization.csv", index=False)
    print("\nSaved to logs/s26_crypto_optimization.csv")

if __name__ == "__main__":
    main()
