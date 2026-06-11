#!/usr/bin/env python3
"""
Parameter optimization script for MNQ bot.
Runs a grid search over:
- ml_threshold
- sl_multiplier
- tp_multiplier
- enable_kill_zone_filter

Using 2026 YTD dollar bars.
"""

import os
import sys
import asyncio
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pytz
import yaml
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent))

# Suppress standard logging to prevent log pollution and slow execution
logging.disable(logging.CRITICAL)

import src.research.tier2_streaming_working as tier2_mod
from src.research.tier2_streaming_working import Tier2StreamingTrader
from src.data.models import DollarBar

CSV_2026 = Path("data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv")
ET_TZ = pytz.timezone("US/Eastern")

def load_bars() -> list[DollarBar]:
    bars = []
    with open(CSV_2026) as f:
        for row in csv.DictReader(f):
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bars.append(DollarBar.model_construct(
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(float(row["volume"])),
                notional_value=float(row["notional"]),
                is_forward_filled=False,
            ))
    return bars

async def run_backtest_with_config(bars: list[DollarBar], config_dict: dict) -> tuple[int, float, float, float]:
    # Write config_dict to a temp yaml file
    temp_yaml = Path("temp_config_opt.yaml")
    with open(temp_yaml, 'w') as f:
        yaml.dump(config_dict, f)
    
    # Set environment variable STRATEGY_CONFIG_PATH
    os.environ["STRATEGY_CONFIG_PATH"] = str(temp_yaml.resolve())
    
    # Re-instantiate trader so it loads the new config
    trader = Tier2StreamingTrader()
    
    # Fast single-window OLS slope calculation for the regime filter
    def fast_allows(bars: list, signal_direction: str) -> bool:
        if not trader.lr_filter.enabled:
            return True
        if len(bars) < trader.lr_filter.slow_len:
            return True
        try:
            closes_slow = np.array([b.close for b in bars[-trader.lr_filter.slow_len:]], dtype=np.float64)
            closes_fast = closes_slow[-trader.lr_filter.fast_len:]
            
            # Slow slope OLS
            L_slow = float(trader.lr_filter.slow_len)
            x_slow = np.arange(L_slow)
            sy_slow = closes_slow.sum()
            xy_slow = np.dot(closes_slow, x_slow)
            sx_slow = 0.5 * L_slow * (L_slow - 1)
            sx2_slow = L_slow * (L_slow - 1) * (2 * L_slow - 1) / 6.0
            denom_slow = L_slow * sx2_slow - sx_slow * sx_slow
            slope_slow = (L_slow * xy_slow - sx_slow * sy_slow) / denom_slow
            
            # Fast slope OLS
            L_fast = float(trader.lr_filter.fast_len)
            x_fast = np.arange(L_fast)
            sy_fast = closes_fast.sum()
            xy_fast = np.dot(closes_fast, x_fast)
            sx_fast = 0.5 * L_fast * (L_fast - 1)
            sx2_fast = L_fast * (L_fast - 1) * (2 * L_fast - 1) / 6.0
            denom_fast = L_fast * sx2_fast - sx_fast * sx_fast
            slope_fast = (L_fast * xy_fast - sx_fast * sy_fast) / denom_fast
            
            # Classify regime
            if slope_slow > 0 and slope_fast > 0:
                regime = "UP"
            elif slope_slow < 0 and slope_fast < 0:
                regime = "DOWN"
            else:
                regime = "SIDEWAYS"
                
            if signal_direction == "bullish":
                return regime in ("DOWN", "SIDEWAYS")
            else:
                return regime in ("UP", "SIDEWAYS")
        except Exception as e:
            return True
            
    trader.lr_filter.allows = fast_allows

    # Mock client and other side effects
    mock_client = MagicMock()
    mock_client.submit_bracket_order = AsyncMock(return_value=(None, None, None))
    mock_client.cancel_order = AsyncMock(return_value=True)
    mock_client.close_position_at_market = AsyncMock(return_value=None)
    mock_client.reconcile_state = AsyncMock(return_value=None)
    trader._ts_client = mock_client
    trader.ml_filter._log_decision = lambda *a, **kw: None
    
    # Suppress save state
    tier2_mod.StatePersistence.save_state = lambda state: None
    
    last_h1_ts = None
    
    for bar in bars[-15000:]:
        trader.dollar_bars.append(bar)
        trader.dollar_bars_dict.append({
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "notional_value": bar.notional_value,
            "is_forward_filled": bar.is_forward_filled
        })
        if len(trader.dollar_bars) > 7500:
            del trader.dollar_bars[:-7500]
            del trader.dollar_bars_dict[:-7500]
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
        else:
            trader._session_high = max(trader._session_high, bar.high)
            trader._session_low  = min(trader._session_low,  bar.low)
        if np.isnan(trader._session_open_price) and bar_et.hour >= 6:
            trader._session_open_price = bar.open

        h1_ts = bar.timestamp.replace(minute=0, second=0, microsecond=0)
        if h1_ts != last_h1_ts:
            trader._update_h1_structure()
            last_h1_ts = h1_ts

        trader._update_m15_choch()
        await trader._advance_active_trade(bar)
        await trader._detect_and_enter(bar, is_backfill=False)
        
    # Clean up temp config file
    if temp_yaml.exists():
        temp_yaml.unlink()
        
    trades = trader.completed_trades
    if not trades:
        return 0, 0.0, 0.0, 0.0
        
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = wins / len(trades)
    
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = sum(t.pnl for t in trades if t.pnl < 0)
    total_pnl = gross_profit + gross_loss
    
    pf = gross_profit / abs(gross_loss) if gross_loss < 0 else float('inf')
    
    return len(trades), win_rate, total_pnl, pf

async def main():
    print("🚀 LOADING 2026 YTD DOLLAR BARS...")
    bars = load_bars()
    print(f"✅ Loaded {len(bars):,} bars.")
    
    # Load base configuration as template
    base_config_path = Path("strategy_config.yaml")
    if base_config_path.exists():
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {}
        
    # Define optimization grid
    ml_thresholds = [0.0, 0.50, 0.60]
    sl_multipliers = [3.0, 5.0]
    tp_multipliers = [4.0, 6.0]
    killzones = [False]
    
    results = []
    
    total_runs = len(ml_thresholds) * len(sl_multipliers) * len(tp_multipliers) * len(killzones)
    print(f"🔍 STARTING GRID SEARCH ({total_runs} combinations)...")
    print("=" * 85)
    print(f"{'ML_TH':<6} | {'SL':<4} | {'TP':<4} | {'KZ':<5} | {'TRADES':<6} | {'WIN%':<6} | {'PF':<6} | {'PNL ($)':<12}")
    print("-" * 85)
    
    run_idx = 0
    for ml_th in ml_thresholds:
        for sl_m in sl_multipliers:
            for tp_m in tp_multipliers:
                for kz in killzones:
                    run_idx += 1
                    
                    # Setup config overrides
                    cfg = dict(base_config)
                    cfg["ml_threshold"] = ml_th
                    cfg["sl_multiplier"] = sl_m
                    cfg["tp_multiplier"] = tp_m
                    cfg["enable_kill_zone_filter"] = kz
                    
                    try:
                        n_trades, wr, pnl, pf = await run_backtest_with_config(bars, cfg)
                        
                        pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                        print(f"{ml_th:<6.2f} | {sl_m:<4.1f} | {tp_m:<4.1f} | {str(kz):<5} | {n_trades:<6d} | {wr:<6.1%} | {pf_str:<6} | {pnl:>+11.2f}")
                        sys.stdout.flush()
                        
                        results.append({
                            "ml_threshold": ml_th,
                            "sl_multiplier": sl_m,
                            "tp_multiplier": tp_m,
                            "enable_kill_zone_filter": kz,
                            "trades": n_trades,
                            "win_rate": wr,
                            "pnl": pnl,
                            "pf": pf
                        })
                    except Exception as e:
                        print(f"Error on run {run_idx}: {e}")
                        
    print("=" * 85)
    print("\n🏆 TOP 10 CONFIGURATIONS RANKED BY PROFIT FACTOR:")
    print("=" * 85)
    print(f"{'Rank':<4} | {'ML_TH':<6} | {'SL':<4} | {'TP':<4} | {'KZ':<5} | {'TRADES':<6} | {'WIN%':<6} | {'PF':<6} | {'PNL ($)':<12}")
    print("-" * 85)
    
    # Sort by PF descending
    # Treat 'inf' as a very high float
    ranked = sorted(results, key=lambda x: (x["pf"] if x["pf"] != float('inf') else 9999.0, x["pnl"]), reverse=True)
    
    rank_idx = 0
    for r in ranked[:10]:
        rank_idx += 1
        pf_str = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "inf"
        print(f"#{rank_idx:<3} | {r['ml_threshold']:<6.2f} | {r['sl_multiplier']:<4.1f} | {r['tp_multiplier']:<4.1f} | {str(r['enable_kill_zone_filter']):<5} | {r['trades']:<6d} | {r['win_rate']:<6.1%} | {pf_str:<6} | {r['pnl']:>+11.2f}")
        
    print("=" * 85)

if __name__ == "__main__":
    asyncio.run(main())
