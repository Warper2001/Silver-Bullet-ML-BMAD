#!/usr/bin/env python3
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import joblib
import httpx
import sys
import csv
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.execution.kraken.market_data.history import KrakenHistoryClient

# Logging setup
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[
    logging.FileHandler(log_dir / 's26_soft_fvg_streaming.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

class S26SoftFVGTrader:
    def __init__(self):
        self.symbol = "PF_XBTUSD"
        self.client = KrakenHistoryClient(httpx.AsyncClient(timeout=15.0))
        self.model_path = Path(__file__).parent.parent.parent / "models/s26_soft_fvg_ml_model.pkl"
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
        
        # S26 Parameters
        self.length = 20
        self.sl_mult = 2.0
        self.tp_mult = 4.0
        self.ml_thresh = 0.62
        self.max_hold = 60
        self.et_tz = pytz.timezone("America/New_York")
        
        self.bars = []
        self.active_trade = None
        self.last_ts = None
        
        self.trade_log_path = log_dir / "s26_soft_fvg_trade_log.csv"
        from src.monitoring.trade_db import TradeDatabase
        self.db = TradeDatabase()

    async def run(self):
        logger.info("Starting S26 Soft-FVG + Sweep Trader...")
        if not self.model:
            logger.error("ML Model not found! Exiting.")
            return
        
        while True:
            try:
                # Fetch 400 bars on first load to cover 360m sweep + padding
                count = 400 if not self.last_ts else 2
                kb_bars = await self.client.fetch_bars(self.symbol, interval="1m", count=count)
                
                if not kb_bars:
                    await asyncio.sleep(10)
                    continue
                    
                for kb in kb_bars:
                    if self.last_ts and kb.time <= self.last_ts:
                        continue
                    
                    bar = {
                        'timestamp': kb.time,
                        'open': float(kb.open),
                        'high': float(kb.high),
                        'low': float(kb.low),
                        'close': float(kb.close),
                        'volume': float(kb.volume)
                    }
                    self.bars.append(bar)
                    self.last_ts = kb.time
                
                # Keep rolling window bounded to 400
                if len(self.bars) > 400:
                    self.bars = self.bars[-400:]
                    
                if len(self.bars) >= 380:
                    self.process_latest()
                    
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
            
            await asyncio.sleep(60)

    def process_latest(self):
        df = pd.DataFrame(self.bars).set_index('timestamp')
        
        # 1. Indicators
        df['prev_close'] = df['close'].shift(1)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['prev_close'])
        df['tr2'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(self.length).mean()
        
        # S26 Sweep & Soft FVG
        df['h1_high'] = df['high'].rolling(360).max()
        df['h1_low'] = df['low'].rolling(360).min()
        df['sweep_bear'] = (df['high'] >= df['h1_high'].shift(1))
        df['sweep_bull'] = (df['low'] <= df['h1_low'].shift(1))
        df['recent_sweep_bear'] = df['sweep_bear'].astype(int).rolling(60).max() > 0
        df['recent_sweep_bull'] = df['sweep_bull'].astype(int).rolling(60).max() > 0
        
        df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2 * df['atr'])
        df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2 * df['atr'])
        
        # Signals
        df['long_cond'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
        df['short_cond'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
        
        # Prevent consecutive signals in same direction
        df['long_cond'] = df['long_cond'] & (~df['long_cond'].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool))
        df['short_cond'] = df['short_cond'] & (~df['short_cond'].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool))
        
        # ML Features
        df['vol_sma'] = df['volume'].rolling(50).mean()
        df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
        df['rvol'] = df['rvol'].fillna(1.0)
        
        df['macro_ema'] = df['close'].ewm(span=200).mean()
        df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']
        
        df['ema'] = df['close'].ewm(span=self.length, adjust=False).mean()
        df['dist_ema'] = (df['close'] - df['ema']) / df['atr']
        
        # Time conversions
        df_et = df.index.tz_convert(self.et_tz)
        df['hour_et'] = df_et.hour
        df['minute_et'] = df_et.minute
        df['dow'] = df_et.dayofweek
        df['is_us_session'] = (((df['hour_et'] == 9) & (df['minute_et'] >= 30)) | 
                               ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | 
                               ((df['hour_et'] == 16) & (df['minute_et'] == 0))).astype(int)

        # Fill NaNs in features
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in ['atr', 'rvol', 'dist_ema', 'dist_macro_ema']:
            df[col] = df[col].fillna(0)
            
        current_bar = df.iloc[-1]
        last_bar = df.iloc[-2]
        
        # 2. Manage active trade
        if self.active_trade:
            self.active_trade['hold_time'] += 1
            t = self.active_trade
            
            exit_reason = None
            exit_price = 0
            
            if t['dir'] == 'L':
                if current_bar['low'] <= t['sl']:
                    exit_reason, exit_price = 'SL', t['sl']
                elif current_bar['high'] >= t['tp']:
                    exit_reason, exit_price = 'TP', t['tp']
            else:
                if current_bar['high'] >= t['sl']:
                    exit_reason, exit_price = 'SL', t['sl']
                elif current_bar['low'] <= t['tp']:
                    exit_reason, exit_price = 'TP', t['tp']
                    
            if not exit_reason and t['hold_time'] >= self.max_hold:
                exit_reason, exit_price = 'TIME_STOP', current_bar['close']
                
            if exit_reason:
                pnl = (exit_price - t['entry']) if t['dir'] == 'L' else (t['entry'] - exit_price)
                logger.info(f"🏁 Trade Closed [{exit_reason}]: PNL=${pnl:.2f} | Hold: {t['hold_time']}m")
                self.log_trade(t, exit_price, exit_reason, pnl)
                self.active_trade = None
                return # Done for this cycle
        
        # 3. Entry Logic (only if flat)
        if not self.active_trade:
            long_cond = last_bar['long_cond']
            short_cond = last_bar['short_cond']
            
            if long_cond or short_cond:
                direction = 1 if long_cond else 0
                dir_str = 'L' if long_cond else 'S'
                
                # Extract features for ML (from last_bar)
                features = pd.DataFrame([{
                    'dir': direction,
                    'atr': last_bar['atr'],
                    'rvol': last_bar['rvol'],
                    'dist_ema': last_bar['dist_ema'],
                    'dist_macro_ema': last_bar['dist_macro_ema'],
                    'hour_et': df_et[-2].hour,
                    'dow': df_et[-2].dayofweek,
                    'is_us_session': last_bar['is_us_session']
                }])
                
                proba = self.model.predict_proba(features)[0, 1]
                
                if proba >= self.ml_thresh:
                    # Realistic execution: enter at the open of the current forming bar
                    entry_price = current_bar['open'] 
                    atr = last_bar['atr']
                    
                    sl = entry_price - (atr * self.sl_mult) if direction == 1 else entry_price + (atr * self.sl_mult)
                    tp = entry_price + (atr * self.tp_mult) if direction == 1 else entry_price - (atr * self.tp_mult)
                    
                    self.active_trade = {
                        'dir': dir_str,
                        'entry': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'atr': atr,
                        'proba': proba,
                        'hold_time': 0,
                        'ts': df.index[-1]
                    }
                    logger.info(f"🔔 S26 Soft-FVG ENTRY {dir_str}: Price={entry_price:.2f} | P(Success)={proba:.3f} | SL={sl:.2f} | TP={tp:.2f}")
                else:
                    # Filtered out by ML
                    pass

    def log_trade(self, t, exit_price, reason, pnl):
        # Log to DB
        self.db.log_trade(
            trader_id='trader-s26',
            timestamp=t['ts'].isoformat(),
            pnl=round(pnl, 2),
            direction=t['dir'],
            entry_price=round(t['entry'], 2),
            exit_price=round(exit_price, 2),
            exit_reason=reason,
            ml_proba=round(t['proba'], 3)
        )
        # Maintain legacy CSV for backward compatibility/redundancy
        write_header = not self.trade_log_path.exists()
        with open(self.trade_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['entry_time', 'direction', 'entry_price', 'exit_price', 'reason', 'pnl', 'ml_proba'])
            writer.writerow([t['ts'].isoformat(), t['dir'], round(t['entry'], 2), round(exit_price, 2), reason, round(pnl, 2), round(t['proba'], 3)])

if __name__ == "__main__":
    trader = S26SoftFVGTrader()
    asyncio.run(trader.run())