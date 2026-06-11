#!/usr/bin/env python3
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import joblib
import httpx
import sys
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3

# Logging setup
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[
    logging.FileHandler(log_dir / 's26_combine_bot.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

SIM_ACCOUNT_ID = "SIM2248559M"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/brokerage/orders"
BARS_URL = "https://api.tradestation.com/v3/marketdata/barcharts/MBTM26?interval=1&unit=Minute"

class S26CombineTrader:
    def __init__(self):
        self.symbol = "MBTM26"
        self.contracts = 5  # 5 MBT contracts = 0.5 BTC
        
        self.model_path = Path(__file__).parent.parent.parent / "models/s26_soft_fvg_ml_model.pkl"
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
        
        # S26 Optimized Parameters for Combine
        self.length = 20
        self.sl_mult = 2.0
        self.tp_mult = 4.0
        self.ml_thresh = 0.62
        self.max_hold = 60
        self.et_tz = pytz.timezone("America/New_York")
        
        self.bars = []
        self.last_ts = None
        
        self.running = False
        self.auth = None
        self.http = None

    async def initialize(self):
        logger.info("=" * 70)
        logger.info(f"S26 COMBINE TRADER - {self.symbol} - {self.contracts} CONTRACTS")
        logger.info(f"Threshold: {self.ml_thresh} | SL: {self.sl_mult}x | TP: {self.tp_mult}x")
        logger.info("=" * 70)
        
        self.auth = TradeStationAuthV3.from_file('.access_token')
        await self.auth.authenticate()
        await self.auth.start_auto_refresh()
        self.http = httpx.AsyncClient(timeout=30.0)

    async def run(self):
        await self.initialize()
        if not self.model:
            logger.error("ML Model not found! Exiting.")
            return
            
        self.running = True
        logger.info("Polling MBTM26 market data from TradeStation SIM...")
        
        try:
            while self.running:
                await self.poll_and_process()
                await asyncio.sleep(60.0)
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            if self.http:
                await self.http.aclose()

    async def poll_and_process(self):
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            since = self.last_ts or (datetime.now(timezone.utc) - timedelta(hours=10))
            url = f"{BARS_URL}&firstdate={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"

            response = await self.http.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} from TS API")
                return

            bars_data = response.json().get("Bars", [])
            if not bars_data:
                return

            now_utc = datetime.now(timezone.utc)
            new_bars = False
            for b in bars_data:
                ts = datetime.fromisoformat(b["TimeStamp"].replace('Z', '+00:00'))
                if ts <= now_utc and (not self.last_ts or ts > self.last_ts):
                    bar = {
                        'timestamp': ts,
                        'open': float(b["Open"]),
                        'high': float(b["High"]),
                        'low': float(b["Low"]),
                        'close': float(b["Close"]),
                        'volume': float(b["TotalVolume"])
                    }
                    self.bars.append(bar)
                    self.last_ts = ts
                    new_bars = True
            
            if len(self.bars) > 400:
                self.bars = self.bars[-400:]
                
            if new_bars and len(self.bars) >= 380:
                await self.process_logic()

        except Exception as e:
            logger.error(f"Polling error: {e}")

    async def process_logic(self):
        df = pd.DataFrame(self.bars).set_index('timestamp')
        
        # 1. Indicators
        df['prev_close'] = df['close'].shift(1)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['prev_close'])
        df['tr2'] = abs(df['low'] - df['prev_close'])
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(self.length).mean()
        
        df['h1_high'] = df['high'].rolling(360).max()
        df['h1_low'] = df['low'].rolling(360).min()
        df['sweep_bear'] = (df['high'] >= df['h1_high'].shift(1))
        df['sweep_bull'] = (df['low'] <= df['h1_low'].shift(1))
        df['recent_sweep_bear'] = df['sweep_bear'].astype(int).rolling(60).max() > 0
        df['recent_sweep_bull'] = df['sweep_bull'].astype(int).rolling(60).max() > 0
        
        df['soft_fvg_bear'] = (df['low'].shift(2) - df['high']) > (0.2 * df['atr'])
        df['soft_fvg_bull'] = (df['low'] - df['high'].shift(2)) > (0.2 * df['atr'])
        
        df['long_cond'] = df['recent_sweep_bull'] & df['soft_fvg_bull']
        df['short_cond'] = df['recent_sweep_bear'] & df['soft_fvg_bear']
        
        # Prevent consecutive signals in same direction (silence pandas downcasting future warning)
        df['long_cond'] = df['long_cond'] & (~df['long_cond'].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool))
        df['short_cond'] = df['short_cond'] & (~df['short_cond'].shift(1).fillna(value=False).infer_objects(copy=False).astype(bool))
        
        df['vol_sma'] = df['volume'].rolling(50).mean()
        df['rvol'] = df['volume'] / df['vol_sma'].replace(0, np.nan)
        df['rvol'] = df['rvol'].fillna(1.0)
        
        df['macro_ema'] = df['close'].ewm(span=200).mean()
        df['dist_macro_ema'] = (df['close'] - df['macro_ema']) / df['atr']
        
        df['ema'] = df['close'].ewm(span=self.length, adjust=False).mean()
        df['dist_ema'] = (df['close'] - df['ema']) / df['atr']
        
        df_et = df.index.tz_convert(self.et_tz)
        df['hour_et'] = df_et.hour
        df['minute_et'] = df_et.minute
        df['dow'] = df_et.dayofweek
        df['is_us_session'] = (((df['hour_et'] == 9) & (df['minute_et'] >= 30)) | 
                               ((df['hour_et'] >= 10) & (df['hour_et'] < 16)) | 
                               ((df['hour_et'] == 16) & (df['minute_et'] == 0))).astype(int)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in ['atr', 'rvol', 'dist_ema', 'dist_macro_ema']:
            df[col] = df[col].fillna(0)
            
        current_bar = df.iloc[-1]
        last_bar = df.iloc[-2]
        
        # 3. Entry Logic
        long_cond = last_bar['long_cond']
        short_cond = last_bar['short_cond']
        
        if long_cond or short_cond:
            direction = 1 if long_cond else 0
            dir_str = 'L' if long_cond else 'S'
            
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
                entry_price = round(current_bar['open'] * 2) / 2  # MBT tick size is 5 pts
                atr = last_bar['atr']
                
                # Round to nearest 5 index points (MBT tick size)
                sl_raw = entry_price - (atr * self.sl_mult) if direction == 1 else entry_price + (atr * self.sl_mult)
                tp_raw = entry_price + (atr * self.tp_mult) if direction == 1 else entry_price - (atr * self.tp_mult)
                
                sl = round(sl_raw / 5.0) * 5.0
                tp = round(tp_raw / 5.0) * 5.0
                
                logger.info(f"🔔 S26 COMBINE ENTRY {dir_str}: Price={entry_price:.2f} | P(Success)={proba:.3f} | SL={sl:.2f} | TP={tp:.2f}")
                await self.submit_bracket_order(dir_str, entry_price, tp, sl)

    async def submit_bracket_order(self, direction, entry, tp, sl):
        entry_action = "BUY" if direction == "L" else "SELL"
        exit_action = "SELL" if direction == "L" else "BUY"
        qty = str(self.contracts)

        payload = {
            "AccountID": SIM_ACCOUNT_ID,
            "Symbol": self.symbol,
            "Quantity": qty,
            "OrderType": "Limit",
            "LimitPrice": str(entry),
            "TradeAction": entry_action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
            "OSOs": [{
                "Type": "BRK",
                "Orders": [
                    {
                        "AccountID": SIM_ACCOUNT_ID,
                        "Symbol": self.symbol,
                        "Quantity": qty,
                        "OrderType": "Limit",
                        "TradeAction": exit_action,
                        "TimeInForce": {"Duration": "GTC"},
                        "LimitPrice": str(tp),
                    },
                    {
                        "AccountID": SIM_ACCOUNT_ID,
                        "Symbol": self.symbol,
                        "Quantity": qty,
                        "OrderType": "StopMarket",
                        "TradeAction": exit_action,
                        "TimeInForce": {"Duration": "GTC"},
                        "StopPrice": str(sl),
                    },
                ],
            }],
        }
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
            response = await self.http.post(SIM_ORDERS_URL, headers=headers, json=payload)
            if response.status_code not in (200, 201):
                logger.warning(f"⚠️ SIM bracket order failed HTTP {response.status_code}: {response.text[:200]}")
            else:
                data = response.json()
                logger.info(f"✅ COMBINE ORDER SUBMITTED: {data}")
        except Exception as e:
            logger.error(f"Submit error: {e}")

if __name__ == "__main__":
    trader = S26CombineTrader()
    asyncio.run(trader.run())
