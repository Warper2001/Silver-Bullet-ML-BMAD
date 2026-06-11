#!/usr/bin/env python3
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import httpx
import sys
import json
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.auth_v3 import TradeStationAuthV3

# Logging setup
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', handlers=[
    logging.FileHandler(log_dir / 'stat_arb_bot.log'),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

SIM_ACCOUNT_ID = "SIM2248559M"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/brokerage/orders"
# Using 5-minute bars, pulling last 150 bars to compute 60-bar rolling metrics
BARS_URL = "https://api.tradestation.com/v3/marketdata/barcharts/{symbol}?interval=5&unit=Minute&barsback=150"

SYM_MNQ = "MNQM26"
SYM_MES = "MESM26"
QTY_MNQ = "2"
QTY_MES = "3"

Z_ENTRY = 2.5
Z_EXIT = 0.0

STATE_FILE = Path(__file__).parent.parent.parent / "data/state/stat_arb_state.json"

class StatArbTrader:
    def __init__(self):
        self.running = False
        self.auth = None
        self.http = None
        self.active_trade = self._load_state()

    def _load_state(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return None

    def _save_state(self):
        os.makedirs(STATE_FILE.parent, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.active_trade, f)

    async def initialize(self):
        logger.info("=" * 70)
        logger.info(f"STAT ARB COMBINE TRADER - {SYM_MNQ} vs {SYM_MES}")
        logger.info(f"Z_ENTRY: {Z_ENTRY} | Z_EXIT: {Z_EXIT}")
        logger.info("=" * 70)
        
        self.auth = TradeStationAuthV3.from_file('.access_token')
        await self.auth.authenticate()
        await self.auth.start_auto_refresh()
        self.http = httpx.AsyncClient(timeout=30.0)

    async def fetch_bars(self, symbol, token):
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        url = BARS_URL.format(symbol=symbol)
        resp = await self.http.get(url, headers=headers)
        if resp.status_code == 200:
            bars = resp.json().get("Bars", [])
            data = []
            for b in bars:
                ts = datetime.fromisoformat(b["TimeStamp"].replace('Z', '+00:00'))
                data.append({"timestamp": ts, "close": float(b["Close"])})
            df = pd.DataFrame(data).set_index("timestamp")
            # Drop the current forming bar to avoid repainting
            return df.iloc[:-1]
        else:
            logger.warning(f"Failed to fetch {symbol}: {resp.status_code}")
            return None

    async def run(self):
        await self.initialize()
        self.running = True
        logger.info("Polling 5-minute market data from TradeStation SIM...")
        
        try:
            while self.running:
                await self.poll_and_process()
                await asyncio.sleep(60.0) # Check every minute
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            if self.http:
                await self.http.aclose()

    async def poll_and_process(self):
        try:
            token = await self.auth.authenticate()
            
            df_mnq = await self.fetch_bars(SYM_MNQ, token)
            df_mes = await self.fetch_bars(SYM_MES, token)
            
            if df_mnq is None or df_mes is None or df_mnq.empty or df_mes.empty:
                return
                
            df_mnq.columns = ["MNQ"]
            df_mes.columns = ["ES"]
            
            df = pd.concat([df_mnq, df_mes], axis=1, join='inner')
            if len(df) < 70:
                logger.warning("Not enough overlapping bars to compute metrics.")
                return
                
            # Compute Stat Arb Metrics
            window = 60
            df['MNQ_ret'] = df['MNQ'].pct_change()
            df['ES_ret'] = df['ES'].pct_change()
            
            cov = df['MNQ_ret'].rolling(window).cov(df['ES_ret'])
            var = df['ES_ret'].rolling(window).var()
            df['beta'] = cov / var
            
            df['spread'] = df['MNQ_ret'] - (df['beta'] * df['ES_ret'])
            df['spread_mean'] = df['spread'].rolling(window).mean()
            df['spread_std'] = df['spread'].rolling(window).std()
            df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            if pd.isna(current['z_score']) or pd.isna(prev['z_score']):
                return
                
            z = current['z_score']
            pz = prev['z_score']
            
            # Log periodic status
            if datetime.now().minute % 15 == 0:
                logger.info(f"Market Status: Z-Score = {z:.2f} | Beta = {current['beta']:.2f}")

            if self.active_trade:
                # Check exit
                if self.active_trade['dir'] == 'SHORT_MNQ' and z <= Z_EXIT:
                    logger.info(f"🏁 Z-Score reverted to {z:.2f}. EXITING SHORT_MNQ / LONG_MES")
                    await self.execute_legs("BUY", SYM_MNQ, QTY_MNQ, "SELL", SYM_MES, QTY_MES)
                    self.active_trade = None
                    self._save_state()
                    
                elif self.active_trade['dir'] == 'LONG_MNQ' and z >= Z_EXIT:
                    logger.info(f"🏁 Z-Score reverted to {z:.2f}. EXITING LONG_MNQ / SHORT_MES")
                    await self.execute_legs("SELL", SYM_MNQ, QTY_MNQ, "BUY", SYM_MES, QTY_MES)
                    self.active_trade = None
                    self._save_state()
            else:
                # Check entry
                if pz < Z_ENTRY and z >= Z_ENTRY:
                    logger.info(f"🚨 Z-Score hit {z:.2f}! ENTERING: SHORT_MNQ / LONG_MES")
                    await self.execute_legs("SELL", SYM_MNQ, QTY_MNQ, "BUY", SYM_MES, QTY_MES)
                    self.active_trade = {'dir': 'SHORT_MNQ', 'z_entry': float(z), 'time': str(current.name)}
                    self._save_state()
                    
                elif pz > -Z_ENTRY and z <= -Z_ENTRY:
                    logger.info(f"🚨 Z-Score hit {z:.2f}! ENTERING: LONG_MNQ / SHORT_MES")
                    await self.execute_legs("BUY", SYM_MNQ, QTY_MNQ, "SELL", SYM_MES, QTY_MES)
                    self.active_trade = {'dir': 'LONG_MNQ', 'z_entry': float(z), 'time': str(current.name)}
                    self._save_state()

        except Exception as e:
            logger.error(f"Processing error: {e}")

    async def execute_legs(self, act1, sym1, qty1, act2, sym2, qty2):
        # Fire both legs simultaneously
        tasks = [
            self.submit_market_order(act1, sym1, qty1),
            self.submit_market_order(act2, sym2, qty2)
        ]
        await asyncio.gather(*tasks)

    async def submit_market_order(self, action, symbol, qty):
        payload = {
            "AccountID": SIM_ACCOUNT_ID,
            "Symbol": symbol,
            "Quantity": qty,
            "OrderType": "Market",
            "TradeAction": action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent"
        }
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
            response = await self.http.post(SIM_ORDERS_URL, headers=headers, json=payload)
            if response.status_code in (200, 201):
                logger.info(f"✅ Executed: {action} {qty} {symbol}")
            else:
                logger.error(f"❌ Failed to execute {action} {symbol}: {response.text}")
        except Exception as e:
            logger.error(f"Execution error on {symbol}: {e}")

if __name__ == "__main__":
    trader = StatArbTrader()
    asyncio.run(trader.run())
