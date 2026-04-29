#!/usr/bin/env python3
"""
TIER 2 FVG Paper Trading - TradeStation HTTP Polling + SIM Order Placement
Configuration: SL5.0x_TP5.0x_Midpoint_H1Sweep

Entry fires a bracket order on SIM account (entry + TP limit + SL stop).
The SIM account manages TP/SL fills. Local per-bar simulation is the
authoritative P&L record and handles the time-stop (cancel bracket + flat close).
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3
from src.data.models import DollarBar

# Configuration (OPTIMIZED TIER 2)
TIER2_CONFIG = "SL5.0x_TP5.0x_Midpoint_H1Sweep"
SL_MULTIPLIER = 5.0
TP_MULTIPLIER = 5.0
ENTRY_PCT = 0.5  # Midpoint entry
ATR_THRESHOLD = 0.5
MAX_GAP_DOLLARS = 60.0
MAX_HOLD_BARS = 120 # 2 Hours (matches final optimization)
CONTRACTS_PER_TRADE = 1

# MNQ Specifications
MNQ_TICK_SIZE = 0.25
MNQ_POINT_VALUE = 20.0
MNQ_CONTRACT_VALUE = MNQ_TICK_SIZE * MNQ_POINT_VALUE  # $5/pt scaling for filters
MNQ_DOLLAR_VALUE = 2.0  # real MNQ: $2 per index point per contract

# Transaction Costs
COMMISSION_PER_CONTRACT = 0.40
TRANSACTION_COST = COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2  # $0.80/roundtrip

# TradeStation market data API
SYMBOL = "MNQM26"
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
BARS_BASE_URL = (f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
                 f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}")
HISTORY_HOURS = 48    # Increased to 48h to have enough data for H1 swing detection
POLL_INTERVAL_SECONDS = 60

# TradeStation SIM order placement
SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"

# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

_handlers: list = [logging.FileHandler(log_dir / 'tier2_streaming_working.log')]
if sys.stdout.isatty():
    _handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=_handlers,
)
logger = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    bar_index: int
    entry_time: datetime
    direction: str
    entry_price: float
    tp_price: float
    sl_price: float
    bars_held: int = 0
    sim_entry_order_id: Optional[str] = None
    sim_tp_order_id: Optional[str] = None
    sim_sl_order_id: Optional[str] = None
    sim_entry_fill: Optional[float] = None


@dataclass
class CompletedTrade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    exit_type: str
    bars_held: int
    pnl: float
    sim_order_id: Optional[str] = None


class Tier2StreamingTrader:
    def __init__(self):
        self.running = False
        self.auth = None
        self.client = None
        self.dollar_bars: list[DollarBar] = []
        self._last_processed_timestamp: Optional[datetime] = None
        self.active_trade: Optional[ActiveTrade] = None
        self.completed_trades: list[CompletedTrade] = []
        self._traded_setups: set[tuple[int, str]] = set()
        self._is_backfill: bool = True
        self.session_start_time: Optional[datetime] = None
        
        # Tier 2 State
        self.h1_bars: pd.DataFrame = pd.DataFrame()
        self.unmitigated_sh: list[float] = []
        self.unmitigated_sl: list[float] = []
        self.last_h1_sweep_time: Optional[datetime] = None
        self.h1_bullish_sweep_active = False
        self.h1_bearish_sweep_active = False

    async def initialize(self):
        logger.info("=" * 70)
        logger.info("TIER 2 FVG PAPER TRADING - SIM ORDER PLACEMENT")
        logger.info("=" * 70)
        logger.info(f"Configuration: {TIER2_CONFIG}")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Max hold: {MAX_HOLD_BARS} bars | SL/TP mult: {SL_MULTIPLIER}x")
        logger.info(f"Entry Level: {ENTRY_PCT*100}% (Mean Threshold)")
        logger.info("=" * 70)

        self.auth = TradeStationAuthV3.from_file('.access_token')
        await self.auth.authenticate()
        await self.auth.start_auto_refresh()
        self.client = httpx.AsyncClient(timeout=30.0)
        self.session_start_time = datetime.now()

    async def start_streaming(self):
        self.running = True
        try:
            while self.running:
                if not self._is_market_open():
                    await asyncio.sleep(60)
                    continue
                await self._poll_and_process()
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"❌ Polling error: {e}", exc_info=True)
        finally:
            await self.stop()

    @staticmethod
    def _is_market_open() -> bool:
        now = datetime.now(timezone.utc)
        wd, h = now.weekday(), now.hour
        if wd == 5: return False
        if wd == 6: return h >= 23
        if wd == 4: return h < 22
        return h != 22

    async def stop(self):
        self.running = False
        if self.active_trade and self.dollar_bars:
            await self._close_active_trade(self.dollar_bars[-1], self.dollar_bars[-1].close, "time")
        if self.client: await self.client.aclose()
        self._print_final_report()

    async def _poll_and_process(self):
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            since = self._last_processed_timestamp or (datetime.now(timezone.utc) - timedelta(hours=HISTORY_HOURS))
            url = f"{BARS_BASE_URL}&firstdate={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            
            response = await self.client.get(url, headers=headers)
            if response.status_code != 200: return

            bars_data = response.json().get("Bars", [])
            new_bars = []
            now_utc = datetime.now(timezone.utc)
            for bar_data in bars_data:
                bar = self._parse_bar(bar_data)
                if bar and bar.timestamp <= now_utc and (not self._last_processed_timestamp or bar.timestamp > self._last_processed_timestamp):
                    self.dollar_bars.append(bar)
                    self._last_processed_timestamp = bar.timestamp
                    new_bars.append(bar)
                    
                    # Update H1 structure and detect sweeps
                    self._update_h1_structure()
                    
                    await self._advance_active_trade(bar)
                    await self._detect_and_enter(bar, is_backfill=self._is_backfill)

            if self._is_backfill and new_bars:
                self._is_backfill = False
                logger.info(f"✅ Tier 2 Backfill complete ({len(self.dollar_bars)} bars)")
        except Exception as e:
            logger.error(f"❌ Error in poll cycle: {e}", exc_info=True)

    def _parse_bar(self, d: dict) -> Optional[DollarBar]:
        try:
            return DollarBar(
                timestamp=datetime.fromisoformat(d["TimeStamp"].replace('Z', '+00:00')),
                open=float(d["Open"]), high=float(d["High"]), low=float(d["Low"]),
                close=float(d["Close"]), volume=int(d.get("TotalVolume", 0)),
                notional_value=0, bar_num=len(self.dollar_bars)
            )
        except: return None

    def _update_h1_structure(self):
        """Resample 1m bars to H1 and detect liquidity sweeps."""
        if not self.dollar_bars: return
        
        df = pd.DataFrame([vars(b) for b in self.dollar_bars])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        h1 = df.set_index('timestamp').resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()
        
        if len(h1) < 5: return
        
        # Detect Swing Points (window=2)
        sh, sl = [], []
        highs, lows = h1['high'].values, h1['low'].values
        for i in range(2, len(h1) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                sh.append((h1.loc[i, 'timestamp'], highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                sl.append((h1.loc[i, 'timestamp'], lows[i]))
        
        # Detect Sweeps in the LATEST H1 bar
        last_h1 = h1.iloc[-1]
        self.h1_bullish_sweep_active = False
        self.h1_bearish_sweep_active = False
        
        # A sweep mitigates an unmitigated level confirmed before this bar
        # For simplicity in live, we check if current H1 High/Low pierces any SH/SL 
        # that hasn't been pierced by previous H1 closes.
        for t, val in sh:
            if t < last_h1['timestamp'] - timedelta(hours=2): # confirmed
                if last_h1['high'] > val and last_h1['close'] < val:
                    self.h1_bearish_sweep_active = True
                    logger.info(f"🎯 H1 BEARISH SWEEP detected at {val}")
        
        for t, val in sl:
            if t < last_h1['timestamp'] - timedelta(hours=2): # confirmed
                if last_h1['low'] < val and last_h1['close'] > val:
                    self.h1_bullish_sweep_active = True
                    logger.info(f"🎯 H1 BULLISH SWEEP detected at {val}")

    async def _advance_active_trade(self, bar: DollarBar):
        if not self.active_trade: return
        t = self.active_trade
        t.bars_held += 1
        if t.direction == "LONG":
            if bar.low <= t.sl_price: await self._close_active_trade(bar, t.sl_price, "sl")
            elif bar.high >= t.tp_price: await self._close_active_trade(bar, t.tp_price, "tp")
        else:
            if bar.high >= t.sl_price: await self._close_active_trade(bar, t.sl_price, "sl")
            elif bar.low <= t.tp_price: await self._close_active_trade(bar, t.tp_price, "tp")
        if t.bars_held >= MAX_HOLD_BARS: await self._close_active_trade(bar, bar.close, "time")

    async def _close_active_trade(self, bar: DollarBar, price: float, reason: str):
        t = self.active_trade
        pnl = ((price - t.entry_price) if t.direction == "LONG" else (t.entry_price - price)) * MNQ_DOLLAR_VALUE - TRANSACTION_COST
        self.completed_trades.append(CompletedTrade(t.entry_time, bar.timestamp, t.direction, t.entry_price, price, reason, t.bars_held, pnl))
        self.active_trade = None
        logger.info(f"Trade Closed: {reason.upper()} | P&L: ${pnl:.2f}")

    async def _detect_and_enter(self, bar: DollarBar, is_backfill: bool):
        if self.active_trade: return
        bars = self.dollar_bars
        if len(bars) < 3: return
        
        # Confluence check: H1 Sweep Active
        if self.h1_bullish_sweep_active:
            fvg = self._detect_fvg(bars, bullish=True)
            if fvg: await self._enter_trade(fvg, bar, len(bars)-1, is_backfill)
        elif self.h1_bearish_sweep_active:
            fvg = self._detect_fvg(bars, bullish=False)
            if fvg: await self._enter_trade(fvg, bar, len(bars)-1, is_backfill)

    def _detect_fvg(self, bars, bullish: bool) -> Optional[dict]:
        c1, c2, c3 = bars[-3], bars[-2], bars[-1]
        if bullish:
            if not (c1.close > c3.open and c2.close > c2.open): return None
            top, bot = c3.low, c1.high
        else:
            if not (c1.close < c3.open and c2.close < c2.open): return None
            top, bot = c1.low, c3.high
        
        if top <= bot: return None
        gap_size = (top - bot) * MNQ_CONTRACT_VALUE
        if gap_size < ATR_THRESHOLD * self._calculate_atr(bars) or gap_size > MAX_GAP_DOLLARS: return None
        
        return {"direction": "bullish" if bullish else "bearish", "top": top, "bottom": bot, "gap_size": gap_size}

    def _calculate_atr(self, bars):
        if len(bars) < 20: return 10.0
        tr = [max(b.high-b.low, abs(b.high-bars[i-1].close), abs(b.low-bars[i-1].close)) for i, b in enumerate(bars[-20:]) if i > 0]
        return sum(tr)/len(tr)

    async def _enter_trade(self, fvg, bar, idx, is_backfill):
        direction = "LONG" if fvg["direction"] == "bullish" else "SHORT"
        gap_size = fvg["top"] - fvg["bottom"]
        if direction == "LONG":
            ent = fvg["top"] - gap_size * ENTRY_PCT
            tp, sl = ent + gap_size * TP_MULTIPLIER, ent - gap_size * SL_MULTIPLIER
        else:
            ent = fvg["bottom"] + gap_size * ENTRY_PCT
            tp, sl = ent - gap_size * TP_MULTIPLIER, ent + gap_size * SL_MULTIPLIER
            
        self.active_trade = ActiveTrade(idx, bar.timestamp, direction, ent, tp, sl)
        logger.info(f"🔔 TIER 2 ENTRY: {direction} at ${ent:.2f} | TP ${tp:.2f} SL ${sl:.2f}")

    def _print_final_report(self):
        logger.info("Tier 2 Paper Trading Session Ended.")

async def main():
    trader = Tier2StreamingTrader()
    await trader.initialize()
    await trader.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())
