#!/usr/bin/env python3
import argparse
import asyncio
import csv
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import httpx
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.execution.kraken.auth.api_key import KrakenFuturesAuth
from src.execution.kraken.exceptions import KrakenAPIError, KrakenAuthError, KrakenOrderError
from src.execution.kraken.market_data.history import KrakenHistoryClient
from src.execution.kraken.orders.submission import KrakenOrdersClient

# Logging setup
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 's27_squeeze_streaming.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SYMBOL = "PF_XBTUSD"


class S27SqueezeTrader:
    def __init__(self, live: bool = False, contracts: int = 1) -> None:
        self.symbol    = SYMBOL
        self.live      = live
        self.contracts = contracts
        self._http     = httpx.AsyncClient(timeout=15.0)
        self.client    = KrakenHistoryClient(self._http)

        self.model_path = Path(__file__).parent.parent.parent / "models/vol_squeeze_ml_model.pkl"
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None

        # S27 Parameters
        self.length   = 20
        self.bb_mult  = 2.0
        self.kc_mult  = 1.5
        self.sl_mult  = 2.0
        self.tp_mult  = 4.0
        self.ml_thresh = 0.56
        self.max_hold  = 60

        self.bars      = []
        self.active_trade: Optional[dict] = None
        self.last_ts   = None

        self.trade_log_path = log_dir / "s27_squeeze_trade_log.csv"

        # Order execution (None → paper simulation only)
        self._orders: Optional[KrakenOrdersClient] = None
        self._entry_order_id: Optional[str]  = None
        self._pending_fill: bool = False
        self._pending_bars: int  = 0
        self._stale_bars: int    = self.max_hold // 2  # cancel pending after this many bars

        self._init_orders_client()

    def _init_orders_client(self) -> None:
        api_key    = os.environ.get("KRAKEN_FUTURES_API_KEY", "")
        api_secret = os.environ.get("KRAKEN_FUTURES_API_SECRET", "")
        if not api_key or not api_secret:
            logger.info("No Kraken credentials found — running in paper-simulation mode")
            return
        try:
            auth = KrakenFuturesAuth()
            self._orders = KrakenOrdersClient(auth, self._http, live=self.live)
            mode = "LIVE" if self.live else "DEMO"
            logger.info(f"Kraken order client initialised ({mode}), contracts={self.contracts}")
        except KrakenAuthError as exc:
            logger.warning(f"Kraken credentials invalid — paper-simulation mode: {exc}")

    async def run(self) -> None:
        logger.info(f"Starting S27 Volatility Squeeze Trader | symbol={self.symbol} | live={self.live} | contracts={self.contracts}")
        if not self.model:
            logger.error("ML model not found at %s — exiting.", self.model_path)
            await self._http.aclose()
            return

        try:
            while True:
                try:
                    count  = 150 if not self.last_ts else 2
                    kb_bars = await self.client.fetch_bars(self.symbol, interval="1m", count=count)

                    if not kb_bars:
                        await asyncio.sleep(10)
                        continue

                    for kb in kb_bars:
                        if self.last_ts and kb.time <= self.last_ts:
                            continue
                        self.bars.append({
                            'timestamp': kb.time,
                            'open':   float(kb.open),
                            'high':   float(kb.high),
                            'low':    float(kb.low),
                            'close':  float(kb.close),
                            'volume': float(kb.volume),
                        })
                        self.last_ts = kb.time

                    if len(self.bars) > 150:
                        self.bars = self.bars[-150:]

                    if len(self.bars) >= 50:
                        # Check fill status before processing new bar
                        await self._check_pending_fill()
                        await self.process_latest()

                except Exception as exc:
                    logger.error(f"Error in poll loop: {exc}")

                await asyncio.sleep(60)
        finally:
            await self._http.aclose()

    async def _check_pending_fill(self) -> None:
        """If an entry limit is outstanding, poll Kraken to see if it filled."""
        if not self._pending_fill or not self._entry_order_id or not self._orders:
            return

        self._pending_bars += 1

        # Stale entry — cancel and abandon
        if self._pending_bars >= self._stale_bars:
            logger.warning(
                "Entry order %s unfilled after %d bars — cancelling",
                self._entry_order_id, self._pending_bars,
            )
            try:
                await self._orders.cancel_order(self._entry_order_id)
            except Exception as exc:
                logger.warning("Cancel failed for %s: %s", self._entry_order_id, exc)
            self._entry_order_id = None
            self._pending_fill   = False
            self._pending_bars   = 0
            self.active_trade    = None
            return

        try:
            open_orders = await self._orders.get_open_orders(self.symbol)
        except Exception as exc:
            logger.warning("get_open_orders failed: %s", exc)
            return

        # Kraken REST may use snake_case "order_id" or camelCase "orderId" — check both
        still_open = any(
            (o.get("order_id") or o.get("orderId")) == self._entry_order_id
            for o in open_orders
        )
        if not still_open:
            logger.info("✅ Entry order %s confirmed filled", self._entry_order_id)
            self._pending_fill = False
            self._pending_bars = 0

    async def _place_market_close(self, direction: str) -> None:
        """Send a market order to close an open position."""
        if not self._orders:
            return
        close_side = "sell" if direction == "L" else "buy"
        try:
            oid = await self._orders.place_order(
                self.symbol, close_side, "mkt", size=float(self.contracts)
            )
            logger.info("🏁 Market close placed: %s %dx → id=%s", close_side, self.contracts, oid)
        except KrakenOrderError as exc:
            logger.warning("Market close order failed: %s", exc)
        except KrakenAPIError as exc:
            logger.warning("Market close API error: %s", exc)

    async def process_latest(self) -> None:
        df = pd.DataFrame(self.bars).set_index('timestamp')

        # 1. Indicators
        df['prev_close'] = df['close'].shift(1)
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['prev_close'])
        df['tr2'] = abs(df['low']  - df['prev_close'])
        df['tr']  = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(self.length).mean()

        df['sma']      = df['close'].rolling(self.length).mean()
        df['std']      = df['close'].rolling(self.length).std()
        df['bb_upper'] = df['sma'] + (self.bb_mult * df['std'])
        df['bb_lower'] = df['sma'] - (self.bb_mult * df['std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

        df['ema']      = df['close'].ewm(span=self.length, adjust=False).mean()
        df['kc_upper'] = df['ema'] + (self.kc_mult * df['atr'])
        df['kc_lower'] = df['ema'] - (self.kc_mult * df['atr'])

        df['squeeze_on']    = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        df['recent_squeeze'] = df['squeeze_on'].astype(int).rolling(window=5).max() > 0

        df['upper_band'] = df['high'].rolling(self.length).max()
        df['lower_band'] = df['low'].rolling(self.length).min()

        # ML features
        df['volume_15m'] = df['volume'].rolling(15).sum()
        df['dist_ema']   = (df['close'] - df['ema']) / df['atr']

        current_bar = df.iloc[-1]
        last_bar    = df.iloc[-2]
        prev_bar    = df.iloc[-3]

        # 2. Manage active trade
        if self.active_trade:
            # Skip exit checks while entry is still pending fill
            if self._pending_fill:
                return

            self.active_trade['hold_time'] += 1
            t = self.active_trade

            exit_reason = None
            exit_price  = 0.0

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
                exit_reason, exit_price = 'TIME_STOP', float(current_bar['close'])

            if exit_reason:
                pnl = (exit_price - t['entry']) if t['dir'] == 'L' else (t['entry'] - exit_price)
                pnl *= self.contracts
                logger.info(
                    "🏁 Trade closed [%s]: PNL=$%.2f | hold=%dm", exit_reason, pnl, t['hold_time']
                )
                await self._place_market_close(t['dir'])
                self.log_trade(t, exit_price, exit_reason, pnl)
                self.active_trade    = None
                self._entry_order_id = None
                return

        # 3. Entry logic (only if flat and no pending order)
        if not self.active_trade and not self._pending_fill:
            long_cond  = bool(last_bar['close'] > prev_bar['upper_band']) and bool(last_bar['recent_squeeze'])
            short_cond = bool(last_bar['close'] < prev_bar['lower_band']) and bool(last_bar['recent_squeeze'])

            if long_cond or short_cond:
                direction = 1 if long_cond else 0
                dir_str   = 'L' if long_cond else 'S'

                features = pd.DataFrame([{
                    'dir':      direction,
                    'atr':      last_bar['atr'],
                    'vol_15m':  last_bar['volume_15m'],
                    'dist_ema': last_bar['dist_ema'],
                    'hour':     df.index[-2].hour,
                    'dow':      df.index[-2].dayofweek,
                    'bb_width': last_bar['bb_width'],
                }])

                proba = self.model.predict_proba(features)[0, 1]

                if proba >= self.ml_thresh:
                    entry_price = float(current_bar['open'])
                    atr = float(last_bar['atr'])
                    sl = entry_price - (atr * self.sl_mult) if direction == 1 else entry_price + (atr * self.sl_mult)
                    tp = entry_price + (atr * self.tp_mult) if direction == 1 else entry_price - (atr * self.tp_mult)

                    self.active_trade = {
                        'dir':       dir_str,
                        'entry':     entry_price,
                        'sl':        sl,
                        'tp':        tp,
                        'atr':       atr,
                        'proba':     proba,
                        'hold_time': 0,
                        'ts':        df.index[-1],
                    }
                    logger.info(
                        "🔔 S27 ENTRY %s: price=%.2f | P=%.3f | SL=%.2f | TP=%.2f",
                        dir_str, entry_price, proba, sl, tp,
                    )

                    if self._orders:
                        side = "buy" if direction == 1 else "sell"
                        try:
                            oid = await self._orders.place_order(
                                self.symbol, side, "lmt",
                                size=float(self.contracts),
                                limit_price=float(round(entry_price)),
                            )
                            self._entry_order_id = oid
                            self._pending_fill   = True
                            self._pending_bars   = 0
                            logger.info("↗ Limit entry placed: %s", oid)
                        except (KrakenOrderError, KrakenAPIError) as exc:
                            logger.warning("Entry order failed — staying flat: %s", exc)
                            self.active_trade = None

    def log_trade(self, t: dict, exit_price: float, reason: str, pnl: float) -> None:
        write_header = not self.trade_log_path.exists()
        with open(self.trade_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['entry_time', 'direction', 'entry_price', 'exit_price', 'reason', 'pnl', 'ml_proba', 'contracts'])
            writer.writerow([
                t['ts'].isoformat(), t['dir'],
                round(t['entry'], 2), round(exit_price, 2),
                reason, round(pnl, 2), round(t['proba'], 3),
                self.contracts,
            ])


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="S27 Volatility Squeeze live/demo trader")
    parser.add_argument("--live",      action="store_true", help="Route to live Kraken Futures (default: demo)")
    parser.add_argument("--contracts", type=int, default=1, help="Contract size (default: 1)")
    args = parser.parse_args()

    trader = S27SqueezeTrader(live=args.live, contracts=args.contracts)
    asyncio.run(trader.run())
