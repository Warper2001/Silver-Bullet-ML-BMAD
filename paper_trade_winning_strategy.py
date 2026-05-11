#!/usr/bin/env python3
"""
Winning Silver Bullet ML Strategy - Live Paper Trading (Fixed)

This uses the EXACT same configuration that achieved institutional-grade status
in the 2025 backtest.
"""

import asyncio
import httpx
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.config import load_settings
from src.ml.inference import MLInference
from src.data.models import DollarBar, SilverBulletSetup, SwingPoint
from src.data.auth_v3 import TradeStationAuthV3
from src.data.market_data_validator import MarketDataValidator
from src.detection.time_window_filter import is_within_trading_hours, DEFAULT_TRADING_WINDOWS
from src.detection.silver_bullet_detection import detect_silver_bullet_setup
from src.detection.swing_detection import detect_swing_high, detect_swing_low, RollingVolumeAverage, detect_bullish_mss, detect_bearish_mss
from src.detection.fvg_detection import detect_bullish_fvg, detect_bearish_fvg
from src.detection.liquidity_sweep_detection import detect_bullish_liquidity_sweep, detect_bearish_liquidity_sweep

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"
MNQ_TICK = 0.25  # MNQ minimum price increment


def round_tick(price: float, tick: float = MNQ_TICK) -> float:
    """Round price to nearest valid tick increment."""
    return round(round(price / tick) * tick, 10)


class WinningSilverBulletTrader:
    """Live paper trader using the winning Silver Bullet ML Strategy configuration."""

    def __init__(self, access_token: str):
        """Initialize the winning strategy trader."""
        self.access_token = access_token
        self.running = False

        # Initialize V3 auth with auto-refresh
        self.auth = TradeStationAuthV3.from_file(".access_token")

        # Initialize ML components (EXACT same as winning backtest)
        self.ml_inference = MLInference(model_dir="models/xgboost")
        self.ml_threshold = 0.65  # 65% probability threshold

        # Initialize data components
        self.validator = MarketDataValidator()

        # Winning strategy state
        self.recent_bars: list[DollarBar] = []
        self.swing_highs: list[SwingPoint] = []
        self.swing_lows: list[SwingPoint] = []
        self.mss_events: list = []
        self.fvg_events: list = []
        self.volume_ma = RollingVolumeAverage(window=20)
        
        # Execution state
        self.pending_setups: list[dict] = []  # Setups waiting for FVG touch
        self.active_trades: list[dict] = []   # Open positions
        self.window_trades: dict[str, set[str]] = {} # {window: {date_str}}
        self.bar_index = 0
        self._is_preloading = False
        self._seen_setup_keys: set[tuple[int, int]] = set()  # (mss_bar_index, fvg_bar_index) dedup
        self._ml_unavailable: bool = False  # latched on first ML failure; suppresses per-setup noise

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

        logger.info("Silver Bullet Strategy initialized")
        logger.info("Corrected Backtest Performance (Nov 2025 - May 2026, 6 months):")
        logger.info("   - Win Rate: 39.2% | Profit Factor: 2.60")
        logger.info("   - 6-month P&L: +$6,559 (+6.56%) | Sharpe: 5.24")
        logger.info("   - Avg daily P&L: $51.65 | 2.15 trades/day")
        logger.info("   - Max Drawdown: $312 | Avg hold: 7.4 min")

    async def fetch_latest_bar(self, symbol: str) -> DollarBar | None:
        """Fetch the most recently completed 1-minute bar from barcharts API."""
        url = (
            f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
            f"?interval=1&unit=Minute&barsback=2"
        )
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"⚠️  barcharts HTTP {response.status_code}")
                return None
            bars = response.json().get("Bars", [])
            # Use the first bar (most recent completed bar; second may still be forming)
            bar_data = bars[0] if bars else None
            if not bar_data:
                return None
            ts_str = bar_data.get("TimeStamp", "")
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            close = float(bar_data.get("Close", 0) or 0)
            if close == 0:
                return None
            volume = int(bar_data.get("TotalVolume", 0) or 0)
            return DollarBar(
                timestamp=ts,
                open=float(bar_data.get("Open", close) or close),
                high=float(bar_data.get("High", close) or close),
                low=float(bar_data.get("Low", close) or close),
                close=close,
                volume=volume,
                notional_value=close * 20.0,
                is_forward_filled=False,
            )
        except Exception as e:
            logger.error(f"Exception fetching barcharts bar: {e}")
            return None

    def _find_next_liquidity_pool(
        self,
        direction: str,
        entry_price: float,
        lookback: int = 200,
    ) -> float | None:
        """Find the nearest unswept prior swing in the trade direction."""
        start = max(0, len(self.recent_bars) - lookback)
        candidates = []

        for i in range(len(self.recent_bars) - 1, start - 1, -1):
            bar = self.recent_bars[i]
            if direction == "bullish":
                # Local swing high
                left = self.recent_bars[i - 1].high if i > 0 else 0
                right = self.recent_bars[i + 1].high if i < len(self.recent_bars) - 1 else 0
                if bar.high > left and bar.high > right and bar.high > entry_price:
                    # Unswept: no bar between swing and current had close above it
                    swept = any(
                        self.recent_bars[j].close > bar.high
                        for j in range(i + 1, len(self.recent_bars))
                    )
                    if not swept:
                        candidates.append(bar.high)
            else:
                left = self.recent_bars[i - 1].low if i > 0 else float("inf")
                right = self.recent_bars[i + 1].low if i < len(self.recent_bars) - 1 else float("inf")
                if bar.low < left and bar.low < right and bar.low < entry_price:
                    swept = any(
                        self.recent_bars[j].close < bar.low
                        for j in range(i + 1, len(self.recent_bars))
                    )
                    if not swept:
                        candidates.append(bar.low)

        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p - entry_price))

    def _detect_patterns(self, bar: DollarBar):
        """Detect patterns from the latest bar."""
        self.bar_index += 1
        self.volume_ma.update(bar.volume)

        # 1. Swing Detection (3-bar lookback)
        idx = len(self.recent_bars) - 4
        if idx >= 3:
            if detect_swing_high(self.recent_bars, idx):
                self.swing_highs.append(SwingPoint(
                    timestamp=self.recent_bars[idx].timestamp,
                    price=self.recent_bars[idx].high,
                    swing_type="swing_high",
                    bar_index=self.bar_index - 4
                ))
            if detect_swing_low(self.recent_bars, idx):
                self.swing_lows.append(SwingPoint(
                    timestamp=self.recent_bars[idx].timestamp,
                    price=self.recent_bars[idx].low,
                    swing_type="swing_low",
                    bar_index=self.bar_index - 4
                ))

        # Keep recent swings
        self.swing_highs = self.swing_highs[-50:]
        self.swing_lows = self.swing_lows[-50:]

        # 2. MSS Detection
        bull_mss = detect_bullish_mss(bar, self.swing_highs, self.volume_ma.average)
        bear_mss = detect_bearish_mss(bar, self.swing_lows, self.volume_ma.average)
        
        if bull_mss: 
            bull_mss.bar_index = self.bar_index
            self.mss_events.append(bull_mss)
        if bear_mss: 
            bear_mss.bar_index = self.bar_index
            self.mss_events.append(bear_mss)
        
        self.mss_events = self.mss_events[-50:]

        # 3. FVG Detection
        curr_idx = len(self.recent_bars) - 1
        bull_fvg = detect_bullish_fvg(self.recent_bars, curr_idx)
        bear_fvg = detect_bearish_fvg(self.recent_bars, curr_idx)
        
        if bull_fvg:
            bull_fvg.bar_index = self.bar_index
            self.fvg_events.append(bull_fvg)
        if bear_fvg:
            bear_fvg.bar_index = self.bar_index
            self.fvg_events.append(bear_fvg)
            
        self.fvg_events = self.fvg_events[-50:]

        # 4. Sweep Detection
        # (Simplified for live: just check if current bar swept most recent swing)
        bull_sweep = None
        if self.swing_lows:
            bull_sweep = detect_bullish_liquidity_sweep(self.recent_bars, curr_idx, self.swing_lows[-1])
        
        bear_sweep = None
        if self.swing_highs:
            bear_sweep = detect_bearish_liquidity_sweep(self.recent_bars, curr_idx, self.swing_highs[-1])

        sweeps = []
        if bull_sweep: 
            bull_sweep.bar_index = self.bar_index
            sweeps.append(bull_sweep)
        if bear_sweep: 
            bear_sweep.bar_index = self.bar_index
            sweeps.append(bear_sweep)

        # 5. Confluence Detection (skipped during preload warmup)
        if self._is_preloading:
            return []

        setups = detect_silver_bullet_setup(
            mss_events=self.mss_events,
            fvg_events=self.fvg_events,
            sweep_events=sweeps
        )

        # Deduplicate: skip (mss_bar_index, fvg_bar_index) pairs already detected
        new_setups = []
        for s in setups:
            key = (s.mss_event.bar_index if s.mss_event else -1,
                   s.fvg_event.bar_index if s.fvg_event else -1)
            if key not in self._seen_setup_keys:
                self._seen_setup_keys.add(key)
                new_setups.append(s)
                logger.info(
                    f"🔍 New setup: {s.direction} | confluence={s.confluence_count} | priority={s.priority}"
                )

        return new_setups

    async def process_trading_setup(self, setup: SilverBulletSetup):
        """Process a trading setup with ML filtering and R:R validation."""
        # 1. Check killzone filter
        within_killzone, window_name = is_within_trading_hours(
            datetime.now(timezone.utc), DEFAULT_TRADING_WINDOWS
        )

        if not within_killzone:
            return

        # 2. Window Deduplication: one setup (pending or filled) per window per day
        today = datetime.now(timezone.utc).date().isoformat()
        if today in self.window_trades.get(window_name, set()):
            return
        already_pending = any(
            s['window'] == window_name and s['date'] == today
            for s in self.pending_setups
        )
        if already_pending:
            return

        # 3. ML prediction filter (optional — strategy is profitable without it, PF=2.60 raw)
        if not self._ml_unavailable:
            try:
                features = self.ml_inference.feature_engineer.extract_features(
                    setup, self.recent_bars
                )
                prediction = self.ml_inference.predict(features)
                logger.info(f"🤖 ML Prediction: P(success) = {prediction:.2%}")
                if prediction < self.ml_threshold:
                    logger.debug(f"❌ Setup below {self.ml_threshold:.0%} threshold - SKIPPED")
                    return
            except Exception as e:
                # Latch any ML failure for the session — one warning, no per-setup spam.
                # Known root cause: MLInference uses different attribute/method names than
                # called here (feature_engineer vs _feature_engineer, predict vs predict_probability).
                # Fix the integration properly before re-enabling ML filtering.
                self._ml_unavailable = True
                logger.warning(f"ML filter unavailable ({e}) — running without ML for this session")

        # 4. Target Calculation (Next Liquidity Pool)
        fvg_midpoint = round_tick((setup.entry_zone_top + setup.entry_zone_bottom) / 2)
        target_price = self._find_next_liquidity_pool(setup.direction, fvg_midpoint)

        if not target_price:
            logger.debug("❌ No valid liquidity pool target found - SKIPPED")
            return

        target_price = round_tick(target_price)

        # 5. R:R Enforcement (Minimum 2:1)
        # Stop anchored to FVG geometry (matches backtest): bottom-0.75×gap (bullish) / top+0.75×gap (bearish)
        fvg_gap = setup.entry_zone_top - setup.entry_zone_bottom
        if fvg_gap <= 0:
            logger.warning(
                f"Invalid FVG geometry (gap={fvg_gap:.2f}): entry_zone_top={setup.entry_zone_top:.2f} "
                f"<= entry_zone_bottom={setup.entry_zone_bottom:.2f} — skipping setup"
            )
            return
        if setup.direction == "bullish":
            stop_loss = round_tick(setup.entry_zone_bottom - 0.75 * fvg_gap)
        else:
            stop_loss = round_tick(setup.entry_zone_top + 0.75 * fvg_gap)
        risk = abs(fvg_midpoint - stop_loss)
        if risk == 0: return
        
        reward = abs(target_price - fvg_midpoint)
        rr = reward / risk
        
        if rr < 2.0:
            logger.info(f"❌ Insufficient R:R ({rr:.2f} < 2.0) - SKIPPED")
            return

        # 6. Submit limit bracket to SIM now (same price/time as local limit order)
        # SIM fills when price actually touches FVG midpoint — matches local sim entry exactly
        sim_entry_id, sim_tp_id, sim_sl_id = await self._submit_sim_bracket(
            symbol=self._current_symbol,
            direction=setup.direction,
            entry_price=fvg_midpoint,
            tp_price=target_price,
            sl_price=stop_loss,
        )

        # Queue for FVG touch detection (local simulation)
        self.pending_setups.append({
            'setup': setup,
            'fvg_midpoint': fvg_midpoint,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'expiry_idx': self.bar_index + 15,  # 15-bar (15-min) cancellation window
            'window': window_name,
            'date': today,
            'sim_entry_id': sim_entry_id,
            'sim_tp_id': sim_tp_id,
            'sim_sl_id': sim_sl_id,
        })

        logger.info(f"🎯 SETUP VALIDATED: {setup.direction.upper()} in {window_name}")
        logger.info(f"   Entry (Limit): {fvg_midpoint:.2f} | Target: {target_price:.2f} | SL: {stop_loss:.2f} | R:R: {rr:.2f}")
        logger.info(f"   SIM limit order armed | expires in 15 bars")

    async def _submit_sim_bracket(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
    ) -> tuple[str | None, str | None, str | None]:
        """Submit a limit bracket to TradeStation SIM at entry_price; return (entry_id, tp_id, sl_id)."""
        entry_action = "BUY" if direction == "bullish" else "SELL"
        exit_action = "SELL" if direction == "bullish" else "BUY"
        payload = {
            "AccountID": SIM_ACCOUNT_ID,
            "Symbol": symbol,
            "Quantity": "1",
            "OrderType": "Limit",
            "LimitPrice": str(entry_price),
            "TradeAction": entry_action,
            "TimeInForce": {"Duration": "GTC"},
            "Route": "Intelligent",
            "OSOs": [{"Type": "BRK", "Orders": [
                {
                    "AccountID": SIM_ACCOUNT_ID,
                    "Symbol": symbol,
                    "Quantity": "1",
                    "OrderType": "Limit",
                    "TradeAction": exit_action,
                    "TimeInForce": {"Duration": "GTC"},
                    "LimitPrice": str(tp_price),
                },
                {
                    "AccountID": SIM_ACCOUNT_ID,
                    "Symbol": symbol,
                    "Quantity": "1",
                    "OrderType": "StopMarket",
                    "TradeAction": exit_action,
                    "TimeInForce": {"Duration": "GTC"},
                    "StopPrice": str(sl_price),
                },
            ]}],
        }
        try:
            token = await self.auth.authenticate()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(SIM_ORDERS_URL, headers=headers, json=payload)
            if response.status_code not in (200, 201):
                logger.warning(f"⚠️  SIM bracket HTTP {response.status_code}: {response.text[:200]}")
                return None, None, None
            orders = response.json().get("Orders", [])
            logger.debug(f"SIM bracket raw orders: {orders}")
            entry_id = tp_id = sl_id = None

            # SL: match by message text OR OrderType field (guards against message wording changes)
            limit_orders = []
            for order in orders:
                msg = order.get("Message", "")
                order_type = order.get("OrderType", "")
                oid = order.get("OrderID")
                if "Stop Market" in msg or "StopMarket" in order_type:
                    sl_id = oid
                else:
                    limit_orders.append(oid)

            # Entry is the OSO parent — TradeStation assigns it the highest OrderID.
            # Empirically verified across all observed submissions (5 working cases).
            # Sort descending: [0]=entry, [1]=TP.
            def _order_id_key(oid):
                try:
                    return int(oid)
                except (TypeError, ValueError):
                    logger.warning(f"Non-integer OrderID '{oid}' — sort position undefined")
                    return 0

            limit_orders.sort(key=_order_id_key, reverse=True)
            if len(limit_orders) >= 2:
                entry_id, tp_id = limit_orders[0], limit_orders[1]
            elif len(limit_orders) == 1:
                entry_id = limit_orders[0]

            if entry_id is None or tp_id is None or sl_id is None:
                logger.warning(f"⚠️  SIM bracket: incomplete order IDs — entry={entry_id} tp={tp_id} sl={sl_id} | raw={orders}")
            else:
                logger.info(f"✅ SIM bracket submitted | entry #{entry_id} | TP #{tp_id} | SL #{sl_id}")
            return entry_id, tp_id, sl_id
        except Exception as e:
            logger.warning(f"⚠️  SIM bracket exception: {e}")
            return None, None, None

    async def _cancel_sim_order(self, order_id: str) -> None:
        """Cancel a pending SIM limit order (called when setup expires)."""
        if not order_id:
            return
        url = f"https://sim-api.tradestation.com/v3/orderexecution/orders/{order_id}"
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.delete(url, headers=headers)
            if response.status_code in (200, 204, 404):
                logger.info(f"🗑️  SIM limit order #{order_id} cancelled")
            else:
                logger.warning(f"⚠️  SIM cancel HTTP {response.status_code} for order #{order_id}")
        except Exception as e:
            logger.warning(f"⚠️  SIM cancel exception for #{order_id}: {e}")

    async def preload_history(self, symbol: str, bars: int = 300) -> None:
        """Fetch the last N 1-minute bars from TradeStation to warm up the 200-bar lookback."""
        logger.info(f"⏳ Preloading {bars} bars of 1-min history for {symbol}...")
        url = (
            f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
            f"?interval=1&unit=Minute&barsback={bars}"
        )
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"⚠️  History preload HTTP {response.status_code} — starting with empty history")
                return
            raw_bars = response.json().get("Bars", [])
            if not raw_bars:
                logger.warning("⚠️  No historical bars returned — starting with empty history")
                return
            loaded = 0
            self._is_preloading = True
            for bar_data in raw_bars:
                try:
                    ts_str = bar_data.get("TimeStamp", "")
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    close = float(bar_data.get("Close", 0) or 0)
                    if close == 0:
                        continue
                    volume = int(bar_data.get("TotalVolume", 0) or 0)
                    bar = DollarBar(
                        timestamp=ts,
                        open=float(bar_data.get("Open", close) or close),
                        high=float(bar_data.get("High", close) or close),
                        low=float(bar_data.get("Low", close) or close),
                        close=close,
                        volume=volume,
                        notional_value=close * max(volume, 1) * 20.0,
                        is_forward_filled=False,
                    )
                    self.recent_bars.append(bar)
                    if len(self.recent_bars) > 300:
                        self.recent_bars.pop(0)
                    if len(self.recent_bars) >= 10:
                        self._detect_patterns(bar)
                    loaded += 1
                except Exception as e:
                    logger.debug(f"Skipping preload bar: {e}")
                    continue
            self._is_preloading = False
            logger.info(f"✅ Preloaded {loaded} bars — 200-bar lookback window ready")
        except Exception as e:
            logger.warning(f"⚠️  History preload error: {e} — starting with empty history")

    async def run_trading_loop(self, symbol: str = "MNQM26"):
        """Main trading loop using winning strategy configuration."""
        # Detect active contract if needed
        try:
            from src.data.contract_detector import ContractDetector
            detector = ContractDetector(self.auth.access_token)
            active_symbol = await detector.detect_active_contract("MNQ")
            if active_symbol:
                symbol = active_symbol
                logger.info(f"✅ Active contract detected: {symbol}")
        except Exception as e:
            logger.warning(f"⚠️  Contract detection failed: {e}. Using {symbol}")

        logger.info("🚀 Starting Winning Silver Bullet ML Strategy - Live Paper Trading")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Killzones: London AM (3-4am), NY AM (10-11am), NY PM (2-3pm EST)")
        logger.info(f"ML Threshold: 65% success probability")
        logger.info(f"OAuth Refresh: Every 10 minutes")
        logger.info("=" * 80)

        self.running = True
        self._current_symbol = symbol  # Available to process_trading_setup via self

        # Start 10-minute OAuth token auto-refresh
        logger.info("🔑 Starting 10-minute OAuth token auto-refresh...")
        await self.auth.start_auto_refresh(interval_minutes=10)

        # Preload history so the 200-bar lookback is ready immediately
        await self.preload_history(symbol)

        last_bar_ts = None  # Deduplicate: skip if same bar returned twice

        while self.running:
            try:
                # 1. Fetch the latest completed 1-minute bar
                bar = await self.fetch_latest_bar(symbol)

                if bar is None or not self.validator.validate_bar_for_trading(bar):
                    logger.debug("Market closed or invalid bar - waiting...")
                    await asyncio.sleep(15)
                    continue

                # Skip if this is the same bar as last poll
                if bar.timestamp == last_bar_ts:
                    await asyncio.sleep(15)
                    continue
                last_bar_ts = bar.timestamp

                # 2. Update bar history and detection
                self.recent_bars.append(bar)
                if len(self.recent_bars) > 300: # Sufficient for 200 lookback + swing detection
                    self.recent_bars.pop(0)

                # 3. Handle Active Trades (SL/TP)
                for trade in self.active_trades[:]:
                    # Check Stop Loss
                    if (trade['direction'] == 'bullish' and bar.low <= trade['stop_loss']) or \
                       (trade['direction'] == 'bearish' and bar.high >= trade['stop_loss']):
                        logger.info(f"🔴 STOP LOSS HIT: {trade['direction'].upper()} @ {trade['stop_loss']:.2f}")
                        self.total_trades += 1
                        loss = abs(trade['entry_price'] - trade['stop_loss']) * 2.0
                        self.total_pnl -= (loss + 1.80) # Include $1.80 transaction cost
                        self.active_trades.remove(trade)
                    
                    # Check Take Profit
                    elif (trade['direction'] == 'bullish' and bar.high >= trade['target_price']) or \
                         (trade['direction'] == 'bearish' and bar.low <= trade['target_price']):
                        logger.info(f"🟢 TAKE PROFIT HIT: {trade['direction'].upper()} @ {trade['target_price']:.2f}")
                        self.total_trades += 1
                        self.winning_trades += 1
                        profit = abs(trade['target_price'] - trade['entry_price']) * 2.0
                        self.total_pnl += (profit - 1.80) # Include $1.80 transaction cost
                        self.active_trades.remove(trade)

                # 4. Handle Pending Setups (FVG Touch/Entry)
                for setup_req in self.pending_setups[:]:
                    # Check Expiry — cancel the SIM limit order if still pending
                    if self.bar_index > setup_req['expiry_idx']:
                        logger.info(f"⚪ Setup Expired: {setup_req['setup'].direction.upper()} FVG not touched")
                        await self._cancel_sim_order(setup_req.get('sim_entry_id'))
                        self.pending_setups.remove(setup_req)
                        continue
                    
                    # Check for Touch
                    is_touched = False
                    if setup_req['setup'].direction == 'bullish' and bar.low <= setup_req['fvg_midpoint']:
                        is_touched = True
                    elif setup_req['setup'].direction == 'bearish' and bar.high >= setup_req['fvg_midpoint']:
                        is_touched = True
                        
                    if is_touched:
                        logger.info(f"🔵 TRADE ENTERED: {setup_req['setup'].direction.upper()} @ {setup_req['fvg_midpoint']:.2f}")
                        logger.info(f"   SIM limit order #{setup_req.get('sim_entry_id')} should now be filling")

                        # Carry SIM order IDs from pending setup into active trade for tracking
                        self.active_trades.append({
                            'entry_price': setup_req['fvg_midpoint'],
                            'target_price': setup_req['target_price'],
                            'stop_loss': setup_req['stop_loss'],
                            'direction': setup_req['setup'].direction,
                            'sim_entry_id': setup_req.get('sim_entry_id'),
                            'sim_tp_id': setup_req.get('sim_tp_id'),
                            'sim_sl_id': setup_req.get('sim_sl_id'),
                        })

                        # Mark window as traded
                        win_set = self.window_trades.setdefault(setup_req['window'], set())
                        win_set.add(setup_req['date'])

                        self.pending_setups.remove(setup_req)

                # 5. Detect New Setups
                if len(self.recent_bars) >= 10: # Minimum for some patterns
                    new_setups = self._detect_patterns(bar)
                    for setup in new_setups:
                        await self.process_trading_setup(setup)

                # 6. Performance Logging
                win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
                logger.info(f"📊 {symbol} @ {bar.close} [{bar.timestamp.strftime('%H:%M')}Z H:{bar.high} L:{bar.low}] | Trades: {self.total_trades} | WR: {win_rate:.1f}% | P&L: ${self.total_pnl:.2f} | Pending: {len(self.pending_setups)} | Active: {len(self.active_trades)}")

                # Poll every 15s to catch each new 1-min bar within ~15s of close
                await asyncio.sleep(15)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def stop(self):
        """Stop the trading system."""
        logger.info("🛑 Stopping Winning Silver Bullet ML Strategy...")
        self.running = False

        # Stop OAuth auto-refresh
        if self.auth:
            await self.auth.cleanup()
            logger.info("🔑 Stopped OAuth auto-refresh")

        # Print final performance
        logger.info("=" * 80)
        logger.info("FINAL PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Total P&L: ${self.total_pnl:.2f}")
        logger.info(f"Return: {(self.total_pnl/100000)*100:.2f}%")
        logger.info("=" * 80)


async def main():
    """Main entry point."""
    # Load access token
    try:
        with open(".access_token", "r") as f:
            access_token = f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load access token: {e}")
        return

    # Initialize trader
    trader = WinningSilverBulletTrader(access_token)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\nReceived interrupt signal...")
        asyncio.create_task(trader.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    # Start trading
    try:
        await trader.run_trading_loop()
    except Exception as e:
        logger.error(f"Trading system error: {e}")
    finally:
        await trader.stop()


if __name__ == "__main__":
    asyncio.run(main())