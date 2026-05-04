#!/usr/bin/env python3
"""
TIER 2 FVG Paper Trading - TradeStation HTTP Polling + SIM Order Placement
Configuration: SL5.0x_TP5.0x_Midpoint_H1Sweep + ML Meta-Labeling Filter

Entry fires a bracket order on SIM account (entry + TP limit + SL stop).
The SIM account manages TP/SL fills. Local per-bar simulation is the
authoritative P&L record and handles the time-stop (cancel bracket + flat close).
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pytz
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3
from src.data.models import DollarBar

# Configuration (OPTIMIZED TIER 2)
TIER2_CONFIG = "SL5.0x_TP5.0x_Midpoint_H1Sweep_MLFilter"
SL_MULTIPLIER = 5.0
TP_MULTIPLIER = 5.0
ENTRY_PCT = 0.5  # Midpoint entry
ATR_THRESHOLD = 0.5
MAX_GAP_DOLLARS = 60.0
MAX_HOLD_BARS = 120  # 2 Hours (matches final optimization)
CONTRACTS_PER_TRADE = 1

# MNQ Specifications
MNQ_POINT_VALUE = 20.0          # used for dollar-bar notional computation
MNQ_DOLLAR_VALUE = 2.0          # $2 per index point — P&L scaling and gap-size dollar filter

# Transaction Costs
COMMISSION_PER_CONTRACT = 0.40
TRANSACTION_COST = COMMISSION_PER_CONTRACT * CONTRACTS_PER_TRADE * 2  # $0.80/roundtrip

# ML Filter
ML_MODEL_PATH = Path(__file__).parent.parent.parent / "models/xgboost/tier2_meta_labeling_model.pkl"
ML_THRESHOLD = 0.52  # PF-optimal threshold from OOS sweep (PF=1.917, 18/204 OOS trades — story 6-6 LR model)

# TradeStation market data API
SYMBOL = "MNQM26"
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
BARS_BASE_URL = (f"https://api.tradestation.com/v3/marketdata/barcharts/{SYMBOL}"
                 f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}")
HISTORY_HOURS = 48  # Enough history for H1 swing detection
POLL_INTERVAL_SECONDS = 60

# TradeStation SIM order placement
SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"

ET_TZ = pytz.timezone('US/Eastern')

# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

_handlers: list = [logging.FileHandler(log_dir / 'tier2_streaming_working.log')]
if sys.stdout.isatty():
    _handlers.append(logging.StreamHandler())
_log_level = logging.DEBUG if os.environ.get("TIER2_DEBUG") else logging.INFO
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=_handlers,
)
logger = logging.getLogger(__name__)


class MetaLabelingFilter:
    """ML-based secondary filter that approves/rejects Tier 2 setups."""

    FEATURE_COLS = [
        'fvg_fill_pct', 'sweep_window_vol', 'volume_ratio', 'signal_direction',
        'h1_trend_slope', 'atr', 'session_displacement', 'session_volume_ratio',
    ]

    def __init__(self, model_path: Path, threshold: float = ML_THRESHOLD):
        self.threshold = threshold
        self.model = None
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                logger.info(f"ML filter loaded from {model_path} (threshold={threshold})")
            except Exception as e:
                logger.warning(f"ML model load failed: {e} — falling back to pass-through")
        else:
            logger.warning(f"ML model not found at {model_path} — falling back to pass-through")

    def predict_proba(self, features: dict) -> float:
        """Return P(success). Returns 1.0 (pass-through) if model unavailable."""
        if self.model is None:
            return 1.0
        try:
            df_feat = pd.DataFrame([features])[self.FEATURE_COLS].copy()
            df_feat['signal_direction'] = 1 if df_feat['signal_direction'].iloc[0] == "bullish" else 0
            # Model is a Pipeline(StandardScaler + LogisticRegression)
            return float(self.model.predict_proba(df_feat)[0, 1])
        except Exception as e:
            logger.warning(f"ML inference failed: {e} — returning pass-through")
            return 1.0


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
        self._is_backfill: bool = True
        self.session_start_time: Optional[datetime] = None

        # ML Filter
        self.ml_filter = MetaLabelingFilter(ML_MODEL_PATH)

        # H1 sweep state — flags persist until 6-hour expiry window lapses
        self.h1_bullish_sweep_active = False
        self.h1_bearish_sweep_active = False
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        self._bullish_sweep_expires: datetime = _epoch
        self._bearish_sweep_expires: datetime = _epoch

        # Feature enrichment state
        self._last_entry_bar: int = -120
        self._bullish_sweep_bar: int = -20
        self._bearish_sweep_bar: int = -20
        self._session_open_price: float = np.nan
        self._session_high: float = float('-inf')
        self._session_low: float = float('inf')
        self._daily_ranges: list[float] = [] # max 20
        self._h1_atr: float = 0.0
        self._h1_slope: float = 0.0
        self._current_day: Optional[datetime.date] = None

    async def initialize(self):
        logger.info("=" * 70)
        logger.info("TIER 2 FVG PAPER TRADING - SIM ORDER PLACEMENT")
        logger.info("=" * 70)
        logger.info(f"Configuration: {TIER2_CONFIG}")
        logger.info(f"Symbol: {SYMBOL}")
        logger.info(f"Max hold: {MAX_HOLD_BARS} bars | SL/TP mult: {SL_MULTIPLIER}x")
        logger.info(f"Entry Level: {ENTRY_PCT*100}% (Mean Threshold)")
        logger.info(f"ML Filter: {'ACTIVE' if self.ml_filter.model else 'PASS-THROUGH'} | threshold={ML_THRESHOLD}")
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
                if bar and bar.timestamp <= now_utc and (
                    not self._last_processed_timestamp or bar.timestamp > self._last_processed_timestamp
                ):
                    self.dollar_bars.append(bar)
                    self._last_processed_timestamp = bar.timestamp
                    new_bars.append(bar)

                    # Update session stats
                    bar_et = bar.timestamp.astimezone(ET_TZ)
                    if self._current_day != bar_et.date():
                        if self._current_day is not None:
                            # Day closed, record range for ADR
                            self._daily_ranges.append(self._session_high - self._session_low)
                            if len(self._daily_ranges) > 20: self._daily_ranges.pop(0)
                        
                        self._current_day = bar_et.date()
                        self._session_open_price = np.nan
                        self._session_high, self._session_low = bar.high, bar.low
                    else:
                        self._session_high = max(self._session_high, bar.high)
                        self._session_low = min(self._session_low, bar.low)
                    
                    if np.isnan(self._session_open_price) and bar_et.hour >= 6:
                        self._session_open_price = bar.open

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
            high_val = float(d["High"])
            low_val = float(d["Low"])
            volume = int(d.get("TotalVolume", 0))
            # Calculate a realistic notional value to pass Pydantic validation
            notional = max(((high_val + low_val) / 2) * volume * MNQ_POINT_VALUE, 0.01)

            return DollarBar(
                timestamp=datetime.fromisoformat(d["TimeStamp"].replace('Z', '+00:00')),
                open=float(d["Open"]), high=high_val, low=low_val,
                close=float(d["Close"]), volume=volume,
                notional_value=notional, bar_num=len(self.dollar_bars)
            )
        except Exception as e:
            logger.warning(f"⚠️ Bar parse failed: {e}")
            return None

    def _update_h1_structure(self):
        """Resample 1m bars to H1 and detect liquidity sweeps in the last completed H1 bar."""
        if not self.dollar_bars: return

        df = pd.DataFrame([vars(b) for b in self.dollar_bars])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        h1 = df.set_index('timestamp').resample('1h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna().reset_index()

        # Need at least 6 completed H1 bars plus the currently forming bar
        if len(h1) < 7: return

        # Swing detection on completed bars only (exclude the forming h1.iloc[-1])
        completed = h1.iloc[:-1].reset_index(drop=True)

        # Calculate H1 ATR and Slope
        tr = np.maximum(completed['high'] - completed['low'], 
                        np.maximum(np.abs(completed['high'] - completed['close'].shift(1)), 
                                  np.abs(completed['low'] - completed['close'].shift(1))))
        h1_atr_val = tr.rolling(20, min_periods=5).mean().iloc[-1]
        self._h1_atr = float(h1_atr_val) if not np.isnan(h1_atr_val) else 0.0
        
        if len(completed) >= 6 and self._h1_atr > 0:
            closes = completed['close'].values[-6:]
            slope = np.polyfit(range(6), closes, 1)[0]
            self._h1_slope = slope / self._h1_atr
        else:
            self._h1_slope = 0.0

        sh, sl = [], []
        highs, lows = completed['high'].values, completed['low'].values
        for i in range(2, len(completed) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                sh.append((completed.loc[i, 'timestamp'], highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                sl.append((completed.loc[i, 'timestamp'], lows[i]))

        # Check the last COMPLETED H1 bar for sweeps (not the still-forming bar)
        last = completed.iloc[-1]
        now_utc = datetime.now(timezone.utc)

        new_bullish_sweep = False
        new_bearish_sweep = False

        for t, val in sh:
            if t < last['timestamp'] - timedelta(hours=2):  # swing must be confirmed
                if last['high'] > val and last['close'] < val:
                    new_bearish_sweep = True
                    # P5: record actual bar index at sweep detection (current tail of dollar_bars)
                    self._bearish_sweep_bar = len(self.dollar_bars)
                    logger.info(f"🎯 H1 BEARISH SWEEP detected at {val:.2f}")

        for t, val in sl:
            if t < last['timestamp'] - timedelta(hours=2):
                if last['low'] < val and last['close'] > val:
                    new_bullish_sweep = True
                    self._bullish_sweep_bar = len(self.dollar_bars)
                    logger.info(f"🎯 H1 BULLISH SWEEP detected at {val:.2f}")

        # Sweep stays active for 6 H1 bars after detection; expires otherwise
        sweep_window = timedelta(hours=6)
        last_ts = last['timestamp'].to_pydatetime().replace(tzinfo=timezone.utc)
        if new_bullish_sweep:
            self.h1_bullish_sweep_active = True
            self._bullish_sweep_expires = last_ts + sweep_window
        elif now_utc > self._bullish_sweep_expires:
            self.h1_bullish_sweep_active = False

        if new_bearish_sweep:
            self.h1_bearish_sweep_active = True
            self._bearish_sweep_expires = last_ts + sweep_window
        elif now_utc > self._bearish_sweep_expires:
            self.h1_bearish_sweep_active = False

    async def _advance_active_trade(self, bar: DollarBar) -> bool:
        if not self.active_trade: return False
        t = self.active_trade
        t.bars_held += 1
        closed = False
        if t.direction == "LONG":
            if bar.low <= t.sl_price:
                await self._close_active_trade(bar, t.sl_price, "sl")
                closed = True
            elif bar.high >= t.tp_price:
                await self._close_active_trade(bar, t.tp_price, "tp")
                closed = True
        else:
            if bar.high >= t.sl_price:
                await self._close_active_trade(bar, t.sl_price, "sl")
                closed = True
            elif bar.low <= t.tp_price:
                await self._close_active_trade(bar, t.tp_price, "tp")
                closed = True
        # Guard: re-check self.active_trade in case close was already called
        if not closed and self.active_trade and t.bars_held >= MAX_HOLD_BARS:
            await self._close_active_trade(bar, bar.close, "time")
            closed = True
        return closed

    async def _close_active_trade(self, bar: DollarBar, price: float, reason: str):
        t = self.active_trade
        pnl = ((price - t.entry_price) if t.direction == "LONG" else (t.entry_price - price)) * MNQ_DOLLAR_VALUE - TRANSACTION_COST
        self.completed_trades.append(CompletedTrade(
            t.entry_time, bar.timestamp, t.direction, t.entry_price, price, reason, t.bars_held, pnl
        ))
        self.active_trade = None
        logger.info(f"Trade Closed: {reason.upper()} | P&L: ${pnl:.2f}")

    async def _detect_and_enter(self, bar: DollarBar, is_backfill: bool):
        if self.active_trade: return
        bars = self.dollar_bars
        if len(bars) < 20: return  # need 20 bars for ATR and volume features

        # Tuesday filter: consistently PF<1.0 across all 5 months of backtest data
        bar_et = bar.timestamp.astimezone(ET_TZ)
        if bar_et.weekday() == 1:  # 1 = Tuesday
            return

        if self.h1_bullish_sweep_active:
            fvg = self._detect_fvg(bars, bullish=True)
            if fvg:
                features = self._extract_features(bars, bar, fvg, "bullish")
                proba = self.ml_filter.predict_proba(features)
                if proba >= ML_THRESHOLD:
                    logger.info(f"Signal ALLOWED by ML threshold | P(Success)={proba:.3f}")
                    await self._enter_trade(fvg, bar, len(bars) - 1, is_backfill)
                else:
                    logger.info(f"Signal FILTERED by ML threshold | P(Success)={proba:.3f} < {ML_THRESHOLD}")
        elif self.h1_bearish_sweep_active:
            fvg = self._detect_fvg(bars, bullish=False)
            if fvg:
                features = self._extract_features(bars, bar, fvg, "bearish")
                proba = self.ml_filter.predict_proba(features)
                if proba >= ML_THRESHOLD:
                    logger.info(f"Signal ALLOWED by ML threshold | P(Success)={proba:.3f}")
                    await self._enter_trade(fvg, bar, len(bars) - 1, is_backfill)
                else:
                    logger.info(f"Signal FILTERED by ML threshold | P(Success)={proba:.3f} < {ML_THRESHOLD}")

    def _extract_features(self, bars: list, bar: DollarBar, fvg: dict, direction: str) -> dict:
        """Extract inference features matching the training data schema (raw index points)."""
        assert direction in ("bullish", "bearish"), f"Invalid direction: {direction!r}"
        atr = self._calculate_atr(bars)

        # gap_size in raw index points — matches training CSV (not dollar-scaled)
        gap_size = fvg["top"] - fvg["bottom"]

        # Volume ratio: directional volume balance over last 20 bars
        recent = bars[-20:]
        up_vol = sum(b.volume for b in recent if b.close >= b.open)
        dn_vol = sum(b.volume for b in recent if b.close < b.open)
        if direction == "bullish":
            vol_ratio = up_vol / dn_vol if dn_vol > 0 else 99.0
        else:
            vol_ratio = dn_vol / up_vol if up_vol > 0 else 99.0

        bar_et = bar.timestamp.astimezone(ET_TZ)
        
        # New Context Features
        session_displacement = (bar.close - self._session_open_price) / atr if not np.isnan(self._session_open_price) and atr > 0 else 0.0
        
        adr_20 = np.mean(self._daily_ranges) if self._daily_ranges else 0.0
        adr_pct_used = np.clip((self._session_high - self._session_low) / adr_20, 0, 2) if adr_20 > 0 else 0.5
        
        current_bar_idx = len(self.dollar_bars)
        sweep_bar = self._bullish_sweep_bar if direction == "bullish" else self._bearish_sweep_bar
        fvg_to_sweep_bars = min(current_bar_idx - sweep_bar, 20)
        prior_setup_proximity = min(current_bar_idx - self._last_entry_bar, 120)

        # New features
        et_hour = bar_et.hour
        sin_hour = np.sin(2 * np.pi * et_hour / 24)
        cos_hour = np.cos(2 * np.pi * et_hour / 24)
        
        recent_vol_mean = np.mean([b.volume for b in recent]) if recent else 0
        session_volume_ratio = bar.volume / recent_vol_mean if recent_vol_mean > 0 else 1.0
        
        # P10/D3: bar close relative to FVG bottom (varies meaningfully; entry_est is always 0.5)
        fvg_fill_pct = (bar.close - fvg["bottom"]) / gap_size if gap_size > 0 else 0.5
        
        bar_range = bar.high - bar.low
        bar_body_ratio = abs(bar.close - bar.open) / bar_range if bar_range > 0 else 0.5
        
        h, m = bar_et.hour, bar_et.minute
        silver_bullet_window = 1 if (h == 3) or (h == 4 and m == 0) or (h == 9 and m >= 30) or (h == 10) else 0
        
        sweep_window_vol = silver_bullet_window * session_volume_ratio
        
        direction_sign = 1 if direction == "bullish" else -1
        slope_direction_match = 1 if np.sign(self._h1_slope) == direction_sign else 0

        features = {
            "atr": atr,
            "gap_size": gap_size,
            "volume_ratio": vol_ratio,
            "et_hour": et_hour,
            "day_of_week": bar_et.weekday(),
            "signal_direction": direction,
            "session_displacement": session_displacement,
            "adr_pct_used": adr_pct_used,
            "fvg_to_sweep_bars": fvg_to_sweep_bars,
            "prior_setup_proximity": prior_setup_proximity,
            "h1_trend_slope": self._h1_slope,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "session_volume_ratio": session_volume_ratio,
            "fvg_fill_pct": fvg_fill_pct,
            "bar_body_ratio": bar_body_ratio,
            "sweep_window_vol": sweep_window_vol,
            "slope_direction_match": slope_direction_match
        }
        
        # Log all 18 feature values at DEBUG level
        logger.debug(f"📊 Feature Vector: ATR={atr:.2f}, Gap={gap_size:.2f}, VolRatio={vol_ratio:.2f}, Slope={self._h1_slope:.4f}, "
                     f"ADR_Pct={adr_pct_used:.2f}, SinH={sin_hour:.2f}, CosH={cos_hour:.2f}, SessVol={session_volume_ratio:.2f}, "
                     f"FVG_Fill={fvg_fill_pct:.2f}, BodyRatio={bar_body_ratio:.2f}, SweepVol={sweep_window_vol:.2f}, "
                     f"SlopeMatch={slope_direction_match}")
        logger.debug(f"Full Features: {features}")
        return features

    def _detect_fvg(self, bars: list, bullish: bool) -> Optional[dict]:
        c1, c2, c3 = bars[-3], bars[-2], bars[-1]
        if bullish:
            # True FVG: candle 1 high strictly below candle 3 low (price void)
            if not (c1.high < c3.low and c2.close > c2.open): return None
            top, bot = c3.low, c1.high
        else:
            if not (c1.low > c3.high and c2.close < c2.open): return None
            top, bot = c1.low, c3.high

        if top <= bot: return None
        gap_pts = top - bot
        # ATR filter in raw points; dollar ceiling uses MNQ_DOLLAR_VALUE ($2/pt) to match backtest
        if gap_pts < ATR_THRESHOLD * self._calculate_atr(bars): return None
        if gap_pts * MNQ_DOLLAR_VALUE > MAX_GAP_DOLLARS: return None

        return {"direction": "bullish" if bullish else "bearish", "top": top, "bottom": bot}

    def _calculate_atr(self, bars: list) -> float:
        """20-bar mean True Range in raw index points."""
        if len(bars) < 20: return 10.0
        sliced = bars[-20:]
        tr = [
            max(b.high - b.low, abs(b.high - sliced[i-1].close), abs(b.low - sliced[i-1].close))
            for i, b in enumerate(sliced) if i > 0
        ]
        return sum(tr) / len(tr)

    async def _enter_trade(self, fvg: dict, bar: DollarBar, idx: int, is_backfill: bool):
        self._last_entry_bar = len(self.dollar_bars)  # track even during backfill for feature consistency
        if is_backfill: return  # never submit orders against historical replay bars

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
