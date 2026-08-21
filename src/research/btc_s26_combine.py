#!/usr/bin/env python3
import asyncio
import logging
import os
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

SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/brokerage/orders"
# Paper-only by default. SIM2797251F is the YANK+MIM combine-MIRROR account;
# placing MBT (Bitcoin) orders here contaminates the mirror equity curve and
# makes the SIM dress-rehearsal diverge from the real MNQ combine. The bot still
# polls, computes signals and logs paper trades to trades.db. Set
# S26_COMBINE_PLACE_ORDERS=1 to restore live SIM order placement.
PLACE_ORDERS = os.environ.get("S26_COMBINE_PLACE_ORDERS", "0") == "1"
# MBT (Micro Bitcoin) rolls MONTHLY on CME. This used to be a hand-set env var, and it
# went stale twice: MBTM26 (June) 404'd silently for a month, then MBTN26 (July) did the
# same from 2026-07-30 — ~3,400 failed polls a day while systemd still reported the
# service "active (running)". Correcting the constant fixes one roll and guarantees the
# next one breaks, so the symbol is now resolved by probing the broker.
#
# S26_COMBINE_SYMBOL still pins a contract (escape hatch, mirrors MIM_NB_SYMBOL) and
# disables auto-roll; S26_COMBINE_AUTOROLL=0 disables it while keeping the seed.
SYMBOL_OVERRIDE = os.environ.get("S26_COMBINE_SYMBOL") or None
AUTOROLL = os.environ.get("S26_COMBINE_AUTOROLL", "1") != "0"
FALLBACK_SYMBOL = "MBTN26"     # seed only — used if probing cannot reach the API at all
DEFAULT_SYMBOL = SYMBOL_OVERRIDE or FALLBACK_SYMBOL
# CME month codes, Jan..Dec. MBT lists monthly, so the front month is normally the
# current month until it expires, then the next.
MONTH_CODES = "FGHJKMNQUVXZ"
ROLL_LOOKAHEAD_MONTHS = 4      # probe current month + next 3
PROBE_WINDOW_H = 24            # a live contract prints continuously; 24h is unambiguous
# Minimum bars in the probe window to accept a contract. Guards against rolling onto a
# barely-quoted far month: measured 2026-08-06, MBTQ26 (front) printed 1197 bars/24h,
# MBTU26 120, MBTV26 only 8. Front-month-first ordering already prefers the liquid one;
# this is the backstop for when the nearer months are genuinely dead.
MIN_PROBE_BARS = 10
# Consecutive empty/failed polls before escalating to a loud CRITICAL log —
# a dead/expired contract 404s forever otherwise with only a WARNING per poll,
# which is easy to miss for a service that stays "active (running)".
STALE_DATA_ALERT_POLLS = 30

class S26CombineTrader:
    def __init__(self):
        from src.monitoring.trade_db import TradeDatabase
        self.db = TradeDatabase()
        self.symbol = DEFAULT_SYMBOL
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
        self.active_trade = None  # paper position being tracked bar-by-bar (see _manage_active_paper_trade)
        self.consecutive_empty_polls = 0
        self._stale_alert_fired = False
        # Set by _note_empty_poll (sync) and consumed by poll_and_process (async), which
        # is the only place that may await a roll.
        self._roll_pending = False

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
        # Resolve the front month BEFORE the first poll, so a restart after an expiry
        # self-heals instead of 404-ing until someone notices.
        await self._resolve_front_month("startup")
        logger.info(f"Polling {self.symbol} market data from TradeStation SIM...")
        
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
            if self._roll_pending:
                # STALE_DATA escalated on a previous (sync) poll — try to self-heal here,
                # where awaiting is allowed. Cleared either way so a failed roll retries
                # only after another full STALE_DATA_ALERT_POLLS run of empty polls,
                # rather than probing the broker every 60s forever.
                self._roll_pending = False
                await self._resolve_front_month("stale data")
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            since = self.last_ts or (datetime.now(timezone.utc) - timedelta(hours=10))
            url = (
                f"https://api.tradestation.com/v3/marketdata/barcharts/{self.symbol}"
                f"?interval=1&unit=Minute&firstdate={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )

            response = await self.http.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} from TS API for {self.symbol}")
                self._note_empty_poll()
                return

            bars_data = response.json().get("Bars", [])
            if not bars_data:
                self._note_empty_poll()
                return

            self.consecutive_empty_polls = 0
            self._stale_alert_fired = False
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

    @staticmethod
    def _candidate_symbols(now_utc=None):
        """Front-month-first MBT candidates: current month, then the next few."""
        now_utc = now_utc or datetime.now(timezone.utc)
        out, y, m = [], now_utc.year, now_utc.month
        for _ in range(ROLL_LOOKAHEAD_MONTHS):
            out.append(f"MBT{MONTH_CODES[m - 1]}{y % 100:02d}")
            m += 1
            if m > 12:
                m, y = 1, y + 1
        return out

    async def _probe_symbol(self, sym: str, headers) -> int:
        """Bars printed by `sym` in the last PROBE_WINDOW_H hours; 0 if dead.

        Must use `firstdate`, NOT `barsback`. `barsback=1` happily returns the final bar a
        contract ever printed, so an EXPIRED contract answers HTTP 200 with a bar and
        looks alive — measured on 2026-08-06, MBTN26 (expired) returned 200/1 bar to
        `barsback=1` while the bot's own `firstdate` poll was 404-ing. A probe that cannot
        distinguish a dead contract from a live one would have made this auto-roll a
        no-op, which is the exact failure it exists to prevent.
        """
        since = (datetime.now(timezone.utc)
                 - timedelta(hours=PROBE_WINDOW_H)).strftime('%Y-%m-%dT%H:%M:%SZ')
        try:
            r = await self.http.get(
                f"https://api.tradestation.com/v3/marketdata/barcharts/{sym}"
                f"?interval=1&unit=Minute&firstdate={since}", headers=headers)
            if r.status_code != 200:
                return 0
            return len(r.json().get("Bars", []))
        except Exception as exc:
            logger.warning("roll probe failed for %s: %s", sym, exc)
            return 0

    async def _resolve_front_month(self, reason: str) -> bool:
        """Point `self.symbol` at the live front month, by asking the SAME API the bot
        polls. Returns True if the symbol changed.

        Probing the broker rather than computing an expiry date deliberately: a date rule
        has to encode CME's calendar and holidays and is itself something that can rot
        silently. "Does this contract return bars?" is the question we actually care
        about, and it is answered by the venue.
        """
        if SYMBOL_OVERRIDE:
            logger.info("roll skipped (%s): S26_COMBINE_SYMBOL pins %s", reason, SYMBOL_OVERRIDE)
            return False
        if not AUTOROLL:
            logger.info("roll skipped (%s): S26_COMBINE_AUTOROLL=0", reason)
            return False
        try:
            token = await self.auth.authenticate()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        except Exception as exc:
            logger.error("roll aborted (%s): auth failed: %s — keeping %s",
                         reason, exc, self.symbol)
            return False

        for sym in self._candidate_symbols():
            n_bars = await self._probe_symbol(sym, headers)
            if n_bars >= MIN_PROBE_BARS:
                logger.info("roll probe: %s printed %d bars in %dh", sym, n_bars, PROBE_WINDOW_H)
                if sym == self.symbol:
                    # Confirmed-live but still frozen: the poll loop's `since` is
                    # `self.last_ts`, and TradeStation 404s (not 200+empty) a
                    # firstdate with zero bars strictly after it -- exactly what
                    # happens once the poll catches up to the most recent bar.
                    # Every subsequent poll re-asks the identical doomed query
                    # forever, since a non-200 response never reaches the code
                    # that would advance last_ts (measured 2026-08-21:
                    # firstdate=<last bar's own timestamp> -> 404 "No data
                    # available"; firstdate a few minutes earlier -> 200,
                    # including that same bar). This probe just proved real
                    # bars exist within PROBE_WINDOW_H hours, so reset the same
                    # way an actual roll does two lines below -- both last_ts
                    # AND bars, not last_ts alone: since the dedupe check is
                    # `not self.last_ts or ts > self.last_ts`, clearing last_ts
                    # without clearing bars would re-append every bar in the
                    # next wide (now-10h) poll as "new", duplicating whatever
                    # self.bars already held.
                    logger.warning("roll check (%s): %s is still live but frozen at last_ts -- resetting to unstick the poll", reason, sym)
                    self.bars, self.last_ts = [], None
                    self.consecutive_empty_polls = 0
                    self._stale_alert_fired = False
                    return False
                logger.warning("AUTOROLL: %s → %s (%s)", self.symbol, sym, reason)
                self.symbol = sym
                self.bars, self.last_ts = [], None   # never splice two contracts' bars
                self.consecutive_empty_polls = 0
                self._stale_alert_fired = False
                return True

        logger.critical(
            "AUTOROLL FAILED (%s): none of %s returned bars — keeping %s. The bot is "
            "NOT trading until this resolves.", reason,
            ",".join(self._candidate_symbols()), self.symbol)
        return False

    def _note_empty_poll(self) -> None:
        """Track consecutive polls that returned no bars and escalate loudly.

        A dead/expired contract 404s on every poll forever with only a WARNING
        each time — that's how MBTM26 ran blind for a month (2026-06-26 to
        2026-07-27) while systemd still reported the service "active (running)".
        """
        self.consecutive_empty_polls += 1
        if self.consecutive_empty_polls >= STALE_DATA_ALERT_POLLS and not self._stale_alert_fired:
            self._stale_alert_fired = True
            logger.critical(
                "STALE_DATA: no bars for %s in %d consecutive polls (~%d min) — "
                "contract may be expired/wrong. Attempting auto-roll.",
                self.symbol, self.consecutive_empty_polls, self.consecutive_empty_polls,
            )
            self._roll_pending = True

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

        # 2. Manage an existing paper position first (mirrors s26_soft_fvg_streaming.py,
        # the sibling Kraken bot with proven exit tracking). PLACE_ORDERS mode relies on
        # the broker's own TP/SL bracket orders instead and skips this bookkeeping.
        if not PLACE_ORDERS and self.active_trade:
            self._manage_active_paper_trade(current_bar)
            return  # one action (exit or nothing) per cycle, same as the sibling

        # 3. Entry Logic (only if flat — the active_trade case already returned above)
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
                await self.submit_bracket_order(dir_str, entry_price, tp, sl, proba, df.index[-1])

    def _manage_active_paper_trade(self, current_bar) -> None:
        """Bar-by-bar TP/SL/time-stop check for the open paper position.

        Ported from s26_soft_fvg_streaming.py's proven implementation — without
        this, a paper entry was logged once and then forgotten forever (every
        historical trader-s26-combine row has exit_price=0.0, pnl=0.0).
        """
        t = self.active_trade
        t['hold_time'] += 1

        exit_reason = None
        exit_price = 0.0

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
            logger.info(f"🏁 S26 COMBINE Trade Closed [{exit_reason}]: PNL=${pnl:.2f} | Hold: {t['hold_time']}m")
            self.db.log_trade(
                trader_id='trader-s26-combine',
                timestamp=t['ts'].isoformat(),
                pnl=round(pnl, 2),
                direction=t['dir'],
                entry_price=round(t['entry'], 2),
                exit_price=round(exit_price, 2),
                exit_reason=exit_reason,
                ml_proba=round(t['proba'], 3),
                metadata={'contracts': self.contracts, 'paper': True, 'sl': t['sl'], 'tp': t['tp']}
            )
            self.active_trade = None

    async def submit_bracket_order(self, direction, entry, tp, sl, proba, entry_ts):
        if not PLACE_ORDERS:
            self.active_trade = {
                'dir': direction,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'proba': proba,
                'hold_time': 0,
                'ts': entry_ts,
            }
            logger.info(f"📝 PAPER (no broker order): {direction} entry={entry} tp={tp} sl={sl} qty={self.contracts}")
            return
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
                # Log to DB
                self.db.log_trade(
                    trader_id='trader-s26-combine',
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    pnl=0.0, # Will be updated upon exit/fill
                    direction=direction,
                    entry_price=entry,
                    exit_price=0.0,
                    exit_reason="PENDING",
                    metadata={'contracts': self.contracts, 'order_data': str(data)}
                )
        except Exception as e:
            logger.error(f"Submit error: {e}")

if __name__ == "__main__":
    trader = S26CombineTrader()
    asyncio.run(trader.run())
