#!/usr/bin/env python3
"""Silver Bullet paper trader for Kraken Futures (PF_XBTUSD).

Ports the TradeStation MNQ paper trader to Kraken Futures with BTC-appropriate
kill zone windows. No ML filter. Credentials from .env.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytz
import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.models import DollarBar, SilverBulletSetup, SwingPoint
from src.detection.fvg_detection import detect_bearish_fvg, detect_bullish_fvg
from src.detection.liquidity_sweep_detection import (
    detect_bearish_liquidity_sweep,
    detect_bullish_liquidity_sweep,
)
from src.detection.silver_bullet_detection import detect_silver_bullet_setup
from src.detection.swing_detection import (
    RollingVolumeAverage,
    detect_bearish_mss,
    detect_bullish_mss,
    detect_swing_high,
    detect_swing_low,
)
from src.execution.kraken import KrakenFuturesClient
from src.execution.kraken.exceptions import KrakenAuthError, KrakenOrderError
from src.execution.kraken.models import KrakenBar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

BTC_TICK = 0.5  # PF_XBTUSD minimum price increment
_CHICAGO = pytz.timezone("America/Chicago")


def round_tick(price: float, tick: float = BTC_TICK) -> float:
    return round(round(price / tick) * tick, 10)


def _load_config(path: str = "config_kraken.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_kill_zones(kill_zones_cfg: list[dict]) -> list[tuple[str, int, int, int, int]]:
    """Parse kill zone config entries into (name, start_h, start_m, end_h, end_m) tuples.

    Times in config are America/Chicago local time. Comparison against current
    time is done by converting UTC → America/Chicago (handles CDT/CST automatically).
    """
    parsed = []
    for kz in kill_zones_cfg:
        name = kz["name"]
        s_h, s_m = map(int, kz["start"].split(":"))
        e_h, e_m = map(int, kz["end"].split(":"))
        parsed.append((name, s_h, s_m, e_h, e_m))
    return parsed


def _cdt_date(now_utc: datetime) -> str:
    """Return the America/Chicago calendar date string for a UTC timestamp."""
    return now_utc.astimezone(_CHICAGO).date().isoformat()


def _is_in_kill_zone(
    now_utc: datetime,
    kill_zones: list[tuple[str, int, int, int, int]],
) -> tuple[bool, str | None]:
    """Check whether now_utc falls inside any America/Chicago kill zone window."""
    local = now_utc.astimezone(_CHICAGO)
    cdt_minutes = local.hour * 60 + local.minute

    for name, s_h, s_m, e_h, e_m in kill_zones:
        start_min = s_h * 60 + s_m
        end_min = e_h * 60 + e_m
        if start_min <= cdt_minutes < end_min:
            return True, name

    return False, None


class KrakenSilverBulletTrader:
    """Silver Bullet paper trader for Kraken Futures (PF_XBTUSD)."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.symbol: str = config["symbol"]
        self.tick: float = config["tick"]
        self.position_size: int = int(config.get("position_size", 1))  # API lot count
        self.btc_size: float = float(config.get("btc_size", 1.0))    # BTC units for P&L
        self.rr_min: float = config["risk"]["rr_min"]
        self.daily_loss_limit: float = config["risk"]["daily_loss_limit"]
        self.kill_zones = _parse_kill_zones(config["kill_zones"])
        self.trade_days: list[int] | None = config.get("trade_days")  # 0=Mon…6=Sun; None = all days

        self.running = False

        # Detection state (mirrors paper_trade_winning_strategy.py)
        self.recent_bars: list[DollarBar] = []
        self.swing_highs: list[SwingPoint] = []
        self.swing_lows: list[SwingPoint] = []
        self.mss_events: list = []
        self.fvg_events: list = []
        self.volume_ma = RollingVolumeAverage(window=20)

        self.bar_index = 0
        self._is_preloading = False
        self._seen_setup_keys: set[tuple[int, int]] = set()

        # Execution state
        self.pending_setups: list[dict] = []
        self.active_trades: list[dict] = []
        self.window_trades: dict[str, set[str]] = {}

        # Performance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self._current_cdt_date = ""  # for daily_pnl reset

    # ------------------------------------------------------------------
    # Bar conversion: KrakenBar → DollarBar for pattern detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _kraken_bar_to_dollar_bar(bar: KrakenBar) -> DollarBar:
        return DollarBar(
            timestamp=bar.time,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=int(bar.volume),
            notional_value=bar.close * max(bar.volume, 1),
            is_forward_filled=False,
        )

    # ------------------------------------------------------------------
    # Pattern detection (identical logic to TradeStation paper trader)
    # ------------------------------------------------------------------

    def _find_next_liquidity_pool(self, direction: str, entry_price: float) -> float | None:
        start = max(0, len(self.recent_bars) - 200)
        candidates = []
        for i in range(len(self.recent_bars) - 1, start - 1, -1):
            bar = self.recent_bars[i]
            if direction == "bullish":
                left = self.recent_bars[i - 1].high if i > 0 else 0
                right = self.recent_bars[i + 1].high if i < len(self.recent_bars) - 1 else 0
                if bar.high > left and bar.high > right and bar.high > entry_price:
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

    def _detect_patterns(self, bar: DollarBar) -> list[SilverBulletSetup]:
        self.bar_index += 1
        self.volume_ma.update(bar.volume)

        idx = len(self.recent_bars) - 4
        if idx >= 3:
            if detect_swing_high(self.recent_bars, idx):
                self.swing_highs.append(SwingPoint(
                    timestamp=self.recent_bars[idx].timestamp,
                    price=self.recent_bars[idx].high,
                    swing_type="swing_high",
                    bar_index=self.bar_index - 4,
                ))
            if detect_swing_low(self.recent_bars, idx):
                self.swing_lows.append(SwingPoint(
                    timestamp=self.recent_bars[idx].timestamp,
                    price=self.recent_bars[idx].low,
                    swing_type="swing_low",
                    bar_index=self.bar_index - 4,
                ))

        self.swing_highs = self.swing_highs[-50:]
        self.swing_lows = self.swing_lows[-50:]

        bull_mss = detect_bullish_mss(bar, self.swing_highs, self.volume_ma.average)
        bear_mss = detect_bearish_mss(bar, self.swing_lows, self.volume_ma.average)
        if bull_mss:
            bull_mss.bar_index = self.bar_index
            self.mss_events.append(bull_mss)
        if bear_mss:
            bear_mss.bar_index = self.bar_index
            self.mss_events.append(bear_mss)
        self.mss_events = self.mss_events[-50:]

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

        sweeps = []
        if self.swing_lows:
            bs = detect_bullish_liquidity_sweep(self.recent_bars, curr_idx, self.swing_lows[-1])
            if bs:
                bs.bar_index = self.bar_index
                sweeps.append(bs)
        if self.swing_highs:
            bs2 = detect_bearish_liquidity_sweep(self.recent_bars, curr_idx, self.swing_highs[-1])
            if bs2:
                bs2.bar_index = self.bar_index
                sweeps.append(bs2)

        if self._is_preloading:
            return []

        setups = detect_silver_bullet_setup(
            mss_events=self.mss_events,
            fvg_events=self.fvg_events,
            sweep_events=sweeps,
        )

        new_setups = []
        for s in setups:
            key = (
                s.mss_event.bar_index if s.mss_event else -1,
                s.fvg_event.bar_index if s.fvg_event else -1,
            )
            if key not in self._seen_setup_keys:
                self._seen_setup_keys.add(key)
                new_setups.append(s)
                logger.info(
                    f"New setup: {s.direction} | confluence={s.confluence_count} | priority={s.priority}"
                )
        return new_setups

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def _submit_bracket(
        self,
        client: KrakenFuturesClient,
        direction: str,
        entry_price: float,
        tp_price: float,
        sl_price: float,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Submit 3 separate orders: entry limit, TP limit, SL stop.

        Returns (entry_id, tp_id, sl_id). Any may be None on failure.
        """
        entry_side = "buy" if direction == "bullish" else "sell"
        exit_side = "sell" if direction == "bullish" else "buy"

        entry_id = tp_id = sl_id = None
        placed: list[str] = []
        try:
            entry_id = await client.orders.place_order(
                symbol=self.symbol,
                side=entry_side,
                order_type="lmt",
                size=self.position_size,
                limit_price=entry_price,
            )
            placed.append(entry_id)
            tp_id = await client.orders.place_order(
                symbol=self.symbol,
                side=exit_side,
                order_type="lmt",
                size=self.position_size,
                limit_price=tp_price,
            )
            placed.append(tp_id)
            sl_id = await client.orders.place_order(
                symbol=self.symbol,
                side=exit_side,
                order_type="stp",
                size=self.position_size,
                stop_price=sl_price,
            )
            placed.append(sl_id)
            logger.info(
                f"Bracket submitted | entry={entry_id} | TP={tp_id} | SL={sl_id}"
            )
        except KrakenAuthError:
            raise
        except Exception as exc:
            logger.warning(f"Bracket leg failed: {exc} — rolling back {len(placed)} placed orders")
            for oid in placed:
                try:
                    await client.orders.cancel_order(oid)
                except Exception:
                    pass
            return None, None, None
        return entry_id, tp_id, sl_id

    # ------------------------------------------------------------------
    # Setup processing
    # ------------------------------------------------------------------

    async def process_trading_setup(
        self, setup: SilverBulletSetup, client: KrakenFuturesClient
    ) -> None:
        # 1. Kill zone filter
        within_kz, window_name = _is_in_kill_zone(datetime.now(timezone.utc), self.kill_zones)
        if not within_kz:
            return

        # 1b. Day-of-week filter (America/Chicago calendar day)
        if self.trade_days is not None:
            local_dow = datetime.now(timezone.utc).astimezone(_CHICAGO).weekday()  # 0=Mon
            if local_dow not in self.trade_days:
                logger.debug(f"DOW filter: weekday {local_dow} not in trade_days — skipping")
                return

        # 2. Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit:
            logger.warning("Daily loss limit reached — skipping new setups")
            return

        # 3. Window dedup: one trade per window per CDT calendar day
        today = _cdt_date(datetime.now(timezone.utc))
        if today in self.window_trades.get(window_name, set()):
            return
        already_pending = any(
            s["window"] == window_name and s["date"] == today
            for s in self.pending_setups
        )
        if already_pending:
            return

        # 4. Target calculation
        fvg_midpoint = round_tick((setup.entry_zone_top + setup.entry_zone_bottom) / 2)
        target_price = self._find_next_liquidity_pool(setup.direction, fvg_midpoint)
        if not target_price:
            logger.debug("No valid liquidity pool target — skipping")
            return
        target_price = round_tick(target_price)

        # 5. R:R enforcement
        fvg_gap = setup.entry_zone_top - setup.entry_zone_bottom
        if fvg_gap <= 0:
            logger.warning(f"Invalid FVG geometry (gap={fvg_gap:.1f}) — skipping setup")
            return
        if setup.direction == "bullish":
            stop_loss = round_tick(setup.entry_zone_bottom - 0.5 * fvg_gap)
        else:
            stop_loss = round_tick(setup.entry_zone_top + 0.5 * fvg_gap)

        risk = abs(fvg_midpoint - stop_loss)
        if risk == 0:
            return
        reward = abs(target_price - fvg_midpoint)
        rr = reward / risk

        if rr < self.rr_min:
            logger.info(f"Insufficient R:R ({rr:.2f} < {self.rr_min}) — skipping")
            return

        # 6. Submit bracket
        entry_id, tp_id, sl_id = await self._submit_bracket(
            client, setup.direction, fvg_midpoint, target_price, stop_loss
        )

        self.pending_setups.append({
            "setup": setup,
            "fvg_midpoint": fvg_midpoint,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "expiry_idx": self.bar_index + 15,
            "window": window_name,
            "date": today,
            "entry_id": entry_id,
            "tp_id": tp_id,
            "sl_id": sl_id,
        })

        logger.info(
            f"SETUP VALIDATED: {setup.direction.upper()} in {window_name} | "
            f"Entry={fvg_midpoint:.1f} Target={target_price:.1f} SL={stop_loss:.1f} R:R={rr:.2f}"
        )

    # ------------------------------------------------------------------
    # Main trading loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        logger.info("=" * 70)
        logger.info("Kraken Futures Silver Bullet Paper Trader")
        logger.info(f"Symbol: {self.symbol} | Tick: {self.tick} | API size: {self.position_size} contract(s) | BTC size: {self.btc_size} BTC")
        logger.info(f"Kill zones (CDT): {[(kz[0], kz[1], kz[3]) for kz in self.kill_zones]}")
        dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        days_str = ", ".join(dow_names[d] for d in self.trade_days) if self.trade_days is not None else "all"
        logger.info(f"Trade days: {days_str} | R:R min: {self.rr_min} | Daily loss limit: ${self.daily_loss_limit}")
        logger.info("=" * 70)

        self.running = True

        async with KrakenFuturesClient() as client:
            # Startup: fetch latest bars and log current price
            try:
                bars = await client.history.fetch_bars(self.symbol, interval="1m", count=2)
                if bars:
                    logger.info(f"{self.symbol} current price: {bars[-1].close:.1f}")
                else:
                    logger.warning("No bars returned on startup — proceeding anyway")
            except KrakenAuthError as exc:
                logger.error(f"AUTH FAILURE: {exc} — halting")
                return

            # Preload 300 bars of history for 200-bar lookback
            await self._preload_history(client)

            last_bar_ts = None

            while self.running:
                try:
                    bars = await client.history.fetch_bars(self.symbol, interval="1m", count=2)
                    if not bars:
                        await asyncio.sleep(15)
                        continue

                    bar_raw = bars[-1]

                    # Deduplicate bars
                    if bar_raw.time == last_bar_ts:
                        await asyncio.sleep(15)
                        continue
                    last_bar_ts = bar_raw.time

                    bar = self._kraken_bar_to_dollar_bar(bar_raw)
                    self.recent_bars.append(bar)
                    if len(self.recent_bars) > 300:
                        self.recent_bars.pop(0)

                    # Reset daily P&L at CDT day boundary
                    cdt_today = _cdt_date(datetime.now(timezone.utc))
                    if cdt_today != self._current_cdt_date:
                        self.daily_pnl = 0.0
                        self._current_cdt_date = cdt_today
                        logger.info(f"New CDT day {cdt_today} — daily P&L reset")

                    # Handle active trades (local sim P&L tracking)
                    # Check TP first — on same-bar crossings prefer the favorable outcome
                    for trade in self.active_trades[:]:
                        tp_hit = (
                            trade["direction"] == "bullish" and bar.high >= trade["target_price"]
                        ) or (
                            trade["direction"] == "bearish" and bar.low <= trade["target_price"]
                        )
                        sl_hit = (
                            trade["direction"] == "bullish" and bar.low <= trade["stop_loss"]
                        ) or (
                            trade["direction"] == "bearish" and bar.high >= trade["stop_loss"]
                        )

                        if tp_hit:
                            profit = abs(trade["target_price"] - trade["entry_price"]) * self.btc_size
                            self.total_pnl += profit
                            self.daily_pnl += profit
                            self.total_trades += 1
                            self.winning_trades += 1
                            logger.info(f"TAKE PROFIT HIT: {trade['direction'].upper()} | +${profit:.2f}")
                            if trade.get("sl_id"):
                                await client.orders.cancel_order(trade["sl_id"])
                            self.active_trades.remove(trade)
                        elif sl_hit:
                            loss = abs(trade["entry_price"] - trade["stop_loss"]) * self.btc_size
                            self.total_pnl -= loss
                            self.daily_pnl -= loss
                            self.total_trades += 1
                            logger.info(f"STOP LOSS HIT: {trade['direction'].upper()} | -${loss:.2f}")
                            if trade.get("tp_id"):
                                await client.orders.cancel_order(trade["tp_id"])
                            self.active_trades.remove(trade)

                    # Handle pending setups (expiry and fill detection)
                    for pending in self.pending_setups[:]:
                        if self.bar_index > pending["expiry_idx"]:
                            logger.info(f"Setup expired ({pending['setup'].direction}) — cancelling orders")
                            for oid in (pending.get("entry_id"), pending.get("tp_id"), pending.get("sl_id")):
                                if oid:
                                    await client.orders.cancel_order(oid)
                            self.pending_setups.remove(pending)
                            continue

                        is_touched = (
                            pending["setup"].direction == "bullish"
                            and bar.low <= pending["fvg_midpoint"]
                        ) or (
                            pending["setup"].direction == "bearish"
                            and bar.high >= pending["fvg_midpoint"]
                        )

                        if is_touched:
                            # Guard: fill bar also breaches stop — skip bad fill (matches backtest)
                            stop_breached = (
                                pending["setup"].direction == "bullish"
                                and bar.low <= pending["stop_loss"]
                            ) or (
                                pending["setup"].direction == "bearish"
                                and bar.high >= pending["stop_loss"]
                            )
                            if stop_breached:
                                logger.info(
                                    f"Fill-bar stop breach — skipping bad fill "
                                    f"({pending['setup'].direction} SL={pending['stop_loss']:.1f})"
                                )
                                self.pending_setups.remove(pending)
                                continue
                            logger.info(
                                f"TRADE ENTERED: {pending['setup'].direction.upper()} @ {pending['fvg_midpoint']:.1f}"
                            )
                            self.active_trades.append({
                                "entry_price": pending["fvg_midpoint"],
                                "target_price": pending["target_price"],
                                "stop_loss": pending["stop_loss"],
                                "direction": pending["setup"].direction,
                                "tp_id": pending.get("tp_id"),
                                "sl_id": pending.get("sl_id"),
                            })
                            self.window_trades.setdefault(pending["window"], set()).add(pending["date"])
                            self.pending_setups.remove(pending)

                    # Detect new setups
                    if len(self.recent_bars) >= 10:
                        new_setups = self._detect_patterns(bar)
                        for setup in new_setups:
                            await self.process_trading_setup(setup, client)

                    # Status log
                    wr = (self.winning_trades / self.total_trades * 100) if self.total_trades else 0
                    logger.info(
                        f"{self.symbol} @ {bar.close:.1f} [{bar.timestamp.strftime('%H:%M')}Z] | "
                        f"Trades={self.total_trades} WR={wr:.1f}% P&L=${self.total_pnl:.2f} | "
                        f"Pending={len(self.pending_setups)} Active={len(self.active_trades)}"
                    )

                    await asyncio.sleep(15)

                except KrakenAuthError as exc:
                    logger.error(f"AUTH FAILURE: {exc} — halting")
                    self.running = False
                    break
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Loop error: {exc}", exc_info=True)
                    await asyncio.sleep(5)

    async def _preload_history(self, client: KrakenFuturesClient) -> None:
        """Warm up the 200-bar lookback using recent 1-minute bars."""
        logger.info("Preloading 300 bars for lookback warmup...")
        try:
            # Fetch in batches of 50 (Kraken charts range limit)
            # We'll fetch the last 5 minutes × 60 bars = 300 bars by adjusting `from`
            import time as _time
            now_ms = int(_time.time() * 1000)
            from_ms = now_ms - 300 * 60_000  # 300 minutes ago

            import httpx as _httpx
            from src.execution.kraken.market_data.history import CHARTS_BASE
            preload_url = f"{CHARTS_BASE}/{self.symbol}/1m"
            async with _httpx.AsyncClient() as tmp:
                response = await tmp.get(
                    preload_url,
                    params={"from": from_ms // 1000, "to": now_ms // 1000},
                    timeout=30.0,
                )
            if response.status_code != 200:
                logger.warning(f"Preload HTTP {response.status_code} — starting cold")
                return
            candles = response.json().get("candles", [])
            from src.execution.kraken.models import KrakenBar as _KB
            loaded = 0
            self._is_preloading = True
            try:
                for c in candles:
                    try:
                        kb = _KB.from_candle(c)
                        db = self._kraken_bar_to_dollar_bar(kb)
                        self.recent_bars.append(db)
                        if len(self.recent_bars) > 300:
                            self.recent_bars.pop(0)
                        if len(self.recent_bars) >= 10:
                            self._detect_patterns(db)
                        loaded += 1
                    except Exception:
                        continue
            finally:
                self._is_preloading = False
            logger.info(f"Preloaded {loaded} bars — lookback ready")
        except Exception as exc:
            logger.warning(f"Preload error: {exc} — starting cold")

    def stop(self) -> None:
        logger.info("Stopping Kraken paper trader...")
        self.running = False
        wr = (self.winning_trades / self.total_trades * 100) if self.total_trades else 0
        logger.info("=" * 70)
        logger.info(f"FINAL | Trades={self.total_trades} WR={wr:.1f}% P&L=${self.total_pnl:.2f}")
        logger.info("=" * 70)


async def main() -> None:
    config = _load_config("config_kraken.yaml")
    trader = KrakenSilverBulletTrader(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, trader.stop)

    try:
        await trader.run()
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
    finally:
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
