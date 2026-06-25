#!/usr/bin/env python3
"""
Kraken Thursday Short — BTC + ETH day-of-week strategy.

Empirical finding: BTC and ETH perpetual futures on Kraken return significantly
negative returns on Thursdays (UTC 00:00 – 23:59). See backtest_dow_effect.py.

  OOS (Oct 2025 – Jun 2026, N=32/36):
    BTC Thu Short: Sharpe +2.95, WR 62%, mean +120 bps/week
    ETH Thu Short: Sharpe +4.34, WR 75%, mean +186 bps/week

Strategy:
  - Short BTC and ETH at Thursday 00:00 UTC
  - Close at Thursday 23:05 UTC
  - Stop: if position moves >5% against entry, close immediately
  - Emergency exit: any position still open at Friday 00:05 UTC

Usage:
  .venv/bin/python kraken_thursday_short.py             # paper simulation
  .venv/bin/python kraken_thursday_short.py --live      # live Kraken Futures perps
  .venv/bin/python kraken_thursday_short.py --margin    # live Kraken Spot margin

Paper mode:  simulates orders and logs P&L without connecting to Kraken.
Live mode:   requires KRAKEN_FUTURES_API_KEY + KRAKEN_FUTURES_API_SECRET in env.
             Instrument: PF_XBTUSD / PF_ETHUSD perps on Kraken Futures.
Margin mode: requires KRAKEN_API_KEY + KRAKEN_API_SECRET in env (Spot keys).
             Instrument: XBTUSD / ETHUSD spot via margin borrow @ 2x leverage.
             Borrowing cost: ~0.06% per 23h hold (~$3 on $5K notional).

Pre-registered: see _bmad-output/preregistration_kraken_thursday_short.md
Live sizing:    see _bmad-output/preregistration_thu_short_live_deploy.md
Decision rule:  PASS if Sharpe > 0.80 after N >= 30 prospective Thursdays
"""
import asyncio
import base64
import csv
import hashlib
import hmac
import logging
import os
import signal
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Under systemd stdout is already redirected to the log file — only one handler
# to avoid double-writing every line.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Strategy configuration
# ──────────────────────────────────────────────
BTC_SIZE    = float(os.environ.get("THU_BTC_SIZE",  "0.039"))  # ≈$2,500 @ $64K
ETH_SIZE    = float(os.environ.get("THU_ETH_SIZE",  "1.4"))    # ≈$2,500 @ $1,728
STOP_PCT    = float(os.environ.get("THU_STOP_PCT",  "5.0"))    # % against position
LOG_FILE    = Path("logs/kraken_thursday_short.csv")
LIVE        = "--live"   in sys.argv
MARGIN      = "--margin" in sys.argv
TSSIM       = "--tssim"  in sys.argv  # CME micro futures MBT/MET on TradeStation SIM
TS_SIM_ACCOUNT = os.environ.get("THU_TS_SIM_ACCOUNT", "SIM2797251F")
MBT_SIZE    = int(os.environ.get("THU_MBT_SIZE", "1"))  # BTC-leg contracts; MET sized to match notional
ALARM_FILE  = Path("data/thursday_ts/ALARM")

ENTRY_HOUR   = 0     # 00:00 UTC Thursday
EXIT_HOUR    = 23
EXIT_MINUTE  = 5     # 23:05 UTC

# Futures (--live)
BTC_SYMBOL    = "PF_XBTUSD"
ETH_SYMBOL    = "PF_ETHUSD"
TICKERS_URL   = "https://futures.kraken.com/derivatives/api/v3/tickers"

# Spot margin (--margin)
BTC_SPOT_PAIR = "XBTUSD"
ETH_SPOT_PAIR = "ETHUSD"
SPOT_BASE_URL = "https://api.kraken.com"


# ──────────────────────────────────────────────
# Public futures price fetcher — no auth needed
# ──────────────────────────────────────────────
async def fetch_mark_price(symbol: str) -> float:
    """Get current mark price via public Kraken Futures tickers endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.get(TICKERS_URL)
        r.raise_for_status()
        for t in r.json().get("tickers", []):
            if t["symbol"] == symbol:
                return float(t["markPrice"])
    raise RuntimeError(f"Symbol {symbol} not found in tickers")


# ──────────────────────────────────────────────
# Kraken Spot margin client — used by --margin
# ──────────────────────────────────────────────
class KrakenSpotMarginClient:
    """Thin async client for Kraken Spot margin orders.

    Auth: HMAC-SHA512 per Kraken REST API docs.
    Reads KRAKEN_API_KEY + KRAKEN_API_SECRET from environment.
    Borrowing cost: 0.01% open + 0.01% per 4h rollover (~0.06% for 23h hold).
    """

    def __init__(self) -> None:
        self.api_key    = os.environ["KRAKEN_API_KEY"]
        self.api_secret = os.environ["KRAKEN_API_SECRET"]

    def _sign(self, urlpath: str, data: dict) -> str:
        postdata = urllib.parse.urlencode(data)
        encoded  = (str(data["nonce"]) + postdata).encode()
        message  = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    async def get_price(self, pair: str) -> float:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{SPOT_BASE_URL}/0/public/Ticker", params={"pair": pair})
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                raise RuntimeError(f"Ticker error: {body['error']}")
            result = body["result"]
            key = next(iter(result))
            return float(result[key]["c"][0])  # last trade price

    async def place_order(self, pair: str, side: str, volume: float, leverage: int = 2) -> str:
        nonce = str(int(time.time() * 1000))
        data = {
            "nonce":     nonce,
            "ordertype": "market",
            "type":      side,       # "buy" or "sell"
            "volume":    str(volume),
            "pair":      pair,
            "leverage":  str(leverage),
        }
        urlpath = "/0/private/AddOrder"
        headers = {
            "API-Key":      self.api_key,
            "API-Sign":     self._sign(urlpath, data),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.post(
                f"{SPOT_BASE_URL}{urlpath}",
                content=urllib.parse.urlencode(data),
                headers=headers,
            )
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                raise RuntimeError(f"Order error: {body['error']}")
            return body["result"]["txid"][0]


# ──────────────────────────────────────────────
# Futures live helpers — only used by --live
# ──────────────────────────────────────────────
async def _live_get_price(client, symbol: str) -> float:
    bars = await client.history.fetch_bars(symbol, interval="1m", count=2)
    if not bars:
        raise RuntimeError(f"No bars for {symbol}")
    return float(bars[-1].close)


async def _live_place_order(client, symbol: str, side: str, size: float) -> str:
    return await client.orders.place_order(
        symbol=symbol, side=side, order_type="mkt", size=size
    )


def fetch_btc_lr_slopes():
    """BTC LR-channel slope (bps/day) for regime tagging — read-only, firewalled.

    Returns (slope_len20, slope_len40) in bps/day from completed daily BTC closes
    through the prior day (the regime known at the Thursday 00:00 UTC entry), or
    (None, None) on any failure. Instrumentation for prospective evaluation of the
    LR-down regime gate; does NOT influence trading (pre-registration-safe).
    """
    try:
        import numpy as np
        from src.research.lr_channel import compute_lr_channel
        with httpx.Client(timeout=10.0) as c:
            r = c.get("https://api.kraken.com/0/public/OHLC",
                      params={"pair": "XBTUSD", "interval": 1440})
            r.raise_for_status()
            body = r.json()
        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        result = body["result"]
        key = next(k for k in result if k != "last")
        closes = np.array([float(c[4]) for c in result[key][:-1]], dtype=float)
        if len(closes) < 45:
            return None, None
        last = closes[-1]
        s20 = float(compute_lr_channel(closes, 20).slope[-1] / last * 1e4)
        s40 = float(compute_lr_channel(closes, 40).slope[-1] / last * 1e4)
        return s20, s40
    except Exception as e:  # firewall — logging tag must never disrupt trading
        logger.warning("LR-slope regime tag failed (logging only): %s", e)
        return None, None


class ChainedCsv:
    """Tamper-evident hash-chained CSV (local copy of the MIM-NB pattern)."""

    def __init__(self, path, fields):
        self.path = path
        self.fields = list(fields) + ["chain"]
        self.head = "GENESIS"
        if path.exists():
            try:
                with open(path) as f:
                    for row in csv.DictReader(f):
                        self.head = row.get("chain", self.head)
            except Exception:
                logger.warning("chain reload failed for %s — restarting chain", path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fields).writeheader()

    def append(self, row):
        payload = "|".join(str(row.get(k, "")) for k in self.fields if k != "chain")
        self.head = hashlib.sha256((self.head + "|" + payload).encode()).hexdigest()[:16]
        row = dict(row); row["chain"] = self.head
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)


class ThursdayShortTrader:
    """Weekly Thursday short on BTC and ETH.

    Paper mode:  simulates fills at mark price, logs as PAPER.
    Live mode:   submits real market orders to Kraken Futures perps.
    Margin mode: submits real market orders via Kraken Spot margin.
    """

    def __init__(self) -> None:
        self.running       = False
        self.btc_entry: Optional[float] = None
        self.eth_entry: Optional[float] = None
        self.btc_order_id: Optional[str] = None
        self.eth_order_id: Optional[str] = None
        self.position_open = False
        self.current_thursday: Optional[str] = None
        self.total_pnl_usd = 0.0
        self.n_trades = 0
        self.n_wins   = 0
        # LR-regime tag cache (computed once per Thursday at first log write).
        # Instrumentation only — does NOT affect entries/exits (pre-reg-safe).
        self._lr_cache_date: Optional[str] = None
        self._lr20: Optional[float] = None
        self._lr40: Optional[float] = None
        # TradeStation SIM (--tssim) per-Thursday execution state
        self.ts_mbt_sym: Optional[str] = None
        self.ts_met_sym: Optional[str] = None
        self.ts_n_mbt = 0
        self.ts_n_met = 0
        self.ts_entry_attempted_date: Optional[str] = None
        self.ts_entry_time = ""
        self._trades_log = None       # ChainedCsv data/thursday_ts/trades.csv (--tssim)
        self._decisions_log = None    # ChainedCsv data/thursday_ts/decisions.csv (--tssim)

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        _hdr = ["date", "asset", "entry_price", "exit_price",
                "size", "direction", "gross_ret_bps", "pnl_usd",
                "exit_reason", "mode", "lr_slope20_bpd", "lr_slope40_bpd"]
        _need_header = True
        if LOG_FILE.exists():
            _lines = LOG_FILE.read_text().splitlines()
            if _lines and _lines[0] == ",".join(_hdr):
                _need_header = False                 # already migrated
            elif len(_lines) > 1:
                _need_header = False                 # data under old header — don't clobber
        if _need_header:
            with LOG_FILE.open("w", newline="") as f:
                csv.writer(f).writerow(_hdr)

    # ── Price fetching ────────────────────────────────────────────────────────

    async def _get_price(self, symbol: str, client=None) -> float:
        if LIVE and client is not None:
            return await _live_get_price(client, symbol)
        return await fetch_mark_price(symbol)

    # ── Entry/exit — Futures (paper + --live) ─────────────────────────────────

    async def _enter_short(self, client=None) -> None:
        mode = "LIVE" if LIVE else "PAPER"
        logger.info("=" * 60)
        logger.info(f"THURSDAY SHORT — ENTERING SHORT POSITIONS [{mode}]")
        logger.info(f"  BTC: SELL {BTC_SIZE} {BTC_SYMBOL}")
        logger.info(f"  ETH: SELL {ETH_SIZE} {ETH_SYMBOL}")
        logger.info("=" * 60)

        btc_price = await self._get_price(BTC_SYMBOL, client)
        eth_price = await self._get_price(ETH_SYMBOL, client)

        if LIVE and client is not None:
            try:
                self.btc_order_id = await _live_place_order(client, BTC_SYMBOL, "sell", BTC_SIZE)
                self.btc_entry = btc_price
                logger.info(f"BTC SHORT LIVE: entry≈{btc_price:.2f} order={self.btc_order_id}")
            except Exception as e:
                logger.error(f"BTC short entry FAILED: {e}")
                self.btc_order_id = None

            try:
                self.eth_order_id = await _live_place_order(client, ETH_SYMBOL, "sell", ETH_SIZE)
                self.eth_entry = eth_price
                logger.info(f"ETH SHORT LIVE: entry≈{eth_price:.2f} order={self.eth_order_id}")
            except Exception as e:
                logger.error(f"ETH short entry FAILED: {e}")
                self.eth_order_id = None
        else:
            self.btc_entry = btc_price
            self.eth_entry = eth_price
            btc_notional = btc_price * BTC_SIZE
            eth_notional = eth_price * ETH_SIZE
            logger.info(f"BTC SHORT PAPER: entry={btc_price:.2f} notional≈${btc_notional:.0f}")
            logger.info(f"ETH SHORT PAPER: entry={eth_price:.2f} notional≈${eth_notional:.0f}")

        self.position_open = True
        self.current_thursday = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _exit_short(self, reason: str, client=None) -> None:
        mode = "LIVE" if LIVE else "PAPER"
        logger.info(f"THURSDAY SHORT — EXITING ({reason}) [{mode}]")

        for symbol, entry, size, label in [
            (BTC_SYMBOL, self.btc_entry, BTC_SIZE, "BTC"),
            (ETH_SYMBOL, self.eth_entry, ETH_SIZE, "ETH"),
        ]:
            if entry is None:
                continue
            try:
                exit_price = await self._get_price(symbol, client)
                if LIVE and client is not None:
                    await _live_place_order(client, symbol, "buy", size)
                ret_bps = (entry - exit_price) / entry * 10_000
                pnl = (entry - exit_price) * size
                self.total_pnl_usd += pnl
                self.n_trades += 1
                if ret_bps > 0:
                    self.n_wins += 1
                logger.info(
                    f"{label} EXIT [{mode}]: {entry:.2f}→{exit_price:.2f} "
                    f"{ret_bps:+.1f}bps ${pnl:+.2f}"
                )
                self._log_trade(label, entry, exit_price, size, ret_bps, pnl, reason, mode)
            except Exception as e:
                logger.error(f"{label} exit FAILED: {e}")

        self._clear_position()

    # ── Entry/exit — Spot margin (--margin) ───────────────────────────────────

    async def _enter_margin(self, client: KrakenSpotMarginClient) -> None:
        logger.info("=" * 60)
        logger.info("THURSDAY SHORT — ENTERING SHORT POSITIONS [MARGIN]")
        logger.info(f"  BTC: SELL {BTC_SIZE} {BTC_SPOT_PAIR} (leverage=2)")
        logger.info(f"  ETH: SELL {ETH_SIZE} {ETH_SPOT_PAIR} (leverage=2)")
        logger.info("=" * 60)

        btc_price = await client.get_price(BTC_SPOT_PAIR)
        eth_price = await client.get_price(ETH_SPOT_PAIR)

        try:
            self.btc_order_id = await client.place_order(BTC_SPOT_PAIR, "sell", BTC_SIZE)
            self.btc_entry = btc_price
            logger.info(f"BTC SHORT MARGIN: entry≈{btc_price:.2f} txid={self.btc_order_id}")
        except Exception as e:
            logger.error(f"BTC margin short entry FAILED: {e}")
            self.btc_order_id = None

        try:
            self.eth_order_id = await client.place_order(ETH_SPOT_PAIR, "sell", ETH_SIZE)
            self.eth_entry = eth_price
            logger.info(f"ETH SHORT MARGIN: entry≈{eth_price:.2f} txid={self.eth_order_id}")
        except Exception as e:
            logger.error(f"ETH margin short entry FAILED: {e}")
            self.eth_order_id = None

        self.position_open = True
        self.current_thursday = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def _exit_margin(self, reason: str, client: KrakenSpotMarginClient) -> None:
        logger.info(f"THURSDAY SHORT — EXITING ({reason}) [MARGIN]")

        for pair, entry, size, label in [
            (BTC_SPOT_PAIR, self.btc_entry, BTC_SIZE, "BTC"),
            (ETH_SPOT_PAIR, self.eth_entry, ETH_SIZE, "ETH"),
        ]:
            if entry is None:
                continue
            try:
                exit_price = await client.get_price(pair)
                await client.place_order(pair, "buy", size)  # close margin short
                ret_bps = (entry - exit_price) / entry * 10_000
                pnl = (entry - exit_price) * size
                self.total_pnl_usd += pnl
                self.n_trades += 1
                if ret_bps > 0:
                    self.n_wins += 1
                logger.info(
                    f"{label} EXIT [MARGIN]: {entry:.2f}→{exit_price:.2f} "
                    f"{ret_bps:+.1f}bps ${pnl:+.2f}"
                )
                self._log_trade(label, entry, exit_price, size, ret_bps, pnl, reason, "MARGIN")
            except Exception as e:
                logger.error(f"{label} margin exit FAILED: {e}")

        self._clear_position()

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _check_stop(self, btc_price: float, eth_price: float) -> bool:
        if self.btc_entry is not None:
            pct_move = (btc_price - self.btc_entry) / self.btc_entry * 100
            if pct_move > STOP_PCT:
                logger.warning(
                    f"BTC STOP HIT: entry={self.btc_entry:.2f} "
                    f"current={btc_price:.2f} +{pct_move:.2f}%"
                )
                return True
        if self.eth_entry is not None:
            pct_move = (eth_price - self.eth_entry) / self.eth_entry * 100
            if pct_move > STOP_PCT:
                logger.warning(
                    f"ETH STOP HIT: entry={self.eth_entry:.2f} "
                    f"current={eth_price:.2f} +{pct_move:.2f}%"
                )
                return True
        return False

    def _log_trade(self, asset: str, entry: float, exit_price: float,
                   size: float, ret_bps: float, pnl: float,
                   reason: str, mode: str) -> None:
        # Tag with the BTC LR-channel regime at entry (slope from completed daily
        # closes through the prior day). Computed once per Thursday and cached;
        # firewalled so a fetch failure never blocks logging the realized fill.
        if self._lr_cache_date != self.current_thursday:
            self._lr20, self._lr40 = fetch_btc_lr_slopes()
            self._lr_cache_date = self.current_thursday
        with LOG_FILE.open("a", newline="") as f:
            csv.writer(f).writerow([
                self.current_thursday, asset, entry, exit_price,
                size, "short", f"{ret_bps:.2f}", f"{pnl:.4f}", reason, mode,
                "" if self._lr20 is None else f"{self._lr20:.3f}",
                "" if self._lr40 is None else f"{self._lr40:.3f}",
            ])

    def _clear_position(self) -> None:
        self.position_open = False
        self.btc_entry = self.eth_entry = None
        self.btc_order_id = self.eth_order_id = None
        self.ts_mbt_sym = self.ts_met_sym = None
        self.ts_n_mbt = self.ts_n_met = 0
        wr = 100 * self.n_wins / self.n_trades if self.n_trades > 0 else 0
        logger.info(f"Session P&L: ${self.total_pnl_usd:+.2f} | Trades: {self.n_trades} | WR: {wr:.0f}%")

    # ── Main loops ────────────────────────────────────────────────────────────

    async def _run_paper(self) -> None:
        """Paper mode loop — no Kraken auth required."""
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                dow = now.weekday()      # 0=Mon … 3=Thu … 6=Sun
                hour, minute = now.hour, now.minute
                today = now.strftime("%Y-%m-%d")
                day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                if (dow == 3 and hour == ENTRY_HOUR and minute < 5
                        and not self.position_open
                        and self.current_thursday != today):
                    await self._enter_short()

                elif (self.position_open and dow == 3
                      and hour == EXIT_HOUR and minute >= EXIT_MINUTE):
                    await self._exit_short("scheduled_exit")

                elif self.position_open and dow == 4:
                    logger.warning("Open position on Friday — emergency close [PAPER]")
                    await self._exit_short("emergency_friday")

                elif self.position_open:
                    btc_p = await fetch_mark_price(BTC_SYMBOL)
                    eth_p = await fetch_mark_price(ETH_SYMBOL)
                    if self._check_stop(btc_p, eth_p):
                        await self._exit_short("stop_loss")
                    else:
                        btc_bps = (self.btc_entry - btc_p) / self.btc_entry * 10_000 if self.btc_entry else 0
                        eth_bps = (self.eth_entry - eth_p) / self.eth_entry * 10_000 if self.eth_entry else 0
                        logger.info(
                            f"PAPER BTC {btc_bps:+.1f}bps ({btc_p:.2f}) "
                            f"ETH {eth_bps:+.1f}bps ({eth_p:.2f}) "
                            f"| {day_names[dow]} {now.strftime('%H:%M UTC')}"
                        )
                else:
                    logger.info(
                        f"Waiting [{day_names[dow]} {now.strftime('%H:%M UTC')}] "
                        f"— next trade: next Thursday 00:00 UTC"
                    )

                await asyncio.sleep(60 if self.position_open else 300)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Loop error: {exc}", exc_info=True)
                await asyncio.sleep(30)

    async def _run_live(self) -> None:
        """Live futures mode loop — requires Kraken Futures API credentials."""
        from src.execution.kraken import KrakenFuturesClient

        async with KrakenFuturesClient(live=True) as client:
            while self.running:
                try:
                    now = datetime.now(timezone.utc)
                    dow = now.weekday()
                    hour, minute = now.hour, now.minute
                    today = now.strftime("%Y-%m-%d")
                    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                    if (dow == 3 and hour == ENTRY_HOUR and minute < 5
                            and not self.position_open
                            and self.current_thursday != today):
                        await self._enter_short(client)

                    elif (self.position_open and dow == 3
                          and hour == EXIT_HOUR and minute >= EXIT_MINUTE):
                        await self._exit_short("scheduled_exit", client)

                    elif self.position_open and dow == 4:
                        logger.warning("Open position on Friday — emergency close [LIVE]")
                        await self._exit_short("emergency_friday", client)

                    elif self.position_open:
                        btc_p = await _live_get_price(client, BTC_SYMBOL)
                        eth_p = await _live_get_price(client, ETH_SYMBOL)
                        if self._check_stop(btc_p, eth_p):
                            await self._exit_short("stop_loss", client)
                        else:
                            btc_bps = (self.btc_entry - btc_p) / self.btc_entry * 10_000 if self.btc_entry else 0
                            eth_bps = (self.eth_entry - eth_p) / self.eth_entry * 10_000 if self.eth_entry else 0
                            logger.info(
                                f"LIVE BTC {btc_bps:+.1f}bps ({btc_p:.2f}) "
                                f"ETH {eth_bps:+.1f}bps ({eth_p:.2f}) "
                                f"| {day_names[dow]} {now.strftime('%H:%M UTC')}"
                            )
                    else:
                        logger.info(
                            f"Waiting [{day_names[dow]} {now.strftime('%H:%M UTC')}] "
                            f"— next trade: next Thursday 00:00 UTC"
                        )

                    await asyncio.sleep(60 if self.position_open else 300)

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Loop error: {exc}", exc_info=True)
                    await asyncio.sleep(30)

        if self.position_open:
            logger.warning("Shutdown with open position — attempting close [LIVE]")
            async with KrakenFuturesClient(live=True) as c2:
                await self._exit_short("shutdown", c2)

    async def _run_margin(self) -> None:
        """Live spot margin mode loop — requires Kraken Spot API credentials."""
        client = KrakenSpotMarginClient()
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                dow = now.weekday()
                hour, minute = now.hour, now.minute
                today = now.strftime("%Y-%m-%d")
                day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                if (dow == 3 and hour == ENTRY_HOUR and minute < 5
                        and not self.position_open
                        and self.current_thursday != today):
                    await self._enter_margin(client)

                elif (self.position_open and dow == 3
                      and hour == EXIT_HOUR and minute >= EXIT_MINUTE):
                    await self._exit_margin("scheduled_exit", client)

                elif self.position_open and dow == 4:
                    logger.warning("Open position on Friday — emergency close [MARGIN]")
                    await self._exit_margin("emergency_friday", client)

                elif self.position_open:
                    btc_p = await client.get_price(BTC_SPOT_PAIR)
                    eth_p = await client.get_price(ETH_SPOT_PAIR)
                    if self._check_stop(btc_p, eth_p):
                        await self._exit_margin("stop_loss", client)
                    else:
                        btc_bps = (self.btc_entry - btc_p) / self.btc_entry * 10_000 if self.btc_entry else 0
                        eth_bps = (self.eth_entry - eth_p) / self.eth_entry * 10_000 if self.eth_entry else 0
                        logger.info(
                            f"MARGIN BTC {btc_bps:+.1f}bps ({btc_p:.2f}) "
                            f"ETH {eth_bps:+.1f}bps ({eth_p:.2f}) "
                            f"| {day_names[dow]} {now.strftime('%H:%M UTC')}"
                        )
                else:
                    logger.info(
                        f"Waiting [{day_names[dow]} {now.strftime('%H:%M UTC')}] "
                        f"— next trade: next Thursday 00:00 UTC"
                    )

                await asyncio.sleep(60 if self.position_open else 300)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Loop error: {exc}", exc_info=True)
                await asyncio.sleep(30)

        if self.position_open:
            logger.warning("Shutdown with open position — attempting close [MARGIN]")
            await self._exit_margin("shutdown", client)

    # ── TradeStation SIM (--tssim): MBT/MET micro futures ─────────────────

    def _alarm(self, reason: str) -> None:
        """Hard integrity alarm: write the ALARM flag + ERROR. Startup refuses to
        run while the flag exists (forces human review after any breach)."""
        try:
            ALARM_FILE.parent.mkdir(parents=True, exist_ok=True)
            ALARM_FILE.write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n")
        except Exception as e:
            logger.error(f"failed to write ALARM flag: {e}")
        logger.error(f"🚨 TSSIM ALARM: {reason}")

    @staticmethod
    def _is_short(p: dict) -> bool:
        return str(p.get("LongShort", "")).lower().startswith("s")

    async def _confirm_positions(self, client, want_syms, timeout: int = 25) -> list:
        """Poll broker positions until all want_syms are present (or timeout)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            pos = await client.get_open_positions()
            if all(any(p.get("Symbol") == s for p in pos) for s in want_syms):
                return pos
            await asyncio.sleep(2)
        return await client.get_open_positions()

    def _log_decision(self, action: str, detail: str = "", mbt=None, met=None,
                      pb=None, pe=None, n_mbt=0, n_met=0) -> None:
        """One row per Thursday entry decision (ENTERED/REJECTED/SKIPPED/...),
        with the LR-regime tags — feeds the prospective N count + LR-gate subset."""
        if self._decisions_log is None:
            return
        self._decisions_log.append({
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "thursday": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "mbt_sym": mbt or "", "met_sym": met or "",
            "mark_btc": pb or "", "mark_eth": pe or "",
            "n_mbt": n_mbt, "n_met": n_met,
            "lr_slope20_bpd": "" if self._lr20 is None else round(self._lr20, 3),
            "lr_slope40_bpd": "" if self._lr40 is None else round(self._lr40, 3),
            "action": action, "detail": detail,
        })

    async def _enter_tssim(self, client) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.ts_entry_attempted_date = today  # G3: record attempt regardless of outcome
        self._lr20, self._lr40 = fetch_btc_lr_slopes()  # regime tag at entry
        self._lr_cache_date = today
        logger.info("=" * 60)
        logger.info("THURSDAY SHORT — ENTERING SHORT POSITIONS [TSSIM]")

        pre = await client.get_open_positions()           # G1 pre-trade reconcile
        if pre:
            self._alarm(f"unexpected open position before entry: {[p.get('Symbol') for p in pre]}")
            self._log_decision("SKIPPED_NOT_FLAT", str([p.get('Symbol') for p in pre]))
            return

        front = await client.resolve_front_month()
        mbt, met = front["MBT"], front["MET"]
        pb = await client.last_price(mbt)
        pe = await client.last_price(met)
        if not pb or not pe:
            self._alarm(f"no marks for {mbt}/{met}")
            self._log_decision("NO_MARKS", f"{mbt}/{met}", mbt, met)
            return
        n_mbt = MBT_SIZE
        n_met = max(1, round(n_mbt * pb / pe))            # equal-notional (~1:38)
        logger.info(f"  {n_mbt} {mbt} (~${n_mbt*pb*0.1:,.0f}) + {n_met} {met} (~${n_met*pe*0.1:,.0f}) short")

        o1 = await client.place_order(mbt, "SELL", n_mbt)
        o2 = await client.place_order(met, "SELL", n_met)

        pos = await self._confirm_positions(client, [mbt, met])   # G1: broker truth
        held = {p.get("Symbol"): p for p in pos}
        ok = (mbt in held and met in held and self._is_short(held[mbt]) and self._is_short(held[met]))
        if not ok:
            self._alarm(f"entry NOT confirmed (mbt#{o1} met#{o2}); broker shows {[p.get('Symbol') for p in pos]}")
            for p in pos:                                  # flatten any partial leg
                act = "BUY" if self._is_short(p) else "SELL"
                await client.place_order(p["Symbol"], act, abs(int(float(p["Quantity"]))))
            self._log_decision("REJECTED", f"mbt#{o1} met#{o2}", mbt, met, pb, pe, n_mbt, n_met)
            return                                         # position_open stays False (anti-phantom)

        self.btc_entry = float(held[mbt].get("AveragePrice") or pb)
        self.eth_entry = float(held[met].get("AveragePrice") or pe)
        self.ts_mbt_sym, self.ts_met_sym = mbt, met
        self.ts_n_mbt, self.ts_n_met = n_mbt, n_met
        self.ts_entry_time = datetime.now(timezone.utc).strftime("%H:%M")
        self.position_open = True
        self.current_thursday = today
        self._log_decision("ENTERED", "", mbt, met, self.btc_entry, self.eth_entry, n_mbt, n_met)
        logger.info(f"✓ TSSIM SHORT confirmed | {mbt} @ {self.btc_entry} | {met} @ {self.eth_entry}")

    async def _exit_tssim(self, reason: str, client) -> None:
        logger.info(f"THURSDAY SHORT — EXITING ({reason}) [TSSIM]")
        exit_t = datetime.now(timezone.utc).strftime("%H:%M")
        for sym, entry, qty, label in [
            (self.ts_mbt_sym, self.btc_entry, self.ts_n_mbt, "MBT"),
            (self.ts_met_sym, self.eth_entry, self.ts_n_met, "MET"),
        ]:
            if not sym or entry is None:
                continue
            try:
                exit_price = await client.last_price(sym)
                await client.place_order(sym, "BUY", qty)            # close short
                ret_bps = (entry - exit_price) / entry * 10_000
                pnl = (entry - exit_price) * qty * 0.1               # micro = 0.1 underlying
                self.total_pnl_usd += pnl
                self.n_trades += 1
                if ret_bps > 0:
                    self.n_wins += 1
                logger.info(f"{label} {sym} EXIT [TSSIM]: {entry:.2f}→{exit_price:.2f} {ret_bps:+.1f}bps ${pnl:+.2f}")
                if self._trades_log is not None:
                    self._trades_log.append({
                        "thursday": self.current_thursday or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        "symbol": sym, "dir": "short", "entry_t": self.ts_entry_time, "entry_px": entry,
                        "exit_t": exit_t, "exit_px": exit_price, "qty": qty,
                        "ret_bps": round(ret_bps, 2), "pnl_usd": round(pnl, 2), "reason": reason,
                        "lr_slope20_bpd": "" if self._lr20 is None else round(self._lr20, 3),
                        "lr_slope40_bpd": "" if self._lr40 is None else round(self._lr40, 3),
                    })
            except Exception as e:
                logger.error(f"{label} TSSIM exit FAILED: {e}")
        pos = await client.get_open_positions()                     # confirm flat
        t0 = time.time()
        while pos and time.time() - t0 < 25:
            await asyncio.sleep(2); pos = await client.get_open_positions()
        if pos:
            self._alarm(f"NOT flat after exit: {[p.get('Symbol') for p in pos]}")
        self._clear_position()

    async def _run_tssim(self) -> None:
        """TradeStation SIM mode — MBT/MET micro futures, broker-confirmed state."""
        from src.data.auth_v3 import TradeStationAuthV3
        from src.research.ts_thursday_client import TradeStationThursdayClient
        if ALARM_FILE.exists():
            raise SystemExit(f"ALARM flag present ({ALARM_FILE}); review and remove to resume.")
        auth = TradeStationAuthV3.from_file(".access_token")
        await auth.authenticate()
        await auth.start_auto_refresh()                              # keep token fresh (d4c0c39)
        async with httpx.AsyncClient(timeout=30) as http:
            client = TradeStationThursdayClient(auth, TS_SIM_ACCOUNT, http)
            _tdir = ALARM_FILE.parent
            self._trades_log = ChainedCsv(_tdir / "trades.csv",
                ["thursday", "symbol", "dir", "entry_t", "entry_px", "exit_t", "exit_px",
                 "qty", "ret_bps", "pnl_usd", "reason", "lr_slope20_bpd", "lr_slope40_bpd"])
            self._decisions_log = ChainedCsv(_tdir / "decisions.csv",
                ["ts_utc", "thursday", "mbt_sym", "met_sym", "mark_btc", "mark_eth",
                 "n_mbt", "n_met", "lr_slope20_bpd", "lr_slope40_bpd", "action", "detail"])
            orphan = await client.get_open_positions()              # G2 startup reconcile
            if orphan:
                logger.warning(f"orphan position(s) at startup: {[p.get('Symbol') for p in orphan]} — flattening")
                for p in orphan:
                    act = "BUY" if self._is_short(p) else "SELL"
                    await client.place_order(p["Symbol"], act, abs(int(float(p["Quantity"]))))
                self._alarm("orphan position at startup (flattened)")
            while self.running:
                try:
                    now = datetime.now(timezone.utc)
                    dow = now.weekday(); hour, minute = now.hour, now.minute
                    today = now.strftime("%Y-%m-%d")
                    day_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

                    if (dow == 3 and hour == ENTRY_HOUR and minute < 5
                            and not self.position_open and self.current_thursday != today):
                        await self._enter_tssim(client)
                    elif (self.position_open and dow == 3
                          and hour == EXIT_HOUR and minute >= EXIT_MINUTE):
                        await self._exit_tssim("scheduled_exit", client)
                    elif self.position_open and dow == 4:
                        logger.warning("Open position on Friday — emergency close [TSSIM]")
                        await self._exit_tssim("emergency_friday", client)
                    elif self.position_open:
                        pos = await client.get_open_positions()     # G2 per-poll reconcile
                        held = {p.get("Symbol") for p in pos}
                        if not ({self.ts_mbt_sym, self.ts_met_sym} <= held):
                            self._alarm(f"local/broker divergence: local {self.ts_mbt_sym}/{self.ts_met_sym}, broker {sorted(held)}")
                        bp = await client.last_price(self.ts_mbt_sym)
                        ep = await client.last_price(self.ts_met_sym)
                        if bp and ep and self._check_stop(bp, ep):
                            await self._exit_tssim("stop_loss", client)
                        else:
                            bbps = (self.btc_entry - bp)/self.btc_entry*10_000 if (self.btc_entry and bp) else 0
                            ebps = (self.eth_entry - ep)/self.eth_entry*10_000 if (self.eth_entry and ep) else 0
                            logger.info(f"TSSIM {self.ts_mbt_sym} {bbps:+.1f}bps ({bp}) {self.ts_met_sym} {ebps:+.1f}bps ({ep}) | {day_names[dow]} {now.strftime('%H:%M UTC')}")
                    else:
                        if (dow == 3 and (hour, minute) > (ENTRY_HOUR, 5)
                                and self.ts_entry_attempted_date != today
                                and self.current_thursday != today):
                            self._alarm(f"no entry attempted on Thursday {today}")    # G3 absence
                            self.ts_entry_attempted_date = today                       # alarm once
                        logger.info(f"Waiting [{day_names[dow]} {now.strftime('%H:%M UTC')}] — next trade: next Thursday 00:00 UTC")

                    await asyncio.sleep(60 if self.position_open else 300)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error(f"Loop error: {exc}", exc_info=True)
                    await asyncio.sleep(30)
            if self.position_open:
                logger.warning("Shutdown with open position — attempting close [TSSIM]")
                await self._exit_tssim("shutdown", client)

    async def run(self) -> None:
        if TSSIM:
            mode = "TSSIM"
        elif MARGIN:
            mode = "MARGIN"
        elif LIVE:
            mode = "LIVE"
        else:
            mode = "PAPER"
        logger.info("=" * 60)
        logger.info(f"KRAKEN THURSDAY SHORT TRADER STARTING [{mode}]")
        logger.info(f"  BTC: {BTC_SIZE} contracts | ETH: {ETH_SIZE} contracts")
        logger.info(f"  BTC notional ≈ ${BTC_SIZE * 64_000:,.0f} | ETH notional ≈ ${ETH_SIZE * 1_728:,.0f}")
        logger.info(f"  Stop: {STOP_PCT}% | Entry: Thu 00:00 UTC | Exit: Thu 23:05 UTC")
        logger.info("=" * 60)
        self.running = True
        if TSSIM:
            await self._run_tssim()
        elif MARGIN:
            await self._run_margin()
        elif LIVE:
            await self._run_live()
        else:
            await self._run_paper()

    def stop(self) -> None:
        logger.info("Thursday Short Trader — stopping")
        self.running = False


async def main() -> None:
    trader = ThursdayShortTrader()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, trader.stop)
    try:
        await trader.run()
    except Exception as exc:
        logger.error(f"Fatal: {exc}", exc_info=True)
    finally:
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())
