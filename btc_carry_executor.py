#!/usr/bin/env python3
"""
BTC Funding-Rate Cash-and-Carry Executor

Delta-neutral strategy: long BTC spot + short BTC perpetual.
Collect 8h funding payments when annualized rate exceeds the hurdle.

Pre-registered parameters sealed in:
  _bmad-output/preregistration_btc_carry_backtest.md (commit 35d9e4d)
Backtest result: PASS — 23.6% ann. yield, Sharpe 12.64, MaxDD 1.93%

Modes:
  Paper (default) — monitors real Kraken funding rates, simulates P&L.
                    No real orders placed. No credentials required.
  Live (--live)   — places real market orders on both legs via Kraken APIs.
                    Requires KRAKEN_SPOT_API_KEY/SECRET and
                    KRAKEN_FUTURES_API_KEY/SECRET in environment.

Usage:
    .venv/bin/python btc_carry_executor.py                    # paper mode
    .venv/bin/python btc_carry_executor.py --live             # live mode
    .venv/bin/python btc_carry_executor.py --status           # status and exit
    .venv/bin/python btc_carry_executor.py --live --status    # status with balance check
    .venv/bin/python btc_carry_executor.py --notional 50000   # override position size
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import httpx

# ---------------------------------------------------------------------------
# Pre-registered parameters (sealed at commit 35d9e4d — do not change without
# a new pre-registration commit in _bmad-output/)
# ---------------------------------------------------------------------------
HURDLE_ANNUAL_PCT  = 10.0       # minimum annualised yield to enter carry
COST_BPS           = 15.0       # round-trip cost per leg transition (bps)
NEG_THRESHOLD             = -0.0001  # -0.01% per 8h; below this counts as negative
NEG_WINDOW_SIZE           = 5        # rolling window length for negative-funding exit
NEG_WINDOW_MIN_NEG        = 3        # exit if >= this many in window below threshold
BELOW_HURDLE_EXIT_PERIODS = 12       # exit if rate < hurdle for this many consecutive payments
SYMBOL             = "PF_XBTUSD"

# Operational constants (not strategy parameters)
DEFAULT_NOTIONAL   = 10_000.0   # USD notional per leg
POLL_INTERVAL_S    = 60         # poll loop interval
FUNDING_TTL_S      = 300        # re-fetch funding rate at most every 5 min
TICKERS_URL        = "https://futures.kraken.com/derivatives/api/v3/tickers"
MIN_BTC_ORDER      = 0.0001     # Kraken minimum spot/perp order size in BTC

STATE_FILE         = Path("data/carry_executor_state.json")
POSITIONS_LOG      = Path("logs/carry_positions.csv")
LOG_FILE           = Path("logs/btc_carry_executor.log")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("carry_executor")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

PositionStatus = Literal["FLAT", "ACTIVE"]


@dataclass
class CarryState:
    """Persisted executor state — survives restarts."""
    status:             PositionStatus = "FLAT"
    entry_time:         Optional[str]  = None   # ISO UTC
    entry_funding_ann:  float          = 0.0    # annualised rate at entry
    payment_history:    list           = field(default_factory=list)  # last NEG_WINDOW_SIZE 8h rates
    below_hurdle_count: int            = 0      # consecutive payments while rate < hurdle
    notional_usd:       float          = DEFAULT_NOTIONAL
    accrued_pnl:        float          = 0.0    # cumulative P&L this trade (fraction of notional)
    n_funding_payments: int            = 0      # payments received this trade
    last_payment_time:  Optional[str]  = None   # ISO UTC of last payment applied
    n_trades:           int            = 0      # completed carry legs
    total_pnl:          float          = 0.0    # all-time cumulative P&L (fraction of notional)
    total_cost:         float          = 0.0    # all-time transaction costs (fraction of notional)
    # Live-mode order tracking (Optional — None in paper mode)
    spot_order_id:      Optional[str]  = None   # Kraken Spot txid at last entry
    perp_order_id:      Optional[str]  = None   # Kraken Futures order_id at last entry
    entry_spot_price:   Optional[float] = None  # actual spot fill price at entry
    entry_perp_price:   Optional[float] = None  # actual perp fill price at entry

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CarryState":
        valid_keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


def load_state() -> CarryState:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return CarryState.from_json(json.load(f))
        except Exception as e:
            logger.warning(f"State load failed ({e}); starting fresh")
    return CarryState()


def save_state(state: CarryState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state.to_json(), f, indent=2)
    tmp.replace(STATE_FILE)


# ---------------------------------------------------------------------------
# Trade / position log (append-only CSV)
# ---------------------------------------------------------------------------

_POSITION_FIELDS = [
    "event_time", "event_type", "mode", "funding_ann_pct",
    "notional_usd", "pnl_frac", "pnl_usd",
    "n_payments", "neg_count", "status_after",
    "spot_order_id", "perp_order_id",
]


def _log_position_event(
    event_type: str,
    mode: str,
    funding_ann: float,
    notional: float,
    pnl_frac: float,
    n_payments: int,
    neg_count: int,
    status_after: str,
    spot_order_id: str = "",
    perp_order_id: str = "",
) -> None:
    POSITIONS_LOG.parent.mkdir(exist_ok=True)
    write_header = not POSITIONS_LOG.exists()
    with open(POSITIONS_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_POSITION_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({
            "event_time":       datetime.now(timezone.utc).isoformat(),
            "event_type":       event_type,
            "mode":             mode,
            "funding_ann_pct":  f"{funding_ann * 100:.4f}",
            "notional_usd":     f"{notional:.2f}",
            "pnl_frac":         f"{pnl_frac:.8f}",
            "pnl_usd":          f"{pnl_frac * notional:.4f}",
            "n_payments":       n_payments,
            "neg_count":        neg_count,
            "status_after":     status_after,
            "spot_order_id":    spot_order_id,
            "perp_order_id":    perp_order_id,
        })


# ---------------------------------------------------------------------------
# Funding rate fetch
# ---------------------------------------------------------------------------

class FundingRateFetcher:
    """TTL-cached fetch of PF_XBTUSD funding rate from Kraken /tickers."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http
        self._rate_8h: Optional[float]  = None
        self._rate_ann: Optional[float] = None
        self._last_fetched: float       = 0.0
        self._last_price: Optional[float] = None

    async def refresh(self) -> None:
        if time.time() - self._last_fetched < FUNDING_TTL_S:
            return
        try:
            resp = await self._http.get(TICKERS_URL, timeout=10.0)
            resp.raise_for_status()
            for t in resp.json().get("tickers", []):
                if t.get("symbol") == SYMBOL:
                    ann = float(t.get("fundingRate", 0.0))
                    self._rate_ann = ann
                    self._rate_8h  = ann / (3 * 365)
                    self._last_price = (
                        float(t["markPrice"]) if t.get("markPrice")
                        else float(t["last"]) if t.get("last")
                        else None
                    )
                    self._last_fetched = time.time()
                    logger.debug(
                        f"Funding fetch: {ann*100:.4f}% ann. "
                        f"({self._rate_8h*100:.6f}%/8h)  btc={self._last_price:,.0f}"
                        if self._last_price else
                        f"Funding fetch: {ann*100:.4f}% ann."
                    )
                    return
            logger.warning("PF_XBTUSD not found in tickers response")
        except Exception as e:
            logger.warning(f"Funding rate fetch failed: {e}")

    @property
    def rate_8h(self) -> Optional[float]:
        return self._rate_8h

    @property
    def rate_ann(self) -> Optional[float]:
        return self._rate_ann

    @property
    def price(self) -> Optional[float]:
        return self._last_price


# ---------------------------------------------------------------------------
# State machine helpers
# ---------------------------------------------------------------------------

def _cost_frac() -> float:
    return COST_BPS / 10_000.0


def _should_enter(rate_ann: float) -> bool:
    return rate_ann > (HURDLE_ANNUAL_PCT / 100.0)


def _hurdle_str() -> str:
    return f"{HURDLE_ANNUAL_PCT:.1f}%"


def _payment_due(state: CarryState) -> bool:
    """True if at least 8h have elapsed since the last payment (or entry, if none yet)."""
    if state.status != "ACTIVE":
        return False
    ref_str = state.last_payment_time or state.entry_time
    if ref_str is None:
        return False
    ref = datetime.fromisoformat(ref_str)
    return datetime.now(timezone.utc) >= ref + timedelta(hours=8)


def _calc_volume_btc(notional_usd: float, btc_price: float) -> float:
    """BTC quantity for both legs. Enforces Kraken minimum lot size."""
    vol = round(notional_usd / btc_price, 8)
    return max(vol, MIN_BTC_ORDER)


# ---------------------------------------------------------------------------
# Carry executor
# ---------------------------------------------------------------------------

class BTCCarryExecutor:
    def __init__(self, notional: float = DEFAULT_NOTIONAL, live: bool = False) -> None:
        self.notional = notional
        self.live     = live
        self.state    = load_state()
        self.state.notional_usd = notional
        self._http    = httpx.AsyncClient(timeout=15.0)
        self.fetcher  = FundingRateFetcher(self._http)
        self._running = False
        # Live-mode clients — created in run() so --status path skips credential checks
        self._spot_client    = None
        self._futures_client = None

    def _mode_tag(self) -> str:
        return "LIVE" if self.live else "PAPER"

    async def _preflight_balance_check(self) -> None:
        """Verify sufficient USD balance before entering live mode. Logs BTC and USD."""
        usd = await self._spot_client.get_usd_balance()
        btc = await self._spot_client.get_btc_balance()
        logger.info(f"Preflight balance — USD: ${usd:,.2f}  BTC: {btc:.8f}")
        required = self.notional * 1.01
        if usd < required:
            raise RuntimeError(
                f"Insufficient USD balance for live carry: "
                f"have ${usd:,.2f}, need ${required:,.2f} (notional + 1% buffer)"
            )
        logger.info(f"Balance check passed — USD ${usd:,.2f} >= required ${required:,.2f}")

    async def _enter(self, rate_ann: float) -> None:
        now  = datetime.now(timezone.utc)
        cost = _cost_frac()

        spot_order_id = ""
        perp_order_id = ""

        if self.live:
            btc_price  = self.fetcher.price
            if btc_price is None:
                logger.error("Cannot enter: BTC price unavailable")
                return
            volume_btc = _calc_volume_btc(self.notional, btc_price)
            if volume_btc < MIN_BTC_ORDER:
                logger.warning(
                    f"Computed volume {volume_btc} BTC below minimum {MIN_BTC_ORDER} — clamped"
                )

            # Perp short FIRST — if spot subsequently fails, perp can be cleanly unwound.
            # The reverse (spot bought, perp fails) leaves naked long BTC.
            try:
                perp_order_id = await self._futures_client.orders.place_order(
                    SYMBOL, "sell", "mkt", volume_btc
                )
                logger.info(f"Perp short placed: {volume_btc} BTC | order_id={perp_order_id}")
            except Exception as exc:
                logger.error(f"Perp short FAILED — aborting entry: {exc}")
                return

            # Spot buy SECOND
            try:
                spot_result = await self._spot_client.place_market_order("buy", volume_btc)
                spot_order_id = spot_result.txid
                self.state.entry_spot_price = spot_result.fill_price
                self.state.entry_perp_price = btc_price   # perp fill price not in sendorder response
                logger.info(
                    f"Spot buy confirmed: {spot_result.vol_exec} BTC "
                    f"@ ${spot_result.fill_price:,.2f} | txid={spot_order_id}"
                )
            except Exception as exc:
                logger.critical(
                    f"Spot buy FAILED after perp short — attempting emergency perp unwind: {exc}"
                )
                # Retry spot once after 5s
                await asyncio.sleep(5.0)
                try:
                    spot_result = await self._spot_client.place_market_order("buy", volume_btc)
                    spot_order_id = spot_result.txid
                    self.state.entry_spot_price = spot_result.fill_price
                    logger.info(f"Spot buy retry succeeded: txid={spot_order_id}")
                except Exception as retry_exc:
                    logger.critical(
                        f"Spot buy retry also failed — placing emergency perp close: {retry_exc}"
                    )
                    try:
                        await self._futures_client.orders.place_order(
                            SYMBOL, "buy", "mkt", volume_btc
                        )
                    except Exception as unwind_exc:
                        logger.critical(f"EMERGENCY PERP UNWIND FAILED — manual action required: {unwind_exc}")
                    return   # remain FLAT; do not update state to ACTIVE

        self.state.status             = "ACTIVE"
        self.state.entry_time         = now.isoformat()
        self.state.entry_funding_ann  = rate_ann
        self.state.payment_history    = []
        self.state.below_hurdle_count = 0
        self.state.accrued_pnl        = -cost
        self.state.n_funding_payments = 0
        self.state.last_payment_time  = None
        self.state.total_cost        += cost
        self.state.spot_order_id      = spot_order_id or None
        self.state.perp_order_id      = perp_order_id or None
        save_state(self.state)
        _log_position_event(
            "ENTER", self._mode_tag(), rate_ann, self.notional, -cost,
            0, 0, "ACTIVE", spot_order_id, perp_order_id,
        )
        logger.info(
            f"ENTER CARRY [{self._mode_tag()}] | "
            f"rate_ann={rate_ann*100:.4f}% > hurdle={_hurdle_str()} | "
            f"notional=${self.notional:,.0f} | entry_cost=${cost*self.notional:.2f}"
        )

    async def _apply_payment(self, rate_8h: float, rate_ann: float) -> None:
        now      = datetime.now(timezone.utc)
        pnl_frac = rate_8h
        self.state.accrued_pnl        += pnl_frac
        self.state.n_funding_payments += 1
        self.state.last_payment_time   = now.isoformat()

        # Sliding-window negative exit (Change A)
        self.state.payment_history.append(rate_8h)
        if len(self.state.payment_history) > NEG_WINDOW_SIZE:
            self.state.payment_history.pop(0)

        # Below-hurdle exit (Change B)
        if rate_ann < HURDLE_ANNUAL_PCT / 100.0:
            self.state.below_hurdle_count += 1
        else:
            self.state.below_hurdle_count = 0

        neg_in_window = sum(r < NEG_THRESHOLD for r in self.state.payment_history)
        save_state(self.state)
        _log_position_event(
            "PAYMENT", self._mode_tag(), rate_ann, self.notional, pnl_frac,
            self.state.n_funding_payments, neg_in_window, "ACTIVE",
        )
        logger.info(
            f"PAYMENT #{self.state.n_funding_payments} | "
            f"rate_8h={rate_8h*100:.4f}%  ({rate_ann*100:.4f}% ann.) | "
            f"pnl_this=${pnl_frac*self.notional:.4f} | "
            f"accrued=${self.state.accrued_pnl*self.notional:.4f} | "
            f"neg={neg_in_window}/{NEG_WINDOW_MIN_NEG}of{NEG_WINDOW_SIZE} | "
            f"bhx={self.state.below_hurdle_count}/{BELOW_HURDLE_EXIT_PERIODS}"
        )

    async def _exit(self, reason: str, rate_ann: float) -> None:
        cost       = _cost_frac()
        net_pnl    = self.state.accrued_pnl - cost
        n_payments = self.state.n_funding_payments
        accrued    = self.state.accrued_pnl

        if self.live:
            btc_price  = self.fetcher.price
            volume_btc = _calc_volume_btc(self.notional, btc_price) if btc_price else MIN_BTC_ORDER

            # Spot sell FIRST, then close perp
            try:
                spot_result = await self._spot_client.place_market_order("sell", volume_btc)
                logger.info(
                    f"Spot sell confirmed: {spot_result.vol_exec} BTC "
                    f"@ ${spot_result.fill_price:,.2f} | txid={spot_result.txid}"
                )
            except Exception as exc:
                logger.error(f"Spot sell failed — retrying once: {exc}")
                await asyncio.sleep(5.0)
                try:
                    spot_result = await self._spot_client.place_market_order("sell", volume_btc)
                    logger.info(f"Spot sell retry succeeded: txid={spot_result.txid}")
                except Exception as retry_exc:
                    logger.critical(f"Spot sell retry also failed — perp still being closed: {retry_exc}")

            try:
                perp_close_id = await self._futures_client.orders.place_order(
                    SYMBOL, "buy", "mkt", volume_btc
                )
                logger.info(f"Perp close confirmed: order_id={perp_close_id}")
            except Exception as exc:
                logger.critical(f"Perp close FAILED — manual action required: {exc}")

        self.state.total_pnl  += net_pnl
        self.state.total_cost += cost
        self.state.n_trades   += 1
        self.state.status             = "FLAT"
        self.state.entry_time         = None
        self.state.entry_funding_ann  = 0.0
        self.state.payment_history    = []
        self.state.below_hurdle_count = 0
        self.state.accrued_pnl        = 0.0
        self.state.n_funding_payments = 0
        self.state.last_payment_time  = None
        self.state.spot_order_id      = None
        self.state.perp_order_id      = None
        self.state.entry_spot_price   = None
        self.state.entry_perp_price   = None
        save_state(self.state)
        _log_position_event(
            "EXIT", self._mode_tag(), rate_ann, self.notional, net_pnl,
            n_payments, 0, "FLAT",
        )
        logger.info(
            f"EXIT CARRY [{self._mode_tag()}] ({reason}) | "
            f"payments={n_payments} | "
            f"gross=${accrued*self.notional:.4f} | "
            f"exit_cost=${cost*self.notional:.2f} | "
            f"net_pnl=${net_pnl*self.notional:.4f} | "
            f"all_time_pnl=${self.state.total_pnl*self.notional:.4f} | "
            f"n_trades={self.state.n_trades}"
        )

    async def _step(self) -> None:
        await self.fetcher.refresh()
        rate_8h  = self.fetcher.rate_8h
        rate_ann = self.fetcher.rate_ann

        if rate_8h is None or rate_ann is None:
            logger.warning("Funding rate unavailable — skipping step")
            return

        if _payment_due(self.state):
            await self._apply_payment(rate_8h, rate_ann)
            neg_in_window = sum(r < NEG_THRESHOLD for r in self.state.payment_history)
            if neg_in_window >= NEG_WINDOW_MIN_NEG:
                await self._exit(f"neg_funding_{NEG_WINDOW_MIN_NEG}of{NEG_WINDOW_SIZE}", rate_ann)
                return
            if self.state.below_hurdle_count >= BELOW_HURDLE_EXIT_PERIODS:
                await self._exit(f"below_hurdle_x{BELOW_HURDLE_EXIT_PERIODS}", rate_ann)
                return

        if self.state.status == "FLAT" and _should_enter(rate_ann):
            await self._enter(rate_ann)

        price_str = f"${self.fetcher.price:,.0f}" if self.fetcher.price else "?"
        neg_in_window = sum(r < NEG_THRESHOLD for r in self.state.payment_history)
        logger.info(
            f"poll [{self._mode_tag()}] | status={self.state.status} | "
            f"funding={rate_ann*100:.4f}%ann | btc={price_str} | "
            f"hurdle_met={_should_enter(rate_ann)} | "
            f"neg={neg_in_window}/{NEG_WINDOW_MIN_NEG}of{NEG_WINDOW_SIZE} | "
            f"bhx={self.state.below_hurdle_count}/{BELOW_HURDLE_EXIT_PERIODS} | "
            f"accrued=${self.state.accrued_pnl*self.notional:.4f}"
        )

    async def run(self) -> None:
        logger.info("=" * 60)
        logger.info(f"BTC CASH-AND-CARRY EXECUTOR  [{self._mode_tag()}]")
        logger.info(f"  Notional : ${self.notional:,.0f}")
        logger.info(f"  Hurdle   : {_hurdle_str()} annualized")
        logger.info(
            f"  Exit (A) : {NEG_WINDOW_MIN_NEG}-of-{NEG_WINDOW_SIZE} payments "
            f"< {NEG_THRESHOLD*100:.3f}%/8h  [pre-reg 79612bc]"
        )
        logger.info(
            f"  Exit (B) : rate < {HURDLE_ANNUAL_PCT:.0f}% ann. "
            f"for {BELOW_HURDLE_EXIT_PERIODS} consecutive payments  [pre-reg 7fc065a]"
        )
        logger.info("  Pre-reg  : _bmad-output/preregistration_btc_carry_backtest.md (35d9e4d)")
        logger.info("=" * 60)
        logger.info(
            f"Resuming: status={self.state.status} "
            f"n_trades={self.state.n_trades} "
            f"all_time_pnl=${self.state.total_pnl*self.notional:.4f}"
        )

        if self.live:
            from src.execution.kraken.spot import KrakenSpotClient
            from src.execution.kraken import KrakenFuturesClient
            self._spot_client    = KrakenSpotClient()
            self._futures_client = KrakenFuturesClient(live=True)
            await self._spot_client.__aenter__()
            await self._futures_client.__aenter__()
            await self.fetcher.refresh()
            await self._preflight_balance_check()

        self._running = True
        try:
            while self._running:
                try:
                    await self._step()
                except Exception as e:
                    logger.error(f"Step error: {e}", exc_info=True)
                await asyncio.sleep(POLL_INTERVAL_S)
        except asyncio.CancelledError:
            pass
        finally:
            if self._spot_client is not None:
                await self._spot_client.__aexit__(None, None, None)
            if self._futures_client is not None:
                await self._futures_client.__aexit__(None, None, None)
            await self._http.aclose()
            logger.info("Executor stopped.")

    def stop(self) -> None:
        self._running = False

    async def print_status(self) -> None:
        await self.fetcher.refresh()
        rate_ann = self.fetcher.rate_ann
        price    = self.fetcher.price

        print()
        print("=" * 60)
        print(f"BTC CASH-AND-CARRY EXECUTOR — STATUS [{self._mode_tag()}]")
        print("=" * 60)
        print(f"  Position       : {self.state.status}")
        if self.state.status == "ACTIVE":
            print(f"  Entry time     : {self.state.entry_time}")
            print(f"  Entry rate     : {self.state.entry_funding_ann*100:.4f}% ann.")
            print(f"  Payments rec'd : {self.state.n_funding_payments}")
            neg_in_window = sum(r < NEG_THRESHOLD for r in self.state.payment_history)
            print(f"  Neg (window)   : {neg_in_window}/{NEG_WINDOW_MIN_NEG} of {NEG_WINDOW_SIZE}")
            print(f"  Below-hurdle   : {self.state.below_hurdle_count}/{BELOW_HURDLE_EXIT_PERIODS}")
            print(f"  Accrued P&L    : ${self.state.accrued_pnl*self.notional:.4f}")
            if self.state.spot_order_id:
                print(f"  Spot txid      : {self.state.spot_order_id}")
            if self.state.perp_order_id:
                print(f"  Perp order_id  : {self.state.perp_order_id}")
        print()
        if rate_ann is not None:
            hurdle = HURDLE_ANNUAL_PCT / 100.0
            print(
                f"  Live funding   : {rate_ann*100:.4f}% ann.  "
                f"({rate_ann/(3*365)*100:.6f}%/8h)"
            )
            if price:
                print(f"  BTC price      : ${price:,.0f}")
            print(f"  Hurdle ({_hurdle_str()})  : {'MET ✓' if rate_ann > hurdle else f'NOT MET (need {_hurdle_str()})'}")
        else:
            print("  Live funding   : unavailable")

        if self.live:
            from src.execution.kraken.spot import KrakenSpotClient
            async with KrakenSpotClient() as spot:
                usd = await spot.get_usd_balance()
                btc = await spot.get_btc_balance()
            print()
            print(f"  USD balance    : ${usd:,.2f}")
            print(f"  BTC balance    : {btc:.8f}")

        print()
        print(f"  Completed trades : {self.state.n_trades}")
        print(f"  All-time P&L     : ${self.state.total_pnl*self.notional:.4f}")
        print(f"  All-time costs   : ${self.state.total_cost*self.notional:.4f}")
        print("=" * 60)
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Cash-and-Carry Executor")
    p.add_argument(
        "--live", action="store_true",
        help="Place real orders on Kraken Spot + Futures (default: paper simulation)",
    )
    p.add_argument(
        "--status", action="store_true",
        help="Print current state and live funding rate, then exit",
    )
    p.add_argument(
        "--notional", type=float, default=DEFAULT_NOTIONAL,
        help=f"USD notional per leg (default: ${DEFAULT_NOTIONAL:,.0f})",
    )
    return p.parse_args()


async def _main() -> None:
    args     = parse_args()
    executor = BTCCarryExecutor(notional=args.notional, live=args.live)

    if args.status:
        await executor.print_status()
        return

    loop = asyncio.get_running_loop()

    def _shutdown() -> None:
        logger.info("Shutdown signal received")
        executor.stop()

    try:
        import signal
        loop.add_signal_handler(signal.SIGTERM, _shutdown)
        loop.add_signal_handler(signal.SIGINT,  _shutdown)
    except NotImplementedError:
        pass   # Windows

    await executor.run()


if __name__ == "__main__":
    asyncio.run(_main())
