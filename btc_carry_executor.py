#!/usr/bin/env python3
"""
BTC Funding-Rate Cash-and-Carry Executor (paper trading)

Delta-neutral strategy: notional long BTC spot + short BTC perpetual.
Collect 8h funding payments when annualized rate exceeds the hurdle.

Pre-registered parameters sealed in:
  _bmad-output/preregistration_btc_carry_backtest.md (commit 35d9e4d)
Backtest result: PASS — 23.6% ann. yield, Sharpe 12.64, MaxDD 1.93%

Paper mode: monitors real Kraken funding rates, simulates P&L,
            logs "WOULD ENTER / WOULD EXIT" events. No real orders placed.

Usage:
    .venv/bin/python btc_carry_executor.py              # run executor
    .venv/bin/python btc_carry_executor.py --status     # print current state and exit
    .venv/bin/python btc_carry_executor.py --notional 50000  # override position size
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
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
NEG_THRESHOLD      = -0.0001    # -0.01% per 8h; below this counts as negative
NEG_STOP_PERIODS   = 3          # consecutive negative-threshold periods to exit
SYMBOL             = "PF_XBTUSD"

# Operational constants (not strategy parameters)
DEFAULT_NOTIONAL   = 10_000.0   # USD notional per leg
POLL_INTERVAL_S    = 60         # poll loop interval
FUNDING_TTL_S      = 300        # re-fetch funding rate at most every 5 min
TICKERS_URL        = "https://futures.kraken.com/derivatives/api/v3/tickers"

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
    neg_count:          int            = 0      # consecutive negative-threshold periods
    notional_usd:       float          = DEFAULT_NOTIONAL
    accrued_pnl:        float          = 0.0    # cumulative P&L this trade (fraction of notional)
    n_funding_payments: int            = 0      # payments received this trade
    last_payment_time:  Optional[str]  = None   # ISO UTC of last payment applied
    n_trades:           int            = 0      # completed carry legs
    total_pnl:          float          = 0.0    # all-time cumulative P&L (fraction of notional)
    total_cost:         float          = 0.0    # all-time transaction costs (fraction of notional)

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
    "event_time", "event_type", "funding_ann_pct",
    "notional_usd", "pnl_frac", "pnl_usd",
    "n_payments", "neg_count", "status_after",
]


def _log_position_event(
    event_type: str,
    funding_ann: float,
    notional: float,
    pnl_frac: float,
    n_payments: int,
    neg_count: int,
    status_after: str,
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
            "funding_ann_pct":  f"{funding_ann * 100:.4f}",
            "notional_usd":     f"{notional:.2f}",
            "pnl_frac":         f"{pnl_frac:.8f}",
            "pnl_usd":          f"{pnl_frac * notional:.4f}",
            "n_payments":       n_payments,
            "neg_count":        neg_count,
            "status_after":     status_after,
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
                    ann = float(t.get("fundingRate", 0.0))   # Kraken: annualised decimal
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


# ---------------------------------------------------------------------------
# Carry executor
# ---------------------------------------------------------------------------

class BTCCarryExecutor:
    def __init__(self, notional: float = DEFAULT_NOTIONAL) -> None:
        self.notional = notional
        self.state    = load_state()
        self.state.notional_usd = notional
        self._http    = httpx.AsyncClient(timeout=15.0)
        self.fetcher  = FundingRateFetcher(self._http)
        self._running = False

    async def _enter(self, rate_ann: float) -> None:
        now  = datetime.now(timezone.utc)
        cost = _cost_frac()
        self.state.status             = "ACTIVE"
        self.state.entry_time         = now.isoformat()
        self.state.entry_funding_ann  = rate_ann
        self.state.neg_count          = 0
        self.state.accrued_pnl        = -cost   # entry cost deducted upfront
        self.state.n_funding_payments = 0
        self.state.last_payment_time  = None
        self.state.total_cost        += cost
        save_state(self.state)
        _log_position_event("ENTER", rate_ann, self.notional, -cost, 0, 0, "ACTIVE")
        logger.info(
            f"ENTER CARRY | rate_ann={rate_ann*100:.4f}% > hurdle={_hurdle_str()} | "
            f"notional=${self.notional:,.0f} | entry_cost=${cost*self.notional:.2f} | "
            f"[PAPER — no real orders placed]"
        )

    async def _apply_payment(self, rate_8h: float, rate_ann: float) -> None:
        now      = datetime.now(timezone.utc)
        pnl_frac = rate_8h
        self.state.accrued_pnl        += pnl_frac
        self.state.n_funding_payments += 1
        self.state.last_payment_time   = now.isoformat()

        if rate_8h < NEG_THRESHOLD:
            self.state.neg_count += 1
        else:
            self.state.neg_count = 0

        save_state(self.state)
        _log_position_event(
            "PAYMENT", rate_ann, self.notional, pnl_frac,
            self.state.n_funding_payments, self.state.neg_count, "ACTIVE"
        )
        logger.info(
            f"PAYMENT #{self.state.n_funding_payments} | "
            f"rate_8h={rate_8h*100:.4f}%  ({rate_ann*100:.4f}% ann.) | "
            f"pnl_this=${pnl_frac*self.notional:.4f} | "
            f"accrued=${self.state.accrued_pnl*self.notional:.4f} | "
            f"neg_count={self.state.neg_count}/{NEG_STOP_PERIODS}"
        )

    async def _exit(self, reason: str, rate_ann: float) -> None:
        cost       = _cost_frac()
        net_pnl    = self.state.accrued_pnl - cost
        n_payments = self.state.n_funding_payments
        accrued    = self.state.accrued_pnl
        self.state.total_pnl  += net_pnl
        self.state.total_cost += cost
        self.state.n_trades   += 1
        self.state.status             = "FLAT"
        self.state.entry_time         = None
        self.state.entry_funding_ann  = 0.0
        self.state.neg_count          = 0
        self.state.accrued_pnl        = 0.0
        self.state.n_funding_payments = 0
        self.state.last_payment_time  = None
        save_state(self.state)
        _log_position_event("EXIT", rate_ann, self.notional, net_pnl, n_payments, 0, "FLAT")
        logger.info(
            f"EXIT CARRY ({reason}) | "
            f"payments={n_payments} | "
            f"gross=${accrued*self.notional:.4f} | "
            f"exit_cost=${cost*self.notional:.2f} | "
            f"net_pnl=${net_pnl*self.notional:.4f} | "
            f"all_time_pnl=${self.state.total_pnl*self.notional:.4f} | "
            f"n_trades={self.state.n_trades} | "
            f"[PAPER]"
        )

    async def _step(self) -> None:
        await self.fetcher.refresh()
        rate_8h  = self.fetcher.rate_8h
        rate_ann = self.fetcher.rate_ann

        if rate_8h is None or rate_ann is None:
            logger.warning("Funding rate unavailable — skipping step")
            return

        # Apply 8h payment if due
        if _payment_due(self.state):
            await self._apply_payment(rate_8h, rate_ann)
            if self.state.neg_count >= NEG_STOP_PERIODS:
                await self._exit(f"neg_funding_x{NEG_STOP_PERIODS}", rate_ann)
                return

        # Entry check (only when FLAT)
        if self.state.status == "FLAT" and _should_enter(rate_ann):
            await self._enter(rate_ann)

        price_str = f"${self.fetcher.price:,.0f}" if self.fetcher.price else "?"
        logger.info(
            f"poll | status={self.state.status} | "
            f"funding={rate_ann*100:.4f}%ann | btc={price_str} | "
            f"hurdle_met={_should_enter(rate_ann)} | "
            f"neg={self.state.neg_count}/{NEG_STOP_PERIODS} | "
            f"accrued=${self.state.accrued_pnl*self.notional:.4f}"
        )

    async def run(self) -> None:
        logger.info("=" * 60)
        logger.info("BTC CASH-AND-CARRY EXECUTOR  [paper mode]")
        logger.info(f"  Notional : ${self.notional:,.0f}")
        logger.info(f"  Hurdle   : {_hurdle_str()} annualized")
        logger.info(
            f"  Exit rule: funding < {NEG_THRESHOLD*100:.3f}%/8h "
            f"for {NEG_STOP_PERIODS} consecutive periods"
        )
        logger.info("  Pre-reg  : _bmad-output/preregistration_btc_carry_backtest.md (35d9e4d)")
        logger.info("=" * 60)
        logger.info(
            f"Resuming: status={self.state.status} "
            f"n_trades={self.state.n_trades} "
            f"all_time_pnl=${self.state.total_pnl*self.notional:.4f}"
        )

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
        print("BTC CASH-AND-CARRY EXECUTOR — STATUS")
        print("=" * 60)
        print(f"  Position       : {self.state.status}")
        if self.state.status == "ACTIVE":
            print(f"  Entry time     : {self.state.entry_time}")
            print(f"  Entry rate     : {self.state.entry_funding_ann*100:.4f}% ann.")
            print(f"  Payments rec'd : {self.state.n_funding_payments}")
            print(f"  Neg count      : {self.state.neg_count}/{NEG_STOP_PERIODS}")
            print(f"  Accrued P&L    : ${self.state.accrued_pnl*self.notional:.4f}")
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
    p = argparse.ArgumentParser(description="BTC Cash-and-Carry Executor (paper mode)")
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
    executor = BTCCarryExecutor(notional=args.notional)

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
