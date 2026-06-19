#!/usr/bin/env python3
"""
TIER 2 FVG Paper Trading - TradeStation HTTP Polling + SIM Order Placement
Configuration: SL5.0x_TP5.0x_Midpoint_H1Sweep + ML Meta-Labeling Filter

Entry fires a bracket order on SIM account (entry + TP limit + SL stop).
The SIM account manages TP/SL fills. Local per-bar simulation is the
authoritative P&L record and handles the time-stop (cancel bracket + flat close).
"""

import asyncio
import csv as _csv_mod
import itertools
import json
import logging
import os
import sys
import time as _time_mod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import pytz
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3
from src.data.models import DollarBar
from src.research.projectx_bars import fetch_px_ts_shaped, ProjectXBarFetchError, _to_contract_id
from src.research.shadow_parity import ShadowParityLogger, bars_by_minute
import src.research.strategy_core as strategy_core
from src.research.strategy_core import (
    Direction,
    EntryDecision,
    ExitDecision,
    ExitReason,
    FVGSignal,
    StrategyConfig,
    SweepSignal,
    calc_atr,
    calc_max_drawdown_pct,
    calc_profit_factor,
    calc_sharpe,
    check_exit,
    detect_fvg,
    detect_liquidity_sweep,
    kill_zone_filter,
    make_entry_decision,
    resample_to_h1,
    volatility_regime_filter,
)

# Strategy parameters live in StrategyConfig (strategy_core.py) — the single source
# of truth shared by live trader and backtest engine.  Do not add duplicates here.
# Default values: SL 5.0×, TP 6.0×, entry 50%, gap≥25% ATR, vol-regime 75th pct,
# max_hold 60 bars, max_pending 240 bars, 5 contracts, daily loss limit −$750.

# ML Filter model path (infrastructure — not a strategy parameter)
ML_MODEL_PATH = Path(__file__).parent.parent.parent / "models/xgboost/tier2_meta_labeling_model.pkl"

# Per-instrument specifications (point value, tick size, default contract count)
SYMBOL_SPECS: dict[str, dict] = {
    "MNQM26": {"point_value": 2.0,  "tick_size": 0.25, "contracts": 5},
    "MNQU26": {"point_value": 2.0,  "tick_size": 0.25, "contracts": 5},  # Sept (active from 06-2026 roll)
    "MNQZ26": {"point_value": 2.0,  "tick_size": 0.25, "contracts": 5},  # Dec (next roll ~2026-09-11)
    "MNQH27": {"point_value": 2.0,  "tick_size": 0.25, "contracts": 5},  # Mar 2027
    "MESM26": {"point_value": 5.0,  "tick_size": 0.25, "contracts": 2},
    "M2KM26": {"point_value": 5.0,  "tick_size": 0.10, "contracts": 2},
}

# Account and trade state types (FR14: SIM ↔ live via config; FR10: per-cycle reconciliation)
TradeStatus = Literal["FLAT", "PENDING", "ACTIVE"]


@dataclass
class AccountConfig:
    """Account configuration for TradeStationClient — SIM vs. live is a config value (FR14)."""
    account_id: str
    execution_mode: Literal["sim", "live"]
    symbol: str
    point_value: float
    tick_size: float
    contracts: int


@dataclass
class TradeState:
    """Broker-reconciled trade state returned by TradeStationClient.reconcile_state() (FR10)."""
    status: TradeStatus
    entry_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    position_qty: int = 0


@dataclass
class TradeRecord:
    """Completed trade record written to the trade log CSV (AR15, FR29).

    Field order matches _COLUMNS in TradeLogger — do not reorder.
    """
    timestamp_entry: datetime
    timestamp_exit: datetime
    direction: str
    entry_price: float
    exit_price: float
    tp_price: float
    sl_price: float
    gap_size: float
    pnl_usd: float
    exit_reason: str
    h1_sweep_bars_ago: int
    m15_confirmed: bool
    kill_zone_active: bool
    vol_regime_pct: float
    contracts: int


# TradeStation market data API
BAR_INTERVAL = "1"
BAR_UNIT = "Minute"
HISTORY_HOURS = 48  # Enough history for H1 swing detection
POLL_INTERVAL_SECONDS = 60

# TradeStation SIM order placement
SIM_ACCOUNT_ID = "SIM2797251F"
SIM_ORDERS_URL = "https://sim-api.tradestation.com/v3/orderexecution/orders"
# Execution backend (Phase B): if PROJECTX_ACCOUNT_ID is set, YANK executes on the
# Topstep combine via ProjectX at YANK_CONTRACTS size; else TradeStation SIM (paper).
YANK_CONTRACTS = int(os.environ.get("YANK_CONTRACTS", "0"))  # >0 overrides per-symbol spec (combine uses 2)

def _default_account_config(symbol: str) -> AccountConfig:
    """Build AccountConfig from SYMBOL_SPECS for *symbol*. Caller must validate symbol first."""
    spec = SYMBOL_SPECS[symbol]
    return AccountConfig(
        account_id=SIM_ACCOUNT_ID,
        execution_mode="sim",
        symbol=symbol,
        point_value=spec["point_value"],
        tick_size=spec["tick_size"],
        contracts=spec["contracts"],
    )


ET_TZ = pytz.timezone('US/Eastern')
_NY_TZ = pytz.timezone('America/New_York')
CT_TZ = pytz.timezone('America/Chicago')  # Topstep session clock
# Topstep combine compliance: auto-flatten at 15:10 CT, no new entries 15:08-17:00 CT,
# no overnight carry. Evening/Globex (17:00 CT+) trading is retained. Modeled in the
# constrained joint MC (results_yank_mim_joint_constrained.md).
TOPSTEP_FLATTEN_MIN = 15 * 60 + 10   # 15:10 CT
TOPSTEP_BLOCK_LO = 15 * 60 + 8       # 15:08 CT — risk managers start flattening
TOPSTEP_BLOCK_HI = 17 * 60           # 17:00 CT — session reopens

# Rolling buffer cap — 125 H1 bars × 60 min covers vol_regime_lookback=120 H1 bars (AR16).
_BUFFER_CAP: int = 7500


def _dollar_bars_to_df(bars: "list[DollarBar]") -> pd.DataFrame:
    """Convert a bounded list of DollarBars to the canonical AR9 DataFrame.

    Timezone is already UTC on DollarBar.timestamp; converts to America/New_York
    at this ingest boundary (AR19 — single conversion, never inside strategy_core).
    """
    rows = {
        "timestamp": pd.to_datetime([b.timestamp for b in bars], utc=True),
        "open": [b.open for b in bars],
        "high": [b.high for b in bars],
        "low": [b.low for b in bars],
        "close": [b.close for b in bars],
        "volume": [b.volume for b in bars],
    }
    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    return df.sort_index()


def _build_strategy_config() -> StrategyConfig:
    """Load StrategyConfig from YAML if available; fall back to dataclass defaults.

    Resolution order:
    1. STRATEGY_CONFIG_PATH env var (explicit override)
    2. strategy_config.yaml at repo root (default YAML)
    3. StrategyConfig() dataclass defaults
    """
    from src.research.config_loader import load_strategy_config

    yaml_path_env = os.environ.get("STRATEGY_CONFIG_PATH")
    if yaml_path_env:
        path = Path(yaml_path_env)
        if path.exists():
            logger.info(f"Loading strategy config from env STRATEGY_CONFIG_PATH: {path}")
            return load_strategy_config(path)
        logger.warning(f"STRATEGY_CONFIG_PATH={yaml_path_env!r} not found; falling through")

    default_yaml = Path(__file__).parent.parent.parent / "strategy_config.yaml"
    if default_yaml.exists():
        logger.info(f"Loading strategy config from {default_yaml}")
        return load_strategy_config(default_yaml)

    logger.info("Using StrategyConfig() dataclass defaults")
    return StrategyConfig()


class StatePersistence:
    """Atomic JSON state file writer/reader for crash-safe active-trade recovery (AR14, AR15).

    Writes to a .tmp file then ``os.replace()`` atomically to avoid partial writes.
    Only this class reads/writes the state file paths (AR15).
    """

    _LOG_DIR = Path(__file__).parent.parent.parent / "logs"
    STATE_PATH = _LOG_DIR / "active_trade_state.json"
    TMP_PATH = _LOG_DIR / "active_trade_state.tmp"

    @classmethod
    def save_state(cls, state: dict) -> None:
        cls._LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.TMP_PATH.write_text(json.dumps(state, default=str), encoding="utf-8")
        os.replace(cls.TMP_PATH, cls.STATE_PATH)

    @classmethod
    def load_state(cls) -> "dict | None":
        try:
            return json.loads(cls.STATE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    @classmethod
    def clear_state(cls) -> None:
        try:
            cls.STATE_PATH.unlink(missing_ok=True)
        except OSError:
            pass


class TradeLogger:
    """Sole appender of the trade log CSV in PRD-mandated column order (AR15, FR29).

    Single-writer pattern: only this class appends to tier2_trade_log.csv.
    Header written only when file is empty (f.tell() == 0) — avoids TOCTOU race (AC#2).
    """

    _LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "tier2_trade_log.csv"
    _COLUMNS = [
        "timestamp_entry", "timestamp_exit", "direction", "entry_price",
        "exit_price", "tp_price", "sl_price", "gap_size", "pnl_usd",
        "exit_reason", "h1_sweep_bars_ago", "m15_confirmed", "kill_zone_active",
        "vol_regime_pct", "contracts",
    ]

    def append_trade(self, record: TradeRecord) -> None:
        import csv as _csv
        self._LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._LOG_PATH.open("a", newline="", encoding="utf-8") as f:
                writer = _csv.DictWriter(f, fieldnames=self._COLUMNS)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow({
                    "timestamp_entry":   record.timestamp_entry.isoformat(),
                    "timestamp_exit":    record.timestamp_exit.isoformat(),
                    "direction":         record.direction,
                    "entry_price":       round(record.entry_price, 4),
                    "exit_price":        round(record.exit_price, 4),
                    "tp_price":          round(record.tp_price, 4),
                    "sl_price":          round(record.sl_price, 4),
                    "gap_size":          round(record.gap_size, 4),
                    "pnl_usd":           round(record.pnl_usd, 2),
                    "exit_reason":       record.exit_reason,
                    "h1_sweep_bars_ago": record.h1_sweep_bars_ago,
                    "m15_confirmed":     record.m15_confirmed,
                    "kill_zone_active":  record.kill_zone_active,
                    "vol_regime_pct":    round(record.vol_regime_pct, 4),
                    "contracts":         record.contracts,
                })
        except Exception as e:
            logger.warning("Trade log write failed: %s", e)


class RiskManager:
    """Phase 1 daily circuit breaker (FR16, NFR12, architecture Risk Layer decision).

    Tracks daily P&L and halts new entries when the configured loss threshold is reached.
    Persists halt state immediately so a crash between trip and next trade close does not
    reset the circuit breaker on restart.

    Phase 2 extension (trailing DD, consistency rule, dynamic contracts) adds evaluators
    inside check_and_update() without changing the public surface.
    """

    def __init__(self) -> None:
        self._daily_pnl: float = 0.0
        self._daily_halted: bool = False
        self._last_trading_date: Optional[datetime.date] = None

    @property
    def is_halted(self) -> bool:
        return self._daily_halted

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def register_close(self, pnl: float) -> None:
        """Update daily P&L after a trade closes. Caller saves full state."""
        self._daily_pnl += pnl

    def check_and_update(self, bar_et: datetime, max_daily_loss: float) -> bool:
        """Reset on new calendar day, then check circuit breaker. Returns True if halted."""
        today = bar_et.date()
        if self._last_trading_date is not None and self._last_trading_date != today:
            logger.info(
                "New trading day %s — resetting daily P&L (was $%.2f)", today, self._daily_pnl
            )
            self._daily_pnl = 0.0
            self._daily_halted = False
        self._last_trading_date = today
        if self._daily_halted:
            return True
        if self._daily_pnl <= max_daily_loss:
            logger.warning(
                "🛑 Daily loss limit hit: $%.2f ≤ $%.0f — halting for today",
                self._daily_pnl, max_daily_loss,
            )
            self._daily_halted = True
            self._persist()
            return True
        return False

    def halt_manually(self) -> None:
        """Externally halt entries for today. Called by emergency stop CLI (FR22)."""
        self._daily_halted = True
        self._persist()

    def restore_from_state(self, state: dict, today: datetime.date) -> None:
        """Restore daily risk state from persisted dict on startup (crash recovery, NFR12)."""
        saved_date_str = state.get("last_trading_date")
        if not saved_date_str:
            return
        try:
            saved_date = (
                datetime.fromisoformat(saved_date_str).date()
                if isinstance(saved_date_str, str)
                else saved_date_str
            )
            if saved_date == today:
                self._daily_pnl = float(state.get("daily_pnl", 0.0))
                self._daily_halted = bool(state.get("daily_halted", False))
                self._last_trading_date = today
                logger.info(
                    "Restored daily risk state: pnl=%.2f halted=%s",
                    self._daily_pnl, self._daily_halted,
                )
        except (ValueError, TypeError) as e:
            logger.warning("Could not parse last_trading_date from state: %s", e)

    def to_state_dict(self) -> dict:
        """Return risk fields for inclusion in the persisted state dict."""
        return {
            "daily_pnl": self._daily_pnl,
            "daily_halted": self._daily_halted,
            "last_trading_date": self._last_trading_date.isoformat() if self._last_trading_date else None,
        }

    def _persist(self) -> None:
        """Persist current risk state immediately (called when circuit breaker trips or halt_manually)."""
        try:
            StatePersistence.save_state(self.to_state_dict())
        except Exception as e:
            logger.warning("RiskManager: failed to persist halt state: %s", e)


class TradeStationClient:
    """Sole network-touching component. Owns auth, bracket-order submission, order
    cancellation, market-close, and per-cycle broker reconciliation (AR3, FR9–FR12).

    SIM vs. live is an AccountConfig value — zero strategy-logic changes for FR14 swap.
    """

    _BROKERAGE_BASE = "https://sim-api.tradestation.com/v3/brokerage"

    def __init__(
        self,
        auth: TradeStationAuthV3,
        account_config: AccountConfig,
        httpx_client: httpx.AsyncClient,
    ):
        self._auth = auth
        self._cfg = account_config
        self._http = httpx_client

    async def _headers(self) -> dict:
        token = await self._auth.authenticate()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def submit_bracket_order(
        self, decision: EntryDecision, account_id: str
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Submit SIM bracket (entry limit + TP limit + SL stop). Returns (entry_id, tp_id, sl_id).

        Returns (None, None, None) on any network or HTTP error — never raises.
        """
        direction = "LONG" if decision.direction == Direction.BULLISH else "SHORT"
        entry_action = "BUY" if direction == "LONG" else "SELL"
        exit_action = "SELL" if direction == "LONG" else "BUY"
        qty = str(self._cfg.contracts)

        payload = {
            "AccountID": account_id,
            "Symbol": self._cfg.symbol,
            "Quantity": qty,
            "OrderType": "Limit",
            "LimitPrice": str(decision.entry_price),
            "TradeAction": entry_action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
            "OSOs": [{
                "Type": "BRK",
                "Orders": [
                    {
                        "AccountID": account_id,
                        "Symbol": self._cfg.symbol,
                        "Quantity": qty,
                        "OrderType": "Limit",
                        "TradeAction": exit_action,
                        "TimeInForce": {"Duration": "GTC"},
                        "LimitPrice": str(decision.tp_price),
                    },
                    {
                        "AccountID": account_id,
                        "Symbol": self._cfg.symbol,
                        "Quantity": qty,
                        "OrderType": "StopMarket",
                        "TradeAction": exit_action,
                        "TimeInForce": {"Duration": "GTC"},
                        "StopPrice": str(decision.sl_price),
                    },
                ],
            }],
        }
        try:
            headers = await self._headers()
            response = await self._http.post(SIM_ORDERS_URL, headers=headers, json=payload)
            if response.status_code not in (200, 201):
                logger.warning(f"⚠️ SIM bracket order failed HTTP {response.status_code}: {response.text[:200]}")
                return None, None, None
            data = response.json()
            orders = data.get("Orders", [])
            entry_id = tp_id = sl_id = None
            for order in orders:
                oid = order.get("OrderID")
                msg = order.get("Message", "")
                
                # TradeStation API often returns only Message and OrderID for OCOs
                # Message format: "Sent order: Buy 1 MNQM26 @ 1000.00 Limit"
                if "Stop Market" in msg:
                    sl_id = oid
                elif exit_action.capitalize() in msg and "Limit" in msg:
                    tp_id = oid
                elif entry_action.capitalize() in msg and "Limit" in msg:
                    entry_id = oid
                else:
                    # Fallback if messages don't match expected pattern
                    if entry_id is None:
                        entry_id = oid
                    elif tp_id is None:
                        tp_id = oid
                    else:
                        sl_id = oid
            logger.info(f"✓ SIM bracket submitted | entry #{entry_id} | TP #{tp_id} | SL #{sl_id}")
            return entry_id, tp_id, sl_id
        except Exception as e:
            logger.warning(f"⚠️ SIM bracket order exception: {e}")
            return None, None, None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order via DELETE. Returns True on 200/204 (success) or 404 (already gone)."""
        try:
            headers = await self._headers()
            url = f"https://sim-api.tradestation.com/v3/orderexecution/orders/{order_id}"
            response = await self._http.delete(url, headers=headers)
            return response.status_code in (200, 204, 404)
        except Exception as e:
            logger.warning(f"⚠️ Cancel order #{order_id} exception: {e}")
            return False

    async def close_position_at_market(self, direction: str, account_id: str, contracts: Optional[int] = None) -> Optional[str]:
        """Submit a market order to flatten the open position. Returns order ID or None on failure."""
        close_action = "SELL" if direction == "LONG" else "BUY"
        payload = {
            "AccountID": account_id,
            "Symbol": self._cfg.symbol,
            "Quantity": str(contracts if contracts is not None else self._cfg.contracts),
            "OrderType": "Market",
            "TradeAction": close_action,
            "TimeInForce": {"Duration": "DAY"},
            "Route": "Intelligent",
        }
        try:
            headers = await self._headers()
            response = await self._http.post(SIM_ORDERS_URL, headers=headers, json=payload)
            if response.status_code in (200, 201):
                oid = response.json().get("Orders", [{}])[0].get("OrderID")
                logger.info(f"✓ SIM flat close order #{oid} submitted")
                return oid
            return None
        except Exception as e:
            logger.warning(f"⚠️ SIM close order exception: {e}")
            return None

    async def reconcile_state(self, account_id: str) -> TradeState:
        """Query broker open orders + positions for this symbol; return FLAT/PENDING/ACTIVE.

        Conservative safe default: FLAT on any error — callers treat in-memory
        active_trade as authoritative; reconciliation is a cross-check (FR10, FR38).
        """
        try:
            headers = await self._headers()
            orders_url = (
                f"{self._BROKERAGE_BASE}/accounts/{account_id}/orders?status=Open"
            )
            pos_url = f"{self._BROKERAGE_BASE}/accounts/{account_id}/positions"
            orders_resp = await self._http.get(orders_url, headers=headers)
            pos_resp = await self._http.get(pos_url, headers=headers)

            symbol_orders = []
            if orders_resp.status_code == 200:
                symbol_orders = [
                    o for o in orders_resp.json().get("Orders", [])
                    if o.get("Symbol") == self._cfg.symbol
                ]
            position_qty = 0
            if pos_resp.status_code == 200:
                # int(float(...)) handles both integer strings ("5") and float strings ("5.0").
                # abs() because short positions have negative Quantity from the broker.
                position_qty = sum(
                    abs(int(float(p.get("Quantity", 0))))
                    for p in pos_resp.json().get("Positions", [])
                    if p.get("Symbol") == self._cfg.symbol
                )

            if position_qty != 0:
                return TradeState(status="ACTIVE", position_qty=position_qty)
            if symbol_orders:
                entry_id = next(
                    (o.get("OrderID") for o in symbol_orders if o.get("OrderType") == "Limit"),
                    None,
                )
                return TradeState(status="PENDING", entry_order_id=entry_id)
            return TradeState(status="FLAT")
        except Exception as e:
            logger.warning(f"⚠️ reconcile_state failed: {e} — assuming FLAT")
            return TradeState(status="FLAT")

    async def cancel_all_pending_orders(self, account_id: str) -> list:
        """Query all open orders for this symbol and cancel each. Returns list of cancelled IDs."""
        cancelled: list = []
        try:
            headers = await self._headers()
            orders_url = f"{self._BROKERAGE_BASE}/accounts/{account_id}/orders?status=Open"
            resp = await self._http.get(orders_url, headers=headers)
            if resp.status_code != 200:
                return cancelled
            orders = [
                o for o in resp.json().get("Orders", [])
                if o.get("Symbol") == self._cfg.symbol
            ]
            for order in orders:
                oid = order.get("OrderID")
                if oid and await self.cancel_order(oid):
                    cancelled.append(oid)
        except Exception as e:
            logger.warning(f"⚠️ cancel_all_pending_orders failed: {e}")
        return cancelled


def _parse_blocked_months(raw: str) -> frozenset:
    """Parse comma-separated month numbers from env var, skipping malformed tokens."""
    months = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            months.add(int(tok))
        except ValueError:
            logging.warning(f"TIER2_BLOCKED_MONTHS: skipping malformed token {tok!r}")
    return frozenset(months)


# Seasonality filter: months where edge is statistically zero (Aug-Oct bearish: $0.50/trade avg).
# Default empty — activate with TIER2_BLOCKED_MONTHS=8,9,10 env var.
# Note: bearish_only is a StrategyConfig field (default True); direction logic reads
# self._strategy_config.bearish_only — do not add a duplicate module-level constant here.
BLOCKED_MONTHS: frozenset = _parse_blocked_months(os.environ.get("TIER2_BLOCKED_MONTHS", ""))

# Setup logging
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

_handlers: list = [logging.FileHandler(log_dir / 'yank_streaming_working.log')]
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

    def __init__(self, model_path: Path, threshold: float = 0.0):  # 0.0 = disabled; matches StrategyConfig.ml_threshold
        self.threshold = threshold
        self.model = None
        if model_path.exists():
            try:
                self.model = joblib.load(model_path)
                logger.info(f"ML filter loaded from {model_path} (threshold={threshold})")
                # Load validated threshold from JSON; overrides constructor default
                _thr_json = Path(__file__).parent.parent.parent / "models/xgboost/tier2_threshold.json"
                if _thr_json.exists():
                    try:
                        import json as _json
                        _data = _json.loads(_thr_json.read_text())
                        self.threshold = float(_data["threshold"])
                        logger.info(
                            f"ML threshold loaded from JSON: {self.threshold} "
                            f"(validated {_data.get('validated_date', '?')})"
                        )
                    except Exception as _e:
                        logger.warning(f"Threshold JSON read failed: {_e} — using {self.threshold}")
            except Exception as e:
                logger.warning(f"ML model load failed: {e} — falling back to pass-through")
        else:
            logger.warning(f"ML model not found at {model_path} — falling back to pass-through")

    def _log_decision(self, timestamp, proba: float, decision: str) -> None:
        """Append filter decision to logs/tier2_filter_log.csv."""
        import csv as _csv
        log_path = Path(__file__).parent.parent.parent / "logs/tier2_filter_log.csv"
        row = {
            "timestamp":       str(timestamp),
            "filter_decision": decision,
            "probability":     round(proba, 4),
            "threshold":       self.threshold,
        }
        write_header = not log_path.exists()
        try:
            with log_path.open("a", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(row.keys()))
                if write_header:
                    _w.writeheader()
                _w.writerow(row)
        except Exception as _e:
            logger.warning(f"Filter log write failed: {_e}")

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


class LRRegimeFilter:
    """LR channel counter-trend pre-filter for Silver Bullet signals.

    Reads config from models/xgboost/lr_regime_config.json.
    Counter-trend: passes signals when LR regime DISAGREES with signal direction.
    Shorting into uptrend and buying into downtrend = Silver Bullet edge.
    SIDEWAYS regime → pass through (no trend to fade).
    """

    def __init__(self) -> None:
        self.enabled  = False
        self.fast_len = 390
        self.slow_len = 1950
        self.ml_threshold = 0.65
        _cfg = Path(__file__).parent.parent.parent / "models/xgboost/lr_regime_config.json"
        if _cfg.exists():
            try:
                import json as _json
                _data = _json.loads(_cfg.read_text())
                self.fast_len     = int(_data.get("fast_len", 390))
                self.slow_len     = int(_data.get("slow_len", 1950))
                self.ml_threshold = float(_data.get("ml_threshold", 0.65))
                self.enabled      = bool(_data.get("enabled", True))
                logger.info(
                    f"LR regime filter loaded: fast={self.fast_len}, "
                    f"slow={self.slow_len}, polarity=counter_trend, "
                    f"ml_threshold={self.ml_threshold}, enabled={self.enabled} "
                    f"(validated {_data.get('validated_date', '?')})"
                )
            except Exception as _e:
                logger.warning(f"LR regime config load failed: {_e} — filter disabled")
        else:
            logger.info("lr_regime_config.json not found — LR regime filter disabled")

    def allows(self, bars: list, signal_direction: str) -> bool:
        """Return True if this signal should proceed past the regime pre-filter."""
        if not self.enabled:
            return True
        if len(bars) < self.slow_len:
            return True  # insufficient bar history during warm-up → pass through
        try:
            import numpy as _np
            from src.ml.regime_detection.lr_channel_detector import LRChannelRegimeDetector
            closes = _np.array([b.close for b in bars[-self.slow_len:]], dtype=float)
            detector = LRChannelRegimeDetector(fast_len=self.fast_len, slow_len=self.slow_len)
            regimes  = detector.fit_predict(closes)
            regime   = regimes[-1]  # current bar regime
        except Exception as _e:
            logger.warning(f"LR regime computation failed: {_e} — passing signal through")
            return True

        # Counter-trend: pass when regime DISAGREES with signal direction
        if regime == "UP":
            passes = (signal_direction == "bearish")
        elif regime == "DOWN":
            passes = (signal_direction == "bullish")
        else:  # SIDEWAYS → neutral, no trend to fade
            passes = True

        if not passes:
            logger.info(
                f"Signal FILTERED by LR regime | regime={regime}, direction={signal_direction}"
            )
        return passes


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
    pending_entry: bool = True  # True until limit order fills
    # Trade-log metadata captured at entry (Story 4-2 — AC#1, AC#7)
    gap_size: float = 0.0
    h1_sweep_bars_ago: int = 0
    m15_confirmed: bool = False
    kill_zone_active: bool = False
    vol_regime_pct: float = 0.0
    # Passive drift canary: meta-model P(success) at entry, paired with realized
    # outcome at close (logs/yank_ml_canary.csv). Logging only — no control effect.
    ml_proba: float = float("nan")


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
    def __init__(self, symbol: str = "MNQM26"):
        spec = SYMBOL_SPECS.get(symbol)
        if spec is None:
            raise ValueError(f"Unknown symbol: {symbol!r}. Valid symbols: {list(SYMBOL_SPECS)}")
        self._symbol: str = symbol
        self._point_value: float = spec["point_value"]
        self._tick_size: float = spec["tick_size"]
        self._contracts: int = YANK_CONTRACTS or spec["contracts"]
        self._on_combine: bool = False        # set in initialize() from PROJECTX_ACCOUNT_ID
        self._exec_account: str = SIM_ACCOUNT_ID
        self._bars_base_url: str = (
            f"https://api.tradestation.com/v3/marketdata/barcharts/{symbol}"
            f"?interval={BAR_INTERVAL}&unit={BAR_UNIT}"
        )

        # Data backend (Stage-1 shadow migration to ProjectX). Default = TradeStation REST.
        # YANK_DATA_SOURCE=projectx makes ProjectX the SIGNAL source; YANK_DATA_SHADOW=1
        # fetches+logs ProjectX in parallel while TradeStation stays the signal source.
        self._data_source: str = os.environ.get("YANK_DATA_SOURCE", "tradestation")
        self._data_shadow: bool = (
            os.environ.get("YANK_DATA_SHADOW", "0") == "1" and self._data_source == "tradestation"
        )
        self._data_px_live: bool = os.environ.get("YANK_DATA_PX_LIVE", "0") == "1"
        self._px_data_contract_id: Optional[str] = None   # set in initialize()
        self._shadow_logger = None                        # set in initialize() if shadow on

        self.running = False
        self.auth = None
        self.client = None
        self._px_auth = None    # ProjectXAuth — set in initialize() for execution and/or data
        self.dollar_bars: list[DollarBar] = []
        self._last_processed_timestamp: Optional[datetime] = None
        self.active_trade: Optional[ActiveTrade] = None
        self.completed_trades: list[CompletedTrade] = []
        self._is_backfill: bool = True
        self.session_start_time: Optional[datetime] = None

        # Strategy config — must be created before any logging that references its fields
        self._strategy_config: StrategyConfig = _build_strategy_config()

        # ML Filter
        self.ml_filter = MetaLabelingFilter(ML_MODEL_PATH)
        self.lr_filter  = LRRegimeFilter()

        # Log active signal filters
        logger.info(f"Signal filters — BEARISH_ONLY={self._strategy_config.bearish_only}, "
                    f"BLOCKED_MONTHS={sorted(BLOCKED_MONTHS) if BLOCKED_MONTHS else 'none'}")

        # H1 sweep state — flags persist until 6-hour expiry window lapses
        self.h1_bullish_sweep_active = False
        self.h1_bearish_sweep_active = False
        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        self._bullish_sweep_expires: datetime = _epoch
        self._bearish_sweep_expires: datetime = _epoch
        # Track which H1 bar timestamp was last logged to avoid re-firing every minute
        self._last_bullish_sweep_h1_ts: datetime = _epoch
        self._last_bearish_sweep_h1_ts: datetime = _epoch

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
        self._h1_atr_history: list[float] = []   # rolling H1 ATR values for percentile gate
        self._vol_regime_high: bool = False       # True when current ATR > 75th pct of history
        self._last_vol_regime_pct: float = 0.0   # ATR percentile at last H1 update (for trade log)
        self._current_day: Optional[datetime.date] = None

        # M15 CHoCH confirmation state (S25 architecture)
        self._m15_choch_active: bool = False
        self._m15_last_bar_ts: datetime = _epoch

        # strategy_core integration (Story 1.5)
        self._cached_sweep: Optional[SweepSignal] = None        # most recent sweep from detect_liquidity_sweep
        self._active_entry_decision: Optional[EntryDecision] = None  # EntryDecision for active trade (check_exit)

        # Per-bar timing (AR17)
        self._bar_processing_times: list[float] = []  # nanoseconds

        # State persistence and trade logging (AR14, AR15)
        self._state_persistence = StatePersistence()
        self._trade_logger: TradeLogger = TradeLogger()
        self._risk_manager: RiskManager = RiskManager()

        # Data quality guards (Story 4-5)
        self._data_stale: bool = False

        # TradeStationClient — created in initialize() once auth + httpx are ready
        self._account_config: AccountConfig = _default_account_config(symbol)
        self._ts_client: Optional[TradeStationClient] = None
        self._ts_sim_mirror = None  # TSSimMirror when YANK_MIRROR_TS_SIM=1 (combine path only)

    async def initialize(self):
        _halt = Path(__file__).parent.parent.parent / "data" / "combine_joint" / "HALT"
        if _halt.exists():
            raise SystemExit(f"HALT flag present ({_halt}) — combine floor monitor halted trading; "
                             "remove the flag after review to resume.")
        logger.info("=" * 70)
        logger.info("TIER 2 FVG PAPER TRADING - SIM ORDER PLACEMENT")
        logger.info("=" * 70)
        cfg = self._strategy_config
        logger.info(
            f"Configuration: SL{cfg.sl_multiplier}x_TP{cfg.tp_multiplier}x_Midpoint_"
            f"H1_M15CHoCH_M1FVG_g{cfg.min_gap_atr_ratio} | ml_threshold={cfg.ml_threshold}"
        )
        logger.info(f"Symbol: {self._symbol} | point_value={self._point_value} tick={self._tick_size} contracts={self._contracts}")
        logger.info(f"Max hold: {cfg.max_hold_bars} bars | SL/TP mult: {cfg.sl_multiplier}x")
        logger.info(f"Entry Level: {cfg.entry_pct * 100}% (Mean Threshold)")
        logger.info(f"M15 CHoCH: REQUIRED (S25) | MIN_GAP_ATR_RATIO={cfg.min_gap_atr_ratio}")
        logger.info(f"ML Filter: {'ACTIVE' if self.ml_filter.model else 'PASS-THROUGH'} | threshold={cfg.ml_threshold}")
        logger.info("=" * 70)

        self.auth = TradeStationAuthV3.from_file('.access_token')
        await self.auth.authenticate()
        await self.auth.start_auto_refresh()
        self.client = httpx.AsyncClient(timeout=30.0)
        # Execution backend: ProjectX/TopstepX combine if PROJECTX_ACCOUNT_ID set, else SIM paper.
        # Market data stays on TradeStation REST (self.auth) either way.
        px_acct = os.environ.get("PROJECTX_ACCOUNT_ID", "")
        if px_acct:
            from src.research.projectx_auth import ProjectXAuth
            from src.research.projectx_client import ProjectXClient
            self._on_combine = True
            self._exec_account = px_acct
            self._px_auth = ProjectXAuth.from_file('.projectx_api_key')
            # Optional best-effort TradeStation SIM order mirror (default OFF). The mirror
            # can never delay/block/crash this authoritative combine path — see ts_sim_mirror.
            if os.environ.get("YANK_MIRROR_TS_SIM", "0") == "1":
                from src.research.ts_sim_mirror import TSSimMirror, MirrorProjectXClient, SimScaler
                _scaler_state = (Path(__file__).parent.parent.parent
                                 / "data" / "ts_sim_mirror" / "yank_scaler.json")
                _scaler = SimScaler("YANK", base_contracts=self._contracts,
                                    state_path=_scaler_state, log=logger)
                self._ts_sim_mirror = TSSimMirror(self.auth, scaler=_scaler, log=logger)
                await self._ts_sim_mirror.start()
                self._ts_client = MirrorProjectXClient(
                    self._px_auth, self._account_config, self.client,
                    projectx_account_id=int(px_acct), ts_mirror=self._ts_sim_mirror)
                logger.info("TS SIM MIRROR: ENABLED — combine orders also copied to %s (best-effort)",
                            SIM_ACCOUNT_ID)
            else:
                self._ts_client = ProjectXClient(self._px_auth, self._account_config, self.client,
                                                 projectx_account_id=int(px_acct))
            logger.info("EXECUTION: ProjectX/TopstepX combine acct %s | %d contracts", px_acct, self._contracts)
        else:
            self._on_combine = False
            self._exec_account = SIM_ACCOUNT_ID
            self._ts_client = TradeStationClient(self.auth, self._account_config, self.client)
            logger.info("EXECUTION: TradeStation SIM paper (%s) | %d contracts", SIM_ACCOUNT_ID, self._contracts)

        # Data backend: ProjectX bars if requested (full source or shadow), else TradeStation REST.
        # Reuses the execution ProjectXAuth when on the combine; constructs one otherwise.
        if self._data_source == "projectx" or self._data_shadow:
            self._px_data_contract_id = _to_contract_id(self._symbol)
            if self._px_auth is None:
                from src.research.projectx_auth import ProjectXAuth
                self._px_auth = ProjectXAuth.from_file('.projectx_api_key')
                await self._px_auth.start_auto_refresh()
            if self._data_shadow:
                self._shadow_logger = ShadowParityLogger(
                    Path(__file__).parent.parent.parent / "logs" / "yank_shadow_parity.csv")
        logger.info("DATA: %s (signal)%s%s", self._data_source,
                    " + projectx SHADOW" if self._data_shadow else "",
                    f" | px_contract={self._px_data_contract_id} live={self._data_px_live}"
                    if (self._data_source == "projectx" or self._data_shadow) else "")
        self.session_start_time = datetime.now()

        # Crash recovery: load persisted state and reconcile with broker (FR38, NFR11, NFR12)
        await self._recover_from_state()

    async def _recover_from_state(self) -> None:
        """Load persisted state and reconcile with broker. Called once in initialize() (AC#4–#6)."""
        state = StatePersistence.load_state()
        if state is None:
            return

        # Restore daily risk state if from the same calendar day (AC#6)
        today = datetime.now(timezone.utc).astimezone(ET_TZ).date()
        self._risk_manager.restore_from_state(state, today)

        # Reconcile active trade if state has trade fields (AC#4, AC#5)
        # Patch: guard entry_price with `is not None` (0.0 is falsy); guard entry_time to avoid KeyError
        if (
            state.get("direction")
            and state.get("entry_price") is not None
            and state.get("entry_time")
        ):
            if self._on_combine:
                # Commingling-safe: the account nets MIM+YANK, so never read net position.
                # Trust our own persisted trade; classify via our OWN entry order ID.
                from src.research.tier2_streaming_working import TradeState
                entry_id = state.get("sim_entry_order_id")
                entry_open = await self._ts_client.is_order_open(entry_id) if entry_id else None
                broker_state = TradeState(status="PENDING" if entry_open else "ACTIVE")
            else:
                broker_state = await self._ts_client.reconcile_state(SIM_ACCOUNT_ID)
            if broker_state.status == "ACTIVE":
                self.active_trade = ActiveTrade(
                    bar_index=0,
                    entry_time=datetime.fromisoformat(state["entry_time"]),
                    direction=state["direction"],
                    entry_price=float(state["entry_price"]),
                    tp_price=float(state["tp_price"]),
                    sl_price=float(state["sl_price"]),
                    sim_entry_order_id=state.get("sim_entry_order_id"),
                    sim_tp_order_id=state.get("sim_tp_order_id"),
                    sim_sl_order_id=state.get("sim_sl_order_id"),
                    pending_entry=False,
                    gap_size=float(state.get("gap_size", 0.0)),
                    h1_sweep_bars_ago=int(state.get("h1_sweep_bars_ago", 0)),
                    m15_confirmed=bool(state.get("m15_confirmed", False)),
                    kill_zone_active=bool(state.get("kill_zone_active", False)),
                    vol_regime_pct=float(state.get("vol_regime_pct", 0.0)),
                    ml_proba=float(state.get("ml_proba", float("nan"))),
                )
                logger.info("✅ Crash recovery: resumed active trade from persisted state")
            elif broker_state.status == "PENDING":
                # Orphaned pending limit order — cancel entry order and clear state
                entry_id = state.get("sim_entry_order_id")
                if entry_id:
                    try:
                        await self._ts_client.cancel_order(entry_id)
                    except Exception as e:
                        logger.warning("Could not cancel orphaned entry order %s: %s", entry_id, e)
                logger.warning(
                    "⚠️ RECONCILIATION_WARNING: state shows pending entry but broker order unconfirmed — cancelled and cleared"
                )
                StatePersistence.clear_state()
            else:
                logger.warning(
                    "⚠️ RECONCILIATION_WARNING: state shows active trade but broker has no position"
                )
                StatePersistence.clear_state()

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

    @staticmethod
    def _is_rth(now_et: datetime) -> bool:
        """Return True if now_et is within RTH (09:30–16:00 ET)."""
        h, m = now_et.hour, now_et.minute
        return (h == 9 and m >= 30) or (10 <= h <= 15) or (h == 16 and m == 0)

    def _check_stale(self, bar: DollarBar) -> bool:
        """Return True if bar timestamp is >5 min old during RTH (sets/clears _data_stale)."""
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(ET_TZ)
        if not self._is_rth(now_et):
            return False
        bar_ts = bar.timestamp if bar.timestamp.tzinfo else bar.timestamp.replace(tzinfo=timezone.utc)
        age = (now_utc - bar_ts).total_seconds()
        if age > 300:
            if not self._data_stale:
                logger.warning(
                    "STALE_DATA: last bar at %s, system time %s — halting entries",
                    bar_ts.isoformat(), now_utc.isoformat(),
                )
            self._data_stale = True
            return True
        if self._data_stale:
            logger.info("Stale data cleared — fresh bar received, resuming entries")
        self._data_stale = False
        return False

    async def stop(self):
        self.running = False
        if self.active_trade and self.dollar_bars:
            await self._close_active_trade(self.dollar_bars[-1], self.dollar_bars[-1].close, "time")
        if self._ts_sim_mirror is not None:
            await self._ts_sim_mirror.stop()
        if self.client: await self.client.aclose()
        self._print_final_report()

    async def _poll_and_process(self):
        try:
            now_utc = datetime.now(timezone.utc)
            since = self._last_processed_timestamp or (now_utc - timedelta(hours=HISTORY_HOURS))

            if self._data_source == "projectx":
                # ProjectX bars (already TS-shaped + 1-min-aligned). Skip poll on fetch error
                # (self-healing: _last_processed_timestamp only advances on a successful append).
                try:
                    bars_data = await fetch_px_ts_shaped(
                        self.client, self._px_auth, self._px_data_contract_id,
                        now_utc=now_utc, live=self._data_px_live, since_utc=since)
                except ProjectXBarFetchError as e:
                    logger.warning("PX_DATA_ERROR: %s — skipping poll", e)
                    return
            else:
                token = await self.auth.authenticate()
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                url = f"{self._bars_base_url}&firstdate={since.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                response = await self.client.get(url, headers=headers)
                if response.status_code != 200: return
                bars_data = response.json().get("Bars", [])

            if not bars_data:
                logger.warning("DATA_GAP: no bars returned at %s", now_utc.isoformat())
                return
            new_bars = []
            for bar_data in bars_data:
                bar = self._parse_bar(bar_data)
                if bar and bar.timestamp <= now_utc and (
                    not self._last_processed_timestamp or bar.timestamp > self._last_processed_timestamp
                ):
                    self.dollar_bars.append(bar)
                    # Bound the buffer to prevent unbounded memory growth (AR16)
                    if len(self.dollar_bars) > _BUFFER_CAP:
                        del self.dollar_bars[:-_BUFFER_CAP]
                    self._last_processed_timestamp = bar.timestamp
                    self._check_stale(bar)
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

                    # Per-bar timing instrumentation (AR17)
                    _t0 = _time_mod.perf_counter_ns()
                    self._update_h1_structure()
                    self._update_m15_choch()
                    await self._advance_active_trade(bar)
                    await self._detect_and_enter(bar, is_backfill=self._is_backfill)
                    self._bar_processing_times.append(_time_mod.perf_counter_ns() - _t0)

            if self._is_backfill and new_bars:
                self._is_backfill = False
                logger.info(f"✅ Tier 2 Backfill complete ({len(self.dollar_bars)} bars)")

            if self._data_shadow:
                await self._run_shadow_parity(now_utc)
        except (httpx.TimeoutException, asyncio.TimeoutError):
            logger.warning("API_TIMEOUT: request timed out — skipping bar")
        except Exception as e:
            logger.error(f"❌ Error in poll cycle: {e}", exc_info=True)

    async def _run_shadow_parity(self, now_utc):
        """Stage-1 shadow: fetch ProjectX bars in parallel and log TS-vs-PX parity to
        logs/yank_shadow_parity.csv. Observation only — never touches trade state or the
        TradeStation signal path. Every failure is swallowed (must not affect trading).

        Uses a fixed ~15-min lookback (NOT the incremental poll `since`): a settled
        minute must still be inside the compared window when it crosses the 2-min
        settle lag, or it falls through the crack (the incremental window is ~1 bar)."""
        try:
            shadow_since = now_utc - timedelta(minutes=15)
            t0 = _time_mod.perf_counter()
            error = ""
            try:
                px = await fetch_px_ts_shaped(
                    self.client, self._px_auth, self._px_data_contract_id,
                    now_utc=now_utc, live=self._data_px_live, since_utc=shadow_since)
            except ProjectXBarFetchError as e:
                px, error = [], str(e)
            fetch_ms = (_time_mod.perf_counter() - t0) * 1000.0
            px_by_min = bars_by_minute(px)
            ts_by_min = {
                b.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"):
                    (b.open, b.high, b.low, b.close, float(b.volume))
                for b in self.dollar_bars if b.timestamp >= shadow_since
            }
            self._shadow_logger.log_poll(ts_by_min, px_by_min, now_utc, fetch_ms, error)
        except Exception as e:
            logger.warning("shadow parity logging failed (non-fatal): %s", e)

    def _parse_bar(self, d: dict) -> Optional[DollarBar]:
        try:
            high_val = float(d["High"])
            low_val = float(d["Low"])
            volume = int(d.get("TotalVolume", 0))
            # Calculate a realistic notional value to pass Pydantic validation
            notional = max(((high_val + low_val) / 2) * volume * strategy_core.MNQ_NOTIONAL_MULTIPLIER, 0.01)

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
        """Resample 1m bars to H1; update H1 ATR, vol regime, and sweep state via strategy_core.

        Replaces all inline swing/sweep/ATR logic with pure strategy_core calls (Story 1.5).
        Buffer is bounded by _BUFFER_CAP (7500 M1 bars ≈ 125 H1 bars) — O(window) per bar.
        """
        if len(self.dollar_bars) < 60:
            return

        # Build canonical AR9 DataFrame from the bounded buffer (tz conversion here only, AR19)
        df = _dollar_bars_to_df(self.dollar_bars)

        # Resample to H1 via strategy_core; exclude the still-forming bar
        try:
            h1_all = resample_to_h1(df)
        except ValueError:
            return
        if len(h1_all) < 2:
            return
        h1_completed = h1_all.iloc[:-1]

        # H1 ATR (used by detect_fvg's H1-ATR ratio gate and ML features)
        self._h1_atr = calc_atr(h1_completed) if len(h1_completed) >= 2 else 0.0

        # H1 slope for ML feature extraction (unchanged)
        if len(h1_completed) >= 6 and self._h1_atr > 0:
            closes = h1_completed["close"].values[-6:]
            slope = float(np.polyfit(range(6), closes, 1)[0])
            self._h1_slope = slope / self._h1_atr
        else:
            self._h1_slope = 0.0

        # Volatility regime via strategy_core (replaces inline ATR-history + pct_rank block)
        try:
            self._vol_regime_high = not volatility_regime_filter(h1_completed, self._strategy_config)
            # Compute percentile for trade log (same formula as volatility_regime_filter)
            _h = h1_completed["high"].to_numpy(dtype=float)
            _lo = h1_completed["low"].to_numpy(dtype=float)
            _c = h1_completed["close"].to_numpy(dtype=float)
            _prev_c = np.roll(_c, 1).astype(float)
            _prev_c[0] = np.nan
            _tr = np.where(np.isnan(_prev_c), _h - _lo,
                           np.maximum(_h - _lo, np.maximum(np.abs(_h - _prev_c), np.abs(_lo - _prev_c))))
            _atr_s = pd.Series(_tr).rolling(20, min_periods=5).mean()
            _hist = [v for v in _atr_s.dropna() if v > 0]
            _hist = _hist[-self._strategy_config.vol_regime_lookback:]
            if len(_hist) >= 20:
                _cur = _hist[-1]
                self._last_vol_regime_pct = sum(1 for v in _hist if v < _cur) / len(_hist)
        except ValueError:
            self._vol_regime_high = True  # safe default: block trading when vol filter errors

        # Sweep detection via strategy_core pure scan (replaces stateful 6-hour expiry machine)
        min_rows = self._strategy_config.h1_sweep_lookback + 5
        prev_bearish_active = self.h1_bearish_sweep_active
        if len(h1_completed) >= min_rows:
            try:
                self._cached_sweep = detect_liquidity_sweep(h1_completed, self._strategy_config)
            except ValueError:
                self._cached_sweep = None
        else:
            self._cached_sweep = None

        new_bearish = (
            self._cached_sweep is not None
            and self._cached_sweep.direction == Direction.BEARISH
        )
        new_bullish = (
            self._cached_sweep is not None
            and self._cached_sweep.direction == Direction.BULLISH
        )

        # Log sweep transitions; reset M15 CHoCH on bearish sweep change
        if new_bearish and not prev_bearish_active:
            logger.info(
                f"🎯 H1 BEARISH SWEEP active (bars_ago={self._cached_sweep.bars_ago}, "
                f"price={self._cached_sweep.sweep_price:.2f})"
            )
            self._m15_choch_active = False
            self._m15_last_bar_ts = datetime.min.replace(tzinfo=timezone.utc)
        elif not new_bearish and prev_bearish_active:
            logger.info("H1 bearish sweep expired (no sweep in last 6 H1 bars)")
            self._m15_choch_active = False
            self._m15_last_bar_ts = datetime.min.replace(tzinfo=timezone.utc)

        self.h1_bearish_sweep_active = new_bearish
        self.h1_bullish_sweep_active = new_bullish

    def _update_m15_choch(self):
        """
        Scan M15 bars for bearish CHoCH (S25 architecture).
        CHoCH = last completed M15 bar closes below the most recent M15 swing low
        by ≥ 0.3 × M15 ATR.

        Only runs when H1 bearish sweep is active and CHoCH has not yet fired.
        When CHoCH fires, sets self._m15_choch_active = True (latches until sweep expires).
        """
        if not self.h1_bearish_sweep_active or self._m15_choch_active:
            return
        if len(self.dollar_bars) < 30:
            return

        recent = self.dollar_bars[-3000:]
        df = pd.DataFrame([vars(b) for b in recent])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        m15 = (
            df.set_index("timestamp")
            .resample("15min")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )

        # Need ≥ 7 completed M15 bars (5 for swing, 1 for ATR baseline, 1 excluded as forming)
        completed = m15.iloc[:-1]   # exclude currently-forming M15 bar
        n = len(completed)
        if n < 7:
            return

        # Only process if the latest completed M15 bar is newer than last checked
        last_m15_ts = completed.iloc[-1]["timestamp"].to_pydatetime()
        if last_m15_ts.tzinfo is None:
            last_m15_ts = last_m15_ts.replace(tzinfo=timezone.utc)
        if last_m15_ts <= self._m15_last_bar_ts:
            return
        self._m15_last_bar_ts = last_m15_ts

        # M15 ATR (20-bar, or all available if fewer)
        period = min(20, n - 1)
        trs = []
        for i in range(n - period, n):
            h  = float(completed.iloc[i]["high"])
            l  = float(completed.iloc[i]["low"])
            pc = float(completed.iloc[i - 1]["close"])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        m15_atr = float(np.mean(trs)) if trs else 0.0
        if m15_atr <= 0:
            return

        # Most recent M15 swing low: 2-bar symmetric radius, must be ≥ 2 bars old
        SWING_R = 2
        lows = completed["low"].values.astype(float)
        swing_low = None
        for i in range(n - 1 - SWING_R, SWING_R - 1, -1):
            lo = lows[i]
            if all(lows[i + k] >= lo for k in range(-SWING_R, SWING_R + 1) if k != 0):
                swing_low = lo
                break

        if swing_low is None:
            return

        # CHoCH check: last completed M15 bar closes below swing_low − 0.3×ATR
        last_close = float(completed.iloc[-1]["close"])
        CHOCH_ATR_MULT = 0.3
        if last_close < swing_low - CHOCH_ATR_MULT * m15_atr:
            self._m15_choch_active = True
            logger.info(
                f"🔑 M15 CHoCH confirmed: close={last_close:.2f} < "
                f"swing_low={swing_low:.2f} − 0.3×ATR({m15_atr:.2f}={swing_low - CHOCH_ATR_MULT*m15_atr:.2f})"
            )

    async def _advance_active_trade(self, bar: DollarBar) -> bool:
        if not self.active_trade: return False
        t = self.active_trade
        t.bars_held += 1

        # ── Topstep combine: flatten by 15:10 CT, never carry across the close ──
        # Only the [15:10, 17:00) CT close window — evening/Globex (17:00 CT+) is a new session.
        bar_ct = bar.timestamp.astimezone(CT_TZ)
        _ct_min = bar_ct.hour * 60 + bar_ct.minute
        if TOPSTEP_FLATTEN_MIN <= _ct_min < TOPSTEP_BLOCK_HI:
            if t.pending_entry:
                for oid in [t.sim_entry_order_id, t.sim_tp_order_id, t.sim_sl_order_id]:
                    if oid: await self._ts_client.cancel_order(oid)
                self.active_trade = None
                self._active_entry_decision = None
                logger.info("🏁 Topstep 15:10 CT flatten — cancelled unfilled pending entry")
                return False
            logger.info("🏁 Topstep 15:10 CT flatten — closing active position at market")
            await self._close_active_trade(bar, bar.close, "time")
            return True

        # ── Pending limit entry: wait for price to reach FVG midpoint ──────────
        if t.pending_entry:
            filled = (
                (t.direction == "SHORT" and bar.high >= t.entry_price) or
                (t.direction == "LONG"  and bar.low  <= t.entry_price)
            )
            if filled:
                t.pending_entry = False
                t.bars_held = 0  # reset so MAX_HOLD_BARS counts from fill, not signal
                logger.info(f"✅ Limit entry FILLED at {t.entry_price:.2f}")
                # ProjectX defers TP/SL until the entry fills — place them now (commingling-safe:
                # they become this bot's own order IDs). TradeStation already placed them at submit.
                if self._on_combine and t.sim_tp_order_id is None and self._active_entry_decision is not None:
                    tp_id, sl_id = await self._ts_client.place_exit_orders(self._active_entry_decision, self._exec_account)
                    t.sim_tp_order_id, t.sim_sl_order_id = tp_id, sl_id
                    logger.info("TP/SL placed on fill (ProjectX): tp #%s sl #%s", tp_id, sl_id)
                # Fall through to TP/SL check — might hit in the same bar
            elif t.bars_held >= self._strategy_config.max_pending_bars:
                for oid in [t.sim_entry_order_id, t.sim_tp_order_id, t.sim_sl_order_id]:
                    if oid: await self._ts_client.cancel_order(oid)
                self.active_trade = None
                self._active_entry_decision = None
                logger.info(f"⏱ Limit entry expired ({self._strategy_config.max_pending_bars} bars, not filled)")
                return False
            else:
                return False  # still waiting for fill

        # ── Active trade: check TP / SL / time-stop via strategy_core.check_exit ──
        if self._active_entry_decision is not None:
            bar_series = pd.Series({"high": bar.high, "low": bar.low, "close": bar.close})
            exit_dec = check_exit(bar_series, self._active_entry_decision, t.bars_held, self._strategy_config)
            if exit_dec is not None:
                _reason_map = {
                    ExitReason.TP: "tp",
                    ExitReason.SL: "sl",
                    ExitReason.TIME_STOP: "time",
                    ExitReason.MANUAL: "time",
                }
                await self._close_active_trade(bar, exit_dec.exit_price, _reason_map[exit_dec.reason])
                return True
        return False

    async def _close_active_trade(self, bar: DollarBar, price: float, reason: str):
        t = self.active_trade
        if reason == "time":
            if t.sim_tp_order_id: await self._ts_client.cancel_order(t.sim_tp_order_id)
            if t.sim_sl_order_id: await self._ts_client.cancel_order(t.sim_sl_order_id)
            await self._ts_client.close_position_at_market(t.direction, self._exec_account, self._contracts)
        else:
            # Bracket leg hit (TP or SL) - cancel the other leg
            other_id = t.sim_sl_order_id if reason == "tp" else t.sim_tp_order_id
            if other_id: await self._ts_client.cancel_order(other_id)

        cfg = self._strategy_config
        pnl = ((price - t.entry_price) if t.direction == "LONG" else (t.entry_price - price)) * self._point_value * self._contracts - cfg.commission_per_roundtrip
        self._risk_manager.register_close(pnl)
        self.completed_trades.append(CompletedTrade(
            t.entry_time, bar.timestamp, t.direction, t.entry_price, price, reason, t.bars_held, pnl
        ))
        self.active_trade = None
        self._active_entry_decision = None
        # Persist risk state after close so daily circuit breaker survives restart (NFR12, AC#7)
        # Patch: save_state() wrapped so failure never prevents the trade record from being written
        try:
            StatePersistence.save_state(self._risk_manager.to_state_dict())
        except Exception as e:
            logger.warning("State persistence failed after trade close (trade record will still be written): %s", e)
        logger.info(f"Trade Closed: {reason.upper()} | P&L: ${pnl:.2f} | Daily P&L: ${self._risk_manager.daily_pnl:.2f}")
        _exit_reason_str = reason.upper() if reason in ("tp", "sl") else "TIME_STOP" if reason == "time" else reason.upper()
        self._trade_logger.append_trade(TradeRecord(
            timestamp_entry=t.entry_time,
            timestamp_exit=bar.timestamp,
            direction=t.direction,
            entry_price=t.entry_price,
            exit_price=price,
            tp_price=t.tp_price,
            sl_price=t.sl_price,
            gap_size=t.gap_size,
            pnl_usd=pnl,
            exit_reason=_exit_reason_str,
            h1_sweep_bars_ago=t.h1_sweep_bars_ago,
            m15_confirmed=t.m15_confirmed,
            kill_zone_active=t.kill_zone_active,
            vol_regime_pct=t.vol_regime_pct,
            contracts=self._contracts,
        ))
        self._log_ml_canary(t, _exit_reason_str, pnl)
        self._log_trade_metrics()
        self._write_equity_curve()

    def _log_ml_canary(self, t: "ActiveTrade", exit_reason: str, pnl: float) -> None:
        """Passive drift canary: pair the entry meta-model P(success) with the realized
        outcome (logs/yank_ml_canary.csv). Logging only — never gates a trade. Rolled up
        weekly (rolling AUC/Brier vs the ~0.50 baseline) per the 2026-06-16 party-mode
        decision: keep the PF<0.90-after-N>=20 P&L guardrail as the actuator; observe here.

        Skipped when the ML model is inactive (pass-through) or no proba was captured
        (e.g. a trade resumed via crash recovery before this field was persisted)."""
        if self.ml_filter.model is None or np.isnan(t.ml_proba):
            return
        import csv as _csv
        log_path = Path(__file__).parent.parent.parent / "logs" / "yank_ml_canary.csv"
        row = {
            "timestamp_entry": str(t.entry_time),
            "direction":       t.direction,
            "ml_proba":        round(float(t.ml_proba), 4),
            "threshold":       self.ml_filter.threshold,
            "exit_reason":     exit_reason,
            "pnl_usd":         round(float(pnl), 2),
            "win":             int(pnl > 0),
        }
        write_header = not log_path.exists()
        try:
            with log_path.open("a", newline="") as _f:
                _w = _csv.DictWriter(_f, fieldnames=list(row.keys()))
                if write_header:
                    _w.writeheader()
                _w.writerow(row)
        except Exception as _e:
            logger.warning(f"ML canary log write failed: {_e}")

    # _log_trade() removed in Story 4-2 — replaced by TradeLogger.append_trade() (AC#1, AC#2)
    # _check_daily_reset_and_halt() removed in Story 4-3 — replaced by RiskManager.check_and_update() (AC#1)

    def _log_trade_metrics(self) -> None:
        """Log PF/Sharpe/MaxDD/trade-count after each close (FR30–FR33, AC#1, AC#2, AC#5)."""
        pnls = [t.pnl for t in self.completed_trades]
        n = len(pnls)
        if not pnls:
            logger.info("PF: N/A | Sharpe: N/A | MaxDD: 0.0%% | Trades: 0")
            return
        pf = calc_profit_factor(pnls)
        pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
        sh = calc_sharpe(pnls)
        cum = list(itertools.accumulate(pnls))
        dd_pct = calc_max_drawdown_pct(cum)
        logger.info("PF: %s | Sharpe: %.2f | MaxDD: %.1f%% | Trades: %d",
                    pf_str, sh, dd_pct * 100, n)

    def _write_equity_curve(self) -> None:
        """Append one row to logs/equity_curve.csv after each trade close (FR34, AC#3)."""
        try:
            pnls = [t.pnl for t in self.completed_trades]
            cum_pnl = sum(pnls)
            n = len(pnls)
            log_path = Path(__file__).parent.parent.parent / "logs" / "equity_curve.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not log_path.exists()
            with log_path.open("a", newline="") as f:
                w = _csv_mod.DictWriter(f, fieldnames=["timestamp", "cumulative_pnl_usd", "trade_count"])
                if write_header:
                    w.writeheader()
                w.writerow({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "cumulative_pnl_usd": round(cum_pnl, 2),
                    "trade_count": n,
                })
        except Exception as e:
            logger.warning("Equity curve write failed: %s", e)

    def _log_filter_decision(
        self,
        bar_timestamp: datetime,
        h1_sweep_active: bool,
        kill_zone_active: bool,
        vol_regime_blocked: bool,
        m15_confirmed: bool,
        fvg_detected: bool,
        action: str,
    ) -> None:
        """Append one per-bar filter decision row to logs/tier2_bar_decisions.csv (FR35, AC#4)."""
        # Do NOT log during the startup backfill: those bars are historical and were
        # re-logged on every restart, ballooning the file (9.2M rows / 631MB observed).
        # Only live (steady-state) bars should produce a decision trail.
        if self._is_backfill:
            return  # backfill bars are historical — see comment above
        try:
            log_path = Path(__file__).parent.parent.parent / "logs" / "tier2_bar_decisions.csv"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not log_path.exists()
            with log_path.open("a", newline="") as f:
                w = _csv_mod.DictWriter(f, fieldnames=[
                    "bar_timestamp", "h1_sweep_active", "kill_zone_active",
                    "vol_regime_blocked", "m15_confirmed", "fvg_detected", "action",
                ])
                if write_header:
                    w.writeheader()
                w.writerow({
                    "bar_timestamp": bar_timestamp.isoformat(),
                    "h1_sweep_active": h1_sweep_active,
                    "kill_zone_active": kill_zone_active,
                    "vol_regime_blocked": vol_regime_blocked,
                    "m15_confirmed": m15_confirmed,
                    "fvg_detected": fvg_detected,
                    "action": action,
                })
        except Exception as e:
            logger.warning("Filter decision log write failed: %s", e)

    async def _detect_and_enter(self, bar: DollarBar, is_backfill: bool):
        if self._data_stale:
            return
        bar_et = bar.timestamp.astimezone(ET_TZ)
        if self.active_trade:
            self._log_filter_decision(bar_et, self.h1_bearish_sweep_active, False, False,
                                      self._m15_choch_active, False, "HOLD")
            return
        bars = self.dollar_bars
        if len(bars) < 20: return  # need 20 bars for ATR and volume features

        # Topstep combine: no new entries 15:08-17:00 CT (positions auto-flatten at 15:10 CT)
        bar_ct = bar.timestamp.astimezone(CT_TZ)
        ct_min = bar_ct.hour * 60 + bar_ct.minute
        if TOPSTEP_BLOCK_LO <= ct_min < TOPSTEP_BLOCK_HI:
            return

        # Tuesday filter: consistently PF<1.0 across all 5 months of backtest data
        if bar_et.weekday() == 1:  # 1 = Tuesday
            return

        # Daily circuit breaker: halt if daily loss limit reached
        if self._risk_manager.check_and_update(bar_et, self._strategy_config.max_daily_loss):
            return

        # Seasonality gate: skip months with statistically zero edge (default: none blocked)
        if bar_et.month in BLOCKED_MONTHS:
            return

        # Volatility regime gate: skip all signals when H1 ATR in top quartile of recent history
        if self._vol_regime_high:
            self._log_filter_decision(bar_et, self.h1_bearish_sweep_active, False, True,
                                      self._m15_choch_active, False, "SKIP")
            return

        # FVG detection via strategy_core (replaces self._detect_fvg)
        # Use the last 20 M1 bars — sufficient for the 3-bar FVG check + 20-bar ATR window.
        m1_df = _dollar_bars_to_df(bars[-20:])
        cached_sweep = self._cached_sweep

        # Direction gate: bearish-only by default (full-year 2025: bearish PF 1.43 vs bullish 1.06)
        if not self._strategy_config.bearish_only and self.h1_bullish_sweep_active and cached_sweep is not None:
            try:
                fvg_signal = detect_fvg(m1_df, self._strategy_config, self._h1_atr)
            except ValueError:
                fvg_signal = None
            if fvg_signal and fvg_signal.direction == Direction.BULLISH:
                fvg_dict = {"direction": "bullish", "top": fvg_signal.high, "bottom": fvg_signal.low}
                if not self.lr_filter.allows(bars, "bullish"):
                    return
                features = self._extract_features(bars, bar, fvg_dict, "bullish")
                proba = self.ml_filter.predict_proba(features)
                if proba >= self.ml_filter.threshold:
                    logger.info(f"Signal ALLOWED by ML threshold | P(Success)={proba:.3f}")
                    self.ml_filter._log_decision(bar.timestamp, proba, "ALLOWED")
                    await self._enter_trade(fvg_signal, bar, len(bars) - 1, is_backfill, ml_proba=proba)
                else:
                    logger.info(f"Signal FILTERED by ML threshold | P(Success)={proba:.3f} < {self.ml_filter.threshold}")
                    self.ml_filter._log_decision(bar.timestamp, proba, "FILTERED")

        if self.h1_bearish_sweep_active and self._m15_choch_active:  # S25: M15 CHoCH required
            try:
                fvg_signal = detect_fvg(m1_df, self._strategy_config, self._h1_atr)
            except ValueError:
                fvg_signal = None
            _fvg_hit = bool(fvg_signal and fvg_signal.direction == Direction.BEARISH and cached_sweep is not None)
            if _fvg_hit:
                fvg_dict = {"direction": "bearish", "top": fvg_signal.high, "bottom": fvg_signal.low}  # type: ignore[union-attr]
                if not self.lr_filter.allows(bars, "bearish"):
                    self._log_filter_decision(bar_et, True, False, False, True, True, "SKIP")
                    return
                features = self._extract_features(bars, bar, fvg_dict, "bearish")
                proba = self.ml_filter.predict_proba(features)
                if proba >= self.ml_filter.threshold:
                    logger.info(f"Signal ALLOWED by ML threshold | P(Success)={proba:.3f}")
                    self.ml_filter._log_decision(bar.timestamp, proba, "ALLOWED")
                    self._log_filter_decision(bar_et, True, False, False, True, True, "ENTER")
                    await self._enter_trade(fvg_signal, bar, len(bars) - 1, is_backfill, ml_proba=proba)  # type: ignore[arg-type]
                else:
                    logger.info(f"Signal FILTERED by ML threshold | P(Success)={proba:.3f} < {self.ml_filter.threshold}")
                    self.ml_filter._log_decision(bar.timestamp, proba, "FILTERED")
                    self._log_filter_decision(bar_et, True, False, False, True, True, "SKIP")
            else:
                self._log_filter_decision(bar_et, True, False, False, True, False, "SKIP")
        else:
            # No bearish sweep / M15 CHoCH this bar — log the live no-setup decision so
            # the steady-state trail isn't silent on the dominant case.
            self._log_filter_decision(bar_et, self.h1_bearish_sweep_active, False, False,
                                      self._m15_choch_active, False, "SKIP")

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

    def _calculate_atr(self, bars: list) -> float:
        """Delegate to strategy_core.calc_atr after converting DollarBar list to DataFrame."""
        if len(bars) < 20:
            return 10.0
        return calc_atr(_dollar_bars_to_df(bars[-20:]))

    def _snap_tick(self, price: float) -> float:
        """Round price to nearest instrument tick. Avoids float artifacts."""
        return round(round(price / self._tick_size) * self._tick_size, 10)

    async def _enter_trade(self, fvg: FVGSignal, bar: DollarBar, idx: int, is_backfill: bool,
                           ml_proba: float = float("nan")):
        """Resolve entry via strategy_core.make_entry_decision and arm the ActiveTrade.

        ``fvg`` is now a ``FVGSignal`` from ``strategy_core.detect_fvg``.
        Tick-snapping is applied to the prices before SIM order submission.
        ``self._active_entry_decision`` stores the *snapped* EntryDecision for ``check_exit``.
        """
        self._last_entry_bar = len(self.dollar_bars)
        if is_backfill:
            return

        sweep = self._cached_sweep
        if sweep is None:
            return

        entry_dec = make_entry_decision(sweep, fvg, self._strategy_config)
        if entry_dec is None:
            return

        # Snap prices to MNQ tick for SIM order submission
        ent = self._snap_tick(entry_dec.entry_price)
        tp = self._snap_tick(entry_dec.tp_price)
        sl = self._snap_tick(entry_dec.sl_price)

        # Rebuild EntryDecision with snapped prices — these are authoritative for check_exit
        snapped_dec = EntryDecision(
            direction=entry_dec.direction,
            entry_price=ent,
            sl_price=sl,
            tp_price=tp,
            contracts=self._contracts,
        )
        self._active_entry_decision = snapped_dec

        direction_str = "LONG" if entry_dec.direction == Direction.BULLISH else "SHORT"
        # Capture entry-context metadata for trade log (AC#1, Story 4-2)
        _gap_size = fvg.gap_size
        _h1_sweep_bars_ago = self._cached_sweep.bars_ago if self._cached_sweep else 0
        _m15_confirmed = self._m15_choch_active
        _kill_zone_active = kill_zone_filter(bar.timestamp, self._strategy_config)
        _vol_regime_pct = self._last_vol_regime_pct

        e_id, tp_id, sl_id = await self._ts_client.submit_bracket_order(snapped_dec, self._exec_account)
        self.active_trade = ActiveTrade(
            idx, bar.timestamp, direction_str, ent, tp, sl,
            sim_entry_order_id=e_id, sim_tp_order_id=tp_id, sim_sl_order_id=sl_id,
            gap_size=_gap_size,
            h1_sweep_bars_ago=_h1_sweep_bars_ago,
            m15_confirmed=_m15_confirmed,
            kill_zone_active=_kill_zone_active,
            vol_regime_pct=_vol_regime_pct,
            ml_proba=ml_proba,
        )
        # Persist active-trade state + daily risk for crash recovery (AR14, AR15, NFR12)
        StatePersistence.save_state({
            "direction": direction_str,
            "entry_price": ent,
            "tp_price": tp,
            "sl_price": sl,
            "entry_time": bar.timestamp.isoformat(),
            "sim_entry_order_id": e_id,
            "sim_tp_order_id": tp_id,
            "sim_sl_order_id": sl_id,
            "gap_size": _gap_size,
            "h1_sweep_bars_ago": _h1_sweep_bars_ago,
            "m15_confirmed": _m15_confirmed,
            "kill_zone_active": _kill_zone_active,
            "vol_regime_pct": _vol_regime_pct,
            "ml_proba": ml_proba,
            **self._risk_manager.to_state_dict(),
        })
        logger.info(f"🔔 TIER 2 LIMIT PLACED: {direction_str} limit=${ent:.2f} | TP ${tp:.2f} SL ${sl:.2f}")

    def _print_final_report(self):
        logger.info("Tier 2 Paper Trading Session Ended.")
        if self.completed_trades:
            logger.info(f"Completed trades this session: {len(self.completed_trades)}")
        # Per-bar timing report (AR17)
        if self._bar_processing_times:
            n = len(self._bar_processing_times)
            sorted_times = sorted(self._bar_processing_times)
            p50 = sorted_times[n // 2] / 1_000
            p95 = sorted_times[int(n * 0.95)] / 1_000
            mx = sorted_times[-1] / 1_000
            logger.info(
                f"Per-bar processing latency ({n} bars): "
                f"p50={p50:.1f}µs  p95={p95:.1f}µs  max={mx:.1f}µs"
            )


async def main():
    import argparse as _argparse
    parser = _argparse.ArgumentParser(description="Tier 2 FVG Paper Trader")
    parser.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", "MNQM26"),
        help="Futures symbol to trade (default: MNQM26 or $SYMBOL env var)",
    )
    args = parser.parse_args()
    trader = Tier2StreamingTrader(symbol=args.symbol)
    await trader.initialize()
    await trader.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())
