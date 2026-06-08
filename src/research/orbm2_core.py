"""orbm2_core.py — ORBM-2 (ORB Breakout Momentum v2) entry/exit logic.

Enters IN THE DIRECTION of the ORB extension. Imports shared ORB primitives
from sorm_core (build_opening_range, detect_extension, load_bars_et).

Pre-registration: 16abdd9 (2026-06-08). Config sealed in orbm2_config.yaml.

Direction convention (inherited from sorm_core):
    Direction.BEARISH = upward extension (close > ORB_high + threshold)
    Direction.BULLISH = downward extension (close < ORB_low − threshold)

ORBM-2 mapping:
    BEARISH extension → LONG  (go with the upward breakout)
    BULLISH extension → SHORT (go with the downward breakout)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import yaml

from src.research.sorm_core import (
    POINT_VALUE_USD,
    TICK_SIZE,
    Direction,
    ExtensionEvent,
    OpeningRange,
)

# ── Config ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ORBM2Config:
    """Immutable ORBM-2 parameters — duck-typed to work with sorm_core functions."""
    # Timing (ET) — same attributes sorm_core functions need
    orb_start_et: time = time(9, 30)
    orb_end_et: time = time(9, 45)
    extension_start_et: time = time(9, 45)
    extension_end_et: time = time(10, 45)
    hard_close_et: time = time(11, 30)
    # Signal
    extension_threshold: float = 0.25   # ORBM-2: 0.25× (lowered for frequency)
    orb_min_size_points: float = 5.0    # skip sessions with tiny ranges
    # Stop
    stop_cap_pts: float = 75.0          # skip if stop > 75 pts ($150/contract)
    # Take profit
    tp_r_multiple: float = 1.5          # 1.5R target in continuation direction
    # Sizing
    contracts_small_stop: int = 2       # stop < 50 pts
    contracts_large_stop: int = 1       # stop 50–75 pts
    contracts_small_threshold_pts: float = 50.0
    # Session management
    max_trades_per_session: int = 1
    daily_loss_limit_usd: float = -200.0
    daily_profit_halt_usd: float = 750.0
    # Market
    point_value_usd: float = POINT_VALUE_USD
    tick_size: float = TICK_SIZE
    commission_per_contract_rt: float = 0.40


def load_orbm2_config(path: Optional[Path] = None) -> ORBM2Config:
    """Load ORBM-2 config from orbm2_config.yaml.

    Falls back to defaults if the file is not found (tests, etc.).
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "orbm2_config.yaml"
    if not path.exists():
        return ORBM2Config()
    raw = yaml.safe_load(path.read_text())

    def _t(key: str, default: time) -> time:
        v = raw.get(key)
        if v is None:
            return default
        if isinstance(v, time):
            return v
        h, m = str(v).split(":")
        return time(int(h), int(m))

    return ORBM2Config(
        orb_start_et=_t("orb_start_et", time(9, 30)),
        orb_end_et=_t("orb_end_et", time(9, 45)),
        extension_start_et=_t("extension_start_et", time(9, 45)),
        extension_end_et=_t("extension_end_et", time(10, 45)),
        hard_close_et=_t("hard_close_et", time(11, 30)),
        extension_threshold=float(raw.get("extension_threshold", 0.25)),
        orb_min_size_points=float(raw.get("orb_min_size_points", 5.0)),
        stop_cap_pts=float(raw.get("stop_cap_pts", 75.0)),
        tp_r_multiple=float(raw.get("tp_r_multiple", 1.5)),
        contracts_small_stop=int(raw.get("contracts_small_stop", 2)),
        contracts_large_stop=int(raw.get("contracts_large_stop", 1)),
        daily_loss_limit_usd=float(raw.get("daily_loss_limit_usd", -200.0)),
        daily_profit_halt_usd=float(raw.get("daily_profit_halt_usd", 750.0)),
        commission_per_contract_rt=float(raw.get("commission_per_contract_rt", 0.40)),
    )


# ── Trade Setup ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ORBM2Trade:
    """A fully specified ORBM-2 entry setup."""
    date_et: date
    direction: Direction          # BEARISH = LONG, BULLISH = SHORT
    entry: float                  # extension bar close
    stop: float                   # ORB boundary ± 1 tick
    tp: float                     # 1.5R in continuation direction
    stop_pts: float               # entry − stop (always positive)
    stop_usd: float               # stop_pts × point_value per contract
    contracts: int                # 0 = skip


def build_orbm2_trade(
    ext: ExtensionEvent,
    orb: OpeningRange,
    cfg: ORBM2Config,
) -> ORBM2Trade:
    """Compute ORBM-2 trade parameters from a detected extension.

    For LONG (BEARISH extension / upward breakout):
        entry = extension_close  (above ORB_high)
        stop  = ORB_high − 1 tick  (re-enters range → thesis dead)
        tp    = entry + 1.5 × (entry − stop)

    For SHORT (BULLISH extension / downward breakout):
        entry = extension_close  (below ORB_low)
        stop  = ORB_low + 1 tick
        tp    = entry − 1.5 × (stop − entry)
    """
    entry = ext.extension_close

    if ext.direction == Direction.BEARISH:  # upward → LONG
        stop = orb.high - cfg.tick_size
        stop_pts = entry - stop
        tp = entry + cfg.tp_r_multiple * stop_pts
    else:                                   # downward → SHORT
        stop = orb.low + cfg.tick_size
        stop_pts = stop - entry
        tp = entry - cfg.tp_r_multiple * stop_pts

    stop_pts = max(stop_pts, 0.0)
    stop_usd = stop_pts * cfg.point_value_usd

    if stop_pts > cfg.stop_cap_pts:
        contracts = 0
    elif stop_pts < cfg.contracts_small_threshold_pts:
        contracts = cfg.contracts_small_stop
    else:
        contracts = cfg.contracts_large_stop

    return ORBM2Trade(
        date_et=ext.detection_bar_ts.date(),
        direction=ext.direction,
        entry=entry,
        stop=stop,
        tp=tp,
        stop_pts=stop_pts,
        stop_usd=stop_usd,
        contracts=contracts,
    )


# ── Trade Simulation ─────────────────────────────────────────────────────────

@dataclass
class ORBM2Result:
    """Outcome of one simulated ORBM-2 trade."""
    date_et: date
    direction: str                # "LONG" or "SHORT"
    entry: float
    exit_price: float
    stop: float
    tp: float
    stop_pts: float
    contracts: int
    pnl_gross: float              # before commission
    pnl_net: float                # after commission
    exit_reason: str              # "TP" | "SL" | "TIME_STOP"
    entry_ts: datetime
    exit_ts: Optional[datetime]
    qualifying: bool              # session P&L ≥ $150


def simulate_orbm2_trade(
    post_entry_df,                # bars AFTER the extension bar (ET DatetimeIndex)
    trade: ORBM2Trade,
    hard_close_str: str,          # e.g. "11:30"
    cfg: ORBM2Config,
) -> ORBM2Result:
    """Walk 1-min bars to determine trade outcome.

    Stop check is applied before TP check on any given bar (conservative).
    If both stop and TP are touched on the same bar, the stop is assumed to
    have been hit first (worst-case for the strategy).
    """
    direction_label = "LONG" if trade.direction == Direction.BEARISH else "SHORT"
    entry_ts = post_entry_df.index[0] if not post_entry_df.empty else None
    exit_price = trade.entry
    exit_reason = "TIME_STOP"
    exit_ts: Optional[datetime] = None

    tradeable_bars = post_entry_df.between_time("00:00", hard_close_str, inclusive="left")

    for ts, row in tradeable_bars.iterrows():
        bar_high = float(row["high"])
        bar_low  = float(row["low"])
        bar_close = float(row["close"])

        if trade.direction == Direction.BEARISH:  # LONG
            if bar_low <= trade.stop:
                exit_price = trade.stop
                exit_reason = "SL"
                exit_ts = ts
                break
            if bar_high >= trade.tp:
                exit_price = trade.tp
                exit_reason = "TP"
                exit_ts = ts
                break
        else:  # SHORT
            if bar_high >= trade.stop:
                exit_price = trade.stop
                exit_reason = "SL"
                exit_ts = ts
                break
            if bar_low <= trade.tp:
                exit_price = trade.tp
                exit_reason = "TP"
                exit_ts = ts
                break

        exit_price = bar_close  # running exit if TIME_STOP

    if exit_ts is None and not tradeable_bars.empty:
        exit_ts = tradeable_bars.index[-1]

    # P&L
    if trade.direction == Direction.BEARISH:  # LONG
        raw_pts = exit_price - trade.entry
    else:                                     # SHORT
        raw_pts = trade.entry - exit_price

    pnl_gross = raw_pts * cfg.point_value_usd * trade.contracts
    commission = cfg.commission_per_contract_rt * trade.contracts
    pnl_net = pnl_gross - commission

    return ORBM2Result(
        date_et=trade.date_et,
        direction=direction_label,
        entry=trade.entry,
        exit_price=exit_price,
        stop=trade.stop,
        tp=trade.tp,
        stop_pts=trade.stop_pts,
        contracts=trade.contracts,
        pnl_gross=pnl_gross,
        pnl_net=pnl_net,
        exit_reason=exit_reason,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        qualifying=False,  # filled by the backtest after session aggregation
    )
