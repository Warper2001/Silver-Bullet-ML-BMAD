# Story 4.2: Trade Logging and StatePersistence with Crash Recovery

Status: ready-for-dev

## Story

As Alex,
I want `TradeLogger` to append every completed trade to a CSV and `StatePersistence` to persist full system state (including daily risk) after every bar,
So that no trade is ever lost and the system can resume cleanly after a crash without double-counting positions.

## Background

**Current state (after Story 4-1):**

- `StatePersistence` class exists (`tier2_streaming_working.py` lines 173–202) with atomic write (`write_text` → `os.replace()`). Works correctly for the atomic-write contract.
- `save_state()` is called in `_enter_trade()` (line 1281) — saves direction, prices, order IDs, and entry_time. **But it is NEVER called after trades close, and it does NOT persist daily P&L or the circuit-breaker state.**
- `load_state()` is **never called** in `initialize()` — crash recovery is half-implemented. On restart, the bot starts with a fresh empty state while TradeStation may still have an open bracket. This is the primary bug this story fixes.
- `_log_trade()` exists (lines 1028–1061) as a raw method on `Tier2StreamingTrader`. Its column schema is wrong: it writes `{timestamp, instrument, direction, entry_price, exit_price, exit_reason, bars_held, pnl_usd}` — missing `tp_price, sl_price, gap_size, h1_sweep_bars_ago, m15_confirmed, kill_zone_active, vol_regime_pct, contracts`. The PRD mandates a specific column order (AR15, FR29).
- `_daily_pnl`, `_daily_halted`, `_last_trading_date` are in-memory only — not persisted. A crash resets the daily circuit breaker (violates NFR12).
- The CSV header check has a TOCTOU race: `write_header = not log_path.exists()` — two simultaneously starting processes can both write the header. Fix: check `f.tell() == 0` inside the `with open(...)` context.

**This story:**
1. Creates a `TradeLogger` class (inside `tier2_streaming_working.py`, AR3 single-file mandate) with `append_trade()` writing the full PRD column set.
2. Extends `StatePersistence.save_state()` to include daily risk state (`daily_pnl`, `daily_halted`, `last_trading_date`).
3. Implements crash recovery in `initialize()`: calls `load_state()` → `reconcile_state()` → resumes or warns.
4. Writes unit tests covering all crash-recovery scenarios.

## Acceptance Criteria

**AC#1 — Trade log column schema:**
Given a trade completes (TP, SL, or TIME_STOP exit),
When `TradeLogger.append_trade(record: TradeRecord)` is called,
Then one row is appended to `logs/tier2_trade_log.csv` with ALL PRD columns in order:
`timestamp_entry, timestamp_exit, direction, entry_price, exit_price, tp_price, sl_price, gap_size, pnl_usd, exit_reason, h1_sweep_bars_ago, m15_confirmed, kill_zone_active, vol_regime_pct, contracts`
where `exit_reason` is `ExitReason.value` string (FR29, NFR9, AR15).

**AC#2 — Append-only concurrency safety:**
Given the trade log CSV is open mid-session and another process reads it concurrently,
When `TradeLogger.append_trade()` writes a row,
Then the CSV is not corrupted — specifically: the header is written only once (TOCTOU race fixed by checking `f.tell() == 0` inside the `with open(...)` context, not via `path.exists()` before open).

**AC#3 — Atomic state write:**
Given `StatePersistence.save_state(state: dict)` is called after every bar with active/pending trade,
When the state file path is checked,
Then the write went to `logs/active_trade_state.tmp` then `os.replace()` to `logs/active_trade_state.json` (atomic write, AR14, already implemented — must remain intact).

**AC#4 — Crash recovery: ACTIVE position:**
Given `StatePersistence.load_state()` returns a dict with active-trade fields,
When `TradeStationClient.reconcile_state(SIM_ACCOUNT_ID)` is called and returns `status="ACTIVE"`,
Then `Tier2StreamingTrader.initialize()` reconstructs `self.active_trade` from the persisted state so the bot resumes managing the open trade (FR38, NFR11).

**AC#5 — Crash recovery: FLAT — RECONCILIATION_WARNING:**
Given `StatePersistence.load_state()` returns a dict with active-trade fields,
When `reconcile_state()` returns `status="FLAT"` (position already closed at broker),
Then `initialize()` logs `RECONCILIATION_WARNING: state shows active trade but broker has no position` and calls `StatePersistence.clear_state()` — no double-entry (FR38, NFR11).

**AC#6 — Circuit breaker survives restart:**
Given `daily_pnl`, `daily_halted`, and `last_trading_date` are saved inside the same state dict,
When the process restarts within the same calendar day and `load_state()` is called,
Then `self._daily_pnl`, `self._daily_halted`, and `self._last_trading_date` are restored from state; if `daily_halted=True` from the persisted state, `_check_daily_reset_and_halt()` returns `True` immediately without re-crossing the threshold (NFR12).

**AC#7 — State includes risk fields after every trade close:**
Given a trade closes (TP, SL, or TIME_STOP),
When `_close_active_trade()` finishes,
Then if another trade was not immediately entered, `StatePersistence.save_state()` is called with a state dict that includes `daily_pnl`, `daily_halted`, `last_trading_date` but no `active_trade` key (or active_trade=None), allowing a restart to see the correct daily loss without an orphaned active-trade confusion.

**AC#8 — Unit tests:**
Given `tests/unit/test_trade_logging_state_persistence.py`,
When pytest runs it,
Then all tests pass covering:
- `TradeLogger.append_trade()` writes correct PRD column order
- `TradeLogger` header written once for empty file, not re-written on second call
- `StatePersistence` roundtrip: save → load returns identical dict
- Crash recovery: active-trade state + broker ACTIVE → `active_trade` reconstructed
- Crash recovery: active-trade state + broker FLAT → RECONCILIATION_WARNING + clear_state
- Circuit breaker: `daily_halted=True` in state → `_check_daily_reset_and_halt()` returns True

## Tasks / Subtasks

- [ ] Task 1: Create `TradeRecord` dataclass and `TradeLogger` class (AC: #1, #2)
  - [ ] Define `@dataclass class TradeRecord` near top of file (after `TradeState`) with fields: `timestamp_entry: datetime`, `timestamp_exit: datetime`, `direction: str`, `entry_price: float`, `exit_price: float`, `tp_price: float`, `sl_price: float`, `gap_size: float`, `pnl_usd: float`, `exit_reason: str`, `h1_sweep_bars_ago: int`, `m15_confirmed: bool`, `kill_zone_active: bool`, `vol_regime_pct: float`, `contracts: int`
  - [ ] Create `TradeLogger` class (placed AFTER `StatePersistence`, BEFORE `TradeStationClient`) with `_LOG_PATH = Path(...) / "logs/tier2_trade_log.csv"` and `_COLUMNS = ["timestamp_entry", "timestamp_exit", ...]` (PRD column order)
  - [ ] Implement `TradeLogger.append_trade(self, record: TradeRecord) -> None` — opens CSV in append mode, checks `f.tell() == 0` to decide whether to write header, writes the row
  - [ ] Error handling: wrap write in try/except, log warning on failure — never raise from this method
  - [ ] Remove or stub out `Tier2StreamingTrader._log_trade()` — the new TradeLogger replaces it

- [ ] Task 2: Extend `StatePersistence.save_state()` schema to include risk state (AC: #3, #6, #7)
  - [ ] Extend the state dict written in `_enter_trade()` (line 1281) to include: `"daily_pnl": self._daily_pnl`, `"daily_halted": self._daily_halted`, `"last_trading_date": self._last_trading_date.isoformat() if self._last_trading_date else None`
  - [ ] After `_close_active_trade()` runs and `StatePersistence.clear_state()` is called, replace that clear with a save of risk-only state (no active_trade keys) so daily P&L survives a restart after a closed trade. Pattern: `StatePersistence.save_state({"daily_pnl": ..., "daily_halted": ..., "last_trading_date": ...})`
  - [ ] Note: `StatePersistence` class itself needs NO changes — only the call sites change

- [ ] Task 3: Implement crash recovery in `initialize()` (AC: #4, #5, #6)
  - [ ] After `self._ts_client = TradeStationClient(...)` is created in `initialize()`, call `StatePersistence.load_state()`
  - [ ] If state is None → no state, continue normally
  - [ ] If state has `daily_pnl` / `daily_halted` / `last_trading_date` fields AND `last_trading_date` matches today's date → restore `self._daily_pnl`, `self._daily_halted`, `self._last_trading_date`
  - [ ] If state has active-trade keys (`direction`, `entry_price`, `tp_price`, `sl_price`, `sim_entry_order_id`) → call `await self._ts_client.reconcile_state(SIM_ACCOUNT_ID)`
  - [ ] If reconcile returns ACTIVE → reconstruct `self.active_trade = ActiveTrade(...)` from state fields; log `"✅ Crash recovery: resumed active trade from persisted state"`
  - [ ] If reconcile returns FLAT or PENDING-only → log `"⚠️ RECONCILIATION_WARNING: state shows active trade but broker has no position"`, call `StatePersistence.clear_state()`
  - [ ] If `load_state()` returns state with no active-trade keys (risk-only state from post-close save) → restore risk state only, no reconciliation needed
  - [ ] All reconciliation calls must be `await` — `initialize()` is already `async`

- [ ] Task 4: Wire `TradeLogger` into `Tier2StreamingTrader` and update trade close path (AC: #1, #2)
  - [ ] Add `self._trade_logger: TradeLogger = TradeLogger()` to `Tier2StreamingTrader.__init__()` (after `self._state_persistence`)
  - [ ] Update `_close_active_trade()` to build a `TradeRecord` from the closed trade fields and call `self._trade_logger.append_trade(record)` instead of `self._log_trade(...)`
  - [ ] The fields needed for `TradeRecord` that `_close_active_trade()` must supply:
    - `timestamp_entry`: `t.entry_time`
    - `timestamp_exit`: `bar.timestamp`
    - `direction`: `t.direction`
    - `entry_price`: `t.entry_price`
    - `exit_price`: price (the closing price)
    - `tp_price`: `t.tp_price`
    - `sl_price`: `t.sl_price`
    - `gap_size`: need to add `gap_size` field to `ActiveTrade` dataclass (set in `_enter_trade()`)
    - `pnl_usd`: pnl
    - `exit_reason`: reason (already a string)
    - `h1_sweep_bars_ago`: add `h1_sweep_bars_ago` to `ActiveTrade` (set in `_enter_trade()` from `self._cached_sweep.bars_ago` if available)
    - `m15_confirmed`: add `m15_confirmed` to `ActiveTrade` (= `self._m15_choch_active` at entry time)
    - `kill_zone_active`: add `kill_zone_active` to `ActiveTrade` (= `kill_zone_filter(bar.timestamp)` at entry time)
    - `vol_regime_pct`: add `vol_regime_pct` to `ActiveTrade` (= `self._last_vol_regime_pct` or 0.0)
    - `contracts`: `self._contracts`

- [ ] Task 5: Write unit tests in `tests/unit/test_trade_logging_state_persistence.py` (AC: #8)
  - [ ] `TestTradeLogger`:
    - [ ] `test_append_trade_writes_correct_columns` — verify CSV row has all PRD fields in order
    - [ ] `test_append_trade_header_written_once` — call twice, verify header appears exactly once
    - [ ] `test_append_trade_no_raise_on_write_error` — mock `open()` to raise, verify no exception propagates
  - [ ] `TestStatePersistence`:
    - [ ] `test_save_load_roundtrip` — save dict with risk + trade fields, load it back, assert equal
    - [ ] `test_clear_state_removes_file` — save then clear, verify load_state() returns None
  - [ ] `TestCrashRecovery`:
    - [ ] `test_crash_recovery_active_broker_position_restores_active_trade` — mock load_state with trade fields + reconcile_state returns ACTIVE → `trader.active_trade is not None`
    - [ ] `test_crash_recovery_flat_broker_logs_warning_and_clears` — mock load_state with trade fields + reconcile_state returns FLAT → RECONCILIATION_WARNING logged, clear_state called
    - [ ] `test_crash_recovery_no_state_skips_reconciliation` — mock load_state returns None → reconcile_state NOT called
    - [ ] `test_circuit_breaker_survives_restart_same_day` — mock load_state with `daily_halted=True`, same today date → `_check_daily_reset_and_halt()` returns True
    - [ ] `test_circuit_breaker_resets_on_new_day` — mock load_state with `daily_halted=True`, yesterday's date → halted flag resets to False

- [ ] Task 6: Run tests and verify no regressions (AC: #8)
  - [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_trade_logging_state_persistence.py -v`
  - [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_orchestrator_order_management.py -v` — 16 tests green
  - [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_strategy_core_detection.py tests/unit/test_config_loader.py -q` — no regressions
  - [ ] `PYTHONPATH=. .venv/bin/python -c "import src.research.tier2_streaming_working"` — import OK

## Dev Notes

### AR3 — Single-File Mandate

Both `TradeRecord`, `TradeLogger`, and all changes to `StatePersistence` call sites go inside `src/research/tier2_streaming_working.py`. Do NOT create `src/research/trade_logger.py` or any new file. The architecture explicitly preserves this as a future escape hatch only.

### Critical File Location Map (Current State)

```
tier2_streaming_working.py

Line 173: class StatePersistence
Line 185:   def save_state(cls, state: dict)         ← atomic write, DO NOT CHANGE
Line 191:   def load_state(cls) -> dict | None        ← NEVER called in initialize() — BUG
Line 198:   def clear_state(cls)

Line 580: @dataclass class ActiveTrade               ← ADD gap_size, h1_sweep_bars_ago,
Line 580:                                              m15_confirmed, kill_zone_active,
Line 580:                                              vol_regime_pct fields

Line 673: self._daily_pnl = 0.0                      ← NOT persisted → NFR12 violation
Line 674: self._daily_halted = False                  ← NOT persisted
Line 675: self._last_trading_date = None              ← NOT persisted

Line 685: self._state_persistence = StatePersistence() ← unused; class methods used directly
Line 691: async def initialize()                      ← ADD load_state + reconcile after ts_client

Line 1018: self._daily_pnl += pnl
Line 1024: StatePersistence.clear_state()             ← CHANGE to risk-state save instead
Line 1026: self._log_trade(...)                       ← REPLACE with TradeLogger.append_trade()

Line 1028: def _log_trade(...)                        ← REMOVE (replaced by TradeLogger)

Line 1063: def _check_daily_reset_and_halt(...)       ← Must respect loaded daily_halted on restart

Line 1281: StatePersistence.save_state({...})         ← EXTEND to include daily_pnl etc.
```

### PRD-Specified Trade Log Column Order (AR15)

```python
_COLUMNS = [
    "timestamp_entry",    # ISO-8601 with TZ offset — entry datetime
    "timestamp_exit",     # ISO-8601 with TZ offset — exit datetime
    "direction",          # "BEARISH" or "BULLISH" (ExitReason.value)
    "entry_price",        # float — index points
    "exit_price",         # float — index points
    "tp_price",           # float — index points (from ActiveTrade)
    "sl_price",           # float — index points (from ActiveTrade)
    "gap_size",           # float — index points (FVG gap size)
    "pnl_usd",            # float — USD P&L after commission
    "exit_reason",        # "TP" / "SL" / "TIME_STOP" / "MANUAL"
    "h1_sweep_bars_ago",  # int — how many H1 bars ago the sweep was
    "m15_confirmed",      # bool — was M15 CHoCH active at entry?
    "kill_zone_active",   # bool — was bar in kill zone window?
    "vol_regime_pct",     # float — ATR percentile at entry (0.0–1.0)
    "contracts",          # int — number of contracts
]
```

This column list is the single truth for all CSV writes. The `TradeLogger` class owns it.

### TradeRecord and TradeLogger Pattern

```python
# Place after TradeState dataclass, before TradeStationClient

@dataclass
class TradeRecord:
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


class TradeLogger:
    """Sole appender of the trade log CSV in PRD-mandated column order (AR15, FR29).

    Single-writer pattern: only this class appends to tier2_trade_log.csv.
    Header is written only when the file is empty (f.tell() == 0 check).
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
                if f.tell() == 0:      # only write header for empty file
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
```

### ActiveTrade Extension

`ActiveTrade` (line 580) needs 4 new optional fields to carry entry-context data through to the close:

```python
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
    pending_entry: bool = True
    # New in Story 4-2 — trade-log metadata
    gap_size: float = 0.0
    h1_sweep_bars_ago: int = 0
    m15_confirmed: bool = False
    kill_zone_active: bool = False
    vol_regime_pct: float = 0.0
```

These are set in `_enter_trade()` at the same time `ActiveTrade` is constructed:
- `gap_size`: `fvg_signal.gap_size` (FVGSignal has this; currently stored in snapped_dec — look at `make_entry_decision` return in strategy_core for the gap size)
- `h1_sweep_bars_ago`: `self._cached_sweep.bars_ago if self._cached_sweep else 0`
- `m15_confirmed`: `self._m15_choch_active`
- `kill_zone_active`: call `kill_zone_filter(bar.timestamp, self._strategy_config)` from strategy_core — already imported
- `vol_regime_pct`: add `self._last_vol_regime_pct: float = 0.0` to `__init__` and update it in `_update_h1_structure()` where the vol regime is computed

**Getting gap_size:** Look at `_enter_trade()` (around line 1260+). The `snapped_dec: EntryDecision` has `entry_price` / `tp_price` / `sl_price`. The gap size = `(sl_price - entry_price) / config.sl_multiplier` (since `sl = entry ± sl_multiplier * gap`). Alternatively, the `fvg_signal.gap_size` is available if `fvg_signal` is passed through. Check `_enter_trade()` signature to see what's available.

**Getting vol_regime_pct:** In `_update_h1_structure()`, the percentile is computed as part of the volatility regime check. Track it as `self._last_vol_regime_pct: float = 0.0` and update it whenever the vol regime is recalculated.

### Crash Recovery Pattern for initialize()

```python
async def initialize(self):
    # ... existing logging ...
    self.auth = TradeStationAuthV3.from_file('.access_token')
    await self.auth.authenticate()
    await self.auth.start_auto_refresh()
    self.client = httpx.AsyncClient(timeout=30.0)
    self._ts_client = TradeStationClient(self.auth, self._account_config, self.client)
    self.session_start_time = datetime.now()

    # Crash recovery: load persisted state and reconcile with broker (FR38, NFR11, NFR12)
    await self._recover_from_state()

async def _recover_from_state(self) -> None:
    """Load persisted state and reconcile with broker. Called once in initialize()."""
    state = StatePersistence.load_state()
    if state is None:
        return

    # Restore daily risk state if from the same calendar day
    today = datetime.now(timezone.utc).astimezone(ET_TZ).date()
    saved_date_str = state.get("last_trading_date")
    if saved_date_str:
        try:
            saved_date = datetime.fromisoformat(saved_date_str).date() if isinstance(saved_date_str, str) else saved_date_str
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

    # Reconstruct active trade if state has trade fields
    if state.get("direction") and state.get("entry_price"):
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
                pending_entry=False,  # position is open, not pending
            )
            logger.info("✅ Crash recovery: resumed active trade from persisted state")
        else:
            logger.warning(
                "⚠️ RECONCILIATION_WARNING: state shows active trade but broker has no position"
            )
            StatePersistence.clear_state()
```

### Extended save_state() Schema

The state dict saved in `_enter_trade()` should include risk fields:

```python
StatePersistence.save_state({
    # Active trade fields
    "direction": direction_str,
    "entry_price": ent,
    "tp_price": tp,
    "sl_price": sl,
    "entry_time": bar.timestamp.isoformat(),
    "sim_entry_order_id": e_id,
    "sim_tp_order_id": tp_id,
    "sim_sl_order_id": sl_id,
    # Daily risk state (NFR12)
    "daily_pnl": self._daily_pnl,
    "daily_halted": self._daily_halted,
    "last_trading_date": self._last_trading_date.isoformat() if self._last_trading_date else None,
})
```

After trade close (`_close_active_trade()`), instead of `StatePersistence.clear_state()`, save risk-only state:

```python
StatePersistence.save_state({
    "daily_pnl": self._daily_pnl,
    "daily_halted": self._daily_halted,
    "last_trading_date": self._last_trading_date.isoformat() if self._last_trading_date else None,
})
```

### `_check_daily_reset_and_halt()` — already correct for new day

The existing `_check_daily_reset_and_halt()` already handles the new-day reset:
- When `today != self._last_trading_date`, it resets `_daily_pnl` and `_daily_halted` to zero/False
- After recovery restores `_daily_halted=True` and `_last_trading_date=today`, the same-day check correctly keeps the halted state
- When a new calendar day arrives, `_last_trading_date != today` resets it — no change needed here

### Unit Test Pattern

```python
# tests/unit/test_trade_logging_state_persistence.py
import csv
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.research.tier2_streaming_working import (
    TradeRecord, TradeLogger, StatePersistence,
    ActiveTrade, Tier2StreamingTrader, SIM_ACCOUNT_ID,
)

class TestTradeLogger:
    def test_append_trade_writes_correct_columns(self, tmp_path):
        logger = TradeLogger()
        logger._LOG_PATH = tmp_path / "trade_log.csv"
        record = TradeRecord(
            timestamp_entry=datetime(2026, 1, 6, 10, 0, tzinfo=timezone.utc),
            timestamp_exit=datetime(2026, 1, 6, 11, 0, tzinfo=timezone.utc),
            direction="BEARISH",
            entry_price=18000.0,
            exit_price=17940.0,
            tp_price=17940.0,
            sl_price=18250.0,
            gap_size=50.0,
            pnl_usd=120.0,
            exit_reason="TP",
            h1_sweep_bars_ago=3,
            m15_confirmed=True,
            kill_zone_active=True,
            vol_regime_pct=0.62,
            contracts=5,
        )
        logger.append_trade(record)
        rows = list(csv.DictReader(logger._LOG_PATH.open()))
        assert len(rows) == 1
        assert rows[0]["direction"] == "BEARISH"
        assert rows[0]["exit_reason"] == "TP"
        assert rows[0]["m15_confirmed"] == "True"
        # Verify column order
        with logger._LOG_PATH.open() as f:
            header = f.readline().strip().split(",")
        assert header == TradeLogger._COLUMNS

    def test_header_written_once(self, tmp_path):
        logger = TradeLogger()
        logger._LOG_PATH = tmp_path / "trade_log.csv"
        record = ... # minimal valid record
        logger.append_trade(record)
        logger.append_trade(record)
        with logger._LOG_PATH.open() as f:
            content = f.read()
        # Header line appears exactly once
        assert content.count("timestamp_entry") == 1
```

### Existing Tests NOT to Break

- `tests/unit/test_orchestrator_order_management.py` — 16 tests for TradeStationClient; no changes needed
- `tests/unit/test_strategy_core_detection.py` — pure strategy_core tests; no changes
- `tests/unit/test_config_loader.py` — config loader; no changes

### Key `_enter_trade()` Call Site (for gap_size derivation)

```python
# Around line 1260 in _enter_trade():
snapped_dec = EntryDecision(...)  # already has entry_price, tp_price, sl_price
# gap_size = gap between SL and entry / sl_multiplier:
gap_size_pts = (snapped_dec.sl_price - snapped_dec.entry_price) / cfg.sl_multiplier  # bearish
# Or: from the fvg_signal directly if passed through
```

The `fvg_signal: FVGSignal` is in scope at `_enter_trade()` call time. Check if `fvg_signal.gap_size` is an attribute — if so, pass it. Otherwise, derive from `abs(snapped_dec.sl_price - snapped_dec.entry_price) / cfg.sl_multiplier`. Verify by reading `_enter_trade()` and `FVGSignal` definition in `strategy_core.py`.

### `_last_vol_regime_pct` Tracking

In `_update_h1_structure()`, the vol regime is computed. Add:
```python
self._last_vol_regime_pct: float = 0.0  # in __init__
# In _update_h1_structure() wherever the percentile is computed, also set:
self._last_vol_regime_pct = computed_percentile
```

Look for where `self._vol_regime_high` is set to find the percentile computation.

### Git Commits to Reference

```
c2cb21f feat: Story 4-1 — TradeStationClient bracket orders + reconciliation
233b097 feat: Epic 8 complete — S25 deployment, YAML config, multi-instrument
```

Key patterns from 4-1:
- `@dataclass` (not frozen) for runtime state objects
- `Optional[str]` return types, never raise from I/O methods
- `logger.warning("... %s", e)` format for recoverable errors (no f-string in log args)
- `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_XXX.py -v` for test runs
- `AsyncMock(spec=httpx.AsyncClient)` for HTTP mocking in tests

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

### File List

- `src/research/tier2_streaming_working.py` (modified)
- `tests/unit/test_trade_logging_state_persistence.py` (new)

### Change Log
