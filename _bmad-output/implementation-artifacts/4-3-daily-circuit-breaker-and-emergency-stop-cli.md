# Story 4.3: Daily Circuit Breaker and Emergency Stop CLI

Status: ready-for-dev

## Story

As Alex,
I want an automatic daily circuit breaker that halts entries at the configured loss threshold and an emergency stop CLI that can immediately cancel all orders,
So that runaway loss days are bounded automatically and I can manually override the system in under 30 seconds.

## Background

**Current state (after Story 4-2):**

The daily circuit breaker logic already exists in `Tier2StreamingTrader` as three in-memory fields (`_daily_pnl`, `_daily_halted`, `_last_trading_date`) and a method `_check_daily_reset_and_halt()`. The logic is correct for in-session use, but has two gaps:

1. **Persistence gap**: `_check_daily_reset_and_halt()` sets `self._daily_halted = True` but does NOT immediately call `save_state()`. State is only saved on trade close. If the process crashes between the circuit-breaker trip and the next trade close, the restart sees `daily_halted=False` in the file (the daily_pnl is correct, so it re-trips on the first bar — but this is fragile).

2. **Architecture gap**: The architecture mandates a `RiskManager` class with a stable `check_entry_allowed / register_fill` interface. Currently the risk logic is scattered as bare fields on `Tier2StreamingTrader`. This story extracts it into a `RiskManager` class (Phase 1 surface only).

3. **Emergency stop CLI gap**: `src/cli/emergency_stop.py` exists but uses the OLD infrastructure (`src.risk.emergency_stop.EmergencyStop`) that has no connection to the deployed Tier2 system. It must be REPLACED with an implementation that uses `TradeStationClient`, `StatePersistence`, and `TradeLogger` from `tier2_streaming_working.py`.

**This story:**
1. Introduces `RiskManager` class inside `tier2_streaming_working.py` (AR3 — single-file mandate; `RiskManager` lives in the same file as `Tier2StreamingTrader`).
2. Migrates `Tier2StreamingTrader` to delegate risk tracking to `self._risk_manager`.
3. Fixes the persistence gap: `RiskManager.check_and_update()` calls `_persist()` immediately when it trips.
4. Rewrites `src/cli/emergency_stop.py` to integrate with the Tier2 system.
5. Writes unit tests covering all circuit-breaker and emergency-stop behaviors.

## Acceptance Criteria

**AC#1 — Circuit breaker halts entries when daily loss limit reached:**
Given `RiskManager` tracking `_daily_pnl` across all closed trades in the current calendar day (ET timezone),
When `_daily_pnl` reaches or drops below `StrategyConfig.max_daily_loss` (e.g., -$750),
Then `RiskManager.is_halted` returns `True` and `_detect_and_enter()` skips all entry checks for the remainder of the day (FR16).

**AC#2 — Circuit breaker resets at calendar day change:**
Given `is_halted` returns `True` at 10:45 ET on day D (because daily_pnl hit threshold),
When the polling loop runs at 09:31 ET on day D+1 (new calendar date),
Then `RiskManager.check_and_update()` resets `_daily_pnl = 0.0` and `_daily_halted = False` — circuit breaker is cleared for the new day (FR16).

**AC#3 — Halted state survives crash and restart same day:**
Given the circuit breaker trips (`_daily_halted = True`) and the process then crashes,
When it restarts within the same calendar day and `_recover_from_state()` calls `risk_manager.restore_from_state()`,
Then `is_halted` returns `True` — state was persisted immediately when the circuit breaker tripped (NFR12).

**AC#4 — Emergency stop CLI cancels orders and closes position:**
Given `python src/cli/emergency_stop.py` is executed (or `python -m src.cli.emergency_stop`),
When it runs successfully,
Then it: (1) calls `TradeStationClient.cancel_all_pending_orders(SIM_ACCOUNT_ID)`, (2) if active position in persisted state calls `TradeStationClient.close_position_at_market(direction, SIM_ACCOUNT_ID)`, (3) calls `RiskManager.halt_manually()` which persists `daily_halted=True` to state file, (4) appends a `TradeRecord` with `exit_reason="MANUAL"` to `TradeLogger` if a position was closed, (5) prints `"EMERGENCY STOP COMPLETE"` to stdout (FR22).

**AC#5 — Emergency stop fail-safe with --force:**
Given `python src/cli/emergency_stop.py --force` is executed when TradeStation API is unreachable (raises `Exception`),
When API calls fail,
Then the script still persists `daily_halted=True` to the state file, logs `"WARNING: Could not confirm order cancellation — API unreachable"` to stderr, and exits with code 1 (fail-safe: local halt succeeds even when API is down).

**AC#6 — RiskManager class introduced with Phase 1 interface:**
Given `tier2_streaming_working.py`,
When it is imported,
Then a `RiskManager` class exists with:
- `is_halted: bool` property
- `daily_pnl: float` property
- `check_and_update(bar_et: datetime, max_daily_loss: float) -> bool`
- `register_close(pnl: float) -> None`
- `halt_manually() -> None`
- `restore_from_state(state: dict, today: date) -> None`
- `to_state_dict() -> dict`

**AC#7 — Tier2StreamingTrader delegates to RiskManager:**
Given `Tier2StreamingTrader.__init__()`,
When instantiated,
Then `self._risk_manager: RiskManager = RiskManager()` is present and the separate `_daily_pnl`, `_daily_halted`, `_last_trading_date` bare fields are REMOVED — all access goes through `_risk_manager`.

**AC#8 — Unit tests:**
Given `tests/unit/test_circuit_breaker_emergency_stop.py`,
When pytest runs it,
Then all tests pass covering:
- `RiskManager` circuit breaker trip and halt
- `RiskManager` calendar-day reset
- `RiskManager.restore_from_state()` same-day vs. new-day
- `RiskManager._persist()` called when circuit breaker trips
- Emergency stop CLI: cancels orders + closes position + halts
- Emergency stop CLI: `--force` path with API failure → still persists halt, exits 1

## Tasks / Subtasks

- [ ] Task 1: Introduce `RiskManager` class in `tier2_streaming_working.py` (AC: #1, #2, #3, #6)
  - [ ] Add `RiskManager` class AFTER `TradeLogger`, BEFORE `TradeStationClient` (same file — AR3)
  - [ ] Fields: `_daily_pnl: float = 0.0`, `_daily_halted: bool = False`, `_last_trading_date: Optional[date] = None`
  - [ ] `is_halted` property: returns `self._daily_halted`
  - [ ] `daily_pnl` property: returns `self._daily_pnl`
  - [ ] `register_close(pnl: float) -> None`: adds pnl to `_daily_pnl` (does NOT call `_persist()` — caller saves state)
  - [ ] `check_and_update(bar_et: datetime, max_daily_loss: float) -> bool`: day-reset logic (copy from `_check_daily_reset_and_halt`); when tripping (`daily_pnl <= max_daily_loss`), set `_daily_halted = True` and call `self._persist()` immediately before returning True
  - [ ] `halt_manually() -> None`: sets `_daily_halted = True`, calls `self._persist()`
  - [ ] `restore_from_state(state: dict, today: date) -> None`: parses `last_trading_date` from state dict; if same calendar day as `today`, restores `_daily_pnl`, `_daily_halted`, `_last_trading_date`
  - [ ] `to_state_dict() -> dict`: returns `{"daily_pnl": ..., "daily_halted": ..., "last_trading_date": ...}`
  - [ ] `_persist() -> None`: calls `StatePersistence.save_state(self.to_state_dict())` — wraps in try/except, logs warning on failure

- [ ] Task 2: Migrate `Tier2StreamingTrader` to use `_risk_manager` (AC: #7)
  - [ ] Add `self._risk_manager: RiskManager = RiskManager()` to `__init__()` (after `self._trade_logger`)
  - [ ] REMOVE `self._daily_pnl`, `self._daily_halted`, `self._last_trading_date` bare fields
  - [ ] `_close_active_trade()`: replace `self._daily_pnl += pnl` with `self._risk_manager.register_close(pnl)`; update `save_state()` call to use `**self._risk_manager.to_state_dict()` merged into the state dict; remove the bare field references
  - [ ] `_detect_and_enter()`: replace `self._check_daily_reset_and_halt(bar_et)` with `self._risk_manager.check_and_update(bar_et, self._strategy_config.max_daily_loss)`
  - [ ] `_recover_from_state()`: replace the inline risk-state restoration block with `self._risk_manager.restore_from_state(state, today)`
  - [ ] `_check_daily_reset_and_halt()`: DELETE this method (logic now lives in `RiskManager.check_and_update()`)
  - [ ] Any remaining `self._daily_pnl` / `self._daily_halted` / `self._last_trading_date` references: update to `self._risk_manager.daily_pnl` / `self._risk_manager.is_halted` / `self._risk_manager._last_trading_date` as appropriate
  - [ ] Verify: existing `save_state()` calls in `_enter_trade()` and `_close_active_trade()` still include risk state (use `**self._risk_manager.to_state_dict()` merge pattern)

- [ ] Task 3: Rewrite `src/cli/emergency_stop.py` (AC: #4, #5)
  - [ ] Replace entire file — the existing implementation uses old infrastructure (`src.risk.emergency_stop.EmergencyStop`)
  - [ ] New implementation structure:
    ```python
    import argparse, asyncio, sys
    from pathlib import Path
    from datetime import datetime, timezone
    import httpx
    from src.research.tier2_streaming_working import (
        TradeStationAuthV3 is imported via auth_v3, TradeStationClient,
        StatePersistence, TradeLogger, TradeRecord, RiskManager,
        SIM_ACCOUNT_ID, _default_account_config, ET_TZ,
    )
    ```
  - [ ] `async def _run_emergency_stop(force: bool) -> int`: the async implementation
    - Load credentials: `auth = TradeStationAuthV3.from_file('.access_token')`
    - Create `async with httpx.AsyncClient(timeout=30.0) as http`
    - Authenticate: `await auth.authenticate()`
    - Create `TradeStationClient(auth, _default_account_config("MNQM26"), http)`
    - Try: `await client.cancel_all_pending_orders(SIM_ACCOUNT_ID)` — if fails and not `--force`, propagate; if fails and `--force`, print warning, set `api_ok=False`
    - Load state: `state = StatePersistence.load_state()`
    - If state has `direction` and `entry_price is not None` and `entry_time`: it's an active trade
      - Try: `close_oid = await client.close_position_at_market(state["direction"], SIM_ACCOUNT_ID)` (skip if api_ok=False)
      - Append MANUAL exit to TradeLogger: create `TradeRecord(timestamp_entry=..., timestamp_exit=now, exit_reason="MANUAL", pnl_usd=0.0, ...)` and call `TradeLogger().append_trade(record)`
    - Call `RiskManager().halt_manually()` — this writes `daily_halted=True` to state file
    - Print `"EMERGENCY STOP COMPLETE"`
    - Return 0 if api_ok else 1
  - [ ] `def main()`: arg-parse `--force` flag; call `sys.exit(asyncio.run(_run_emergency_stop(args.force)))`
  - [ ] Handle `--force`: catch all API exceptions, log to stderr, still persist halt, return exit code 1
  - [ ] `if __name__ == "__main__": main()`

- [ ] Task 4: Write unit tests in `tests/unit/test_circuit_breaker_emergency_stop.py` (AC: #8)
  - [ ] `TestRiskManager`:
    - [ ] `test_circuit_breaker_trips_when_pnl_below_threshold` — register_close brings pnl below limit, check_and_update returns True
    - [ ] `test_circuit_breaker_does_not_trip_above_threshold` — pnl still above limit, check_and_update returns False
    - [ ] `test_day_reset_clears_halted_flag` — halted on day D, check_and_update with day D+1 → is_halted=False, pnl=0.0
    - [ ] `test_restore_from_state_same_day_restores_risk` — state has today's date, pnl and halted restored
    - [ ] `test_restore_from_state_new_day_does_not_restore` — state has yesterday → pnl=0.0, halted=False
    - [ ] `test_persist_called_when_circuit_breaker_trips` — patch StatePersistence.save_state, verify called when check_and_update trips
    - [ ] `test_halt_manually_persists` — patch StatePersistence.save_state, verify called by halt_manually()
    - [ ] `test_to_state_dict_includes_all_keys` — verify dict has daily_pnl, daily_halted, last_trading_date keys
  - [ ] `TestEmergencyStopCLI`:
    - [ ] `test_emergency_stop_cancels_orders_and_halts` — mock client, cancel_all_pending_orders called, state updated, prints COMPLETE
    - [ ] `test_emergency_stop_closes_position_when_active_trade_in_state` — state has direction/entry_price/entry_time, close_position_at_market called
    - [ ] `test_emergency_stop_appends_manual_exit_to_trade_logger` — TradeLogger.append_trade called with MANUAL exit when position closed
    - [ ] `test_emergency_stop_force_flag_persists_halt_on_api_failure` — API raises exception, --force → halt persisted, exit code 1
    - [ ] `test_emergency_stop_no_active_trade_skips_close` — state is risk-only (no direction), close_position_at_market NOT called

- [ ] Task 5: Run tests and verify no regressions (AC: #8)
  - [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_circuit_breaker_emergency_stop.py -v` → all pass
  - [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_trade_logging_state_persistence.py -v` → 16/16 pass (no regressions)
  - [ ] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_orchestrator_order_management.py -v` → 16/16 pass
  - [ ] `PYTHONPATH=. .venv/bin/python -c "import src.research.tier2_streaming_working"` → import OK
  - [ ] `PYTHONPATH=. .venv/bin/python -c "import src.cli.emergency_stop"` → import OK

## Dev Notes

### AR3 — Single-File Mandate

`RiskManager` lives in `src/research/tier2_streaming_working.py` — NOT in a separate file. The architecture mandates that the live orchestrator is a single file. Do NOT create `src/risk/risk_manager.py` or any new file under `src/`.

The `src/cli/emergency_stop.py` IS a separate file (CLI entry point is always separate), but it imports its dependencies from `tier2_streaming_working.py`.

### RiskManager Class Placement

Place `RiskManager` **after `TradeLogger`** and **before `TradeStationClient`** in the file. The class dependency order in the file:
1. `AccountConfig` (dataclass)
2. `TradeState` (dataclass)
3. `TradeRecord` (dataclass)
4. `StatePersistence` (class) — RiskManager depends on this
5. `TradeLogger` (class)
6. **`RiskManager` (class)** ← NEW HERE
7. `TradeStationClient` (class)
8. `Tier2StreamingTrader` (class) — uses all of the above

### RiskManager Implementation Pattern

```python
class RiskManager:
    """Phase 1 daily circuit breaker. Tracks daily P&L and halts entries when threshold hit.

    Architecture: see architecture.md Risk Layer — Phase 1 interface only.
    Phase 2 extension (trailing DD, consistency, dynamic contracts) adds evaluators inside
    check_and_update() without changing the public surface.
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
        """Update daily P&L after a trade closes. Caller is responsible for saving state."""
        self._daily_pnl += pnl

    def check_and_update(self, bar_et: datetime, max_daily_loss: float) -> bool:
        """Reset on new calendar day, then check circuit breaker. Returns True if halted."""
        today = bar_et.date()
        if self._last_trading_date != today:
            if self._last_trading_date is not None:
                logger.info("New trading day %s — resetting daily P&L (was $%.2f)", today, self._daily_pnl)
            self._daily_pnl = 0.0
            self._daily_halted = False
            self._last_trading_date = today
        if self._daily_halted:
            return True
        if self._daily_pnl <= max_daily_loss:
            logger.warning(
                "🛑 Daily loss limit hit: $%.2f ≤ $%.0f — halting for today",
                self._daily_pnl, max_daily_loss
            )
            self._daily_halted = True
            self._persist()  # immediately persist — crash between trips and next trade close won't lose halt
            return True
        return False

    def halt_manually(self) -> None:
        """Externally halt for today (called by emergency stop CLI). Persists immediately."""
        self._daily_halted = True
        self._persist()

    def restore_from_state(self, state: dict, today: datetime.date) -> None:
        """Restore daily risk state from persisted dict on startup (crash recovery)."""
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
        """Persist current risk state immediately (used when circuit breaker trips or halt_manually)."""
        try:
            StatePersistence.save_state(self.to_state_dict())
        except Exception as e:
            logger.warning("RiskManager: failed to persist halt state: %s", e)
```

### Migrating Tier2StreamingTrader

**Fields to REMOVE** from `__init__()` (lines ~747-750):
```python
# DELETE these:
self._daily_pnl: float = 0.0
self._daily_halted: bool = False
self._last_trading_date: Optional[datetime.date] = None
```

**Field to ADD** in `__init__()` (after `self._trade_logger`):
```python
self._risk_manager: RiskManager = RiskManager()
```

**`_close_active_trade()` — update pnl and state:**
```python
# BEFORE (lines ~1183, ~1189-1195):
self._daily_pnl += pnl
...
StatePersistence.save_state({
    "daily_pnl": self._daily_pnl,
    "daily_halted": self._daily_halted,
    "last_trading_date": self._last_trading_date.isoformat() if self._last_trading_date else None,
})

# AFTER:
self._risk_manager.register_close(pnl)
...
try:
    StatePersistence.save_state(self._risk_manager.to_state_dict())
except Exception as e:
    logger.warning("State persistence failed after trade close: %s", e)
```

**`_detect_and_enter()` — circuit breaker check:**
```python
# BEFORE:
if self._check_daily_reset_and_halt(bar_et):
    return

# AFTER:
if self._risk_manager.check_and_update(bar_et, self._strategy_config.max_daily_loss):
    return
```

**`_recover_from_state()` — risk state restoration:**
```python
# BEFORE (inline block restoring _daily_pnl, _daily_halted, _last_trading_date):
if saved_date == today:
    self._daily_pnl = float(state.get("daily_pnl", 0.0))
    self._daily_halted = bool(state.get("daily_halted", False))
    self._last_trading_date = today
    logger.info(...)

# AFTER (single delegation):
self._risk_manager.restore_from_state(state, today)
```

**DELETE `_check_daily_reset_and_halt()` method entirely** — its logic is now in `RiskManager.check_and_update()`.

**Update `_enter_trade()` save_state** to use merged risk dict:
```python
StatePersistence.save_state({
    "direction": direction_str,
    "entry_price": ent,
    ...all existing trade fields...,
    **self._risk_manager.to_state_dict(),  # merge risk state
})
```

**Other references** — grep for any remaining `self._daily_pnl`, `self._daily_halted`, `self._last_trading_date` and update:
- Log messages that print daily P&L: use `self._risk_manager.daily_pnl`
- `_close_active_trade()` log line: `self._risk_manager.daily_pnl`
- `_print_final_report()` if it references pnl: use `self._risk_manager.daily_pnl`

### Emergency Stop CLI Design

The CLI is a **separate process** from the live trader. When it runs:
- It reads credentials from `.access_token` (same as the live trader)
- It uses the SAME `StatePersistence` state file as the live trader
- The RUNNING trader will NOT immediately see the halt (it uses in-memory state) — the user should kill the trader process separately after running the emergency stop; on the next restart, `daily_halted=True` from the state file will prevent new entries

**Complete emergency_stop.py structure:**

```python
"""Emergency stop CLI for Tier2StreamingTrader.

Cancels all pending orders, closes any active position, and sets daily_halted=True
in the persisted state so the trader does not enter new trades on restart.

Usage:
    python src/cli/emergency_stop.py
    python src/cli/emergency_stop.py --force   # persist halt even if API unreachable
"""
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Ensure repo root is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.auth_v3 import TradeStationAuthV3
from src.research.tier2_streaming_working import (
    ET_TZ,
    RiskManager,
    SIM_ACCOUNT_ID,
    StatePersistence,
    TradeLogger,
    TradeRecord,
    TradeStationClient,
    _default_account_config,
)


async def _run_emergency_stop(force: bool) -> int:
    """Execute emergency stop. Returns exit code (0 = success, 1 = API failure with --force)."""
    api_ok = True

    auth = TradeStationAuthV3.from_file(".access_token")
    async with httpx.AsyncClient(timeout=30.0) as http:
        await auth.authenticate()
        cfg = _default_account_config("MNQM26")
        client = TradeStationClient(auth, cfg, http)

        # Step 1: Cancel all pending orders
        try:
            cancelled = await client.cancel_all_pending_orders(SIM_ACCOUNT_ID)
            print(f"Cancelled {len(cancelled)} pending orders: {cancelled}")
        except Exception as e:
            msg = f"WARNING: Could not confirm order cancellation — API unreachable: {e}"
            print(msg, file=sys.stderr)
            if not force:
                raise
            api_ok = False

        # Step 2: Close any active position recorded in persisted state
        state = StatePersistence.load_state()
        if (
            state
            and state.get("direction")
            and state.get("entry_price") is not None
            and state.get("entry_time")
        ):
            direction = state["direction"]
            entry_time_str = state["entry_time"]
            entry_price = float(state["entry_price"])

            if api_ok:
                try:
                    await client.close_position_at_market(direction, SIM_ACCOUNT_ID)
                    print(f"Closed active {direction} position at market.")
                except Exception as e:
                    print(f"WARNING: Could not close position — {e}", file=sys.stderr)
                    if not force:
                        raise
                    api_ok = False

            # Append MANUAL exit to trade log
            now = datetime.now(timezone.utc)
            try:
                TradeLogger().append_trade(TradeRecord(
                    timestamp_entry=datetime.fromisoformat(entry_time_str),
                    timestamp_exit=now,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=entry_price,  # unknown — use entry as placeholder
                    tp_price=float(state.get("tp_price", 0.0)),
                    sl_price=float(state.get("sl_price", 0.0)),
                    gap_size=float(state.get("gap_size", 0.0)),
                    pnl_usd=0.0,  # unknown — manual stop
                    exit_reason="MANUAL",
                    h1_sweep_bars_ago=int(state.get("h1_sweep_bars_ago", 0)),
                    m15_confirmed=bool(state.get("m15_confirmed", False)),
                    kill_zone_active=bool(state.get("kill_zone_active", False)),
                    vol_regime_pct=float(state.get("vol_regime_pct", 0.0)),
                    contracts=cfg.contracts,
                ))
            except Exception as e:
                print(f"WARNING: Could not write trade log — {e}", file=sys.stderr)

    # Step 3: Persist halt (runs regardless of API success/failure)
    RiskManager().halt_manually()

    print("EMERGENCY STOP COMPLETE")
    return 0 if api_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emergency stop: cancel all orders, close position, halt entries."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Persist local halt even if TradeStation API is unreachable (exits code 1).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run_emergency_stop(args.force)))


if __name__ == "__main__":
    main()
```

### Existing Tests That Must NOT Break

```bash
tests/unit/test_trade_logging_state_persistence.py   # 16 tests — Story 4-2
tests/unit/test_orchestrator_order_management.py     # 16 tests — Story 4-1
tests/unit/test_strategy_core_detection.py           # ~40 tests
```

The 4-2 crash-recovery tests patch `StatePersistence.load_state` and `StatePersistence.clear_state` and call `trader._recover_from_state()`. After this story's migration:
- `_recover_from_state()` calls `self._risk_manager.restore_from_state(state, today)` instead of inline restoration
- The test assertions on `trader._daily_pnl` and `trader._daily_halted` must change to `trader._risk_manager.daily_pnl` and `trader._risk_manager.is_halted`

**UPDATE the Story 4-2 tests** (`test_circuit_breaker_survives_restart_same_day`, `test_circuit_breaker_resets_on_new_day`, `test_crash_recovery_risk_only_state_no_reconciliation`) to use `trader._risk_manager.is_halted` and `trader._risk_manager.daily_pnl` instead of the now-deleted bare fields. Do this BEFORE running the 4-2 test suite to check for regressions.

### Unit Test Patterns (from Story 4-2)

```python
# RiskManager tests — no async, no mocking needed for basic behavior
class TestRiskManager:
    def test_circuit_breaker_trips_when_pnl_below_threshold(self):
        rm = RiskManager()
        rm.register_close(-800.0)
        bar_et = datetime(2026, 1, 6, 10, 0, tzinfo=ET_TZ)
        result = rm.check_and_update(bar_et, max_daily_loss=-750.0)
        assert result is True
        assert rm.is_halted is True

    def test_persist_called_when_circuit_breaker_trips(self):
        rm = RiskManager()
        rm.register_close(-800.0)
        bar_et = datetime(2026, 1, 6, 10, 0, tzinfo=ET_TZ)
        with patch.object(StatePersistence, "save_state") as mock_save:
            rm.check_and_update(bar_et, max_daily_loss=-750.0)
        mock_save.assert_called_once()
```

```python
# Emergency stop CLI tests — use AsyncMock + patch
class TestEmergencyStopCLI:
    @pytest.mark.asyncio
    async def test_emergency_stop_cancels_orders_and_halts(self):
        with patch("src.cli.emergency_stop.TradeStationAuthV3") as MockAuth, \
             patch("src.cli.emergency_stop.TradeStationClient") as MockClient, \
             patch("src.cli.emergency_stop.StatePersistence") as MockSP, \
             patch("src.cli.emergency_stop.RiskManager") as MockRM:
            MockSP.load_state.return_value = None  # no active trade
            mock_client_inst = AsyncMock()
            mock_client_inst.cancel_all_pending_orders = AsyncMock(return_value=["O1"])
            MockClient.return_value = mock_client_inst
            MockAuth.from_file.return_value.authenticate = AsyncMock()

            result = await _run_emergency_stop(force=False)

        assert result == 0
        mock_client_inst.cancel_all_pending_orders.assert_called_once()
        MockRM.return_value.halt_manually.assert_called_once()
```

**Import in test file:**
```python
from src.cli.emergency_stop import _run_emergency_stop
from src.research.tier2_streaming_working import (
    ET_TZ, RiskManager, StatePersistence,
)
```

### Key Code Location Map (Current State)

| Symbol | File:Line | Notes |
|---|---|---|
| `_daily_pnl` | tier2_streaming_working.py:748 | REMOVE — replace with `_risk_manager.daily_pnl` |
| `_daily_halted` | tier2_streaming_working.py:749 | REMOVE — replace with `_risk_manager.is_halted` |
| `_last_trading_date` | tier2_streaming_working.py:750 | REMOVE — replace with `_risk_manager._last_trading_date` |
| `_check_daily_reset_and_halt()` | tier2_streaming_working.py:1221 | DELETE — logic moves to `RiskManager.check_and_update()` |
| `self._daily_pnl += pnl` | tier2_streaming_working.py:1183 | → `self._risk_manager.register_close(pnl)` |
| Circuit breaker check in `_detect_and_enter()` | tier2_streaming_working.py:1252 | → `self._risk_manager.check_and_update(...)` |
| Risk state restoration in `_recover_from_state()` | tier2_streaming_working.py:806-813 | → `self._risk_manager.restore_from_state(state, today)` |
| `cancel_all_pending_orders()` | tier2_streaming_working.py:457 | Already exists — used by emergency stop CLI |
| `close_position_at_market()` | tier2_streaming_working.py:389 | Already exists — used by emergency stop CLI |
| `src/cli/emergency_stop.py` | Full rewrite | Old implementation uses `src.risk.emergency_stop.EmergencyStop` — DELETE everything |

### Log References (after migration)

All log lines that currently reference `self._daily_pnl` directly need updating:
```python
# In _close_active_trade() logger.info:
# BEFORE: f"Trade Closed: ... Daily P&L: ${self._daily_pnl:.2f}"
# AFTER:  f"Trade Closed: ... Daily P&L: ${self._risk_manager.daily_pnl:.2f}"
```

### Git Commits to Reference

```
0454e3d fix: Story 4-2 review — 5 crash-recovery patches, 4 new tests (16 total)
58fb25c feat: Story 4-2 — TradeLogger, crash recovery, circuit-breaker persistence
c2cb21f feat: Story 4-1 — TradeStationClient bracket orders + reconciliation
```

Key patterns from prior stories:
- `@dataclass` (not frozen) for runtime state objects
- `Optional[str]` returns, never raise from I/O methods
- `logger.warning("... %s", e)` format for recoverable errors (no f-string in log args for exceptions)
- `AsyncMock` from `unittest.mock` for async method mocking
- `patch.object(StatePersistence, "save_state", ...)` — tested extensively in 4-2
- `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_XXX.py -v` for test runs

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

### Change Log
