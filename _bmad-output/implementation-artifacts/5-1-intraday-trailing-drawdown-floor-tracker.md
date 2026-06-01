# Story 5.1: Intraday Trailing Drawdown Floor Tracker

Status: review

## Story

As Alex,
I want a trailing drawdown floor that updates in real-time with every unrealized equity peak,
So that I can manage a TopStep $50K Combine without accidentally breaching the intraday trailing drawdown limit.

## Background

TopStep Combine accounts use an *intraday trailing* drawdown: the floor rises with every unrealized equity peak, not just at trade close. A trade that runs +$800 unrealized then reverses to flat has permanently raised the DD floor by $800. This is meaningfully different from an end-of-day (EOD) trailing approach used in XFA funded accounts, where the floor only moves at session end.

This story extends the existing `RiskManager` Phase 2 evaluator slot (see architecture `planning-artifacts/architecture.md` lines 393–396) to implement the trailing DD tracker. The `AccountConfig` dataclass (already in `tier2_streaming_working.py:81`) is the configuration source.

Phase 1 circuit breaker (`RiskManager`) already exists at `src/research/tier2_streaming_working.py:280`. This story adds evaluators *inside the same two methods* (`check_and_update` / `register_close`) — no new public surface.

## Acceptance Criteria

**Given** a session begins with starting equity `E₀ = $50,000` and `config.topstep_trailing_dd_amount = $2,000`,
**When** the first bar is processed,
**Then** `trailing_floor = E₀ - topstep_trailing_dd_amount = $48,000`

**Given** an open trade running in profit and unrealized equity peaks at `E₀ + $800 = $50,800`,
**When** the bar is processed,
**Then** `trailing_floor` rises to `$50,800 - $2,000 = $48,800` — floor updates on every unrealized peak, not only at trade close (FR17)

**Given** current equity `= $48,700` and `trailing_floor = $48,800`,
**When** the bar is processed,
**Then** the system logs `TRAILING_DD_BREACH: current equity $48,700 is at or below floor $48,800` and triggers an emergency halt (trading stops immediately)

**Given** current equity is within `config.trailing_dd_alert_pct = 10%` of the floor,
**When** the bar is processed,
**Then** the system logs `TRAILING_DD_ALERT: equity at 90% of floor` and displays the warning (pre-emptive alert, FR17)

**Given** `config.dd_type = 'eod'` (XFA funded account mode),
**When** bars are processed intraday,
**Then** `trailing_floor` only updates at session end, not intrabar — XFA EOD trailing behavior implemented separately from Combine intraday behavior

**Given** `trailing_floor` state is persisted via `StatePersistence`,
**When** the process crashes and restarts,
**Then** `trailing_floor` is restored to its pre-crash value — floor tracking is not reset on restart

## Tasks / Subtasks

- [x] Task 1: Extend `AccountConfig` with trailing DD fields (AC: #1, #4, #5)
  - [x] Add `dd_type: Literal["intraday", "eod"] = "intraday"` field
  - [x] Add `topstep_trailing_dd_amount: float = 2000.0` field
  - [x] Add `trailing_dd_alert_pct: float = 0.10` field
  - [x] Add `starting_equity: float = 50000.0` field (basis for initial floor calculation)

- [x] Task 2: Add `TrailingDDTracker` inside `RiskManager` (AC: #1, #2, #3, #4, #5)
  - [x] Initialize `_trailing_floor: float` from `starting_equity - topstep_trailing_dd_amount` on first bar
  - [x] Implement intraday mode: update floor when `current_equity > floor + dd_amount` (i.e., new peak)
  - [x] Implement EOD mode: floor update only called from session-close hook, not per-bar
  - [x] Add `check_trailing_dd(current_equity: float, bar_et: datetime, account_config: AccountConfig) -> bool` method; returns True if breach
  - [x] Log `TRAILING_DD_ALERT` when equity within `trailing_dd_alert_pct` of floor
  - [x] Log `TRAILING_DD_BREACH` and set `_daily_halted = True` on breach

- [x] Task 3: Wire into bar processing loop (AC: #1–#4)
  - [x] In `Tier2StreamingTrader._poll_and_process()`, compute `current_equity` from `starting_equity + daily_pnl + unrealized_pnl`
  - [x] Call `risk_manager.check_trailing_dd(current_equity, bar_et, account_config)` on every bar
  - [x] If breach, halt trading same as circuit breaker (skip `_detect_and_enter`, skip `_advance_active_trade` new entries)

- [x] Task 4: Persist and restore `trailing_floor` (AC: #6)
  - [x] Extend `RiskManager.to_state_dict()` to include `trailing_floor`
  - [x] Extend `RiskManager.restore_from_state()` to restore `_trailing_floor` when same-day state found
  - [x] Reset `_trailing_floor` on new calendar day (same as `_daily_pnl` reset in `check_and_update`)

- [x] Task 5: Unit tests (all ACs)
  - [x] `test_trailing_floor_initializes_from_starting_equity` — first bar sets floor = E₀ − DD amount
  - [x] `test_trailing_floor_rises_on_unrealized_peak` — floor advances when equity peaks above prior high
  - [x] `test_trailing_floor_does_not_fall` — floor never decreases even if equity drops
  - [x] `test_breach_halts_trading` — equity at/below floor → `is_halted = True`
  - [x] `test_alert_logged_near_floor` — equity within `trailing_dd_alert_pct` → alert emitted
  - [x] `test_eod_mode_floor_not_updated_intrabar` — dd_type='eod' → floor stable intrabar
  - [x] `test_trailing_floor_survives_crash_recovery` — `to_state_dict` / `restore_from_state` round-trip

## Dev Notes

### Existing Patterns to Follow

- `RiskManager` lives at `src/research/tier2_streaming_working.py:280`. The Phase 1 circuit breaker is `check_and_update(bar_et, max_daily_loss) -> bool`. Pattern: same two public methods, extend internally.
- `StatePersistence.save_state()` / `load_state()` at line 204. JSON dict stored at `logs/active_trade_state.json`.
- `AccountConfig` dataclass at line 81 (simple frozen-style dataclass, no `__post_init__`). Add new fields with defaults — backward compat is automatic.

### Unrealized PnL Calculation

The live trader does not currently compute unrealized PnL per bar. To avoid polling the broker every bar, compute it from the `ActiveTrade` object:

```python
unrealized_pnl = 0.0
if self.active_trade is not None and self.active_trade.status == "ACTIVE":
    direction_sign = -1 if self.active_trade.direction == "SHORT" else 1
    unrealized_pnl = direction_sign * (bar.close - self.active_trade.entry_price) \
                     * self._point_value * self._contracts
current_equity = account_config.starting_equity + self._risk_manager.daily_pnl + unrealized_pnl
```

### Floor Update Logic (Intraday)

```python
# Only update floor upward, never downward
equity_peak = max(current_equity, self._equity_high_water)
self._equity_high_water = equity_peak
new_floor = equity_peak - account_config.topstep_trailing_dd_amount
if new_floor > self._trailing_floor:
    self._trailing_floor = new_floor
```

### Test File

Add to `tests/unit/test_risk_manager.py` (create if it doesn't exist, following `tests/unit/test_strategy_core_*.py` pattern).

### References

- FR17: `_bmad-output/planning-artifacts/prd.md:576`
- Epic 5 story spec: `_bmad-output/planning-artifacts/epics.md:932`
- Architecture risk layer decision: `_bmad-output/planning-artifacts/architecture.md:380`
- `AccountConfig`: `src/research/tier2_streaming_working.py:81`
- `RiskManager`: `src/research/tier2_streaming_working.py:280`
- `StatePersistence`: `src/research/tier2_streaming_working.py:204`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Fixed: `_last_trading_date` not updated in `check_trailing_dd` → `to_state_dict` had `None` date → `restore_from_state` returned early without restoring floor. Fixed by setting `self._last_trading_date = today` inside `check_trailing_dd`.
- Fixed: pre-existing test `test_filter_log_skip_when_vol_regime_high` expected `"SKIP"` but code logs `"SKIP:VOL_REGIME"` — corrected assertion.

### Completion Notes List

- Task 1: Added 4 fields with defaults to `AccountConfig` dataclass — fully backward-compatible.
- Task 2: `RiskManager` extended with `_trailing_floor: Optional[float]`, `_equity_high_water: Optional[float]`, `trailing_floor` property, and `check_trailing_dd()` method. Intraday and EOD modes both implemented. Alert threshold is `cushion < dd_amount × alert_pct`. Breach sets `_daily_halted = True` and persists immediately.
- Task 3: Per-bar unrealized PnL computed inline in `_poll_and_process` loop using `ActiveTrade.entry_price` and direction sign. `check_trailing_dd` called before `_advance_active_trade` so a breach halts before new entries.
- Task 4: `to_state_dict` now includes `trailing_floor` and `equity_high_water`. `restore_from_state` restores both when `saved_date == today`. Day-transition reset is handled in `check_trailing_dd` (resets to `None` on new date, re-initializes on next call).
- Task 5: 13 tests written covering all ACs. All pass.

### Review Findings

### File List

- `src/research/tier2_streaming_working.py` — AccountConfig (4 new fields), RiskManager (trailing DD logic + property), _poll_and_process (per-bar equity calc + check_trailing_dd call)
- `tests/unit/test_risk_manager.py` — new, 13 tests
- `tests/unit/test_realtime_metrics_display.py` — fix pre-existing assertion ("SKIP" → "SKIP:VOL_REGIME")

## Change Log

- 2026-06-01: Story implemented — AccountConfig trailing DD fields, RiskManager.check_trailing_dd(), bar loop wiring, state persistence. 13 unit tests passing.
