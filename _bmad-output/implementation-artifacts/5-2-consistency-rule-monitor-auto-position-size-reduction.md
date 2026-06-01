# Story 5.2: Consistency Rule Monitor and Auto Position Size Reduction

Status: review

## Story

As Alex,
I want the system to monitor my best-day profit as a percentage of total accumulated profit and automatically reduce position size when approaching the 50% threshold,
So that I never inadvertently violate the TopStep consistency rule during a strong trading day.

## Background

TopStep Combine accounts enforce a consistency rule: no single trading day's profit may exceed 50% of the total accumulated profit across the evaluation period. Violation disqualifies the Combine attempt.

Example: If total accumulated profit = $1,370 and today's net PnL = $520, the ratio is 37.9% — safe. But if the session runs further and daily PnL approaches $685, the ratio hits 50% and the day becomes disqualifying. The system must auto-reduce to 1 contract when the ratio approaches the threshold so the user cannot accidentally cross it.

This extends `RiskManager` (Phase 2 evaluator slot) and `AccountConfig`. No strategy-core logic changes required — the contract override happens at the entry decision boundary in `_enter_trade()`.

## Acceptance Criteria

**Given** accumulated session PnL history with `best_day_pnl = $520` and `total_accumulated_pnl = $1,370`,
**When** `calc_consistency_ratio(session_pnls: list[float]) -> float` is called,
**Then** it returns `520 / 1370 * 100 = 37.96%` (FR18)

**Given** consistency ratio ≥ `config.consistency_alert_pct` (e.g., 40%),
**When** a new bar is processed,
**Then** the system logs `CONSISTENCY_ALERT: ratio at 40.2% — approaching 50% threshold` and displays the alert

**Given** consistency ratio ≥ `config.consistency_reduce_pct` (e.g., 45%),
**When** `_enter_trade()` resolves `contracts`,
**Then** `contracts` is overridden to `1` (minimum) for the remainder of the session (FR19)

**Given** position size was reduced due to consistency rule,
**When** the next calendar day begins,
**Then** normal contract sizing per `StrategyConfig.contracts_per_trade` is restored — the override does not persist across days

**Given** the metrics display,
**When** consistency ratio is shown,
**Then** output includes: `Consistency: 37.9% (alert at 40%, reduce at 45%, limit at 50%)` and whether size reduction is currently active

## Tasks / Subtasks

- [x] Task 1: Add consistency rule configuration to `AccountConfig` (AC: #1–#3)
  - [x] Add `consistency_alert_pct: float = 0.40` field
  - [x] Add `consistency_reduce_pct: float = 0.45` field
  - [x] Add `consistency_hard_limit_pct: float = 0.50` field (for display reference only)

- [x] Task 2: Implement `calc_consistency_ratio()` pure function (AC: #1)
  - [x] `def calc_consistency_ratio(session_pnls: list[float]) -> float` — add to `strategy_core.py` alongside other pure functions
  - [x] Definition: `best_day = max(positive_pnls)`, `total = sum(positive_pnls)`, return `best_day / total * 100` if `total > 0` else `0.0`
  - [x] Only count positive days in both numerator and denominator (losing days don't count toward or against consistency)
  - [x] Return `0.0` if no profitable days exist yet

- [x] Task 3: Add `ConsistencyEvaluator` inside `RiskManager` (AC: #2–#4)
  - [x] Track `_session_pnls: list[float]` — one entry per completed trading day (positive and negative)
  - [x] Track `_today_pnl_for_consistency: float` — current day's running PnL (updated via `register_close`)
  - [x] Track `_consistency_size_reduced: bool` — whether 1-contract override is active today
  - [x] `check_consistency(account_config: AccountConfig) -> tuple[float, bool]` — returns (ratio, should_reduce)
  - [x] On new calendar day: append previous day's PnL to `_session_pnls`, reset `_today_pnl_for_consistency = 0.0`, reset `_consistency_size_reduced = False`
  - [x] Log `CONSISTENCY_ALERT` when ratio ≥ `consistency_alert_pct`
  - [x] Set `_consistency_size_reduced = True` when ratio ≥ `consistency_reduce_pct` (never set back to False within same day)

- [x] Task 4: Wire contract cap into `_enter_trade()` (AC: #3, #4)
  - [x] After `make_entry_decision()` resolves `entry_dec.contracts`, check `risk_manager.consistency_size_reduced`
  - [x] If reduced: replace contracts with `1` via new `EntryDecision` with `contracts=1`
  - [x] Log `CONSISTENCY_REDUCE: capping contracts to 1 (ratio=X.X%)`

- [x] Task 5: Persist and restore consistency state (AC: #4)
  - [x] Extend `RiskManager.to_state_dict()` to include `session_pnls`, `today_pnl_for_consistency`, `consistency_size_reduced`
  - [x] Extend `RiskManager.restore_from_state()` to restore these fields when same-day state found
  - [x] `session_pnls` list persists across days; only `_consistency_size_reduced` and `_today_pnl_for_consistency` reset on new day

- [x] Task 6: Update metrics display (AC: #5)
  - [x] Add consistency ratio to the existing metrics display output
  - [x] Format: `Consistency: 37.9% (alert at 40%, reduce at 45%, limit at 50%) [SIZE REDUCED]`
  - [x] `[SIZE REDUCED]` tag only shown when `_consistency_size_reduced = True`

- [x] Task 7: Unit tests (all ACs)
  - [x] `test_calc_consistency_ratio_typical` — known inputs → expected ratio
  - [x] `test_calc_consistency_ratio_no_profitable_days` → returns 0.0
  - [x] `test_calc_consistency_ratio_ignores_losing_days` — losing days not included
  - [x] `test_alert_logged_at_threshold` — ratio ≥ alert_pct → alert emitted
  - [x] `test_size_reduction_triggered_at_reduce_pct` — ratio ≥ reduce_pct → `_consistency_size_reduced = True`
  - [x] `test_size_reduction_resets_on_new_day` — new calendar day → reduction cleared
  - [x] `test_session_pnls_accumulate_across_days` — multi-day state persists correctly
  - [x] `test_contracts_capped_to_1_when_reduced` — `_enter_trade` produces contracts=1 when reduced

## Dev Notes

### `calc_consistency_ratio()` Implementation

Add to `src/research/strategy_core.py` alongside pure functions (e.g., after `check_exit`):

```python
def calc_consistency_ratio(session_pnls: list[float]) -> float:
    """Return best profitable day / total profitable days * 100.

    TopStep consistency rule: best day ≤ 50% of cumulative profit.
    Returns 0.0 if no profitable sessions yet.
    """
    profitable = [p for p in session_pnls if p > 0]
    if not profitable:
        return 0.0
    return max(profitable) / sum(profitable) * 100.0
```

### Tracking Today's PnL for Consistency

The running daily PnL is already tracked in `RiskManager._daily_pnl` (updated by `register_close`). Use the same field for consistency tracking — no duplication needed. The `_session_pnls` list is appended at day-rollover time (inside `check_and_update` new-day logic).

### Contract Override in `_enter_trade()`

At `tier2_streaming_working.py:1640` (after `entry_dec = make_entry_decision(...)`):

```python
entry_dec = make_entry_decision(sweep, fvg, self._strategy_config)
if entry_dec is None:
    return

# Apply consistency-rule size reduction if active
effective_contracts = entry_dec.contracts
if self._risk_manager.consistency_size_reduced:
    effective_contracts = 1
    logger.info("CONSISTENCY_REDUCE: capping contracts to 1")
snapped_dec = EntryDecision(..., contracts=effective_contracts)
```

### State Dict Keys to Add

```python
# In to_state_dict():
{
    "session_pnls": self._session_pnls,           # list[float], grows across days
    "consistency_size_reduced": self._consistency_size_reduced,  # bool, resets daily
    # daily_pnl already present — doubles as today_pnl_for_consistency
}
```

### References

- FR18, FR19: `_bmad-output/planning-artifacts/prd.md:577–578`
- Epic 5 story spec: `_bmad-output/planning-artifacts/epics.md:968`
- Journey 4 narrative: `_bmad-output/planning-artifacts/prd.md:192–200`
- `RiskManager`: `src/research/tier2_streaming_working.py:280`
- `make_entry_decision` + `EntryDecision`: `src/research/strategy_core.py:613`
- `_enter_trade`: `src/research/tier2_streaming_working.py:1624`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

- Fixed test `test_typical_case`: initial assertion used 520/1370 but actual best day is 850 → corrected to 850/1370*100=62.04%.

### Completion Notes List

- Task 1: Three consistency fields added to `AccountConfig` with defaults (backward-compatible).
- Task 2: `calc_consistency_ratio()` added to `strategy_core.py` after `calc_max_drawdown_pct`. Only positive days count.
- Task 3: `_session_pnls`, `_consistency_size_reduced` added to `RiskManager`. `check_consistency()` computes ratio including today's `_daily_pnl`. Day rollover inside `check_and_update` appends previous day's PnL before reset.
- Task 4: After `make_entry_decision()` in `_enter_trade()`, `effective_contracts` is set to 1 if `consistency_size_reduced` is True.
- Task 5: `to_state_dict`/`restore_from_state` extended with `session_pnls` and `consistency_size_reduced`. `session_pnls` restored always (persists across days); `consistency_size_reduced` restored same-day only.
- Task 6: `_log_trade_metrics()` now logs consistency ratio + reduce tag after PF/Sharpe/MaxDD line.
- Task 7: 12 new tests (5 pure function + 7 RiskManager). All pass.

### Review Findings

### File List

- `src/research/strategy_core.py` — add `calc_consistency_ratio()`
- `src/research/tier2_streaming_working.py` — AccountConfig (3 new fields), RiskManager (consistency state + check_consistency + day rollover + persist/restore), `_enter_trade` (contract cap), `_log_trade_metrics` (consistency display)
- `tests/unit/test_risk_manager.py` — extended with 7 consistency tests
- `tests/unit/test_strategy_core_consistency.py` — new, 5 pure function tests

## Change Log

- 2026-06-01: Story implemented — calc_consistency_ratio, AccountConfig fields, RiskManager.check_consistency(), contract cap in _enter_trade, metrics display. 12 tests passing.
