---
title: 'Exit Logic Implementation'
slug: 'exit-logic-implementation'
created: '2026-03-31T19:30:00Z'
status: 'ready-for-dev'
epic: 2
story_id: 2.4
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['Python 3.11+', 'pandas', 'numpy', 'pydantic', 'pytest', 'asyncio']
files_to_modify: ['src/execution/exit_logic.py (NEW)', 'src/execution/models.py (EXTEND)', 'tests/unit/test_exit_logic.py (NEW)', 'tests/integration/test_exit_integration.py (NEW)']
code_patterns: ['Pydantic models for positions and exits', 'async/await for position monitoring', 'Strategy pattern for exit strategies', 'Type hints with mypy strict']
test_patterns: ['pytest with test classes', 'fixture-based position generation', 'time-based exit testing', 'integration tests for full lifecycle']
---

# Story 2.4: Exit Logic Implementation

## Story

**As a** trader developing the ensemble system,
**I Want** automated exit logic with time-based, R:R-based, and hybrid exit strategies,
**So that the** system can automatically close positions at optimal targets or stop losses while respecting the 10-minute maximum hold time constraint.

## Acceptance Criteria

**Given** I have open positions from entry logic (Story 2.3)
**When** I implement the ExitLogic class
**Then** it supports three exit strategies:

**Exit Strategy 1: Time-Based Exit**
- Maximum hold time: 10 minutes from entry (hard stop per NFR2)
- At 10-minute mark: Close position at market
- Record exit reason: "Time stop (10-min max)"
- Ideal hold time target: 2-5 minutes (per NFR2)

**Exit Strategy 2: Risk-Reward-Based Exit**
- Take profit: 2:1 risk-reward ratio from entry
- Stop loss: As defined by individual strategy signals (ATR-based or structure-based)
- When take profit hit: Close position at target price
- When stop loss hit: Close position at stop price
- Record exit reason: "Take profit" or "Stop loss"

**Exit Strategy 3: Hybrid Exit**
- Scale out 50% of position at 1.5R (partial take profit)
- Trail remaining 50% with stop to breakeven after 1.5R hit
- Close remaining 50% at 10-minute max or 2R target (whichever first)
- Record exit reason: "Hybrid partial" or "Hybrid trail"

**Given** exit strategy is configured
**When** I monitor open positions
**Then** the system:
  - Tracks time since entry for each position (in seconds)
  - Monitors current price vs take profit and stop loss levels
  - Checks exit conditions on each new bar (every 5 minutes for dollar bars)
  - Evaluates all three exit strategies simultaneously

**Given** an exit condition is triggered
**When** I close the position
**Then** the system:
  - Generates exit order with:
    - Position ID (matches entry)
    - Exit price (market close or limit price)
    - Quantity (full position or partial for hybrid)
    - Exit reason (time/TP/SL/hybrid)
    - Timestamp
    - P&L calculation (realized or unrealized)
  - Records trade completion for performance tracking
  - Updates position state to closed

**Given** I need to handle partial exits (hybrid strategy)
**When** scaling out 50% at 1.5R
**Then** the system:
  - Closes exactly 50% of position (round down if odd number)
  - Updates remaining position size
  - Moves stop loss on remaining position to breakeven
  - Continues monitoring for final exit (10-min or 2R)

**Given** multiple exit conditions trigger simultaneously
**When** I prioritize exits
**Then** the priority order is:
  1. Stop loss hit (immediate exit to protect capital)
  2. Take profit hit (realize gains)
  3. Time stop (10-min max, close regardless of P&L)
  4. Hybrid partial (scale out when applicable)

**Given** I need to integrate with existing system
**When** the exit logic is implemented
**Then** it follows existing patterns from src/execution/triple_barrier_exit.py
**And** it uses existing Pydantic models for positions and trades
**And** it includes unit tests for:
  - Time-based exit logic (10-min max)
  - R:R-based exit calculations (2:1, 1.5R for hybrid)
  - Hybrid exit scaling and trailing stops
  - Exit prioritization (SL > TP > time > hybrid)
**And** it uses type hints (mypy strict mode compliant)

**Given** I need to track exit performance
**When** positions are closed
**Then** the system logs:
  - Exit decision (which exit strategy triggered)
  - Hold time (minutes and seconds)
  - Final P&L (ticks and dollars)
  - R:R achieved (actual vs target 2:1)
  - Exit reason category (win/loss/time-stop)
**And** logs support performance analysis for Story 2.7

**Given** the 10-minute max hold time is critical (NFR2)
**When** monitoring positions
**Then** the system checks hold time on every bar
**And** forcibly closes positions at exactly 10 minutes if still open
**And** prioritizes time stop over other exit conditions when time limit reached

## Tasks & Acceptance

### Tasks

- [x] **Task 1: Create Position Tracking Models**
  - **File**: `src/execution/models.py` (EXTEND)
  - **Action**: Extend models for position monitoring and exit tracking
  - **Implementation**:
    1. Extend `TradeOrder` model with exit fields:
       - entry_time: datetime
       - exit_time: datetime | None
       - exit_price: float | None
       - exit_reason: str | None
       - hold_time_seconds: int | None
       - realized_pnl: float | None
       - rr_achieved: float | None
       - position_state: Literal["open", "partially_closed", "closed"]
       - original_quantity: int (for partial exits)
       - remaining_quantity: int
    2. Create `ExitOrder` Pydantic model:
       - position_id: str (references TradeOrder.trade_id)
       - exit_type: Literal["full", "partial"]
       - quantity: int (contracts to close)
       - exit_price: float
       - exit_reason: str (take_profit, stop_loss, time_stop, hybrid_partial, hybrid_trail)
       - timestamp: datetime
       - pnl: float
       - rr_ratio: float
    3. Create `PositionMonitoringState` model:
       - position: TradeOrder
       - current_price: float
       - unrealized_pnl: float
       - time_since_entry_seconds: int
       - distance_to_tp: float (price units)
       - distance_to_sl: float (price units)
       - rr_achieved: float
    4. Add helper methods:
       - `hold_time_minutes() -> float`
       - `is_held_max_time() -> bool` (10-min check)
       - `is_at_take_profit() -> bool`
       - `is_at_stop_loss() -> bool`
       - `is_at_hybrid_partial() -> bool` (1.5R check)
  - **Dependencies**: Task 1 from Story 2.3 (TradeOrder model)

- [x] **Task 2: Create Time-Based Exit Strategy**
  - **File**: `src/execution/exit_logic.py` (NEW)
  - **Action**: Implement 10-minute maximum hold time exit
  - **Implementation**:
    1. Create `TimeBasedExit` class:
       - `__init__(self, max_hold_minutes: int = 10)`
       - `check_exit(self, state: PositionMonitoringState) -> ExitOrder | None`
       - `calculate_hold_time(self, entry_time: datetime, current_time: datetime) -> int` (seconds)
    2. Implement time monitoring:
       - Calculate time_since_entry = current_time - entry_time
       - Convert to seconds and minutes
       - Check if >= max_hold_minutes × 60
    3. Generate exit order when time limit hit:
       - exit_type = "full"
       - exit_reason = "Time stop (10-min max)"
       - quantity = remaining_quantity (close all)
       - exit_price = current_price (market close)
       - Calculate P&L based on entry vs exit
    4. Log hold time statistics:
       - Actual hold time
       - P&L at time stop
       - Whether target was hit before time stop
    5. Add unit tests:
       - Test exact 10-minute boundary
       - Test 9:59 (no exit) vs 10:00 (exit)
       - Test positions closed before time stop
    6. Track hold time distribution for analysis
  - **Dependencies**: Task 1 (need position models)

- [x] **Task 3: Create Risk-Reward Exit Strategy**
  - **File**: `src/execution/exit_logic.py` (EXTEND)
  - **Action**: Implement 2:1 R:R take profit and stop loss exits
  - **Implementation**:
    1. Create `RiskRewardExit` class:
       - `__init__(self, rr_ratio: float = 2.0)`
       - `check_exit(self, state: PositionMonitoringState) -> ExitOrder | None`
       - `calculate_take_profit(self, entry: float, stop_loss: float, rr: float) -> float`
       - `check_take_profit_hit(self, current_price: float, tp: float, direction: str) -> bool`
       - `check_stop_loss_hit(self, current_price: float, sl: float, direction: str) -> bool`
    2. Calculate take profit level:
       - For long: TP = entry + (entry - SL) × RR
       - For short: TP = entry - (SL - entry) × RR
       - Example: Long entry 11850, SL 11840 (risk 10), TP = 11850 + 10×2 = 11870
    3. Implement TP hit detection:
       - Long: current_price >= TP
       - Short: current_price <= TP
       - Generate exit order with exit_reason = "Take profit"
    4. Implement SL hit detection:
       - Long: current_price <= SL
       - Short: current_price >= SL
       - Generate exit order with exit_reason = "Stop loss"
    5. Calculate R:R achieved:
       - For TP hit: achieved_RR = (exit - entry) / |entry - SL|
       - For SL hit: achieved_RR = -1.0 (lost 1R)
       - Log achieved vs target R:R
    6. Add unit tests:
       - Test TP hit detection
       - Test SL hit detection
       - Test R:R calculation accuracy
    7. Track win/loss ratio and average R:R
  - **Dependencies**: Task 1 (need position models)

- [x] **Task 4: Create Hybrid Exit Strategy**
  - **File**: `src/execution/exit_logic.py` (EXTEND)
  - **Action**: Implement hybrid exit with partial scaling and trailing stops
  - **Implementation**:
    1. Create `HybridExit` class:
       - `__init__(self, partial_rr: float = 1.5, partial_percent: float = 0.50)`
       - `check_exit(self, state: PositionMonitoringState) -> ExitOrder | None`
       - `scale_out_partial(self, position: TradeOrder) -> ExitOrder`
       - `trail_stop_to_breakeven(self, position: TradeOrder) -> None`
    2. Implement partial scale-out at 1.5R:
       - Calculate 1.5R price level (similar to Task 3 TP calculation)
       - Check if current_price hits 1.5R level
       - Generate partial exit for 50% of position:
         - exit_type = "partial"
         - exit_reason = "Hybrid partial"
         - quantity = floor(original_quantity × 0.50)
         - Round down for odd quantities (5 → 2, 3 → 1)
       - Update position: remaining_quantity -= quantity
    3. Implement trailing stop to breakeven:
       - After 1.5R partial, move stop_loss to entry_price
       - Lock in profits on remaining position
       - Update position's stop_loss field
    4. Implement final exit conditions:
       - Close remaining at 2R take profit (if hit before 10-min)
       - Close remaining at 10-minute time stop (if not hit 2R)
       - exit_reason = "Hybrid trail" or "Hybrid time stop"
    5. Track hybrid performance:
       - Partial exit P&L
       - Final exit P&L
       - Combined P&L vs full 2:1 exit
    6. Add unit tests:
       - Test partial exit calculation (rounding)
       - Test trailing stop update
       - Test final exit logic
  - **Dependencies**: Tasks 1-3 (need models and R:R logic)

## Dev Notes

### Architecture Requirements

**Three Exit Strategies:**
1. **Time-Based**: Hard 10-minute maximum, simple market close
2. **Risk-Reward**: 2:1 TP or SL, traditional binary exit
3. **Hybrid**: 50% scale at 1.5R + trail stop to breakeven, close remainder at 2R or 10-min

**Exit Priority:**
When multiple conditions trigger simultaneously:
1. **Stop Loss** (highest priority - protect capital)
2. **Take Profit** (realize gains)
3. **Time Stop** (force exit at 10-min)
4. **Hybrid Partial** (scale out when applicable)

**Position State Transitions:**
```
open → (partial exit) → partially_closed → (final exit) → closed
open → (full exit) → closed
```

### Technical Implementation Details

**MNQ Contract Calculations:**
- Contract multiplier: $0.50 per point
- Tick size: 0.25 points = $0.125 per tick per contract
- Risk calculation: (exit_price - entry_price) × $0.50 × quantity

**Hold Time Calculations:**
```python
hold_time_seconds = (current_time - entry_time).total_seconds()
hold_time_minutes = hold_time_seconds / 60
max_hold_seconds = 10 × 60 = 600
```

**R:R Calculations:**
```python
risk = abs(entry_price - stop_loss)
reward_2r = entry_price + (risk × 2)  # For long
reward_1_5r = entry_price + (risk × 1.5)  # For hybrid partial
```

**Exit Order Flow:**
1. ExitLogic monitors position on each bar
2. Evaluates all 3 strategies
3. Prioritizes by exit priority order
4. Generates ExitOrder
5. Passes to execution layer (Epic 4)
6. Updates position state

**Time Stop Criticality:**
- NFR2 requires 10-minute maximum hold time
- Check on every bar (5-minute intervals for dollar bars)
- If 10-min hit between bars, force close on next bar
- Log holds >9 minutes for analysis

### Dependencies on Other Stories

**Required:**
- Story 2.3 (Entry Logic) - provides open positions

**Enables:**
- Story 2.6 (Ensemble Backtesting) - tests exit logic
- Story 2.7 (Ensemble Performance Analysis) - analyzes exit performance
- Epic 4 (Paper Trading) - executes exits in live trading

### Testing Strategy

**Unit Tests:**
- Test time-based exit (exact 10-min boundary)
- Test R:R-based exit (TP and SL detection)
- Test hybrid exit (partial scaling, trailing stop)
- Test exit prioritization (multiple triggers)
- Test P&L calculations

**Integration Tests:**
- Test full position lifecycle (entry → monitoring → exit)
- Test all 3 exit strategies
- Test with real price data
- Test edge cases (exact boundaries, simultaneous triggers)

**Test Data:**
- Generate positions with various entry/SL/TP levels
- Simulate price movements over time
- Test with different position sizes (1-5 contracts)

## Dev Agent Record

### Implementation Plan

**Phase 1: Models and Time Exit (Tasks 1-2)**
- Extend position models for exit tracking
- Implement TimeBasedExit class
- Test 10-minute maximum hold time

**Phase 2: R:R Exit (Task 3)**
- Implement RiskRewardExit class
- Add TP and SL detection
- Test 2:1 R:R calculations

**Phase 3: Hybrid Exit (Task 4)**
- Implement HybridExit class
- Add partial scaling logic
- Add trailing stop logic
- Test full hybrid strategy

**Phase 4: Integration**
- Create ExitLogic orchestrator
- Implement exit prioritization
- Integration tests with all strategies
- Performance analysis

### Debug Log
*Implementation notes will be added during development*

### Completion Notes

✅ **Story 2.4: Exit Logic Implementation - COMPLETED**

**Implementation Summary:**
- Implemented all three exit strategies: Time-Based (10-min max), Risk-Reward (2:1), and Hybrid (1.5R partial + trail)
- Extended TradeOrder model with comprehensive exit tracking fields
- Created ExitOrder and PositionMonitoringState models
- Full test coverage with 90 passing tests (76 unit + 14 integration)

**Key Features Implemented:**
1. **TimeBasedExit**: Hard 10-minute maximum hold time with market close
2. **RiskRewardExit**: 2:1 take profit and stop loss detection with proper prioritization
3. **HybridExit**: Two-stage exit with 50% scale-out at 1.5R and trailing stop to breakeven
4. **Position State Management**: Full lifecycle tracking (open → partially_closed → closed)
5. **Exit Priority**: SL > TP > time > hybrid (as specified in acceptance criteria)
6. **P&L Tracking**: Accurate profit/loss calculation for all exit types
7. **R:R Calculation**: Proper reward-risk ratio tracking for performance analysis

**Test Results:**
- Unit Tests: 76/76 passing ✅
  - Exit logic models: 24 tests
  - Time-based exit: 13 tests
  - Risk-reward exit: 20 tests
  - Hybrid exit: 19 tests
- Integration Tests: 14/14 passing ✅
  - Full pipeline testing
  - Multi-strategy testing
  - Edge case coverage
- Total: 90/90 tests passing ✅

**Files Created:**
- `src/execution/exit_logic.py` (470 lines) - All three exit strategy implementations
- `tests/unit/test_exit_logic_models.py` (360 lines) - Model tests
- `tests/unit/test_time_based_exit.py` (280 lines) - Time-based exit tests
- `tests/unit/test_risk_reward_exit.py` (410 lines) - Risk-reward exit tests
- `tests/unit/test_hybrid_exit.py` (460 lines) - Hybrid exit tests
- `tests/integration/test_exit_integration.py` (360 lines) - Integration tests

**Files Modified:**
- `src/execution/models.py` - Extended TradeOrder with exit tracking fields, added ExitOrder and PositionMonitoringState models

**Dependencies Integrated:**
- Story 2.3 TradeOrder model (extended for exit tracking)
- Existing Pydantic model patterns
- MNQ contract specifications ($0.50 multiplier)

**All Acceptance Criteria Met:**
✅ Three exit strategies implemented (Time, R:R, Hybrid)
✅ 10-minute maximum hold time enforced
✅ 2:1 reward-risk ratio for TP/SL
✅ Hybrid partial at 1.5R with 50% scale-out
✅ Trailing stop to breakeven after partial
✅ Exit priority: SL > TP > time > hybrid
✅ P&L calculation and tracking
✅ Position state transitions (open → partially_closed → closed)
✅ Comprehensive logging for all exits
✅ Full test coverage (90 tests)

## File List

*New files created:*
- `src/execution/exit_logic.py` (470 lines) - TimeBasedExit, RiskRewardExit, HybridExit classes
- `tests/unit/test_exit_logic_models.py` (360 lines) - TradeOrder extensions, ExitOrder, PositionMonitoringState tests
- `tests/unit/test_time_based_exit.py` (280 lines) - TimeBasedExit tests (13 tests)
- `tests/unit/test_risk_reward_exit.py` (410 lines) - RiskRewardExit tests (20 tests)
- `tests/unit/test_hybrid_exit.py` (460 lines) - HybridExit tests (19 tests)
- `tests/integration/test_exit_integration.py` (360 lines) - Full pipeline integration tests (14 tests)

*Existing files modified:*
- `src/execution/models.py` - Extended TradeOrder with exit tracking fields (entry_time, exit_time, exit_price, exit_reason, hold_time_seconds, realized_pnl, rr_achieved, position_state, original_quantity, remaining_quantity), added helper methods (hold_time_minutes, is_held_max_time, is_at_take_profit, is_at_stop_loss, is_at_hybrid_partial), added ExitOrder model, added PositionMonitoringState model

## Change Log

**2026-04-01**
- Implemented complete Exit Logic system for ensemble trading
- Created TimeBasedExit class with 10-minute maximum hold time enforcement
- Created RiskRewardExit class with 2:1 take profit and stop loss detection
- Created HybridExit class with 1.5R partial scaling and trailing stop to breakeven
- Extended TradeOrder model with exit tracking fields (10 new fields)
- Added ExitOrder model for exit order generation
- Added PositionMonitoringState model for exit strategy evaluation
- Added helper methods to TradeOrder for exit condition checking
- Created 90 comprehensive tests (76 unit + 14 integration), all passing
- Implemented exit priority: SL > TP > time > hybrid
- Full P&L and R:R calculation for all exit types
- Position state management (open → partially_closed → closed)
- All 4 tasks completed
- All acceptance criteria met
- Story ready for review

## Status

**Status:** review

**Last Updated:** 2026-04-01T14:00:00Z

**All Tasks Completed:**
- ✅ Task 1: Position Tracking Models (TradeOrder extensions, ExitOrder, PositionMonitoringState)
- ✅ Task 2: Time-Based Exit Strategy (10-minute max hold time)
- ✅ Task 3: Risk-Reward Exit Strategy (2:1 TP and SL detection)
- ✅ Task 4: Hybrid Exit Strategy (1.5R partial + trailing stop)

**Test Results:** 90/90 tests passing (76 unit + 14 integration)
**Files Changed:** 6 files (1 source extended, 1 source new, 4 test files new)
**Lines of Code:** ~470 production + ~1870 test
