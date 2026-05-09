---
title: 'Entry Logic Implementation'
slug: 'entry-logic-implementation'
created: '2026-03-31T19:25:00Z'
status: 'ready-for-dev'
epic: 2
story_id: 2.3
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['Python 3.11+', 'pandas', 'pydantic', 'pytest', 'asyncio']
files_to_modify: ['src/execution/entry_logic.py (NEW)', 'src/execution/models.py (NEW)', 'src/risk/ (EXISTING)', 'tests/unit/test_entry_logic.py (NEW)', 'tests/integration/test_entry_integration.py (NEW)']
code_patterns: ['Pydantic models for trade orders', 'async/await for order processing', 'Risk validation before entry', 'Position sizing algorithms']
test_patterns: ['pytest with test classes', 'fixture-based signal generation', 'mock risk management', 'integration tests for full pipeline']
---

# Story 2.3: Entry Logic Implementation

## Story

**As a** trader developing the ensemble system,
**I Want** automated entry logic that determines position sizing (1-5 contracts) and executes trades when ensemble signals meet quality criteria,
**So that the** system can automatically enter trades with proper risk management for futures trading.

## Acceptance Criteria

**Given** I have ensemble signals from the weighted scorer (Story 2.2)
**When** I implement the EntryLogic class
**Then** it validates entry criteria:
  - Composite confidence > threshold (from Story 2.2)
  - Daily loss limit not exceeded ($1000 daily loss limit)
  - Max drawdown not exceeded (12% of account)
  - Not currently in a position (no overlapping positions)

**Given** entry criteria are validated
**When** I calculate position size
**Then** the system determines position size using:
  - Base position: 1 contract (default)
  - Scale up to 5 contracts based on confidence strength:
    - Confidence 0.50-0.60: 1 contract
    - Confidence 0.60-0.70: 2 contracts
    - Confidence 0.70-0.80: 3 contracts
    - Confidence 0.80-0.90: 4 contracts
    - Confidence >0.90: 5 contracts
  - Max position size: 5 contracts (hard limit)
  - Minimum position size: 1 contract
**And** position size is recorded for P&L calculation

**Given** position size is calculated
**When** multiple signals arrive simultaneously
**Then** the system:
  - Prioritizes signals by composite confidence score (highest first)
  - Executes highest-confidence signal first
  - Re-evaluates remaining signals after first entry
  - Skips signals if max position count reached (5 total positions)

**Given** entry is ready to execute
**When** I create the trade order
**Then** the order includes:
  - Symbol (MNQ futures contract)
  - Direction (long/short)
  - Quantity (number of contracts from position sizing, 1-5)
  - Order type (market or limit based on market conditions)
  - Entry price (or limit price if using limit order)
  - Stop loss price (initial stop)
  - Take profit price (initial target)
  - Timestamp
  - Trade ID (unique identifier)

**Given** I need to integrate with risk management (FR13)
**When** I validate entry risk
**Then** the system checks:
  - Daily P&L (today's realized + unrealized losses < $1000 limit)
  - Account drawdown (current equity < 12% below peak)
  - Open positions count (max 5 total positions across all strategies)
  - Per-trade stop loss defined (from individual strategy signals)
**And** entries are rejected if any risk limit is breached
**And** rejections are logged with reason for monitoring

**Given** I need to integrate with existing system
**When** the entry logic is implemented
**Then** it follows existing patterns from src/execution/
**And** it uses existing risk management from src/risk/
**And** it uses Pydantic models for trade orders
**And** it includes unit tests for:
  - Position sizing calculations (1-5 contracts)
  - Risk validation ($1000 daily loss, 12% drawdown)
  - Signal prioritization
  - Order creation
**And** it uses type hints (mypy strict mode compliant)

**Given** entry orders are created
**When** they are ready for execution
**Then** orders are passed to execution layer (Epic 4 - Paper Trading)
**And** Story 2.3 does NOT execute orders directly (maintains dependency rule)
**And** orders include all metadata needed for execution and monitoring

**Given** I need to track entry decisions
**When** entries are validated or rejected
**Then** the system logs:
  - Entry decision (accepted/rejected)
  - Composite confidence score
  - Position size (1-5 contracts)
  - Risk check results (all limits verified)
  - Rejection reason (if applicable)
**And** logs support audit trail for compliance (NFR5)

## Tasks & Acceptance

### Tasks

- [x] **Task 1: Create Trade Order Models**
  - **File**: `src/execution/models.py` (NEW)
  - **Action**: Define Pydantic models for trade orders
  - **Implementation**:
    1. Create `TradeOrder` Pydantic model:
       - trade_id: str (unique UUID)
       - symbol: str = "MNQ"
       - direction: Literal["long", "short"]
       - quantity: int (1-5 contracts)
       - order_type: Literal["market", "limit"]
       - entry_price: float
       - limit_price: float | None
       - stop_loss: float
       - take_profit: float
       - timestamp: datetime
       - status: Literal["pending", "submitted", "filled", "rejected", "cancelled"]
       - ensemble_signal: EnsembleTradeSignal (reference)
       - position_size: int (1-5)
    2. Create `EntryDecision` Pydantic model:
       - signal: EnsembleTradeSignal
       - position_size: int (1-5)
       - risk_checks_passed: bool
       - risk_check_details: dict (daily_pnl, drawdown, open_positions, stop_loss_defined)
       - decision: Literal["ACCEPT", "REJECT"]
       - rejection_reason: str | None
       - timestamp: datetime
    3. Add field validators:
       - Quantity must be 1-5
       - Stop loss direction validated
       - Take profit respects 2:1 from entry
    4. Add helper methods:
       - `notional_value(self) -> float` - Calculate contract notional
       - `risk_per_contract(self) -> float` - Entry to stop loss distance
  - **Dependencies**: Story 2.2 (needs EnsembleTradeSignal model)

- [x] **Task 2: Create Position Sizing Algorithm**
  - **File**: `src/execution/entry_logic.py` (NEW)
  - **Action**: Implement confidence-based position sizing
  - **Implementation**:
    1. Create `PositionSizer` class:
       - `__init__(self, min_contracts: int = 1, max_contracts: int = 5)`
       - `calculate_position_size(self, confidence: float) -> int`
       - `get_confidence_tier(self, confidence: float) -> str`
    2. Implement confidence-based sizing:
       - Define confidence tiers:
         - Tier 1: 0.50-0.60 → 1 contract
         - Tier 2: 0.60-0.70 → 2 contracts
         - Tier 3: 0.70-0.80 → 3 contracts
         - Tier 4: 0.80-0.90 → 4 contracts
         - Tier 5: >0.90 → 5 contracts
       - Enforce min/max limits (1-5)
    3. Add position size validation:
       - Verify quantity within [1, 5]
       - Raise `ValueError` if confidence outside [0, 1]
       - Log tier assignment for transparency
    4. Create position size history tracker:
       - Track position sizes over time
       - Calculate average position size
       - Distribution by tier (1-5 contracts)
    5. Add unit tests for all confidence ranges
  - **Dependencies**: None (standalone algorithm)

- [x] **Task 3: Create Risk Validation Integration**
  - **File**: `src/execution/entry_logic.py` (EXTEND)
  - **Action**: Integrate with existing risk management
  - **Implementation**:
    1. Create `RiskValidator` class (or integrate with existing src/risk/):
       - `__init__(self, risk_manager: RiskManager)`
       - `validate_entry(self, decision: EntryDecision) -> EntryDecision`
       - `check_daily_loss_limit(self, current_pnl: float, limit: float = 1000.0) -> bool`
       - `check_drawdown_limit(self, current_equity: float, peak_equity: float, max_drawdown: float = 0.12) -> bool`
       - `check_open_positions(self, open_positions: int, max_positions: int = 5) -> bool`
       - `check_stop_loss_defined(self, signal: EnsembleTradeSignal) -> bool`
    2. Implement risk checks:
       - Daily loss limit: Today's P&L > -$1000
       - Drawdown limit: Current equity > peak_equity × (1 - 0.12)
       - Position count: Open positions < 5
       - Stop loss: Signal has valid stop loss price
    3. Create risk check result model:
       - check_name: str
       - passed: bool
       - current_value: float
       - limit_value: float
       - message: str
    4. Aggregate all checks:
       - All must pass for entry to proceed
       - Any fail → reject with specific reason
       - Log each check result
    5. Return updated EntryDecision with risk_check_details
  - **Dependencies**: Task 1 (need EntryDecision model), existing src/risk/

- [x] **Task 4: Create EntryLogic Class**
  - **File**: `src/execution/entry_logic.py` (EXTEND)
  - **Action**: Implement main entry logic orchestrator
  - **Implementation**:
    1. Create `EntryLogic` class:
       - `__init__(self, position_sizer: PositionSizer, risk_validator: RiskValidator)`
       - `process_signal(self, signal: EnsembleTradeSignal) -> EntryDecision`
       - `create_trade_order(self, decision: EntryDecision) -> TradeOrder`
       - `prioritize_signals(self, signals: list[EnsembleTradeSignal]) -> list[EnsembleTradeSignal]`
    2. Implement signal processing:
       - Calculate position size from confidence
       - Run all risk checks
       - Generate EntryDecision (ACCEPT/REJECT)
       - Log decision with full details
    3. Implement order creation:
       - Convert accepted EntryDecision to TradeOrder
       - Generate unique trade_id (UUID)
       - Set initial status to "pending"
       - Include reference to ensemble signal
    4. Implement signal prioritization:
       - Sort signals by composite_confidence (descending)
       - Handle multiple simultaneous signals
       - Re-evaluate risk limits after each entry
    5. Add comprehensive logging:
       - Log all entry decisions
       - Log rejection reasons
       - Log position size calculations
       - Log risk check results
  - **Dependencies**: Tasks 1-3

## Dev Notes

### Architecture Requirements

**Position Sizing Logic:**
- Scale 1-5 contracts based on confidence
- Higher confidence → larger position
- Hard limits: min 1, max 5 contracts
- Simple tiered approach (not fractional contracts)

**Risk Management Integration:**
- Use existing src/risk/ infrastructure if available
- If not available, create basic risk checks:
  - Daily loss limit: $1000
  - Max drawdown: 12% of account
  - Max positions: 5 open positions
  - Stop loss must be defined

**Signal Prioritization:**
- When multiple signals arrive simultaneously
- Sort by composite confidence (highest first)
- Execute highest confidence first
- Re-check risk limits before each subsequent entry

### Technical Implementation Details

**Position Sizing Table:**
| Confidence Range | Position Size | Notional at $11800 (approx) |
|-----------------|---------------|---------------------------|
| 0.50 - 0.60     | 1 contract    | $5,900                    |
| 0.60 - 0.70     | 2 contracts   | $11,800                   |
| 0.70 - 0.80     | 3 contracts   | $17,700                   |
| 0.80 - 0.90     | 4 contracts   | $23,600                   |
| > 0.90          | 5 contracts   | $29,500                   |

**MNQ Contract Specifications:**
- Contract multiplier: $0.50 per point
- 1 point = 4 ticks
- Tick size: 0.25 points
- Notional value = price × $0.50 × quantity

**Risk Limits:**
- Daily loss limit: $1000 (configurable)
- Max drawdown: 12% (configurable)
- Max positions: 5 (hard limit for this story)
- Per-trade risk: Based on signal stop loss

**Entry Order Types:**
- Market order: Execute immediately at current price
- Limit order: Execute at specified price or better
- This story creates orders but does NOT execute (Epic 4 executes)

### Dependencies on Other Stories

**Required:**
- Story 2.1 (Ensemble Signal Aggregation) - provides EnsembleSignal
- Story 2.2 (Weighted Confidence Scoring) - provides EnsembleTradeSignal

**Enables:**
- Story 2.4 (Exit Logic Implementation) - manages positions created here
- Story 2.6 (Ensemble Backtesting) - tests entry logic
- Epic 4 (Paper Trading Deployment) - executes orders

### Testing Strategy

**Unit Tests:**
- Test position sizing for all confidence ranges
- Test all risk checks (daily loss, drawdown, positions, stop loss)
- Test signal prioritization
- Test order creation
- Test rejection logic

**Integration Tests:**
- Test full entry logic pipeline with ensemble signals
- Test risk management integration
- Test multiple signal handling
- Test rejection scenarios

**Test Data:**
- Generate signals with varying confidences
- Mock risk manager for testing
- Test edge cases (confidence = 0, 1, exactly on tier boundaries)

## Dev Agent Record

### Implementation Plan

**Phase 1: Models and Position Sizing (Tasks 1-2)**
- Create TradeOrder and EntryDecision models
- Implement PositionSizer class
- Test confidence-based position sizing

**Phase 2: Risk Integration (Task 3)**
- Create RiskValidator class
- Integrate with existing src/risk/ or create new
- Implement all 4 risk checks
- Test risk validation

**Phase 3: Entry Logic Orchestrator (Task 4)**
- Create EntryLogic class
- Implement signal processing
- Implement order creation
- Add signal prioritization
- Test full pipeline

### Debug Log
*Implementation notes will be added during development*

### Completion Notes

✅ **Story 2.3: Entry Logic Implementation - COMPLETED**

**Implementation Summary:**
- Created complete trade order and entry decision models with Pydantic validation
- Implemented confidence-based position sizing with 5 tiers (1-5 contracts)
- Integrated with existing risk management system for comprehensive validation
- Built full entry logic orchestrator for signal processing and order creation

**Key Features Implemented:**
1. **TradeOrder Model**: Complete order model with validation, helper methods for notional value and risk calculation
2. **EntryDecision Model**: Decision model with detailed risk check results
3. **PositionSizer**: Confidence-based position scaling (0.50-0.60→1, 0.60-0.70→2, 0.70-0.80→3, 0.80-0.90→4, >0.90→5)
4. **RiskValidator**: Integration with risk orchestrator for all 4 risk checks (daily loss, drawdown, positions, stop loss)
5. **EntryLogic**: Main orchestrator for signal processing, decision making, and order creation

**Test Coverage:**
- 41 tests total (7 + 14 + 11 unit tests, 9 integration tests)
- All tests passing
- Coverage includes: model validation, position sizing, risk validation, full pipeline, edge cases

**Files Created:**
- 2 source files (models.py, entry_logic.py)
- 4 test files with comprehensive coverage
- Total: ~850 lines of production code + ~900 lines of test code

**Dependencies Integrated:**
- Story 2.2 EnsembleTradeSignal model
- Existing src/risk/ RiskOrchestrator
- Existing config-sim.yaml for risk limits

## File List

*New files created:*
- `src/execution/entry_logic.py`
- `src/execution/models.py`
- `tests/unit/test_entry_logic.py`
- `tests/unit/test_position_sizing.py`
- `tests/unit/test_entry_logic_integration.py`
- `tests/integration/test_entry_integration.py`

*Existing files modified:*
- `src/execution/__init__.py` (added exports for TradeOrder, EntryDecision)

## Change Log

**2026-04-01**
- Created `src/execution/models.py` with TradeOrder and EntryDecision Pydantic models
- Created `src/execution/entry_logic.py` with PositionSizer, RiskValidator, and EntryLogic classes
- Updated `src/execution/__init__.py` to export new models
- Created comprehensive unit tests:
  - `tests/unit/test_entry_logic.py` (7 tests for models)
  - `tests/unit/test_position_sizing.py` (14 tests for position sizing)
  - `tests/unit/test_entry_logic_integration.py` (11 tests for risk validation and entry logic)
- Created integration tests:
  - `tests/integration/test_entry_integration.py` (9 tests for full pipeline)
- All 41 tests passing
- Full implementation of confidence-based position scaling (1-5 contracts)
- Full integration with risk management (daily loss, drawdown, positions, stop loss)

## Status

**Status:** review

**Last Updated:** 2026-04-01T00:00:00Z

**All Tasks Completed:**
- ✅ Task 1: Trade Order Models (TradeOrder, EntryDecision)
- ✅ Task 2: Position Sizing Algorithm (PositionSizer with 5 tiers)
- ✅ Task 3: Risk Validation Integration (RiskValidator)
- ✅ Task 4: Entry Logic Orchestrator (EntryLogic with signal prioritization)

**Test Results:** 41/41 tests passing
**Files Changed:** 6 files (2 source, 4 test)
**Lines of Code:** ~850 production + ~900 test
