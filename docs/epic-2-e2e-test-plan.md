# Epic 2 E2E Test Plan: Ensemble Integration

**Document Version:** 1.0
**Date:** 2026-04-01
**Epic:** Epic 2 - Ensemble Integration
**Author:** Test Team
**Status:** Draft

---

## 1. Overview

### 1.1 Purpose

This document defines the end-to-end (E2E) testing strategy for Epic 2: Ensemble Integration. The E2E tests validate that all 5 strategies from Epic 1 integrate correctly into the ensemble system, producing reliable composite signals with dynamic weight optimization.

### 1.2 Objectives

- ✅ Validate ensemble signal aggregation from all 5 Epic 1 strategies
- ✅ Verify weighted confidence scoring produces valid composite scores (0-1 range)
- ✅ Test dynamic weight optimization rebalances correctly
- ✅ Confirm entry/exit logic executes with ensemble signals
- ✅ Validate ensemble performance meets or exceeds individual strategies
- ✅ Detect integration issues before Epic 3 (Walk-Forward Validation)

### 1.3 Scope

**In Scope:**
- Integration of all 5 Epic 1 strategies into ensemble
- Ensemble signal aggregation and normalization
- Weighted confidence scoring
- Dynamic weight optimization
- Entry logic with ensemble signals
- Exit logic (time-based, R:R-based, hybrid)
- Performance comparison: ensemble vs individual strategies

**Out of Scope:**
- Walk-forward validation (Epic 3)
- Parameter optimization (Epic 3)
- Paper trading integration (Epic 4)
- Live market data testing
- Performance benchmarking under load

---

## 2. Test Strategy

### 2.1 Testing Approach

**Bottom-Up Integration Testing:**
1. Test individual strategy outputs (already done in Epic 1)
2. Test ensemble signal aggregation with mock strategy outputs
3. Test ensemble with real strategy outputs on sample data
4. Test full ensemble pipeline on historical dataset
5. Performance comparison and analysis

**Test Pyramid:**
```
        E2E (1 suite)
       /              \
  Integration (6 suites)  Unit (Epic 1)
 /    |    \             |
Mocks Real Data Perf     |
```

### 2.2 Test Data Strategy

**3-Tier Data Approach:**

| Tier | Size | Purpose | Source |
|------|------|---------|--------|
| **Synthetic** | 100 bars | Unit/integration tests | Generate programmatically |
| **Sample** | 500 bars | Quick E2E validation | Epic 1 test data |
| **Full** | 116K bars | Performance validation | All HDF5 files |

**Synthetic Data Characteristics:**
- Trending market (50 bars)
- Ranging market (50 bars)
- Known signal locations (embedded for validation)
- Edge cases (gaps, outliers, low volume)

---

## 3. Test Scenarios

### Scenario 1: Ensemble Signal Aggregation

**Objective:** Verify ensemble correctly collects and normalizes signals from all strategies

**Preconditions:**
- All 5 strategies initialized
- Historical data loaded (500 bars)
- Ensemble aggregator configured

**Test Steps:**
1. Load 500 bars of historical MNQ data
2. Process bars through all 5 strategies
3. Collect individual strategy signals
4. Run ensemble signal aggregation
5. Verify ensemble output contains all strategy signals

**Expected Results:**
- Ensemble receives signals from all active strategies
- Confidence scores normalized to 0-1 range
- Signal timestamps aligned correctly
- No signals lost in aggregation

**Success Criteria:**
- ✅ All strategy signals captured
- ✅ Confidence scores in valid range [0, 1]
- ✅ Signal count matches expected (based on Epic 1 results)

---

### Scenario 2: Weighted Confidence Scoring

**Objective:** Validate weighted scoring produces valid composite confidence scores

**Preconditions:**
- Ensemble aggregator initialized with default weights (0.20 each)
- Historical data loaded
- Individual strategy signals available

**Test Steps:**
1. Configure equal weights for all 5 strategies (0.20 each)
2. Process 500 bars through ensemble
3. Calculate weighted confidence scores
4. Validate score distribution
5. Test edge cases (all long, all short, mixed signals)

**Expected Results:**
- Composite scores in range [0, 1]
- Higher confluence = higher composite score
- Equal weights → average of individual confidences
- Conflicting signals reduce composite score appropriately

**Success Criteria:**
- ✅ 100% of scores in valid range
- ✅ Score distribution follows expected pattern
- ✅ Edge cases handled correctly
- ✅ No NaN or infinite values

---

### Scenario 3: Dynamic Weight Optimization

**Objective:** Verify weight optimizer rebalances based on performance

**Preconditions:**
- 4 weeks of historical data available
- Initial weights configured (0.20 each)
- Performance metrics available for each strategy

**Test Steps:**
1. Calculate performance scores for each strategy (Win Rate × Profit Factor)
2. Run weight optimization algorithm
3. Verify new weights calculated correctly
4. Check weight constraints (floor 0.05, ceiling 0.40)
5. Validate weight sum = 1.0

**Expected Results:**
- Weights adjust based on relative performance
- Poor-performing strategies reduced (but not below 0.05)
- High-performing strategies increased (but not above 0.40)
- Sum of all weights = 1.0

**Success Criteria:**
- ✅ All weights in range [0.05, 0.40]
- ✅ Weight sum = 1.0 ± 0.001
- ✅ Weight changes correlate with performance
- ✅ No strategy eliminated or over-concentrated

---

### Scenario 4: Entry Logic Integration

**Objective:** Validate entry logic triggers on ensemble signals correctly

**Preconditions:**
- Ensemble configured with confidence threshold (default 0.50)
- Historical data loaded
- Entry logic module initialized

**Test Steps:**
1. Set confidence threshold to 0.50
2. Process 500 bars through ensemble
3. Count entry signals generated
4. Verify entries only when confidence ≥ threshold
5. Test with different thresholds (0.40, 0.60, 0.70)

**Expected Results:**
- Entry signals generated only above threshold
- Higher threshold = fewer entries
- All entries have valid entry price, stop loss, take profit
- Position sizing calculated correctly

**Success Criteria:**
- ✅ 0 entries below threshold
- ✅ 100% of entries above threshold
- ✅ All entries have complete trade parameters
- ✅ Position sizing respects risk limits (1% of account)

---

### Scenario 5: Exit Logic Integration

**Objective:** Verify all 3 exit modes work correctly with ensemble positions

**Preconditions:**
- Entry logic has generated positions
- Historical data continues after entries
- Exit logic module initialized

**Test Steps:**
1. Generate 10 positions from ensemble entry logic
2. Apply time-based exit (10-min max)
3. Apply R:R-based exit (2:1 target)
4. Apply hybrid exit (scale 50% at 1.5R, trail 50%)
5. Compare exit performance across modes

**Expected Results:**
- Time-based exits trigger at 10 minutes
- R:R-based exits hit 2:1 target or stop loss
- Hybrid exits scale correctly and trail stop
- All exits have valid exit prices and timestamps

**Success Criteria:**
- ✅ 100% of positions exited
- ✅ Exit conditions met correctly
- ✅ No orphaned positions
- ✅ Exit P&L calculated correctly

---

### Scenario 6: Performance Comparison

**Objective:** Compare ensemble performance vs individual strategies

**Preconditions:**
- All strategies backtested individually (Epic 1 results)
- Ensemble backtested on same data
- Performance metrics calculated

**Test Steps:**
1. Run individual strategy backtests (500 bars)
2. Run ensemble backtest (same 500 bars)
3. Calculate performance metrics for all
4. Compare metrics: win rate, profit factor, expectancy
5. Document ensemble advantages/disadvantages

**Expected Results:**
- Ensemble win rate ≥ best individual strategy OR
- Ensemble profit factor ≥ best individual strategy OR
- Ensemble expectancy ≥ best individual strategy
- Lower drawdown due to diversification

**Success Criteria:**
- ✅ Ensemble meets minimum performance targets (60% win rate, 2:1 R:R)
- ✅ Ensemble performance ≥ average of individual strategies
- ✅ Documented performance characteristics
- ✅ Clear recommendation for Epic 3

---

## 4. Test Cases

### TC-E2E-001: Ensemble Initialization

| Field | Value |
|-------|-------|
| **Priority** | P0 (Critical) |
| **Type** | Integration |
| **Data** | Synthetic (100 bars) |

**Steps:**
1. Initialize ensemble with default config
2. Verify all 5 strategies loaded
3. Check initial weights = 0.20 each
4. Validate confidence threshold = 0.50

**Expected:** All components initialized without errors

---

### TC-E2E-002: Signal Aggregation - All Strategies

| Field | Value |
|-------|-------|
| **Priority** | P0 (Critical) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Load 500-bar sample
2. Run all 5 strategies
3. Run ensemble aggregation
4. Count signals per strategy
5. Verify total signal count

**Expected:**
- Triple Confluence: ~155 signals
- Wolf Pack: 0-5 signals
- Adaptive EMA: 0-5 signals (warming up)
- VWAP Bounce: 10-20 signals
- Opening Range: 5-10 signals
- Ensemble captures all signals

---

### TC-E2E-003: Confidence Score Distribution

| Field | Value |
|-------|-------|
| **Priority** | P1 (High) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Run ensemble on 500 bars
2. Extract all confidence scores
3. Calculate distribution stats (min, max, mean, std)
4. Verify all scores in [0, 1]
5. Check for NaN/infinite values

**Expected:**
- Min ≥ 0.0
- Max ≤ 1.0
- Mean between 0.3-0.7
- No NaN/infinite values

---

### TC-E2E-004: Weight Optimization - Performance-Based

| Field | Value |
|-------|-------|
| **Priority** | P1 (High) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Run backtest for 4-week period
2. Calculate performance per strategy
3. Run weight optimizer
4. Verify new weights
5. Check constraints applied

**Expected:**
- Weights adjusted based on performance
- All weights in [0.05, 0.40]
- Sum = 1.0 ± 0.001

---

### TC-E2E-005: Entry Logic - Confidence Threshold

| Field | Value |
|-------|-------|
| **Priority** | P0 (Critical) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Set threshold to 0.50
2. Run ensemble
3. Count entries with confidence ≥ 0.50
4. Verify 0 entries with confidence < 0.50
5. Test with thresholds: 0.40, 0.60, 0.70

**Expected:**
- Entries only above threshold
- Higher threshold → fewer entries
- All entries valid

---

### TC-E2E-006: Exit Logic - Time-Based

| Field | Value |
|-------|-------|
| **Priority** | P1 (High) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Generate 10 positions
2. Apply 10-minute max hold time
3. Track exit timestamps
4. Verify exit conditions

**Expected:**
- All positions exited by 10-min mark
- No positions exceed max hold time

---

### TC-E2E-007: Exit Logic - R:R-Based

| Field | Value |
|-------|-------|
| **Priority** | P1 (High) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Generate 10 positions with 2:1 R:R
2. Apply R:R-based exit
3. Track exits at TP or SL
4. Calculate P&L

**Expected:**
- Exits at 2:1 TP or SL
- No premature exits
- P&L matches expected

---

### TC-E2E-008: Exit Logic - Hybrid

| Field | Value |
|-------|-------|
| **Priority** | P1 (High) |
| **Type** | Integration |
| **Data** | Sample (500 bars) |

**Steps:**
1. Generate 10 positions
2. Apply hybrid exit (scale 50% at 1.5R, trail 50%)
3. Verify partial exits
4. Verify trailing stops

**Expected:**
- 50% scaled at 1.5R
- Remaining 50% trailed to 10-min
- Proper position sizing throughout

---

### TC-E2E-009: Performance Comparison - Sample Data

| Field | Value |
|-------|-------|
| **Priority** | P0 (Critical) |
| **Type** | E2E |
| **Data** | Sample (500 bars) |

**Steps:**
1. Backtest all 5 strategies individually
2. Backtest ensemble (equal weights)
3. Calculate metrics for all
4. Compare performance
5. Document findings

**Expected:**
- Ensemble metrics calculated
- Comparison report generated
- Ensemble ≥ average of individuals

---

### TC-E2E-010: Performance Comparison - Full Dataset

| Field | Value |
|-------|-------|
| **Priority** | P1 (High) |
| **Type** | E2E |
| **Data** | Full (116K bars) |

**Steps:**
1. Load all 28 HDF5 files
2. Backtest all strategies
3. Backtest ensemble
4. Generate comprehensive report
5. Identify best-performing configurations

**Expected:**
- All metrics calculated
- Ensemble advantages identified
- Clear recommendation for Epic 3

---

### TC-E2E-011: Edge Case - No Strategy Signals

| Field | Value |
|-------|-------|
| **Priority** | P2 (Medium) |
| **Type** | Integration |
| **Data** | Synthetic (100 bars, no signals) |

**Steps:**
1. Create data with no tradeable patterns
2. Run ensemble
3. Verify no entries generated
4. Verify no errors thrown

**Expected:**
- Ensemble runs without error
- 0 entries generated
- Graceful handling

---

### TC-E2E-012: Edge Case - All Strategies Agree

| Field | Value |
|-------|-------|
| **Priority** | P2 (Medium) |
| **Type** | Integration |
| **Data** | Synthetic (100 bars, strong signal) |

**Steps:**
1. Create data with strong, obvious pattern
2. Run ensemble
3. Verify all strategies signal
4. Check ensemble confidence = maximum

**Expected:**
- All 5 strategies signal
- Ensemble confidence ≥ 0.90
- Entry generated

---

### TC-E2E-013: Edge Case - Conflicting Signals

| Field | Value |
|-------|-------|
| **Priority** | P2 (Medium) |
| **Type** | Integration |
| **Data** | Synthetic (100 bars, mixed signals) |

**Steps:**
1. Create data with mixed long/short signals
2. Run ensemble
3. Verify netting of signals
4. Check confidence reduced appropriately

**Expected:**
- Long and short signals both present
- Ensemble nets or rejects conflicting signals
- Confidence score lower than unanimous case

---

## 5. Success Criteria

### 5.1 Must-Have (P0) Criteria

All MUST pass for Epic 2 to be considered complete:

- ✅ **SC-001:** Ensemble aggregator receives signals from all 5 strategies
- ✅ **SC-002:** Weighted confidence scores always in range [0, 1]
- ✅ **SC-003:** Entry logic only triggers when confidence ≥ threshold
- ✅ **SC-004:** All positions exited correctly (no orphaned trades)
- ✅ **SC-005:** Ensemble backtest runs successfully on 500-bar sample
- ✅ **SC-006:** Performance metrics calculated for all modes

### 5.2 Should-Have (P1) Criteria

At least 80% must pass:

- ✅ **SC-007:** Weight optimizer respects floor/ceiling constraints
- ✅ **SC-008:** All 3 exit modes work correctly
- ✅ **SC-009:** Ensemble performance ≥ average of individuals
- ✅ **SC-010:** Full dataset backtest completes without errors
- ✅ **SC-011:** No NaN/infinite values in any calculations
- ✅ **SC-012:** Memory usage stays within bounds (< 2GB)

### 5.3 Nice-to-Have (P2) Criteria

Optional but desirable:

- ⭐ **SC-013:** Ensemble outperforms best individual strategy
- ⭐ **SC-014:** Backtest completes in < 5 minutes (full dataset)
- ⭐ **SC-015:** Performance report auto-generated in markdown

---

## 6. Test Infrastructure

### 6.1 Required Components

**Test Runner:** pytest with asyncio plugin

**Test Data:**
- `tests/fixtures/synthetic_data.py` - Generate test data
- `tests/fixtures/sample_data.h5` - 500-bar sample
- `data/processed/dollar_bars/` - Full dataset (28 files)

**Test Utilities:**
- `tests/utils/ensemble_helpers.py` - Ensemble test helpers
- `tests/utils/performance_helpers.py` - Performance comparison tools
- `tests/utils/backtest_helpers.py` - Backtest utilities

**Mock Components:**
- `tests/mocks/mock_strategies.py` - Mock strategy outputs
- `tests/mocks/mock_weights.py` - Mock weight optimizer

### 6.2 Test File Structure

```
tests/integration/
├── test_ensemble_aggregation.py        # Signal aggregation tests
├── test_weighted_confidence.py         # Weighted scoring tests
├── test_weight_optimization.py         # Weight optimizer tests
├── test_entry_integration.py           # Entry logic tests
├── test_exit_integration.py            # Exit logic tests (all 3 modes)
└── test_ensemble_backtest.py           # Performance comparison tests

tests/e2e/
├── test_ensemble_e2e.py                # Full E2E suite
└── fixtures/
    ├── synthetic_data_generator.py
    ├── sample_data_loader.py
    └── test_data_validator.py
```

### 6.3 Continuous Integration

**CI Pipeline:**
1. Run unit tests (Epic 1) - must pass
2. Run integration tests (Epic 2) - must pass
3. Run E2E tests (sample data) - must pass
4. Generate coverage report - target >80%
5. Run linting (black, flake8, mypy) - must pass

**CI Time Budget:**
- Unit tests: < 2 minutes
- Integration tests: < 5 minutes
- E2E tests (sample): < 10 minutes
- Full backtest: Manual trigger (not in CI)

---

## 7. Test Execution Plan

### 7.1 Phases

**Phase 1: Integration Tests (Week 1)**
- Run TC-E2E-001 through TC-E2E-008
- Focus on component integration
- Use synthetic + sample data
- Fix any integration issues

**Phase 2: Performance Comparison (Week 1)**
- Run TC-E2E-009 on sample data
- Compare ensemble vs individuals
- Identify performance gaps
- Adjust weights/thresholds as needed

**Phase 3: Full Dataset Validation (Week 2)**
- Run TC-E2E-010 on full 116K bars
- Generate comprehensive report
- Validate ensemble edge
- Document findings

**Phase 4: Edge Cases & Regression (Week 2)**
- Run TC-E2E-011 through TC-E2E-013
- Regression testing on fixes
- Final validation before Epic 3

### 7.2 Execution Schedule

| Day | Tasks | Duration |
|-----|-------|----------|
| **Day 1** | Phase 1: TC-E2E-001 to TC-E2E-003 | 4 hours |
| **Day 2** | Phase 1: TC-E2E-004 to TC-E2E-006 | 4 hours |
| **Day 3** | Phase 1: TC-E2E-007 to TC-E2E-008 + fixes | 4 hours |
| **Day 4** | Phase 2: TC-E2E-009 + analysis | 4 hours |
| **Day 5** | Phase 3: TC-E2E-010 (may take 1-2 hours) | 4 hours |
| **Day 6** | Phase 4: TC-E2E-011 to TC-E2E-013 | 2 hours |
| **Day 7** | Regression + documentation | 4 hours |

**Total Time Budget:** 26 hours (~3.5 days focused work)

### 7.3 Entry Criteria

Before starting Epic 2 E2E tests:

- ✅ Epic 1 complete with all tests passing
- ✅ All 5 strategies implemented and tested
- ✅ Backtest engine functional (Epic 1)
- ✅ Sample data available (500 bars)
- ✅ Test infrastructure setup (pytest, fixtures)

### 7.4 Exit Criteria

Epic 2 E2E testing complete when:

- ✅ All P0 (must-have) test cases pass
- ✅ ≥80% of P1 (should-have) test cases pass
- ✅ Performance report generated
- ✅ Known issues documented
- ✅ Go/No-Go decision for Epic 3 made

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Integration issues between strategies** | High | High | Use mock outputs first, then real outputs |
| **Slow performance on full dataset** | Medium | Medium | Optimize data loading, use caching |
| **Memory issues with 116K bars** | Medium | Medium | Process in chunks, monitor memory |
| **NaN/infinite values in calculations** | Low | High | Add validation, handle edge cases |
| **Strategy initialization inconsistencies** | High | Low | Already documented in Epic 1 |

### 8.2 Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Test data loading too slow** | Medium | Medium | Pre-load data, use smaller samples |
| **Integration bugs take time to fix** | High | High | Buffer time in schedule, prioritize P0 |
| **Full backtest takes too long** | Low | Low | Run overnight, use sample for iteration |
| **Environment setup issues** | Low | Medium | Document setup, use Docker if needed |

### 8.3 Mitigation Strategies

**Performance Issues:**
- Profile code before full backtest
- Use vectorized operations (pandas/numpy)
- Cache loaded data in memory
- Parallelize independent calculations

**Integration Issues:**
- Start with mock outputs (deterministic)
- Gradually introduce real strategy outputs
- Extensive logging for debugging
- Incremental testing (1 strategy → 2 → all 5)

**Data Issues:**
- Validate test data before running
- Use multiple data samples (synthetic, real)
- Handle edge cases explicitly
- Data quality checks before tests

---

## 9. Deliverables

### 9.1 Test Artifacts

**Test Code:**
- ✅ Integration test suites (6 files)
- ✅ E2E test suite (1 file)
- ✅ Test fixtures and utilities
- ✅ Mock components for testing

**Test Data:**
- ✅ Synthetic data generator
- ✅ 500-bar sample dataset
- ✅ Test data validation scripts

**Documentation:**
- ✅ This test plan
- ✅ Test execution report
- ✅ Performance comparison report
- ✅ Known issues and recommendations

### 9.2 Reports

**Test Execution Report:**
- Test case results (pass/fail)
- Coverage metrics
- Defect summary
- Blockers and issues

**Performance Comparison Report:**
- Individual strategy metrics
- Ensemble metrics (all configurations)
- Comparative analysis
- Recommendation for Epic 3

**Final Report Structure:**
```markdown
# Epic 2 E2E Test Results

## Executive Summary
- Overall status: ✅ PASS / ❌ FAIL
- P0 cases: X/Y passed
- P1 cases: X/Y passed
- Go/No-Go for Epic 3

## Test Results by Scenario
[Detailed results per scenario]

## Performance Analysis
[Ensemble vs individual comparison]

## Issues and Blockers
[List of all issues found]

## Recommendations
[Recommendations for Epic 3]
```

---

## 10. Sign-Off

### 10.1 Approval Process

**Test Plan Approval:**
- [ ] Test plan reviewed by development team
- [ ] Test plan approved by project lead
- [ ] Test infrastructure ready
- [ ] Test data prepared

**Test Execution Approval:**
- [ ] Epic 1 complete and verified
- [ ] All prerequisites met
- [ ] Schedule confirmed
- [ ] Resources allocated

**Epic 2 Completion:**
- [ ] All P0 test cases pass
- [ ] ≥80% of P1 test cases pass
- [ ] Performance report generated
- [ ] Go/No-Go decision made
- [ ] Epic 3 readiness confirmed

### 10.2 Stakeholders

**Primary:**
- Development Team: Implementation and testing
- Project Lead: Approval and sign-off
- QA Team: Test execution and reporting

**Secondary:**
- Architecture Team: Design review
- Product Owner: Requirements validation
- Operations Team: Deployment readiness

---

## 11. Appendix

### 11.1 Glossary

- **E2E:** End-to-End testing
- **P0/P1/P2:** Priority levels (Critical/High/Medium)
- **TC:** Test Case
- **SC:** Success Criterion
- **R:R:** Risk-Reward ratio
- **TP:** Take Profit
- **SL:** Stop Loss

### 11.2 References

- Epic 1 Summary: `EPIC1_SUMMARY.md`
- Epic 1 Retrospective: `docs/epic-1-retrospective.md`
- Epic Breakdown: `_bmad-output/planning_artifacts/epics.md`
- Test Infrastructure: `tests/` directory

### 11.3 Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-04-01 | Test Team | Initial test plan creation |

---

**Document Status:** 📝 **DRAFT - PENDING REVIEW**

**Next Steps:**
1. Review test plan with team
2. Approve test strategy
3. Set up test infrastructure
4. Begin Phase 1 testing

---

*This test plan ensures Epic 2 (Ensemble Integration) is thoroughly validated before proceeding to Epic 3 (Walk-Forward Validation). By testing E2E with Epic 1 outputs, we catch integration issues early and build confidence in the ensemble system.*
