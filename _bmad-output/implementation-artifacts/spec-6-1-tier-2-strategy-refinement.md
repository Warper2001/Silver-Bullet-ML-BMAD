---
title: 'Tier 2 Strategy Refinement: HTF Alignment and Liquidity Sweeps'
type: 'feature'
created: '2026-04-28'
status: 'done'
baseline_commit: '19676c5320faa223e20421ef71ff42458b3c1186'
context: ['_bmad-output/planning-artifacts/research/technical-tier-2-fvg-strategy-refinements-research-2026-04-28.md']
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The current Tier 1 FVG system shows negative expectancy (-$442 P&L) when subjected to "Zero-Bias" realistic backtesting (limit entry, real MNQ commissions). Simple pattern matching lacks the structural context needed to filter out low-probability setups in volatile futures markets.

**Approach:** Upgrade to a Tier 2 architecture by implementing stateful multi-timeframe confluence: H4 EMA trend alignment, H1 Liquidity Sweep (stop-hunt) detection, and optimized "Mitigation Mapping" for unmitigated swing points.

## Boundaries & Constraints

**Always:**
- Apply the `shift(1)` protocol to all resampled HTF data before joining with LTF data to prevent look-ahead bias.
- Use `pd.merge_asof(direction='backward')` for robust time-series synchronization.
- Maintain the $1.80 round-trip MNQ cost ($0.80 commission + 2 ticks slippage).

**Ask First:**
- If the `smartmoneyconcepts` library performance is insufficient for large-scale grid searches, ask to switch to custom Numba-optimized logic.

**Never:**
- Allow entries on the same bar that confirms an FVG signal.
- Hardcode timeframe intervals; keep them configurable for optimization.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| HTF Alignment | H4 EMAs 9 > 21 > 50 | Long signals enabled; Short signals disabled | Return neutral if EMAs cross/tangled |
| Liquidity Sweep | High > Unmitigated SH; Close < SH | Bearish Sweep event flagged for H1/M15 level | Ignore if Close > SH (Market Structure Break) |
| Mitigation | Price > Swing High | Level marked as 'mitigated' and removed from active search | N/A |
| Look-ahead check | Join H4 data at 10:00:00 | 09:00:00-10:00:00 H4 bar data available only AFTER 10:00:01 | Raise error or assert if timestamp alignment leaks future |

</frozen-after-approval>

## Code Map

- `src/research/backtest_zero_bias_optimized.py` -- Primary backtest engine to be refactored for Tier 2 logic.
- `src/detection/fvg_detection.py` -- Reference for existing FVG logic (ensure consistency).

## Tasks & Acceptance

**Execution:**
- [x] `src/research/backtest_zero_bias_optimized.py` -- Integrate `smartmoneyconcepts` and implement H4 EMA alignment logic.
- [x] `src/research/backtest_zero_bias_optimized.py` -- Implement "Mitigation Map" state machine to track unmitigated Swing Highs/Lows.
- [x] `src/research/backtest_zero_bias_optimized.py` -- Implement Liquidity Sweep detection (pierce-and-reject) and integrate into the main signal pipeline.
- [x] `src/research/backtest_zero_bias_optimized.py` -- Run a Tier 2 parameter sweep [H4 Align + H1 Sweep + 1m FVG].

**Acceptance Criteria:**
- Given a dataset with confirmed bearish sweeps, when the backtester runs, then it must correctly identify sweeps without including breaks of structure (closes above level).
- Given resampled H4 EMA data, when joined with 1-minute bars, then no 1-minute bar at $T$ should know the H4 close at $T + 59min$.
- Given the Tier 2 confluence requirements, when the backtest completes, then the Profit Factor must be reported inclusive of all transaction costs.

## Design Notes

The **Mitigation Map** should decouple sparse swing points from the dense candle DataFrame. 
```python
# Conceptual optimization
active_levels = levels[levels.mitigated_at > current_idx]
nearest_sweep = active_levels[active_levels.price < current_high].max()
```
The implementation should prioritize the `smartmoneyconcepts` library for structural mapping unless O(N) constraints force a custom Numba implementation.

## Verification

**Commands:**
- `.venv/bin/python src/research/backtest_zero_bias_optimized.py` -- expected: Successful run with Tier 2 metrics showing improved Profit Factor vs Tier 1 baseline.

## Suggested Review Order

**Data Architecture & Synchronization**

- Orchestrates H4 alignment and H1 sweep detection with look-ahead bias guards.
  [`backtest_zero_bias_optimized.py:100`](../../src/research/backtest_zero_bias_optimized.py#L100)

- Correct UTC-to-ET conversion handling Daylight Savings to prevent session timing leaks.
  [`backtest_zero_bias_optimized.py:92`](../../src/research/backtest_zero_bias_optimized.py#L92)

**Algorithmic Refinements**

- Strict pierce-and-reject logic using forward-looking Mitigation Maps to prevent O(N²) bottlenecks.
  [`backtest_zero_bias_optimized.py:141`](../../src/research/backtest_zero_bias_optimized.py#L141)

- Fractal swing detection using local extrema to provide levels for liquidity sweeps.
  [`backtest_zero_bias_optimized.py:31`](../../src/research/backtest_zero_bias_optimized.py#L31)

**Signal Pipeline**

- Integrates H4 Trend and H1 Sweep as mandatory filters for the 1m FVG entry.
  [`backtest_zero_bias_optimized.py:245`](../../src/research/backtest_zero_bias_optimized.py#L245)
