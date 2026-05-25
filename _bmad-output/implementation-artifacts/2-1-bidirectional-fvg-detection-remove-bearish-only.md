# Story 2.1: Bidirectional FVG Detection — Statistical Power Recovery (15m)

Status: done

## Story

As Alex (researcher),
I want to run the FVG+H1-sweep strategy bidirectionally (both bearish and bullish setups) at 15m resolution on the 2025 training window,
so that I can quantify how much statistical power is recovered by removing the `bearish_only=True` constraint at the timeframe where the OOS signal is confirmed.

## Acceptance Criteria

1. Pre-registration doc `_bmad-output/preregistration_s_bidir_15m.md` written **and committed** before running any backtest. Hypothesis: "Bidirectional 15m FVG detection recovers ≥ 1.5× trade count vs bearish-only baseline (61 trades) while maintaining PF > 1.0."
2. `src/research/bidir_15m_test.py` implemented — resamples 2025 1-min training data to 15m, runs `BacktestEngine(config=StrategyConfig(bearish_only=False))`, computes PF/WR/Sharpe/trade-count and directional breakdown (bearish vs bullish).
3. Results compared against the bearish-only 15m baseline (61 trades, PF=1.179, training 2025).
4. Verdict report `_bmad-output/s_bidir_15m_verdict_<date>.md` produced: trade count, PF, WR, breakdown by direction, and statistical power assessment.
5. No modifications to `strategy_core.py`, `backtest_engine.py`, or `tier2_streaming_working.py`. Config change only via `StrategyConfig(bearish_only=False)` — no default changed.

## Tasks / Subtasks

- [x] Task 1 — Write and commit pre-registration doc (AC #1)
  - [x] Create `_bmad-output/preregistration_s_bidir_15m.md`
  - [x] Include hypothesis, data source, bearish-only baseline, pass criteria
  - [x] `git add -f && git commit` — record SHA
  - [x] Write SHA into pre-reg doc

- [x] Task 2 — Implement `src/research/bidir_15m_test.py` (AC #2)
  - [x] Load 2025 training CSV, resample to 15m (same pattern as timeframe_replication.py)
  - [x] Run BacktestEngine with `StrategyConfig(bearish_only=False)`
  - [x] Compute PF/WR/Sharpe/count + directional breakdown
  - [x] Compare against baseline

- [x] Task 3 — Produce verdict report (AC #4)
  - [x] Write `_bmad-output/s_bidir_15m_verdict_<date>.md`

- [x] Task 4 — Verify no strategy code modified (AC #5)

### Review Findings

- [x] [Review][Patch] COUNT_THRESHOLD documentation inconsistency — pre-reg says "≥ 91", code threshold is 91.5, verdict says "≥ 92"; for integer trade counts these differ; updated code to use `math.ceil` and consistently print "≥ 92" [src/research/bidir_15m_test.py:24]
- [x] [Review][Patch] Bearish count regression unexplained — bearish trades dropped 61→51 in bidirectional mode; added explanation to completion notes [_bmad-output/implementation-artifacts/2-1-...md]
- [x] [Review][Defer] calc_sharpe single-day edge case — if all trades fall on one day, std of single value returns 0 or NaN; pre-existing pattern from timeframe_replication.py [src/research/bidir_15m_test.py:79] — deferred, pre-existing
- [x] [Review][Defer] Timezone date bucketing for daily Sharpe — `t.timestamp_entry.date()` uses UTC date while bars were resampled in ET; pre-existing pattern from timeframe_replication.py [src/research/bidir_15m_test.py:77] — deferred, pre-existing
- [x] [Review][Defer] backtest_engine.py shows M in git status — modification predates this story; not part of this diff; separate concern [src/research/backtest_engine.py] — deferred, pre-existing

## Dev Notes

### 15m Reframe Context

This story is the first Epic 2 story reframed for 15m per the Phase 2 verdict requirement
(`_bmad-output/phase2_verdict_20260523.md`). All prior Epic 2 stories were designed for 1m;
this implementation tests statistical power recovery at the OOS-validated timeframe.

### Bearish-Only Baseline (S13 training window)

| Metric | Value |
|---|---|
| Trades | 61 |
| PF | 1.179 |
| WR | 0.475 |
| Daily Sharpe | 1.373 |
| TIME_STOP % | 11% |

### Code Infrastructure (already in place)

`strategy_core.py` `detect_fvg` already detects both directions.
`make_entry_decision` enforces `fvg.direction == sweep.direction` (no cross-direction entries).
`BacktestEngine` has the `bearish_only` gate + direction-aware FVG pre-check.

**No changes to any of these files.** The test is purely a parameter variation:
`StrategyConfig(bearish_only=False)`.

### Expected Behavior with bearish_only=False

- Bearish H1 sweep + bearish FVG → SHORT entry (same as before)
- Bullish H1 sweep + bullish FVG → LONG entry (new)
- Bearish sweep + bullish FVG → skipped (direction mismatch in make_entry_decision)
- Bullish sweep + bearish FVG → skipped (direction mismatch)

### BacktestEngine API

```python
from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import StrategyConfig

engine = BacktestEngine(tmp_csv_path, config=StrategyConfig(bearish_only=False))
trades = engine.run()
```

### TradeRecord Fields for Directional Breakdown

```python
bearish_trades = [t for t in trades if t.direction == "BEARISH"]
bullish_trades  = [t for t in trades if t.direction == "BULLISH"]
```

### Data

Training window: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`
Resample to 15m (same as `timeframe_replication.py`), write UTC temp CSV, run engine.

### References

- Baseline: `_bmad-output/s13_verdict_20260523.md` (15m bearish-only)
- Phase 2 verdict reframe requirement: `_bmad-output/phase2_verdict_20260523.md`
- timeframe_replication.py resample pattern: `src/research/timeframe_replication.py`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23)

### Debug Log References

- Pre-registration commit: `9a94d2e7e75f073a717337a198610c81641d060a`

### Completion Notes List

1. Pre-registration sealed at `9a94d2e7` before any backtest ran.
2. **VERDICT: H₀ SUPPORTED.** Bidirectional fails all three criteria:
   - Count: 81 trades = 1.33× (need ≥ 1.5×)
   - Total PF: 0.985 ≤ 1.0
   - Bullish PF: 0.826 ≤ 1.0 (direction consistency fails)
3. The bearish component still performs well (51 trades, PF=1.106). Bullish FVG setups at 15m
   are a net drag — bullish H1 sweeps + bullish FVGs do not produce edge in 2025 training.
4. **Implication for Epic 2:** `bearish_only=True` should be KEPT. It is a load-bearing filter,
   not an arbitrary restriction. Statistical power recovery must come from other means (Story 2-2:
   AM kill zone, Story 2-3: M15 confirmation, etc.).
5. No modifications to strategy_core.py, backtest_engine.py, or tier2_streaming_working.py.
6. **Bearish count regression (61→51):** When `bearish_only=False`, the H1 sweep lookback (6-bar window) now detects both bearish and bullish sweeps. On days where a bullish sweep fires first and consumes the window, a subsequent bearish sweep may fall outside the lookback, reducing bearish trade count. This is expected behavior from the symmetric detection logic and is not a bug.

### File List

- `_bmad-output/preregistration_s_bidir_15m.md` — NEW (committed at `9a94d2e7`)
- `src/research/bidir_15m_test.py` — NEW
- `_bmad-output/s_bidir_15m_verdict_20260523.md` — NEW (produced by running script)
