# Story 7.1: Phase 2 Pre-Registration + 15m BacktestEngine Adapter

Status: review

## Story

As Alex (researcher),
I want to write and commit the Phase 2 pre-registration document and implement the 15m sealed holdout OOS test script,
so that Story 7.2 can run the definitive 15m OOS test against the sealed holdout with a clean audit trail.

## Acceptance Criteria

1. Pre-registration doc `_bmad-output/preregistration_phase2_15m.md` written **and committed to git** before any holdout data is read. Doc must include: hypothesis, exact test procedure, pass/fail threshold (PF > 1.1), StrategyConfig defaults, prior holdout access disclosure.
2. `src/research/holdout_15m_oos_test.py` implemented — resamples `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv` to 15m, runs `BacktestEngine`, computes PF/WR/Sharpe/trade-count, logs holdout access.
3. Script produces verdict report `_bmad-output/s_phase2_15m_verdict_<date>.md` when run.
4. No modifications to `strategy_core.py`, `backtest_engine.py`, or `tier2_streaming_working.py`.

## Tasks / Subtasks

- [x] Task 1 — Write and commit pre-registration doc (AC #1)
  - [x] Create `_bmad-output/preregistration_phase2_15m.md`
  - [x] Include hypothesis, StrategyConfig defaults, pass/fail threshold, prior access disclosure
  - [x] `git add -f _bmad-output/preregistration_phase2_15m.md && git commit` — record SHA
  - [x] Write the SHA into the pre-reg doc

- [x] Task 2 — Implement holdout_15m_oos_test.py (AC #2, #3)
  - [x] Load `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
  - [x] Resample to 15m (same resample_bars() from timeframe_replication.py)
  - [x] Write UTC temp CSV, run BacktestEngine(tmp, config=StrategyConfig())
  - [x] Compute PF/WR/Sharpe/trade-count
  - [x] Append access log entry to `data/sealed_holdout/ACCESS_LOG.md`
  - [x] Write verdict report

- [x] Task 3 — Verify no strategy code modified (AC #4)
  - [x] Confirm only `holdout_15m_oos_test.py` created in `src/research/`

## Dev Notes

### Prior Holdout Access Disclosure

The sealed holdout has been accessed before (see `data/sealed_holdout/ACCESS_LOG.md`). Relevant prior accesses:

| Date | SHA | Purpose | 15m Result |
|---|---|---|---|
| 2026-05-20 | `910e95c` | Old S13 (original Phase 1 pre-reg) | 15m PF=1.8157, **14 trades** (insufficient sample) |

The prior 15m test used a different pre-registration, different script, and produced only 14 trades — statistically insufficient. This Story 7.1 test is the definitive pre-registered 15m holdout validation with a dedicated script and full audit trail.

### StrategyConfig Defaults (frozen, no adjustment for 15m)

| Parameter | Value |
|---|---|
| `sl_multiplier` | 5.0 |
| `tp_multiplier` | 6.0 |
| `atr_threshold` | 0.5 |
| `max_gap_dollars` | 60.0 |
| `max_hold_bars` | 60 |
| `max_pending_bars` | 240 |
| `contracts_per_trade` | 5 |
| `max_daily_loss` | -750.0 |
| `vol_regime_lookback` | 120 |
| `vol_regime_threshold` | 0.75 |
| `min_gap_atr_ratio` | 0.25 |
| `bearish_only` | True |
| `h1_sweep_lookback` | 6 |
| `commission_per_roundtrip` | 4.0 |

**No adjustments for 15m resolution.** If the pattern is real, it should survive at default settings.

### Behavioral Deltas at 15m (same as S13)

| Behavior | 1m baseline | 15m |
|---|---|---|
| FVG span | 3 min | 45 min |
| `max_hold_bars=60` | 60 min | 900 min |
| `max_pending_bars=240` | 4 hr | 60 hr |
| H1 resample | 60 bars/H1 | 4 bars/H1 |
| Bar ATR | true 1-min ATR | 15-min ATR (larger) |

### Expected Sample Size

Training window (2025, full year): 61 trades at 15m.
Holdout (2026-03-01 to 2026-05-19, ~2.5 months ≈ 21% of year): expected ~13 trades.
This is a small sample — document this limitation in the verdict report.

### Pass/Fail Threshold

**PF > 1.1** on sealed holdout at 15m. Pre-committed in Epic 7 outline (`phase1_verdict_20260523.md`).

### Script Structure

Reuse `resample_bars()` and UTC temp-file pattern from `timeframe_replication.py`:

```python
from src.research.timeframe_replication import resample_bars, load_1min_bars
# Or inline the logic — either is fine, just don't import from timeframe_replication
# if that creates circular issues
```

Must append to `data/sealed_holdout/ACCESS_LOG.md` before printing any results:

```python
log_entry = (
    f"| {date_str} | {PRE_REG_SHA} | holdout_15m_oos_test.py | "
    f"Phase 2 definitive 15m OOS test | <filled after run> |\n"
)
```

### References

- Phase 1 verdict: `_bmad-output/phase1_verdict_20260523.md`
- Epic 7 outline: `_bmad-output/phase1_verdict_20260523.md` § Epic 7 Outline
- Access log: `data/sealed_holdout/ACCESS_LOG.md`
- Holdout data: `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
- Training-window 15m result (S13): 61 trades, PF=1.179, Sharpe=1.373

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23)

### Debug Log References

- Pre-registration commit: `5b581f4d88e5bf66216e23c4b66eb331ffb9b43b` (before holdout data read)

### Completion Notes List

1. Pre-registration sealed at `5b581f4d` before any holdout data was accessed. Prior 15m access (old S13, 14 trades) disclosed in pre-reg doc.
2. `holdout_15m_oos_test.py` reuses same UTC-temp-file pattern from `timeframe_replication.py`. Appends to `data/sealed_holdout/ACCESS_LOG.md` before printing any results.
3. Pass/fail threshold pre-committed: PF > 1.1.
4. No modifications to `strategy_core.py`, `backtest_engine.py`, or `tier2_streaming_working.py`. Confirmed via `git status --short src/research/`.

### File List

- `_bmad-output/preregistration_phase2_15m.md` — NEW (committed at `5b581f4d`)
- `src/research/holdout_15m_oos_test.py` — NEW
- *(Story 7.2 produces `_bmad-output/s_phase2_15m_verdict_<date>.md` and ACCESS_LOG entry)*
