# YANK Compressed-Cascade Phase 1 Verdict — 20260819

## Pre-Registration
Sealed at git SHA: `445a9ba` (`_bmad-output/preregistration_yank_compressed_cascade.md`,
config corrected in Amendment 1, same commit range as this verdict — see PR #50).

## Candidate
Sweep leg resampled to **15min** (was H1), CHoCH leg resampled to **5min** (was M15), FVG
entry unchanged at M1 (data floor — no tick data exists in this repo). All other
`StrategyConfig` fields at the Amendment-1 baseline (`bearish_only=True,
m15_confirmation=True, h1_sweep_lookback=6, tuesday_exclusion=True, min_gap_atr_ratio=0.25,
max_gap_atr_ratio=0.426, ml_threshold=0.0` — clean Program-C baseline, not a live-YANK
config replica; see Amendment 1 for why).

## Simulation Parameters
- N = 100 random-null simulations, seeds 0-99, `np.random.default_rng(seed)`
- Data: `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` (training window only;
  sealed holdout not accessed)
- Entry gates shared with the real strategy: M15 sweep active (within last 6 M15 bars), M5
  CHoCH confirmed, vol regime filter passes, daily circuit-breaker not tripped, not Tuesday
- Random entry: coin flip calibrated to the candidate's own measured rate
  (`p_enter = 199 / 33208 = 0.60%`) — **not** S12's original 2025 rate, per §3 of the seal
- Exit logic: `strategy_core.check_exit()` unchanged (SL -> TP -> TIME_STOP)
- Implementation: `yank_compressed_cascade_phase1.py` (repo root). New, isolated
  `resample_to_timeframe()`; zero edits to `resample_to_h1` / `resample_to_m15` /
  `backtest_engine.py`, per the seal's engineering constraint

## Null Distribution (N=100 random simulations, all 100 produced >=1 trade)

| Metric | Value |
|---|---|
| Median PF | 0.912 |
| P90 PF | 1.218 |

## Candidate Result
- PF: **1.397** (199 trades, 33,208 candidate bars over 2025)
- Percentile rank: **100%** — exceeds all 100 random-null simulations

## Decision Rule (verbatim from the seal, §3)

| Condition | Verdict |
|---|---|
| Candidate PF < median of null PFs | PIVOT |
| Candidate PF > 90th percentile of null PFs | PROCEED to Phase 2 |
| Candidate PF in 50th-90th percentile | AMBIGUOUS = TREATED AS FAIL = PIVOT |

## Verdict

**PROCEED to Phase 2.**

Candidate PF=1.397 clears the p90 null bar (1.218) with room, not marginally — unlike S12's
own AMBIGUOUS result at the 70th percentile. This is evidence the compressed cascade is not
simply firing more often and mistaking event count for edge; it is evidence worth spending
the Phase 2 clock on.

## What this does NOT establish

- Not a live-fidelity read: ML filter, kill-zone, and current daily-breaker sizing are not
  exercised (Amendment 1). This is a structure/timeframe read in isolation.
- Not yet OOS-validated. Per the seal, Phase 2 requires a **fresh prospective window**
  starting from the seal's commit date forward — the 2026-03-01/05-19 holdout is not reused.
  Phase 2 has **not started**; it cannot be fast-forwarded, and N=100 random sims on 2025
  data says nothing about live 2026-08+ behavior.
- Phase 1's cheapness (this run) was always expected; Phase 2's cost (N>=30 trades,
  prospective, likely 8-12+ weeks) is the real bill, disclosed in the seal §6, and is now due.

---
_Produced by `yank_compressed_cascade_phase1.py`._
_Sealed pre-registration: `_bmad-output/preregistration_yank_compressed_cascade.md`_
