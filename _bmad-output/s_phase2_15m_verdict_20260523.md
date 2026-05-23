# Phase 2 — 15m Holdout OOS Test Verdict — 20260523

## Pre-Registration
Sealed at git SHA: `5b581f4d88e5bf66216e23c4b66eb331ffb9b43b`
Doc: `_bmad-output/preregistration_phase2_15m.md`

## Holdout Data
- File: `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`
- Window: 2026-03-01 → 2026-05-19
- 1m bars: 75,081 | 15m bars after resample: 5,171

## Results

| | Holdout (OOS) | Training (S13 2025) |
|---|---|---|
| Trades | 6 | 61 |
| PF | 2.586 | 1.179 |
| WR | 0.667 | 0.475 |
| Daily Sharpe | 7.684 | 1.373 |
| TIME_STOP % | 0% | 11% |

## Exit Breakdown (Holdout)

| TP | SL | TIME_STOP |
|---|---|---|
| 4 | 2 | 0 |

## Pass/Fail Threshold

**Pre-committed threshold:** PF > 1.1
**Observed PF:** 2.586
**Result:** PASS ✓

## Sample Size Caveat

N=6 trades over ~2.5 months. Expected ~13 based on training rate.
Small sample — treat result as directional evidence, not high-confidence conclusion.

## Verdict

**PASS — H₁ SUPPORTED**

PF > 1.1 on sealed holdout. H₁ (15m edge is real and generalises) is supported.

**Next step (Story 7.3):** Synthesise with Epic 6 Phase 1 context. Consider unblocking Epic 2
(strategy enhancement) starting from 15m infrastructure.

## Access Log

This run was logged in `data/sealed_holdout/ACCESS_LOG.md` before results were printed.
Pre-reg SHA: `5b581f4d88e5bf66216e23c4b66eb331ffb9b43b`

_Produced by `src/research/holdout_15m_oos_test.py` (Story 7.1/7.2)._
