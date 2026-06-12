# Pre-Commitment Addendum 2: 5-Minute Timeframe Extension (Gate 0, one shot)

**Date:** 2026-06-12
**Parent:** `precommit_pair_divergence_survey_2026-06.md` (`b54fb08`) + BTC–ETH
addendum (`9f4ae9a`); parent results: all 7 pairs FAIL at 1-min (`db3abd0`, `ce47f14`).
**Status:** committed BEFORE any 5m simulation runs. Frozen config:
`pair_survey_5m_config.yaml` (same commit).

## Hypothesis (mechanism, stated before results)

The 1-min survey produced two specific failure modes, both of which 5m bars
directly address:

1. **Cost fraction** (SI–GC, PL–GC long): real gross edge (PF 1.05, ~+$2/trade)
   destroyed by fixed costs. 5m residuals are ~√5 ≈ 2.24× larger in dollars, so
   the same reversion structure carries ~2× the per-trade edge against the SAME
   fixed costs. At the new $80 primary, SIL breakeven WR falls 54.7% → 52.3%
   and the slippage-stressed bar 67.2% → 58.6%.
2. **Frequency** (RTY–ES, BTC–ETH): gross-positive but divergences too rare at
   the 1-min-scaled threshold. Larger 5m swings cross dollar thresholds more
   often.

**Counter-hypothesis (also stated up front):** signal quality may dilute — a
$40 5-bar divergence is a sharper anomaly at 1-min than $80 is at 5m — and the
S26 precedent (1-min edge, 15m Gate 0 FAIL) shows structure does not
automatically survive rescaling. Win rate at 5m is the open question the test
answers.

## Multiplicity disclosure

This is the **second look at the same 7 pairs** and the **only timeframe
extension** authorized under this survey family. If 5m fails, the
divergence-fade family is closed across timeframes for these pairs; any further
work requires a new hypothesis class with fresh pre-registration. No 15m/1h
follow-up sweep is authorized by this document.

## Frozen specification (delta from parent only)

- **Bars:** 5-minute, resampled from the same 1-min CSVs (right-labeled,
  right-closed; identical convention both legs — no relative lookahead).
- **Grid:** THRESH_USD ∈ {40, 60, 80, 100} × STOP_MULT ∈ {1.0, 2.0};
  **primary $80 / 1.0×** (√5-scaling of the validated 1-min $40 primary,
  rounded down). Stop cap stays $150 — 2.0× cells above $75 self-cap toward
  zero trades by construction; this is accepted, not a bug.
- **Bar-count parameters unchanged in bar units:** beta 60, spread 5, hold 30
  (= 5 h / 25 min / 150 min at 5m).
- Everything else identical to parent: RTH 09:30–15:55 ET, dev window
  2025-05-01 → 2026-02-28 (hard-coded), both directions, roll-date skip,
  per-pair breakeven WR gates, five Gate 0 criteria, slippage stress, decision
  rule, PL–GC informational-only, BTC–ETH venue-proxy + prospective-Gate-2
  caveats carried forward.
- **MNQ–ES is a candidate at 5m, not a control** (no validated 5m template
  exists). Harness regression vs the 1-min control is re-verified in the same
  execution by re-running `--pair MNQ_ES` on the default 1-min config and
  requiring bit-identical output (N=633, WR 58.0%, PF 1.27).

## Decision rule

Identical to parent. A 5m pass on any non-informational pair-direction
authorizes only the writing of a Gate 1 pre-registration (which must address
5m-specific slippage evidence). Fail → family closed across timeframes.
