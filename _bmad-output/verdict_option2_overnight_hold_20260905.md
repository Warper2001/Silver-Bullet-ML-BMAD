# Verdict — Option 2: MNQ overnight/Globex hold

**Date:** 2026-09-05
**Pre-registration:** `_bmad-output/preregistration_option2_overnight_hold.md`
**Script:** `tools/option2_overnight_hold_backtest.py`
**Data:** dev window only (< 2026-03-01), 230 qualifying nights (301 total sessions with a valid prior close, minus post-cutoff).

## Result

| metric | value |
|---|---|
| N nights | 230 |
| Win rate | 54.3% |
| Net PF (after $4.00/night RT cost) | 1.154 |
| Total net P&L | +$4,698 |
| Ex-top-5-nights PF | **0.967** |
| Friday nights | N=43, PF=0.881, −$710 |
| Non-Friday nights | N=187, PF=1.220, +$5,408 |
| Random-direction null (200 draws) | median 0.956, **p95 1.323** |

## Gate 0

| check | result |
|---|---|
| N floor | PASS (230) |
| net PF > 1.15 | PASS (1.154, barely) |
| net PF > null p95 | **FAIL** (1.154 < 1.323) |
| ex-top5 PF > 1.0 | **FAIL** (0.967) |

**VERDICT: FAIL.**

## Reading it

The always-long overnight hold clears the weak-edge floor by a hair and beats the null's *median* (0.956 — a random direction call loses money after costs on this sample, so the always-long side is doing something), but it does not clear the null's 95th percentile: on this sample, a purely random per-night coin flip would beat 1.154 about 5% of the time. That's not a strong enough separation to call it a real, exploitable drift rather than a lucky sample.

The more damning number is **ex-top-5-nights PF = 0.967** — pull out five nights out of 230 and the whole thing is a loser. That's the same fat-tail-dependency shape MIM-NB has, but MIM-NB earned trust for that shape through a real validation chain; this is a first look that shows the same fragility with none of the supporting evidence.

**One real, unplanned observation, reported but not acted on:** Fridays (weekend holds) are the losing half (PF 0.881) and non-Fridays alone would have cleared PF 1.22. Per the stopping rule, this is *not* grounds to re-run excluding Fridays on this same seal — that would be exactly the same post-hoc subset-selection pattern this project has rejected before. If it's worth testing, it needs its own pre-registration and, ideally, a reason to expect it beyond "it happened to help in this sample" (GAP-1 already excludes Fridays for its own stated reasons — whether that reasoning transfers to an overnight hold is a real, separate question, not an obvious yes).

## Disposition

**Option 2 (MNQ overnight/Globex hold) closed, FAIL**, on the primary always-long test. The day-type filter variant (scheduled high-vol days only) that the pre-registration reserved as a follow-up is **not run** — the primary's own fat-tail dependency and null-beat failure make it unlikely a filtered variant would fare better without addressing the same fragility, and per the stopping rule, this seal doesn't authorize chasing it further. If overnight holding is revisited, it should be as a fresh pre-registration incorporating the Friday-split observation as a stated hypothesis, not a re-test of this one.
