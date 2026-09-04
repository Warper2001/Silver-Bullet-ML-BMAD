# Verdict — Option 1: GAP-1 time-stop horizon sweep

**Date:** 2026-09-04
**Pre-registration:** `_bmad-output/preregistration_option1_gap_fade_horizon.md` (sealed commit `1d28de9`, before this run)
**Script:** `tools/option1_gap_fade_horizon_sweep.py`
**Data:** `data/processed/dollar_bars/1_minute/mnq_1min_{2025,2026_ytd}.csv`, 301 valid sessions, N=117 qualifying gap trades (identical across every grid cell by construction).

## Result

| hour | N | PF | PF ex-top3 | gross P&L |
|---|---|---|---|---|
| 13 (baseline) | 117 | 1.761 | 1.515 | $9,878 |
| 14 | 117 | 1.760 | 1.508 | $10,400 |
| **15 (best)** | 117 | **1.829** | 1.588 | $11,883 |
| 16 (= EOD) | 117 | 1.790 | 1.561 | $11,916 |

Random-hour null (200 draws, seed 20260904): **median PF 1.783, p95 1.949.**

## Gate 0

| check | result |
|---|---|
| best PF (1.829 @ 15h) > baseline PF (1.761) | PASS |
| best PF (1.829) > null p95 (1.949) | **FAIL** |
| N ≥ 60 | PASS (117) |
| ex-top3 PF at best (1.588) > ex-top3 PF at baseline (1.515) | PASS |
| monotonic across grid | FAIL (13→14 flat, 14→15 up, 15→16 down — a lone interior spike) |

**VERDICT: FAIL.**

## Reading it

This is the null test doing exactly its job. Taken alone, "15h beats baseline, and ex-top-3 also improves" would have read as a promising, seemingly-robust result — precisely the shape that got a pass in isolation before (the order-flow ballpark's 60-second window, TSMOM-1's raw signal before its own look-ahead check). Here the null catches it: a *randomly* assigned exit hour, drawn per trade with no relationship to anything, produces a PF distribution (median 1.783) centered **above** every deliberately-chosen grid cell except the best one, and its p95 (1.949) sits above even that best cell. In plain terms — on this sample, GAP-1's PF is close to indifferent to which hour between 13:00 and 16:00 you exit at; the 13→14→15→16 differences are within the noise band of "exit at some arbitrary hour," not a real reversion-completion effect. The lone-spike-at-15h pattern (not 16h, the natural endpoint) is the visible symptom of that.

**This does not indict the broader "give it more room" hypothesis** — it specifically rules out same-session RTH exit-hour extension (9:30–16:00) as a lever for GAP-1. It says nothing about overnight/Globex extension (a genuinely different mechanism — Option 2, still open) or about YANK, which uses a different exit structure entirely and was not tested in this seal (deferred, one knob at a time, per the pre-registration).

## Disposition

- **Option 1a (GAP-1 RTH time-stop extension): closed, FAIL.** Per the pre-registration's stopping rule, no re-sweep on this seal.
- **Option 1b (YANK exit-horizon test): not yet run** — genuinely open, needs its own pre-registration (different strategy, different exit mechanism; testing it under this same seal would violate one-knob-at-a-time).
- Recommend folding Option 1b into the queue rather than treating "Option 1" as fully closed — only the GAP-1 half was tested.
