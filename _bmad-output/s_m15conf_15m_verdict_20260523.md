# S-M15CONF-15m Verdict — 20260523

## Pre-Registration
Sealed at git SHA: `cfe7cb3`
Doc: `_bmad-output/preregistration_s_m15conf_15m.md`

## Results

| Run | N | PF | WR | Daily Sharpe |
|---|---|---|---|---|
| M15-confirmed (A) | 61 | 1.179 | 0.475 | 1.373 |
| Full window (B) | 61 | 1.179 | 0.475 | 1.373 |
| S13 baseline | 61 | 1.179 | 0.475 | 1.373 |

## Exit Breakdown

| | TP | SL | TIME_STOP |
|---|---|---|---|
| M15-confirmed | 29 | 25 | 7 |
| Full window | 29 | 25 | 7 |

## M15 Confirmation Verification

PASS — all 61 M15-confirmed trades have `m15_confirmed=True`

## Pass Criteria

- PF > 1.3: ✗ (1.179)
- N ≥ 15: ✓ (61)
- M15 confirmation verification: ✓

## Verdict

**H₀ SUPPORTED — fails: PF 1.179 ≤ 1.3**
