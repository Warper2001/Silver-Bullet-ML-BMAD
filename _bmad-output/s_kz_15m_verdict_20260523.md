# S-KZ-15m Verdict — 20260523

## Pre-Registration
Sealed at git SHA: `df66bd9`
Doc: `_bmad-output/preregistration_s_kz_15m.md`

## Results

| Run | N | PF | WR | Daily Sharpe |
|---|---|---|---|---|
| KZ-filtered (A) | 5 | 0.826 | 0.400 | -1.441 |
| Full window (B) | 61 | 1.179 | 0.475 | 1.373 |
| S13 baseline | 61 | 1.179 | 0.475 | 1.373 |

## Exit Breakdown

| | TP | SL | TIME_STOP |
|---|---|---|---|
| KZ-filtered | 2 | 3 | 0 |
| Full window | 29 | 25 | 7 |

## DST Verification

PASS — all 5 KZ-filtered entries in [09:30, 11:00) ET

## Pass Criteria

- PF > 1.3: ✗ (0.826)
- N ≥ 15: ✗ (5)
- DST verification: ✓

## Verdict

**H₀ SUPPORTED — fails: PF 0.826 ≤ 1.3; N 5 < 15**
