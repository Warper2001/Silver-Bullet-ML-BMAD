# S-BIDIR-15m Verdict — 20260523

## Pre-Registration
Sealed at git SHA: `9a94d2e7e75f073a717337a198610c81641d060a`
Doc: `_bmad-output/preregistration_s_bidir_15m.md`

## Results

| Direction | Trades | PF | WR | Daily Sharpe |
|---|---|---|---|---|
| TOTAL (bidir) | 81 | 0.985 | 0.457 | -0.126 |
| BEARISH only | 51 | 1.106 | 0.471 | 0.810 |
| BULLISH only | 30 | 0.826 | 0.433 | -1.602 |
| BASELINE (bearish-only S13) | 61 | 1.179 | 0.475 | 1.373 |

## Exit Breakdown

| | TP | SL | TIME_STOP |
|---|---|---|---|
| BEARISH | 24 | 20 | 7 |
| BULLISH | 10 | 14 | 6 |

## Consistency Criterion

- Count ≥ 1.5× baseline (≥ 92): ✗ (81 trades, 1.33×)
- Total PF > 1.0: ✗ (0.985)
- Bearish PF > 1.0: ✓ (1.106)
- Bullish PF > 1.0: ✗ (0.826)

## Verdict

**H₀ SUPPORTED — fails: count 81 < 92; PF 0.985 ≤ 1.0; bullish PF 0.826 ≤ 1.0**
