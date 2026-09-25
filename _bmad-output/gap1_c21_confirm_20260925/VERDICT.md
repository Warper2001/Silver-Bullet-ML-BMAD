# GAP-1-C21 verdict: INCONCLUSIVE (2026-09-25)

| Item | Value |
|---|---|
| Pre-registration | `preregistration_gap1_confirm_2021_2024.md`, sealed `9351402` |
| Script | `run_c21.py`, committed `799eb70` before running |
| Runs | One run, one look |
| Figures | `results.json` |
| Primary trade list | `trades_primary.csv` |

## Gates

| Gate | Result |
|---|---|
| **Reproduction** | PASS: unchanged `replay()` reproduced corrected Gate-0 trade for trade (N=115, PF 1.646, $8,281) |
| **Setups** | PASS: replayed dates equal the outcome-blind lists exactly (307 unseen, 278 primary) |

## Result: primary, N=278, net of $5.45/trade, 1ct

| Mean net $/trade | SD | t | p (one-sided) | Bootstrap 95% CI of the mean | Gross PF | Net PF | WR |
|---|---|---|---|---|---|---|---|
| **+$10.39** | $251.48 | 0.689 | **0.246** | [−$18.63, +$40.05] | 1.165 | 1.106 | 56.1% |

**Verdict per §5: INCONCLUSIVE.** p ≥ 0.05 and the mean is above 0.

Pre-committed actions:
- **No edge claim.** Ambiguous evidence counts as a FAIL for any claim.
- **Live GAP-1 continues under its sealed rule** at 2ct on TS SIM paper.
- **This window is spent for GAP-1.**

Month-clustered t (42 clusters): 0.762, p = 0.225, which reads the same.

## Descriptives (never a verdict)

- **The effect is much smaller than the dev estimate.**
  - Mean net was $10.39/trade, against **$66.56** in the dev window: about 16% of it.
  - Gross PF was 1.165, against 1.646.
  - At d ≈ 0.04, confirming an effect this size would take roughly **3,600 trades** (normal approximation, 80% power).
  - The CI does include values up to about 0.6× the dev edge, so this is not a refutation.
- **Profits are concentrated.** The top 3 trades are 54% of gross P&L (72% on all 307).
- **By year:**

  | Year | N | Mean net | Gross PF |
  |---|---|---|---|
  | 2021 | 67 | −$19.86 | 0.87 |
  | 2022 | 101 | +$13.55 | 1.18 |
  | 2023 | 55 | +$31.89 | 1.61 |
  | 2024 | 55 | +$19.94 | 1.28 |

  There is no monotone decay pattern.
- **By side:**

  | Side | N | Net | Gross PF |
  |---|---|---|---|
  | Short | 147 | +$3,315 | 1.34 |
  | Long | 131 | −$426 | 1.02 |

  This is the **reverse** of the dev window, where long was the stronger half (PF 2.25 against 1.44). It is a descriptive split of the test data. **Choosing a side from it would be the forbidden pattern**: a filter chosen on favourable past data, with no unseen data left to retest it.
- **Exits:** 168 time-stops, 98 target fills, 12 stops. The engine's target-first rule never decided a trade: no exit bar touched both target and stop.
- **Sensitivity (all 307, including roll weeks):** mean net +$5.97, p = 0.340, gross PF 1.116. Same reading.

## What this means

- GAP-1's rules were not confirmed on four years of data they were never fitted to.
- The point estimate is positive but small, and the dev window's edge looks largely specific to that window.
- It does not show the edge is absent: the CI's upper end is +$40/trade.
- The live N=60 re-evaluation and N≈88 target still stand. At an effect near +$10/trade, neither can resolve it.
