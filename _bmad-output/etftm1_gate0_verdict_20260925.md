# ETFTM-1 Gate 0 verdict: FAIL, final (2026-09-25)

**Chain:**

| Step | Record |
|---|---|
| A0 data | GO (a158b70) |
| A1 power gate | MARGINALLY_POWERED (bdc9f07), MDE timing IR 0.55 |
| Alex's decision | "Go, Gate 0 first" |
| A2 seal | c3365a9 |
| Runner | d99809e, committed before running |
| Results | `etftm1_gate0_results.json` |

One run, on development data only (1994-01 to 2021-08, 332 months, median 26 ETFs a month). `data/sealed_holdout/` was never read.

| Gate 0 condition | Result | Needed | Pass? |
|---|---|---|---|
| (i) Significance | z = 0.576, one-sided p = 0.282 | z ≥ 1.645 | ✗ |
| (ii) Persistence-preserving random null | m̂ at the **90.5th** percentile (null p95 = 0.0255, m̂ = 0.0191) | ≥ 95th; 50–95th = ambiguous = FAIL | ✗ |
| (iii) Leave one asset class out | 8 of 8 positive | ≥ 7 of 8 | ✓ |

**Verdict: FAIL, final.** A3 (engine), the deployment gate, Gate 1 (holdout) and A5 (paper) are **not built or run**. The ETF holdout stays sealed and unused.

## What it means

- **Estimated timing effect:** IR ≈ **0.15** a year, small and positive. It's under A1's 80%-power detection threshold of 0.55, so this is **"no timing edge of IR ≳ 0.55"**, not proof of zero. That's consistent with Huang, Li, Wang & Zhou (2020): time-series momentum is weak once each asset's own mean is removed.
- **Decay (descriptive only):**

  | Period | Mean monthly timing return |
  |---|---|
  | Before 2012 (216 months) | 0.027 |
  | 2012 on (116 months) | **0.005** |
  | Excluding 2008 | **0.006** |

  Most of the effect is the 2008 crisis year, matching published post-publication decay.
- **Breadth:** the 31 ETFs behave like about **4 independent bets** (A1), far below the 58 futures in MOP's universe. A US-listed ETF universe can't add much breadth.

## Conditions for revisiting (a new pre-registration is required)

1. A materially broader and less correlated universe, with effective breadth well above 4. For example, futures across many markets, which needs larger capital (see the 09-20 micro-futures NO-GO), or international single-country ETFs, whose history is shorter.
2. A different claim: a *diversified risk-premia allocation* (always-long, inverse-volatility), which the decision memo marked as a separate pre-registration and **not an edge**.
3. No re-sweep of lookback, volatility window, universe subset or filters on this development data. Those were sealed out.

## Descriptive, by asset class (mean timing return, never gating)

| Asset class | Mean |
|---|---|
| Intl equity | 0.085 |
| Commodities | 0.050 |
| Treasuries | 0.046 |
| Sectors | 0.035 |
| Currency | 0.030 |
| US equity | 0.014 |
| Real estate | −0.001 |
| Credit / inflation | −0.008 |

Picking the positive classes from this table would be forbidden: it is a filter chosen on past data.
