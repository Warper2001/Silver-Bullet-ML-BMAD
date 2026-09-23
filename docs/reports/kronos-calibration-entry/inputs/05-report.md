# Kronos readiness

PARK_PENDING_EVIDENCE

Strategy testing permitted: false. Trading authorized: false.

Documentary planning only. No collection, forecasting, scoring or trading admitted; no automated evidence-sufficiency assessment is implemented.

**This run: [3/6/12-month horizon comparison](comparison.md), with full accounting in [scenarios.json](scenarios.json).** The inherited sessions/SE-inflation table below is GENERIC readiness planning only; it is not this run's horizon analysis.

Actual power: UNASSESSABLE. Admitted sessions: 0.

- **economics — unresolved:** Independent useful K and K-M dollar effects, dated before sample-size selection; own-strategy variances, covariance and dependence evidence, not another strategy's estimates.
- **calendar — unresolved:** Reviewed CME MNQ RTH calendar with UTC/DST boundaries, holidays, early closes and coverage over the entire selected horizon; proxy weekdays are insufficient.
- **operations — unresolved:** Validated endpoint entitlements, completion/arrival/revision semantics, quote coverage, request budget, clock-error bounds, latency and outage/recovery coverage for proposed cadence.
- **costs — unresolved:** Independently supported commissions, fees, slippage and latency; freeze before evaluation without consulting concealed outcomes.
- **population — unresolved:** Separate preregistered calibration and untouched evaluation populations; power gate before performance calibration and another before evaluation; contamination/exposure audit.
- **collection — unresolved:** Separate collection and shadow-forecast admission, isolated auth, operational review and preregistration. This command does not implement admission assessment.
- **contract — unresolved:** Causal volume-derived front-contract decisions, same-contract history and resets; independently justify warmup and roll deductions.

Conditional normal planning only; per-test alpha .025, marginal power .90, joint lower bound .80.

| Sessions | SE inflation | Detectable standardized effect |
| ---: | ---: | ---: |
| 20 | 1.0 | 0.7248 |
| 20 | 1.5 | 1.0872 |
| 20 | 2.0 | 1.4496 |
| 60 | 1.0 | 0.4185 |
| 60 | 1.5 | 0.6277 |
| 60 | 2.0 | 0.8370 |
| 120 | 1.0 | 0.2959 |
| 120 | 1.5 | 0.4439 |
| 120 | 2.0 | 0.5918 |
| 252 | 1.0 | 0.2042 |
| 252 | 1.5 | 0.3063 |
| 252 | 2.0 | 0.4084 |
| 504 | 1.0 | 0.1444 |
| 504 | 1.5 | 0.2166 |
| 504 | 2.0 | 0.2888 |

Full evidence and descriptive measurements are in report.json; COMPLETE.json fingerprints all outputs.
