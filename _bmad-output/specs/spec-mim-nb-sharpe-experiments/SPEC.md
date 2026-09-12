---
id: SPEC-mim-nb-sharpe-experiments
companions: [protocol.md]
sources: []
---
# MIM-NB Sharpe improvement experiments

## Why
Alex seeks a simulation-proven revenue strategy with higher Sharpe and controlled drawdown, accepting lower profit. Four predefined entry filters are hypotheses, not proven improvements.

## Capabilities
- **CAP-1**
  - **intent:** Reproduce baseline.
  - **success:** 1323 selected eligible days and identical decisions/trades for both timings, all three costs, within1e-8 dollars accounting and1e-9 indicators, ignoring only new metadata.
- **CAP-2**
  - **intent:** Calculate causal features.
  - **success:**  R/E use trailing30 current-session completed closes including decision close; P uses respective minute bands; future changes cannot alter earlier output.
- **CAP-3**
  - **intent:** Simulate all fixed candidates.
  - **success:**  independent entry gates preserve exits, stop/guard accounting, timing and eligibility; all four run without combinations.
- **CAP-4**
  - **intent:** Evaluate paired results.
  - **success:**  common daily grid with flats, fixed10000 capital, zeroRF Sharpe sqrt252*mean/std(ddof1); thresholds and synchronized20k bootstrap with three dependence settings applied exactly.
- **CAP-5**
  - **intent:** Publish reproducible evaluation.
  - **success:**  immutable source/input/config manifests, ledgers, Markdown and standalone HTML; failures and none-qualifies outcomes explicit.

## Constraints
- Follow protocol.md exactly; freeze definitions before computing returns.
- Preserve all frozen comparison and production artifacts; new research is isolated.

## Non-goals
- Deployment, sizing changes, parameter search, new acquisition, or alteration of the prospective A/B experiment.

## Success signal
All four experiments complete with verified baseline parity, reconciled results, dependence-aware uncertainty and an explicit historical-only recommendation, including none qualifies.

## Assumptions
- Common fixed10000 capital, zeroRF and252-session annualization; exposed history cannot establish prospective improvement.

