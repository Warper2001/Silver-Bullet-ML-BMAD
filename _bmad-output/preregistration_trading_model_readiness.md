---
id: trading-model-readiness-20260921
status: preregistered-readiness-only
created: 2026-09-21
baseline_commit: f2524c0d12b36ed61175cceb9da766e4d9942ae8
economic_testing_authorized: false
sealed_holdout_access_authorized: false
---

# Trading model: readiness-only preregistration

## Question and permitted work

Can existing unsealed MNQ data and available computing resources support designing an intraday market-model experiment? Before implementing or running this audit, commit this document. Work is descriptive data/compute readiness only, not a strategy test or a seal of any economic decision threshold.

Permitted inputs are explicitly named development CSVs: the MIM-X contract-labeled minute history, the known-contaminated 2025 minute CSV as a diagnostic comparator, and the reconstructed 2025 splice CSV if available. No source is designated unseen. Overlapping dates across files cannot add independent evidence. Never access data/sealed_holdout, including through aliases, or live trade/account ledgers.

## Measurements

Record byte hashes, bytes, schema, row counts, valid OHLCV/timestamp counts, observed date bounds, contract identifiers, non-increasing within-contract timestamps, duplicate contract-minutes, and descriptive weekday RTH minute masks under both start/end timestamp hypotheses. Count full candidate 15-minute groups and complete regular-length grids. These grids do not authenticate a historical exchange calendar, full coverage, finalization or receipt times. Do not read strategy signals, form forward-return labels, choose contracts, compute PnL, train a model, or select profitable observations.

Record elapsed CSV audit time and rows/second, interpreter/package availability and NVIDIA device/tool visibility. Parameter-weight storage arithmetic may be reported as a lower bound only. Without a real measured model workload, report GPU training throughput, hours and cost as unmeasured. Do not install packages or spend money.

## Gate semantics

The audit cannot authorize strategy testing. Data remains HOLD_DATA because source provenance, causal front-month selection, provider label/finalization semantics, historical calendar, costs and a genuinely untouched evaluation interval require separate evidence. Report missing/invalid files and malformed records, never silently repair them.

Power is UNASSESSABLE: no economic target effect, admissible evaluation population, dependence-adjusted variance model or multiple-comparison allocation has yet been justified. Raw bars, overlapping windows, prompts and repeated training seeds are not independent market outcomes. Do not borrow an earlier strategy's power verdict or numerical thresholds. Run an experiment-specific power gate only after those inputs are established and registered; UNDERPOWERED is a valid stop.

The next economic registration must cite development artifacts and their commits for any calibrated thresholds, predeclare the comparisons, chronological splits, dependency treatment, costs, risk/account constraints and statistical decisions, and respect the repository's unseen-data and one-change-per-experiment policies. Model pretraining exposure needs its own audit. No live deployment follows from this readiness registration.

## Reporting and continuation

Produce machine-readable and Markdown readiness reports binding inputs and audit code to hashes and source revision, plus a runbook and an explicitly unsealed future comparison protocol. Record limitations without turning unresolved fields into defaults. If gates remain unresolved, deliver the readiness evidence and concrete missing requirements; do not run the trading experiment.
