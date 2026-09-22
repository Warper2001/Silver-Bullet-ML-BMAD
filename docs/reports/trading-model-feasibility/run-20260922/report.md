# Trading-model readiness

**HOLD — strategy testing is not authorized by this audit.**

Source revision: `eba41d90484ab8779ad0892ad0e6b58d5b072c48`.
Readiness preregistration: `ff7fbeb74491704ea0a5c36e5ac1aff2d5cf633f`.

## Dataset observations

### mnq_1min_by_contract.csv

Path: `/root/Silver-Bullet-ML-BMAD/data/mim_x/mnq_1min_by_contract.csv`.
Status: AUDITED; gate: HOLD_DATA.
Rows: 2,028,965; structurally valid rows: 2,028,965; invalid rows: 0.
Observed weekday dates: 1,481; dates with multiple contracts: 43.
Contract identity: OBSERVED_NOT_AUTHENTICATED; structural quality: STRUCTURAL_CHECKS_ONLY.
Range: 2020-12-18T00:01:00+00:00 to 2026-08-28T15:19:00+00:00.
CPU CSV audit: 19.145 seconds, 105981 rows/second (hashing excluded).

- start-label hypothesis: 1,415 dates with any full regular grid; 37,508 candidate contract-groups.
- end-label hypothesis: 1,415 dates with any full regular grid; 37,508 candidate contract-groups.

SHA-256: `ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea`.

### mnq_1min_2025.csv

Path: `/root/Silver-Bullet-ML-BMAD/data/processed/dollar_bars/1_minute/mnq_1min_2025.csv`.
Status: AUDITED; gate: HOLD_DATA.
Rows: 289,230; structurally valid rows: 289,230; invalid rows: 0.
Observed weekday dates: 260; dates with multiple contracts: 0.
Contract identity: ABSENT; structural quality: STRUCTURAL_CHECKS_ONLY.
Range: 2025-01-01T23:01:00+00:00 to 2025-12-31T22:00:00+00:00.
CPU CSV audit: 2.162 seconds, 133775 rows/second (hashing excluded).

- start-label hypothesis: 207 dates with any full regular grid; 5,507 candidate contract-groups.
- end-label hypothesis: 207 dates with any full regular grid; 5,506 candidate contract-groups.

SHA-256: `3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822`.

### mnq_1min_2025_frontmonth.csv

Path: `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/gapfade-splice-sensitivity/_bmad-output/diagnostics_gap_fade_splice_20260916/mnq_1min_2025_frontmonth.csv`.
Status: AUDITED; gate: HOLD_DATA.
Rows: 281,645; structurally valid rows: 281,645; invalid rows: 0.
Observed weekday dates: 260; dates with multiple contracts: 0.
Contract identity: ABSENT; structural quality: STRUCTURAL_CHECKS_ONLY.
Range: 2025-01-01T23:01:00+00:00 to 2025-12-31T22:00:00+00:00.
CPU CSV audit: 2.273 seconds, 123923 rows/second (hashing excluded).

- start-label hypothesis: 207 dates with any full regular grid; 5,507 candidate contract-groups.
- end-label hypothesis: 207 dates with any full regular grid; 5,506 candidate contract-groups.

SHA-256: `f1fe5b36abba90681d8b1439a3975f94e4b4d1040093c7c0368629a807d219d4`.

These are overlapping, previously researched development inputs. Counts cannot be added across files, contracts or windows. A full regular-length weekday grid is not authenticated calendar completeness or decision-time availability.

## Gates

Data: HOLD_DATA. Missing evidence:

- SOURCE_PROVENANCE_NOT_ADMITTED
- CAUSAL_FRONTMONTH_SELECTION_NOT_ADMITTED
- BAR_LABEL_AND_DECISION_TIME_AVAILABILITY_UNVERIFIED
- HISTORICAL_EXCHANGE_CALENDAR_UNVERIFIED
- FILL_AND_COST_EVIDENCE_UNVERIFIED
- UNTOUCHED_EVALUATION_INTERVAL_NOT_REGISTERED
- PRETRAINING_EXPOSURE_NOT_AUDITED

Power: UNASSESSABLE. No registered economic effect, eligible untouched evaluation sample, dependence-adjusted variance or comparison allocation. No strategy returns were computed.

## Compute

Python 3.12.3; 4 reported logical CPUs; NVIDIA probe: NVIDIA_TOOL_UNAVAILABLE.
Package visibility: `{"numpy": true, "pandas": true, "scipy": true, "torch": false, "transformers": false}`.
GPU training throughput, training hours and training cost: NOT MEASURED. Preprocessing throughput cannot substitute for a model benchmark.

## Next requirements

Establish source/causal-roll/calendar/availability/cost provenance; register evaluation data and effect-size justification; run the experiment-specific power gate. Then benchmark the pinned model in an isolated training environment before budgeting training. No data acquisition, dependency installation, model training or deployment was performed.
