---
id: mnq-wick-short-calibration-phase-a-execution-r2-20260916
phase: A
status: completed-independently-audited
sample_role: calibration-development-only
evaluation_allowed: false
original_gate_verdict: POWER_UNDETERMINED
confirmation_authorized: false
collector_activated: false
implementation_revision: 7ab4c9a20fde3d9aa3358abc4de42d2dc7060229
---

# Corrected Phase A historical calibration execution

The reviewed runner completed the same registered historical calculation from
committed code. This record identifies the corrected `r2` publication. The
[first execution](mnq-wick-short-calibration-phase-a-execution-20260916.md) and
all seven original result files are preserved byte for byte. Both runs describe
the same previously exposed calibration sample; they are not additional or
independent observations and neither is confirmation evidence.

## Execution and bindings

- Code committed before the corrected run: `7ab4c9a20fde3d9aa3358abc4de42d2dc7060229`.
- Started `2026-09-16T18:57:56.047828+00:00`; completed `2026-09-16T18:58:14.649988+00:00`; process exit status zero.
- Input `/root/mnq_historical.json`, SHA-256 `e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924`; only rows strictly before `2026-03-01T00:00:00Z` enter calibration.
- [Calibration registration](preregistration_mnq_wick_short_calibration_20260916.md), canonical revision `3167d5354d2b71af413e9e2404b69efa82df96fc`, SHA-256 `429dcb566b6b03a7565fc389d9c0aa048099e48718f20024bc52cd4919135a84`.
- [Corrected report](mnq-wick-short-calibration-phase-a-20260916-r2/report.md) and [JSON](mnq-wick-short-calibration-phase-a-20260916-r2/report.json); output [manifest](mnq-wick-short-calibration-phase-a-20260916-r2/manifest.json) SHA-256 `6aeb7869658fbfc2ad57ec7385ad82175ef9e925af8ab3eb88dfeecf6835726e`.
- [Run log](mnq-wick-short-calibration-phase-a-20260916-r2.run.log), [independent audit log](mnq-wick-short-calibration-phase-a-20260916-r2.audit.log), and [exact comparison to the initial publication](mnq-wick-short-calibration-phase-a-20260916-r2.comparison.json).

The completed JSON binds the original registration, gate, specifications and
results to their canonical revisions/hashes, plus current calculation code,
input and runtime provenance. `COMPLETE.json` binds the output manifest.
The results commit is the outer publication record. All nine original and
calibration pins matched their recorded hashes after the corrected run.

## Measurements and interpretation

Counts reconciled before aligned outcomes: 576 observed dates, 515 eligible
sessions, 585 signals on 351 sessions, and 164 eligible zero-signal sessions.
There were 23 incomplete and 38 mixed-contract excluded sessions; 62,277
post-cutoff records were skipped. Mean frequency is 1.1359223300970873 signals
per eligible session; gross per-signal sample dispersion is $50.9390874237179.

| Historical reference outcome | Mean dollars per signal | Outer envelope of approximate 95% intervals |
| --- | ---: | --- |
| Gross | -4.09 | [-8.45, 0.28] |
| Assumed net, $1.22 cost | -5.31 | [-9.67, -0.94] |
| Assumed net, $2.22 cost | -6.31 | [-10.67, -1.94] |
| Assumed net, $3.22 cost | -7.31 | [-11.67, -2.94] |

The report retains both session- and ISO-week-clustered Student-t intervals,
cluster diagnostics, all cost scenarios, eligible-session totals including
zeros, and empirical distributions. The envelope is a sensitivity summary,
not a third confidence procedure. These reference-price measurements estimate
the size, variability and frequency seen in the exposed calibration sample.
Costs remain prospective assumptions applied to historical prices. They do
not establish an executable future edge or support a profitability verdict.

## Verification and resolved review findings

The [post-review verification record](mnq-wick-short-calibration-phase-a-review-verification-20260916.md)
documents 96 relevant synthetic tests and passing formatting, lint and targeted
type checks before the corrected run. All three review lenses completed and
all triaged findings were addressed. Final annotation/message-only corrections
also passed the affected three synthetic cases and static checks.

The hardened independent audit passed on both the initial and corrected
publications. It verifies the exact manifest inventory and hash chain,
recorded interval/component mapping, cross-ledger counts/contracts/coverage,
monetary arithmetic, all nine estimands, both uncertainty calculations,
envelopes, cluster diagnostics and empirical distributions. It uses separate
exact decimal-rational arithmetic and does not import the calibration runner.
It does not rescan the raw source or independently authenticate copied source
prices. Its archived files, committed before this run, have these hashes:

- `mnq_phase_a_audit_outputs.py`: `f56c7cb4df3205a1ff216f8f438830ebcb61a83e38be7d5ae69f841994765e97`.
- `mnq_phase_a_independent_audit.py`: `a38656718db467f63b1065a5f20fb69f0cbf57426af9a9bf681bb2af764dea58`.
- `independent-oracles.json`: `645249706b217cd8d7659b6944abe7d989e7c54e0eb60e0e3c314acb5b1bb2dc`.

The three ledgers and all counts are byte-identical or exactly equal across
runs. Only 22 numerical summary fields differ, with a maximum absolute
difference of `1.7763568394002505e-15`; all displayed two-decimal measurements
are unchanged. The rerun verifies the cost-rounding/publication corrections;
it is not a replication on new data.

Concurrent main-branch work added a warning about MNQ CSV contract/roll
quality. This calculation uses the frozen hash-bound JSON and its registered
complete, single-contract session rule. Single-contract eligibility does not
by itself verify front-month identity or source accuracy, and this calibration
makes no such claim. The fixed sample and rule were not revised after seeing
outcomes; a later design must separately resolve contract/feed applicability.

The original `POWER_UNDETERMINED` and `evaluation_allowed=false` remain in
force. Phase B has not been built or activated. No new power decision,
confirmation, holdout access, live-code change, order, service operation,
deployment or strategy parameter change was performed. Confirmation requires
a separate prospective registration and a new justified power assessment.
