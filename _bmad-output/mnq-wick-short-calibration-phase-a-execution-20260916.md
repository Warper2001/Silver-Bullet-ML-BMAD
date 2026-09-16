---
id: mnq-wick-short-calibration-phase-a-execution-20260916
phase: A
status: completed-independently-audited
sample_role: calibration-development-only
evaluation_allowed: false
original_gate_verdict: POWER_UNDETERMINED
confirmation_authorized: false
collector_activated: false
implementation_revision: 646f27d620d8543a89b9396e1c3ac57e0d4f6d40
---

# Phase A historical calibration execution record

The verified, committed runner completed the registered Phase A calculation
using only `/root/mnq_historical.json`, retaining records strictly before
`2026-03-01T00:00:00Z`. This is calibration/development evidence only. The
original `POWER_UNDETERMINED` gate and `evaluation_allowed=false` remain binding;
no fresh power assessment, profitability verdict, confirmation, Phase B,
collector activation or trading action occurred.

## Execution and provenance

- Implementation committed before input access: `646f27d620d8543a89b9396e1c3ac57e0d4f6d40`.
- Input SHA-256: `e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924`.
- Calibration registration SHA-256: `429dcb566b6b03a7565fc389d9c0aa048099e48718f20024bc52cd4919135a84`; canonical revision `3167d5354d2b71af413e9e2404b69efa82df96fc`.
- Run began `2026-09-16T18:24:07.847851+00:00`; calculation completed `2026-09-16T18:24:57.334094+00:00`; process exit status zero.
- Output directory: [mnq-wick-short-calibration-phase-a-20260916](mnq-wick-short-calibration-phase-a-20260916/).
- [Run log](mnq-wick-short-calibration-phase-a-20260916.session.run.log) records provenance verification, count reconciliation before alignment, and completion.
- Output manifest SHA-256: `bc2ddd9d95e8c95658fdda72259e266dd6c62d14e9f65005205e8d3200521779`.

The runner used `nohup` with an exclusive output directory and an exclusive log.
The first detached launch disappeared before creating an output directory and
left an empty log; the successful launch retained its shell wait in a tool
session. No output was overwritten. The completed report binds all canonical
artifacts, current implementation file revisions/hashes, source input, Python
interpreter and installed dependency inventories. The manifest hashes all three
ledgers plus JSON/Markdown reports, and `COMPLETE.json` binds the manifest. This
record and the final results commit provide the outer publication record.

## Reconciliation and descriptive measurements

Counts matched the bound original gate before aligned outcomes were calculated:
576 observed RTH dates, 515 eligible sessions, 61 exclusions (23 incomplete and
38 mixed-contract), 585 signals on 351 sessions, and 164 eligible zero-signal
sessions. Exactly 62,277 records at or after the cutoff were skipped. Frequency
is 1.1359223300970873 signals per eligible session.

| Reference outcome | Mean dollars per signal | Aggregate dollars |
| --- | ---: | ---: |
| Gross | -4.085470085470085 | -2390.00 |
| Assumed net, $1.22 cost | -5.305470085470085 | -3103.70 |
| Assumed net, $2.22 cost | -6.305470085470085 | -3688.70 |
| Assumed net, $3.22 cost | -7.305470085470085 | -4273.70 |

These are frozen historical reference-price measurements, with all registered
cost assumptions retained. Full distributions, corrected session/week
Student-t intervals and their envelopes, session totals and frequency are in
the [report](mnq-wick-short-calibration-phase-a-20260916/report.md) and its JSON
companion. They establish no future edge, fill availability or profitability
verdict.

## Independent persisted-output audit

The parent independently implemented and ran an audit without importing the
calibration runner. Its scope is internal consistency of the persisted output
ledgers and reports: the output hash chain, 576 eligibility rows, 585
interval/outcome records and their recorded component-minute mapping, 515
session totals including 164 zeros, monetary signs and all cost scenarios, and
all nine estimands with both clustered intervals, envelopes and empirical
distributions. It did not independently rescan `/root/mnq_historical.json`,
reconstruct all RTH source rows, or independently prove that copied component
prices match that source. Input provenance and cutoff access were verified by
the production runner; the audit compares the recorded cutoff count to the
registered prior count. Agreement between persisted artifacts must not be
described as an independent raw-source replay.
The independent arithmetic uses exact decimal-rational observations, separate
from the runner's exact binary-rational residual arithmetic; comparisons passed.

The initial audit scripts were archived with the results commit. Review later
hardened those scripts; the original versions and their hashes remain available
in commit `9f291030bc451e7cb3aabe69183a401dc0bf412b`. Current script hashes are:

- [Persisted-output audit](mnq-wick-short-calibration-phase-a-audit-20260916/mnq_phase_a_audit_outputs.py), SHA-256 `f56c7cb4df3205a1ff216f8f438830ebcb61a83e38be7d5ae69f841994765e97`.
- [Independent arithmetic](mnq-wick-short-calibration-phase-a-audit-20260916/mnq_phase_a_independent_audit.py), SHA-256 `a38656718db467f63b1065a5f20fb69f0cbf57426af9a9bf681bb2af764dea58`.
- [Archived oracle fixture](mnq-wick-short-calibration-phase-a-audit-20260916/independent-oracles.json), SHA-256 `645249706b217cd8d7659b6944abe7d989e7c54e0eb60e0e3c314acb5b1bb2dc`. The helper resolves this file beside its own script, without any `/tmp` dependency.
- [Initial archived audit log](mnq-wick-short-calibration-phase-a-audit-20260916/audit.log): **INDEPENDENT AUDIT PASSED** using the initial script versions. This original log is preserved; it is not a claim that the hardened scripts were executed on the market outputs during the review-fix task.

Reproduce from the repository root using the root `.venv` interpreter:

```text
.venv/bin/python _bmad-output/mnq-wick-short-calibration-phase-a-audit-20260916/mnq_phase_a_audit_outputs.py _bmad-output/mnq-wick-short-calibration-phase-a-20260916
```

The [pre-execution verification record](mnq-wick-short-calibration-phase-a-verification-20260916.md)
documents 61 passing synthetic tests plus formatting, lint and targeted type
checks. No original gate/spec/registration bytes, strategy parameters, live
code, live services, or production ledgers were changed; sealed holdout data
was not accessed. The parent performs the final merge after workflow review.

## Review corrections and focused verification

The review fixes preserve the completed run directory and its original hashes.
They correct cost-rounding degeneracy by deriving per-signal uncertainty from
gross centered observations and session uncertainty from exact gross minus
registered cost times signal count. Descriptive distributions continue to use
the persisted ledger float values. Completion now publishes a fully written,
closed temporary marker atomically without overwriting a marker, and a final
logging failure cannot create a contradictory failure record after publication.

The hardened independent audit requires the exact manifest inventory and
publication schema/status, both interval endpoints, the full quantile set,
all report counts, internally derivable per-date coverage/contracts/counts,
cluster label/count mappings and concentration, Student-t critical values, and
ECDF cumulative probabilities. Both audit scripts refuse optimized execution
so that their assertions cannot silently disappear.

Focused verification used only synthetic fixtures and the two affected test
files: `tests/unit/test_mnq_wick_short_calibration.py` and
`tests/unit/test_mnq_wick_short_calibration_audit.py`: **85 passed in 12.63s**.
This includes balanced unequal/equal cluster costs, constant/varying session
counts, interrupted final-marker writes, logging failure after completion,
real initial provenance refusal before input access, and 22 rehashed-corruption
cases. A final targeted check after moving the optimization guards below the
imports passed the three optimized-execution/relocated-oracle tests.

These corrections have not been used to rerun the historical container in this
review-fix task. The parent handles full verification, the fix commit, any new
committed-code run in a fresh output directory, and final merge.
