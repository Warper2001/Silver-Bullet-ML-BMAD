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
calibration runner. It verified the output hash chain, all 576 eligibility rows,
585 interval/outcome records and their component-minute mapping, 515 session
totals including 164 zeros, monetary signs and all cost scenarios, and all nine
estimands with both clustered intervals, envelopes and empirical distributions.
The independent arithmetic uses exact decimal-rational observations, separate
from the runner's exact binary-rational residual arithmetic; comparisons passed.

The original audit scripts are archived unchanged for reproducibility:

- [Persisted-output audit](mnq-wick-short-calibration-phase-a-audit-20260916/mnq_phase_a_audit_outputs.py), SHA-256 `3735c63ee37440db4b06af6923234dee73150d9a29be26ad6043400a92972072`.
- [Independent arithmetic](mnq-wick-short-calibration-phase-a-audit-20260916/mnq_phase_a_independent_audit.py), SHA-256 `33bd0e268fabc0c8abaeacee44469d8762b5603771fb8f53615b75fa8b0885d0`.
- [Archived audit log](mnq-wick-short-calibration-phase-a-audit-20260916/audit.log): **INDEPENDENT AUDIT PASSED**. The archived copy was also executed successfully to verify that the sibling import remains reproducible.

Reproduce from the repository root using the root `.venv` interpreter:

```text
.venv/bin/python _bmad-output/mnq-wick-short-calibration-phase-a-audit-20260916/mnq_phase_a_audit_outputs.py _bmad-output/mnq-wick-short-calibration-phase-a-20260916
```

The [pre-execution verification record](mnq-wick-short-calibration-phase-a-verification-20260916.md)
documents 61 passing synthetic tests plus formatting, lint and targeted type
checks. No original gate/spec/registration bytes, strategy parameters, live
code, live services, or production ledgers were changed; sealed holdout data
was not accessed. The parent performs the final merge after workflow review.
