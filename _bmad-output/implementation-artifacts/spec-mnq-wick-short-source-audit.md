---
title: 'MNQ wick-short source audit and worktree bootstrap repair'
type: 'bugfix'
created: '2026-09-16'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
baseline_commit: '2547dd76913668e57cce3e079f0061cae0d30e13'
context:
  - AGENTS.md
  - _bmad-output/preregistration_mnq_wick_short_calibration_20260916.md
  - _bmad-output/mnq-wick-short-calibration-phase-a-execution-r2-20260916.md
---

<frozen-after-approval reason="The user directed repair of the blocked audit and completion of the bounded source analysis.">

## Intent

**Problem:** An isolated worktree lacks the machine-local, ignored `_bmad/`
workflow installation, so the required build renderer cannot start there. The
completed wick-short calibration also shows a large loss concentration in March
and April 2025 that must be checked for a contract-selection defect before the
project is stopped or any follow-on observation is considered.

**Approach:** Add a guarded bootstrap utility that exposes the primary
checkout's existing BMAD configuration to a selected worktree without
overwriting its files or sharing generated render output. Add a read-only,
hash-bound diagnostic that checks the bound historical source, the
already-published calibration ledger, contract labels, session integrity and
the official 2025 CME roll calendar; publish an auditable decision record.

## Boundaries & Constraints

**Always:** Work only in this isolated worktree and keep the root checkout's
tracked files, live code, services, strategy configuration and trade records
unchanged. Treat `/root/mnq_historical.json` as the registered external input:
verify its existing SHA-256 before decoding it, access no sealed holdout path,
and never alter it. Preserve both Phase A result directories and their hashes.
Use the full RTH 09:31–16:00 New York session, exact raw contract labels and
the calibration outcome ledger only for diagnosis. State that any exclusion
after seeing results is diagnostic, never an adopted strategy filter. Cite the
CME Equity Index roll-date page for the March 17, 2025 lead-month transition.

**Never:** Rerun, rewrite or reinterpret the calibration as confirmation;
change strategy mechanics, costs or eligibility; launch Phase B; send orders;
deploy/restart a service; add dependencies; read the sealed holdout; modify or
overwrite an existing `_bmad` entry in a worktree; or claim that a contract
label by itself proves trade execution or a future edge.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Bootstrap | Empty ignored `_bmad/` worktree overlay and valid source install | Creates only required symlinks; generated render remains local | Refuse source/layout mismatch |
| Existing path | Target entry exists or is not the expected link | Leave it untouched | Nonzero status naming the entry |
| Audit baseline | Matching input/report hashes and raw rows | JSON evidence plus Markdown decision record | No strategy verdict beyond data validity |
| Provenance failure | Changed input, report or incomplete source rows | No completed audit publication | Nonzero status with precise reason |
| Pre-roll label | MNQM25 RTH session before 2025-03-17 | Count, list and quantify its recorded calibration outcomes | Flag contaminated; do not silently remove it |
| Mixed/incomplete day | Duplicate/mixed contracts or missing RTH labels | Report separately from complete single-contract days | Never infer a substitute bar |

</frozen-after-approval>

## Code Map

- `tools/mnq_wick_short_calibration.py` -- pins the source hash, frozen cutoff,
  session definition and original artifacts; audit must not modify or invoke its
  production calculation path.
- `_bmad/scripts/render_skill.py` and `_bmad/scripts/config_utils.py` in the
  primary checkout -- require local `_bmad` config but publish render output
  below the supplied project root; bootstrap may link only config/input paths.
- `_bmad-output/mnq-wick-short-calibration-phase-a-20260916-r2/` -- immutable
  outcome and session ledgers to reconcile to the raw source.
- `AGENTS.md` -- records existing MNQ roll-contamination history and all live
  safety rules.

## Tasks & Acceptance

**Execution:**
- [x] `tools/bootstrap_bmad_worktree.py` -- create explicit, non-overwriting
  links for the local BMAD runtime needed by isolated worktrees.
- [x] `tools/audit_mnq_wick_short_source.py` -- implement the bound source and
  ledger audit with deterministic JSON/Markdown output.
- [x] `tests/unit/test_bootstrap_bmad_worktree.py` and
  `tests/unit/test_audit_mnq_wick_short_source.py` -- cover refusal, local
  render isolation, pre-roll classification and no-source-access failures.
- [x] `_bmad-output/mnq-wick-short-source-audit-20260916-r3/` -- publish the
  completed read-only evidence, commands and decision recommendation.

**Acceptance Criteria:**
- Given an empty test worktree, when bootstrap runs against a valid source
  installation, then it creates only the named configuration links and the
  render destination resolves inside that test worktree.
- Given an existing file or divergent link, when bootstrap runs, then it exits
  without replacing that target.
- Given the bound source and Phase A ledger, when the audit runs, then every
  analysed outcome maps to raw same-contract RTH components and pre-March-17
  MNQM25 sessions are disclosed with their exact count and dollars.
- Given a changed source or report hash, when the audit runs, then it refuses
  before publication.

## Implementation Notes

The initial renderer failure occurred because `_bmad/` is ignored and therefore
absent in a new Git worktree. A local manual overlay demonstrated that symlinks
to the root configuration/scripts permit rendering while `_bmad/render/` stays
inside this worktree. The final utility must make that behavior explicit and
testable rather than relying on manual setup.

The final audit publication is the separate
`mnq-wick-short-source-audit-20260916-r3` directory, generated after source
integrity and reviewer hardening. It maps all 585 Phase A outcomes to
same-contract raw RTH components and reports 576 observed sessions: 515
complete single-contract sessions, 61 incomplete sessions and 40
mixed-contract sessions (40 overlap the incomplete count). It identifies eight
pre-March-17 MNQM25 sessions with 18 outcomes totalling -$549 gross. This is a
diagnostic contamination finding; it does not retroactively remove observations
or produce a strategy verdict.

## Review Triage Log

| Layer | Finding | Verdict | Evidence / route |
| --- | --- | --- | --- |
| Blind / edge | The raw source could change between hashing and decoding. | Patched | The audit reads bytes once, hashes those bytes, then decodes the same buffer. |
| Blind / edge | Missing or malformed input/publication could escape as a traceback. | Patched | Input, ledger and Phase A publication failures now become `AuditError`; no output is published. |
| Blind | Outcome evidence could omit interval order, values or reconstruction. | Patched | It verifies each contiguous 5+5 minute window, local/UTC labels, OHLC, geometry and reference-dollar arithmetic. |
| Blind | Eligibility could omit contract minute counts, labels or exclusions. | Patched | It reconciles exact raw labels, contract counts and published exclusion fields for every observed session. |
| Blind / verification | The audit orchestration and immutable destination were untested. | Patched | Added orchestration and existing-output refusal tests. |
| Edge | A partially created bootstrap overlay could remain after link failure. | Patched | Bootstrap removes links and a newly created empty target directory on failure; rollback is tested. |
| Edge | Tests depended on the primary checkout’s runtime. | Patched | Bootstrap tests construct synthetic source and target worktrees. |
| Blind | The initial report lacked session-integrity evidence. | Superseded | Only the fresh r3 publication is committed; it contains the full session-integrity summary. |
| Blind | Audit every 2025 CME roll date. | Out of scope | The approved diagnostic is limited to the March/April anomaly and the March 17 lead-month transition; no wider filter is adopted. |

## Verification

- `/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/test_bootstrap_bmad_worktree.py tests/unit/test_audit_mnq_wick_short_source.py -q` -- expected: all synthetic tests pass without market data.
- `/root/Silver-Bullet-ML-BMAD/.venv/bin/black --check` and `flake8` on new
  tools/tests -- expected: no formatting or lint errors.
- Run the completed audit once with the bound input and inspect the JSON and
  Markdown hashes, raw/ledger reconciliation, contract table and stated limits.
