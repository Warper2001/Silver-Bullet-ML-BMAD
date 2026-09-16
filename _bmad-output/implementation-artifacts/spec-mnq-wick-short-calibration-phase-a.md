---
title: MNQ wick-short Phase A historical calibration
type: feature
created: 2026-09-16
status: in-review
baseline_commit: 403a9542e60891f22e4f14e7411a90557acf660e
route: dispatch
review_loop_iteration: 0
context:
  - AGENTS.md
  - _bmad-output/preregistration_mnq_wick_short_calibration_20260916.md
---

<frozen-after-approval reason="Phase A intent approved by the user's go ahead after the registered implementation plan">

## Intent

**Problem:** The registered wick-short candidate has timing/frequency and shifted
dispersion evidence, but lacks aligned historical calibration measurements.
**Approach:** Implement, synthetically verify and commit a standalone Phase A
runner, execute the bound historical input, independently verify its ledgers
and summaries, and commit/merge code and results. The calibration registration
is the authoritative scientific contract; this spec only maps it to code.

## Boundaries & Constraints

**Always:** Preserve all frozen mechanics, eligibility, UTC cutoff and all three
cost scenarios. Include eligible zero-signal sessions. Report both corrected
session/week Student-t intervals and their envelope, including session-level
totals/frequency. Mark invalid uncertainty unassessable. Bind original artifacts,
input, registration, implementation and output hashes and canonical revisions.
Reconcile counts before aligned outcomes. Keep `evaluation_allowed=false` and
the original `POWER_UNDETERMINED` gate. Label all results calibration only.

**Never:** Change original gate/spec/registration files, search parameters,
declare profitability, infer future edge, access sealed holdout, start Phase B,
trade, touch live code/config/services or production ledgers, install libraries
in `.venv`, or overwrite previous outputs. No fresh power verdict is computed:
the prior gate remains binding; this is the separate authorized calibration.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
| --- | --- | --- | --- |
| Bound sample | Committed code, matching input/artifacts/counts | Eligibility, outcome and session ledgers; JSON/Markdown summaries | Calibration only |
| Provenance mismatch | Changed input, pinned artifact or uncommitted implementation | No aligned calculation or completed output | Nonzero exit and reason |
| Invalid input/counts | Malformed rows, duplicates, gate reconciliation mismatch | Refuse before alignment | Nonzero exit and reason |
| Ineligible session | Incomplete or mixed contract | Retain reason/coverage; omit outcomes | Never impute zeros |
| Eligible no-signal | Complete single-contract session | Zero count and totals | Included in session estimands |
| Invalid inference | Too few observations/clusters, invalid/zero variance | Null interval/SE with reason; partial companion retained | Envelope unassessable |
| Output collision | Existing run directory or prohibited destination | Preserve existing data | Refuse before market input |

</frozen-after-approval>

## Code Map

- `tools/mnq_wick_short_power_gate.py`: reuse pure `load_minutes`, `sessionize`,
  `bars_for_session`, `signal_slots`, `week_key`, constants and dataclasses.
  Never call `run`, shifted outcomes, old `cluster_se`, or its publishing path.
- `tests/unit/test_mnq_wick_short_power_gate.py`: existing synthetic mechanics
  regression suite; keep unchanged. Its old SE tests describe the original gate.
- `_bmad-output/preregistration_mnq_wick_short_calibration_20260916.md`: exact
  pin `429dcb566b6b03a7565fc389d9c0aa048099e48718f20024bc52cd4919135a84`,
  canonical revision `3167d5354d2b71af413e9e2404b69efa82df96fc`; table binds
  all original artifacts and the external input.
- `pyproject.toml`: existing numpy/scipy/pytest dependencies; no install needed.

## Tasks & Acceptance

**Execution:**
- [x] `tools/mnq_wick_short_calibration.py` -- add protected CLI, provenance,
  reconciled eligibility, aligned ledgers, registered statistics and publication.
- [x] `tests/unit/test_mnq_wick_short_calibration.py` -- verify synthetic mechanics,
  independent monetary/statistical arithmetic, refusal and publication behavior.
- [x] `_bmad-output/mnq-wick-short-calibration-phase-a-20260916/` -- after code
  verification and commit, generate and independently reconcile the real results.

**Acceptance Criteria:**
- Given the bound sample, when eligibility and signals are counted, then the
  run reconciles 576 observed/515 eligible sessions, 585 signals, 351 signal
  sessions and original exclusion/cutoff counts before aligned outcomes.
- Given signals, when outcomes are computed, then their own next-bar interval
  produces `2*(open-close)` and all costs; provenance identifies component
  minutes, contract, slots, labels and reference interval boundaries.
- Given unequal clusters, when uncertainty is computed, then exact independent
  arithmetic agrees with `G/(G-1)*sum(R_g^2)/N^2`, t degrees of freedom and both
  interval endpoints/envelope, including zero-valued nonempty clusters.
- Given completed outputs, when independently audited, then counts, signs,
  monetary totals, sample roles and provenance reconcile, with no live writes.

## Implementation Notes

Execution repository: `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/mnq-wick-short-calibration-phase-a`.
Use the interpreter and check tools from `/root/Silver-Bullet-ML-BMAD/.venv/bin/`
without installing anything. All edits, commits and outputs belong in that
worktree; the parent will perform the final merge.

No unresolved intent gaps. Footprint is a new runner/test pair plus research
artifacts; no consumers or live deployment. User's existing approval covers
this Phase A scope and execution. Collector development/activation is later.
Production CLI binds `/root/mnq_historical.json` and refuses alternate/sealed
paths; unit tests call pure helpers on synthetic inputs. New outputs live under
the worktree's `_bmad-output/`; an exclusive run directory prevents overwrite.
Record interpreter/dependency versions and hashes, start/end UTC timestamps,
schema version and JSON null reasons. Quantiles are descriptive summaries,
not trading thresholds. Calendar audit flags are observations, not new filters.

## Spec Change Log

## Review Triage Log

| Finding | Verdict | Evidence and route |
| --- | --- | --- |
| Blind 1: rounded constant-cost residuals | medium | Confirmed zero gross cluster variance becomes tiny positive net variance after float subtraction; patch scenario uncertainty using gross centered values/exact pre-subtraction arithmetic. Actual recorded intervals passed independent audit. |
| Blind 2: partial completion marker | medium | Direct final-name write can fail after creating a truncated marker; patch atomic publication of fully written marker. |
| Blind 3: final logging failure | medium | The final flushed print is inside the failure handler's try after COMPLETE publication; patch so logging cannot publish FAILED after success. |
| Blind 4: empty manifest inventory accepted by audit | medium | Audit loops only over supplied hash keys and never requires the five expected files; patch exact inventory/status/schema validation. |
| Blind 5: missing interval endpoints/quantiles accepted | medium | zip and supplied-quantile iteration skip absent entries; patch expected shapes and quantile keys. |
| Blind 6: ledger/report counts not reconciled by audit | medium | Existing audit checks row counts but omits report counts and per-date eligibility counts/contracts; patch all internally derivable cross-ledger checks. |
| Blind 7: no independent rescan of raw source | low | Audit explicitly operates on persisted outputs and has not authenticated copied minute values against raw source. Clarify that boundary in execution record; no new source access or changed strategy calculation is needed. |
| Blind 8: unchecked diagnostics | medium | Label/count mapping, HHI, t critical values and ECDF cumulative probabilities are not compared; patch these independently derivable fields. |
| Blind 9: optimized Python disables audit assertions | low | Python optimization disables assert-based checks; direct refusal of optimized execution is a small correction to the audit entry points. |
| Blind 10: helper self-check uses temporary fixture | low | Main audit reproduces, but running the archived helper standalone depends on /tmp; archive fixture alongside helper and use its directory. |
| Edge 1: cost subtraction violates zero-variance invariance | medium | Same verified cause as Blind 1; same patch preserves all registered mechanics and costs. |
| Edge 2: marker write interruption | medium | Same verified cause as Blind 2; same atomic-marker patch. |
| Verification 1: pre-access provenance regression gap | medium | Reviewer mutation moving initial verification after load leaves 61 tests passing. Add a CLI test with real corrupted synthetic pin and forbidden input access to lock down the already-correct ordering. |


## Verification

- Root `.venv/bin/python -m pytest` on both wick-short test files from the
  worktree: all tests pass without real market data or credentials.
- Root `.venv/bin/black --check`, `.venv/bin/flake8` and targeted mypy on new
  code/tests, respecting existing formatting/type configuration.
- Confirm code committed and original hashes unchanged before the real run.
- Launch real calibration via `nohup`, then inspect log and completion output;
  independently recompute ledger statistics before committing/merging results.

Independent fixture guidance: `[0,2,4,6,8]` with clusters `[a,a,b,c,c]`
has residual sums `[-6,0,6]`, variance `4.32`, and df `2`; retain cluster b.
An exactly constant `[0.1,0.1,0.1]` must yield unassessable inference despite
floating-point centering. Add valid bearish OHLC and actual adjacent interval
price/time mapping tests; the inherited gate tests do not fully cover these.

Review execution: three lenses completed. The host initially lacked a third
review slot; verification-gap review launched after the blind review completed,
before parent triage. No layer was skipped. All findings were checked against
code or filed mutation evidence; bounded corrections preserve the frozen intent.

All triaged corrections are implemented. Parent verification passed 96 relevant
synthetic tests, all targeted static checks, the archived arithmetic self-check,
and the hardened persisted-output audit against the unchanged initial run.
See `mnq-wick-short-calibration-phase-a-review-verification-20260916.md` for
precise commands, scope and the final annotation/message-only corrections.
The patch route requires a committed-code rerun in a fresh directory before
final publication; the original seven result files retain their original hashes.
