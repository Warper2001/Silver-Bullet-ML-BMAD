---
title: Native reclaim conditional power feasibility gate
type: feature
created: 2026-09-14
status: done
route: dispatch
review_loop_iteration: 0
baseline_commit: 42d905e2914a0d901e933dfa86651c223e0594f3
context: []
---

<frozen-after-approval>

## Intent

User approved proceeding from native profile measurement to the properly gated strategy study. Implement and run the preregistered metadata-only power feasibility gate. No independent calibration or transferable net effect exists for this exact construct in inspected sources; make that blocker quantitatively useful with conditional sample-size and detectable-effect calculations rather than inventing a plausible edge. The registration is already committed before numerical calculations.

## Boundaries & Constraints

Use only the existing interpreter `/root/Silver-Bullet-ML-BMAD/.venv/bin/python` (SciPy1.17.1 available), no installs. Work only in `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/valentini-power`. Do not commit; parent handles commits/review and actual gate run. Synthetic tests only during implementation; do not run calculations on the actual six-session ledger. Existing strategy parameters, data exclusions and market-evaluation refusal remain unchanged. No signals, prices decoded, simulated trades, P&L, credential access, network or sealed holdout. Hashing artifacts is permitted but is not decoding price paths.

Read `_bmad-output/preregistration_valentini_native_power_20260914.md` fully. Its committed SHA256 is8defef100fe89382f331858e5d02e5cc495d8280189a2fe37fabca2acac09005, commit42d905e2914a0d901e933dfa86651c223e0594f3. Follow every frozen numerical/model choice. Treat hypothetical effects as scenarios, never evidence. Always POWER_UNDETERMINED/evaluation_allowed=false in this release, with explicit reasons; do not call missing evidence UNDERPOWERED.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Valid pinned measurement and prereg | Count eligible sessions, emit all declared conditional calculations, terminal gate |
| Missing/corrupt/hash-mismatched artifact, prereg or inventory | Refuse with exit2 before model output; no output publication |
| Fake POWERED evidence or changed eligible count | Cannot promote; pinned manifest/ledger validation rejects tampering |
| n<2 | Explicit unassessable conditional model, no invented MDE |
| Invalid finite domain or root/search failure | Reject invalid inputs; cap reports unresolved rather than sufficient |
| Output existing/alias/input/protected path | No overwrite or protected access |

</frozen-after-approval>

## Code Map

- New `tools/valentini_power_gate.py`: typed pure conditional-model functions and metadata gate CLI.
- New `tests/unit/test_valentini_power_gate.py`: model and input/output/integrity firewall tests.
- New `docs/valentini-power.md`: commands, interpretation and scope; parent owns results and inventory notes.
- Reuse `tools/valentini_reclaim.py` protected_path/strict JSON/hash helpers and `tools/valentini_native_audit.py` safe_input/child/directory-descriptor publish if useful; import no decoder, trader or native replay. Do not alter either existing module.
- Existing `_bmad-output/valentini-native-20260914/run-final` measurement: manifest SHA25623f2205518d7be751fa147274441efe0af89c2a2aec52f76a8810e443a4b1667. Validate all six artifact bytes before and after metadata parsing/model calculations; parse only report.json, sessions.json and provenance.json. No histogram/snapshot/observed-bar rows parsed.
- `_bmad-output/valentini-power-20260914/inventory.json`: scoped metadata-only native inventory, already committed. Bind its current exact SHA in code; include this trusted evidence in output, not arbitrary user booleans.

## Tasks & Acceptance

- [x] Implement CLI --audit-dir --prereg --inventory --output-dir, requiring new output directory; pin the above input hashes and emit canonical JSON with code/dependency/input hashes and prereg commit. No promotion option. Valid terminal report exit0; bad input exit2.
- [x] Read registration and implement exact noncentral-t power, bracketed80%-MDE, minimum-n search, and separate known-variance normal cross-check. No borrowed effect estimate; fixed scenario grid0.1/0.2/0.3/0.5/1.0. Normalized net session outcomes are hypothetical, not returns produced by this code. All outputs finite or explicit null/unassessable.
- [x] Validate pinned metadata/count consistency, eligible booleans, unique nonoverlapping session IDs, counts vs report and per-session summaries. Preserve original exclusions and verify provenance code hashes against current repo files. Data-source bytes were already audited; record that no new raw-native decode/reconciliation occurs here. Existing trading admission remains NOT_ADMITTED.
- [x] Include session-based MDE frontier, all hypothetical effect scenarios, assumptions, blockers, and next required evidence. The six profiles are not independent trades or a calibrated effective sample size. The iid-normal assumption is a scenario, not a demonstrated bound.
- [x] Test independent chi-square integral oracle for nct power (synthetic n7/d0.4 and n12/d0.7, avoiding actual-ledger calculations), alpha under null, monotonicity, minimum-n minimality, MDE target inversion, invalid/nonfinite types, data/schema tampering, artifact mutation, output collisions and structural no-price/no-trade calls.
- [x] Write concise docs and exact command. Parent runs numerical gate only after reviewed script/tests committed, verifies outputs and writes results.

Acceptance criteria:
- Given a committed registration, when a valid gate runs, then all prespecified scenarios are reported with immutable evidence bindings and no strategy result is calculated.
- Given synthetic normal outcomes, when power is evaluated, then it agrees with independently integrated chi-square theory within tight numerical tolerance, and required integer n satisfies the adjacent-n boundary.
- Given tampering or missing data, when the CLI runs, then it refuses before publishing a model result and preserves inputs.
- Given absent independent calibration, when any hypothetical cell exceeds80%, then the operational verdict remains POWER_UNDETERMINED with evaluation_allowed=false.

## Implementation Notes

Parent owns pre-registration/inventory and actual results; implementation agent owns new code/tests/docs only, no commits. No new approval needed: user explicitly proceeded. The existing template xsmom1 is conceptual only; its strategy-specific effect constants and flawed identity-only bootstrap guard are not reusable evidence. No market resampling is needed or authorized for this sample-count feasibility calculation. The prereg defines normal-theory assumptions explicitly; do not invent dollar costs or translate standardized effects to dollars/trade/years.

## Spec Change Log

## Review Triage Log


Review execution: the host refused additional fresh reviewer threads. Two agents independent of the power implementation performed the three lenses: a fresh blind reviewer also ran verification as a separate pass, and the earlier native implementation agent reviewed power edge cases. All lens assignments were sent before triage; the blind and verification contexts were shared. This is a review-isolation limitation, not three independent reviews.

| ID | Verdict | Route | Evidence and action |
|---|---|---|---|
| B1 conservation object | low | patch | Synthetic repinning can reach AttributeError on null conservation; direct object validation restores the documented structured refusal. Fixed production manifest prevents this malformed input today. |
| B2 calendar duration | low | rejected | Standalone validator accepts mutually adjusted synthetic counts, but production hashes bind the audited calendar and ledger before validation. Re-implementing calendar scheduling adds branches for an input no caller can supply without changing source pins. No wrong production n demonstrated. |
| B3 status vocabulary | low | rejected | Standalone validator accepts invented synthetic labels, but version and fixed manifest bind the exact audited statuses. Adding a second status-schema implementation for repinned test fixtures is unnecessary complexity in this fixed-evidence gate. |
| B4 committed provenance | medium | patch (parent evidence) | Runtime report copies a known prereg commit but does not prove repository history. Registration explicitly delegates reviewed-code commit before actual run to the operator; parent will verify committed bytes and record exact code/test commit before execution. No new git subprocess surface in gate. |
| B5 df=1 oracle | low | patch | Existing independent oracle covers n7/n12, while n2 is an actual frontier boundary. Add a synthetic n2 oracle point near target without calculating on the six-session ledger. |
| B6 code/runtime tests | medium | patch | Separate final integrity comparisons are unexercised. Add controlled digest/runtime mutation tests without changing installed dependencies. Same root cause as V3; retained as separate row. |
| B7 transitive sentinel test | low | rejected | Direct structural scan cannot prove arbitrary transitive behavior. Current helper implementations were read and do not decode/trade; opaque invalid row bytes in end-to-end tests separately prove those rows are not parsed. No actual forbidden transitive call was found. A comprehensive behavioral sandbox would add unrelated complexity. |
| E1 provenance snapshot race | medium | patch | A helper file can change after metadata validates its digest but before the result's code snapshot, producing contradictory historical/current hashes that remain stable until publication. Compare the snapshot's provenance subset to pinned provenance immediately. |
| E2 nonregular input | medium | patch | safe_input permits a FIFO/device and bound_bytes reads it before hash verification. Reject nonregular files before reading prereg/inventory/manifest bytes. |
| V1 eligible-count wiring | medium | patch | End-to-end assertions omit observed n and its powers; replacing eligible count with total ledger length could pass. Assert synthetic n7 and its calculations. |
| V2 frontier completeness | medium | patch | Tests check individual inversions and empty frontier only; dropping final positive-n row would pass. Assert exact positive frontier count/sequence and observed endpoint. |
| V3 code/runtime tests | medium | patch | Pre-verified missing mutation coverage for final code/runtime comparisons; same defect as B6, separate row retained. |
| V4 refusal JSON | medium | patch | Refusal tests check exit/no output but not stdout/stderr JSON contract. Capture a representative malformed input and assert fields and streams. |

## Verification

Run the new unit suite with the repository interpreter, Black/flake8 and default strict mypy if installed. Do not use --follow-imports=skip because it discards typed helpers; do not add overrides. Parent rechecks the unchanged existing119 native+simulator tests when integrating. Actual CLI result generated only after code/prereg commits; independently check minimum sample boundaries and all model/operational verdict distinctions.

Pre-run implementation verification: 214 focused tests passed; Black, flake8 and strict mypy passed. Reviewed fixes are implemented. Actual ledger run remains pending code commit.

Completed after reviewed-code commit a9509494b8f6b20456f75824c94833e05e188c98: actual metadata gate returned POWER_UNDETERMINED/evaluation_allowed=false. Independent integration verified all five frontier cells and all five scenario powers; exhaustive integer checks confirmed minimum counts620/156/71/27/8. All model/code/input hashes matched; pre-run committed-byte verification and final evidence are retained. No strategy outcomes calculated. No findings deferred.
