---
title: 'Complete and consolidate the YANK pilot'
type: 'feature'
created: '2026-09-08'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="user supplied implementation plan">

## Intent

Restore only the frozen audit module, command, tests and documentation from commit 84f2382ed238ff66f42c470bf8021c0fd7e4e4e2 into feat/yank-native-minute-pilot. Adapt fixed local input paths and use the private loader. Preserve hashes, settings, five cases, thirty scenarios, both timestamps, three delays and the 240-opportunity lifetime. Consolidate audit, provenance, corrected replay and May 28 evidence with HOLD_VALIDATION. Run 48 audit, 83 minute/book and 133 legacy tests plus import isolation; reproduce the audit twice, compare findings and hashes, verify unchanged archives and corrected replay, save evidence and commit locally. No acquisition, tuning, invented fills, revised frozen P&L, push or merge.

</frozen-after-approval>

## Implementation Notes

No intent gaps or irreversible actions. Small adaptation of an existing audited module, plus restored historical artifacts and completion evidence. Acquisition and bars live in the main checkout; archive and frozen engine remain in the sibling replay checkout. Reuse the native-minute private package loader. Preserve historical report bytes; save integrated results separately. Protect the relocated input tree from output writes. No strategy or native-minute source changes.

Restored audit and native-minute suites passed together: 132 tests (48 historical audit, one command isolation, 83 minute/book). Frozen replay checkout independently passed 133 tests. Existing Pydantic deprecation warnings only. Optimized-Python published-ledger verification retained both arms' accounting results. Full audit runs use separate fresh output directories; verification compares full reports except code hashes, not merely outcome totals.

## Review Triage Log

- false: Main-checkout output restriction is intentional: the authorized command runs in the isolated minute worktree, and the main checkout holds protected inputs. Updated stale reproduction documentation to the isolated checkout.
- false: Adding SDK package dependencies is outside this fixed-environment integration; the existing Databento environment is explicitly required and used by both pilot commands.
- Integration review: no actionable findings. Completion report and verifier review: no substantive findings. No deferred work.

Final verification passed: both fresh runs completed with byte-identical canonical artifacts; all complete findings and 240-opportunity schedules match the original, including 11 supported and 19 unassessable scenarios. Only report code hashes changed. All input/archive hashes, original saved repeat, corrected native artifact pairs, native source and follow-up files remain unchanged. Evidence: docs/reports/yank-pilot-completion/verification.json. HOLD_VALIDATION retained.
