---
title: YANK decision evidence and TradeStation pilot preparation
type: feature
created: 2026-09-09
status: done
route: dispatch
review_loop_iteration: 0
baseline_commit: a478d674e8d507928cec843b139f086d14d85296
context: []
---

<frozen-after-approval reason="User explicitly requested implementation of the supplied plan">

## Intent

Prepare defensible decision-time evidence for YANK and the May 2025 TradeStation pilot in the isolated validation worktree. External declarations currently cannot establish account or loaded runtime identity. Add a trusted-collector admission path, retain legacy captures as diagnostics, and prepare disabled prospective observation and historical archive tooling.

## Boundaries & Constraints

Always preserve HOLD_VALIDATION, installed strategy, original snapshots, archived audit results and frozen P&L. Work only in /root/Silver-Bullet-ML-BMAD-yank-validation, starting at a478d67. Prepare local conventional commits; no push or merge. Production signing secrets are external to repositories and captures. Trusted public keys and release expectations arrive independently at offline verification. Signatures prove collector provenance, not host integrity.

Never authenticate, acquire paid data, spend, deploy, restart, collect live observations, retrain, search parameters or access holdouts. Installation, entitlements and costs remain explicit operational blockers. Keep ProjectX deployed reconciliation behavior unchanged.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Trusted evidence | Valid signed bundle and independently pinned runtime/config/model/checkpoint | Separate signature, identity, checkpoint, account and coverage verdicts | HOLD_VALIDATION remains |
| Tampering | Altered/unsigned bundle, untrusted key, substituted model/config, process restart, wrong checkpoint, sequence discontinuity | No live evidence upgrade | Explicit failed or unknown verdict |
| Account uncertainty | Failed/partial/stale/inconsistent account responses, missing replies | Unknown, never inferred FLAT | Preserve response boundaries and redacted fields |
| Capture failure | Overflow, serializer/writer failure, missing records | Invalid coverage; original trading result unchanged | Bounded queue and no trading-path disk writes |
| Historical bars | Partial/future bars, conflicting flags, revisions, duplicate/missing minutes, DST, overlap | Preserve raw labels, BarStatus and IsEndOfHistory; both interval candidates | Report ambiguity separately |
| Warm-up | Separate April 1–May 30 archive and May 19–30 evaluation | Explicit insufficient warm-up/coverage status | Acquisition receipt is never 2025 arrival evidence |

</frozen-after-approval>

## Code Map

- src/research/yank_deployed_validation/adapter.py: pinned private runtime, canonical serialization and complete Adapter.state contract; reuse existing replay and do not modify snapshots.
- src/research/yank_deployed_validation/capture.py: bounded asynchronous DecisionCapture, disabled default, install_poll_observer rollback and raw bar evidence; add verified startup integration separately.
- src/research/yank_deployed_validation/compare.py: offline replay and exact/feature differences; extend with independent trusted evidence arguments, keep legacy behavior.
- src/research/yank_deployed_validation/replay.py: existing bar admission to inspect and extend as needed.
- src/research/projectx_client.py and installed snapshot source: inspect underlying HTTP response schemas and runtime/model/config fields; do not change deployed reconciliation.
- tests/unit/yank_deployed_validation and tests/research/test_yank_deployed_capture.py: existing preservation and capture cases.
- docs/reports/yank-deployed-validation: immutable prior results; new findings go in a new directory.

## Tasks & Acceptance

**Execution:**
- [x] src/research/yank_deployed_validation/evidence.py and startup.py — implement versioned Ed25519 signed bundles, runtime-object verification against pinned release before wrapping, effective model/configuration and checkpoint binding, process start and ordered capture binding; disabled startup entry point and rollback. Reject caller-supplied identity upgrades.
- [x] src/research/yank_deployed_validation/account.py — observe redacted underlying ProjectX request/response boundaries, pseudonym, contract, status and available balance/orders/positions. Fail closed for incomplete, stale or contradictory observations without altering reconciliation results.
- [x] src/research/yank_deployed_validation/compare.py and CLI — separate offline admission verdicts and integrate verified evidence into replay eligibility; receipt-order reconciliation, exact quantity/tick/decision comparisons distinct from features and unavailable evidence.
- [x] src/research/yank_deployed_validation/tradestation.py and a local-only CLI — generate seven-day one-minute-overlap request ledger for April 1 through May 30 with May 19–30 evaluation, rate schedule and explicit symbol verification gate; archive validation with raw response/request metadata and separate warm-up/evaluation coverage. No network or auth commands execute.
- [x] tests/unit/yank_deployed_validation — cover all matrix scenarios and preserve existing deterministic replay behavior. Benchmark startup-sized responses with both existing shadow features and record decision equality, bounded memory, failure coverage and storage-thread behavior.
- [x] docs/yank-evidence-pilot-preparation.md and deploy/yank-observation.disabled.json — prepare next-maintenance-window release/rollback and two-session observation protocol, cost/entitlement worksheet and execution evidence fields; no fabricated sessions or trades.

**Acceptance Criteria:**
- Given legacy captures, when caller flags assert trust, then verifier cannot promote their identity/account evidence.
- Given enabled startup with mismatched loaded runtime objects, when observation installation is requested, then no wrappers are installed and the mismatch is reported.
- Given trusted signed complete evidence, when replay comparisons run, then independently computed admission verdicts permit eligible live comparisons without declaring strategy validation.
- Given dry-run preparation, when commands run, then outputs are deterministic and no credentials, network, service changes or paid acquisition occur.

## Implementation Notes

No intent gaps for preparation. Acquisition and deployment details are intentionally unresolved operational gates. Root agent independently researches official TradeStation documentation and writes a cited semantics report; implementation agent need not duplicate web research. Root agent verifies final diff and runs preservation checks. Implementation agent owns the code, tests and operational package tasks above, may delegate bounded testing work if useful, and reports gaps candidly.

## Spec Change Log

## Review Triage Log

| ID | Verdict | Verified evidence and resolution required |
|---|---|---|
| B1 | high | Close restores wrapped bindings before checking them; per-poll method checks are excluded. Check ownership/loaded originals before restoration and invalidate unexpected substitutions. |
| B2 | high | Function hashes omit referenced defining-module globals; imported helper behavior can change without bytecode changing. Include relevant loaded global/helper bindings. |
| B3 | high | `MetaLabelingFilter.predict_proba` reads effective `FEATURE_COLS`; the initial runtime pin includes model/threshold but omits that class/instance setting. Bind inference configuration. |
| B4 | medium | The positive account fixture contains reply rows without starts; verifier checks outstanding starts but not missing starts. Require matched boundary pairs. |
| B5 | medium | Signing/publication exception changes only returned coverage; persisted coverage can remain valid. Invalidate persisted coverage and use final manifest publication as completion. |
| B6 | high | Close has no idempotence/ownership guard; repeated rollback can overwrite a later observer. Make repeated close inert and preserve other owners' bindings. |
| B7 | medium | Parsed maximum/minimum datetime values overflow interval arithmetic outside the malformed-row handler. Retain a finding rather than abort diagnostics. |
| B8 | medium | Duplicate keys include symbol, while revision suppression initially uses timestamp alone. Keep contract scope in suppression so rejected contracts cannot erase selected-contract coverage. |
| B9 | medium | Boolean IsClosed is checked but malformed supplied values are ignored. Reject malformed supplied completion indicators from coverage credit. |
| B10 | high | Startup success test closes an empty session and never consumes its emitted files. Add a nonempty startup/poll/sign/file-comparison test. |
| E1 | high | Runtime class traversal ignores property accessors, including risk properties used by the trader. Pin accessor implementations. |
| E2 | high | Pending account state is compared only as PENDING; quantity/side/entry-price mismatches are not checked. Compare corresponding pending exposure fields, with ambiguity UNKNOWN. |
| V1 | medium | Restart fixture aliases expected process into the signed payload, so mutation tests signature failure instead of identity rejection. Deep-copy expectations and assert each verdict. |
| V2 | medium | Stale fixture sets receipt before request, failing chronology even with freshness disabled. Test valid chronology at both sides of freshness threshold. |
| V3 | high | Per-poll guards and emitted bundle usability have no startup integration consumer test. Cover actual guarded polling, persisted outputs and model/config drift; same integration root cause as B10. |
| V4 | medium | No assertion consumes captured decision_times. Assert one ordered boundary per decision transition and account-change rejection at that boundary. |
| R1 | high | Root inspection found continuous sequence numbers can coexist with reversed receipt times. Add chronological consistency and receipt-order validation without reordering evidence. |

## Verification

Use /root/Silver-Bullet-ML-BMAD/.venv/bin/python for repository-compatible dependency versions. Run new and relevant existing YANK suites, deterministic replay/capture checks, and preserve baseline manifests. Report the existing SL5-versus-SL2 failure separately if reproduced. Report engineering complete versus evidence blockers versus operations awaiting approval.

### Completed verification

Final relevant regressions: 575 passed; the single pre-existing SL5-versus-SL2 failure remains unchanged. All four observation benchmarks passed. Two full native replay runs were byte-identical to each other and the preserved prior result; repeated legacy comparison remained unassessable. Preservation checks found zero mismatches across 177 prior pins and 42 local protected files. All review findings above were corrected; two fresh-context reviews and one explicitly disclosed implementation self-review were used. Engineering preparation is complete; HOLD_VALIDATION and all acquisition/installation gates remain. Detailed evidence and limitations: docs/reports/yank-evidence-preparation/README.md.
