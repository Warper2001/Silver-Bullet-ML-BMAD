---
title: YANK zero-cost pilot acquisition and maintenance proposal
type: feature
created: 2026-09-10
status: done
route: dispatch
review_loop_iteration: 0
baseline_commit: f2528491bcc4bead748eb01b0caf5a909abf91ea
context: []
---

<frozen-after-approval reason="User approved execution of the conversational plan">

## Intent

Complete the next YANK evidence phase: acquire the April–May 2025 pilot only after independently confirmed zero incremental cost, validate/replay acquired evidence where possible, and prepare exact maintenance approval without installing anything. Missing external evidence is a reportable blocker, never an inferred success.

## Boundaries & Constraints

Always work in /root/Silver-Bullet-ML-BMAD-yank-validation. Preserve HOLD_VALIDATION, installed strategy, original snapshots, prior reports and frozen P&L. Existing credentials may support authenticated verification and confirmed-zero-cost acquisition. No purchase/subscription changes, service installation/restart, live collection, push/merge, retraining or holdout access. Never expose credentials or refresh/write the running trader's shared token cache. Root owns public research, actual operational inspection/acquisition decisions, maintenance report and commits. Implementation agent owns code/tests and must not perform authenticated calls. The root's operational research is independent of the local engineering work.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Missing cost/entitlement/symbol evidence | Stop before token/network use; report gate |
| Confirmed zero-cost plan | Nine fixed ledger GET requests; preserve raw bytes and metadata |
| Transient error / rate limit | At most two retries per request, spacing >=60 seconds, honor Retry-After |
| Auth/entitlement rejection or unknown charge | Stop, preserve diagnostics, no token refresh/subscription action |
| Interrupted run | Keep completed per-attempt records; never overwrite prior evidence; no false COMPLETE archive |
| Provider pagination or partial/malformed response | Preserve evidence and reject completeness; never silently drop pages |
| Credentials in response/error | Do not persist tokens or headers; explicit redaction/invalidation, no valid raw claim |
| Full startup collector integration | Compare decisions/state against unobserved runtime with both shadows, record runtime overhead and bounded queue/failure evidence |

</frozen-after-approval>

## Code Map

- src/research/yank_deployed_validation/tradestation.py: existing deterministic ledger, validate_archive, export_replay; keep offline imports and existing interfaces intact. Ledger covers April1–May31 exclusive, evaluation May19–May31 exclusive, nine windows with one-minute overlap, candidate MNQM25. New separate acquisition CLI/module should reuse ledger/envelope format.
- src/data/tradestation_auth.py: cache under ~/.tradestation/token_cache.json; get_valid_access_token auto-refreshes and writes shared cache. Do NOT call it. Reuse cache format read-only via explicit token provider that refuses expiry; no new browser flow.
- src/data/tradestation_client.py: existing normalized historical client uses different bars endpoint, 100000 limit and70day chunks, so do not reuse its downloader or change it. Use narrow httpx GET transport with redirect disabled and fixed official host/path.
- src/research/yank_deployed_validation/startup.py and tests/unit/yank_deployed_validation/test_evidence.py: verified runtime setup/integration fixtures. test_observation_benchmark.py has direct observer sizing only; add full verified-startup benchmark in a separate file to compare complete guard overhead.
- docs/reports/yank-evidence-preparation: immutable prior evidence, root writes new findings in docs/reports/yank-pilot-acquisition. Root prepares service/rollback proposal separately.

## Tasks & Acceptance

- [x] src/research/yank_deployed_validation/acquire.py: separate explicit CLI and testable injected transport/clock/token-provider entry point. Gate using evidence references pinned to exact contract, endpoint, zero incremental USD cost, entitlement and approval; evidence declarations must be transparently operator-supplied, not cryptographic provider proof. Fixed nine ledger requests, bounded retries, fresh output directory, per-attempt atomic files, raw base64 and SHA256 compatible envelopes, final archive only when all complete. Preserve failures and report unknown pagination without following arbitrary URLs. Do not require solved bar-label/calendar semantics to collect raw evidence.
- [x] tests/unit/yank_deployed_validation/test_acquire.py: mocked tests covering matrix, actual ledger overlap/query, JSON raw preservation, pagination detection, Retry-After seconds/date, timeout/auth errors, interruption, missing/expired cache and redaction.
- [x] tests/unit/yank_deployed_validation/test_startup_benchmark.py: full prepare_observation integration benchmark with representative2880 startup bars and both shadow paths, independent baseline, same decisions/state, code pins, disk-access/writer boundaries, configured queue bounds and valid signed completion; report descriptive shared-host latency and limitations, no production certification. Run once after code stabilizes; retain machine-readable metrics for root. Use separate existing fixtures or narrow helpers, avoid editing protected adapter/source.
- [x] docs/yank-pilot-acquisition.md: CLI usage, gate format, auth behavior, acquisition outputs/recovery and benchmark scope. Root adds actual findings, evidence blockers and concrete maintenance/session proposal.

Acceptance: Given unresolved evidence, acquisition performs no network and reports precise blockers. Given mocked eligible replies, all ledger boundaries/raw bytes survive validate_archive without claiming historical arrivals. Given the fully guarded collector, measured decisions/state remain unchanged and failures invalidate coverage. Actual acquisition is conditional on independent cost evidence; absent evidence is an accepted external blocker, not engineering completion.

## Implementation Notes

Use /root/Silver-Bullet-ML-BMAD/.venv/bin/python. No intent gaps remain; external facts stay explicit gates. Default retry budget is two per window, maximum27 attempts; rate wait may exceed60 seconds but orchestration should remain communicative. Unknown timezone/labels/calendar do not become guessed admission. Root will review diff, run relevant regressions and preservation, and create local conventional commits.

## Spec Change Log

## Review Triage Log

## Verification

Run new acquisition tests and existing validation/capture suites. Record complete startup benchmark separately. Existing SL5-versus-SL2 failure must remain unchanged and separately reported.

### Final implementation and verification

Completed engineering and conditional execution investigation. One authenticated metadata GET verified MNQM25; no historical bar requests were made because account-specific entitlement and zero-cost evidence remain unavailable. The actual pending gate was run and blocked before token/network access. Maintenance and two actual-session dates are proposals only.

58 acquisition tests pass;633 broad regression tests pass with the known preserved SL5-versus-SL2 failure. Full startup benchmark passed: baseline80.354s, guarded133.908s; equal state and2637 filter decisions,2880 transitions. Guarded overhead and larger capacity16 fixture mean production settings/latency remain unqualified.177 prior pins,49 additional prior files and6 installed files passed preservation; service process unchanged; two legacy comparisons matched prior. See docs/reports/yank-pilot-acquisition/README.md for all evidence and blockers.

# Review triage

Three fresh-context agents reviewed the change: blind, edge-case and verification-gap. A concurrency limit delayed the third launch until a slot opened; all findings were collected before triage. Corrections remain within the approved acquisition/evidence boundaries.

| ID | Verdict | Evidence and resolution |
|---|---|---|
| B1 | high | Nonfinite parsed numbers fail JSON publication after a reply arrives; preserve raw bytes with explicit representation diagnostics. Confirmed independently with a NaN response. |
| B2 | medium | Lone surrogate strings fail the credential scanner's UTF8 encoding; handle them explicitly without mislabeling a response as transport failure. |
| B3 | medium | File fsync without directory fsync does not support the durable-before-network claim; durably publish directory entries. |
| B4 | high | Cost evidence is account-specific while a mutable token provider can change credentials; require an approved credential digest and compare it before every attempt. |
| B5 | medium | A future review timestamp parses and passes; reject future reviews with a bounded clock-skew allowance. |
| B6 | medium | Retry/control decisions were not retained; persist safe derived control flags and retry timing, excluding unrestricted response headers. |
| B7 | medium | HTTPX inactivity timeout is not a total response deadline; add a cancellable total deadline. |
| B8 | medium | Unbounded Retry-After can wait effectively forever; stop when its required delay exceeds the remaining one-hour run budget, never retry early. |
| B9 | medium | Reviewed diff preceded benchmark completion and lacked the final report; retain the final actual log/metrics and findings before completion. |
| E1 | medium | Same total-deadline defect as B7; same correction, separate finding retained. |
| E2 | medium | Same durable-directory publication defect as B3; same correction. |
| E3 | high | Overflowing JSON numbers share B1's raw-publication defect; include exponent-overflow coverage. |
| V1 | medium | Existing tests do not advance a real token provider through expiry after successful acquisition/retry; add both paths with evidence retention and unchanged token-file assertions. |
| R1 | high | Root reproduced NaN response as transport_failure with only a partial .tmp result; include exact raw-byte retention assertion in B1 correction. |

All listed corrections are complete. Final verification:58 focused acquisition cases pass,633 broad regression cases pass with the unchanged known configuration failure, and the full startup benchmark passes. Actual benchmark artifacts and final outcomes are retained in README.md. No finding authorizes subscription changes, acquisition with unverified cost, installation or live collection.
