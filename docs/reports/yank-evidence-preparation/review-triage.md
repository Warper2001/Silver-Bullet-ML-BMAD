# Preparation review triage

Two reviewers ran with fresh context (blind and edge-case review). A platform thread limit prevented a fresh verification reviewer; the implementation agent explicitly performed that layer as a self-review. This is not represented as three independent reviews. All layers reported before triage. Findings are engineering defects or test gaps, not new operational authorization.

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

All confirmed findings were corrected and verified within the authorized preparation scope. Root also caught filesystem reads in the initial per-poll guard correction: the final guard is memory-only during polling, with a regression test that traps filesystem/process-file checks. Original strategy and audit defects are not changed. The existing SL5-versus-SL2 regression remains separately documented. Final test evidence and completion state are recorded in the findings report.
