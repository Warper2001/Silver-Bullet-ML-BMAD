# Proposed maintenance and observation

**PROPOSED, NOT APPROVED. HOLD_VALIDATION.** This task performed no installation, service restart or live collection. The exact proposal is also available as [JSON](maintenance-proposal.json).

## Window and current service

The September 12 candidate window expired without approval. No replacement window is proposed while the release, operator, key, checkpoint, latency and rollback gates remain unresolved. A future proposal must confirm broker maintenance availability and current [CME Micro E-mini trading hours](https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html) before assigning dates.

Earlier read-only inspection found `trader-yank.service` active, PID 455819, started September 4 at 20:36:45 UTC. The September 10 observer-overhead inspection instead found PID 1247178, started at 19:35 UTC. This task performed only a read-only `systemctl show`; it did not restart the service. Its working directory is the main checkout and its command invokes `src/research/yank_streaming_working.py`. The installed unit has `TIER2_DEBUG=1`, absent from the checked-in template, and configures MNQU26 with two contracts. Its comment anticipates a September roll; verify the actual approved live contract at the window. Do not change contracts automatically or copy the historical MNQM25 symbol into live configuration.

At the September 10 inspection, the on-disk strategy differed from the preserved validation snapshot because of existing diagnostic edits. Those bytes were later recorded in repository history outside this task, and subsequent commits changed the checked-in service contract copy to MNQZ26. This task did not edit or restore the installed strategy. Current loaded-process identity and the relationship between the running unit and current HEAD remain UNKNOWN. See the [earlier service fingerprints](service-inspection.json) and [September 10 read-only check](../yank-observer-overhead/installed-preservation.json). The old rollback candidate and release expectations require renewed comparison and independent review; they cannot be treated as current release evidence.

## Measured observer readiness

The [observer report](../yank-observer-overhead/README.md) retains three fresh-process counterbalanced repetitions before and after the candidate change. Guarded median poll time changed from 76.113 to 73.828 seconds for 2,880-bar startup and from 3.355 to 3.010 seconds for 30 incremental polls after a 7,500-bar warm-up. Shared-host variation and the private Clock cache bypass limit attribution; these results do not establish accepted production latency.

The tested proposal remains capacity 1, 64,000,000 bytes per queued record and four million nodes. All official guarded timing cells retained complete queue/write coverage, but a separate unpaced fixture dropped two of three records and correctly failed coverage. Queue limits are not a whole-process memory limit; observed guarded startup RSS approached 333MiB. Timing packages are not admission evidence; separate signed synthetic correctness packages also retain account UNKNOWN. The current collector inventory is a candidate, not an approved release.

## Approval packet and responsibilities

The existing service owner is the proposed operator; a named person's acceptance must be recorded before approval. The operator owns service commands and rollback. An independent release reviewer supplies loaded runtime/model/configuration expectations from the approved release/dependency environment and verifies checkpoint expectations. A key custodian supplies the signer and pseudonym salt externally, and delivers the trusted public key separately to the offline verifier. These roles are responsibilities to assign, not claims that anyone has accepted them.

The [rollback candidate](rollback-candidate.tar.gz) preserves the inspected installed unit and selected strategy/config/model bytes. [Verification](rollback-verification.json) checks all members. It is not a full host backup or a substitute for current persisted state. Recheck every source hash immediately before the window; if anything changed, regenerate and review the candidate before proceeding.

No runnable production launcher is approved. The release proposal uses a separate observer launcher so the original source can remain unchanged: construct the ordinary trader and perform its ordinary initialization once, call `prepare_observation` after initialization and before `start_streaming`, and arrange quiescent close on normal shutdown. Review this launcher with the concrete runtime and shutdown behavior before installation. Do not append initialization calls or extract a checkpoint from the process seeking admission and treat it as independent proof. A crash or forced kill produces incomplete evidence, never a COMPLETE session.

## Proposed sequence after approval

1. **Preflight:** Verify named operator, exact unit/source/model/config/dependency pins, contract-roll decision, independently prepared runtime/checkpoint expectations, external signer/public-key distribution, fresh capture directory, full startup benchmark and explicit production-latency acceptance. Missing evidence means no enablement. The offline benchmark does not itself establish acceptable live latency.
2. **Quiescent installation:** Preserve current state using existing operator procedures and recheck the rollback artifact. Keep credentials and live state outside the repository/captures. Install only the reviewed collector overlay and separate launcher using an approved service change; retain original unit bytes. No strategy or shared configuration reset.
3. **Approved start and verification:** Verify the actual new process receipt, object identity, complete checkpoint and guard installation. Reject a mismatch before wrappers are installed. Check storage health and event coverage; account UNKNOWN remains explicit and prevents claims of admitted live comparison.
4. **Reserved rollback:** If any gate fails, close at a quiescent boundary, retain failed evidence, remove the observer service change and restore the inspected original unit through the operator's approved restart procedure. Restore source/config/model only if this approved installation changed them and their current hashes still match the installation record; never overwrite unrelated concurrent changes. Verify ordinary service behavior. Reserve enough approved window time for rollback rather than rushed enablement.

## Evidence sessions

The September 14 and 16 candidate sessions are withdrawn because the prerequisite maintenance window expired without approval. After all gates are satisfied, propose two new evidence intervals: one for initial plumbing, timing and coverage inspection and one for independent repeat verification. Dates do not authorize a stop, start or restart. The reviewed launcher must support quiescent observer session boundaries before any schedule is proposed.

Record actual request/reply/decision boundaries, process/release/checkpoint pins, coverage failures, resource metrics and signed completion. Preserve ordinary strategy behavior, including sessions with no trades. Retain actual broker acknowledgements/fills/cancels only when available. Close and compare in receipt order with independent keys/release pins. Report exact decisions/quantities/tick prices, features and missing account/execution evidence separately. No trade is manufactured and no result promotes HOLD_VALIDATION.
