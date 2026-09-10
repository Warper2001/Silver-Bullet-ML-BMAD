# YANK pilot acquisition and maintenance findings

September 10, 2026 — **HOLD_VALIDATION**. The acquisition tooling, provider investigation, full startup benchmark and maintenance proposal are complete. Actual historical acquisition remains blocked by missing account-specific zero-cost/entitlement evidence. No installation, restart or live observation was performed.

## Completed work and actual findings

- Added a separate raw acquisition CLI with credential-safe evidence gates, fixed nine-request ledger, bounded retries, interruption records and lossless response handling. See [usage and gate format](../../yank-pilot-acquisition.md).
- Made **one authorized authenticated read-only symbol request**. TradeStation confirmed MNQM25, June 2025 Micro E-mini Nasdaq-100, CME, expiration June 20, tick 0.25 and point value 2. [Exact response evidence](symbol-verification.json). This does not prove historical entitlement or availability.
- Completed the [cost/entitlement and provider report](provider-findings.md). Existing credentials were inspected read-only; no refresh or shared credential-file write occurred. General subscription pricing does not establish this account's incremental cost. The pending gate correctly returned BLOCKED before token/network use and created no acquisition directory; [result](acquisition-gate-check.json).
- Inspected the active service without changing it. Its installed unit contains `TIER2_DEBUG=1`, absent from the repository template. The on-disk strategy matches the preserved snapshot despite being dirty relative to main HEAD; loaded-process identity remains unverified. [Fingerprints](service-inspection.json).
- Prepared a [maintenance proposal](maintenance-proposal.md) for September 12, 14:00–15:00 UTC, a verified [rollback candidate](rollback-verification.json), and proposed sessions September 14 and 16. These dates and operations are **not approved or scheduled**.

## Full startup benchmark

The final benchmark passed with 2,880 synthetic provider-shaped bars, both existing shadow paths, equal final state and equal 2,637 actual filter-decision records. It exercised the complete prepare/guard/close/sign/verify lifecycle. Signature, identity, checkpoint and coverage passed; account remained UNKNOWN and overall admission remained ineligible. The signed invalid-coverage negative control stayed ineligible.

| Measurement | Observed result |
|---|---:|
| Baseline poll |80.354s|
| Guarded poll |133.908s|
| Difference |+53.554s /1.666×|
| Observer preparation /close |0.062s /0.078s|
| Capture bytes |10,975,591|
| Captured transitions |2,880|
| Baseline /guarded process high-water RSS |274,216 /352,732KiB|

This is a material measured overhead finding, not an isolated production estimate. Both modes used identical passive profiling on a shared host; the benchmark is one sample per mode. JSONL writes occurred on the capture worker, with no poll-thread accesses through the monitored file APIs. Monitoring is not a kernel syscall audit. The feed, account identity, release reference, signing key and shadow sinks are synthetic; no trades were generated.

The full benchmark used capacity 16, max_bytes 64,000,000 and max_nodes 4,000,000. Its configured queue byte ceiling is 1,024,000,000, not an observed memory peak. This differs from the earlier disabled proposal's capacity 1; **neither the benchmark's capacity 16 nor production latency is approved**. Qualification against the actual approved production limits remains a maintenance gate. See [machine-readable metrics](startup-benchmark.json) and [final log](startup-benchmark.log). Two development harness runs were superseded: an overly broad dependency pin and a wrong dependency path were corrected before the final passing run.

## Verification and review

The final post-review broad regression run passed **633 tests**, with only the known SL5-versus-SL2 failure; [log](final-regressions.log). The focused acquisition suite passed all 58 cases, and the full startup benchmark passed separately. The preserved configuration remains SL2; no parameter change was made to satisfy the pre-existing SL5 expectation.

Three fresh-context reviewers identified raw JSON retention, credential-context binding, durable publication, review-time validation, retry auditability, deadline/budget and token-expiry verification issues. All confirmed issues were corrected and verified; [individual triage](review-triage.md) records each finding. The full benchmark's retained results close the artifact-retention finding.

The existing 177-pin preservation check passed without mismatches. Final checks also verified 49 prior preparation/collector/snapshot files, six installed source/config/model/unit files and unchanged service process identity; [evidence](preservation-final.json). Repeated legacy comparisons were byte-identical to each other and the prior result, remaining ineligible; [evidence](legacy-repeat-verification.json). Native replay code and its preserved outcomes remain unchanged; no May2025 provider replay result is invented when no archive exists.

## Remaining evidence and operation gates

1. **Acquisition:** Provide an existing-subscription record or provider confirmation tying expired MNQ historical API access to this account at no additional charge. No bar requests were sent; historical availability, label semantics, calendar coverage and warm-up therefore remain unassessed. The approved zero-cost-only boundary persists.
2. **Account admission:** ProjectX's published separate REST replies and user-hub events do not establish the required cross-entity atomic snapshot/replay barrier. Account remains UNKNOWN. No new live subscription or relaxed admission was introduced.
3. **Maintenance:** Named operator acceptance, reviewed launcher/release, independent runtime/checkpoint and key inputs, actual live-contract verification, production limits/latency acceptance and window approval remain required. The rollback candidate is selected on-disk material, not a full host/state backup or loaded-process attestation.
4. **Observation:** The two sessions remain proposals. After approved installation, retain only actual polling/execution evidence. Never force trades, infer missing fills or promote HOLD_VALIDATION from engineering results.

No purchases, subscription changes, service changes, retraining, parameter search, holdout access, push or merge occurred. The main checkout and frozen P&L were preserved.

Engineering commit: `f0d741b1e280b70944c587f0ba809693510bc3f6`. The [evidence inventory](evidence-inventory.json) pins retained artifacts and the acquisition/benchmark sources. Public documentation and one symbol response are actual observations; historical bars and live session evidence are absent.
