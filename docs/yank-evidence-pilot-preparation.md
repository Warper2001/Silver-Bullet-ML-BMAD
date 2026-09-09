# YANK evidence pilot preparation

**HOLD_VALIDATION.** Engineering preparation only. No observation sessions, trades, acquisitions, authentication, installation, service restarts, deployment, push or merge have been performed by this preparation. Frozen strategy, snapshots, audit results and P&L remain authoritative and unchanged.

## Local historical preparation

Run these offline commands from the validation worktree with the repository Python environment:

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/research/yank_deployed_validation/tradestation.py ledger
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/research/yank_deployed_validation/tradestation.py validate path/to/local-archive.json
```

The ledger proposes April 1 through May 30, 2025 inclusive, with May 19–30 evaluation separated from warm-up. Seven-day UTC windows overlap one minute. The default 60-second spacing is a conservative preparation policy. Documented limits are 500 requests per five minutes and 57,600 bars; see the [official-documentation semantics report](yank-tradestation-semantics.md). It does not send requests. Provider contract spelling, expiration, session calendar, timestamp interval semantics, entitlements and costs must be independently resolved before acquisition. Continuous symbols are prohibited. The ledger includes GET `/v3/marketdata/barcharts/{symbol}` and `interval`, `unit`, `firstdate`, `lastdate` query values. Half-open planning boundaries do not assert provider boundary inclusion; that remains an explicit gate.

The validator accepts a JSON list of envelopes. Each has `request` fields `request_id`, `symbol`, `interval`, `unit`, `first_inclusive`, `last_exclusive`, `started_at`; `response_metadata` fields `received_at`, `http_status`, `raw_sha256`; and `response` containing the original provider `Bars`. Retain optional server time/version, chunk identity, precision, adjustment policy and request hash. Each envelope must include `raw_response_base64` containing the actual credential-free raw response bytes. The validator checks their SHA-256 against `response_metadata.raw_sha256` and checks decoded JSON equality against `response` before awarding coverage credit. Missing or altered bytes receive no coverage credit; hashes alone do not prove provider provenance. Never include authorization headers, cookies, tokens or account identifiers.

Diagnostics preserve original row order and flags, revisions, duplicates, gaps and both interval-label candidates. `IsEndOfHistory` describes historical delivery; it does not close an open bar. Calendar gaps include market closures until a pinned exchange calendar distinguishes missing sessions. For usable calendar coverage, supply `--expected-calendar independent-calendar.json`: an independently prepared object with `symbol`, unique aligned `minute_starts` within the archive window, and optional independently verified `label_semantics` (`start` or `end`). Never derive expected minutes from observed bars. The report separately compares warm-up/evaluation against both interval candidates; without supplied semantics, admission remains UNKNOWN. Partial, revised, malformed, wrong-contract and integrity-failed rows receive no coverage credit. A PASS establishes only coverage of that supplied calendar; full H1 ATR/LR/daily-range readiness remains a separate blocker. Historical acquisition receipts never establish arrival at decisions made in 2025.

## Conditional replay bridge

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/research/yank_deployed_validation/tradestation.py bridge path/to/local-archive.json --output-directory path/to/fresh-bridge
```

The optional `--expected-calendar` argument uses the same independent calendar as validation. The bridge requires raw-verified usable bars from exactly one explicit contract. It emits `start/` and `end/` candidate directories, each with a hash-pinned `manifest.json`, `polls.jsonl`, separate `warmup.jsonl` and `evaluation.jsonl`, and the preserved archive-validation report. These manifests and streams pass the existing replay admission interface without changing the replay engine.

Each candidate replays all warm-up followed by evaluation in one continuous synthetic state. The separate files mark the assessment boundary; do not restart from an empty buffer for evaluation. Candidate bars use normalized start labels and synthetic interval-end receipt times, retain raw references, and never assert actual 2025 arrival or poll history. The computational receipt interval extends one minute beyond the archive end solely to include the final completed minute. Account/risk initialization is explicitly `SYNTHETIC_UNKNOWN_ACCOUNT`; parameters here are diagnostic inputs, not observed account state. Coverage admission is distinct from feature readiness. No candidate establishes interval semantics, live identity or trading validation.

## Next maintenance window

The disabled configuration in `deploy/yank-observation.disabled.json` is a proposal, not a runnable deployment manifest. A separately approved maintenance window must identify the exact service, operator, release and rollback artifact before any installation or restart.

1. Independently build and distribute expected collector/runtime/configuration/model identities and the startup checkpoint expectation. Pin trusted public keys outside capture bundles. Keep production private keys and pseudonym salts outside repositories and captures. A signature proves collector provenance, not host integrity.
2. Confirm the loaded runtime objects match that release while polling is quiescent. Reject mismatches before wrapping methods. Verify effective contract/quantity, model, strategy configuration and process start identity. Do not use a caller's trust flags to upgrade legacy captures.
3. Resolve the supported account integration before enabling. The observer wraps the underlying ProjectX `_http` boundary, including mirror clients inheriting that interface. Account/order/position endpoints produce non-atomic snapshots; separate successful replies alone cannot establish a coherent decision-time account state. Keep that verdict UNKNOWN unless every actual reply carries the same snapshotId; current ProjectX response schemas do not supply this coherence evidence. Preserve deployed reconciliation behavior.
4. Check the bounded queue, storage worker and failure coverage under representative startup-size responses with both shadow features. Record exact trading-decision equality, peak queue/memory, latency distribution, dropped records and thread identity for disk writes. Any overflow, writer/serializer failure, sequence discontinuity or restart invalidates coverage.
5. Install only through the verified startup entry point after the approved gates are satisfied. Start a fresh process-bound evidence interval and retain its actual initial checkpoint. Stop at a quiescent poll boundary, drain storage, sign the bundle and verify it offline against independently supplied expectations.
6. Roll back by invoking the returned restoration/close path at a quiescent boundary. If a service release was separately installed, restore the approved prior release/configuration using the recorded maintenance procedure. Verify original callables and ordinary reconciliation behavior. Preserve failed captures as diagnostics; never overwrite evidence or frozen results.

## Two-session observation protocol

Choose two actual, separately approved sessions after maintenance approval; dates remain unset. Session 1 establishes capture completeness and runtime/account coverage under ordinary polling. Session 2 repeats verification independently, including a new process identity if restarted. Do not trigger trades, manufacture fills, retune parameters or train models for the protocol. Record only actual existing provider callbacks and ProjectX execution events, with venue attribution.

For each session retain: approval/window identifier; UTC start/end; release and collector hashes; public-key identifier; process identity; checkpoint/configuration/model hashes; contract and pseudonym; actual request/receipt boundaries; queue/storage metrics; coverage and signing results; decision comparison results; actual acknowledgement/fill/cancel references; quantity/tick differences; unresolved account/ordering gaps; and rollback verification. Missing or inconsistent evidence stays UNKNOWN. Two sessions alone do not establish strategy validation or profitability.

## Cost and entitlement worksheet

| Field | Prepared value / evidence required |
|---|---|
| Incremental spending authorization | $0 |
| Contract spelling and expiration | UNKNOWN; provider verification required |
| Historical endpoint and boundary convention | Verify against official documentation before acquisition |
| Rate and maximum bars/request | Documented 500 requests per five minutes; 57,600 bars; see linked semantics report. Entitlements and operational applicability still require confirmation |
| Historical entitlement | UNKNOWN; credential-free entitlement record required |
| Per-call/per-row/subscription charge | UNKNOWN; quote and approval required |
| Planned calls | Number of ledger windows; retry allowance remains unset |
| Estimated incremental total | UNKNOWN until applicable costs and retry policy are confirmed |
| Acquisition approval and operator | UNSET |
| Maintenance approval, window and rollback release | UNSET |
| Actual sessions/trades | NONE recorded by this preparation |

Engineering test results belong in the implementation evidence/report. Operational gates, unknown data semantics and unavailable live evidence remain explicit even when local tests pass.

## Collector API and independent trust inputs

The maintenance integration calls `startup.prepare_observation(trader, enabled=False)` by default. A separately approved, quiescent installation supplies `enabled=True`, `expected_release`, a fresh `output_dir`, an external Ed25519 `signer` object implementing `sign(bytes)`, `key_id`, external `pseudonym_salt`, and optional `capture_limits` containing only positive `capacity`, `max_bytes`, `max_nodes`. No key-loading or authentication path is provided. The returned `ObservationSession.close()` checks owned bindings and the process/runtime/configuration and collector files before restoring its original methods, drains storage, and writes `evidence.json` plus `manifest.json`. Close only between polls. Failed installation returns disabled with an error; a setup exception closes the failed capture and restores original bindings.

Independent release engineering must pin the following JSON fields; never create the expected values from the process seeking admission:

- `runtime`: `methods` mapping (loaded Python method/code/default/closure fingerprints), `effective_configuration_sha256`, and `model_sha256` (SHA-256 of canonical joblib model hash). Includes fitted model, threshold, strategy/LR settings, symbol, quantity, tick, shadow/data flags and module constants. Build pins using approved loaded release objects and exact dependency versions; source file equality alone is insufficient.
- `collector_sha256`: mapping of `startup.py`, `evidence.py`, `account.py`, `capture.py`, `adapter.py` to file SHA-256.
- `snapshot_identity`: the fixed reference identity from `capture.identities(approved_snapshot)`; this describes reference files, never process identity.
- `checkpoint_sha256`: SHA-256 of canonical complete `Adapter.checkpoint()`; must independently match the decision boundary to be admitted.
- `account_pseudonym`: HMAC-SHA256 of the actual ProjectX client's account ID with the external salt; `contract`: exact ProjectX contract ID; `account_max_age_seconds`: positive approved freshness ceiling.
- For offline verification also add `process`: `{ "boot_id": "<kernel boot UUID>", "pid": 123, "start_ticks": 456 }`, independently transferred from the approved process-start receipt. These are placeholders, not an observed process.

Offline `--trusted-keys` JSON maps independently provisioned key IDs to base64 raw 32-byte Ed25519 public keys. `--expected-release` holds the independent release/checkpoint/process/account pins above; `--evidence` names the signed bundle. The bundle never supplies its own trusted key. The capture file is streamed into a SHA-256 digest at close; account requests/replies stay in that signed file, avoiding an unbounded in-process copy of session records. A signature proves collector provenance, not host integrity. Memory-only runtime configuration/model and wrapper-ownership checks bracket every poll. Collector-file and full process-start checks run only at quiescent startup/close. The final COMPLETE manifest is published last; failed signing/publication cannot admit a capture. Unknown closures or unsupported loaded objects fail the installation gate.

The optional collector verifier requires `cryptography` (verified locally with **49.0.0**) and the repository-compatible pinned model dependencies. Provision it in a separate approved collector environment during release preparation; do not alter the installed strategy environment here. The disabled startup and legacy diagnostic imports do not import cryptography. No dependency installation occurred in this preparation.

Account PASS requires successful complete fresh actual Account/search, Order/searchOpen and Position/searchOpen replies with a common provider snapshot ID. The inspected ProjectX schema does not provide this atomic consistency evidence; ordinary observed responses therefore remain UNKNOWN, including apparent flat responses. Request-start markers make missing or in-flight replies explicit. The legacy ProjectX reconciliation result, including its error-to-FLAT fallback, remains unchanged.

The disabled proposal records collector-only limits `capacity=1`, `max_bytes=64000000`, `max_nodes=4000000`. A real private-poller 2880-bar backfill with both existing shadow paths retained all 2880 transition/label records in 10,975,592 bytes; measured poll-and-close was 71.69 seconds on the shared offline host. This is functional sizing evidence, **not production latency qualification**. See `docs/reports/yank-evidence-preparation/observation-48h-benchmark.json`. Default smaller bounds are still tested for permanent coverage invalidation. JSONL payload writes occur only on the storage worker; startup creates its directory and quiescent close writes summaries/signatures on the maintenance caller. Polls perform no filesystem writes in the new observer.

The collector records wall-clock boundaries immediately before each actual decision call. Offline account admission uses the final decision boundary (receipt for no-decision polls), requires fresh coherent replies, and rejects account requests/replies occurring within the poll/decision interval. Checkpoint position/pending state and signed quantity must agree with account observations. This conservative rule does not upgrade apparent flatness from the deployed reconciliation fallback.
