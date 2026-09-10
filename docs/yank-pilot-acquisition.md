# YANK pilot raw acquisition

The separate `src.research.yank_deployed_validation.acquire` CLI collects the fixed nine MNQM25 one-minute windows from April 1 through May 31, 2025 exclusive. Adjacent windows overlap by one minute. It preserves `HOLD_VALIDATION`; acquisition does not establish exchange-calendar coverage, timestamp semantics, historical arrivals, account state or strategy readiness.

Only run acquisition after independently confirming **zero incremental USD cost**, exact contract and endpoint entitlement, and approval. The module checks operator-supplied declarations and evidence references before reading a token or opening a network connection. These declarations are not cryptographic provider proof. A missing gate returns `BLOCKED`, exit code 2, with specific blockers and creates no acquisition directory.

## Gate file

Supply a JSON object with these fields:

```json
{
  "credential_sha256": null,
  "symbol": "MNQM25",
  "endpoint": "https://api.tradestation.com/v3/marketdata/barcharts/MNQM25",
  "incremental_cost_usd": 0,
  "contract_verified": true,
  "entitlement_verified": true,
  "acquisition_authorized": true,
  "evidence_references": {
    "contract": {},
    "endpoint": {},
    "zero_cost": {},
    "entitlement": {},
    "approval": {}
  }
}
```

The top-level `credential_sha256` must contain the lowercase SHA256 of the exact reviewed access token; each evidence record must bind to the same digest. This binds account-specific operator review to each dispatched credential without publishing token values. Token rotation requires new review and a fresh run. Every empty evidence object above must be replaced with a reviewed record containing `reference`, `reviewed_by`, `reviewed_at` (timezone-aware ISO timestamp, no more than 60 seconds ahead of the injected current clock), `symbol: "MNQM25"`, and the exact `endpoint` URL above. References should identify retained independent evidence and its applicable account/request scope. Do not place credentials in gate files. This deliberately incomplete example cannot authorize acquisition. Calendar and bar-label interpretation are separate validation gates and are not required to preserve raw responses.

## Invocation and authentication

From `/root/Silver-Bullet-ML-BMAD-yank-validation`:

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m src.research.yank_deployed_validation.acquire \
  --gate /path/to/reviewed-gate.json \
  --output-directory /path/to/new-acquisition-directory \
  --token-cache /path/to/read-only-token-cache.json
```

The default JSON cache is `~/.tradestation/token_cache.json`, with `access_token` and timezone-aware `expires_at` fields. The module reads it immediately before each attempt and refuses missing, malformed, expired or near-expiry tokens (60-second margin). It never imports or calls refreshing authentication code and never modifies a token cache.

For the current runner's raw `.access_token` format, replace `--token-cache` with `--token-file /path/to/.access_token`. JWT `exp` is parsed locally only to reject expired/unknown expiry; this is **unverified JWT metadata**, not signature verification or entitlement proof. Opaque raw tokens additionally require `--token-expires-at <timezone-aware-ISO-timestamp>` established independently. A later explicit timestamp cannot override an expired JWT claim. No browser login, token refresh or subscription action occurs.

GET requests use only `https://api.tradestation.com/v3/marketdata/barcharts/MNQM25` and the existing deterministic ledger's queries. Redirects and environment proxy configuration are disabled. Each request has a cancellable 60-second total deadline covering connection and full body download. A synchronous wrapper runs the async HTTPX fetch under `asyncio.timeout`, awaiting cancellation and client cleanup without leaving background network work. The entire acquisition has a one-hour budget; it stops incomplete if required spacing/retry delay plus a full request deadline exceeds the remaining budget, never retrying early. The spacing is at least 60 seconds between attempts; transient network errors and HTTP 408/429/500/502/503/504 receive at most two retries per window, giving at most 27 attempts. `Retry-After` numeric seconds and HTTP dates may extend the delay. Sleeps are split into at most 60-second chunks. Authentication, entitlement, payment and other HTTP rejections stop immediately.

## Evidence and recovery

A fresh output directory is mandatory. Acquisition writes these files atomically without replacing existing files:

- `plan.json`: unchanged fixed ledger, gate SHA256, and explicit operator-declaration classification. The original gate remains with its operator; free-form evidence declarations are not copied into the archive.
- `may2025-NNN-attempt-N-started.json`: request identity, exact window/query, attempt number and request start time, durable before network use, with file and directory entries fsynced.
- Matching `-result.json`: receipt time, status, parsed body, raw bytes as base64 and their SHA256; or `-error.json` with a safe failure category. Response headers and exception strings are never persisted. Safe derived pagination/extra-field flags, Retry-After presence and computed delay, budget-exceeded flag, and run deadline are retained. Nonfinite JSON values preserve exact raw bytes with `response: null` and a parse diagnostic; escaped surrogate strings remain safely representable.
- `status.json`: `COMPLETE`, `INCOMPLETE` or `BLOCKED`, completed request count and reason. An ungraceful process kill may leave this file absent; absence never means success.
- `archive.json`: compatible with `tradestation.validate_archive`, written only when all nine responses satisfy the narrow raw-response checks. `COMPLETE` means all nine raw requests completed; it does not assert complete market coverage.

Pagination indicators, extra top-level fields, missing required bar fields, malformed bar objects, empty arrays, or redirects stop acquisition and preserve the attempted response. The collector never follows arbitrary pagination URLs or discards pages. Bar completion flags, timestamp interpretations, revisions and invalid OHLCV remain preserved with per-envelope validator diagnostic counts; they do not turn transport-complete HTTP replies into missing raw evidence. In particular, the collector does not assume REST responses end with a true `IsEndOfHistory` stream marker. Additional validation should use the existing archive validator and conditional replay bridge; unresolved calendar/labels remain unresolved.

Detected credentials (the active token, credential fields, bearer strings or JWT-shaped text) cause the whole response body to be omitted, explicit redaction/invalidation to be recorded, and acquisition to stop. Redacted records deliberately have no raw hash/base64 claim. Detection cannot identify every unlabeled arbitrary secret, so responses and provider fields remain an operational review boundary.

Interrupted attempts retain their start records and prior results; graceful interruption writes an incomplete status. The collector has no automatic resume: preserve the entire directory, inspect completed attempts, and use a fresh directory for a separately reviewed run. Temporary files from interrupted atomic writes are evidence, not completion. Re-running the CLI against any existing output path fails before token use.

## Verification and benchmark scope

`tests/unit/yank_deployed_validation/test_acquire.py` exercises the gates, fixed requests, overlap/query preservation, exact raw bytes, retries, interruptions, pagination/malformed replies, token refusal, and redaction using injected transports and clocks. Tests never use real credentials or network.

`tests/unit/yank_deployed_validation/test_startup_benchmark.py` covers the full verified `prepare_observation` startup with 2,880 bars, independent unobserved baseline and both shadow paths. Its retained machine-readable results should be read with the root acquisition report. The benchmark checks decisions/state, code pins, writer/disk boundaries, queue bounds, signed completion and failure invalidation. Shared-host latency is descriptive and does not certify production performance. Actual provider findings and service/rollback proposals belong in `docs/reports/yank-pilot-acquisition`; nothing in this CLI installs or activates collection.
