# Prospective decision observation: installation proposal

Decision: HOLD_VALIDATION. No hooks were enabled on the installed trader, no live collection was launched, and no service, authentication, order or trading operation was performed. Enabled observer tests use private synthetic traders only. DecisionCapture defaults to enabled=False. The new _decision_capture attribute is separate from settled _shadow_logger feed parity and bullish _shadow_trade_logger; their existing consumers remain unchanged.

## Concrete integration

capture.py provides install_poll_observer(trader, capture, identity=..., readiness=..., state_reader=...). When disabled it immediately returns a no-op rollback function without touching the trader. When explicitly enabled after a future reviewed installation, it wraps the exact existing _poll_and_process, its HTTP client response reader, _detect_and_enter, _log_filter_decision, model predict_proba and execution intention methods. It records the original method results and returns them unchanged. A context-local datetime proxy records every original datetime.now call during the poll, including the pre-HTTP poll clock and each post-response staleness clock; other datetime operations delegate to the original object. No initialization is invoked by the helper.

The reviewable future call site is after trader.initialize() completes and before start_streaming(), using the already-constructed trader. The capture configuration must remain separate from signal/execution parameters. Construct a dedicated read-only state reader using the Adapter.state contract; it must capture every state field and accepted-buffer identity. For example, the offline-tested reader shape is:

```python
reader = object.__new__(Adapter)
reader.trader = trader
reader._buffer_cache_key = None
reader._buffer_hash = None
reader._warmup = {}
observer = DecisionCapture(enabled=False)  # proposed configuration remains disabled
rollback = install_poll_observer(
    trader, observer,
    identity=verified_identity,
    readiness={"equivalent_feed_and_state": False},
    state_reader=reader.state,
)
```

Do not claim identity from a copied filename. Before any future enabled run, record loaded source/model/config identity, effective runtime overrides, service/drop-in identity, and explicit signal/execution venue. The current on-disk snapshot alone cannot authenticate the running process. Capture an explicit complete checkpoint, including accepted bars, sweep and M15 latches, expiry clocks, active/pending state, daily risk and session histories. Pin its redacted account-evidence source hash and receipt time. Missing identity, clock, state or feed evidence must leave equivalent_feed_and_state false. The comparator rejects incomplete observed checkpoints rather than assuming flat state or available funds.

Each poll record includes safe hashed request identity, raw Bars in original response order, response status, actual receipt time, original per-call clocks, source/model/YAML/threshold/LR/service hashes, before/after decision state, consumed-history readiness, filter/model outputs and order intentions. Returned execution-method values are retained separately as observed method replies so offline replay can use actual order IDs; they do not establish fills. The helper hashes the request URL and does not retain headers, tokens or account IDs. Capture payload validation refuses credential-named fields and rejects oversized/custom objects.

ProjectX acknowledgements/fills and TradeStation SIM mirror observations must be emitted only from actual observed response/fill callbacks via DecisionHooks.broker_observed. Call immediately after receiving and validating the provider response, retaining a safe receipt time, order identity, observed quantity/price and venue. Do not manufacture an acknowledgement from a local simulated fill or cross-attribute TS SIM evidence to ProjectX. These callbacks are an installation proposal; no live callback has been changed here.

## Bounds and invalid coverage

The writer is a separate thread with a bounded queue (default 64 records), per-record byte/node limits and a cumulative per-poll observation budget. Hot-path producers never wait for the queue lock or disk. Full queues, contention, oversized inputs, observation failures and writer failures permanently invalidate coverage; subsequent apparently good rows cannot erase the gap. Once the per-poll budget is reached, further observation is dropped without changing the original trading method. Disabled mode creates no directory/thread. Close emits a coverage summary with accepted/written/dropped counts, failure reasons and a capture hash. Missing or invalid coverage makes comparison unassessable. Storage hangs are bounded at shutdown; an unfinished writer cannot produce valid coverage.

Before enabling, use offline fault injection to verify that blocked storage, queue overflow, logger errors and malformed responses preserve the original result/state, and measure observer overhead on startup-sized responses. The bounded observer still performs CPU work to copy/hash state; overhead evidence is required before deployment. Capture buffers and worker lifecycle are independent of the two existing shadow features.

## Offline comparison

Pin capture.jsonl and coverage.json in an explicit manifest containing capture_sha256, coverage_sha256 and initial_state. Synthetic initial state must be labeled SYNTHETIC_UNKNOWN_ACCOUNT. Observed initial state requires the complete checkpoint and account_evidence; omitted risk/position fields cannot default to zero/flat.

```
/root/Silver-Bullet-ML-BMAD/.venv/bin/python src/cli/compare_yank_decision_capture.py \
  --manifest /path/to/capture-manifest.json \
  --input-dir /path/to/finished-capture \
  --snapshot docs/yank-validation/snapshot/v1 \
  --output-dir /tmp/NEW-yank-capture-comparison
```

This replays raw receipt order through the offline adapter, including future 2026+ captured dates, rather than comparing only precomputed traces. It checks reference-file identities, ordered poll sequences and predecision state. Reference-file hashes and caller-supplied readiness flags do not authenticate live loaded code, model or account state. Only the private pinned synthetic runtime can currently receive an engineering match; live observations remain unassessable pending independent identity/account attestation integration. Recorded original clock calls and execution-method replies are consumed in order; exhausted/unused evidence is unassessable. Exact discrete decisions, quantities and tick prices are compared independently from floating feature/probability differences. Feed/state differences, missing clocks, bad coverage or poll errors are unassessable; a matching synthetic constant-clock test is not prospective provider evidence. Floating feature differences are reported as a qualified discrete match; risk/configuration values require exact agreement. The already admitted poll is replayed without inventing a second scheduler decision. Explicit non-200/timeout envelopes retain failed observations. Venue observations are reported separately, with no inferred fills or economic promotion.

## Rollback proposal

Call the returned rollback closure to restore original bound methods, logger and datetime object and remove only _decision_capture. Then close the dedicated capture worker and preserve its coverage status/artifacts. Restore the proposed disabled configuration. Do not remove settled parity/bullish loggers or alter strategy parameters. A service restart, trading change, deployment or enabled collection would require the separately approved installation plan; none has occurred in this task.


## Retained offline example

`docs/reports/yank-deployed-validation/shadow/` contains a synthetic three-poll entry/fill/SL-exit capture, its complete initial checkpoint, coverage, manifest and CLI comparison. Two CLI runs produce byte-identical comparisons with zero state/decision/feature differences. Each result is nevertheless `UNASSESSABLE: account_evidence_unverified`: the account declaration is deliberately a negative control, not an authenticated account. The companion test exercises a privately seeded synthetic state and verifies exact MATCH, qualified feature differences, and missing-reply refusal.

To replay that example, pass `--manifest docs/reports/yank-deployed-validation/shadow/manifest.json --input-dir docs/reports/yank-deployed-validation/shadow` to the comparator command above, with a fresh output directory. It does not connect to a broker. Labels retain original values plus UTC-normalized start/end candidate intervals, both marked unverified. Response hashes distinguish bounded raw HTTP bytes (when observed) from canonical parsed Bars; absent raw bytes remain unknown. Coverage applies only to the accepted prefix before close; an active producer during shutdown invalidates it.

## Monitoring and acceptance before a future run

After separately approved installation, keep strategy settings unchanged and first verify the disabled call site creates no capture directory, thread or attribute. Validate the loaded runtime/account attestation design independently; the current comparator intentionally cannot certify a live declaration. Measure capture overhead using the full startup response and both existing shadow features before enabling. Select explicit capacity, record byte/node limits, local output destination, retention limit and operator stop criteria in the approved configuration. Reserve disk space based on measured capture bytes per poll times planned polls; logger storage pressure must never drive a strategy change.

During an approved run, monitor queue drops, invalid reasons, accepted versus written records, worker health, disk space and poll latency outside the trading decision path. Stop observation on any missing clocks, overflow, writer failure or unexplained latency; mark the entire affected capture interval invalid and retain evidence. Rotate only at a quiescent poll boundary: rollback the observer, close the writer, verify the coverage hash and counters, pin the complete ending checkpoint, and start a separately identified segment. A restart requires a new capture identity, complete checkpoint, observed recovery/broker state and explicit discontinuity; never stitch segments as uninterrupted coverage.

At capture verification, pin both capture and coverage files, check receipt and sequence order, reconcile all broker observations by venue, and replay through the offline comparator. Any account/source/clock ambiguity or capture gap blocks an equivalence claim. Rollback restores only observation wrappers and closes the writer at a quiescent boundary; preserve captures and existing settled/bullish shadow logs. A short run or exact local comparison cannot establish profitability.
