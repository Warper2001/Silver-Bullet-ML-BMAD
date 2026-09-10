# Published versus deployed MIM research

See [the historical assessment](RESULTS.md) and [verification record](VERIFICATION.md). Generated immutable run directories remain local artifacts and are excluded from Git.

This package diagnoses exposed MNQ history. Historical results, sizing results, and prospective results never authorize deployment. It imports no production strategy, broker client, or study script. Independent source evidence is under [evidence](evidence/).

Run from the repository with `.venv/bin/python`; NumPy and pandas are required. Every CLI invocation creates a unique directory under `research/mim_comparison/runs/`. Inputs, configuration, protocol and research sources are hashed before return computation. Research source snapshots preserve uncommitted implementations. Completed artifacts have SHA256 inventories and read-only modes; exclusive creation prevents accidental overwrite. This is tamper detection, not a signature or protection against a privileged administrator.

```
.venv/bin/python -m research.mim_comparison audit --data data/mim_x/mnq_1min_by_contract.csv --labels end
.venv/bin/python -m research.mim_comparison historical --data data/mim_x/mnq_1min_by_contract.csv --labels end
.venv/bin/python -m pytest tests/unit/mim_comparison -q
```

The `end` declaration means offset-aware 09:31–16:00 ET minute completions. The author uses 09:30–15:59 start labels; `--labels start` adds one minute. Quarterly MNQ codes supply explicit contract identity, with third-Friday expiry inferred conservatively: the expiry-date session is unavailable. This is documented inference, not exchange metadata. Continuous contractless data and existing contractless operational bars are refused. A dated, independently established contract mapping must be added to a new research input before it can qualify; never mutate operational logs.

Prior full-session volume chooses the next unexpired contract, with nearest expiry breaking ties. The winner is selected before inspecting its current-session completeness. Missing winner data never triggers fallback. Duplicate minutes, partial/early closes, nonfinite/invalid OHLCV, and unknown intervening weekday closures are excluded. Entire absent weekdays are reported as missing-or-exchange-closed, with no invented holiday calendar. Sigma uses 14 complete prior selected sessions and dimensionless moves across rolls; the prior close always belongs to the selected contract.

| Specification requirement | Implementation / evidence |
|---|---|
| Independent author and deployed references | `references.py`; original-source vectorized exposure semantics; `evidence/deployed_fixture_audit.py` extracts unchanged live AST methods into an in-memory broker fixture; `evidence/recorded-indicator-audit.json` reports serialized operational indicator agreement |
| A/B/C/D and mechanisms | `engine.py`: guarded deployed/published, unguarded deployed/published, gap-only, confirmation-only, neutral-exit-only |
| Signal/execution separation | Half-hour decisions; primary i+2 open, optimistic i+1; stop signal-close±250 active only after fill; adverse gaps use worse open; EOD16:00 close proxy |
| Guard | Realized gross one-contract reference P&L at signal/stop references, ≤−1000 blocks reentry; actual modeled fills/costs remain separate |
| Costs and accounting |2.24 primary,3.24 and6.24 round trip, half per contract-side; reversals charge two sides; eligible no-trade days zero |
| Historical diagnostics | `daily.csv`, `ledger.csv`, `decisions.csv`, report with paired expectancy, drawdown, yearly outcomes, turnover/exposure and largest-winner removal |
| Mechanism attribution | Each single change minus A; interaction B−A minus all singles; disclosed first eligible-day author sigma-depth difference |
| Separate sizing | Published unguarded signal,100000 initial equity,.02 target,4x pre-round leverage ceiling, multiplier2, Python bankers rounding; daily returns[d−15:d−1],ddof1; NaN→4x; own-equity1x comparator |
| Prospective collection | Streaming first observations in SQLite, FULL synchronous commits before next row; replay/correction immunity; finite sandboxed invocation |
| Final inference |20,000 stationary bootstrap draws, seed7, mean block5 with10/20 sensitivity;95% percentile intervals; failure/support/inconclusive thresholds below |
| No deployment | Every protocol and decision explicitly denies deployment authorization |

Sizing is an MNQ adaptation, not SPY replication. The4x author ceiling applies before integer rounding; actual notional leverage can exceed4x after rounding, and `sizing.csv` reports it. Reference vectorized author close-difference returns are diagnostic; common engine fills and dollar costs are authoritative for arm comparisons. Historical ledger costs show the primary scenario; `daily.csv` contains all three costs computed from actual turnover.

`evidence/recorded-indicator-audit.json` is an independent rounded-log reconciliation. Exact action/broker parity cannot be reconstructed from rounded contractless operational files. Historical runs freeze the decision log snapshot they analyze. A threshold ambiguity envelope preserves uncertainty from six-decimal sigma and two-decimal bands; serialized agreement is never presented as full-precision equality.

Prospective protocol and operation:

1. Produce and inspect a historical run, including the historical120-session minimum detectable increment estimate. The approximation assumes80% power, two-sided5% alpha and Bartlett lag5 long-run variance; underpowering is disclosed without changing the horizon.
2. Provide a research-only, offset-aware CSV with `contract,timestamp,open,high,low,close,volume,received_at`. It must contain first-observed bars; operational contractless files fail. Include all candidate contracts needed for prior-volume selection. Audit the warmup source first; no purchased acquisition or broker requests occur here.
3. Run `shadow` to collect currently available observations under the historical run’s frozen protocol. Start is the first complete ET session opening strictly after that original freeze. Historical warmup must be the exact audited source. Actual collector time and receipt time must both fall within60 seconds of the completed bar. Historical replay never earns prospective eligibility.
4. Invoke the same command as needed, passing `--state FIRST_SHADOW_RUN/collector` on subsequent calls. The original freeze, nine-calendar-month deadline, source hashes and warmup hash remain binding. Each call writes a new immutable output run and snapshots the durable journal. Do not modify research code during collection: source drift fails closed.
5. Invoke `evaluate` on a completed shadow run. It verifies inventory hashes, frozen warmup and implementation, replays only immutable first observations, and checks modeled decisions against the decisions committed during collection. No efficacy is returned until120 eligible paired sessions have accumulated. Missed/late/invalid or silent sessions are unavailable for both arms; weekdays without verified closure metadata are conservatively excluded. No automatic extension occurs at nine months.

```
.venv/bin/python -m research.mim_comparison shadow --data /absolute/research/feed.csv --labels end --warmup /absolute/audited/history.csv --historical-run research/mim_comparison/runs/HISTORICAL_RUN
.venv/bin/python -m research.mim_comparison shadow --data /absolute/research/feed.csv --labels end --warmup /absolute/audited/history.csv --historical-run research/mim_comparison/runs/HISTORICAL_RUN --state research/mim_comparison/runs/FIRST_SHADOW_RUN/collector
.venv/bin/python -m research.mim_comparison evaluate --shadow-run research/mim_comparison/runs/LATEST_SHADOW_RUN
```

Shadow uses Linux bubblewrap with a new user/mount/network namespace, an empty environment, selected read-only Python runtime/source/input mounts, and only isolated research output/state writable. Production paths and host home credentials are absent. An x86-64 seccomp filter rejects socket/socketpair syscalls. Missing sandbox support fails closed; no service starts. Inspect shadow failures/status, session exclusions and journal decisions for operational monitoring, without computing interim efficacy.

Final support means the lower95% CI for paired B−A is greater than5 dollars per eligible session and B's lower CI is above zero, with highest-cost B and incremental point estimates positive. Failure means primary incremental upper CI below5 or B upper CI below zero. Everything else is inconclusive. Any categorical conflict under block10/20 also makes the result inconclusive. Insufficient coverage is incomplete/inconclusive. Support authorizes only consideration of further validation.

Current available history ends before the present prospective freeze. Historical output and source evidence must not be mistaken for future120-session results. Operational contract identity and fresh timely warmup/feed coverage remain prerequisites, not assumed facts.

The governing experimental freeze comes from the compatible historical run's protocol, sealed before historical return computation. Launching or resuming collection later does not move that freeze or its nine-month deadline. The original historical manifest, execution settings, power estimate, source, and warmup remain bound. Growing shadow CSVs are described as streams; their durable first observations and raw invalid-row tombstones, rather than a whole-file hash, establish what the collector consumed.

Malformed first observations are journaled before proceeding. Identifiable invalid rows permanently exclude their session; unidentified rows preserve file/line evidence and disqualify the collector state because their session cannot be established. Corrections never restore eligibility. A session also needs all durable eligible half-hour decisions: completing an out-of-order set of390 bars cannot erase an earlier unavailable decision. Finalization waits through the60-second close grace period unless all valid bars and decisions are already complete. Collection stops after120 eligible sessions or the original deadline.

Historical and final runs include `paired-daily.csv` with A, B and B−A for both timings and all three cost scenarios. Historical `scenario_contrasts` reports corresponding paired means and mechanism interactions. Shadow exports durable `decisions.csv`, `eligibility.csv`, `ledger.csv`, and raw `daily.csv` outcomes; finalized session outcomes are persisted once. These are operational records, without interim efficacy conclusions. Every CLI output has a readable `report.md`; unavailable/not-applicable ledgers retain column schemas. Final `report.md` states the decision, coverage, confidence intervals and deployment prohibition.

Ledger `event_timestamp` remains the completed bar label. `modeled_fill_timestamp` for market fills is that label minus one minute, and EOD uses the close label. Intrabar stops have no fabricated exact timestamp: their fill is located only within the event minute and `fill_time_basis` says so. The deployed reference intentionally preserves the source quirk that an opposite entry can follow a rejected exit; controlled broker fixtures document this behavior. The `full_notional` comparator remains1x of its own previous equity, matching the author. Author sizing also preserves the missing first daily return, so its day15 volatility slice is NaN and falls back to4x.

## Existing-project feed adapter

See the [verified feed integration status and concrete poll commands](FEED_STATUS.md).

The [contract feed adapter](feed_adapter/README.md) reconciles `data/mim_nb/bars_raw.csv` with the timestamped TradeStation request and DATA-context evidence in `logs/mim_nb_live.log`. It adds inferred contract provenance in isolated research output, preserving original event and receipt timestamps. Missing or ambiguous evidence is excluded; earlier replay cannot earn prospective eligibility. No production files or services are changed.

The adapter is a separate package so it does not change the frozen trading engine, historical result, or original prospective deadline. Its own source/configuration and consumed input prefixes are frozen separately. Its output can be passed to the existing sandboxed `shadow` command with the original historical run and warmup. Recent operational bars are not added to historical performance. Warmup gaps and incomplete roll coverage remain subject to the existing common eligibility checks; the adapter cannot manufacture missing candidate-contract volume or prior closes.
