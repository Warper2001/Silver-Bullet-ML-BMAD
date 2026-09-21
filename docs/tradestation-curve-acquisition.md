# TradeStation commodity coverage acquisition

The completed v3 pilot and its remaining source requirements are summarized in the [v3 readiness assessment](tradestation-curve-v3-readiness.md).

The [2025 coverage and reported-activity analysis](tradestation-curve-feasibility-2025.md) measures pair availability, basic volume/open-interest screens, missing observations and repeated closes using the saved pilot.

This read-only workflow measures accessible daily contract bars for the twelve-root reference universe. All output is development/feasibility evidence. **HOLD-DATA** remains until settlement provenance, historical calendars and the other [data-contract requirements](../_bmad-output/specs/spec-commodity-curve-carry/data-contract.md) are resolved. No returns, rankings or P&L are calculated, and no subscription is purchased.

## Reproduce the acquisition

From the repository root, with the existing `.access_token` and configured refresh credentials:

```bash
.venv/bin/python -m src.cli.acquire_tradestation_curve all \
  --year 2025 --max-rpm 30 \
  --output-dir data/commodity_curve/coverage-pilot-2025-20260906-v3
```

Use a fresh directory for each independent acquisition. To continue an interrupted run, repeat the same command with `--resume`. Mode, year, rate and algorithm identity must match the manifest. `pilot` acquires the calendar year; `audit` takes only the historical samples; `all` performs both. Requests are sequential and paced to at most 30/minute, including retries. Server retry/reset delays can reduce that rate. Expect a full run to take substantially longer than the two-contract probe.

The pilot discovers delivery candidates across 2025 and 2026 plus January 2027, then selects verified broker expirations in 2025–2026. The last candidate matters for energy contracts that expire in the preceding December. Canonical roots and broker aliases remain separate. Discovery verifies returned symbol, root, asset class, exchange, USD currency and plausible expiration; it does not certify expiration as the official last trading day.

Each accepted contract gets twelve daily-bar requests with adjacent monthly endpoints. Live responses show that a UTC-midnight first date can include the preceding date’s closing bar. The acquisition permits at most one day of leading spill, preserves it in source evidence and records overlap reconciliation. Numerically equal market observations are reconciled once, while differing retrieval flags remain documented; conflicting market values and duplicate timestamps within a request fail validation. OHLC relationships and nonnegative integral volume/open-interest counts are checked without rejecting legitimate negative prices. The final panel includes timestamps in `[2025-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`. Daily-bar timestamps are broker timestamps, not reconstructed exchange sessions. Missing sessions are never filled.

For every root, the audit samples January 1–15 and July 1–15 (exclusive ending timestamp in the report) in 2025, 2020, 2015, 2010, 2005 and 2000. Discovery includes the current delivery month, selects a nearby contract whose verified expiration is after the window starts, and, when available, another expiration 90–180 days away. These are sparse availability observations: a successful 2000 sample does not establish uninterrupted history from 2000 onward, and an unsuccessful sample does not establish the earliest available date.

## Evidence and recovery

`manifest.json` binds request identities, states and artifact SHA256 checksums. Request artifacts preserve allowlisted original market-data values and types with canonical JSON checksums, separately from normalized records. They are source-field snapshots, **not original HTTP response bytes**; arbitrary response diagnostics, account data and credentials are excluded. Normalized record checksums identify the transformed representation and must never be cited as raw-source hashes. Full useful price-format fields remain source evidence for later quote-unit reconciliation.

Successful, empty, denied, unavailable, invalid and exhausted transient outcomes remain distinguishable. HTTP 200 alone does not mean success: broker error envelopes are classified too. A terminal authentication failure is persisted in sanitized form and stops the whole acquisition immediately. Restore valid existing credentials before resuming; the workflow does not purchase access or initiate browser authorization. Retried requests preserve earlier failed or partial attempts separately. Unresolved delivery candidates are tried under fallback aliases; temporary discovery failures remain retryable on resume.

Resume validates identity and all manifest-bound artifact hashes before reuse. An unverified orphan request file is rejected, even if its contents appear plausible; retain that run for investigation and use a fresh output directory if its provenance cannot be recovered. A process lock prevents concurrent writers. Interrupted report generation can be regenerated from the verified request evidence; corruption in reports marked complete fails verification.

Offline integrity verification requires no broker connection:

```bash
.venv/bin/python -c 'from pathlib import Path; from src.data.tradestation_curve_batch import verify_run; print(verify_run(Path("data/commodity_curve/coverage-pilot-2025-20260906-v3")))'
```

`contracts.csv` preserves separate contract identities. `contract_bars.csv` contains normalized daily bars. The JSON and Markdown acquisition reports give contract dates/counts, chunk states, boundary reconciliation and simultaneous maturity-pair evidence. The availability report and root/year tables describe the sparse historical windows. The missing-metadata report records unresolved requirements. Keep these files together with the manifest and request artifacts; a report alone is not the acquisition evidence.

## Interpret breadth and rerun the metadata gate

Observed-bar breadth counts roots with any pilot bars. Simultaneous maturity-pair breadth counts roots with common timestamps in two expirations 90–180 days apart. Each is compared with nine roots across three sectors. Strategy eligibility remains unresolved even if both breadth tests pass: first-notice/last-trade safety, official settlements, vintages, calendars, permissions and costs need independent evidence. See the [exchange metadata source assessment](tradestation-curve-metadata-sources.md).

Rerun the metadata-only audit after acquisition:

```bash
.venv/bin/python -m src.cli.audit_commodity_curve \
  --data-root data \
  --output-dir _bmad-output/specs/spec-commodity-curve-carry/metadata-audit/tradestation-coverage-2025
```

The metadata audit inspects schemas and excludes sealed-holdout directories. Broker bar files do not satisfy the settlements table. Keep its `HOLD-DATA` result separate from observed acquisition breadth. The pilot neither supplies roughly 25 years of uninterrupted history nor creates an untouched reserved test.
