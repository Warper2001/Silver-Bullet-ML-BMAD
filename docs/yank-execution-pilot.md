# Frozen-order execution evidence audit

Run from the isolated pilot worktree with the existing Databento 0.85.0 environment:

```sh
cd /root/Silver-Bullet-ML-BMAD-yank-minute
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m src.cli.check_yank_execution_pilot --output-dir docs/reports/yank-execution-pilot/run2
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_execution_pilot -q
```

The output directory must be new and outside source/input trees. The command reads only the pinned acquisition and archived replay evidence, verifies manifest hashes before trusting their contents, and verifies every consumed input before and after processing. No API client, key, live trader, replay engine, or network operation is invoked. A failed run cannot retain a PASS report.

`report.json` contains five cases and six conditional findings per case, exact archived source records, input/code hashes, decoder versions, native schema diagnostics, status and definition records, and May 28 crossing timelines. `report.md` presents the results for inspection. `reconciliation.jsonl` contains every original-minute/native-T comparison for both label conventions, using capture time and a separate exchange-time diagnostic. `event-extracts.jsonl` preserves source file, zero-based native record index, sequence, event identity, capture and exchange timestamps for timeline crossings. `artifacts.json` hashes the canonical artifacts. No wall-clock run time or output-directory path is embedded.

The stream uses bounded NumPy chunks and retains unfinished events across chunks. Books are rebuilt from daily synthetic resets/adds on case days; native trade reconciliation and format/timestamp/boundary checks cover every purchased day. T prints contribute trade evidence, F notifications do not contribute volume, and neither changes resting orders. C subtracts quantity and M replaces price and quantity. Completed F_LAST events alone produce usable book evidence. Synthetic timestamps never become arrival observations or trades. Snapshots without F_LAST await a completed live event.

Arrival uses the last strictly prior completed live book. An event straddling arrival makes the arrival book unavailable. Arrival-time equality is excluded from trade support. The pending lifetime ends at the exclusive end of the 240th subsequent actual original bar opportunity, including that last bar’s trades, and is bounded by purchased evidence. Archived assumed fill times never shorten it. Events ending after expiry retain eligible trades whose own capture timestamps precede expiry.

Evidence outcomes are supported, touch-only, unsupported or unassessable. Book, native-data, scheduled coverage and status gaps are explicit. “Supported” means strict trade-through under a no-impact assumption, never an inferred fill, partial fill or queue position. Immediately executable bid size and same-price ask queue context are separately reported. A brief May 28 non-trading interval remains part of pending-window assessability even where later trade-through exists. Historical fills and P&L remain reference data; the historical $4 cost is not a verified broker charge.

Both start- and end-labelled bar interpretations remain conditional because empirical reconciliation cannot independently establish original bar provenance. Timeline crossings within one native event or at equal capture times remain ambiguous; invalid completed events make ordering unassessable. Actual fill and queue outcomes remain unobserved in every case. `PASS_AUDIT_CHECKS` is separate from `HOLD_VALIDATION` and does not validate strategy returns.

Vendor conventions consulted: [MBO snapshots](https://databento.com/docs/standards-and-conventions/mbo-snapshot) and [resting-order tracking](https://databento.com/docs/examples/order-book/order-tracking).

Reconciliation retains raw finite T observations and labels each minute with detected contamination reasons, coverage qualification, volume delta, and whether its capture day received book replay. Basic action/side/price/size and native ordering checks also run outside case days. A whole event's failures qualify all of its observed minutes. Coverage credits each eligible live record's minute only when that record's entire event validates, including events that finish after pending expiry.

Gap counts are exact and representative samples are the canonical smallest 100 source/index/reason/time references. Affected timestamps are compacted within minute partitions split at all scenario and timeline assessment boundaries; this preserves exact interval overlap without retaining every malformed record. Archived timeline extraction is independent of pending expiry. Status, native gaps and missing validated minutes anywhere in the archived interval qualify its ordering conclusion.

Final input and implementation verification completes before any PASS artifact is published. Cleanup catches ordinary failures and interruption during verification or publication and removes any partially written PASS artifacts.

The restored command loads its audit package privately, without executing `src.research` initialization. Acquisition and original bars remain pinned in `/root/Silver-Bullet-ML-BMAD`; archived replay and engine remain in `/root/Silver-Bullet-ML-BMAD-yank-replay`. Only the output directory is configurable. The [completion report](reports/yank-pilot-completion/README.md) connects this full-window audit to provenance, corrected minute replay and subsequent May 28 observations. Historical `final-run1` artifacts retain their original bytes and code hashes.
