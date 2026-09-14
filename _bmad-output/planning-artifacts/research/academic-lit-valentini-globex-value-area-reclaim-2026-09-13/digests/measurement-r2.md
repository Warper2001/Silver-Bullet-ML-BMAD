# Measurement digest — round 2, targeted closure

Accessed: 2026-09-13. Nine total batched web calls across both rounds; round 2 retains two additional primary sources. No project data read. Builds on [round 1](measurement-r1.md).

| ID | Publisher / source | pub_date | accessed | confidence / class | Supported claim |
|---|---|---|---|---|---|
| M9 | Databento, [Venues and datasets — CME Globex MDP 3.0](https://databento.com/docs/knowledge-base/datasets) | Undated living documentation | 2026-09-13 | High / vendor CME normalization specification | Every native CME trade summary becomes one T record; passive order detail entries become F records. An existing aggressor order can also receive F detail. Trades/MBP schemas exclude F details. Implied executions can have unspecified aggressor side. Records for an instrument retain their message order even with equal timestamps. |
| M10 | Databento, [Schemas and data formats — OHLCV](https://databento.com/docs/knowledge-base) | Undated living documentation | 2026-09-13 | High / vendor aggregation specification | No trade in an interval means no OHLCV record. Bar ts_event labels the inclusive start and aggregation uses underlying trades' ts_recv. Daily bars use UTC dates. Vendor recommends constructing aggregates from trades for transparent timestamp/condition handling; electronic volume can differ from official reported volume/settlement definitions. |

## Closed measurement issues

M9 verifies the CME-specific volume rule: sum quantities from T only, or request trades schema directly. F quantities are execution detail, not additional volume. Do not deduplicate legitimate equal-timestamp records or collapse all T records within one native message; the feed can contain multiple trade summaries. This is distinct from removing accidental duplicate downloads, which must rely on provenance and record identity.

M10 means consecutive stored rows need not be consecutive elapsed minutes. A missing minute alone is neither proof of a data outage nor proof of zero market activity: distinguish no-trade intervals with coverage evidence from incomplete capture, exchange halts, and missing files. For replica bars, use the same timestamp convention for both bars and profile cutoffs; a start-labelled 09:30 minute becomes available after its end. Avoid comparing a receive-time vendor bar with an event-time profile without documenting boundary differences. These are direct timing and completeness inferences.

Round-1 M8 was rechecked against the full current Sierra page at calculation lines 1038–1062: POC and value-area tie/expansion semantics match the indexed legacy extract. No change to the result.

## Recommended measurement acceptance gate (design, not performance evidence)

1. Manifest explicit MNQ expiry, date-aware scheduled session, request bounds, files, quality flags, and missing intervals. Historical contract rolls must be predetermined and profiles must not silently mix expiries or adjusted prices.
2. Verify T-only MBO volume equals trades-schema volume for the same contract, request coverage, and timestamp convention. Reconcile self-built minute volume with vendor OHLCV; explain differences before using signals.
3. Compare true and uniform-proxy profiles on predeclared complete sessions. Report VAL/VAH displacement in ticks, excursion/reclaim label disagreement, event-count differences, and time-of-day distributions. Do not tune the proxy against favorable P&L or silently equate agreement in total volume with agreement in profile.
4. Freeze cutoff, algorithm, ties, bin width, moving-versus-fixed reference, volume comparisons, eligibility, and execution timing. Keep incomplete sessions in an exclusion ledger with reasons determined independently of outcomes.

Steps 1–4 are analyst recommendations derived from M1–M10 and the round-1 information-loss proof. They do not certify that any existing dataset passes. No arbitrary numerical accept/reject tolerance or trading threshold was selected.

## Absent evidence and final stop reason

OHLCV direct URL returned an error; same documentation was retrieved successfully through its primary knowledge-base URL. Search found a vendor roadmap item saying trade corrections/busts are dropped; not promoted into a CME-specific conclusion because the retrieved item lacks clear dated dataset scope. Corrections policy should therefore be captured from the actual purchased dataset/version before reconciliation.

No direct empirical MNQ comparison of uniform minute allocation versus trade-level value area was retrieved. No evidence shows the stated falling/rising-volume sequence identifies absorption or produces an edge. Current regular hours were verified, but a complete historical MNQ holiday schedule was not obtained. Stop: core measurability and the two targeted normalization/timing gaps are resolved within budget; remaining needs require actual data/calendar acquisition and preregistered validation, not more generic literature.
