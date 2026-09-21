# Exchange send-time evidence audit

**PASS_TIMESTAMP_RECONSTRUCTION_CHECKS; research remains HOLD-DATA.** The existing January–August 2025 export contains sufficient timestamp fields to reconstruct exchange transmission times for all **95,537 records**. The earlier blanket assessment that timestamp evidence was unavailable was too strong.

For modern GLBX data, Databento documents exchange send time as `ts_recv - ts_in_delta`. The signed delta can saturate, so endpoint values must not be treated as exact. This audit is confined to the 2025 acquired sample; legacy data requires a separate mapping. [Databento timestamp fields](https://databento.com/docs/standards-and-conventions/common-fields-enums-types), [GLBX conventions](https://databento.com/docs/venues-and-datasets/glbx-mdp3).

The offline audit established:

- All 95,537 rows have valid, unsaturated deltas and strict event-before-send-before-receive ordering.
- Observed send-to-receive deltas range from 10,074 to 752,004,320 nanoseconds.
- All timestamp fields and contract IDs match the corresponding native DBN records by original source ordinal. The native file's receipt hash was verified first.
- Comparing send-time eligibility with the existing availability-bound eligibility produces zero differing rows at each of the eight month-end cutoffs.
- Nine hand-calculated checks cover normal reconstruction, zero/negative deltas, clock-order failures, saturation endpoints and invalid values. Repeated offline runs produce identical evidence CSV and report JSON bytes.

Artifacts are in `data/commodity_curve/send-time-audit-20260907/`: `audit.py`, `verify.py`, `send-time-evidence.csv`, `report.json` and `verification.json`. The evidence table contains timestamps and identities, not prices. Parent, native-source, code and output hashes are recorded.

This closes the question of whether useful send-time fields exist and reconcile to the acquired bytes. It does **not** establish that each message represents the first publication of its contents, or resolve recovery, restatement and definition-derived statistics provenance. Those questions are detailed in the [Databento recon](../_bmad-output/planning-artifacts/research/technical-databento-commodity-missing-data-2026-09-07/research.md). No row has been promoted to an accepted `published_at_utc` value.

The cutoff comparison concerns row eligibility only. It does not establish identical vintage ordering, eligible contract pairs, strategy decisions or P&L. Existing exports and earlier reports remain preserved. The next source request should seek the missing provenance and calendar guarantees, rather than assume a replacement timestamp dataset is required.
