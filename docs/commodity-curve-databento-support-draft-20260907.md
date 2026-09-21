# Draft only — Databento quality inquiry

Recipient: support@databento.com (verified at https://databento.com/support).

Subject: GLBX.MDP3 degraded dates: settlement and definition/status impact

Hello Databento Support,

We downloaded GLBX.MDP3 statistics, definition and status data on September 7, 2026 using Python SDK 0.85.0. The request interval was 2025-01-01 inclusive through 2026-01-01 exclusive, with `stype_in=raw_symbol`, restricted to 214 outright contracts across CL, NG, RB, HO, HG, ZC, ZW, ZS, ZM, ZL, LE and HE.

The request warned that September 17, September 24 and November 28, 2025 were degraded. The saved dataset-condition response gives last-modified dates of June 11, June 12 and June 9, 2026 respectively.

We retain all native records and settlement vintages. Grouping settlement records by `ts_ref`, we observe 414 records / 138 instruments on September 17, 411 / 137 on September 24, and 474 / 120 on November 28. All twelve roots have records on each date. These counts do not establish completeness.

Could you clarify:

1. What incident or normalization issue caused each warning? Does the condition date refer to a UTC capture date or an exchange trading date?
2. Which schemas, channels, instruments and UTC intervals are affected? In particular, are settlement statistics (`stat_type=3`), contract definitions or trading-status events missing, corrupted, or timestamp-affected?
3. Have these dates been repaired? If so, what was repaired, when, and does our September 7 extraction include it? Are original event, send and capture timestamps preserved on repaired records, or reconstructed? Please identify any remaining limitations.
4. Can you provide an incident reference or machine-readable coverage information sufficient to distinguish affected from unaffected records? We need to preserve preliminary/final/theoretical/actual status and assess only information available at historical cutoffs.
5. Your feed documentation notes that CME does not publish MDP settlements for instruments without open interest or volume. Is there an entitled source for complete listed-contract settlements with publication vintages, and for historical session/notice calendars? Please describe availability and any quoted charges; do not enable services or initiate billable downloads.

We can provide the exact symbol list, request parameters, file hashes and small record samples if useful. Please let us know what would help.

Thank you.

---

This draft has not been sent. It contains no API keys, account credentials, trading results or raw market-price attachments.
