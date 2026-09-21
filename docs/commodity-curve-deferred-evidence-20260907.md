# Deferred-contract evidence and quality follow-up

**HOLD-DATA.** The dated last-trade evidence now corroborates **119 pilot contracts**, up from 95. No additional market data was purchased and no historical performance or holdout analysis was performed.

Three additional CME notices supplied 57 event observations for 24 new contracts. All 26 last-trade observations agree with the vendor definitions; two contracts appear in two notices, whose separate source vintages were retained. The notices cover January, February and June 2026, including next-month CL/NG tables for February, March and July. [January notice](https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2025/chadv25-391.pdf), [February notice](https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2026/01/chadv26-033.pdf), [June notice](https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2026/05/chadv26-192.pdf).

The publication dates are December 24, 2025, January 27, 2026 and May 26, 2026 respectively. These notices corroborate dates; they cannot supply information at an earlier historical cutoff. Exact UTC publication timestamps and original PDF bytes remain unavailable. Other 2026 monthly delivery notices were not located by the public searches in this pass; this is not evidence that they do not exist. Missing dates were not inferred.

## Degraded-date follow-up

| Vendor-flagged date | Settlement records | Instruments | Roots |
| --- | ---: | ---: | ---: |
| 2025-09-17 | 414 | 138 | 12 |
| 2025-09-24 | 411 | 137 | 12 |
| 2025-11-28 | 474 | 120 | 12 |

These are counts of all returned vintages by settlement reference date, not accepted final prices or a completeness finding. All three dates remain quarantined for acceptance.

CME's November 28 notice reports a derivatives trading halt affecting intraday index calculations. This supports the occurrence of an exchange disruption but does not identify which acquired settlement records are affected or establish the cause of Databento's degradation. The PDF extraction title says October, while the body and URL identify November; this inconsistency is recorded. No sufficiently specific public official explanation for the September dates was located. [CME interruption notice](https://www.cmegroup.com/market-data/cme-group-benchmark-administration/files/28-november-2025-cme-group-intraday-indices-cvol-and-petroleum.pdf).

Artifacts live in `data/commodity_curve/deferred-evidence-20260907/`: source facts, date comparisons, quality counts, an offline reconciliation script, verification and hashes. Checks passed for all date matches, 57 unique source/event keys, and identical output bytes on repeated runs. Earlier evidence files remain unchanged.

Remaining requirements include the full session calendar and first-notice evidence through applicable safety boundaries, earlier metadata publication vintages, and provider clarification or independent reconciliation of degraded dates. The broader long-history and executable-data requirements also remain open.
