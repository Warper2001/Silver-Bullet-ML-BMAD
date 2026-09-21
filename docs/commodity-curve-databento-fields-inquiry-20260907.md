# Draft only — historical reference fields and settlement provenance

To: support@databento.com

Subject: GLBX.MDP3 2025: historical reference backfill and settlement SendingTime provenance

Hello Databento Support,

We are assessing your products for historical commodity-futures research. We already downloaded GLBX.MDP3 statistics, definitions and status for 214 outright contracts in 2025. The attached contract list contains instrument IDs, symbols and roots only. Please identify the relevant existing products or custom services, without enabling subscriptions or starting billable work.

**1. Historical contract reference data and calendars**

Can you supply historical CME reference files containing first-notice, first-intent, last-trade and delivery dates, physical/cash settlement applicability, and dated amendments? Our normalized definitions provide expiration/last-trade information but do not expose the remaining fields we need. We found an older roadmap reference to fixed-cost historical CME security-definition backfills. Is that service available now, and does it supply MDP definitions, FIXML/FPRF reference files, or another source?

Can you also supply product session calendars with holidays, early closes, exchange trade dates and the versions available at historical decision times? Observed status transitions alone do not establish the published schedule. Please provide a field dictionary, a small sample, available years, mapping keys and a scoped quote. The immediate scope is January–August 2025 decisions, their preceding risk history and applicable contract safety boundaries.

**2. Settlement message provenance**

Our 95,537 January–August settlement records all contain unsaturated `ts_in_delta`. Reconstruction using `ts_recv - ts_in_delta` gives consistent event-before-send-before-receive ordering and matches the native records. We have not yet treated these send times as original-publication timestamps.

For these 2025 records, does the reconstructed time preserve original CME SendingTime? How can we distinguish initial dissemination, later corrections, recovery messages and restated values? Your August 8, 2026 release notes describe moving settlement prices from definitions into statistics: was historical data reprocessed, and how are the original message type, reference date and timing preserved? Please describe any limitations or record-level flags we should check.

**3. Settlements absent from the MDP feed**

Your GLBX documentation describes settlement omissions for instruments without volume or open interest. Do you offer another source covering such listed contracts while retaining publication and revision history? If so, please distinguish its coverage and timing from MDP statistics and PCAPs.

Please confirm existing entitlement, historical ranges, license scope and costs before proposing a download. A sample and written description are sufficient for the next assessment; no new service or paid job should be initiated.

Thank you.

---

Prepared attachment: `data/commodity_curve/databento-fields-inquiry-20260907/contracts.csv`.

This inquiry is unsent. It complements the earlier degraded-date inquiry; January–August development excludes those flagged late-2025 dates. No credentials, prices or trading results are attached.
