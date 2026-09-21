# Commodity calendar evidence — 2025 pilot

**HOLD-DATA.** Additional exchange evidence has been reconciled with the acquired pilot. A complete exchange-session calendar has not been established.

The January 9, 2025 national day of mourning had a **12:15 Chicago time (18:15 UTC) agricultural close**, announced December 30, 2024. Energy and metals retained normal hours. The downloaded status file corroborates nontrading transitions at that time for **87 instruments across HE, LE, ZC, ZL, ZM, ZS and ZW**. All twelve roots have settlement records referencing January 9, so treating it as a full holiday would be incorrect. [CME announcement](https://investor.cmegroup.com/news-releases/news-release-details/cme-group-announces-trading-hours-us-national-day-mourning-honor).

A newly located January 1 clearing advisory states that CME would produce no settlement file that day. The acquired feed contains no January 1 settlement reference dates. This is clearing-file evidence, not a full trading-session schedule. The nine previously sourced no-settlement holiday dates likewise have no settlement reference records in this extraction. [CME New Year advisory](https://www.cmegroup.com/content/dam/cmegroup/tools-information/holiday-calendar/files/2025-new-years-advisory.pdf).

Lean-hog Rule 15203.A specifies cash settlement with no physical delivery. The 16 HE contracts in the pilot now have an explicit **NOT_APPLICABLE_CURRENT_CASH_RULE** physical-notice convention in the diagnostic inventory. This does not supply their historical last-trade calendar or backdate the current rulebook. [CME Chapter 152](https://www.cmegroup.com/rulebook/CME/II/150/152/152.pdf).

The inventory covers 214 contracts and preserves separate columns for explicit last trade, explicit first notice, notice day and first intent. It contains 95 sourced last-trade dates and 12 copper first-notice dates. CBOT intent and delivery notice remain distinct; no first-notice date was inferred from an intent date. [CBOT Rule 713](https://www.cmegroup.com/rulebook/CBOT/I/7/7.pdf).

Artifacts are in `data/commodity_curve/calendar-audit-2025-20260907/`: `source-facts.json`, `contract-evidence-inventory.csv`, `january-9-status-events.csv`, and `report.json`. The offline audit reads the 496,233 acquired status records, definitions, settlement-date export and existing source facts. SDK trading flags are nullable booleans; false is preserved separately from unknown. Event dates are not silently promoted to exchange session identifiers.

The full calendar, including coverage through deferred contracts' 2026 safety boundaries, remains missing. No-settlement days may still have trading, early closes count differently from closures, and historical effective dates/publication times remain incomplete. No calendar was synthesized from weekdays or observed bars. The raw acquisition and earlier audit artifacts remain unchanged; no additional spending, holdout access or strategy analysis occurred.

Verification passed for inventory counts, seven independent early-close root counts, the Chicago-to-UTC conversion and holiday reference-date checks. A repeated offline run produced identical bytes for all three generated tables/reports. Hashes and dependencies are recorded in `manifest.json`.
