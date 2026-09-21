# Commodity settlement audit — 2025 development pilot

**HOLD-DATA.** The acquired feed contains identifiable settlement vintages, but eventual final settlements cannot be used indiscriminately at historical month-end cutoffs. This audit performs no ranking, strategy returns, trades or holdout access. No additional data purchase was made.

## Findings

- All 130,579 settlement records were preserved across 214 instruments and 40,720 instrument/session groups. These totals include messages received in 2025 referring to the December 31, 2024 session.
- 48,984 records are final actual settlements, 81,580 preliminary actual, and 15 final theoretical. Twelve theoretical records have zero prices; they remain visible and flagged. No undefined prices, invalid timestamp diagnostics, exact decoded duplicates or simultaneous same-flag price conflicts were found.
- Forty instrument/session groups contain different prices across their messages. One group contains a changed final actual price: **ZLU5, March 31, 2025**, reported as **45.48** on March 31 at 23:11:41 UTC and **47.74** on April 1 at 22:04:10 UTC (times shown to seconds). Only the first final record was available at the March cutoff. Original nanoseconds are retained in the export.
- 1,004 instrument/session groups have no final actual record in this extraction. The December endpoint is right-censored: records arriving in January 2026 were not requested. Missing final records are not automatically missing preliminary settlements.
- All **95** comparable CME last-trade date observations agree with vendor expiration dates. This corroborates date identity for those contracts; it does not certify first-notice dates, exact event times or historical publication availability.
- All 51,285 definition records have an undefined `contract_multiplier` sentinel. It must not be used numerically. Contract quantities, quote units and point values still require explicit reconciliation.
- No gaps were found against same-root peer-observed dates within vendor-reported contract life. This is a relative diagnostic: a date missing for every peer would escape it. Full exchange-calendar coverage remains unproven.

## Month-end timing

The existing protocol cutoff is 23:59:59 UTC on the last calendar day. The table uses each root's latest **observed** session as a provisional reference; it does not certify the required exchange session. Counts are contracts across all 12 roots, not selected nearby/deferred pairs.

| Month | Contracts with records on provisional session | Clean record available by cutoff | Final actual available by cutoff |
| --- | ---: | ---: | ---: |
| January | 201 | 201 | 0 |
| February | 197 | 197 | 0 |
| March | 187 | 186 | 184 |
| April | 183 | 182 | 179 |
| May | 171 | 171 | 0 |
| June | 167 | 167 | 164 |
| July | 155 | 155 | 153 |
| August | 146 | 146 | 143 |
| September | 135 | 135 | 133 |
| October | 127 | 127 | 0 |
| November | 120 | 120 | 118 |
| December | 110 | 110 | 0 |

Final-only counts are diagnostic, not a change to the protocol's latest-permissible-vintage rule. Preliminary values must retain their status. Declining counts reflect this fixed pilot contract set and cannot establish complete listing coverage. No missing leg was replaced with an active contract or an older session.

## Source and timing conventions

For GLBX.MDP3, bit values 1, 2, 4 and 8 identify final, actual, trading-tick and intraday status respectively. `ts_ref` carries a trading date, so converting it to a local timezone would incorrectly shift some dates. The vendor also documents that CME omits MDP settlement messages for instruments without open interest or volume. [Databento feed specification](https://databento.com/docs/venues-and-datasets/glbx-mdp3).

The audit preserves `ts_recv`, `ts_event`, `ts_in_delta` and the raw reference timestamp. `max(ts_recv, ts_event)` is used only as a conservative diagnostic availability bound. **Official `published_at_utc` remains null**; capture receipt is not silently renamed publication. Prices are decoded exactly from integer nanounits with Decimal arithmetic. [Databento statistics schema](https://databento.com/docs/schemas-and-data-formats/statistics).

The local export is an inspection dataset, not an accepted production settlements table. Each row retains the source-record ordinal and a hash of the SDK-decoded DBN v3 record. Those hashes are distinct from compressed source-file hashes, which are retained separately.

## Verification and artifacts

Artifacts live in `data/commodity_curve/settlement-audit-2025-20260907/`. `audit.py` reads only the explicitly named acquired files and CME date-fact file, with network calls prohibited. CSVs preserve settlement and definition vintages, per-instrument coverage, monthly availability and date reconciliation. `audit.json` contains reason counts and source/code hashes; `source-conventions.json` records the source interpretation.

`verify_audit.py` passed **155 independent checks**, including SQL aggregation, all observed flag combinations, 144 monthly root comparisons, and seven records checked directly against native source data. Parser checks found no truncation warnings. A second offline run produced identical bytes and SHA-256 hashes for all seven generated audit tables/reports (`determinism.json`). The research verdict remains separate from these diagnostic verification results.

## Next acceptance work

Reconcile quote units and point values against effective-dated specifications; establish a complete historical session and first-notice calendar; resolve the three degraded dates (September 17, September 24 and November 28); and validate compatible as-of vintages for actual calendar-selected pairs. Any supplementary records beyond the approved download need a new bounded acquisition decision. This one-year pilot does not establish the long history, untouched test period or executable quote/account inputs required by the wider protocol.
