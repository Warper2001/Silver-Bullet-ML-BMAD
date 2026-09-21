# Commodity pilot acceptance and acquisition handoff

**Timestamp assessment update:** the subsequent [send-time audit](commodity-curve-send-time-audit-20260907.md) reconstructed and reconciled exchange transmission times for all 95,537 bounded rows. The original publication fields remain blank, but useful timestamp evidence is present. The remaining timing gate concerns message lineage and original-publication interpretation; it should not be described as categorical absence of vendor timestamps. The dated assessment below preserves the earlier gate snapshot.

**Mechanics: PASS_FIXTURES. Research: HOLD-DATA. Full historical-data readiness is not complete.**

The current synthetic CLI passed 143/143 fixtures twice with network access prohibited. Both runs matched the saved Markdown/JSON reports byte for byte. The targeted mechanics, metadata-audit (including CLI), TradeStation batch and development-export tests passed **412 tests**, with two existing Pydantic deprecation warnings. This verifies tested implementation behavior; it does not verify the historical inputs below.

The bounded January–August sample has 95,537 settlement-vintage records across 214 acquired instruments and 12 roots. Its eight monthly diagnostic indexes prevent later observed revisions from leaking into earlier checkpoints. Official publication timestamps remain absent. The [combined date inventory](commodity-curve-contract-date-gaps-20260907.md) has 129 exchange-corroborated last-trade dates and 16 explicit exchange first-notice dates. These counts concern the full acquired inventory; they do not establish which subset would be required by causal pair selection.

| Canonical gate | Current result | Evidence still required |
| --- | --- | --- |
| Source metadata | HOLD | License/use scope, complete version lineage and immutable historical references |
| Contract metadata | HOLD | Applicable dated specifications, first-notice applicability, required lifecycle dates and product session calendars |
| Settlements | HOLD | Official publication/revision timestamps, compatible vintages and expected-versus-observed required sessions |
| Breadth | HOLD | At least nine causally eligible roots in three sectors; observed symbols alone are insufficient |
| Quotes | HOLD | Contract-specific bid/ask, size, tradeability and sequences for executable claims |
| Costs/account | HOLD | Effective costs, permissions, margins and explicit capital/risk/loss constraints |
| Access/seal | HOLD | Prior-access declaration, credible history/splits and immutable carry-specific seal before outcomes |

The underlying criteria remain the canonical `data-contract.md` and `validation-plan.md` under `_bmad-output/specs/spec-commodity-curve-carry/`. A JSON copy of this assessment, with parent hashes, is saved at `data/commodity_curve/pilot-acceptance-20260907/acceptance.json`. No historical ranking, returns, P&L or reserved-test inspection was performed. No additional paid acquisition occurred.

## Concrete source request

For the already accessed 2025 development pilot, request the following from an existing entitled provider or CME. This is a prepared request only; nothing has been sent and no service has been purchased.

1. Daily point-in-time contract reference vintages for the 214 identities in `data/commodity_curve/may-2026-date-evidence-20260907/combined-contract-gaps.csv`. Include listing/expiry/last-trade/first-notice fields, explicit not-applicable semantics, settlement method, units, currency, tick/multiplier conventions, corrections and effective/publication timestamps. Identify which fields are historical snapshots versus subsequently corrected values.
2. Product session calendars and exceptional closures/early sessions sufficient for the 2025 decisions, their risk lookbacks and applicable safety boundaries. Supply the schedule versions available at each decision. Do not replace a scheduled session with a weekday assumption or a clearing-file date.
3. Contract/session settlement vintages for January–August 2025 and the necessary preceding risk history. Preserve preliminary/final and actual/theoretical distinctions, exchange publication timestamps, identifiers and revisions. Confirm coverage for listed contracts without volume/open interest; identify omissions rather than substitute active contracts.
4. Dataset history range, license rights, sample schema, existing entitlement and a price quote before any new download. State whether historical dissemination timestamps exist or only file availability times. Full-history work separately requires sufficient duration and a credible untouched reserved period under the validation plan.
5. Quote, cost, margin and account evidence only when advancing to executable claims. These inputs are separate from proving the settlement-only data gates.

## Source discovery and retrieval limits

CME describes its reference data as covering specifications, schedules and lifecycle dates, with historical delivery options. Its older futures reference-file schema explicitly distinguishes `FirstIntDt`, `FirstNoticeDt` and `LastTrdDt`; the file's business date represents processing context, not an exact publication timestamp. Historical field/version coverage and entitlement still require confirmation. [Reference data catalog](https://www.cmegroup.com/market-data/browse-data/catalog/reference-data.html), [reference-file schema](https://www.cmegroup.com/clearing/files/cme-group-product-reference-file-futures.pdf).

The public reference-file directory showed dated November 25, 2024 candidates and recent 2026 files. Direct retrieval of the four 2024 exchange ZIP candidates timed out; web CSV retrieval also failed. No reference rows were acquired or promoted into the inventory. The directory listing is discovery evidence only. Attempt results are saved in `data/commodity_curve/reference-file-probe-20260907/download-results.json`. [Public directory](https://www.cmegroup.com/ftp/pub/fprf/csv/).

CME's settlement FAQ describes multiple delivery channels with different delays. A channel schedule does not supply the missing historical record-level publication time. DataMine is a candidate access route, subject to entitlement and schema verification. [Settlement FAQ](https://www.cmegroup.com/articles/faqs/access-to-cme-group-settlement-data-faq.html), [DataMine API](https://www.cmegroup.com/datamine/datamine-api.html).

The existing [Databento inquiry draft](commodity-curve-databento-support-draft-20260907.md) remains unsent. The January–August workaround avoids the three flagged late-2025 dates; it does not resolve calendar, publication or long-history gaps. The full objective remains open until the required evidence is obtained and passes its gates.
