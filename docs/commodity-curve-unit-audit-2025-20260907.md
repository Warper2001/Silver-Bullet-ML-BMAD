# Commodity quote-unit reconciliation — 2025 pilot

**PASS_PILOT_UNIT_RECONCILIATION**, with research still **HOLD-DATA**. All 51,285 acquired definition records, covering 214 instruments and 12 roots, agree with the inspected exchange contract quantities and tick conventions. Eight comparisons per record passed (410,280 comparisons), and no instrument changed its unit convention within the downloaded definitions.

The table gives dollars per **one whole quoted price-unit move**, per contract. This is the multiplier needed when converting changes in the vendor's conventional quoted prices into dollars.

| Root | Vendor quote unit | USD per quote point | Outright tick | USD per tick |
| --- | --- | ---: | ---: | ---: |
| CL | dollars/barrel | 1,000 | 0.01 | 10 |
| NG | dollars/MMBtu | 10,000 | 0.001 | 10 |
| RB | dollars/gallon | 42,000 | 0.0001 | 4.20 |
| HO | dollars/gallon | 42,000 | 0.0001 | 4.20 |
| HG | dollars/pound | 25,000 | 0.0005 | 12.50 |
| ZC | cents/bushel | 50 | 0.25 | 12.50 |
| ZW | cents/bushel | 50 | 0.25 | 12.50 |
| ZS | cents/bushel | 50 | 0.25 | 12.50 |
| ZM | dollars/short ton | 100 | 0.10 | 10 |
| ZL | cents/pound | 600 | 0.01 | 6 |
| LE | cents/pound | 400 | 0.025 | 10 |
| HE | cents/pound | 400 | 0.025 | 10 |

Each root's official rulebook URL, page and extracted numeric facts are retained in `data/commodity_curve/unit-audit-2025-20260907/exchange-unit-facts.json`. For example, [corn Chapter 10](https://www.cmegroup.com/rulebook/CBOT/I/10.pdf) specifies 5,000 bushels and a quarter-cent tick worth $12.50; [crude oil Chapter 200](https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf) specifies 1,000 barrels and a $0.01/barrel tick. These independently support $50 and $1,000 per whole quoted point respectively.

The vendor-derived calculation uses contract quantity and a quote-to-dollar scale. For nonfractional contracts, that scale is `(min_price_increment_amount / display_factor) / (min_price_increment × unit_of_measure_qty)`. Fractionally quoted grains use the vendor-documented hundredths convention. The resulting tick value and point value were checked against the separately transcribed exchange facts using 34-digit Decimal arithmetic. [Databento calculation guide](https://databento.com/docs/examples/instrument-definitions/contract-notional).

The undefined `contract_multiplier` field remains undefined. Derived `usd_per_quote_point` is a separate field; it is never filled with the sentinel. Display factors must not be applied again to already normalized settlement prices. Outright trading ticks are not used to reject finer clearing settlement prices.

`definition-unit-checks.csv` preserves per-record timestamps, raw definition identity, calculation method, values and decoded-record hashes. `report.json` records comparison counts and source/code hashes. The audit reads only the acquired definition file and its receipt, with network access prohibited. No strategy prices were transformed, returns calculated, holdout data read or additional data purchased.

This resolves the numerical unit ambiguity for the observed pilot definitions. The inspected current rulebooks corroborate those observations but do not supply a complete archive of effective-dated historical amendments. Original PDF bytes and exact historical publication timestamps were not obtained. Full calendars, notice dates, settlement availability and the three degraded data dates remain acceptance work; this result does not lift HOLD-DATA.

Verification also independently checked all 12 root point/tick-value pairs (24 checks). A repeated offline run produced identical bytes for the facts, per-record checks and JSON report. `verification.json` and `manifest.json` preserve this evidence and file hashes.
