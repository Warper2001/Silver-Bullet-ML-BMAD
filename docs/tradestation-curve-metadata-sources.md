# TradeStation commodity-curve metadata source assessment

Research date: 2026-09-06. Classification: feasibility/development. Decision: **HOLD-DATA**.

This source inventory supports the 2025 acquisition pilot. It does not certify broker bars as exchange settlements, supply a historical contract calendar, or release the sealed holdout. Sources below were inspected on the research date; current rulebook URLs can change in place. No subscription was purchased. Contract-specific broker evidence and acquisition outcomes belong in the run manifest and report.

## Contract size and price-unit checks

The point values below are arithmetic conversions of the cited exchange contract sizes into the stated quote units, not an assertion about TradeStation's numeric encoding. Preserve the broker's `PriceFormat`, `PointValue`, currency, symbol identity and raw response; reconcile those fields before any later normalization. One whole quote-unit move means, for example, one cent/bushel for grains, not one dollar/bushel. All listed monetary conventions are USD.

| Canonical root | Exchange/product | Physical contract unit | Quote unit used in this table | USD per whole quote-unit move | Outright minimum tick / USD tick value | Exchange source |
| --- | --- | --- | --- | ---: | --- | --- |
| CL | NYMEX WTI light sweet crude | 1,000 barrels | dollars/barrel | 1,000 | 0.01 / 10 | [Chapter 200, 200102.B–C](https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf) |
| NG | NYMEX Henry Hub natural gas | 10,000 MMBtu | dollars/MMBtu | 10,000 | 0.001 / 10 | [Chapter 220, 220102.B–C](https://www.cmegroup.com/rulebook/NYMEX/2/220.pdf) |
| RB | NYMEX RBOB gasoline | 42,000 gallons | dollars/gallon | 42,000 | 0.0001 / 4.20 | [Chapter 191, 191102.B–C](https://www.cmegroup.com/rulebook/NYMEX/1a/191.pdf) |
| HO | NYMEX NY Harbor ULSD | 42,000 gallons | dollars/gallon | 42,000 | 0.0001 / 4.20 | [Chapter 150, 150102.B–C](https://www.cmegroup.com/rulebook/NYMEX/1a/150.pdf) |
| HG | COMEX copper | 25,000 pounds | dollars/pound | 25,000 | 0.0005 / 12.50 | [Chapter 111, 111102.B–C](https://www.cmegroup.com/rulebook/COMEX/1a/111.pdf) |
| ZC | CBOT corn | 5,000 bushels | cents/bushel | 50 | 0.25 / 12.50 | [Chapter 10, 10102.B–C](https://www.cmegroup.com/rulebook/CBOT/I/10.pdf) |
| ZW | CBOT Chicago SRW wheat | 5,000 bushels | cents/bushel | 50 | 0.25 / 12.50 | [Chapter 14, 14102.B–C](https://www.cmegroup.com/rulebook/CBOT/II/14/14.pdf) |
| ZS | CBOT soybeans | 5,000 bushels | cents/bushel | 50 | 0.25 / 12.50 | [Chapter 11, 11102.B–C](https://www.cmegroup.com/rulebook/CBOT/II/11/11.pdf) |
| ZM | CBOT soybean meal | 100 short tons (2,000 pounds each) | dollars/short ton | 100 | 0.10 / 10 | [Chapter 13, 13102.B–C](https://www.cmegroup.com/rulebook/CBOT/II/13/13.pdf) |
| ZL | CBOT soybean oil | 60,000 pounds | cents/pound | 600 | 0.01 / 6 | [Chapter 12, 12102.B–C](https://www.cmegroup.com/rulebook/CBOT/II/12/12.pdf) |
| LE | CME live cattle | 40,000 pounds | cents/pound | 400 | 0.025 / 10 | [Chapter 101, 10102.B–C](https://www.cmegroup.com/rulebook/CME/II/100/101/101.pdf) |
| HE | CME lean hogs | 40,000 pounds | cents/pound | 400 | 0.025 / 10 | [Chapter 152, 15201–15202](https://www.cmegroup.com/rulebook/CME/II/150/152/152.pdf) |

## Expiration, notice and delivery evidence

These are source-backed rules or explicit unresolved items, not populated historical dates. “Business day” requires the applicable exchange calendar, exceptional closure notices and effective rule version. A broker `ExpirationDate` must remain a broker field until its relationship to last trading day is verified. Notice of intention, assignment/notice day, first position day and first delivery day are different events.

| Root(s) | Last trading rule supported by inspected source | Notice evidence / unresolved normalization |
| --- | --- | --- |
| CL | Third business day before the 25th of the preceding month; if the 25th is not a business day, count from the preceding business day. Original listed expiration generally survives later holiday-schedule changes, subject to the rule's holiday exception. | Intent due first business day after trading ends; notice day second business day after trading ends. [200102.F, 200105](https://www.cmegroup.com/rulebook/NYMEX/2/200.pdf). |
| NG | Third business day before the delivery month's first day; rule specifies treatment of later holiday changes. | Notice day first business day after final trading day. [220102.F, 220105.B](https://www.cmegroup.com/rulebook/NYMEX/2/220.pdf). |
| RB | Last business day before delivery month. | Intent first business day of delivery month; notice day second business day. [191102.F, 191106](https://www.cmegroup.com/rulebook/NYMEX/1a/191.pdf). |
| HO | Last business day before delivery month. | Intent first business day of delivery month; notice day second business day. [150102.F, 150106](https://www.cmegroup.com/rulebook/NYMEX/1a/150.pdf). |
| HG | Third-last business day in delivery month. | First notice is last business day of preceding month under general metal delivery rules. [111102.F](https://www.cmegroup.com/rulebook/COMEX/1a/111.pdf), [joint Chapter 7, 706.C](https://www.cmegroup.com/rulebook/NYMEX/1/7.pdf). |
| ZC | Business day before the 15th of delivery month. | Exact historical notice dates remain to be acquired and reconciled with clearing delivery schedules. [10102.G](https://www.cmegroup.com/rulebook/CBOT/I/10.pdf). |
| ZW | Business day before the 15th of delivery month. | Same unresolved calendar requirement. [14102.G](https://www.cmegroup.com/rulebook/CBOT/II/14/14.pdf). |
| ZS | Business day before the 15th of delivery month. | Same unresolved calendar requirement. [11102.G](https://www.cmegroup.com/rulebook/CBOT/II/11/11.pdf). |
| ZM | Business day before the 15th of delivery month. | Same unresolved calendar requirement. [13102.G](https://www.cmegroup.com/rulebook/CBOT/II/13/13.pdf). |
| ZL | Business day before the 15th of delivery month. | Same unresolved calendar requirement; final delivery deadline differs from grains/meal. [12102.G](https://www.cmegroup.com/rulebook/CBOT/II/12/12.pdf). |
| LE | Last business day of contract month. | Tender cannot occur on/before first Friday; exact assignment/notice date needs contract calendar and full tender exceptions. [10102.H, 10104.A](https://www.cmegroup.com/rulebook/CME/II/100/101/101.pdf). |
| HE | Tenth business day in contract month (also identified as July's last trade date in current 15202.D). | Cash settlement; physical first notice is inapplicable, requiring explicit N/A treatment rather than an invented date. [Chapter 152](https://www.cmegroup.com/rulebook/CME/II/150/152/152.pdf), [exchange historical contract fact sheet](https://www.cmegroup.com/content/dam/cmegroup/education/interactive/moore-report/pdf/AC-167_MoorePorkFinalwDemo.pdf). |

For CBOT agricultural contracts, [Chapter 7, 713](https://www.cmegroup.com/rulebook/CBOT/I/7/7.pdf) describes intent, delivery notices and notice to buyers. Historical clearing advisories show that exchange-specific columns differ: the [July 2009 delivery schedule](https://www.cmegroup.com/tools-information/lookups/advisories/clearing/ChAdv09-264.html) separates CBOT first holding/intent from NYMEX notice day and COMEX first notice day. A single date copied between those columns would lose meaning. The pilot has not reconstructed a versioned CBOT notice calendar.

## Calendar and settlement sources still needed as datasets

The [CME expiration calendar](https://www.cmegroup.com/tools-information/calendars/expiration-calendar.html) and [holiday/trading hours page](https://www.cmegroup.com/trading-hours.html) are official discovery points. Their current interactive pages do not establish a complete, archived 2000–2026 session/expiration calendar. Obtain dated contract schedules, settlement holiday notices and exceptional closures, retaining original sources and retrieval hashes. Calendar completeness cannot be inferred from bars or `IsEndOfHistory`.

| Roots | Inspected official settlement-method source | What it supports, and what it does not |
| --- | --- | --- |
| CL | [CME NYMEX crude oil procedure](https://cmegroupclientsite.atlassian.net/wiki/display/EPICSANDBOX/NYMEX%2BCrude%2BOil) | Exchange determination based on Globex activity; no mapping of TradeStation daily `Close` to the resulting settlement. |
| NG | [SER-8427, amended natural-gas settlement procedure](https://www.cmegroup.com/notices/ser/2019/08/SER-8427.pdf) | Dated 2019 procedure with active/deferred month and expiration handling; later amendments and pre-2019 versions still need collection. |
| RB | [CME RBOB procedure](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457088078/NYMEX%2BRBOB%2BGasoline) | Daily/final exchange methodology; broker-close equivalence unresolved. |
| HO | [CME NYMEX heating-oil/ULSD procedure](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457415161/NYMEX%2BHeating%2BOil) | Active month VWAP and deferred spread-based pricing/fallbacks; “last trade” alone cannot establish settlement. |
| HG | [CME copper procedure](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457415464/Copper%3Fredirect%3D%252Ftrading%252Fmetals%252Ffiles%252Fdaily-settlement-procedure-copper-futures.pdf) | Active-month 12:59–13:00 ET window and separate spread treatment; historical versions and broker lineage unresolved. |
| ZC, ZW, ZS, ZM, ZL | [CME grains procedure](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457414829/Grains) | Ordinary settlements based on 13:14–13:15 CT activity, with product/expiration rules. Current methodology is not a historical settlement-price dataset. |
| LE, HE | [CME livestock procedure](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457317920/Livestock) | Ordinary 12:59:30–13:00 CT settlement window; HE final cash settlement uses its index, distinct from an ordinary bar close. |

The [Daily Bulletin](https://www.cmegroup.com/market-data/daily-bulletin.html) supplies official product sections and distinguishes preliminary from final publication. The [daily settlement-time documentation](https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/457085528) further distinguishes settlement windows and publication times. Neither proves that the account includes an archived settlement feed or gives permission to relabel broker bars. The missing dataset is dated, contract-level official settlements with preliminary/final/revision status and provenance; a documented vendor mapping plus reconciliation evidence could also resolve that requirement.

## Remaining evidence and acceptance

| Field/evidence | Status after this research | Acquisition follow-up |
| --- | --- | --- |
| Broker roots, identity, exchange, currency, expiry | Must be verified from actual run responses; exchange root names do not prove broker aliases | Keep canonical and broker roots separate; reject unresolved identities. |
| Current contract sizes/ticks | Supported above | Reconcile broker point values and quote-unit encoding without changing raw records. |
| Historical multipliers/product identity | Unresolved version coverage | Obtain effective-dated exchange notices; do not backfill current rules automatically. |
| Per-contract last trade/first notice | Rule sources partly supported; historical date dataset unresolved | Acquire explicit schedules and notices; preserve event type and time zone. |
| Trading calendars | Current discovery sources supported; archived session completeness unresolved | Collect annual and exceptional schedules, including early settlements and closures. |
| Settlement provenance | Unresolved for TradeStation bars | Keep all acquired prices classified as bars; source/reconcile official settlements separately. |
| Approximately 25 years and untouched reserved test | Not established by one-year acquisition or sampled windows | Preserve existing protocol and sealed holdout. |

Historical versioning is material: [live-cattle Chapter 101](https://www.cmegroup.com/rulebook/CME/II/100/101/101.pdf) explicitly separates settlement/delivery provisions through August 2026 from October 2026 onward. The [livestock enhancement notices](https://www.cmegroup.com/company/livestock-market-enhancements.html) also identify earlier effective contract months. For older HO and RB samples, investigate product-specification/launch transitions using dated notices before treating predecessor products as interchangeable history.

An observed 90–180-day expiry gap with simultaneous bars is maturity-spacing evidence only. Even if acquisition reaches **9 roots across 3 sectors**, the metadata audit must retain **HOLD-DATA** until settlement provenance and required historical calendars/metadata are satisfied. This document creates no strategy returns, ranking or P&L.
