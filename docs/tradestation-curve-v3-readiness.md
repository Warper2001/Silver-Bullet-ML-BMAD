# TradeStation v3 coverage readiness

Assessment date: 2026-09-06. Scope: development/metadata feasibility. Verdict: **HOLD-DATA**.

## Evidence established

The [completed v3 acquisition](../data/commodity_curve/coverage-pilot-2025-20260906-v3/acquisition-report.md) contains 214 contracts and 40,375 normalized 2025 daily bars. All twelve roots across four sectors have observed bars and simultaneous pairs with broker-expiration spacing of 90–180 days. Offline verification passed for 3,059 request artifacts and six reports. The downloader's 68 unit tests passed before acquisition.

This establishes observed bar breadth, not settlement eligibility, a historical exchange calendar, or continuous history. The [metadata-only audit](../_bmad-output/specs/spec-commodity-curve-carry/metadata-audit/tradestation-v3-coverage-20260906/metadata-audit.md) still holds all seven gates. Its "usable roots: none" refers to complete required schemas, not absence of broker bars. The baseline statement at the end of the canonical data contract predates this acquisition; this report records the new bar evidence without changing its acceptance criteria.

The saved requests comprise 2,365 successful, 689 unavailable, and five invalid outcomes. Unavailable results include post-expiry months, absent early deferred-contract history, fallback aliases and historical gaps; they are not all subscription failures.

## Quote-convention reconciliation

All 214 contracts' broker `PriceFormat.PointValue` and `Increment` values match the conventions in the existing [exchange source assessment](tradestation-curve-metadata-sources.md). Decimal comparison found no mismatches. Each root has one observed format across its pilot contracts. Tick value below is point value multiplied by increment.

| Root | Broker alias | Point value USD | Increment | Tick value USD |
|---|---|---:|---:|---:|
| CL | CL | 1000 | 0.01 | 10 |
| NG | NG | 10000 | 0.001 | 10 |
| RB | RB | 42000 | 0.0001 | 4.20 |
| HO | HO | 42000 | 0.0001 | 4.20 |
| HG | HG | 25000 | 0.0005 | 12.50 |
| ZC | C | 50 | 0.25 | 12.50 |
| ZW | W | 50 | 0.25 | 12.50 |
| ZS | S | 50 | 0.25 | 12.50 |
| ZM | SM | 100 | 0.1 | 10 |
| ZL | BO | 600 | 0.01 | 6 |
| LE | LC | 400 | 0.025 | 10 |
| HE | LH | 400 | 0.025 | 10 |

Point values apply to the quote units in that assessment (including cents for grains, soybean oil and livestock). This is consistency with the recorded convention inventory, not independent historical version certification or evidence of publication times. No raw or normalized price records were changed.

## Historical samples and rejected evidence

Each root has January and July windows in each of six years: 144 root/year/month windows total. A root counts below if at least one of its two windows contains retained bars or a qualifying pair. These counts do not imply both windows passed, uninterrupted history, or an earliest available date.

| Sample year | Roots with bars | Roots with simultaneous pairs | Main limitation |
|---|---:|---:|---|
| 2000 | 1 | 1 | Only ZC has retained bars and pairs |
| 2005 | 11 | 10 | RB has no retained bars; LE has bars but no qualifying pair |
| 2010 | 12 | 12 | Sparse windows only |
| 2015 | 12 | 12 | Sparse windows only |
| 2020 | 12 | 12 | Sparse windows only |
| 2025 | 12 | 12 | Sample results are separate from the full-year pilot |

The five invalid requests are retained in the manifest:

- `CLK05`, January 2005 bars: `inconsistent_bar_ohlc`. The source snapshot remains evidence; the response supplied no accepted normalized bars.
- `ZC`, January and July 2010 fallback metadata: `metadata_identity_mismatch` among rejected candidates. The verified `C` alias supplied usable evidence for these windows.
- `LE`, January and July 2010 fallback metadata: `metadata_identity_mismatch` among rejected candidates. The verified `LC` alias supplied usable evidence for these windows.

These outcomes do not justify loosening identity or OHLC checks. Any investigation of the CL inconsistency must distinguish source semantics from genuine corruption before changing validation or reprocessing evidence.

## Settlement-source findings

TradeStation's [Daily Data help](https://help.tradestation.com/10_00/eng/tradestationhelp/data_definitions/daily_data.htm) describes daily data as settlement prices. This is supporting vendor documentation, but does not specify the v3 futures `Close` mapping, preliminary/final treatment, historical corrections, or publication timestamps for the acquired contracts. The [API HTTP example](https://api.tradestation.com/docs/fundamentals/http-requests/) demonstrates OHLC bars, not a futures settlement-vintage guarantee. The specification page did not expose substantive text through the browsing reader. API-specific lineage therefore remains unresolved, rather than disproven.

CME's [settlement access FAQ](https://www.cmegroup.com/articles/faqs/access-to-cme-group-settlement-data-faq.html) directs legacy settlement files to DataMine and maps exchange CSV/XML files to End of Market Summary offerings. Delivery timing varies by offering; it cannot be substituted for a historical record's publication timestamp. The [DataMine API](https://www.cmegroup.com/datamine/datamine-api.html) requires an entitled API ID and provides file listing and download access. No CME/DataMine-named configuration keys were found in the process environment or repository `.env`; this limited check does not establish whether the user owns a subscription elsewhere.

The public [Daily Bulletin page](https://www.cmegroup.com/market-data/daily-bulletin.html) presents the previous trade date and distinguishes preliminary and final updates. It is a discovery source, not evidence that the complete 2025 or 2000–2026 archive is available here. These web pages were inspected on the assessment date; they are documentation references, not downloaded settlement datasets or immutable publication evidence.

## Next evidence required

1. Identify an existing licensed settlement source or an export available to the user. For DataMine, list entitled files before selecting a dataset; do not infer entitlement from possession of a TradeStation token. No purchase or external message has been made.
2. Establish API-specific TradeStation daily-close lineage, or acquire official contract/session settlements with price status, publication time, retrieval time and revision/source identifiers. Reconcile contract identity and quote units against the preserved pilot. Numerical agreement alone does not establish point-in-time provenance.
3. Acquire effective-dated contract schedules covering last trade, applicable first notice, session calendars and historical product changes. Broker expiry and current rules alone are insufficient.
4. Resolve license terms, costs, margins, executable quotes/fills, continuous-history coverage and access/seal declarations before advancing the relevant gates. No returns, ranking, P&L or holdout access is authorized by this assessment.

Next dependency: the user's available historical settlement source/access. A further full broker-bar download would not itself close settlement provenance or publication-vintage gaps.

## Parent evidence hashes

SHA256 of the completed run's files, all relative to `data/commodity_curve/coverage-pilot-2025-20260906-v3/`:

| File | SHA256 |
|---|---|
| `manifest.json` | `ec882c92ba47a9ffe87cb29c8e085955d805eef60e868434acca1f53ee31e877` |
| `contracts.csv` | `2b626abc97e5839d2f44c4b93a4a84d1d2d25fb93e3b420b6b40dd0cc45390b0` |
| `acquisition-report.json` | `0e5f9cdea66b682a4004f5367c73f692438369cb8eb645eac88e908f98a3eed6` |
| `availability-report.json` | `de6a423f06b2618d13fd346afebc0e55bc6f3035003976057fa7469eb1e9db58` |
