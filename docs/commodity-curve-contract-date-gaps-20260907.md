# Contract-date evidence and remaining gaps

**HOLD-DATA.** CME's April 24 notice for May 2026 adds last-trade corroboration for ten acquired contracts: May CL, NG, HO, RB, HG, ZC, ZW, ZS, ZM and ZL. Its June CL/NG entries repeat dates already sourced; these observations are retained separately. May copper first notice is April 30, 2026. [CME May delivery notice](https://www.cmegroup.com/content/dam/cmegroup/notices/clearing/2026/04/chadv26-158.pdf).

All 12 last-trade observations match the acquired vendor definitions. The new artifact records 29 event observations and keeps first intent, explicit first notice and energy notice day separate. The notice was published after the 2025 development cutoffs, so it supplies corroboration only. Exact UTC publication timestamps remain unknown.

The consolidated inventory covers 214 acquired contracts. Four older source observations concern January 2025 energy contracts outside that inventory and are excluded from its contract totals.

| Root | Acquired contracts | Exchange last-trade corroboration | Explicit exchange first notice |
| --- | ---: | ---: | ---: |
| CL | 24 | 17 | 0 |
| HE | 16 | 0 | 0 |
| HG | 24 | 16 | 16 |
| HO | 24 | 15 | 0 |
| LE | 12 | 8 | 0 |
| NG | 24 | 17 | 0 |
| RB | 24 | 15 | 0 |
| ZC | 10 | 6 | 0 |
| ZL | 16 | 10 | 0 |
| ZM | 16 | 10 | 0 |
| ZS | 14 | 9 | 0 |
| ZW | 10 | 6 | 0 |
| Total | 214 | 129 | 16 |

There remain 85 contracts without explicit exchange last-trade corroboration and 182 physically settled contracts without explicit exchange first-notice dates. The 16 HE contracts retain the current cash-rule N/A convention, pending historical rule versions. USDA's two first-notice examples and one hog last-trade example remain separate corroborating evidence, excluded from these exchange-source counts. Zero in this table describes the evidence collected, not absence of the event.

`data/commodity_curve/may-2026-date-evidence-20260907/combined-contract-gaps.csv` identifies every contract, sourced dates, URLs and missing-field reason codes. The same directory contains source facts, reconciliation checks, input/output hashes and the offline script. Repeated offline runs produced identical CSV and JSON bytes. The May PDF's web text extraction succeeded, but page rendering failed and direct download timed out; original PDF bytes are not archived.

The next acquisition targets are the historical session calendar, missing contract dates, applicable rule versions and metadata/publication vintages. [March](https://www.cmegroup.com/notices/clearing/2026/02/26-080.html) and [April](https://www.cmegroup.com/notices/clearing/2026/03/26-115.html) memo landing pages were located, but their PDF contents were not retrieved in this pass. Existing evidence remains unchanged. No extra purchase, holdout access or strategy analysis occurred.
