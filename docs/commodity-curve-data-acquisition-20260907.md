# Commodity curve data acquisition — 2026-09-07

Research status: **HOLD-DATA**. The three Databento historical datasets were downloaded for the already accessed 2025 development pilot after explicit authorization of up to **$1**. The refreshed combined quote was **$0.622945073992**; the final account charge has not been independently confirmed. Reserved holdout data and acquired bar files were not read during this acquisition task.

## Acquired evidence

Local artifacts are in `data/commodity_curve/source-evidence-20260907/`:

- Databento dataset availability, schema fields, symbol-resolution responses and cost estimates. All 214 pilot contracts have a nonempty raw-symbol resolution in 2025; natural-gas year-suffix aliases required explicit resolution. Resolution establishes identifiers, not complete settlement coverage.
- `cme-delivery-date-facts-2025.json` and `.csv`: 237 event facts for 99 delivery contracts across 11 roots, extracted from 12 official monthly CME clearing notices. Each observation retains its source URL, page and notice date. First intent, first notice, notice day and last trade retain distinct meanings.
- `cme-holiday-settlement-facts-2025.json`: nine no-settlement holiday dates and four product-specific settlement-time observations. These are incomplete historical calendar evidence. Settlement determination times are neither dissemination timestamps nor session closing times.
- `acquisition-status.json` and `manifest.json`: count/uniqueness validation and verified SHA-256 hashes for all saved local artifacts, including the three native Databento files. Hashes identify saved artifacts, not original CME PDF bytes.

Original CME PDF downloads failed. Dates were read through web extraction of official PDFs; no original PDFs were archived. Exact UTC publication timestamps remain unknown. The Presidents Day notice has a stale update footer and the Thanksgiving dairy note contains an inconsistent weekday/date; neither is used to infer publication time.

## Completed download

`download-plan.json` contains exact requests and quote receipts for GLBX.MDP3, 2025-01-01 inclusive through 2026-01-01 exclusive, restricted to 214 resolved outright symbols:

| Schema | Estimated USD | Billable bytes |
| --- | ---: | ---: |
| statistics | 0.506778210402 | 544148960 |
| definition | 0.042222384363 | 26668200 |
| status | 0.073944479227 | 19849320 |
| Total | 0.622945073992 | 590666480 |

The quote was refreshed before downloading and remained within the approved $1 budget. Each request completed once, without a paid retry. Native records retain all returned updates, timestamps and flags; settlement status, session attribution and price conventions still need validation before normalization. This quote excludes executable bid/ask history and additional years.

Databento documents exchange settlement retrieval through its [statistics schema](https://databento.com/docs/examples/futures/retrieving-oi-and-settlement-prices). The retrieved dataset range begins June 2010, so this source alone cannot supply the approximately 25 years implied by a 20% reserved test segment lasting five years.

## Remaining acceptance gaps

Settlement time series have been acquired; their vintage semantics and complete required-session coverage still need validation. Full exchange session coverage, exact publication evidence, CBOT/cattle first-notice semantics, HE cash-settled last-trade evidence and deferred 2026 contract schedules remain incomplete. A future execution study also needs causal quote, cost, margin and stress inputs. These artifacts support source reconciliation; they do not establish historical strategy performance or lift HOLD-DATA.

Verification passed for local artifact hashes, event-key uniqueness, date parsing, expected source/record counts, distinct symbols per request and exact decimal quote totals. No application code changed, so the existing mechanics suite was not rerun for this acquisition-only task.


## Download verification

All three native DBN files passed SHA-256 receipt checks and full record parsing with no parser warnings or truncation detected:

| Schema | Records | Distinct instrument IDs | Saved bytes |
| --- | ---: | ---: | ---: |
| statistics | 6801862 | 214 | 157658032 |
| definition | 51285 | 214 | 242588 |
| status | 496233 | 214 | 851076 |

The statistics file includes **130579 settlement records across 214 instruments**, with raw settlement flags 1, 2, 3 and 5 retained. These counts include updates and do not imply one accepted final settlement per required session. Definition and status event timestamps can precede the requested range; reception and effective-time interpretation must be validated before using them causally.

Databento flagged **2025-09-17, 2025-09-24 and 2025-11-28** as degraded. The metadata response is preserved in `databento-2025-dataset-condition.json`; dataset condition is not a certified exchange-session calendar. No strategy returns were calculated.

Exact requests, refreshed quotes, per-file receipts, record counts and hashes are saved locally in `approved-download-quotes.json`, `databento-2025-*.download.json`, `download-verification.json` and `manifest.json`. The local `verify_download.py` script can recheck receipt hashes and parse files (using saved prior parse checks for unchanged statistics/definition files). Compressed saved sizes differ from quoted billable sizes.


## Follow-up settlement audit

The [2025 settlement audit](commodity-curve-settlement-audit-2025-20260907.md) now documents flag semantics, cutoff availability, revisions, and 95 matching last-trade date comparisons. It passed 155 independent checks and repeated-output checks. Calendar completeness, price-unit reconciliation and degraded-source dates keep the research at HOLD-DATA.
