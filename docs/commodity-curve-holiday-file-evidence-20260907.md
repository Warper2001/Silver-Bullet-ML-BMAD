# Holiday clearing files and settlement prices

Research remains **HOLD-DATA**. The September 7 evidence pass distinguishes new settlement price formation from clearing-file distribution; it does not establish a complete trading calendar.

CME's July 4, 2025 settlement notice says no settlement prices would be derived or disseminated for its four exchanges. Its clearing advisory separately specifies prior-day prices for end-of-day settlement variation and normal distribution of clearing files. These statements describe different processes: a distributed file need not contain a newly formed price. Do not use its existence to fill a missing required settlement session. [Settlement notice](https://www.cmegroup.com/tools-information/holiday-calendar/files/2025/us-independence-day-settlement-times-2025.pdf), [clearing advisory](https://www.cmegroup.com/tools-information/holiday-calendar/files/2025/2025-4th-of-july-clearing-advisory.pdf).

January 1 differs: CME's clearing advisory explicitly specifies no settlement files in either cycle. Neither advisory supplies the complete product trading-session schedule. [New Year advisory](https://www.cmegroup.com/content/dam/cmegroup/tools-information/holiday-calendar/files/2025-new-years-advisory.pdf).

An offline scan of all 130,579 previously audited settlement-vintage rows found zero January 1 and July 4 reference-date rows. January 9 has 472 rows across all 12 roots, consistent with the earlier early-close audit. These are observations in the acquired extraction, not independent proof of clearing-file delivery or completeness.

The source facts, source hash, counts and network-blocked verification script are saved in `data/commodity_curve/holiday-file-evidence-20260907/`. Repeated verification produced identical report bytes. The source PDFs were inspected through web extraction and page rendering. A direct PDF download timed out; original PDF bytes and exact publication timestamps remain unavailable. Existing acquisition manifests and earlier audits were preserved.

Next required evidence remains the historical product session calendar and first-notice coverage through safety boundaries, plus applicable metadata and official publication vintages. No new market-data purchase, holdout access or strategy-performance calculation occurred.
