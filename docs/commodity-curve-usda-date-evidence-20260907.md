# USDA date corroboration for July 2025 contracts

**HOLD-DATA.** USDA's 2025 Livestock Gross Margin for Swine Commodity Exchange Endorsement, marked released April 2024, supplies three explicit date examples. It is an insurance document, not an exchange calendar. [USDA original PDF](https://www.rma.usda.gov/sites/default/files/2024-07/LGM%20Swine%20Commodity%20Exchange%20Endorsement%202025.pdf).

| Acquired contract | Example event | Date | PDF page |
| --- | --- | --- | --- |
| ZCN5 | First notice | 2025-06-30 | 3 |
| ZMN5 | First notice | 2025-06-30 | 4 |
| HEN5 | Last trade | 2025-07-15 | 3 |

The corn and meal examples distinguish first notice from the June 27 first-intent dates already recorded from CME notices. The original inventory remains unchanged; this evidence is retained separately and does not automatically approve either contract for historical selection.

Offline reconciliation confirmed all three acquired contract identities, matched the hog date against vendor expiration metadata, and found records for all 13 instrument/date examples checked: June 25–27 for corn and meal, and July 3, 7–11 and 14 for hogs. No price averages or strategy outcomes were computed. Presence of records does not establish publication eligibility or completeness.

Artifacts are saved in `data/commodity_curve/usda-date-evidence-20260907/`: the original downloaded PDF, local text extraction, `source-facts.json`, `verify.py` and `verification.json`. The original bytes and dependencies are hashed. The offline verifier passed twice with identical report bytes. Web page rendering failed, but the local PDF extraction independently confirmed the three dates and release-month text.

Exact publication timestamps, the original historical version, exchange-authoritative first-notice coverage for the remaining contracts and complete product calendars remain unresolved. The printed release month is retained at month precision. No date was extrapolated to other contracts, and no additional market-data purchase or holdout access occurred.
