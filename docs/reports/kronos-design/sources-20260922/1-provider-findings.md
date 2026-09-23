# Provider findings — September 10, 2026

## Observed TradeStation evidence

One authorized, read-only authenticated GET to `/v3/marketdata/symbols/MNQM25` returned HTTP 200. The retained [response envelope](symbol-verification.json) binds request/receipt boundaries and exact response bytes to SHA256. It identifies Micro E-mini Nasdaq-100 Jun 2025, CME, expiration 2025-06-20, price increment 0.25 and point value 2. No historical bars were requested. Symbol metadata success proves this token could access that endpoint; it does not prove expired-history availability, entitlement or incremental price.

The running trader loads the plain `.access_token` through `src/data/auth_v3.py`. A read-only check found a JWT with a future expiry claim; that claim was not independently signature-verified. The provider accepted the metadata request. The separate legacy JSON cache existed but its recorded expiry was June 5, 2026. Neither credential file was written or refreshed by this work. Both existing auth helpers can refresh shared credentials, so acquisition must not call them.

The older normalized downloader uses a different `/marketdata/bars` endpoint, 100,000-bar and 70-day defaults, and a `MNQ_MULTIPLIER=0.5` constant. These are not adopted or corrected in this task. The new raw acquisition command uses the pinned barcharts ledger. Symbol metadata and [CME contract specifications](https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html) distinguish the $2 point value from the $0.50 tick value.

## Cost and entitlement worksheet

| Item | Finding |
|---|---|
| Authorized acquisition | Existing access, confirmed $0 incremental cost only |
| Symbol / expiration | Provider verified MNQM25 /2025-06-20 |
| Expired April–May 2025 bar availability | UNKNOWN; symbol lookup alone cannot establish it |
| Account-specific historical entitlement | UNKNOWN; no subscription record supplied or found in inspected pilot artifacts |
| Incremental data charge | UNKNOWN, not assumed zero |
| Calls / retries | Nine planned GETs; at most two retries each, 27 attempts maximum; minimum 60s spacing and provider Retry-After |
| Historical-rate credits | Small seven-day windows are below the documented nonzero credit threshold; credits are a rate limit, not billing evidence |
| Historical acquisition performed | None |
| Paid changes / purchases | None |

[TradeStation's historical limits](https://api.tradestation.com/docs/fundamentals/rate-limiting/historical-bar/) specify 57,600 intraday bars/request and 500 bar-chart requests per five minutes. General [market-data pricing](https://www.tradestation.com/pricing/market-data-pricing/) and [customer FAQs](https://www.tradestation.com/faqs/) describe subscription-dependent data fees. They do not identify this account's subscription or prove that these expired-contract requests are included. No public entitlement/billing query was identified in the inspected API specification; this is a search finding, not proof that no provider interface exists. A credential-free subscription record or provider confirmation is needed to complete the zero-cost worksheet.

## ProjectX account coherence

The documented [account](https://gateway.docs.projectx.com/docs/api-reference/account/search-accounts/), [open-order](https://gateway.docs.projectx.com/docs/api-reference/order/order-search-open/) and [open-position](https://gateway.docs.projectx.com/docs/api-reference/positions/search-open-positions/) examples return separate state objects and success/error fields. They show no common atomic snapshot/version identifier. Separate request success cannot prove one coherent decision-boundary account state.

The [realtime user hub](https://gateway.docs.projectx.com/docs/realtime/) offers account/order/position/trade events and reconnect/resubscribe examples. Its published payloads do not establish a common sequence, atomic initial snapshot or replay barrier spanning those entities. Our inference is that adding this stream alone would not meet the current evidence-admission contract. Account admission stays UNKNOWN; no relaxation or live subscription was made.

To resolve this blocker, provider evidence must explain initial-snapshot consistency, cross-entity ordering, missing-event detection and reconnect replay, or provide an atomic snapshot usable at the decision boundary. Mere absence of observed changes is insufficient.

## Calendar and labeling

No April–May 2025 raw archive was acquired, so archive coverage, revisions and feature warm-up remain unassessed. The existing validator retains both interval interpretations. The public TimeStamp schema alone does not settle endpoint label semantics.

[CME's current trading-hours page](https://www.cmegroup.com/trading-hours.html) focuses on 2026/2027 schedules. A [2025 Good Friday clearing notice](https://www.cmegroup.com/tools-information/holiday-calendar/files/2025/2025-good-friday-clearing-advisory.pdf) is a clearing notice, not a complete MNQ minute-session calendar. Do not substitute it, an equities calendar or the observed bar timestamps for independent April–May 2025 MNQ session coverage. Calendar admission stays UNKNOWN until the correct historical schedule is pinned.
