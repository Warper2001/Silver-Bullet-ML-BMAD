# Digest r1-1: TradeStation equities/ETF account economics and frictions (as of 2026-09-25)

Researcher: r1-ts-equities. Accessed: 2026-09-25 for all sources. Tool calls: 18 of 18. Sources: 7 (5 read directly, 2 from search-result summary only).

Source IDs
- S1 https://www.tradestation.com/pricing/ | TradeStation Securities | no page date; (c) 2026 footer; read directly (raw HTML) | primary, pricing
- S2 https://www.tradestation.com/pricing/market-data-pricing/ | TradeStation | no page date (doc code "D1225"); read directly, but the Equities tab is JS-rendered and did not come through | primary
- S3 https://api.tradestation.com/docs/specification (the v3 OpenAPI spec embedded in the docs JS bundle, chunk 42059736.*.js) plus /docs/faq and /docs/fundamentals/sim-vs-live | TradeStation | undated; read directly | primary, capability
- S4 https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2 | SEC | 2026-02-27 | primary, regulatory
- S5 https://www.finra.org/investors/insights/intraday-margin-requirements | FINRA | 2026-04-20 | primary, regulatory (via WebFetch summary with quotes)
- S6 https://www.tradestation.com/learn/market-basics/stocks-etfs/day-trading-rules/day-trading-rules-and-cash-accounts/ | TradeStation | undated | only the search-result summary was seen; the page was not opened
- S7 Schwab (https://www.schwab.com/learn/story/avoid-these-violations-when-trading-cash) and Fidelity (https://www.fidelity.com/learning-center/trading-investing/trading/avoiding-cash-trading-violations) | broker education | undated | only the search-result summary was seen; count as a second, non-TradeStation publisher for GFV/free-riding definitions

## Claims

### Pricing

1. **Claim:** "TS SELECT" and "TS GO" no longer appear on the pricing page. Pricing now uses volume tiers, Tier 1 to Tier 4, reset monthly. Each asset class has its own tier, and the page has separate "U.S. Clients" and "Clients Outside the U.S." tabs. | S1 | TradeStation | n.d. (2026) | high | pricing
2. **Claim:** Stocks and ETFs, U.S. clients: $0 commission per share. The second (non-U.S.) tab shows a $5.00 ticket charge per trade. Matching the $0 to U.S. and the $5 to non-U.S. is inferred from tab order. | S1 | TradeStation | n.d. | high for $0 to U.S., medium for the tab mapping | pricing
3. **Claim:** Other stock/ETF trading fees, for Tier 1 (0 to 100,000 shares a month) through Tier 4 (more than 10M):
   - Sub-dollar/OTC: $0.005 per share.
   - Direct routing: $0.0048, $0.0046, $0.0043, $0.0032 per share.
   - "Clearing fee (per share)": $0.003, $0.002, $0.001, $0.
   - A $25K–$50K monthly rebalancer is Tier 1. It is **unclear from the page whether the $0.003/share clearing fee applies to all equity orders or only to direct-routed ones.**
   - | S1 | TradeStation | n.d. | high for the numbers, low for how the clearing fee applies | pricing
4. **Claim:** Account fees:
   - Inactivity fee: $10/month, "FREE if account meets minimum activity". The criteria are not stated on the page.
   - IRA: $35 annual administration fee and a $50 termination fee.
   - Outgoing account transfer: $125. Sending a wire: $25 (U.S. tab).
   - Mutual funds: $14.95 per transaction.
   - | S1 | TradeStation | n.d. | high | pricing
5. **Claim:** Margin interest runs 11.75% below $50,000 debit, 10.75% for $50,000–$499,999, 6.25% for $500K+ and 4.25% for $2M+. The rate is set off a base rate that TradeStation can change at its discretion. | S1 | TradeStation | n.d. | high | pricing
6. **Claim:** Interest on uninvested cash is 0.15% APR, and only on free credit balances **over $100,000** in **non-IRA** securities and futures accounts. No interest is paid on days below that threshold. So a $25K–$50K account, or any IRA, earns 0 on cash. The page mentions no sweep program. | S1 | TradeStation | n.d. | high | pricing
7. **Claim:** Market data: "Certain basic market data is available at no charge to non-professional subscribers. Additional market data fees may apply for added markets." Optional real-time fees are debited monthly in advance and are not prorated. The specific equities (non-pro) package prices were not extracted. | S1, S2 | TradeStation | n.d. | high for the wording, the equities fee amounts are **unverified** | pricing
8. **Claim:** Short selling:
   - Short stock/ETF needs 30% intraday / 50% overnight margin, and is "Not available" in cash accounts.
   - At least $2,000 is required to open and maintain a short stock position.
   - "Short debit fees: Fees vary." Some symbols carry special margin (list at my.tradestation.com/lists/borrow-special-margin).
   - | S1 | TradeStation | n.d. | high | pricing/policy

### Regulatory

9. **Claim:** Section 31 fee: from 2026-04-04 the SEC rate is **$20.60 per million dollars** of covered sales. Sales on charge dates through 2026-04-03 were $0.00 per million. The rate stays in effect until 60 days after the FY2027 appropriation is enacted. It applies to sales only. | S4 | SEC | 2026-02-27 | high | regulatory
10. **Claim:** The FINRA pattern-day-trader rule was replaced by intraday margin requirements, **effective 2026-06-04, with a permitted transition period through 2027-10-20**:
    - "There's no 'pattern day trader' designation based on counting trades."
    - "There's no $25,000 minimum equity requirement for day trading."
    - $2,000 is the minimum equity for margin trading.
    - Maintenance margin is 25% of current market value, monitored in real time.
    - | S5 | FINRA | 2026-04-20 | high (a primary source, read through a WebFetch summary that quoted it) | regulatory
11. **Claim (second source for 10):** TradeStation's pricing page now describes the new regime:
    - With margin equity at or above $2,000, intraday buying power is 4x margin excess and overnight buying power is 2x. Below $2,000, both are 1x.
    - $2,000 is the minimum to open and maintain a margin position.
    - The page links to the FINRA intraday-margin page.
    - The page does not mention $25K or a PDT count.
    - **Caveat:** the v3 API spec still exposes `PatternDayTrader`, `DayTrades` and `CanDayTrade` fields, described with the old "4 or more times in 5 business days" rule. That may be stale documentation.
    - | S1, S3 | TradeStation | n.d. | medium-high that TradeStation has already adopted the regime, since the transition window allows a firm to still apply the old rule | regulatory/policy
12. **Claim:** Relevance to a monthly rebalance: a sell-then-buy of *different* ETFs, or buying and holding, is not a day trade. A day trade needs an open and a close of the same security in one session. Under the post-June-2026 regime, day trades are no longer counted anyway. So PDT is not a binding constraint for this use case in a margin account at $25K–$50K. | S5, S1 (definitions); applying them to this use case is analysis | FINRA/TradeStation | 2026 | high | regulatory
13. **Claim:** Cash-account rules. Day trading in a TradeStation cash account is allowed only with settled funds, to avoid Reg T free-riding.
    - A good-faith violation (GFV) happens when a security bought with unsettled sale proceeds is **sold before those proceeds settle**.
    - Free-riding is buying without settled cash and paying for the purchase by selling that same security. It triggers a 90-day freeze.
    - Four GFVs in 12 months leads to an account restriction.
    - | S6 (TradeStation, summary only), S7 (Schwab/Fidelity, summary only) | TradeStation + other broker | n.d. | medium, because the pages were not opened | regulatory
14. **Claim:** Relevance to a monthly rebalance in a cash account: selling ETF A and buying ETF B with the unsettled proceeds on the same day is **not** a GFV. It becomes one only if B is sold before A's sale settles (T+1). Holding for a month avoids GFVs. The rebalance cannot buy before the sale fills (no margin float), and proceeds cannot be withdrawn until T+1. | Derived from S6/S7 definitions; T+1 settlement **not retrieved this run** (see "Looked for") | — | medium | regulatory
15. **Claim:** Accounts and minimums:
    - Margin needs $2,000 minimum equity (S1).
    - IRAs offered: Traditional, Roth, Rollover and SEP. Zero-commission equities in IRAs (TradeStation accounts and retirement pages, tradestation.com/accounts/retirement/, read directly).
    - Cash-account and IRA opening minimums were **not found**.
    - | S1 + retirement page | TradeStation | n.d. | high for $2K margin, gap otherwise | policy

### API capability (v3)

16. **Claim:** The v3 order request takes OrderType ∈ {Market, Limit, StopMarket, StopLimit} and TimeInForce.Duration ∈ {DAY, DYP, GTC, GCP, GTD, GDP, **OPG, CLO**, IOC, FOK, 1/3/5 MIN}.
    - CLO means "On Close; orders that target the closing session of an exchange".
    - OPG is "only valid for listed stocks at the opening session".
    - So MOC is Market + CLO and LOC is Limit + CLO. That mapping is inferred from the spec text, and no explicit MOC/LOC wording was found.
    - "Allowed durations vary by Asset Type."
    - | S3 | TradeStation | n.d. | high for the enums, medium for the MOC/LOC mapping | capability
17. **Claim:** v3 AccountType values are **Cash, Margin, Futures, DVP**. There is no IRA account type in the API schema, so an IRA presumably appears as a Cash type. That is **unverified**: no spec text mentions IRA or retirement accounts either way. The spec also exposes account Status codes including "90 Day Restriction-Closing Transaction Only", the free-riding freeze. | S3 | TradeStation | n.d. | high for the schema, low for the IRA inference | capability
18. **Claim:** Fractional shares: the v3 spec's `Quantity` is a string with no mention of fractional or notional equity orders. The only "fractional" hits are price-display and rate-limit credit fields. **There is no documented fractional-share order support via the API.** This is an absence of evidence, not a confirmed prohibition. | S3 | TradeStation | n.d. | medium | capability
19. **Claim:** SIM API at https://sim-api.tradestation.com/v3: "identical to the Live API in all ways except it uses fake trading accounts seeded with fake money... only simulated executions occur with instant 'fills'." So SIM will not model the closing-auction (CLO) fill realism or slippage. | S3 | TradeStation | n.d. | high | capability
20. **Claim:** Getting an API key requires a funded TradeStation account and an email to ClientExperience@tradestation.com. The docs FAQ lists no API fee. | S3 (FAQ) | TradeStation | n.d. | high | capability/policy

## Leads

- Clarify whether the Tier-1 $0.003/share "clearing fee" on S1 applies to every equity order or only to direct-routed orders. It is the main open commission-equivalent friction. Ask TradeStation or read the Commission/Fee Schedule PDF on the Agreements & Disclosures page.
- FINRA TAF current rate: try finra.org/rules-guidance/guidance/trading-activity-fee through WebFetch. Curl gets a 403 from finra.org, but WebFetch worked for S5.
- TradeStation's help center or support article on API trading in IRA accounts. Also check whether IRA accounts return AccountType "Cash" from GET /v3/brokerage/accounts. The quickest test is empirical: the operator's own API key against a TradeStation IRA, if one exists.
- Test MOC/LOC on the SIM API with a Market+CLO order on an ETF, then check the live 3:50pm ET closing-order cutoff that TradeStation applies (NYSE/Nasdaq close cutoffs).
- Inactivity-fee "minimum activity" criteria. A monthly rebalance probably qualifies, but that is unverified.
- The S2 equities tab needs a JS-rendering fetch to get non-pro NYSE/Nasdaq/OPRA fees, and whether API streaming quotes need a paid subscription.
- FINRA PDT transition: confirm the date TradeStation actually switched (a TradeStation notice or blog, 2026-06). S1's wording suggests it already has.

## Looked for, could not find

- The TS SELECT / TS GO plan structure: it is no longer on the pricing page (replaced by tiers). No current page explains what happened to it.
- FINRA TAF rate: finra.org returned 403 to curl, and the budget ran out before a WebFetch retry. My belief is about $0.000166/share for sales, with a per-trade cap, but that is **unverified this run**.
- An SEC/Investor.gov primary page for T+1 settlement (effective 2024-05-28) and for free-riding: investor.gov returned 403. T+1 is **unverified this run** beyond the settlement references in S6/S7.
- Minimum opening deposit for cash and IRA accounts.
- Any TradeStation terms-of-service or API-agreement clause on automated trading for personal accounts. It was not searched due to budget, and the docs FAQ and SIM pages have none.
- Explicit MOC/LOC wording in the v3 docs. Only the CLO duration was found.
- An explicit statement on fractional shares via the API, in either direction.
- Specific borrow-fee rates for ETF shorts ("Fees vary").
