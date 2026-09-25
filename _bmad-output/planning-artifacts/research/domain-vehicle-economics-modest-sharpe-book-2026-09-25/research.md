---
title: 'domain research: vehicle economics for a modest-Sharpe ETF book'
type: 'domain'
topic: 'vehicle economics for a modest-Sharpe ETF book'
decision: 'Which account should hold a diversified low-frequency ETF strategy ($25-50K), and what does it realistically pay after cost and tax?'
source: 'native run'
status: complete
preset: 'standard (4 researchers)'
validation: 'normal'
created: '2026-09-25'
updated: '2026-09-25'
---

# domain research: vehicle economics for a modest-Sharpe ETF book

**Decision this research serves:** Which account should hold a diversified, low-frequency ETF strategy ($25–50K), and what does it realistically pay after cost and tax?

## Executive summary

**Pick: a personal TradeStation account, long/flat, cash-type (IRA if its API supports IRAs, otherwise a taxable cash account).** Prop firms are out for this strategy.

1. **No futures prop firm allows multi-week holds.**
   - Topstep requires all positions closed daily and says "No swing trading" [9]. Apex, Alpha and Tradeify impose the same daily flat rule [11][12][13].
   - Topstep also bans automation in Live Funded accounts: "Live funded accounts are not allowed to trade through the ProjectX API" [10] (verified verbatim). **This also blocks the MNQ combine's automated bots after promotion to Live.**
   - The one swing-capable equity prop, Trade The Pool, has "beta" automation, a 30% best-position cap and a 14-day inactivity rule [14]. All three conflict with a few-big-winners monthly book.
2. **TradeStation's personal account costs are low; its idle-cash treatment is the real frictional cost.**
   - US ETF commissions are $0. A $0.003/share clearing fee applies at Tier 1, but its scope is unclear [1]. The SEC Section 31 fee is $20.60 per $1M of sales [3].
   - **Cash earns 0% below $100K, and never in an IRA** [1]. A long/flat book that's often partly in cash should hold a T-bill ETF instead of broker cash.
   - The API supports Cash and Margin accounts with market or limit orders at the close (`CLO`). **IRA via the API and fractional shares are both unverified** [2].
   - FINRA's pattern-day-trader $25K rule was replaced on 2026-06-04 [4][5] (verified). It's irrelevant at monthly rebalancing anyway.
3. **Taxes favour an IRA.**
   - In a taxable account, monthly trend realizes mostly short-term gains at ordinary rates [6]. Wash sales defer losses on quick re-entries [7]. GLD is taxed at the 28% collectibles rate [6][16].
   - An IRA removes current tax on trades [8], and long-only fits it.
   - A monthly strategy is unlikely to qualify for trader tax status [15]. **That and several other items are CPA questions.**
4. **Expect little.**
   - Over the last decade diversified trend delivered roughly **Sharpe 0–0.4**: the SG CTA Index had Sharpe 0.07 and DBMF's strategy 0.39 over 2016-07 to 2026-06 [17] (issuer deck, single source).
   - Two academic critiques weaken the classic case: time-series momentum is weak asset-by-asset [19], and its alpha comes mostly from volatility scaling [20]. Post-publication decay averages 58% [21].
   - At Sharpe 0.5, a $25–50K book earns about $1,250–2,500 a year of excess return, and loses money in about 31% of years (local calculation, §Income).

**Biggest caveat:** several load-bearing TradeStation items are unverified: IRA through the API, the scope of the clearing fee, fractional shares, and the terms of service on automation. They must be confirmed with TradeStation before A5, the paper-trading step.

## D1 — TradeStation personal-account terms

- **Commissions:** US clients pay $0 per share for stocks and ETFs [1].
- **Clearing fee:** "clearing fee" $0.003/share at Tier 1. Whether it applies to all orders or only direct-routed ones is unclear [1] (medium confidence). Even if it applies to everything, a $50K book trading about $10K a month at around $100/share pays about $0.30 a month.
- **Inactivity fee:** $10/month unless "minimum activity" is met; the criteria aren't stated [1].
- **IRA fees:** $35 a year, plus $50 on termination [1].
- **Interest on cash:** 0.15%, only on balances over $100K and never in an IRA [1].
- **Margin:** 11.75% on debits under $50K.
- **Shorting:** ETF shorts need a margin account with at least $2,000 equity; borrow fees "vary" [1].
- **Regulatory fees:** the SEC Section 31 fee is $20.60 per $1M of sales from 2026-04-04 [3]. The FINRA TAF rate wasn't retrieved (unverified).
- **PDT:** replaced by FINRA's intraday margin requirements, effective 2026-06-04, with a transition to 2027-10-20. There is no $25K minimum; $2K is the margin minimum [4][5] (verified by two sources).
- **Cash-account settlement:** a good-faith violation happens only if a security bought with unsettled proceeds is sold before those proceeds settle. Selling one ETF and buying another the same day is fine for monthly holds [22] (summary-level, medium confidence).
- **API (v3):**
  - Order types: Market, Limit, StopMarket, StopLimit.
  - Durations include `OPG` and `CLO`, so market-on-close and limit-on-close exist (mapping inferred).
  - Account types: Cash, Margin, Futures, DVP. **No IRA type is documented, and fractional shares aren't documented** [2].
  - SIM fills instantly, so it won't show how closing-auction orders really fill [2].

## D2 — Taxes (US federal; CPA questions flagged)

- **Short-term vs long-term:** short-term gains (held one year or less) are taxed as ordinary income. Net capital losses offset only $3,000 a year of other income [6].
- **Wash sales:** a loss is disallowed if substantially identical securities are bought within 30 days before or after the sale (§1091(a)). The loss is added to the new shares' cost basis, so it's deferred, not lost [7]. Whether different ETFs on the same index are "substantially identical" is a **CPA question**.
- **Commodity ETFs:** PDBC is a 1940 Act fund with no K-1 but may "recognize more ordinary income" [18]. DBC's K-1 and 60/40 status rests on secondary sources only (medium confidence).
- **Gold:** GLD long-term gains are taxed at up to 28% as collectibles [6][16].
- **Futures:** Section 1256 contracts get 60/40 treatment and are marked to market at year end [23].
- **Prop payouts:** Topstep traders are independent contractors, paid on a 1099-NEC [24] (medium confidence). Self-employment tax is a **CPA question**.
- **IRA:** tax-exempt (§408(e)(1)). Pledging the account as loan security is treated as a distribution, which is why true margin doesn't fit an IRA [8]. Funding requires a rollover; the plan's capital exceeds annual contribution limits.
- **Trader tax status:** Topic 429 sets qualitative tests; a monthly strategy is unlikely to qualify [15]. This is a **CPA question**.

## D3 — Prop-firm fit for multi-week, automated holds

| Firm | Overnight / multi-week holds | Automation | Source |
|---|---|---|---|
| Topstep | **No**: close by 3:10 PM CT, "No swing trading" | Allowed in Combine and XFA; **not in Live Funded** (verified) | [9][10] |
| Apex | No: close by 4:50 PM ET | "strictly prohibited on all account types" (snippet) | [11] |
| Alpha Futures | No: close by 4:20 PM EST | Prohibited on all account types (page dated 2025-10) | [12] |
| Tradeify | No: flat by 4:45 PM ET | Personal bots allowed (snippet) | [13] |
| Trade The Pool (equities) | **Yes**, on Swing accounts | "Beta", with approval; 30% best-position cap; 14-day inactivity rule; 70/30 split | [14] |

**Conclusion:** prop firms don't fit a monthly-rebalanced ETF book.

## D4 — The do-nothing baselines

| Fund | Inception | Fee | 3-year (to 2026-08-31) | Since inception | Source |
|---|---|---|---|---|---|
| CTA (Simplify) | 2022-03 | 0.75% | 10.42% | 8.85% | [25] |
| KMLM | 2020-12 | 0.90% | **0.42%** | 7.61% | [26] |
| RPAR (risk parity) | 2019-12 | 0.52% | 8.07% (5-year: 1.23%) | 4.31% | [27] |

- CTA reports on a 1099, not a K-1 [25].
- KMLM trails its own index by about 1.9 points a year since inception [26].
- DBMF's strategy (a decade that includes pre-ETF history): 6.6% a year, Sharpe 0.39. The SG CTA Index returned 3.0% a year at Sharpe 0.07 over the same period [17] (issuer deck).
- **Fund choice is itself a big bet:** KMLM made 0.42% a year and CTA 10.42% over the same three years.
- These funds go long/short with leverage. A long/flat ETF book is closer to Faber's tactical asset allocation (in-sample Sharpe about 0.81, 1972–2005 [28]). The out-of-sample figure of about 0.68 for 2006–2025 comes from a blog summary only (low confidence).

## Income (local calculation: `income_math.py` / `.json`; arithmetic, not a research claim)

Figures assume 10% portfolio volatility. They are before tax and before the T-bill yield on cash.

| Sharpe | $/yr at $25K | $/yr at $50K | P(losing year) | Median / worst-1-in-10 drawdown over 10 years | Capital needed for $240K/yr |
|---|---|---|---|---|---|
| 0.3 | 750 | 1,500 | 38% | 22% / 36% | $8.0M |
| 0.5 | 1,250 | 2,500 | 31% | 19% / 30% | $4.8M |
| 0.7 | 1,750 | 3,500 | 24% | 16% / 26% | $3.4M |
| 1.0 | 2,500 | 5,000 | 16% | 14% / 21% | $2.4M |

## Cross-dimension insights

- **The vehicle question and the edge question collapse into one for low-frequency strategies.** Prop firms can't hold the positions, so the only vehicle is personal capital. Its payoff scales with capital times Sharpe, so at $25–50K the dollar stakes are small whatever the Sharpe.
- **Automation risk runs one way.** Topstep's Live Funded API ban means the current MNQ combine path also can't stay automated past the funded stage.
- **An IRA is the best vehicle on taxes but the least documented through the API.** Settle that one question first.

## Recommendations

1. **Vehicle for ETFTM-1: a TradeStation personal account, long/flat, gross exposure ≤ 1.0.** IRA if TradeStation confirms API trading in IRAs; otherwise a taxable cash account. Hold a T-bill ETF instead of broker cash, since cash earns 0% below $100K. Confidence: high on prop exclusion, medium on the IRA-via-API question.
2. **Proceed with A, framed as research-process value (goal 6) plus a possible foundation, not income.** A's test, with a per-asset intercept, directly addresses the Huang et al. critique. Expect UNDERPOWERED or FAIL, and treat either as a cheap, valid outcome. Confidence: medium.
3. **Record the Topstep Live-Funded API ban** as a goal-5 constraint on the MNQ combine route. Confidence: high (verified verbatim).
4. **Before A5, get written TradeStation answers** on: IRA via the API, the clearing fee's scope, fractional shares, inactivity criteria, and any terms-of-service clause on automation. These are appended to `research/project_goals/account-questions.md`.

## Open questions

- How TradeStation's clearing fee applies; whether IRA accounts can trade through the API; fractional shares; the automation terms of service; the market-data fee table (JS-rendered).
- CPA: wash sales across ETFs on the same index; IRA replacement purchases (Rev. Rul. 2008-5); self-employment tax on prop payouts; trader tax status.
- DBMF's ETF-only fee, AUM and tax form; the numeric Sharpe from Hurst/Ooi/Pedersen and MOP; any post-publication decay figure specific to futures trend.

## Source appendix

| # | Supports | Publisher | Pub date | Accessed | Confidence |
|---|---|---|---|---|---|
| [1] | TradeStation fees, cash interest, shorting, inactivity | [TradeStation pricing](https://www.tradestation.com/pricing/) | 2026 | 2026-09-25 | medium (403 on re-check) |
| [2] | TradeStation API order types, account types | [TradeStation API spec](https://api.tradestation.com/docs/specification) | 2026 | 2026-09-25 | medium |
| [3] | SEC Section 31 fee | [SEC fee advisory 2026-2](https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2) | 2026-02-27 | 2026-09-25 | high |
| [4] | PDT replaced by intraday margin | [FINRA investor insight](https://www.finra.org/investors/insights/intraday-margin-requirements) | 2026-04-20 | 2026-09-25 | high |
| [5] | PDT change, second source | [FINRA Regulatory Notice 26-10](https://www.finra.org/rules-guidance/notices/26-10) | 2026 | 2026-09-25 | high |
| [6] | ST/LT rates, $3K limit, 28% collectibles | [IRS Topic 409](https://www.irs.gov/taxtopics/tc409) | 2026-09-24 | 2026-09-25 | high |
| [7] | Wash sale §1091 | [26 USC 1091 (LII)](https://www.law.cornell.edu/uscode/text/26/1091) | current | 2026-09-25 | high |
| [8] | IRA exemption, loan pledge | [26 USC 408 (LII)](https://www.law.cornell.edu/uscode/text/26/408) | current | 2026-09-25 | high |
| [9] | Topstep: no overnight, no swing | [Topstep help 8284206](https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade) | 2026-07-13 | 2026-09-25 | high |
| [10] | Topstep: no API in Live Funded | [Topstep help 11187768](https://help.topstep.com/en/articles/11187768-topstepx-api-access) | 2026 | 2026-09-25 | high (verified) |
| [11] | Apex hours, automation ban | [Apex prohibited activities](https://support.apextraderfunding.com/hc/en-us/articles/40463668243099-Prohibited-Activities) | 2026 | 2026-09-25 | medium (snippet) |
| [12] | Alpha hours, automation ban | [Alpha Futures help](https://help.alpha-futures.com/en/articles/9508585-prohibited-trading-practices) | 2025-10-30 | 2026-09-25 | medium |
| [13] | Tradeify hours | [Tradeify help](https://help.tradeify.co/en/articles/10495876-rules-permitted-times-to-trade) | 2026 | 2026-09-25 | medium (snippet) |
| [14] | Trade The Pool swing terms | [Trade The Pool program terms](https://tradethepool.com/program-terms/) | 2026 | 2026-09-25 | high |
| [15] | Trader tax status | [IRS Topic 429](https://www.irs.gov/taxtopics/tc429) | current | 2026-09-25 | high |
| [16] | GLD collectibles treatment | [SEC EDGAR, GLD filing](https://www.sec.gov/Archives/edgar/data/0001222333/000119312514192745/d725076dfwp.htm) | 2014 | 2026-09-25 | medium |
| [17] | DBMF strategy and SG CTA figures, 2016–2026 | [iMGP DBi presentation](https://www.imgp.com/documents/iMGP_DBi_Managed_Futures_Presentation.pdf) | 2026 | 2026-09-25 | medium (issuer) |
| [18] | PDBC tax structure | [Invesco PDBC tax doc](https://www.invesco.com/us-rest/contentdetail?contentId=30764e28-1344-4e69-9e99-2bad61fa3a53) | 2024-12-20 | 2026-09-25 | high |
| [19] | Time-series momentum weak asset-by-asset | [Huang, Li, Wang, Zhou, JFE 2020](https://ideas.repec.org/a/eee/jfinec/v135y2020i3p774-794.html) | 2020 | 2026-09-25 | high |
| [20] | Vol scaling drives the alpha | [Kim, Tse, Wald, SSRN 2786955](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2786955) | 2016 | 2026-09-25 | medium |
| [21] | Post-publication decay 58% | [McLean & Pontiff, JF 2016](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365) | 2016 | 2026-09-25 | high |
| [22] | Cash-account settlement violations | [Schwab: cash-trading violations](https://www.schwab.com/learn/story/avoid-these-violations-when-trading-cash) | n/d | 2026-09-25 | medium (summary) |
| [23] | §1256 60/40 treatment | [26 USC 1256 (LII)](https://www.law.cornell.edu/uscode/text/26/1256) | current | 2026-09-25 | high |
| [24] | Topstep payouts on 1099 | [Topstep funded-trader tax](https://help.topstep.com/en/articles/8284238-funded-trader-tax-questions) | 2026 | 2026-09-25 | medium (snippet) |
| [25] | CTA fund data | [Simplify CTA](https://www.simplify.us/etfs/cta-simplify-managed-futures-strategy-etf) | 2026-08-31 | 2026-09-25 | medium (issuer) |
| [26] | KMLM fund data | [KraneShares KMLM](https://kraneshares.com/kmlm) | 2026-08-31 | 2026-09-25 | medium (issuer) |
| [27] | RPAR fund data | [RPAR ETF](https://www.rparetf.com/rpar) | 2026-08-31 | 2026-09-25 | medium (issuer) |
| [28] | Faber tactical asset allocation | [Faber, SSRN 962461](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461) | 2007/2013 | 2026-09-25 | medium |

## Staleness map

Fastest-ageing items:

| Items | Re-check within |
|---|---|
| TradeStation pricing [1] and prop-firm rules [9]–[14] | ≤ 3 months, by **2026-12-25** |
| Fund figures [25]–[27] and the issuer deck [17] | ≤ 3 months |
| FINRA margin transition [4][5] | when TradeStation implements it (by 2027-10-20) |
| Academic claims | slow-moving |

This is a select-shaped report, so refresh it after two quarters (by 2027-03-25) before acting on it.
