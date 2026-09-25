# Digest r2-taxes-r1-1 — US federal tax treatment: ETF trend (taxable vs IRA) vs futures vs prop payouts

Scope: US federal only. Accessed 2026-09-25 for all sources. 17 tool calls, 10 sources retrieved.
Not tax advice. Items marked `cpa-question` are open questions, not findings.

## Claims

### Q1 — Short- vs long-term capital gains; holding period
1. **Claim:** Long-term treatment requires holding an asset *more than one year*. Otherwise the gain or loss is short-term. Verbatim: "Generally, if you hold the asset for more than one year before you dispose of it, your capital gain or loss is long-term."
   - Source: https://www.irs.gov/taxtopics/tc409 · IRS (Topic 409) · last reviewed 2026-09-24 · accessed 2026-09-25 · confidence HIGH · class regulatory
2. **Claim:** Short-term gains are taxed as ordinary income at graduated rates. Long-term rates are 0/15/20%. The 2025 brackets shown: 0% up to $48,350 (single) / $96,700 (MFJ); 15% up to $533,400 (single) / $600,050 (MFJ); 20% above. The page still showed 2025 thresholds; 2026 figures were not retrieved.
   - Source: same (tc409) · confidence HIGH (rates) / MEDIUM (bracket year) · regulatory
3. **Claim:** Net capital losses offset ordinary income only up to $3,000 a year; the rest carries forward indefinitely. The Net Investment Income Tax (NIIT) may also apply (Topic 559, not fetched).
   - Source: same (tc409) · HIGH · regulatory
4. **Inference (not a separate source):** A monthly-rebalanced trend strategy that exits most positions within 12 months will realize mostly short-term gains, taxed at ordinary rates in a taxable account. This follows directly from claims 1–2. Positions held more than one year get long-term rates.

### Q2 — Wash sales (§1091)
5. **Claim:** A loss is disallowed if, within the window from 30 days before to 30 days after the sale, the taxpayer acquires "substantially identical stock or securities", or enters a contract or option to acquire them. The only exception is a dealer acting in the ordinary course of business. So a trend strategy that sells an ETF at a loss and re-buys the same ETF within 30 days has that loss disallowed.
   - Source: https://www.law.cornell.edu/uscode/text/26/1091 · Cornell LII (IRC text) · current code · HIGH · regulatory
6. **Claim:** The disallowed loss is not lost permanently. §1091(d) adds it to the basis of the replacement shares: the new basis is the old basis, adjusted by the difference between the repurchase price and the sale price. Pub 550 (2025) states the same 30-days-before-or-after rule.
   - Sources: 26 USC 1091(d) (above); https://www.irs.gov/publications/p550 · IRS · 2025 edition · HIGH · regulatory
7. **Claim:** The statute does not define "substantially identical". Neither fetch of Pub 550 returned a definition or any ETF-specific guidance. Whether two different ETFs tracking the same index are "substantially identical" is not settled by any primary source retrieved this run.
   - Source: 26 USC 1091 (no definition in text) · HIGH (that the statute is silent) · class **cpa-question** (whether same-index ETFs from different sponsors are substantially identical)

### Q3 — Commodity ETFs: partnership (K-1) vs '40 Act (1099)
8. **Claim:** PDBC is "a 1940 Act commodity fund" structured "in a manner that avoids generating a K-1 and K-3". It gets commodity exposure through a Subsidiary (the Cayman subsidiary is named only in the secondary source). This "may cause the Fund to recognize more ordinary income". The fund's RIC status depends on futures income counting as qualifying income: "If the IRS were to determine that the Fund's income is derived from the futures did not constitute qualifying income…" it would have to cut that exposure. PDBC paid a $0.57471/share income distribution in Dec 2024.
   - Source: https://www.invesco.com/us-rest/contentdetail?contentId=30764e28-1344-4e69-9e99-2bad61fa3a53 · Invesco (sponsor PDF) · 2024-12-20 · HIGH · regulatory (sponsor tax doc)
9. **Claim:** DBC is a partnership for tax purposes and issues a Schedule K-1 (tax-package portal: taxpackagesupport.com/DBC). The fund's Section 1256 futures gains pass through with 60/40 treatment (K-1 Box 11c) and are marked to market. By contrast, PDBC reports on Form 1099 and its payouts are fund distributions. The 60/40 treatment is not confirmed by an Invesco primary document, and neither is how PDBC's distributions are characterized (ordinary dividend or capital gain).
   - Sources: https://247wallst.com/investing/2026/05/26/pdbc-promises-diversified-commodities-without-k-1-tax-forms-but-the-workaround-hides-a-long-term-roll-cost/ (24/7 Wall St, 2026-05-26, secondary); claimyr.com K-1 article (secondary, 2025-04-11); consistent with the §1256 text (claim 12) · confidence MEDIUM · regulatory (needs primary confirmation — see Leads)
10. **Claim (unverified belief, not evidenced this run):** USO is also a K-1 partnership. No USO source was retrieved.

### Q4 — Physically backed metal grantor trusts (GLD, SLV)
11. **Claim:** Net gains from collectibles are taxed at a maximum 28% rate (IRS). Holders of GLD, a grantor trust, are treated as owning a pro-rata share of the trust's gold. Gains on shares held more than one year, and gains from the trust's own sales of gold held more than one year, are "generally … taxed at a maximum U.S. federal income tax rate of 28%" (language from the GLD offering documents, quoted in search results). The trust's sales of gold to cover expenses are also taxable events for holders.
   - Sources: https://www.irs.gov/taxtopics/tc409 (IRS, HIGH, 28% collectibles rate); https://www.sec.gov/Archives/edgar/data/0001222333/000119312514192745/d725076dfwp.htm (SPDR Gold Trust FWP on SEC EDGAR, 2014; retrieved as a search-result excerpt, not fetched in full) · confidence MEDIUM-HIGH · regulatory
   - SLV: not retrieved. Treating SLV the same way is an unverified belief.
   - Short-term gains on GLD are presumably taxed at ordinary rates like any short-term gain (claim 2); no GLD-specific short-term source was retrieved.

### Q5 — §1256 futures
12. **Claim:** Section 1256 contracts held at year end are treated as sold at fair market value on the last business day of the year (mark-to-market). Gains and losses are split 60% long-term and 40% short-term regardless of how long the position was held. Regulated futures contracts qualify. Securities futures contracts, such as single-stock futures, are excluded unless they are dealer contracts.
   - Source: https://www.law.cornell.edu/uscode/text/26/1256 · Cornell LII · HIGH · regulatory
13. **Claim:** A non-corporate taxpayer with a net §1256 contracts loss can elect to carry it back 3 years, but only against net §1256 gains in those years. The carryback cannot create a net operating loss and is re-characterized 60% long-term / 40% short-term. The election is reported on Form 6781 (named in Pub 550).
   - Sources: https://www.law.cornell.edu/uscode/text/26/1212 (§1212(c)) · HIGH; Pub 550 (2025) for Form 6781 · regulatory
14. **Inference:** Mark-to-market taxation of futures means the wash-sale timing problem from claim 5 does not arise the same way. No source was retrieved that says so explicitly, so this is marked **cpa-question**.

### Q6 — Prop-firm payouts
15. **Claim:** Topstep's help center says funded traders are independent contractors, not employees. US persons must submit a W-9 with their payout request. Topstep issues a 1099-NEC to anyone paid more than $600 in a year; the 2025 forms were emailed 2026-01-31. Traders report only the payouts actually received, as regular income. Example: a funded account earns $5,000 but the trader requests only $1,000, so $1,000 is reported.
   - Source: https://help.topstep.com/en/articles/8284238-funded-trader-tax-questions · Topstep Help Center · undated (mentions 2026-01-31 mailing) · content seen as search-result excerpt, not fetched in full · MEDIUM · regulatory (firm doc)
16. **Contrast:** IRS Topic 429 says gains from securities trading, even by a qualifying trader, are "not subject to self-employment tax". A prop payout reported on a 1099-NEC is contractor income, not trading gain. No IRS primary source on self-employment tax for 1099-NEC income was retrieved.
   - **cpa-question:** Is a prop-firm payout subject to self-employment tax (Schedule SE), and can evaluation fees be deducted against it? Strong prior (1099-NEC nonemployee compensation → Schedule C/SE), but not evidenced this run.

### Q7 — IRA
17. **Claim:** An IRA is exempt from tax (§408(e)(1)), so trades inside it create no current tax; taxes apply only on distribution (traditional) or never, if a Roth meets its conditions (Roth rules not retrieved). If the owner engages in a §4975 prohibited transaction, the account stops being an IRA as of the first day of that tax year. All its assets are treated as distributed at their fair market value on that day (§408(e)(2)).
   - Source: https://www.law.cornell.edu/uscode/text/26/408 · Cornell LII · HIGH · regulatory
18. **Claim:** If the owner uses any part of the IRA as security for a loan, that part is treated as distributed (§408(e)(4)). This is the statutory reason IRAs cannot hold true margin loans. Brokers' "limited margin" IRAs (settlement-only, no borrowing) and short-sale restrictions were **not** evidenced this run.
   - Source: 26 USC 408(e)(4) · HIGH (statute) · the inference about margin and short selling is a **cpa-question** or needs a broker source
19. **Claim:** An indirect rollover must be redeposited within 60 days. Only one such rollover is allowed in any 1-year period (§408(d)(3)). Trustee-to-trustee transfers were not retrieved.
   - Source: 26 USC 408(d)(3) · HIGH · regulatory
20. **Implication:** Futures inside an IRA lose the benefit of 60/40 treatment, because the IRA's income is untaxed until distribution. A monthly ETF trend strategy's main tax cost in a taxable account is short-term gains at ordinary rates plus wash-sale disallowances; inside an IRA that cost is zero. Annual IRA contribution limits were not retrieved, so the practical question of funding $25K–$50K through contributions versus a rollover is left open.
   - **cpa-question:** Does a wash sale triggered by buying the replacement in one's own IRA permanently disallow the loss? Rev. Rul. 2008-5 is believed to say yes, but it was not retrieved this run.

### Q8 — Trader tax status / §475(f)
21. **Claim:** Trader status requires all three of the following. First, seeking profit "from daily market movements … and not from dividends, interest, or capital appreciation". Second, activity that is "substantial". Third, activity carried on "with continuity and regularity". The factors weighed are typical holding periods, the frequency and dollar amount of trades, whether the activity produces income for a livelihood, and time devoted. The IRS publishes **no numeric thresholds**.
   - Source: https://www.irs.gov/taxtopics/tc429 · IRS · last reviewed 2026-09-24 · HIGH · regulatory
22. **Claim:** The §475(f) election must be made by the original due date (not counting extensions) of the prior year's return. Under it, gains and losses are ordinary, wash-sale rules do not apply, and the $3,000 loss limit does not apply. Trader gains are not subject to self-employment tax.
   - Source: same · HIGH · regulatory
23. **Assessment (inference from claim 21):** A monthly-rebalanced strategy that captures multi-week trends seeks profit from trend persistence and capital appreciation, not "daily market movements". It trades roughly 12 times a year per position. On the IRS factors as written, it very likely does **not** qualify for trader tax status, so §475(f) would not be available. Automation and time devoted also weigh against qualifying.
   - class **cpa-question** (the facts-and-circumstances test is decided by a CPA or the courts, not here) · confidence in "unlikely to qualify" MEDIUM-HIGH

## Leads
- Invesco DBC prospectus or the K-1 tax package at taxpackagesupport.com/DBC. Check whether the 60/40 §1256 pass-through and the mark-to-market treatment are primary-confirmed.
- PDBC Form 1099-DIV history. Is the December distribution an ordinary dividend, and are sales of PDBC shares ordinary capital gains?
- The iShares SLV prospectus tax section (for the collectibles rate). USO's K-1 tax package.
- Rev. Rul. 2008-5 on irs.gov (IRA wash sale). The full Pub 550 "Wash Sales" section (Pub 550 was too large to fetch cleanly).
- IRS Topic 554 / Schedule SE instructions for nonemployee compensation, to confirm self-employment tax on prop payouts. The full Topstep help page. Other firms' pages (Apex, TPT) on their 1099 forms.
- 2026 long-term capital gains brackets (Rev. Proc. 2025-32 or similar). NIIT (Topic 559): 3.8% above the MAGI thresholds.
- IRA annual contribution limit for 2026, and broker "limited margin IRA" documentation.

## Looked for, could not find (this run)
- Any IRS primary guidance defining "substantially identical" for ETFs tracking the same index. Not found; the statute is silent.
- The verbatim wash-sale, IRA-purchase, and Form 6781 text in Pub 550 (the fetch returned it only partially).
- An Invesco primary document on DBC's 60/40 / K-1 treatment (the sponsor PDF found covers only PDBC).
- The spdrgoldshares.com FAQ (returned 404). GLD evidence comes from a search excerpt of an SEC EDGAR FWP, and the sponsor's 2025 grantor-trust tax statement was listed in results but not fetched.
- The IRS IRA prohibited-transactions FAQ page (returned 404). Replaced by the §408 statute.
- An IRS page on self-employment tax for 1099-NEC prop payouts. Not retrieved within budget.
