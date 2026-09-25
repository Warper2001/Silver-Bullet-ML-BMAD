# Digest r4-retail-r1-1 — Retail / small-trader edge: loss rates, persistent-skill subgroups, prop-firm economics, who takes the other side

Accessed: 2026-09-25 · Tool calls: 18 (budget 18) · Distinct publishers/sources used: 9

**Definitions used in the claim fields**
- **source-type**: peer-reviewed / working paper / regulatory / industry.
- **class**:
  - **primary-read**: I read the claim in the source's own text, retrieved this run.
  - **search-summary**: the claim came from a search engine's summary of the named primary page. I did not open the page itself, so the numbers should be checked before anyone relies on them.
  - **derived**: my own arithmetic on retrieved figures.
  - **unverified**: the figure surfaced this run, but I could not trace it to a primary text.
- **confidence**: high / medium / low that the claim faithfully represents the source.

**Retrieval note:** WebFetch returned the two Barber et al. PDFs (Odean faculty pages) as binary files. I converted those downloaded copies to text with pdftotext and read passages directly. No project files were read.

---

## Bottom line (evidence retrieved this run only)

1. **Complete-account datasets show that almost everyone loses and a very small tail persists.**
   - In Taiwan (all day traders, 1992–2006), the day-trader population lost money in aggregate, net of fees, in every one of the 15 years.
   - About 20% are profitable net of fees in a typical year, but fewer than 1% (about 1,000–4,000 of about 450,000) are *predictably* profitable year over year.
   - In Brazil's mini-index futures (2013–2015 starters who persisted at least 300 days), 97% lost money and 0.4% earned more than a bank teller's wage (US$54 per day).
2. **Persistent skill exists but is rare, and in the best-documented case it is not liquidity provision.**
   - Taiwan's top 500 prior-year day traders earn 37.9 bps per day net of fees in the following year.
   - The best predictor of that skill is past performance.
   - Their profits concentrate in hard-to-value stocks and around earnings announcements.
   - Even the most profitable group sends about two-thirds of its trades as aggressive orders that demand immediacy.
   - An older US study cited inside the same paper found that 15 proprietary day traders did profit mainly from passive limit orders (2000, before decimalization; I did not read that study).
3. **Retail derivatives lose on average, largely through costs.**
   - EU regulators found that 74–89% of retail CFD accounts lose money.
   - US retail options traders lost $2.1bn in aggregate from Nov 2019 to Jun 2021, mostly through indirect costs (spreads).
   - Retail options buyers lose 5–9% around earnings announcements, and 10–14% when expected volatility is high.
4. **Prop-firm economics (one firm's own disclosure, not audited).** Topstep reports these 2025 figures:
   - 16.8% of Trading Combines were passed.
   - 51.8% of individual participants reached the Funded Level at least once.
   - 33.3% of Funded-Level participants received a payout.
   - 0.71% of Express Funded participants were called up to a Live Funded Account.
   - Industry blogs claim that about 7% of challenge buyers ever receive a payout. I could not trace that figure to a primary source.
5. **The "retail flow is dumb money" story needs a caveat.** In US equities, the direction of marketable retail order flow (the net buy/sell imbalance) *predicts* weekly returns cross-sectionally (Boehmer, Jones, Zhang & Zhang, JF 2021). Retail flow is therefore not uniformly uninformed at multi-day horizons. That is a different question from whether individual retail traders come out ahead after costs.

---

## Claims

### Q1 — Day-trader studies with complete account data

**C1. Aggregate day-trader performance in Taiwan was negative net of fees in every year.**
- Claim: "using complete data for the Taiwan market, the aggregate performance of day traders net of fees is negative in each of the 15 years that we study" (Taiwan, 1992–2006).
- Source: https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trading%20and%20Learning%20110217.pdf
- Authors: Barber, Lee, Liu, Odean & Zhang, "Do Day Traders Rationally Learn About Their Ability?"
- Date: October 2017 draft
- Source type: working paper (an earlier version was presented at the AEA 2019 meeting as "Learning, Fast or Slow")
- Accessed 2026-09-25 · confidence **high** · class **primary-read**

**C2. Losing day traders keep trading at almost the same rate as profitable ones.**
- Claim: traders with at least 50 days of experience who had been unprofitable have a 95.3% probability of day trading again in the next 12 months. The figure for previously profitable traders is 96.4%. The abstract says "the vast majority of day traders are unprofitable, and many persist despite an extensive experience of losses."
- Source: same as C1
- Accessed 2026-09-25 · confidence **high** · class **primary-read**

**C3. Most day traders quit quickly, and poor performers quit sooner.**
- Claim: "more than 75% of all day traders quit within two years, and poor performers are more likely to quit."
- Source: same as C1
- Accessed 2026-09-25 · confidence **high** · class **primary-read**

**C4. About 20% profit in a given year, but fewer than 1% do so predictably.**
- Claim: in the average year about 450,000 individuals day trade in Taiwan.
  - "Approximately 20% earn profits net of fees in the typical year."
  - "Less than 1% of day traders (4,000 out of 450,000) are able to outperform consistently."
  - Traders are ranked on year-*y* returns and tracked in year *y*+1. On that basis, the top 500 earn **61.3 bps per day gross and 37.9 bps per day net**.
  - The bottom-ranked earn **−11.5 bps per day gross and −28.9 bps per day net**.
- Source: https://faculty.haas.berkeley.edu/odean/papers/day%20traders/The%20Cross-Section%20of%20Speculator%20Skill.pdf (also https://www.sciencedirect.com/science/article/abs/pii/S1386418113000190)
- Authors: Barber, Lee, Liu & Odean
- Date: *Journal of Financial Markets* 18 (2014) 1–24
- Source type: **peer-reviewed**
- Accessed 2026-09-25 · confidence **high** · class **primary-read**
- The minus signs were lost in PDF extraction. The search summary of the same abstract confirms that both bottom-ranked figures are negative.

**C5. The count of predictably profitable traders depends on the commission assumed.**
- Claim: at the 14.25 bps statutory maximum commission, the top three groups earn net alphas of 24.1 bps (t = 39.0), 7.6 bps (t = 10.9) and 1.1 bps (t = 1.6). "The number of predictably profitable day traders ranges from a low of 1,000 to a high of 4,000."
- Source: same as C4
- Accessed 2026-09-25 · confidence **high** · class **primary-read**

**C6. In Brazil, 97% of persistent futures day traders lost money.**
- Sample: every individual who began day trading Brazilian equity futures (the "third in terms of volume in the world") in 2013–2015 and persisted for at least 300 days.
- Results: "97% of them lost money, only 0.4% earned more than a bank teller (US$54 per day), and the top individual earned only US$310 per day with great risk (a standard deviation of US$2,560). We find no evidence of learning by day trading."
- Source: https://ideas.repec.org/p/fgv/eesptd/525.html (SSRN 3423101 returned HTTP 403)
- Authors: Chague, De-Losso & Giovannetti, "Day trading for a living?"
- Date: 2020, FGV EESP Textos para discussão 525
- Source type: **working paper**
- Accessed 2026-09-25 · confidence **high** · class **primary-read** (abstract only)

**C7. An earlier version of the Brazil paper reports a minimum-wage figure.**
- Claim: "only 1.1% earned more than the Brazilian minimum wage and only 0.5% earned more than the initial salary of a bank teller."
- Source: search summary pointing to https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3423101 and https://sites.google.com/site/bcaragiovannetti/home
- Authors and date: Chague, De-Losso & Giovannetti, 2019/2020 versions
- Source type: working paper
- Accessed 2026-09-25 · confidence **medium** · class **search-summary**
- This is a version discrepancy: 0.5% here versus 0.4% in the 2020 abstract (C6). The minimum-wage figure does not appear in the 2020 abstract.

### Q2 — Retail futures, FX and CFDs

**C8. EU regulators found that 74–89% of retail CFD accounts lose money.**
- Claim: national regulators' analyses of CFD trading in several EU jurisdictions show "74-89% of retail accounts typically lose money," with average losses per client of €1,600–€29,000.
- ESMA's response:
  - leverage caps from 30:1 to 2:1, depending on the underlying's volatility
  - a per-account margin close-out rule
  - negative-balance protection
  - restrictions on incentives
  - a mandated risk warning
- ESMA's Board of Supervisors agreed these measures on 23 March 2018.
- Source: https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-and-restrict-cfds-protect-retail-investors (press release PDF: https://www.esma.europa.eu/sites/default/files/library/esma71-98-128_press_release_product_intervention.pdf)
- Publisher: ESMA
- Date: March 2018
- Source type: **regulatory**
- Accessed 2026-09-25 · confidence **medium-high** · class **search-summary**
- Several national-regulator reprints (AFM, CNMV, CMS) surfaced in the same search. I did not open them.
- These are pre-intervention figures. I did not retrieve any broker's "X% of retail accounts lose money" figures or post-2018 loss rates.

**C9. US retail futures profitability data.** Nothing was retrieved. See "Looked for, could not find."

### Q3 — Retail options

**C10. US retail options traders lost $2.1bn in aggregate, mostly through spreads.**
- Claim: the aggregate retail options portfolio lost **$2.1 billion from Nov 2019 to Jun 2021**, assuming a 10-day holding horizon. "The bulk of the losses comes from the indirect costs of trading." Retail investors prefer cheap weekly options, whose average bid-ask spread is 12.6%.
- Sources:
  - https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13285
  - https://ideas.repec.org/a/bla/jfinan/v78y2023i6p3465-3514.html
  - PDF: https://lbsresearch.london.edu/id/eprint/2827/
- Authors: Bryzgalova, Pavlova & Sikorskaya, "Retail Trading in Options and the Rise of the Big Three Wholesalers"
- Date: *Journal of Finance* 78(6):3465–3514, 2023
- Source type: **peer-reviewed**
- Accessed 2026-09-25 · confidence **medium** · class **search-summary** (the journal citation is confirmed by the RePEc URL)
- The correct title ends in "Wholesalers", not "Wireless Carriers" as the brief had it.

**C11. Retail options buyers lose money around earnings announcements.**
- Claim: retail investors buy options in a concentrated way before earnings announcements, especially those with high expected abnormal volatility. Three behaviours cost them money:
  - overpaying relative to realized volatility
  - paying very wide bid-ask spreads
  - reacting slowly after the announcement
- Losses average **5–9%**, and **10–14%** for announcements with high expected volatility.
- Data: Nasdaq equity-option trades covering 32,791 announcements, Jan 2010–Feb 2021.
- Sources:
  - https://academic.oup.com/rof/article-abstract/30/2/489/8301159
  - SSRN 4050165: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4050165
- Authors: de Silva, Smith & So, "Losing is Optional: Retail Option Trading and Expected Announcement Volatility"
- Date: *Review of Finance* 30(2), March 2026
- Source type: **peer-reviewed**
- Accessed 2026-09-25 · confidence **medium** · class **search-summary**
- The loss unit (percent of premium or percent of position) was not confirmed.

### Q4 — Prop-firm / funded-trader economics

**C12. Topstep's own 2025 figures.**
- Claim: for January–December 2025, Topstep discloses:
  - **16.8%** of all Trading Combines started were passed and could advance to the Funded Level.
  - **51.8%** of individual participants who entered at least one Combine reached the Funded Level in at least one of them.
  - **33.3%** of individual participants at the Funded Level received a payout.
  - **0.71%** of individual participants trading an Express Funded Account were called up to a Live Funded Account.
- Source: topstep.com performance disclosure, surfaced by a search restricted to topstep.com (candidate pages: https://www.topstep.com/ and https://www.topstep.com/topstep-prop). I did not identify the exact page.
- Publisher: Topstep
- Date: 2025 figures, published 2026
- Source type: **industry** (the firm's self-disclosure; not audited)
- Accessed 2026-09-25 · confidence **medium** · class **search-summary**

**C13. Roughly 17% of Combine entrants may have received a payout (my estimate, not Topstep's).**
- Claim: *if* the denominators align, about **17%** (0.518 × 0.333) of individual Combine entrants received at least one payout in 2025.
- The denominators may not align: people at the Funded Level in 2025 may have qualified in 2024. A payout also does not mean net profit after Combine fees and resets. Topstep does not disclose the net-of-fees share.
- Source: derived from C12
- Accessed 2026-09-25 · confidence **low** · class **derived**

**C14. Industry-wide pass and payout rates are untraceable.**
- Claim: industry posts claim that "about 7% of all challenge buyers ultimately receive a payout." They also cite evaluation pass rates of 5–10%, and "one published analysis of 300,000+ accounts found a 14% pass rate."
- Sources:
  - https://www.quantvps.com/blog/prop-firm-statistics
  - https://www.track360.io/blog/prop-trading-industry-statistics-2026
  - https://apextraderfunding.com/resources/prop-trading/what-percentage-of-traders-get-a-payout-from-prop-firms/
  - https://tradersyard.com/blog-posts/how-many-people-get-payouts-from-prop-firms
- Publishers: marketing, affiliate and prop-firm blogs
- Date: 2026
- Source type: **industry (downgraded)**
- Accessed 2026-09-25 · confidence **low** · class **unverified**
- No upstream dataset was identified, and the posts appear to recycle each other. Treat them as one unverified publisher.

### Q5 — Subgroups with persistent skill

**C15. What Taiwan's persistent winners look like.**
- Claim:
  - "Past performance (either returns or dollar profits) is, by a large margin, the best predictor of future performance."
  - Profits come from "hard-to-value stocks and around earnings announcements … day traders forecast short-term price movements in stocks or periods with high levels of information asymmetry."
  - The *liquidity-provision* explanation is rejected: "the most profitable day traders tend to lean somewhat more on passive orders, but even for this select group … nearly two-thirds of trades emanate from aggressive orders." Order aggressiveness is "an economically weak predictor of future day trader profitability."
- Source: Barber, Lee, Liu & Odean, JFM 2014 (URL in C4)
- Source type: **peer-reviewed**
- Accessed 2026-09-25 · confidence **high** · class **primary-read**

**C16. Some US prop day traders profited mainly as passive liquidity providers.**
- Claim: Garvey & Murphy (2005a) studied 96,000 trades by 15 proprietary day traders over three months in 2000. These traders use firm capital, pay no commissions and share profits with the firm. They made money "primarily by placing limit orders on electronic crossing networks (ECNs) that are inside the current best quotes offered by NASDAQ dealers." A follow-up (2005b) studied 1,386 day traders.
- Source: cited in Barber et al. JFM 2014 (URL in C4). I did not retrieve the original papers.
- Source type: peer-reviewed (secondary citation)
- Accessed 2026-09-25 · confidence **medium** · class **primary-read of the citing paper**
- This is the one retrieved piece of evidence of a persistent retail-adjacent edge from liquidity provision. It predates decimalization (2001) and modern HFT competition.

**C17. No evidence of learning among Brazilian day traders.**
- Claim: "We find no evidence of learning by day trading." The Brazil paper's framing is that individuals cannot "compete with HFTs."
- Source: Chague et al. 2020 (URL in C6)
- Source type: working paper
- Accessed 2026-09-25 · confidence **high** · class **primary-read** (abstract)

**C18. Algorithmic or automated retail traders.** Nothing was retrieved. See "Looked for, could not find."

### Q6 — Retail flow as the other side

**C19. Net retail buying in US equities predicts returns over the following weeks.**
- Method: retail marketable orders are identified by sub-penny price improvement, which appears because wholesalers internalize or buy most US retail order flow.
- Findings:
  - Stocks with net retail buying outperform stocks with net retail selling by **about 10 bps over the following week (about 5% annualized)**.
  - Retail investors appear informed "at horizons up to 12 weeks," more so in smaller, low-priced stocks.
  - They show **no market-timing ability**.
  - Less than half of the predictive power comes from order-flow persistence. Contrarian trading (a proxy for liquidity provision) and public news sentiment explain little of the rest.
- Sources: https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13033 ; SSRN 2822105
- Authors: Boehmer, Jones, Zhang & Zhang, "Tracking Retail Investor Activity"
- Date: *Journal of Finance*, 2021
- Source type: **peer-reviewed**
- Accessed 2026-09-25 · confidence **medium-high** · class **search-summary**
- This **cuts against** a blanket "retail flow is uninformed" premise. It measures aggregate imbalances across stocks, not individual accounts' returns net of costs, so it is compatible with C1–C7.

**C20. In options, the costs retail pays go to liquidity providers (my inference).**
- Claim: C10's finding that "the bulk of losses comes from indirect costs" implies that spread-capturing liquidity providers are the main counterparties profiting from retail options flow. The paper's title points to three wholesalers dominating that flow.
- Source: derived from C10
- Accessed 2026-09-25 · confidence **low-medium** · class **derived**
- I did not retrieve how the paper quantifies wholesaler profit.

---

## Leads (not verified this run)

- **Kuo et al., "The Profitability of Day Trading and the Characteristics of Traders: Evidence from the Taiwan Futures Market."** https://www.irabf.org/upload/journal/prog/2.%20Final%20-%20The%20Profitability%20of%20Day%20Trading%20and%20the%20Characteristics%20of%20Traders%20%20Evidence%20from%20the%20Taiwan%20Futures%20Market.pdf. This is directly relevant to *futures* day traders with complete account data. It surfaced in search but I did not open it.
- **Barber, Lee, Liu, Odean & Zhang, "Learning, Fast or Slow"** (AEA 2019 preliminary): https://www.aeaweb.org/conference/2019/preliminary/paper/ZKnGb4Zh. This may be the published successor to C1–C3.
- **Unverified figures from a search summary.** Nearly 40% of traders day trade for only one month, 13% continue after three years, and 7% after five years. Separately, 1.6% of day traders are profitable in the average year and account for 12% of day-trading activity. The summary attributed these to the Barber et al. learning paper, but a grep of the October 2017 draft found neither the 1.6% figure nor the attrition percentages. They may come from another version, or from the truewealth.ch blog in the results.
- **"Retail Order Flow Imbalances: Informed Trading or Liquidity Provision?"** https://microstructure.exchange/papers/Dog_s_Tail_02212022.pdf. This appears to challenge C19's interpretation.
- **"Retail Trader's Ruin: An Anatomy of Popular Signal Failure,"** arXiv 2607.20093 (July 2026): https://arxiv.org/pdf/2607.20093. Unread.
- **Prior-knowledge hypotheses to query next** (training data only, *not evidence*):
  - The CFTC action against My Forex Funds (Traders Global Group, 2023) may give regulator-sourced prop-firm economics.
  - Heimer & Simsek's JFE paper on US retail FX leverage limits may hold US retail FX loss data.
  - Barber, Huang, Odean & Schwarz on Robinhood attention-induced trading (JF 2022).
  - The FCA's 2016 CFD consultation reported a single loss rate.
  - Kelley & Tetlock (2013) on retail order-flow informativeness.
- **Garvey & Murphy (2005a/b):** the original sources behind C16.

## Looked for, could not find

- **US retail futures profitability data (CFTC or academic).** No such dataset appeared in the searches that ran. I did not run a dedicated CFTC or NFA query because of the budget, so this is *not searched*, not *confirmed absent*.
- **Performance evidence for algorithmic or automated retail traders.** Nothing surfaced incidentally. I ran no dedicated query (budget). Treat as not searched.
- **Independent, audited prop-firm payout statistics.** Only Topstep's self-disclosure (C12) and affiliate blogs with untraceable figures (C14) surfaced. No FTC or CFTC data were retrieved.
- **Chague et al.'s minimum-wage figure in primary text.** It appears only in a search summary (1.1%, C7). The 2020 abstract I read reports only the bank-teller share (0.4%). SSRN returned HTTP 403.
- **Quantified wholesaler or PFOF profits from retail equity or options flow.** Not retrieved.
- **Post-2018 ESMA-era CFD loss rates** (the percentages brokers must disclose). Not retrieved; only the pre-intervention 74–89% range.
