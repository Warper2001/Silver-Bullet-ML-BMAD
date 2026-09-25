# Red-team digest R1-1: the "edges exist only for structurally advantaged firms" conclusion

**Role:** adversarial skeptic. **Date:** 2026-09-25. **Budget used:** 18 tool calls, 8 sources.
**Conclusion under attack:** "Durable, verifiable trading edges (risk-adjusted, net of all costs) exist only for a small number of firms with structural advantages (speed, scale, order flow, risk-bearing capacity). For small independent, retail or automated traders with modest capital (~$25K–$150K), net-of-cost edges that can be verified are essentially unobtainable. Published anomalies mostly decay after publication or are eaten by costs, and the rare profitable individuals are statistically indistinguishable from luck."

Everything below comes from sources retrieved in this run. For the two central papers (Barber et al. 2014; Coval et al. 2021), I read the full text and quote it. For the others, I relied on abstracts or search-result summaries, and the confidence ratings say so.

---

## (A) Strongest counter-evidence

### A1. A small top tail of individual day traders profits persistently, net of fees, and the luck null is rejected
- **Claim:** The sample covers Taiwan day traders from 1992 to 2006, about 450,000 per year. Traders were ranked on year-y returns and then observed in year y+1. The top 500 went on to earn **61.3 bps/day gross and 37.9 bps/day net of fees** on their day-trading portfolio. About 1,000–4,000 traders predictably profit net of costs. The range depends on the commission assumption, from 5 bps up to the 14.25 bps statutory maximum. At the maximum, the top group's net alpha is still 24.1 bps/day (t = 39.0). The paper also tests whether being a winner or loser in one year is independent of the next. That luck null is "comfortably reject[ed] (p<0.01)." The authors write: "luck is not the whole story." The same abstract says **"Less than 1% of the day trader population is able to predictably and reliably earn positive abnormal returns net of fees"**, and that about 20% are profitable in any single year. So the population is mostly luck, while the top tail is skill.
- **What distinguishes the top tail:** The paper reports that returns are higher in hard-to-value stocks (small or volatile) and around certain periods. It tested liquidity provision as an explanation and found it is *not* the main driver.
- **Source:** https://faculty.haas.berkeley.edu/odean/papers/day%20traders/The%20Cross-Section%20of%20Speculator%20Skill.pdf (also https://www.sciencedirect.com/science/article/abs/pii/S1386418113000190)
- **Authors:** Brad M. Barber, Yi-Tsung Lee, Yu-Jane Liu, Terrance Odean
- **Pub date:** Journal of Financial Markets 18 (2014) 1–24; online 2 July 2013
- **Source type:** peer-reviewed; administrative exchange data covering the whole population, so there is no survivorship selection
- **Confidence:** high (full text read)
- **Dent:** **material** to the clause "statistically indistinguishable from luck". It directly falsifies that clause for a subset. It *supports* "essentially unobtainable" as a base rate: below 1% of participants. The limits are that the data are Taiwanese cash equities from 1992–2006, not index futures, and that account capital was not reported. The inclusion threshold was about US$20K of annual day-trade volume, not account size.

### A2. The top decile of US discount-broker households shows persistent skill, and a mimicking strategy appears to clear costs
- **Claim:** The data cover 63,652 households and 115,856 accounts at a large US discount broker from January 1991 to November 1996. Households in the top performance decile in the first half of the sample "subsequently earn risk-adjusted returns of about 6% per year." The abstract says this is "not confined to stocks in which the investors are likely to have inside information, nor ... driven by illiquid stocks". It also says the returns are above the size, value, momentum and earnings-announcement strategies. A strategy that mimics past winners and shorts past losers earns more than 7%/yr abnormal, at roughly 128%/yr turnover. The authors compare this with about 2.44%/yr cost for the highest-turnover mutual-fund quintile and call it "perhaps our strongest evidence that individual investors can beat the market."
- **What distinguishes them:** Successful households buy stocks with *higher* bid-ask spreads but pay *less* on the day they buy. The paper says: "our households sometimes provide liquidity and avoid paying the spread."
- **Source:** https://www.tylergshumway.org/Individual_Trading_Coval_Hirshleifer_Shumway_raps3.pdf ; journal page https://academic.oup.com/raps/article-abstract/11/3/552/6311677
- **Authors:** Joshua D. Coval, David Hirshleifer, Tyler Shumway
- **Pub date:** Review of Asset Pricing Studies 11(3), 2021 (working paper from about 2002; the version read is dated June 21, 2021)
- **Source type:** peer-reviewed
- **Confidence:** medium-high on persistence. **Medium-low on "net of costs":** the net-of-cost argument is a back-of-envelope comparison, not a computed net return for each individual. The holding-period strategy excluded the smallest tercile of stocks. The data are from the 1990s. The author-reported correlation of an individual's performance across periods is only about 10%, a weak signal.
- **Dent:** **material** to the "luck" clause; **minor** to "unobtainable", because the data are old, costs are not netted for individual investors, and the effect size is modest.

### A3. Low-turnover anomalies still earn significant spreads net of trading costs
- **Claim:** "Most anomalies with one-sided monthly turnover lower than 50% continue to generate statistically significant net spreads, at least when designed to mitigate transaction costs." The most effective simple fix is a buy/hold spread, where stocks the investor would not actively trade into are simply kept. Few higher-turnover strategies survive. Mid-turnover costs run 20–57 bps/month, "often exceeding half the strategies' gross spreads."
- **Source:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2535173 ; https://academic.oup.com/rfs/article-abstract/29/1/104/1844518 ; NBER w20721
- **Authors:** Robert Novy-Marx, Mihail Velikov
- **Pub date:** Review of Financial Studies 29(1), 2016 (NBER WP 2014)
- **Source type:** peer-reviewed
- **Confidence:** medium (abstract and search summary only; the full text was not read in this run)
- **Dent:** **material** to "eaten by costs" as a blanket statement. The survivors are slow, low-turnover strategies with no speed or order-flow requirement. The authors also say capacity *falls* as turnover rises, so the survivors are exactly what a small trader can run. Caveats: these are long-short equity spreads, which need shorting (costly or hard for retail). They are largely pre-publication and in-sample relative to discovery. The authors themselves flag that costs raise data-snooping concerns.

### A4. Most factors replicate and work out of sample across 93 countries
- **Claim:** "The majority of asset pricing factors can be replicated, can be clustered into 13 themes (the majority of which are significant parts of the tangency portfolio), work out-of-sample in a new large data set covering 93 countries," and the evidence is "strengthened (not weakened) by the large number of observed factors." The code and data are public.
- **Source:** https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249 ; https://www.nber.org/papers/w28432 ; code https://github.com/bkelly-lab/ReplicationCrisis
- **Authors:** Theis Ingerslev Jensen, Bryan T. Kelly, Lasse Heje Pedersen
- **Pub date:** Journal of Finance 78(5), 2023 (NBER WP 2021)
- **Source type:** peer-reviewed
- **Confidence:** medium (abstract and summary)
- **Dent:** **minor-to-material** against "published anomalies mostly decay". It shows the premia are real out of sample, not data-mining artifacts. It does *not* show net-of-cost returns for a small trader. These are gross factor returns, and it says nothing about the publication decay in A5.

### A5. Post-publication decay is large but partial: about 42% of anomaly returns remain
- **Claim:** The study covers 97 predictors. Returns are 26% lower out of sample (the upper bound on data mining) and **58% lower post-publication**. That implies about 32 percentage points of decay from publication-informed trading, with roughly 42% of the in-sample return remaining on average. Remaining returns are higher in stocks with high idiosyncratic risk and low liquidity, where arbitrage is limited.
- **Source:** https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623
- **Authors:** R. David McLean, Jeffrey Pontiff
- **Pub date:** Journal of Finance, 2016
- **Source type:** peer-reviewed
- **Confidence:** high on the headline numbers, which are widely reproduced; the summary was retrieved but the full text was not read
- **Dent:** **minor**. It confirms "mostly decay" (58%) but refutes "decay to nothing". The residual sits where arbitrage is limited, which is plausibly where a small trader's lack of price impact helps. This source partly *supports* the conclusion.

### A6. Crypto has large, recurring cross-exchange arbitrage that persists for days or weeks
- **Claim:** "Cryptocurrency markets exhibit periods of large, recurrent arbitrage opportunities across exchanges." The spreads "persist over several days and weeks" and are much larger across countries than within them. The authors attribute this to capital controls and a lack of regulatory oversight. The study uses tick data from 34 exchanges in 19 locations (BTC, ETH, XRP).
- **Source:** https://ideas.repec.org/a/eee/jfinec/v135y2020i2p293-319.html ; https://mitsloan.mit.edu/cfi/trading-and-arbitrage-cryptocurrency-markets
- **Authors:** Igor Makarov, Antoinette Schoar
- **Pub date:** Journal of Financial Economics 135(2), February 2020 (data centred on 2017–2018)
- **Source type:** peer-reviewed
- **Confidence:** medium (summary only)
- **Dent:** **minor**. It shows a large mispricing that is not about speed, but the barrier is *capital mobility across borders* (for example, Korea). That is not something a small US trader can exploit. No post-publication persistence evidence was retrieved.

### A7. Top hedge-fund performance is not explained by luck (bootstrap), and it persists
- **Claim:** "Using a robust bootstrap procedure ... top hedge fund performance cannot be explained by luck, and hedge fund performance persists at annual horizons." Results "are neither confined to small funds, nor driven by incubation bias, backfill bias, or serial correlation." Alpha persistence was found in all six strategy categories. Sorting on Bayesian alphas raises the top-minus-bottom decile spread by 5.5%/yr.
- **Source:** https://www.sciencedirect.com/science/article/abs/pii/S0304405X06002017 ; https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=1007&context=bnp_research
- **Authors:** Robert Kosowski, Narayan Y. Naik, Melvyn Teo
- **Pub date:** Journal of Financial Economics 84(1), 2007
- **Source type:** peer-reviewed
- **Confidence:** medium (summary). Pre-2007 data; database returns are self-reported.
- **Dent:** **minor** for the small-trader clause, since these are institutions. **Minor-to-material** against "only firms with speed, scale, order flow or risk-bearing capacity". Hedge-fund skill across six strategy styles is not obviously any of those four.

### A8. Small, young ("emerging") hedge funds outperform, consistent with capacity limits
- **Claim:** After adjusting for database biases, the study finds "strong evidence of outperformance during the first two to three years of existence": about 2.3%/yr in the first two years *relative to later years*. Each additional year of age costs about 42 bps. Early outperformance persists for up to five years. The authors suggest emerging funds "because of their size, may be more nimble."
- **Source:** https://www.sciencedirect.com/science/article/abs/pii/S0304405X0900258X ; https://econpapers.repec.org/RePEc:eee:jfinec:v:96:y:2010:i:2:p:238-256
- **Authors:** Rajesh K. Aggarwal, Philippe Jorion
- **Pub date:** Journal of Financial Economics 96(2), 2010
- **Source type:** peer-reviewed
- **Confidence:** medium (summary). The 2.3% is *relative* to the same funds' later years, not an absolute net alpha.
- **Dent:** **minor**. This is the best retrieved support for the idea that being small is itself an edge, but the funds are still far larger than $25K–$150K.

---

## (B) Overall assessment: the conclusion survives only with qualification. As worded, one clause fails and two are too strong.

1. **"The rare profitable individuals are statistically indistinguishable from luck": fails as written.** Two large-sample, peer-reviewed studies, one of them covering an entire market's population, reject the luck null for a small top tail of individuals (A1, A2). In A1 the result holds net of fees, including at the statutory maximum commission. Correct restatement: *most* profitable individuals in any one year are lucky (about 20% profitable, under 1% persistently so), but a verifiable skilled tail exists.
2. **"Only a small number of firms with structural advantages": too strong.** Low-turnover anomaly strategies survive realistic costs without speed or order-flow access (A3). Factor premia replicate internationally (A4). Hedge-fund alpha survives a luck bootstrap (A7). Small, young managers outperform older and larger ones, consistent with capacity limits (A8).
3. **"Published anomalies mostly decay or are eaten by costs": broadly supported, but not to zero.** The average decay is 58%, leaving about 42% (A5). High-turnover strategies mostly die after costs, while low-turnover ones mostly survive (A3). This clause holds for *high-turnover* strategies, which describes most intraday retail systematic trading.
4. **"Essentially unobtainable" for small traders: survives as a base-rate statement.** Every piece of counter-evidence either concerns a population tail below about 1% (A1), rests on old data (A1: 1992–2006; A2: 1991–1996), concerns slow, low-turnover equity long-short books, not intraday futures (A3–A5), or concerns institutions (A7, A8). None of the retrieved evidence shows a small, intraday, automated *futures* trader achieving verifiable, persistent, net-of-cost skill.

**Strongest defensible rewrite:** "Verifiable net-of-cost edges for small independents are rare. In the best population dataset, fewer than 1% of active day traders profit persistently. Those that exist cluster in low-turnover, capacity-limited or hard-to-value niches, and most evidence predates 2010. They are not impossible, and the top tail *is* statistically distinguishable from luck. For high-turnover intraday strategies, the conclusion is well supported."

---

## (C) Searched for but not found, or not reached within budget (absence is a finding)

- **Audited or verified-account track records of small independent traders that survive a luck adjustment** (competitions with verified accounts, regulatory filings, CTA/NFA records): not searched within budget, and none turned up incidentally. This remains an open gap.
- **Post-2010 individual-level persistence studies** (retail skill in the decimalized, HFT-era market): none retrieved. The strongest individual-skill evidence (A1, A2) is from 1991–2006.
- **Retail skill in index futures specifically:** not retrieved. A Taiwan futures day-trading paper (Kuo et al., IRABF) appeared in search results but was not read.
- **Post-publication or out-of-sample evidence for crypto arbitrage (A6), or for crypto inefficiencies accessible to small traders:** not retrieved.
- **Structural small-trader edges** (tax treatment, long horizon or patience, liquidity provision as a business, sports-betting-style closing-line-value analogues): no peer-reviewed evidence retrieved. The only partial hint is A2's liquidity-provision observation. A1 explicitly found that liquidity provision is *not* the main driver.
- **Evidence cutting the other way** (appeared in results but not read, so recorded here only as unverified pointers): an arXiv paper, "Retail Trader's Ruin: An Anatomy of Popular Signal Failure" (arXiv:2607.20093), and "Publication Bias in Asset Pricing Research" (arXiv:2209.13623). **Unverified belief, not retrieved:** Chen & Velikov, "Zeroing in on the expected returns of anomalies" (JFQA), which I recall reports near-zero net post-publication anomaly returns. If it holds, it would *strengthen* the conclusion against A3 and A5.
