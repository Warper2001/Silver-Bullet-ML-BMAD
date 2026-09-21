# Cluster B digest r1-1: scheduled/recurring event and session-clock effects, non-equity-index futures

Accessed 2026-09-21. Web research only. Budget was ~15 calls; ~19 were used, and several key PDFs would not parse (NY Fed sr1188, INSEAD Krohn PDF, CORE 403). Where I only have a search-engine summary of a paper, I say so. **Bottom line: I found NO candidate in this cluster that has (a) a usable per-trade net effect size, (b) independent replication, and (c) post-publication evidence. Every candidate is a pointer, not a strategy.**

## Candidates

### C1. Crude oil: EIA Wednesday-report intraday momentum (third half-hour to last half-hour)
- Rule (per abstract, Wen, Indriawan, Lien, Xu, Energy Journal 44(5), Sep 2023, peer-reviewed): EIA report released Wed 10:30 ET. On EIA days the return in the *third half-hour* of the session significantly and positively predicts the *last half-hour* return; on non-EIA days only the first half-hour predicts. Direction is inferred from the sign of the third-half-hour return, not from the surprise. Authors say the effect comes from informed traders and thinner liquidity around the release; "substantial economic gains" claimed. Wen et al. sample period and instrument were not in the text I could read (abstract pages only; CORE returned 403).
- Effect size: NOT RETRIEVED. No mean/trade, SD, win rate, or cost-adjusted figure obtained. Events/yr: ~52 (Wednesdays, minus holidays), which satisfies the cadence requirement.
- OOS/replication: none found by independent authors. Caveat: the closest related paper I actually read (Intraday return predictability, commodity ETFs, PMC7480318, 2020) does an OOS R^2 check but does NOT address EIA days. There, USO first-half-hour predicting last-half-hour has in-sample R^2 0.67% (t=2.86), OOS R^2 0.38%; OVX failed OOS (-0.67%). USO is an ETF, sample Jan 2007-Jul 2019.
- Decay: not assessed anywhere I found. Sample ends before ~2022; nothing on 2023-2026.
- Slippage: the paper itself says liquidity is reduced around the release; a Journal of Futures Markets 2026 paper (Miao, on intraday liquidity in international crude futures, abstract only) says EIA news affects returns and to a lesser extent liquidity. No spread/slippage numbers retrieved. The signal window (11:00-11:30 ET to ~15:30-16:00 ET) is well after 10:30, so announcement-time spread widening is less relevant to fills, but that is my inference.
- Fat-tail dependence: unknown.
- Micro feasibility: MCL exists (belief from training data, tick value and margin unverified this run). Small single-trade risk is plausible with a stop, but untested.
- Verdict: worth a power gate and a replication attempt on the operator's own data, not a trade-ready edge. Direction rule is mechanical (no surprise inference required), which is good.

### C2. Natural gas: Thursday EIA storage report (surprise-driven reaction)
- Rule/evidence: Linn and Zhu (J. Futures Markets 2004, abstract via search snippet only): the weekly storage report drives elevated volatility at release and for ~30 minutes afterwards. The PMC4122141 paper (heating oil and NG, NYMEX, 2003-2006, 988/991 daily observations) finds expected inventory surprises significantly and negatively affect far-month NG returns (coefficients -0.0095 to -0.0141, p 0.044-0.074, i.e., marginal). Authors claim no trading application and no forward validation.
- Direction must be inferred from the surprise (actual minus consensus), which requires a consensus feed.
- Effect size, win rate, decay, slippage: NOT FOUND. Only 2003-2006 data. Recent news snippets (NGI) show first-move reversals ("buying lost momentum as the session progressed"), an anecdote, not evidence.
- Micro feasibility: I know of no micro NG future on CME equal in liquidity to MCL (unverified); full-size NG (10,000 MMBtu) has very large per-tick dollar risk and violent announcement moves, which is incompatible with a $2K trailing limit (belief, not sourced).
- Verdict: weak evidence base, contract-size problem. Deprioritize.

### C3. FX: WM/Reuters 4pm London fix and the "W-shaped" around-the-clock pattern
- Rule (Krohn, Mueller, Whelan, Journal of Finance 79(1), 2024, peer-reviewed; SSRN/BoC working paper 2021, 2020 AEA): USD tends to appreciate into FX benchmark fixes and depreciate afterwards. Traders exploiting these patterns earn Sharpe 0.5-0.7 (this number comes from a search-engine summary of the paper; I could not open the tables to confirm and could not confirm whether it is net of costs, or the sample period). Direction is predetermined (long USD into fix, reverse after), no surprise inference.
- Cadence: daily, so far more than 1-2 trades/week (exceeds requirement, but each trade is tiny in bp).
- Earlier evidence: Evans (arXiv 1501.07778, preprint, 12 pairs, 2008/2010-Apr 2014, minute data): extreme returns around 4pm of ~8.3-16 bp in the minute before the fix, versus ~5 bp typical daily average, extreme-move probability 15.0% vs 9.5% at other times. This is a volatility/extreme-return finding, not a directional edge. Authors themselves warn that using the fix "could re-introduce execution risk".
- Market-practice source (Pragma, via Euromoney/LeapRate trade-press, not peer-reviewed): conditioned on first-minute direction, returns continue about 6 bp (month-end) to 10 bp (quarter-end) then revert 4-6 bp. Month/quarter-end concentrated, i.e., few events per year, and largely a dealer-client flow story.
- Structural break: On 2015-02-15 WM/Reuters widened the fixing window from 1 min to 5 min and added data sources (from arXiv/FSB-related results), intended to curb manipulation. Krohn et al. section headings mention post-2015 analysis; I could not read the result. Whether the effect decayed after the change is UNVERIFIED.
- Slippage: 6-10 bp move at quarter-end is comparable to spread plus slippage on FX futures around 4pm London; no cost analysis retrieved. Krohn's PDF has a section "Intraday Profitability and Transaction Costs" that I could not read.
- Micro feasibility: M6E (micro EUR/USD) exists (belief); trades run in London afternoon/US morning hours. FX futures (CME) versus the OTC interdealer data used in the paper is a data-mismatch risk: the paper's edge is measured on an interdealer platform, not on CME futures, so tradable net edge on M6E is unverified.
- Verdict: best-published item in cluster (top-tier journal), but Sharpe 0.5-0.7 gross-or-net is unconfirmed and the futures-vs-spot mapping is unproven. Needs the PDF read.

### C4. Gold: London PM fix (15:00 GMT) leakage
- Evidence: Caminschi and Heaney, Journal of Futures Markets 2014 (peer-reviewed, via search summary of the abstract): GC futures and GLD show elevated volume/volatility right after the fixing starts, before the fix is published; trades in the first minutes predict the fixing direction, in some cases >90%; statistically significant return advantages in the first 4 minutes for informed traders; NO significant returns after publication of the fix.
- Decay: strongly post-publication. Barclays was fined USD 43.8m in May 2014; the London gold fix was reformed (LBMA Gold Price, electronic auction, from 2015). A later paper (Nilsson, "Did the New Fix, Fix the Fix?", SSRN 2657767) exists; I did not read it.
- Verdict: an informed-trader/leak finding, not a tradable retail edge, and the mechanism was legally targeted after 2014. Drop unless Nilsson finds residual effects. Not a weekly-cadence strategy in any case (it is daily).

### C5. Gold, silver, crude: generic intraday momentum (first/last half-hour) in commodity ETFs
- Rule: sign of a specific half-hour return predicts last half-hour return (PMC7480318): USO r1 (coef 0.0118, R^2 0.67%, OOS R^2 0.38%), GLD r5 (0.0436, 0.49%, OOS 0.26%), SLV r12 (0.1260, 1.72%, OOS 0.18%). Sample 2004/2006/2007 to 2019/2020. Stronger in high-volatility subsamples.
- The "market timing" Sharpe numbers reported in the paper are clearly implausible for tradable use (GVZ timing Sharpe 34.16, mean 26.96%; volatility index, not a tradable instrument) and the paper reports no transaction-cost adjustment. Treat as a statistical-predictability result, not a profit claim.
- Out-of-sample R^2 values of 0.2-0.4% are tiny; mean per trade after costs is almost certainly negative or unresolvable without a power gate. Multiple testing: the authors pick the best half-hour per instrument from 13, an obvious selection caveat.
- Verdict: pointer only.

### C6. Treasury auction-day price pressure (yields rise before and reverse after auctions)
- Evidence: NY Fed staff report (sr1188, PDF would not parse; search summary only): 33 years of intraday Treasury data; yields rise in hours before the auction and reverse afterwards; stronger when dealer risk constraints are tight; order flow explains it. I could not read the title, authors, date or magnitude. Sigaux (ECB WP 2208 2018; J. Banking and Finance 2024 per RePEc listing; Italian government bonds, 2.4 bp yield rise explained by announcements) is a different market (Italy).
- Events: 2y/5y/7y/10y/30y auctions occur multiple times per month (roughly weekly cadence; unverified count). Direction predetermined (short into auction, cover after) but pressure is a few bp of yield.
- Feasibility: ZN/ZF/ZB are full-size (DV01 large for a $2K trailing limit; belief). CME micro Treasury yield futures exist (belief, unverified, liquidity likely poor). Magnitude in bp of yield vs bid-ask spread in micro yield futures could kill it.
- Verdict: interesting, unread magnitude. Follow up only if the sr1188 numbers are large relative to tick size.

### C7. Jobless claims, other weekly releases
- Not found: no rigorous futures-based paper on weekly initial claims trading found in this run (see "could not find").

## Claims

| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| On EIA days, third-half-hour return positively predicts last-half-hour return in crude; non-EIA days only first half-hour | https://ideas.repec.org/a/sae/enejou/v44y2023i5p149-172.html | The Energy Journal (SAGE), via RePEc | 2023-09 | 2026-09-21 | med (abstract only) | quantitative (direction only, no size) |
| Authors claim "substantial economic gains" from EIA-day intraday predictors; no numbers retrieved | same as above | The Energy Journal | 2023-09 | 2026-09-21 | low | effect-size (unquantified) |
| USO first-half-hour predicts last: coef 0.0118, IS R^2 0.67%, OOS R^2 0.38%; GLD r5 OOS 0.26%; SLV r12 OOS 0.18%; OVX OOS -0.67% | https://pmc.ncbi.nlm.nih.gov/articles/PMC7480318/ | PubMed Central (peer-reviewed article) | ~2020 | 2026-09-21 | med | effect-size |
| Commodity-ETF timing Sharpe (e.g., 34.16 on GVZ) not tradable; no transaction costs modeled | https://pmc.ncbi.nlm.nih.gov/articles/PMC7480318/ | PubMed Central | ~2020 | 2026-09-21 | high | quantitative (caveat) |
| NG/heating-oil 2003-2006: expected inventory surprises affect far-month NG returns, coeff -0.0095 to -0.0141, p 0.044-0.074; no trading claims; 4-year sample | https://pmc.ncbi.nlm.nih.gov/articles/PMC4122141/ | PubMed Central (peer-reviewed article) | ~2014 | 2026-09-21 | med | effect-size |
| Storage report raises NG volatility at release and for ~30 minutes after | https://www.ou.edu/content/dam/price/Management/Energy%20Institute/docs/Linn%20and%20Zhu%20JFM%202004.pdf | Linn and Zhu, J. Futures Markets | 2004 | 2026-09-21 | med (search snippet) | quantitative |
| USD appreciates into FX fixes and depreciates after (W-shaped); Sharpe 0.5-0.7 for exploiting traders | https://ideas.repec.org/a/bla/jfinan/v79y2024i1p541-578.html | Journal of Finance | 2024 | 2026-09-21 | med (search summary; net-of-cost status unknown) | effect-size |
| 4pm fix: extreme returns 8.3-16 bp vs ~5 bp average daily; extreme-move probability 15.0% vs 9.5%; authors flag execution risk | https://ar5iv.labs.arxiv.org/html/1501.07778 | arXiv preprint (Evans) | 2015-01 | 2026-09-21 | med | quantitative (preprint) |
| WM/Reuters widened fix window from 1 to 5 minutes on 2015-02-15 | https://ar5iv.labs.arxiv.org/html/1501.07778 | arXiv | 2015 | 2026-09-21 | med | decay (structural break) |
| Pragma: after first-minute direction, month-end momentum ~6 bp reverting ~4 bp; quarter-end 10 bp reverting 6 bp | https://www.euromoney.com/article/b12kp4rx0dj2gd/trading-is-predictable-during-wm-4pm-fix-says-pragma | Euromoney (trade press; vendor research) | ~2015 | 2026-09-21 | low | effect-size |
| London PM gold fix: informed trading in first minutes before publication; no significant returns after publication; leak >90% direction predictive in some cases | https://onlinelibrary.wiley.com/doi/10.1002/fut.21636 | Journal of Futures Markets | 2014 | 2026-09-21 | med (search summary) | quantitative |
| Barclays fined USD 43.8m (May 2014) for manipulating gold fix; regulatory scrutiny followed research | https://researchimpact.uwa.edu.au/research-impact-stories/fixing-a-leaky-fixing/ | Univ. of Western Australia | ~2015 | 2026-09-21 | med | decay |
| Treasury auction-day pressure: yields rise in hours pre-auction, reverse after (33 years intraday); size not read | https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr1188.pdf | NY Fed staff report | unknown | 2026-09-21 | low (PDF unreadable; search summary) | quantitative |
| Sigaux: pre-auction price decline mechanism; 2.4 bp yield increase from announcements in Italian Treasuries | https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2208.en.pdf | ECB working paper (JBF 2024 per RePEc) | 2018-11 | 2026-09-21 | med | quantitative |
| Pre-FOMC drift disappeared after publication (analogy for decay risk; not this cluster) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7525326/ | PMC (title only, not read) | ~2020 | 2026-09-21 | low | decay |

## Leads worth chasing

1. Read the full Wen et al. (2023) EIA paper: sample period, instrument (futures or ETF?), the trading-rule mean/trade, costs, subperiods. Then replicate on the operator's own MCL/CL 1-min data with a power gate: ~52 events/yr, so N would reach ~100 in two years. Best cadence-fit in the cluster, and direction rule is mechanical.
2. Read Krohn, Mueller, Whelan (JF 2024) tables directly (BoC PDF https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf): confirm Sharpe 0.5-0.7 is net, sample period, post-2015 subsample, and whether CME 6E futures show the same pattern. Only viable if the pattern survives on futures bars at fix times.
3. NY Fed sr1188 Treasury auction paper: get the bp magnitude and the futures-based tradability; check if micro yield futures have tight enough spreads.
4. Nilsson (SSRN 2657767) on post-reform gold/silver fix.

## What I looked for and could not find

- Any independent replication of the EIA intraday momentum result, or any paper on post-2022 EIA-day performance.
- Effect sizes for any candidate in per-trade dollar/point units net of costs: none found.
- Natural-gas storage report trading-rule evidence beyond volatility/regression studies (all pre-2010 or unread).
- Weekly jobless-claims trading evidence on any futures (none in results).
- Rigorous (non-blog) evidence for gold/silver Asian/London/NY session-open effects: search returned only blog/broker content (Medium, Investing.com, Substack, edgeful); I did not cite any as evidence.
- Announcement-time spread/slippage numbers for MCL, M6E, or micro yield futures.
- Contract specs (tick value, margin, round-trip cost) for MCL, M6E, MGC, micro yield futures were not sourced this run; any mention above is labelled belief.
- Dependence on extreme events: no source reported per-event distributions.
