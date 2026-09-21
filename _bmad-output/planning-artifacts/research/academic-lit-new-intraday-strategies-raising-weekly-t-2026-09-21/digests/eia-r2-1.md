# EIA-report-day crude intraday momentum (Wen, Indriawan, Lien, Xu 2023) — depth digest

Accessed 2026-09-21. Paper text read: the ACCEPTED VERSION (post-peer-review, pre-typeset) deposited at University of Adelaide, obtained via CORE (https://core.ac.uk/download/612285763.pdf; handle http://hdl.handle.net/2440/141224). Peer-reviewed journal: The Energy Journal 44(5):149-172, Sept 2023. All numbers below are from that PDF unless labelled DERIVED (my arithmetic on the paper's numbers) or UNVERIFIED.

## Findings

### 1. Sample, instrument, windows
- **Instrument is NOT crude futures. It is the USO ETF (United States Oil Fund)**, 1-minute data from Refinitiv Tick History (trade price, volume, trade count, bid/ask). USO trades on NYSE Arca 9:30-16:00 ET regular session (extended 4:00-20:00).
- Sample 2006-04-10 (USO launch) to **2019-07-31**. Days with <500 trades dropped -> 3,240 trading days. 685 EIA reports released; the paper uses only the **591 that fell on Wednesday at 10:30 ET** (Thursday/Friday/Monday releases at 11:00 excluded). Non-EIA days = 2,649.
- Windows: 13 half-hour intervals 9:30-16:00. r1 = 16:00 prior close -> 10:00 (includes overnight); r3 = **10:30 -> 11:00** (the half-hour beginning at the EIA release); r13 = **15:30 -> 16:00**. Signal known at 11:00; trade is long/short USO at 15:30, flat at 16:00 close. No look-ahead in construction. Holding period is 30 min.
- 2020 negative-price episode: **not in sample** (ends July 2019). 2022+ regime: not covered. Paper is silent on both.

### 2. Regression results (Table 1, Newey-West t in brackets)
- EIA days (N=591): r3 coefficient 0.038 (t=2.12**), adj R2 3.10%; with other half-hour dummies 0.042 (t=2.85***), adj R2 8.15%. r1 coefficient 0.015 (t=1.61, n.s.).
- Non-EIA days (N=2,649): r1 = 0.011** (t=2.44), adj R2 0.68%; r3 ~ -0.011 (n.s.).
- Pooled interaction EIA*r3 = 0.050*** (t=2.89); with controls 0.043** (t=2.27).
- Out-of-sample R2: **none reported** (searched full text; no out-of-sample or holdout exercise). All results are in-sample, full-sample.
- Table 7: r3 windows shorter than 15 min (10:30-10:35, 10:30-10:40) are NOT significant (EIA*r3 = -0.006, 0.022); significance appears only at 10:30-10:45 and longer (0.052**, 0.056**, 0.053**, 0.050***).

### 3. Trading-rule results (Table 6, EIA days, signal r3, hold last half-hour)
- Reported: mean 4.14% (t=1.88, significant only at 10% level, marked *), Std 0.21, Sharpe 19.54, skewness 0.66, kurtosis 10.09, success rate 58% (success defined as zero-or-positive payoff, so ties count as wins). Number of trades not stated in the table; the EIA sample is 591 days.
- **Units decoded (DERIVED, checks against the paper's own t-stats):** "Mean" is an annualised % (daily mean x252), "Std" is a DAILY % of the ETF price. Check: long-only 1.04/252 = 0.0041%/day, / 0.20% x sqrt(3240) = 1.17 vs paper t=1.16; EIA rule 4.14/252 = 0.0164%/day, /0.21% x sqrt(591) = 1.90 vs paper t=1.88. So the "Sharpe 19.54" is annualised-mean / daily-SD (mean/std of the printed columns), not a proper annualised Sharpe; do not cite it.
- DERIVED per-trade figures for the EIA rule: mean ~ +1.6 bp of USO price per trade (~0.0164%), SD ~ 21 bp, per-trade standardised mean ~ 0.078. At ~44 EIA Wednesdays/yr (591/13.3 yr), event-only annualised Sharpe ~ 0.078 x sqrt(44) ~ 0.5 GROSS (derived; t=1.88 over 13.3 yr is consistent with this).
- **Costs: NOT charged.** Full text contains no transaction-cost, commission, slippage or spread deduction for the strategy (searched cost/commission/slippage/fee: only hits are "adverse selection costs" in the mechanism section). Payoffs are close-to-close-type log returns at half-hour prices, gross. Abstract's "substantial economic gains" is a gross, in-sample, ETF number.
- Other rows: non-EIA r1 rule 1.88% (t=2.10); overnight rule 2.39% (t=2.45); r3 on non-EIA days -0.36% (n.s.).

### 4. Subperiod stability and concentration
- Figure 4 (5-year rolling, 1,250-day windows, EIA*r3 coefficient): the authors write the effect is "strong and statistically significant over our sample period, albeit with a declining trend", and that **"during the crude oil price plunge between 2014 and 2016, the intraday return predictability is not observed"**. No post-2016 or post-2020 evidence is possible from this sample. Rolling windows are 5 years so subperiod robustness is coarse, and only a figure is given (no table).
- Volatility (Table 3, median split of daily realised vol): EIA*r3 is significant only in the high-vol half (r3=0.043**, adj R2 3.89%, N=347); low-vol half r3=0.013 (n.s., adj R2 0.25%, N=244). Pooled interaction 0.061*** high-vol vs 0.0013 low-vol. Footnote: persists in the highest RV quartile.
- Month half (Table 4): EIA*r3 = 0.078*** in the first half of the month vs 0.014 (n.s.) in the second half (authors note it is "stronger if EIA announcements occur during the first half of the month"). This is a post-hoc split with a null in half the data.
- Extreme days: no winsorisation, jackknife, outlier-removal or leave-out-top-days test. Kurtosis 10.09 and skew 0.66 on the EIA rule show fat tails; the long-only benchmark kurtosis is 14.49. Whether a few days drive the t=1.88 is untested. Note that Table 3/4 splits already show the effect is confined to about a third to a half of EIA days.
- The strategy t-stat (1.88) is weaker than the regression t (2.12-2.89); the regression coefficient may be leveraged by large-|r3| days.

### 5. Independent replication / related evidence
- **No independent replication found.** Semantic Scholar shows 3 citations to the paper (retrieved via API); I did not retrieve or read the citing papers. No critical paper found.
- Same-group earlier work: Wen, Gong, Ma, Xu, "Intraday momentum and return predictability: Evidence from the crude oil market" (Economic Modelling 2021; SSRN 3553682) — abstract page was blocked (403), NOT READ; it is by overlapping authors, so not independent.
- Related but different questions: "Intraday return predictability in China's crude oil futures market" (Economic Modelling 96, 2021; RePEc listing only seen in search results, not read). Miao 2026 J. Futures Markets "Intraday Liquidity in International Crude Oil Futures Markets: News Impacts, Commonality, and Spillovers" (Wiley 403; only the title seen in a search list). The 2016 Energy Economics paper (Ye/Karali line, "informational content of inventory announcements: intraday evidence from crude oil futures") shows up in searches; only titles seen.
- Related within the paper: intraday momentum in non-EIA days rests on r1/overnight (SPY-style Gao et al. 2018 pattern), a separate hypothesis.

### 6. Contract feasibility (MCL)
- CME Micro WTI (MCL): 100 barrels (1/10 of CL), tick $0.01/bbl = **$1.00 per tick**, cash-settled, ~23h/day Sun-Fri (5:00 PM-4:00 PM CT, 1h halt). Source: CME Education page and broker pages as shown in search-result snippets (CME fact card and contract-specs pages timed out / returned non-PDF this run; snippet-level only, confidence medium).
- **Typical bid-ask spread, liquidity in 11:00-16:00 ET, round-trip commission, and EIA-time spread widening for MCL/CL: NOT RETRIEVED.** No number in the paper for USO spreads by time of day either, apart from a qualitative statement that spreads are J-shaped (higher at open and close) and liquidity (Amihud) is worse in the high-predictability days (Table 5: Amihud high r3=0.078**, low r3=0.021 n.s.; small vs large trade-size difference small).
- DERIVED order-of-magnitude: at WTI ~ $60/bbl, MCL notional ~ $6,000; a 1.6 bp gross mean = ~$1.0 per contract per trade, i.e. about ONE tick, against an SD of ~ 21 bp ~ $12.6. Minimum plausible costs (one-tick spread at entry+exit and exchange/broker fees per side) are of the same order as or larger than the gross mean edge. Price level assumption ($60) is mine, not sourced; at $80 both scale up 33% and the ratio is unchanged (edge in bp vs cost in ticks depends on price only via tick/notional).
- Timing note: USO 15:30-16:00 ET is after the NYMEX CL 14:30 ET settlement; futures mapping (15:30-16:00 vs 13:30-14:30 settle) is not tested by the paper. The strategy would trade MCL in the post-settlement window; USO's close at 16:00 is not identical to futures. USO is a futures-holding ETF (contango roll cost visible in its -15%/yr buy-and-hold), so it is a proxy, not the same price series.

## Claims

| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| Sample is USO ETF 1-min data, 2006-04-10 to 2019-07-31, 3,240 days, not futures | https://core.ac.uk/download/612285763.pdf | Energy Journal / IAEE (accepted version via U Adelaide/CORE) | 2023-09 | 2026-09-21 | high | peer-reviewed (accepted manuscript) |
| 591 Wednesday 10:30 EIA days; other-weekday releases at 11:00 excluded | same | same | 2023-09 | 2026-09-21 | high | peer-reviewed |
| EIA-day r3 coefficient 0.038-0.042, adj R2 3.10%/8.15% | same (Table 1) | same | 2023-09 | 2026-09-21 | high | peer-reviewed |
| Rule mean 4.14% (annualised), t=1.88, Std 0.21, success 58%, kurtosis 10.09 | same (Table 6) | same | 2023-09 | 2026-09-21 | high | peer-reviewed |
| Table 6 mean is annualised, Std is daily; "Sharpe 19.54" is mean/Std of those columns | same (my arithmetic, reproduces the paper's t-stats within rounding) | derived | n/a | 2026-09-21 | med | derived |
| Per-trade mean ~1.6 bp, SD ~21 bp of USO price; event-only gross Sharpe ~0.5 | same (derived) | derived | n/a | 2026-09-21 | med | derived |
| No transaction costs charged | same (full-text search for cost terms) | same | 2023-09 | 2026-09-21 | high | peer-reviewed (absence) |
| No out-of-sample test reported | same | same | 2023-09 | 2026-09-21 | high | peer-reviewed (absence) |
| Predictability not observed during 2014-2016 oil price plunge; declining trend | same (Section 4.2, Fig. 4) | same | 2023-09 | 2026-09-21 | high | peer-reviewed |
| Effect significant only in high-vol half; first half of month only | same (Tables 3, 4) | same | 2023-09 | 2026-09-21 | high | peer-reviewed |
| r3 windows <15 min not significant | same (Table 7) | same | 2023-09 | 2026-09-21 | high | peer-reviewed |
| Published as Energy Journal 44(5):149-172, Sept 2023; abstract text | https://ideas.repec.org/a/sae/enejou/v44y2023i5p149-172.html | RePEc/IDEAS | 2023-09 | 2026-09-21 | high | index |
| Paper has 3 citations on Semantic Scholar; green-OA PDF at Adelaide | https://api.semanticscholar.org/graph/v1/paper/DOI:10.5547/01956574.44.4.zwen | Semantic Scholar | n/a | 2026-09-21 | med | index |
| MCL: 100 bbl, tick $0.01 = $1.00, cash-settled, 23h trading | search-result snippets citing https://www.cmegroup.com/education/courses/basic-principles-of-micro-wti-crude-oil-futures/micro-wti-crude-oil-futures-overview and broker pages (Ironbeam, Metro) | CME Group / brokers | n/a | 2026-09-21 | med | vendor doc (snippet only) |

## Inputs a power test would need

| input | value | source | status |
|---|---|---|---|
| Events per year | ~44 (591 EIA Wednesdays / 13.3 yr); real-world ~52 less holidays/Thursday shifts | paper (591, 2006-2019) | have (in-sample count) |
| Gross mean per trade | ~ +1.6 bp of price (DERIVED from 4.14% annualised /252); regression slope 0.038-0.042 on r3 | paper Table 6/1, my decode | have, but derived and in-sample (upward-biased by selection of the r3 window) |
| SD per trade | ~ 21 bp (0.21% daily) of price | paper Table 6 | have (USO, pre-2019) |
| Per-trade standardised effect | ~0.078 (t=1.88 on N=591) | paper, derived | have |
| Cost per round trip (MCL) | NOT RETRIEVED: spread at 15:30-16:00 ET (in ticks), commission/exchange/NFA fees, slippage | none | **MISSING** — needs CME/broker fee schedule and observed MCL quotes |
| Price level / notional per contract | assumption ~ $60 x 100 bbl (unsourced) | mine | **MISSING**; replace with current front-month |
| Out-of-sample / post-2019 effect | none | none | **MISSING**; effect may be smaller (paper reports decline; 2014-16 null) |
| Futures-vs-ETF mapping and 15:30-16:00 vs 14:30 settlement | not tested | none | **MISSING** |
| Days per regime (vol filter) | high-vol half only carries effect (N=347 EIA days in high-vol) | paper Table 3 | have, but post-hoc split |

Power sketch (DERIVED, not in paper): even at the reported gross effect of ~0.078 SD per trade, N=591 gave only t=1.88; 80% power at alpha=5% one-sided needs about (2.49/0.078)^2 ~ 1,020 trades ~ 23 years at 44/yr, before any cost or decay haircut. Order-of-magnitude, arithmetic only.

## Contrary evidence found
- Strategy t-stat is only 1.88 (10% level) in the paper's own headline result, with no cost charge and no OOS test.
- Predictability absent in 2014-2016 and trending down; authors themselves attribute it to low-price regimes.
- Confined to high-volatility days and to the first half of the month (post-hoc splits, each n.s. in the complement).
- Requires r3 windows >=15 min; shorter windows fail (Table 7).
- Instrument mismatch: USO ETF, sample ended 2019; futures/MCL, 2020 and 2022+ untested.
- Derived gross edge (~1.6 bp) is about one MCL tick in dollars, before costs.
- No independent replication located; the only other same-topic paper found is by overlapping authors.

## What I could not find
- Independent replication or critical citations (3 citing papers per Semantic Scholar; not read). Routes not read: Google Scholar citation list, the SSRN Wen-Gong-Ma-Xu abstract (403), ScienceDirect Economic Modelling (403), Wiley Miao 2026 (403), SAGE journal page (403), SSRN PDF (403), ResearchGate (403), web.archive.org (blocked by tool). The web-reader/web-search-prime tools were unavailable (subscription expired).
- Working-paper vs published differences (only the accepted manuscript was read; SSRN version 3822093 not opened).
- MCL bid-ask spread by time of day, liquidity 11:00-16:00 ET, typical round-trip commission, and EIA-time spread evidence for futures: CME contract-spec and fact-card fetches timed out or returned no PDF; only search-snippet specs obtained. Exact fee schedule not retrieved.
- Any post-2019 evidence (2020 negative prices, 2022+).
- Winsorised / top-days-removed results (not in paper).
