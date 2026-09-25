---
title: 'academic-lit research: does a verifiable market edge exist'
type: 'academic-lit'
topic: 'does a verifiable market edge exist'
decision: 'Should the project keep hunting trading edges, and which class of edge (if any) is plausibly reachable by a small automated operation?'
source: 'native run'
status: complete
preset: 'standard (4 researchers + red team)'
validation: 'normal (load-bearing claims two-source)'
created: '2026-09-25'
updated: '2026-09-25'
---

# academic-lit research: does a verifiable market edge exist

**Decision this research serves:** Should the project keep hunting trading edges, and which class of edge, if any, can a small automated operation plausibly reach?

## Executive summary

**Yes, edges exist, and some check out. But nearly every verifiable one belongs to a few players with a structural advantage, or to a tiny tail of skilled individuals. Almost all of the evidence says the kind this project has been hunting doesn't survive: price patterns in liquid index futures, traded intraday.**

1. **The verifiable edges are mostly structural rents: speed, spread capture at scale, order flow.**
   - Virtu reported **one losing day in 1,238** in its SEC filing [18].
   - HFT profits are concentrated in the fastest firms, the concentration is "high and non-declining", and new entrants earn less and exit [19].
   - Latency-arbitrage races take about 20% of equity volume, and **6 firms win >80%** of them [20].
   - Medallion's returns (66%/yr gross, 1988–2018 [21]) are the famous exception. They rest on a journal article and a book, not audited filings, and the fund is closed.
2. **Professionals with huge resources mostly don't beat the market net of fees.**
   - About **92%** of US domestic funds trailed their benchmarks over 20 years [1].
   - The best estimate of truly skilled funds net of fees is **0.6%** (2006), down from 14.4% in 1990 [4].
   - Skill exists before fees, about 9.6% of funds [4], but fees capture it [4][6]. Hedge-fund investors' realized alpha is about zero [7].
3. **Published "edges" mostly shrink to almost nothing once you trade them.**
   - Returns fall **58% after publication** [13].
   - Net of spreads, post-publication, in the modern trading era, the average of 204 anomalies earns **4 bps/month**, and the best at most 10 bps [15].
   - Only low-turnover designs keep significant net spreads [14]. The durable factors (value, momentum, trend) are partly compensation for risk [16].
4. **A tiny tail of individuals does have real, persistent skill. It isn't luck, but it's rare.**
   - Among *all* Taiwan day traders over 1992–2006, **fewer than 1%** were predictably profitable net of fees. The luck null is rejected for that top tail [24] (corroborated for US households [32]).
   - Their edge was short-horizon forecasting in hard-to-value stocks around earnings, not liquidity provision [24].
   - Everyone else loses. **97%** of persistent Brazilian futures day traders lost money, and only 0.4% earned more than a bank teller [26]. 74–89% of retail CFD accounts lose [27], and retail options traders lost $2.1B, mostly to spreads [28].
   - Topstep's own figures: **0.71%** of Express Funded traders reached a Live account [30].

**Biggest caveat:** almost all individual-skill evidence is **pre-2010 equities**, and none studies automated retail traders in index futures. "Essentially unreachable for small intraday futures systems" is a strong base rate, not a proof. The red team's qualified wording (below) is the version to cite.

## D1 — Professional managers and hedge funds

- **Underperformance:** about 92% of US domestic active funds underperformed over 20 years to 2025. In 2025, 79% of large-cap funds trailed the S&P 500 [1] (corroborated by the industry reports that republish SPIVA; SPIVA's primary pages returned 403, so medium confidence). A 2026 working paper challenges SPIVA's method (Open questions).
- **Persistence:** only 29% of 2023 top-quartile large-cap funds stayed top-quartile for the next two years. Almost none in any category stayed there over five years [2].
- **Luck vs skill:**
  - Fama & French: net of costs, few funds cover their costs, and true skill shows only in the extreme tails [3].
  - Barras, Scaillet & Wermers, 2,076 funds over 1975–2006: 75.4% zero-alpha, 24.0% unskilled, **0.6% skilled** net of fees. Before expenses, 9.6% are skilled [4].
  - Harvey & Liu (2022) show the older tests are either underpowered (Fama–French) or too permissive (Kosowski et al.), so the true size of the skilled share is still debated [5].
- **Where skill shows up:** skill exists and persists in *dollar value added*, about $3.2M a year for the average fund, but it goes to managers as fees and size [6].
- **Hedge funds:**
  - Investors' dollar-weighted returns were 3–7% a year below buy-and-hold, with alpha near zero [7].
  - Top-fund alpha does survive a bootstrap luck test [8], though that is the kind of test Harvey & Liu call too permissive [5].

## D2 — Published anomalies: replication, decay, costs

- **Replication, both sides:**
  - Hou, Xue & Zhang: 64% of 447 anomalies are insignificant and 85% fail t > 3 under value-weighting [9]. Harvey, Liu & Zhu argue for a t > 3 hurdle and say most claimed findings are likely false [10].
  - Jensen, Kelly & Pedersen: most factors replicate, cluster into 13 themes and work in 93 countries (gross returns) [11]. Chen & Zimmermann: 98% of originally significant predictors reproduce [12].
  - These measure different things: whether an in-sample result reproduces vs whether it's a robust, tradable premium.
- **Decay:** out-of-sample returns are 26% lower, and post-publication returns **58%** lower. What's left sits in illiquid, high-idiosyncratic-risk stocks [13] (independently corroborated by [15]).
- **Costs:**
  - Every anomaly loses to costs, but most with monthly turnover under 50% keep significant net spreads when built to limit costs [14].
  - The modern-era net average is **4 bps/month** (JFQA 2023; verified via 3 listings) [15].
- **Risk vs mispricing:** value and momentum work across 8 markets and asset classes, and funding-liquidity risk is only a partial explanation [16].
- **Machine learning:** its gains concentrate in hard-to-arbitrage stocks and periods, and mostly vanish with microcaps excluded and realistic costs included [17].

## D3 — Who verifiably has an edge

| Holder | What kind of edge | Evidence | Confidence |
|---|---|---|---|
| Virtu and other market makers | Spread capture at huge volume | S-1: 1 losing day in 1,238 (2009–13); $182M net income on 151 staff (2013) [18] | High (regulatory filing) |
| Fastest HFT firms | Speed / latency | Performance ranks by relative speed; concentration is high and non-declining; entrants slower and exit [19]. Latency races: about 20% of volume, 6 firms win >80%, about $5B/yr [20] | High (JFQA, QJE) |
| Medallion (Renaissance) | Unknown, statistical | 66%/yr gross, no losing year, negative beta, 1988–2018 [21] | Medium (unaudited; book plus a practitioner journal) |
| Index arbitrageurs | Structural flow | The S&P 500 inclusion effect fell from 7.4% (1990s) to under 1%, *traded away* by arbitrageurs [22] | High |
| Pairs traders (historical) | Relative value | Up to 11%/yr over 1962–2002 [23]; decay since then is not retrieved here | Medium |

**What they share:** each has an identifiable *reason it persists*. That is either a barrier to entry (speed, capital, order flow) or a service someone pays for (liquidity). The one flow edge without a barrier, index inclusion, was arbitraged away [22]. Nothing retrieved shows an edge available to a small trader without colocation or large capital.

## D4 — Retail and small traders

| Population | Result | Source |
|---|---|---|
| Taiwan day traders (all, 1992–2006) | About 20% profitable in a given year; **<1% predictably profitable** net of fees. The top 500 earn 37.9 bps/day net the next year | [24] |
| Taiwan day traders | The population lost money net of fees **every year for 15 years**; >75% quit within 2 years; losers keep trading almost as much as winners | [25] |
| Brazil equity-futures day traders (≥300 days) | **97% lost money**; 0.4% beat a bank teller's wage; no learning | [26] |
| Retail CFD accounts (EU, 2018) | **74–89% lose**; average loss €1.6K–€29K | [27] |
| Retail options (US, 2019–21) | Lost **$2.1B**, mostly spreads (12.6% on weeklies) | [28]; earnings-event losses 5–14% [29] |
| Topstep 2025 cohort (firm-disclosed, unaudited) | 16.8% of Combines passed; 33.3% of Funded-level traders got a payout; **0.71%** of XFA traders reached Live | [30] |
| Retail order flow (stocks) | Net retail buying predicts about 10 bps/week of stock returns: informative in aggregate, not proof that accounts profit after costs | [31] |

## Cross-dimension insights

1. **Every durable edge answers "who pays me, and why can't others take it?"** Market makers are paid for liquidity and protected by scale and speed [18][19][20]. Factor premia are paid for bearing risk [16]. The tiny tail of skilled individuals is paid for processing hard-to-value information in capacity-limited niches [24]. Edges that are just patterns in price data get published or arbitraged away [13][15][22]. That's the common thread across all four dimensions.
2. **Turnover is the dividing line.** Net-of-cost survival falls steeply with turnover [14][15], and HFT is the exception only because its costs are tiny and its speed is unmatched [19][20]. A small intraday strategy faces high turnover *without* that cost or speed advantage, which is the worst of both.
3. **The persistent individuals aren't doing what automated index-futures systems do.** Their edge is in hard-to-value single stocks around information events [24], not in the most liquid, most-watched instrument where HFTs set prices.

## Contrary evidence (red team: a fresh-context skeptic searched for evidence that small traders *can* reach edges)

**Verdict: the prior conclusion survives only in qualified form.**
- **"Profitable individuals are indistinguishable from luck" is FALSE as written.** The luck null is rejected for a small top tail, net of fees [24]. US households' top decile earns about 6%/yr risk-adjusted afterwards [32].
- **"Only structural firms have edges" is too strong.** Low-turnover anomalies survive costs [14], factors replicate out of sample [11], and emerging small funds outperform early by about 2.3%/yr [33].
- **"Mostly decay or eaten by costs" holds for high-turnover strategies,** but decay isn't to zero: about 42% remains [13]. The net figure is small, though, at 4 bps/month [15].
- **"Essentially unobtainable" survives as a base rate.** The counter-evidence covers the top <1%, slow long-short equity books, institutions, or pre-2010 data. None shows a small intraday automated futures trader with verified, persistent net skill.
- Crypto cross-exchange arbitrage gaps were large and persistent, but blocked by capital controls, not speed [34].

**Qualified wording to cite:** "Verifiable net-of-cost edges for small independents are rare (<1% persistence in the best population data). They cluster in low-turnover, capacity-limited or hard-to-value niches, with evidence mostly predating 2010. They are not impossible, and the top tail is distinguishable from luck. For high-turnover intraday strategies, the negative conclusion is well supported."

## Recommendations

1. **Stop hunting price-pattern edges in liquid index futures intraday.** The literature [13][14][15][19][20][24][26] and the project's own record agree. Confidence: **high.** This is an architecture constraint on the research queue.
2. **If research continues, pre-screen every idea with "who pays me, and why can't a faster or bigger player take it?"** Make it a required section in the pre-registration template, alongside the power gate. Ideas without a named payer and barrier get closed before any data work. Confidence: medium-high.
3. **Only two directions have evidence behind them for a small operator.**
   - (a) **Harvesting risk premia** (diversified value, momentum or trend exposure), framed as investing with drawdowns, not an edge [16].
   - (b) **Capacity-limited, information-processing niches**: small, hard-to-value stocks around events [24][13]. That needs a different skill and data set, and its evidence is mostly pre-2010.

   Neither is intraday futures. Confidence: medium.
4. **Treat the prop-firm route as low odds** (0.71% of XFA traders reach Live, firm-disclosed [30]). Combined with the Topstep Live-API ban found earlier today, the automated combine path is not an income route. Confidence: medium (unaudited).
5. **Revisit the income aspiration against this evidence.** Income from this operation is unlikely to come from an edge that checks out. A conscious decision is needed: pause and invest passively, pivot to (a) or (b), or keep a low-cost research hobby. Confidence: high on the evidence, judgment on the choice.

## Open questions

- Individual-skill persistence data after 2010, and for futures specifically. Lead: Kuo et al., Taiwan futures day traders (IRABF). Nothing retrieved covers automated retail systems.
- Audited records of small independent traders that survive luck adjustment. None found.
- Whether 0.6% understates skill: Andrikogiannopoulou & Papakonstantinou argue the Barras–Scaillet–Wermers method lacks power (lead).
- The 2026 critique of SPIVA's method (SSRN 6710358); Medallion's audited evidence; Frazzini/Israel/Moskowitz's lower institutional cost estimates.

## Source appendix

| # | Supports | Publisher | Pub date | Accessed | Confidence |
|---|---|---|---|---|---|
| [1] | 92% trail benchmark over 20y | [S&P DJI SPIVA U.S. YE2025](https://www.spglobal.com/spdji/en/spiva/article/spiva-us/) | 2026 | 2026-09-25 | medium |
| [2] | Persistence | [S&P DJI Persistence YE2025](https://www.spglobal.com/spdji/en/spiva/article/us-persistence-scorecard/) | 2026 | 2026-09-25 | medium |
| [3] | Luck vs skill | [Fama & French, JF 2010](https://econpapers.repec.org/RePEc:bla:jfinan:v:65:y:2010:i:5:p:1915-1947) | 2010 | 2026-09-25 | high |
| [4] | 0.6% skilled net of fees | [Barras, Scaillet & Wermers, JF 2010](https://terpconnect.umd.edu/~wermers/FDR_published.pdf) | 2010 | 2026-09-25 | high (full text) |
| [5] | Test power critique | [Harvey & Liu, JF 2022](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13123) | 2022 | 2026-09-25 | medium-high |
| [6] | Skill captured as fees | [Berk & van Binsbergen, JFE 2015](https://ideas.repec.org/a/eee/jfinec/v118y2015i1p1-20.html) | 2015 | 2026-09-25 | medium-high |
| [7] | Hedge-fund investor alpha ≈ 0 | [Dichev & Yu, JFE 2011](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1354070) | 2011 | 2026-09-25 | medium-high |
| [8] | Top hedge-fund alpha not luck | [Kosowski, Naik & Teo, JFE 2007](https://www.sciencedirect.com/science/article/abs/pii/S0304405X06002017) | 2007 | 2026-09-25 | medium |
| [9] | 64% of anomalies insignificant | [Hou, Xue & Zhang, RFS 2020](https://www.nber.org/papers/w23394) | 2020 | 2026-09-25 | high |
| [10] | t > 3 hurdle | [Harvey, Liu & Zhu, RFS 2016](https://www.nber.org/papers/w20592) | 2016 | 2026-09-25 | high |
| [11] | Factors replicate | [Jensen, Kelly & Pedersen, JF 2023](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249) | 2023 | 2026-09-25 | medium |
| [12] | 98% reproduce | [Chen & Zimmermann, CFR 2022](https://doi.org/10.17016/feds.2021.037) | 2022 | 2026-09-25 | medium |
| [13] | 58% post-publication decay | [McLean & Pontiff, JF 2016](https://ideas.repec.org/a/bla/jfinan/v71y2016i1p5-32.html) | 2016 | 2026-09-25 | high |
| [14] | Low-turnover survives costs | [Novy-Marx & Velikov, RFS 2016](https://academic.oup.com/rfs/article-abstract/29/1/104/1844518) | 2016 | 2026-09-25 | medium |
| [15] | 4 bps/month net | [Chen & Velikov, JFQA 2023](https://www.federalreserve.gov/econres/feds/zeroing-in-on-the-expected-returns-of-anomalies.htm) | 2023 | 2026-09-25 | high (verified via 3 listings) |
| [16] | Value/momentum everywhere | [Asness, Moskowitz & Pedersen, JF 2013](https://ideas.repec.org/a/bla/jfinan/v68y2013i3p929-985.html) | 2013 | 2026-09-25 | high |
| [17] | ML gains in hard-to-arbitrage stocks | [Avramov, Cheng & Metzker, Mgmt Sci 2023](https://ideas.repec.org/a/inm/ormnsc/v69y2023i5p2587-2619.html) | 2023 | 2026-09-25 | medium |
| [18] | Virtu: 1 losing day in 1,238 | [Virtu S-1, SEC EDGAR](https://www.sec.gov/Archives/edgar/data/1592386/000104746914002070/a2218589zs-1.htm) | 2014 | 2026-09-25 | high |
| [19] | HFT profits rank by speed | [Baron, Brogaard, Hagströmer & Kirilenko, JFQA 2019](https://econpapers.repec.org/RePEc:cup:jfinqa:v:54:y:2019:i:03:p:993-1024_00) | 2019 | 2026-09-25 | high |
| [20] | Latency-arbitrage races | [Aquilina, Budish & O'Neill, QJE 2022](https://academic.oup.com/qje/article/137/1/493/6368348) | 2022 | 2026-09-25 | high |
| [21] | Medallion returns | [Cornell, JPM 2020](https://jpm.pm-research.com/content/46/4/156) | 2020 | 2026-09-25 | medium |
| [22] | Disappearing index effect | [Greenwood & Sammon, JF 2025](https://onlinelibrary.wiley.com/doi/10.1111/jofi.13410) | 2025 | 2026-09-25 | high |
| [23] | Pairs trading 1962–2002 | [Gatev, Goetzmann & Rouwenhorst, RFS 2006](https://academic.oup.com/rfs/article-abstract/19/3/797/1646694) | 2006 | 2026-09-25 | medium |
| [24] | <1% predictably profitable | [Barber, Lee, Liu & Odean, JFM 2014](https://faculty.haas.berkeley.edu/odean/papers/day%20traders/The%20Cross-Section%20of%20Speculator%20Skill.pdf) | 2014 | 2026-09-25 | high (full text, 2 readers) |
| [25] | Population loses every year; persistence of losers | [Barber et al., learning draft](https://faculty.haas.berkeley.edu/odean/papers/Day%20Traders/Day%20Trading%20and%20Learning%20110217.pdf) | 2017 | 2026-09-25 | medium (working paper) |
| [26] | 97% of Brazil futures day traders lose | [Chague, De-Losso & Giovannetti](https://ideas.repec.org/p/fgv/eesptd/525.html) | 2020 | 2026-09-25 | medium (working paper) |
| [27] | 74–89% of CFD accounts lose | [ESMA product intervention](https://www.esma.europa.eu/press-news/esma-news/esma-agrees-prohibit-binary-options-and-restrict-cfds-protect-retail-investors) | 2018 | 2026-09-25 | medium |
| [28] | Retail options −$2.1B | [Bryzgalova, Pavlova & Sikorskaya, JF 2023](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13285) | 2023 | 2026-09-25 | medium |
| [29] | Retail options losses at earnings | [de Silva, Smith & So, RoF 2026](https://academic.oup.com/rof/article-abstract/30/2/489/8301159) | 2026 | 2026-09-25 | medium |
| [30] | Topstep 2025 cohort metrics | [Topstep](https://www.topstep.com/topstep-prop) | 2026 | 2026-09-25 | medium (firm-disclosed; echoed by secondary sites) |
| [31] | Retail flow informative | [Boehmer, Jones, Zhang & Zhang, JF 2021](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13033) | 2021 | 2026-09-25 | medium |
| [32] | US household top-decile persistence | [Coval, Hirshleifer & Shumway, RAPS 2021](https://www.tylergshumway.org/Individual_Trading_Coval_Hirshleifer_Shumway_raps3.pdf) | 2021 | 2026-09-25 | medium-high |
| [33] | Emerging funds outperform early | [Aggarwal & Jorion, JFE 2010](https://www.sciencedirect.com/science/article/abs/pii/S0304405X0900258X) | 2010 | 2026-09-25 | medium |
| [34] | Crypto arbitrage behind capital controls | [Makarov & Schoar, JFE 2020](https://ideas.repec.org/a/eee/jfinec/v135y2020i2p293-319.html) | 2020 | 2026-09-25 | medium |

## Staleness map

Computed with `recon_kit.py staleness`, using this pack's windows: industry data and state-of-the-art claims 12 months; seminal work has no bar but is checked for being superseded.

| Claim | Class | Re-check by |
|---|---|---|
| SPIVA ~92% underperformance [1] | industry data | 2027-03 (next year-end scorecard) |
| Topstep 2025 cohort metrics [30] | industry data | 2027-01 (next cohort disclosure) |
| "No post-2010 individual-skill persistence study found" | state of the art | **2027-09**: the absence should be re-searched |
| Chen–Velikov 4 bps/month [15] | state of the art | **already past its 12-month window** (2023). Check for a follow-up or a replication of the net-return estimate before relying on it for a new decision |
| Barras–Scaillet–Wermers [4]; HFT concentration [19][20] | seminal | no bar; watch for supersession, e.g. the Andrikogiannopoulou–Papakonstantinou reassessment |

The earliest dated re-check is the Chen–Velikov follow-up, which is already due, but it doesn't change the direction of the conclusion.
