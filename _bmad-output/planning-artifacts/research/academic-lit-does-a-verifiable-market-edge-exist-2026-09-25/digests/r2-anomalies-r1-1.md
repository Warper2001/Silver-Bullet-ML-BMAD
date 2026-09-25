# Digest r2-anomalies-r1-1: Published anomalies and factors (replication, costs, post-publication decay)

Decision served: do durable, verifiable trading edges (risk-adjusted excess returns net of costs) exist, and do published ones survive?
Dimension: published cross-sectional anomalies and factors.
Accessed: 2026-09-25. Tool calls used: 18 of 18. Sources: 10 papers (cap 10).

Retrieval notes: SSRN returned HTTP 403 every time. Abstracts were therefore taken from NBER abstract pages (HLZ, HXZ, JKP, NMV) and from the OpenAlex API (`abstract_inverted_index`). OpenAlex serves the publisher's own abstract text, so each paper counts as one primary source. The IDEAS/RePEc pages for MP and AMP gave bibliographic data only. Every number below comes from the paper's own abstract. None comes from a secondary summary. Abstract-level evidence cannot show methods detail, so the confidence ratings reflect that limit.

Class key: `replication`, `decay`, `costs`, `risk-vs-mispricing`, `ML`, `methodology`.

---

## Claims

### Q1: Do published anomalies replicate? (both sides)

1. **Claim:** Hundreds of papers propose hundreds of factors. After adjusting for multiple testing, a new factor should clear a t-ratio above 3.0, not the traditional 2.0. The authors conclude that "most claimed research findings in financial economics are likely false."
   - Source: https://www.nber.org/papers/w20592
   - Authors: Campbell R. Harvey, Yan Liu, Heqing Zhu
   - Pub date: NBER WP Oct 2014; *Review of Financial Studies* 29(1):5–68, 2016
   - Peer-reviewed: yes (RFS)
   - Accessed: 2026-09-25 · Confidence: high (abstract wording) · Class: methodology / replication
   - Caveat: the abstract gives no percentage of false findings. The specific figure the brief asked for was **not retrieved** (see "Looked for").

2. **Claim:** Of 447 anomaly variables, 286 (64%) are insignificant at the 5% level and 380 (85%) fail a t>3 cutoff. This holds once microcaps are muted with NYSE breakpoints and value-weighted returns. In the liquidity category, 95 of 102 (93%) are insignificant. For the 161 anomalies that remain significant, the originally reported magnitudes are much larger than the replicated ones. The authors conclude that "capital markets are more efficient than previously recognized."
   - Source: https://www.nber.org/papers/w23394
   - Authors: Kewei Hou, Chen Xue, Lu Zhang
   - Pub date: NBER WP May 2017; *Review of Financial Studies* 33(5):2019–2133, 2020
   - Peer-reviewed: yes (RFS)
   - Accessed: 2026-09-25 · Confidence: high · Class: replication

3. **Claim (counter-side):** A Bayesian model of factor replication finds that the majority of asset pricing factors:
   - (i) can be replicated;
   - (ii) cluster into 13 themes, most of which are significant parts of the tangency portfolio;
   - (iii) work out-of-sample in a new dataset covering 93 countries;
   - (iv) have evidence that is *strengthened*, not weakened, by the large number of observed factors.

   The authors reject the "replication crisis" framing.
   - Source: https://www.nber.org/papers/w28432 (abstract also via OpenAlex, DOI 10.1111/jofi.13249)
   - Authors: Theis Ingerslev Jensen, Bryan T. Kelly, Lasse Heje Pedersen
   - Pub date: NBER WP Feb 2021; *Journal of Finance* 78(5):2465–2518, 2023
   - Peer-reviewed: yes (JF)
   - Accessed: 2026-09-25 · Confidence: high for the qualitative claim · Class: replication
   - Caveat: the abstract says "majority". The exact replication percentage was **not retrieved**. The abstract also gives no magnitude for out-of-sample performance, so "works out-of-sample" does not show that returns survived undiminished.

4. **Claim (counter-side):** The authors release open code that reproduces "nearly all" cross-sectional predictors, 319 characteristics in total.
   - Of the 161 characteristics that were clearly significant in the original papers, 98% of the reproduced long-short portfolios have t>1.96.
   - The 44 characteristics with mixed original evidence reproduce at t≈2 on average.
   - Regressing reproduced t-stats on the originals gives slope 0.90 and R² 83%.
   - The remaining 114 were either insignificant in the original papers or are modifications created by Hou, Xue & Zhang (2020).
   - Source: https://doi.org/10.17016/feds.2021.037 (FEDS abstract via OpenAlex); site https://www.openassetpricing.com/
   - Authors: Andrew Y. Chen, Tom Zimmermann
   - Pub date: FEDS WP 2021-037; *Critical Finance Review* 2022, pp. 207–264 (DOI 10.1561/104.00000112)
   - Peer-reviewed: the numbers quoted are from the working-paper (FEDS) abstract. The CFR version is peer-reviewed, but its abstract was not available (null in OpenAlex).
   - Accessed: 2026-09-25 · Confidence: high for the FEDS numbers; medium that they carry over unchanged to CFR · Class: replication

5. **Synthesis (my inference from claims 2–4, not a sourced claim):** The two sides measure different things, so they do not directly contradict each other.
   - Chen & Zimmermann (and JKP) ask whether the original in-sample result can be reproduced under the original specification. It usually can.
   - Hou, Xue & Zhang ask whether the anomaly survives value-weighting, NYSE breakpoints and a stricter hurdle. It usually does not, and their 447 includes variants that the CZ abstract says were never significant originally.
   - Neither side speaks to *net-of-cost, post-publication* profitability. That question is covered by claims 6–8.
   - Class: replication (interpretation)

### Q2: Out-of-sample and post-publication decay

6. **Claim:** Across 97 published cross-sectional predictors, portfolio returns are **26% lower out-of-sample** (after the sample ends, before publication) and **58% lower post-publication**.
   - The 26% is "an upper bound estimate of data mining effects".
   - The authors attribute **32% (58% − 26%)** to publication-informed trading.
   - Post-publication declines are larger for predictors with higher in-sample returns.
   - Returns are higher in portfolios concentrated in high-idiosyncratic-risk, low-liquidity stocks.
   - Correlations between predictor portfolios and other published-predictor portfolios rise after publication.
   - Interpretation: "investors learn about mispricing from academic publications."
   - Source: DOI 10.1111/jofi.12365 (abstract via OpenAlex; bibliographic data at https://ideas.repec.org/a/bla/jfinan/v71y2016i1p5-32.html)
   - Authors: R. David McLean, Jeffrey Pontiff
   - Pub date: *Journal of Finance* 71(1):5–32, Feb 2016 (online 2015)
   - Peer-reviewed: yes (JF)
   - Accessed: 2026-09-25 · Confidence: high · Class: decay
   - Corroboration: independently consistent with Chen & Velikov (claim 8), who treat post-publication effects as one of three return-reducing adjustments.

### Q3: Transaction costs

7. **Claim:** Transaction costs reduce the profitability and statistical significance of **all** anomaly strategies examined, which heightens data-snooping concerns.
   - Most strategies with monthly turnover **below 50%** still earn statistically significant net spreads, at least when designed to mitigate transaction costs.
   - "Few of the strategies with higher turnover do."
   - A buy/hold spread is "the single most effective simple cost mitigation strategy."
   - Source: https://www.nber.org/papers/w20721
   - Authors: Robert Novy-Marx, Mihail Velikov
   - Pub date: NBER WP Dec 2014; *Review of Financial Studies* 29(1):104–147, 2016
   - Peer-reviewed: yes (RFS)
   - Accessed: 2026-09-25 · Confidence: high (abstract) · Class: costs
   - Caveat: the abstract does not name which anomalies survive. Named survivors were **not retrieved**.

8. **Claim:** Chen & Velikov study long-short portfolios on 204 anomalies. They account for (i) effective bid-ask spreads, (ii) post-publication effects, and (iii) the modern trading-technology era that began in the early 2000s.
   - Net of all three, **the average anomaly's expected return is "a measly 4 bps per month."**
   - The strongest anomalies net "at best, 10 bps" after controlling for data mining.
   - Several methods of combining anomalies net about 20 bps.
   - These expected returns are negligible even though cost mitigations produce impressive net returns in-sample, and even though the estimates *omit* price impact. The true net figure is therefore an upper bound.
   - Source: DOI 10.1017/S0022109022000874 (abstract via OpenAlex; also https://www.federalreserve.gov/econres/feds/zeroing-in-on-the-expected-returns-of-anomalies.htm, FEDS 2020-039)
   - Authors: Andrew Y. Chen, Mihail Velikov
   - Pub date: *Journal of Financial and Quantitative Analysis* 58(3):968–1004, 2023 (online 2022)
   - Peer-reviewed: yes (JFQA)
   - Accessed: 2026-09-25 · Confidence: high · Class: costs / decay
   - **This is the single most decision-relevant number in the dimension.**

### Q4: Risk premia vs mispricing; long-run evidence

9. **Claim:** Value and momentum return premia are consistent across **eight diverse markets and asset classes**, with a strong common factor structure.
   - Value and momentum returns correlate more strongly across asset classes than passive exposures do.
   - Value and momentum are negatively correlated with each other, both within and across asset classes.
   - The authors infer "common global risks", modelled with a three-factor model.
   - **Global funding liquidity risk is a *partial* source.**
   - The findings "present a challenge to existing behavioral, institutional, and rational asset pricing theories."
   - Source: DOI 10.1111/jofi.12021 (abstract via OpenAlex; bibliographic data at https://ideas.repec.org/a/bla/jfinan/v68y2013i3p929-985.html)
   - Authors: Clifford S. Asness, Tobias J. Moskowitz, Lasse Heje Pedersen
   - Pub date: *Journal of Finance* 68(3):929–985, June 2013
   - Peer-reviewed: yes (JF)
   - Accessed: 2026-09-25 · Confidence: high (abstract) · Class: risk-vs-mispricing
   - Note: the authors lean toward a risk explanation, but only partially. The abstract explicitly leaves every theory challenged, so this is not a clean "compensation for risk" result.

10. **Claim:** Time-series momentum (trend-following) across global markets since 1880 delivered positive average returns **in every decade**, with low correlation to traditional asset classes. It performed well in **8 of the 10 largest crisis periods** (the largest drawdowns of a 60/40 stock/bond portfolio) and across macro regimes: recession and boom, war and peace, high and low rates, high and low inflation.
    - Source: DOI 10.3905/jpm.2017.44.1.015 (abstract via OpenAlex); https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing
    - Authors: Brian K. Hurst, Yao Hua Ooi, Lasse H. Pedersen
    - Pub date: *Journal of Portfolio Management* 44(1):15, 2017
    - Peer-reviewed: practitioner journal with editorial review. It is weaker than JF/RFS peer review. The authors are from AQR, which sells trend strategies, so there is a conflict of interest.
    - Accessed: 2026-09-25 · Confidence: medium (single industry-authored source; decade-level returns and cost assumptions not visible in the abstract) · Class: risk-vs-mispricing
    - Not established here: net-of-fee magnitudes, and performance after 2017. The AQR page summary did not cover either.

### Q5: Machine-learning return prediction

11. **Claim:** Investments based on deep-learning signals "extract profitability from **difficult-to-arbitrage stocks** and during **high limits-to-arbitrage market states**."
    - **Excluding microcaps, distressed stocks, or high-volatility episodes "considerably attenuates profitability."**
    - Performance "further deteriorates in the presence of reasonable trading costs" because of high turnover and extreme positions.
    - The ML methods identify mispriced stocks in ways consistent with most anomalies.
    - "Beyond economic restrictions," deep-learning signals remain profitable in long positions and in recent years, with low downside risk.
    - Source: DOI 10.1287/mnsc.2022.4449 (abstract via OpenAlex); https://ideas.repec.org/a/inm/ormnsc/v69y2023i5p2587-2619.html
    - Authors: Doron Avramov, Si Cheng, Lior Metzker
    - Pub date: *Management Science* 69(5):2587–2619, 2023 (online 2022)
    - Peer-reviewed: yes (Management Science)
    - Accessed: 2026-09-25 · Confidence: high (abstract) · Class: ML / costs
    - Caveat: I did **not** retrieve Gu, Kelly & Xiu (2020) itself (source cap), so this digest cannot state GKX's own reported gains. See "Looked for".

---

## Net reading for the decision (inference, traceable to claims 2, 6, 7, 8, 11)

- **Reproducing the in-sample statistic mostly succeeds** (claims 3–4). **Surviving as a net-of-cost, post-publication edge mostly fails.**
  - Average post-publication returns are 58% lower (claim 6).
  - The average anomaly nets about 4 bps/month after spreads, publication and the modern era, before price impact (claim 8).
  - High-turnover anomalies rarely survive costs (claim 7).
  - ML gains sit in hard-to-arbitrage stocks and shrink with costs (claim 11).
- Where the published evidence holds up across a century or across asset classes (claims 9–10), the authors themselves partly attribute it to **priced risk** (funding liquidity) and crisis behaviour. That makes it a risk premium to be *harvested*, with drawdowns, rather than a free edge. Even that attribution is partial and industry-sourced in the trend case.
- **Scope limit:** every source here studies monthly-rebalanced cross-sectional equity anomalies or multi-asset time-series factors. None studies intraday single-instrument futures strategies. Carrying these findings over to intraday futures is an analogy, not evidence.

---

## Leads (unverified; proposed from prior knowledge, not retrieved this run)

- Chordia, Subrahmanyam & Tong (2014), "Have capital market anomalies attenuated in the recent era of high liquidity and trading activity?", *Journal of Accounting and Economics*. Believed to find anomaly attenuation after decimalization. Would corroborate the "modern era" adjustment in claim 8.
- Green, Hand & Zhang (2017), "The characteristics that provide independent information about average U.S. monthly stock returns", RFS. Believed to find few independent characteristics after about 2003.
- Frazzini, Israel & Moskowitz, "Trading Costs of Asset Pricing Anomalies" (working paper) and "Trading Costs" (2018 WP). Uses live AQR execution data and believed to find *lower* costs than Novy-Marx & Velikov. This is the main counter-argument to claims 7–8 and should be retrieved next.
- Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning", RFS 33(5). This is the original ML result that claim 11 qualifies.
- Jacobs & Müller (2020), JFE, on post-publication decline outside the US, which is believed to be absent or smaller internationally. Tests how general claim 6 is.
- Linnainmaa & Roberts (2018), RFS, "The History of the Cross-Section of Stock Returns". Pre-sample (out-of-sample backward) tests.
- Chen & Zimmermann (2020), "Publication bias and the cross-section of stock returns", RAPS. Believed to estimate small publication-bias shrinkage, a counterweight to HLZ.
- Harvey & Liu (2020), "False (and Missed) Discoveries in Financial Economics", JF. Would supply the numeric false-discovery estimate that HLZ's abstract lacks.
- Post-2017 performance of trend and value: for example, the 2010s value drawdown and trend-following's weak 2010s. Needed to answer "recent performance/decay" for claim 10.

## Looked for, could not find (this run)

- **HLZ's numeric share of likely-false findings, and its factor count.** The abstract says only "most" and "hundreds". The SSRN full text returned 403.
- **JKP's exact replication rate** and the magnitude of out-of-sample decay. The abstract says only "majority".
- **Which named anomalies survive costs** in Novy-Marx & Velikov. The abstract gives only the <50% monthly turnover rule.
- **Frazzini/Israel/Moskowitz cost estimates.** Not searched: tool and source budget exhausted.
- **Gu/Kelly/Xiu's own performance and cost figures.** Not retrieved because of the 10-source cap.
- **Chordia/Subrahmanyam/Tong and Green/Hand/Zhang decay magnitudes.** Not retrieved because of budget.
- **Trend-following and value performance after 2017** from a primary source. Not found.
- **Full text of any paper.** Every claim rests on abstracts, so methods, sample periods and robustness tables were not checked.
