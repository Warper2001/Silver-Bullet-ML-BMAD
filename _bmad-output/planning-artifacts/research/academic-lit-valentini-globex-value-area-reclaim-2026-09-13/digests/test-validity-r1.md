# Test validity literature digest — round 1

Accessed: 2026-09-13. Scope: statistical design for a proposed intraday futures value-area reclaim, not evidence that the pattern earns returns. No project documents or market data were read. Eight distinct works admitted; five web calls. Confidence refers to the stated methodological claim, never to strategy profitability.

## Retrieved evidence

### TV1 — Sample Size Justification
- Author: Daniël Lakens. Publisher: University of California Press, *Collabra: Psychology* 8(1), 33267. Publication date: 2022-03-22. Class: peer-reviewed methodological review; primary author exposition, not original futures evidence. Confidence: high.
- URL: https://doi.org/10.1525/collabra.33267
- Evidence: sections “A-priori Power Analysis,” “Plot a Sensitivity Power Analysis,” and “Post-hoc Power Analysis.” Power is conditional on an assumed effect. With fixed available sample size, sensitivity analysis identifies detectable effects; the effect need not be estimated from the final test sample. Generic effect-size defaults and optimistic literature borrowing lack adequate justification. Observed post-hoc power adds no information beyond the p-value. Simulation supports designs without analytic solutions.
- Application: report MDE curves over available sessions and nuisance assumptions; do not label the design adequately powered for an unknown true edge. A practical effect threshold needs independent justification. No equity or cross-sectional effect estimate transfers to this rule.
- Limitation: this is general design guidance, not a trading-return model or futures effect estimate.

### TV2 — Panel Data and Experimental Design
- Authors: Fiona Burlig, Louis Preonas, Matt Woerman. Publisher: NBER Working Paper 26250. Publication date: 2019-09. Class: working paper as retrieved; later journal status not verified. Confidence: high for abstract-level methodological claim, moderate for transfer.
- URL: https://www.nber.org/papers/w26250 ; DOI https://doi.org/10.3386/w26250
- Evidence: publisher abstract analytically derives variance allowing arbitrary serial correlation; simulations and real panels show ex-ante power errors when that correlation is ignored. Appendix: https://back.nber.org/appendix/w26250/BPW_power_calculations_NBER_appx.pdf discusses simulation for cluster randomization and distinguishes it from the main analytic scope.
- Application: minute bars and repeated signals are not automatically independent replications. Simulate dependence at the relevant session/block level, rather than applying an iid trade-count gate.
- Limitation: panel randomized experiments differ from a single observational futures series; its exact panel formulas cannot simply be transplanted.

### TV3 — The Stationary Bootstrap
- Authors: Dimitris N. Politis, Joseph P. Romano. Publisher: Taylor & Francis / American Statistical Association, *JASA* 89(428), 1303–1313. Publication date: 1994-12 (online archive date 2012-02-27). Class: peer-reviewed original methods article. Confidence: high for scope; abstract retrieved, full publisher open failed.
- URL: https://doi.org/10.1080/01621459.1994.10476870
- Evidence: develops resampling for standard errors and confidence regions with weakly dependent stationary observations, extending consecutive-block approaches.
- Application (inference): aggregate outcomes within sessions, retain zero-trade sessions for a calendar-session estimand, and resample consecutive session blocks when across-session dependence matters. Sweep plausible block lengths on development data and report sensitivity.
- Limitation: stationary bootstrap does not repair structural breaks, selected histories, or too few informative blocks. Session independence is an assumption to examine, not an automatic consequence of aggregation.

### TV4 — Exact testing with random permutations
- Authors: Jesse Hemerik, Jelle Goeman. Publisher: Springer, *TEST* (publication record: December 2018); author manuscript on arXiv. Class: peer-reviewed original methods article; retrieved full manuscript. Confidence: high.
- URLs: https://arxiv.org/html/1411.7565 ; https://pure.eur.nl/en/publications/exact-testing-with-random-permutations/
- Evidence: sections 2–3 require an appropriate null invariance and transformation structure; arbitrary appealing subsets of permutations can be anti-conservative. Randomly sampled transformations need correct sampling and inclusion of the identity for the presented valid procedure.
- Application (inference): a timing/volume null must break signal–future-outcome alignment while preserving relevant dependence and the eligibility conditions. Shuffling completed trade P&Ls changes order but leaves mean P&L unchanged, so it cannot test whether entries predict outcomes. Random entries with different risk geometry test another hypothesis. Naive minute shuffles or circular shifts across Globex boundaries are not established valid here.
- Limitation: no theorem retrieved establishes exchangeability for this exact market setup; matched/block surrogates remain assumption-dependent diagnostics until their validity is demonstrated.

### TV5 — The Probability of Backtest Overfitting
- Authors: David H. Bailey, Jonathan M. Borwein, Marcos López de Prado, Qiji Jim Zhu. Publisher: Infopro Digital Risk, *Journal of Computational Finance* 20(4), 39–69. Publication date: 2017-04; retrieved author manuscript dated 2015-02-27. Class: peer-reviewed original methods article, with prepublication text retrieved. Confidence: high.
- URLs: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf ; publisher publication confirmation: https://www.risk.net/journal-of-computational-finance/volume-20-number-4-april-2017
- Evidence: develops combinatorially symmetric cross-validation to assess whether selection of an in-sample winner leads to out-of-sample underperformance. Repeated strategy selection makes ordinary backtest evidence vulnerable to overfitting.
- Application: preserve the full variant/research ledger; account for choices of value-area construction, volume pattern, timing, stop and exit. PBO is a search-process diagnostic, not proof of positive net expectation and not replacement for untouched forward evidence.
- Limitation: exact protocol profitability and effective number of variants are unknown.

### TV6 — A Reality Check for Data Snooping
- Author: Halbert White. Publisher: Wiley / Econometric Society, *Econometrica* 68(5), 1097–1126. Publication date: 2000-09 (archive online date 2003-12-10). Class: peer-reviewed original methods article. Confidence: high.
- URL: https://doi.org/10.1111/1468-0262.00152
- Evidence: reused data for inference/model selection create chance discoveries; proposes testing whether the best model encountered in a specification search has predictive superiority over a benchmark.
- Application: a later favorable test on repeatedly consulted data is not a fresh confirmation. Define the benchmark and searched family, or freeze the final rule and obtain unconsulted chronological data. Distinguish an exploratory screen from a confirmatory test.
- Limitation: an unrecorded historical search cannot be retroactively treated as one prespecified test.

### TV7 — Inference and missing data
- Author: Donald B. Rubin. Publisher: Oxford University Press / Biometrika Trust, *Biometrika* 63(3), 581–592. Publication date: 1976-12-01. Class: peer-reviewed original methods article. Confidence: high for theorem scope; moderate for application.
- URL: https://academic.oup.com/biomet/article-abstract/63/3/581/270932 ; DOI https://doi.org/10.1093/biomet/63.3.581
- Evidence: ignoring missingness is valid only under specified conditions, differing for sampling-distribution and likelihood/Bayesian inference. Missingness is not automatically ignorable.
- Application (inference): selected tick days cannot establish representative event frequency or an unconditional power gate. Record universe, coverage and selection reasons. Restrict claims to an explicitly defined observed population unless an adequate sampling/missingness model is justified. Bootstrap repetition cannot create information about omitted regimes.
- Limitation: this source does not identify the actual mechanism of missing tick data; that must be audited separately.

### TV8 — CME Liquidity Tool User Guide
- Author/publisher: CME Group. Publication date: not displayed in retrieved guide; accessed 2026-09-13. Class: official exchange methodology/documentation, not peer-reviewed research. Confidence: high for tool definitions; unknown for exact instrument/date availability.
- URL: https://www.cmegroup.com/education/demos-and-tutorials/cme-liquidity-tool-user-guide
- Evidence: tool reports bid–ask spread, book depth and cost to trade, with order-size and time-zone selections; cost to trade is in ticks for a specified lot size.
- Application (inference): define net outcomes using fees plus execution assumptions appropriate to instrument, time and order size. Quotes/depth or recorded fills can inform cost scenarios; the guide supplies no universal slippage constant. A profitable gross pattern does not itself establish profitable execution.
- Limitation: no MNQ-specific numerical cost estimate, stop-fill model, or current fee schedule retrieved. Stop/target intrabar ordering still requires adequate data.

## Decision implications — explicitly an analyst synthesis

No retrieved source supplies a defensible positive effect size for this exact Globex 70% developing-value-area reclaim with adjacent-minute volume sequence, excursion stop and VAH exit. Therefore do not insert an unrelated published alpha, a generic standardized effect, or the selected backtest mean into a power calculation and call it validated [TV1, TV5–TV6].

A defensible pre-test gate is conditional: freeze estimand, rule and costs; inventory prospectively eligible sessions; use a separate, unselected development sample to estimate event frequency and variability; then run the intended dependence-aware test in simulation under a grid of net effects, costs and plausible dependence. Report the MDE surface and nuisance uncertainty [TV1–TV3, TV7–TV8]. A practical detection target can be derived from externally justified economics, but no numeric target is established by this digest.

If relevant data are selectively acquired or too incomplete to estimate event incidence, nuisance variance or fills, label the gate NOT ESTIMABLE / DATA INADEQUATE; use UNDERPOWERED only when power can be estimated and falls short of a declared target. These labels are proposed decision terminology, not conclusions about actual project data [TV1, TV7]. Collect a contiguous prospective pilot to resolve those uncertainties, without using it later as untouched confirmation [TV6–TV7].

Use separate tests for (a) net deployable expectancy and (b) incremental timing/volume information. A surrogate that destroys pairing can address (b), subject to its invariance assumptions, while same-P&L reshuffling cannot [TV4]. Neither favorable surrogate ranking nor a gross edge alone establishes positive net returns [TV4, TV8].

## Gaps, queries and stopping record

- Gaps: exact strategy effect size; event rate on representative tick history; full coverage/missingness mechanism; stability of session dependence; valid conditional randomization scheme for deterministic volume signals; actual fills/fees; complete prior trial ledger. No numerical MDE can be produced from literature alone.
- Queries (round 1): `site.nber.org "Power Calculations" "Serial Correlation" experiment`; `site.davidhbailey.com probability backtest overfitting pdf`; `site.afajof.org "Evaluating Trading Strategies" transaction costs`; `Lakens 2022 sample size justification smallest effect size sensitivity power journal`; `Hemerik Goeman exact testing random permutations exchangeability 2018`; `Politis Romano stationary bootstrap 1994 original paper`; `site.cmegroup.com futures liquidity tool bid ask spread cost to trade historical data`; `Halbert White 2000 reality check data snooping Econometrica bootstrap`; `Donald Rubin 1976 inference missing data Biometrika missing at random`; publication checks for Burlig and Bailey.
- Rejected: blogs, Reddit discussions, generic effect-size defaults, and unrelated financial effect estimates. Search hits were discovery only unless included above. General statistical methods were transferred explicitly as design inferences, not as alpha evidence.
- Retrieval limitations: PMC Hemerik open hit browser check; author arXiv full manuscript was accessible. NBER direct open failed but publisher search abstract and NBER appendix were retrieved. Politis publisher direct open failed but publisher abstract was retrieved. No paywall was bypassed.
- Stop reason: first-round evidence covers each methodological issue with eight works; unresolved quantities require representative strategy-specific data or a declared decision utility, not another broad literature round. Five web calls used, below ten-call cap; no second round.
