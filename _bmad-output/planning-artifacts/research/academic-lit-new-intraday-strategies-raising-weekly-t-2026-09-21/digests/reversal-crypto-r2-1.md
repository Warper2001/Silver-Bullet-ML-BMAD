# Digest r2-1: overnight-intraday reversal (equity-index futures) and crypto session/day-of-week rules

Accessed 2026-09-21. Web-only research. Access notes: MDPI, SSRN, ScienceDirect and assets.super.so returned HTTP 403 (also via curl with browser UA); webReader returned a quota error. The DKW conference PDF was read in full (WebFetch saved the binary; text extracted with pdftotext). Everything about MDPI JRFM 19(9):692 and SSRN 6776934 below is from search-engine abstract snippets or a blog excerpt, NOT the full text.

---
## CANDIDATE 1: Overnight-intraday reversal (CO-OC), equity-index futures

### Findings
**(a) Later/published version.**
- The cicfconf.org PDF is "Conference Copy", "This version is incomplete", "This Version: November 2015", authors Della Corte, Kosowski, Wang (Imperial College). No journal version was found.
- A later SSRN paper "Overnight-Intraday Reversal Everywhere" (SSRN 2730304, dated 2016-12-31) lists authors Chun Liu, Yang Liu, Tianyu Wang, Guofu Zhou, Yingzi Zhu (changed author set; Della Corte/Kosowski appear on a separately-hosted copy of the same title). Search snippets say it reports Sharpe ratios "two to five times larger" than traditional reversal across asset classes. I could not open it (403), so I have NO final numbers and cannot confirm a journal publication. Status: preprint/working paper only.

**(b) Equity-index-futures-only results (from the Nov 2015 draft, Table 3 / Table 4).**
- Rule (Eq. 1 and Table 1): zero-investment, cross-sectional. Weight on asset i = -(1/N)(r_i - mean r) using the overnight (close-to-open) return of day t as signal; hold from that same day's open to close (open-to-close). Signal and entry both use the day-t open price.
- Universe: 5 equity-index futures on CME: DJIA, NASDAQ, NIKKEI 225, S&P400, S&P500. Sample July 1982 - Dec 2014 (not all contracts start 1982). Data: TickData daily open/close. The paper does not (in text I read) say which session defines "open/close" for a nearly-24h contract. Unverified.
- Full sample equity index: CO-OC mean 0.252 %/day, t=13.18 (Newey-West), st.dev 0.980, annualized Sharpe 4.078, skew 1.324, kurt 23.5. Comparison: CC-CC 0.087 %/day (t 5.20, SR 1.15); OO-OO 0.201 (SR 2.43); OC-OC 0.024 (t 1.67, SR 0.36).
- Subperiods (Table 4 Panel B): 1982-2006 CO-OC 0.166 %/day, t 7.39, SR 2.508; 2007-2014 0.364 %/day, t 12.32, SR 6.683. So no decay inside the sample, but the sample ends Dec 2014. Nothing post-2014 in the primary source.
- FOMC vs non-FOMC days (Table 10): equity-index CO-OC 0.21 vs 0.26 %/day, SR 3.06 vs 4.00.
- Net of costs: NOT net. The paper states results are gross; the CXO summary quotes the authors that costs of daily reformation "would dramatically reduce" performance (about 100% daily turnover; stock-sample statement). No futures cost analysis in the draft.
- Scale caveat (my inference from Eq. 1): weights are -(r_i - mean)/N, not normalized to $1 gross exposure, so "%/day" is in units of that weighting and is not a return on committed capital. A per-contract P&L in dollars cannot be read off these numbers.

**(c) Independent replication / contradiction.**
- Serial-correlation check inside DKW itself (Section 4, Fig. 2): in equity-index futures the first-lag overnight-return t-statistic is "slightly below 2" (in absolute terms), versus t=13 for the CO-OC portfolio; the paper says reversal in liquid futures "only lasts for one period". Read: the pooled regression evidence is much weaker than the strategy t-stat suggests (paper does not reconcile).
- Implementability test (Table 7, US STOCKS only, TAQ S&P500 constituents 2011-2014): CO-OC average 0.36 %/day (t 10.98) if trades fill one second after 9:30; 0.11 %/day (t 3.81) at 9:31; 0.04 %/day (t 1.99) at 9:45. With signals formed from 9:29 prices 0.34 %/day; from 9:25 0.32 %/day, again collapsing with later entry. The authors themselves flag that open price enters both formation and holding return, so it "may not be implementable in real-time". No equivalent futures delayed-entry test in the draft. This is the largest threat: most of the return is earned in the first minute after the open.
- Secondary (non-peer-reviewed website, QuantReturns): claims CO-OC on equity futures (ES, YM, NQ etc.) 2007-2025 CAGR 30.31%, Sharpe 2.09, max DD -19.93%; the site itself says execution at the open is essential and "small daily percentage moves ... even minor execution delays can erode most or all of the edge"; no live/post-publication evidence; costs not stated in the summary I retrieved. Low confidence (page content seen only through a summarizer).
- Abstract-level only (search snippet, unopened): Overnight returns of stock indexes: evidence from ETFs and futures (ScienceDirect S1059056016301563): overnight returns of S&P500/sector ETFs and most international index futures are positive and daytime returns negative over 1999-2014; overnight returns predict first-30-min returns negatively and last-30-min returns positively, nothing in between. Consistent with DKW's "first minute after open" concentration, and it also shows the always-long-overnight drift (see (e)).
- Not found: Nagel-style VIX conditioning at the index-futures level from an independent author. DKW Table 11 (in-house) regresses change in CO-OC return on lagged change in VIX and overnight VIX increment; for equity-index futures adjusted R2 rises from 0.048 to 0.086 with VIX-overnight; dispersion is used as the volatility proxy for futures. Cheema et al. "Overnight returns, daytime reversals ... China" (2022) and Dangl & Salbrechter "Overnight Reversal and the Asymmetric Reaction to News" (SSRN 4307675) exist as related work but I only saw titles.

**(d) Trades per week / structure.** Daily rebalanced, so ~5 rebalances/week across the whole 5-contract basket; each leg is long or short depending on that day's overnight return relative to the cross-sectional mean. Every day, all 5 legs are non-zero (dollar-neutral, long-short). For a micro-Nasdaq-only operator: the signal for MNQ is defined relative to the other indices (needs MES/MYM/M2K legs; the NIKKEI 225 contract in the sample has different trading hours, unverified effect). The paper does not evaluate a single-contract version. Round-trip count would be about 5/week per leg if traded; trades/week is not a constraint here, cost/latency is.

**(e) Is it the "always long overnight" artifact?** Partly ruled out by construction, not by test. Because the portfolio is cross-sectionally demeaned (Eq. 1), a uniform overnight drift cancels; and the OC-OC strategy (intraday-only) is ~0 for index futures (0.024 %/day, t 1.67). The DKW draft contains no explicit decomposition against an always-long-overnight benchmark. Separate abstract-level evidence (ScienceDirect snippet above) confirms a positive overnight/negative daytime drift exists in index instruments, so a naive single-index "buy after down overnight" rule would carry that drift, and the demeaned version would not. A residual concern (inference): the effect concentrates in the first minute after the open, which is the signature of opening-auction/stale-price and bid-ask effects, and the draft's own Table 7 shows most stock-side return vanishes with a 1-15 minute delay.

### Claims
| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| DKW draft is "incomplete", Nov 2015; 5 index futures, 1982-2014, CME/TickData | https://www.cicfconf.org/sites/default/files/paper_357.pdf | CICF conference / authors | 2015-11 | 2026-09-21 | high | primary, preprint |
| CO-OC eq-index futures 0.252 %/day, t 13.18, SR 4.08; 1982-2006 SR 2.51; 2007-2014 SR 6.68 | https://www.cicfconf.org/sites/default/files/paper_357.pdf (Tables 3, 4) | authors | 2015-11 | 2026-09-21 | high (as reported) | primary, preprint, gross of costs |
| Rule is cross-sectional demeaned reversal, signal and entry at same open price | same (Eq. 1, Table 1, Robustness IV) | authors | 2015-11 | 2026-09-21 | high | primary |
| US-stock delayed-entry: 0.36 -> 0.11 -> 0.04 %/day at 9:30/9:31/9:45 | same (Table 7 text) | authors | 2015-11 | 2026-09-21 | high | primary, preprint |
| Eq-index futures first-lag overnight t-stat slightly below 2 (Fig. 2 text) | same (Section 4) | authors | 2015-11 | 2026-09-21 | medium (text only, figure not viewed) | primary |
| Reported figures gross; daily reformation costs would "dramatically reduce" performance; snooping bias caveat | https://www.cxoadvisory.com/technical-trading/overnightintraday-return-reversal-trading/ | CXO Advisory | 2016-04-13 | 2026-09-21 | medium (summarizer output) | secondary |
| Later version titled "Overnight-Intraday Reversal Everywhere", authors Liu, Wang, Liu, Zhou, Zhu, 2016-12-31 | https://papers.ssrn.com/sol3/Delivery.cfm/2730304.pdf?abstractid=2730304 (search snippet only) | SSRN | 2016-12-31 | 2026-09-21 | low-medium | preprint, not opened |
| Website backtest eq-futures CO-OC 2007-2025 Sharpe 2.09, CAGR 30.31% | https://quantreturns.com/strategy-review/overnight-mean-reversion/ | QuantReturns | ~2025-09 | 2026-09-21 | low | secondary, unverified, cost basis unknown |
| Overnight index returns positive, daytime negative 1999-2014; overnight predicts first/last 30 min | https://www.sciencedirect.com/science/article/abs/pii/S1059056016301563 (snippet only) | ScienceDirect (journal not identified) | ~2016 | 2026-09-21 | low-medium | peer-reviewed abstract, not opened |

### Inputs a power test would need (mean, SD, trades/yr, cost)
- Mean: gross 0.252 %/day full sample; 0.364 %/day 2007-2014 (Table 3/4) but in units of a non-normalized zero-investment weighting. MISSING: per-dollar-of-gross-exposure return, per-contract dollar P&L, and any post-2014 number from a primary source.
- SD: 0.980 %/day (same units; annualized Sharpe 4.08 = 0.252/0.980 x sqrt(252), checks out).
- Trades/yr: ~252 rebalances/yr. MISSING: whether single-instrument (MNQ vs peers) version retains anything.
- Cost: MISSING for futures. Operator's $2.24 round trip is not in the paper. Not verified: notional-based bp equivalent (needs the current index level, which I did not retrieve). Slippage at the open is the binding cost; only the stock delayed-entry table speaks to it (edge fell by ~70% in one minute, ~90% by 15 minutes).
- Entry-timing: MISSING for futures (no delayed-entry table). Required before any power test.

### Contrary evidence found
- Own-paper contradictions: futures lag-1 regression t below 2 vs portfolio t 13; edge vanishes with delayed entry (stocks); authors flag same-price formation/entry and multiple-strategy snooping.
- Reported gross only; ~100% daily turnover.
- QuantReturns page says retail cannot reliably get open prices; no post-publication live evidence.
- Extreme Sharpe ratios (stocks 24, futures 4-6.7) far above anything tradable at scale are themselves a red flag for microstructure/timing artifacts (my judgment, not a sourced claim).

### What I could not find
- Any journal version and the final-version numbers (SSRN 2730304 blocked).
- Post-2014 results from a primary source; any post-2015 subperiod split.
- Futures-specific net-of-cost or delayed-entry results.
- Independent replication with the same rule on index futures by unaffiliated authors (only a website backtest).
- Independent Nagel-style VIX conditioning at the index level.
- Definition of the futures "open" and "close" (pit vs electronic session).

---
## CANDIDATE 2: Crypto (JRFM 19(9):692; SSRN 6776934)

### Findings
**MDPI JRFM 19(9):692 (2026), peer-reviewed journal (MDPI).**
- Full text NOT retrieved (403 to WebFetch and curl; webReader quota error). Only search-engine abstract-level text is available. From that: hourly Kraken prices 2016-2025; 25 ordered combinations of {cash, long, short, momentum, reversal} positions across two 12-hour sessions with different hourly boundaries; selected rules: Bitcoin Reversal/Reversal with 08:00 UTC daytime start; Ethereum Long/Reversal with 05:00 UTC start. Bitcoin: conditional reversal in both sessions; Ethereum: positive overnight drift plus daytime reversal. Selected rules have higher terminal wealth and better drawdown/Sharpe than buy-and-hold along the full-sample path, but "these realized differences are not statistically significant in paired bootstrap tests."
- Inference (from the abstract wording, not verified in full text): rules are picked from a grid (25 combinations x start hour) on the same 2016-2025 sample they are evaluated on, i.e. in-sample selection with no stated holdout. Confirming this needs the full text.
- Not obtainable: mean return per trade, costs, trade count, 2022-2025 subsample results.

**SSRN 6776934 (May 2026; preprint), via blog summary only.**
- The mlquants Substack excerpt says weekly patterns "collapse into a single hour", Sunday 23:00-00:00 UTC (7-8 PM ET); effects are localized, asset-specific, stronger among actively traded coins and largely absent among less active ones; meme coins amplified; daily-bar patterns attributed to aggregation bias. The excerpt gives NO effect size in bps, sample dates, t-stats, 2024-2026 survival, or costs. SSRN page blocked (403).
- Trade frequency: a Sunday 23:00 UTC rule is 1 trade/week/asset (~52/yr), below the 1-2/week target for a single asset unless several assets are pooled (pooling would be highly correlated crypto; inference).

**Related secondary evidence.** Wen, Bouri, Xu, Zhao (2022, N. Am. J. Econ. Finance 62): BTC high-frequency 2013-03-03 to 2020-05-31 plus ETH, LTC, XRP; intraday momentum and reversal predictability; timing strategy beats always-long/buy-and-hold; transaction costs not explicitly quantified (per RePEc summary). Sample ends 2020, so it predates the MDPI sample's 2022-2025 window.

### Claims
| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| Kraken hourly 2016-2025, 25 ordered combos; selected BTC Rev/Rev 08:00 UTC, ETH Long/Rev 05:00 UTC | https://www.mdpi.com/1911-8074/19/9/692 (search snippet) | MDPI JRFM | 2026 | 2026-09-21 | medium (abstract only) | peer-reviewed, abstract |
| Selected rules beat buy-and-hold on the path but differences not significant in paired bootstrap | same | MDPI JRFM | 2026 | 2026-09-21 | medium (abstract only) | peer-reviewed, abstract |
| Sunday 23:00-00:00 UTC concentration of weekly effect; active coins only; meme coins amplified | https://mlquants.substack.com/p/are-day-of-the-week-effects-in-cryptocurrencies | ML Quants (Substack) | ~2026-05/06 (not shown) | 2026-09-21 | low-medium | secondary summary of preprint |
| SSRN 6776934 is the underlying paper | same page (ID cited there) | SSRN | 2026-05 | 2026-09-21 | low | preprint, not opened |
| Intraday momentum + reversal in BTC 2013-2020; no explicit costs | https://ideas.repec.org/a/eee/ecofin/v62y2022ics1062940822000833.html | RePEc / NAJEF | 2022 | 2026-09-21 | medium | peer-reviewed abstract |

### Inputs a power test would need (mean, SD, trades/yr, cost)
- Mean per trade / per session: MISSING for both papers (MDPI full text and SSRN blocked).
- SD: MISSING. Would have to be computed from Kraken hourly data by the operator, not from these papers.
- Trades/yr: MDPI rules trade twice daily-sessions (up to ~730 session decisions/yr per asset, but position may be cash, so realized trades unknown); Sunday-hour rule ~52/yr/asset. Both counts are my arithmetic from the descriptions, not reported figures.
- Cost: operator's own ~10 bps round trip; the papers' assumed costs are MISSING. A one-hour Sunday effect would need a gross edge well above 10 bps per trade to matter; its size is unknown.

### Contrary evidence found
- The MDPI authors' own bootstrap does not distinguish the selected rules from buy-and-hold; rule selection appears to be over a 25-combination grid (multiple-testing exposure; inference).
- SSRN summary: effect is asset-specific, confined to one hour, absent in less active coins; this is the profile of a fragile / data-mined seasonal (my judgment).
- Wen et al. found patterns shift with jumps, FOMC, and COVID (regime dependence).

### What I could not find
- Full text of MDPI 692 (per-trade mean, costs, trade count, 2022-2025 subsample, selection protocol).
- Full text of SSRN 6776934 (bps effect, sample, 2024-2026 survival, costs, multiple-testing correction).
- Any out-of-sample or post-publication test of either.
