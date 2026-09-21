# Red-team digest r3-1 (accessed 2026-09-21)

Scope: web-only hunt for evidence that Conclusions 1-3 are wrong. Budget ~20 calls used. Several primary sources returned 403 (ScienceDirect, Wiley, SSRN); those are cited from search snippets only and flagged low confidence.

## Counterexamples that survive scrutiny

**None that pass all four tests (replication/OOS + net-of-cost + post-2020 survival + >=1-2 trades/week).** Three near-misses, each failing at least one test:

1. **Baltussen, Da, Lammers, Martens, "Hedging demand and market intraday momentum", JFE 142 (2021) 377-403 (peer-reviewed).** Last-30-min return predicted by rest-of-day return, 60+ futures (17 equity index, 16 bond, 21 commodity, 8 FX), Dec 1974-May 2020. Independent extension of Gao et al. (JFE 2018) to futures; pooled regressions are similar in 1974-1999 and 2000-2020 subsamples (read from the PDF). Trade rate ~1/day (~250/yr) - passes the rate test. Costs: the paper states "we do not consider transaction costs" in the main strategy table, but says S&P 500 futures yield a positive net Sharpe at a one-tick cost (no figure read). FAILS: sample ends May 2020 (no post-publication data); the authors themselves note the effect was weak in the last four months of the sample (Feb-May 2020). Strongest pre-2020 counterexample to "no replication"; not evidence of post-2020 survival.
2. **Zarattini, Aziz, Barbon, "Beat the Market" (SSRN 4824172 / SFI WP 24-97, May 2024; preprint, not peer-reviewed).** Intraday momentum with noise-area breakout on SPY, 2007-early 2024, net of costs per abstract: 19.6% ann., Sharpe 1.33. Independent-looking replication (GitHub, giovannibrusco/zarattini-2024-momentum-spy, single-author, unreviewed): SPY Jul 2020-Jul 2026 and ES May 2024-Jul 2026, ~0.4 bp/round-trip cost assumed for ES, full-period Sharpe 1.11, +2.6 bp/trade, beta ~0; but "edge compressed since 2025", recent Sharpe ~0 on both instruments; author verdict "not allocable today, not dismissible as dead". Pre-2025 partial OOS confirmation (2020-2024) then decay. FAILS survival test after 2025 and is not peer-reviewed; the 0.4 bp cost is optimistic vs. 1 tick (0.25 pt ES ~ 0.5 bp+ commissions).
3. **Quantitativo Substack (2025-01-16), ES/NQ intraday momentum reimplementation.** 2010-Jan 2025, costs $0.85 commission + $1.40 fees + 0.5 tick slippage, Sharpe 1.57 (after the author's own "improvements": 90-day vs 14-day lookback, leverage targeting), ~1,000+ trades/yr across portfolio, "2 negative years in 16". Author states no confirmed OOS - forward test only. REJECTED as practitioner, tuned, unaudited; kept only as a rate/cost data point.

## Counterexamples rejected and why

- **Rosa (2022), "Understanding intraday momentum strategies", J. Futures Markets 42:2218-2234 (peer-reviewed).** Abstract (via search snippet, page 403): OOS predictability of overnight return -> last half hour "disappears in the out-of-sample period"; regime-switching/threshold version does better. This SUPPORTS Conclusion 1 (post-publication decay), not a counterexample. Threshold variant is an in-paper fix, not independently confirmed.
- **Seeck, "Intraday Momentum in Spot FX and Currency Futures: ... Cost Barrier to Retail Exploitability", SSRN 7008318, July 2026 (preprint, single author).** Per search snippet only (page 403): London-Open 30-min signal significant on 5 of 6 instruments in IS 2012-2018 and OOS 2019-2024; 6J CME futures M1 data 2019-2024; the title itself names a cost barrier to retail exploitation. Statistical predictability, not net-positive trading; not read in full. Rejected on cost test as far as I can verify.
- **arXiv 2605.04004, MNQ OHLCV falsification study (preprint, v. Sept 15 2026).** 14 signal families, 947 days 5-min MNQ 2021-2025, walk-forward, 2.0-pt friction floor; none passed all five criteria; 11 of 14 had gross returns too small for friction. SUPPORTS Conclusion 1 for MNQ 2021-2025.
- **VIX-futures intraday momentum (ResearchGate 366146088) and Chinese index futures ITSM (ScienceDirect S1544612319304337):** seen only as titles/snippets; not opened; cannot assess costs or post-2020 behavior. Not counted.
- **Calendar anomalies in stock index futures (Carchano & Pardo, SSRN 1958587):** search snippet says TOTM is "the only calendar effect statistically and economically significant and persistent" with "net cumulative profit 27.5%" and mentions data "through Q3 2024" - but the 2024 extension appears to be from a different (practitioner) page I could not open; the SSRN paper is dated ~2011. Unverified; not counted.

## Conclusion 2 (EIA Wednesday crude intraday momentum)

- SSRN/Energy Journal record confirmed to exist (Indriawan, Lien, Wen, Xu; SSRN 3822093/3907324; RePEc ej44-5-xu). Search snippet: on EIA days the third half-hour predicts the last half-hour; on non-EIA days only the first half-hour does. I could not open the paper (RePEc 404, SSRN 403), so I could NOT verify or challenge t=1.88, "no costs", the USO sample, or "absent 2014-2016".
- Searched for independent replications on CL/MCL, post-2019 evidence, or per-trade net >1.6 bp: **found none.** Only practitioner blog posts on EIA 10:30 volatility (no rule, no stats). Conclusion 2 stands as "unchallenged by anything I could retrieve", not "confirmed".

## Conclusion 3 (TOTM, RSI(2)/IBS, pre-holiday in index futures)

- TOTM: a paperswithbacktest summary of Hensel-Ziemba-type/Lakonishok-style work states "TOTM effects for S&P 500 futures disappear after 1990" (original sample 1982-1999, published 2000) and lists ~12 trades/yr; its own 1990-2026 backtest shows 1.61% ann., Sharpe 0.25 (practitioner aggregator; strategy cost treatment not shown). Supports Conclusion 3 on decay and rate.
- Counter-signal: unverified claim of persistence through Q3 2024 (see above). No peer-reviewed post-2015 futures-specific TOTM evidence retrieved.
- RSI(2)/IBS/pre-holiday: not searched separately (budget); no evidence retrieved either way.

## Claims

| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| Last-30-min return predicted by rest-of-day return in 60+ futures, 1974-May 2020; similar in 1974-99 and 2000-20 subsamples | https://academicweb.nd.edu/~zda/intramom.pdf | Journal of Financial Economics (author copy) | 2021 | 2026-09-21 | high (PDF read) | peer-reviewed |
| Main strategy results exclude costs; S&P futures net Sharpe positive at 1-tick cost (no number given); effect weak Feb-May 2020 | https://academicweb.nd.edu/~zda/intramom.pdf | JFE (author copy) | 2021 | 2026-09-21 | high (PDF read) | peer-reviewed |
| Overnight-return -> last-half-hour predictability disappears out of sample | https://econpapers.repec.org/RePEc:wly:jfutmk:v:42:y:2022:i:12:p:2218-2234 (snippet via search) | J. Futures Markets | 2022 | 2026-09-21 | medium (abstract via search, page 403) | peer-reviewed |
| Zarattini-Aziz-Barbon SPY intraday momentum 2007-2024 net Sharpe 1.33, 19.6% ann. | https://ssrn.com/abstract=4824172 | SSRN / Swiss Finance Institute WP 24-97 | 2024-05 | 2026-09-21 | medium (search snippet) | preprint |
| Independent replication: Sharpe 1.11 Jul 2020-Jul 2026, +2.6 bp/trade, edge compressed since 2025, recent Sharpe ~0 | https://github.com/giovannibrusco/zarattini-2024-momentum-spy | GitHub (individual) | 2026 (undated) | 2026-09-21 | medium (single unreviewed replicator, fetched summary) | practitioner |
| ES/NQ intraday momentum 2010-2025 Sharpe 1.57 net, tuned, no OOS | https://www.quantitativo.com/p/intraday-momentum-for-es-and-nq | Quantitativo (Substack) | 2025-01-16 | 2026-09-21 | medium | practitioner |
| MNQ 5-min, 14 signal families 2021-2025, none pass; 11 lack gross edge over 2-pt friction | https://arxiv.org/abs/2605.04004 | arXiv | 2026-05-05 (rev. 2026-09-15) | 2026-09-21 | medium-high (abstract) | preprint |
| FX/6J London-Open 30-min intraday momentum significant IS/OOS 2019-2024; title flags cost barrier to retail | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7008318 | SSRN | 2026-07 | 2026-09-21 | low (snippet only) | preprint |
| EIA-day third-half-hour predicts last half-hour in crude (paper exists) | https://ideas.repec.org/a/aen/journl/ej44-5-xu.html (snippet via search) | Energy Journal / SSRN | 2023 | 2026-09-21 | low-medium (snippet only) | peer-reviewed |
| TOTM S&P futures effect disappears after 1990 (orig. sample 1982-1999) | https://paperswithbacktest.com/strategies/closing-the-question-on-the-continuation-of-turn-of-the-month-effects-evidence-from-the-s-p-index-futures-contract | Papers With Backtest (aggregator) | 2000 paper | 2026-09-21 | medium | practitioner (summary of peer-reviewed) |
| TOTM "persistent" through Q3 2024 in S&P futures | https://dx.doi.org/10.2139/ssrn.1958587 (snippet only) | SSRN | ~2011 / unclear | 2026-09-21 | low - unverified belief | preprint/unverified |

## What I searched and did not find

- Any peer-reviewed post-2020 out-of-sample confirmation, net of realistic costs, of an index-futures intraday momentum rule with weekly-or-higher trade rate: none. The closest (Baltussen 2021) stops at May 2020 and does not report a net-of-cost figure.
- Independent CL/MCL replication of the EIA-Wednesday rule, or any post-2019 evidence: none.
- Peer-reviewed post-2015 futures-specific TOTM, RSI(2)/IBS, or pre-holiday evidence: none retrieved.
- Not searched (budget): overnight/intraday decomposition, MOC-imbalance rules, VIX/bond lead-lag to equity index, order-flow imbalance from public data, FOMC/PEAD futures drift, 1-5 day index trend, BTC/ETH rules. Absence of results here reflects unsearched space, not a negative finding; Conclusion 1 is therefore only partly stress-tested.
- Full text unavailable (403/404): Rosa 2022, Seeck 2026, Indriawan et al. 2023, Baltussen main net-Sharpe table figure (grep only).
