# Cluster C digest (r1-1): short-horizon BTC/ETH strategies, literature screen

Accessed 2026-09-21. Web-only. ~19 tool calls, several sources 403'd (ScienceDirect, Wiley, MDPI, Springer, Reading PDF blocked by Anubis), so many papers were read only via abstracts, search snippets or secondary summaries. Where a number was not retrieved it is marked NOT RETRIEVED, not estimated.

**Bottom line:** no candidate reached the bar of "independent replication + net of 10 bps + recent-2y evidence + >=1-2 trades/week". Best-evidenced items are weak on cost or recency; the most cost-robust idea (multi-hour/1-day reversal-vs-momentum) has only one recent, non-significant paper.

## Candidates

### C1. Intraday time-series momentum (first half-hour -> last half-hour), Bitcoin
- Rule: sign of first-30-min return (session defined by volume, since BTC has no open/close) predicts last-30-min return; trade last 30 min in that direction. Strongest in high-volume/high-volatility sessions.
- Sample/venue: Shen, Urquhart, Wang, Financial Review 57(2) 2022 (peer-reviewed), pp 319-344. Sample period and exchange NOT RETRIEVED (full text blocked).
- Effect size: NOT RETRIEVED (abstract gives only "economic gains in market timing and asset allocation, especially in downturns").
- OOS/independent replication: none found. Same-family paper (Wen, Bouri, Xu, Zhao 2022, J. Int. Financial Markets/Fin. Analysis, ecofin v62) finds both momentum and reversal, sample Mar 2013-May 2020, pattern changes with large jumps, FOMC, liquidity, COVID -> regime dependence acknowledged by the authors. Not independent of the Shen data lineage on the question of recency.
- Recent 2y vs original: no evidence retrieved. Sample ends 2020 (Wen) or earlier than 2021 (Shen); pre-ETF, pre-2024.
- 10 bps sensitivity: a 30-min holding window earns tiny gross per trade; at 10 bps round trip this is very likely dead (inference, unverified, since effect size not retrieved). Also trades ~daily-per-session so trade count is fine but edge per trade is the problem.
- Extreme-day dependence: NOT RETRIEVED; Wen says jump periods change the sign, which suggests concentration.
- Long-only artifact: strategy is long/short, so not a bull artifact by construction; shorts require CME micros or Kraken shorts.
- Verdict: LOW credibility for this operator (no post-2020 data, cost-fragile, session definition is researcher-choice).

### C2. Hour-of-day seasonality (buy 21:00 UTC, sell 23:00 UTC), Bitcoin
- Rule: long BTC 21:00-23:00 UTC daily (Quantpedia summary of Bitcoin seasonality paper; the strategy variant also filters high-volatility regimes).
- Sample: Gemini hourly, 9 Oct 2015 - 30 Jun 2023. Quantpedia summary reports ~40.6% annualized, MDD -22.7%, "rough period in 2022 and 2023". Underlying academic paper ("The Seasonality of Bitcoin", ResearchGate/SSRN; ~Oct 2015-Feb 2022 sample per the search snippet) - peer-review status NOT VERIFIED.
- Effect size: ~0.06% avg over the 2h window per Quantpedia-derived estimate (i.e. ~6 bps gross per trade). Costs: not stated in the source. SD/win rate not retrieved. ~250-365 trades/yr (daily) - Quantpedia's "~730" is my reading of entry+exit and should be treated as ~365 round trips.
- OOS: in-sample-selected hours (multiple-testing across 24 hours x 7 days, no correction seen). Later-sample "rough patch" in 2022-23 is the only quasi-OOS signal.
- Independent replication: Quantpedia/Concretum/QuantifiedStrategies are re-implementations, not independent data; the arXiv-style "Bitcoin Never Sleeps" is a blog. Related independent-ish paper: SSRN 6776934 (May 2026) says weekly patterns collapse on hourly data and only Sunday 23:00-00:00 UTC survives -> the 22-23h "effect" may be a partial view of the same narrow bursts.
- 10 bps sensitivity: gross ~6 bps per trade < 10 bps round trip -> NEGATIVE net on Kraken taker; only viable at maker/CME cost levels. FAILS the operator's cost bar.
- Extreme-day dependence, long-only artifact: strategy is long-only, ~8% time in market; the paper claims risk-adjusted improvement vs buy-and-hold but the 2015-2023 window is dominated by BTC's bull run (bull-market drift alone gives ~ 0.5-1 bp/hour on average - unverified arithmetic).
- Verdict: LOW; fails costs.

### C3. Turn-of-the-candle effect (minutes 0/15/30/45)
- Source: PMC10015199 (peer-reviewed, PLoS/Sci-Rep-type venue; journal name not confirmed in retrieved text). Seven exchanges (Bitfinex, Bittrex, Binance, Gemini, Kucoin, Bitstamp, FTX) from inception to 31 Dec 2021.
- Rule: hold at candle turns; positive avg ~0.58 bps per minute (0.55-0.66 bps across exchanges in 2021), other minutes negative. Claims $5k start -> 74.18% net annual return vs B&H 60.27% after fees and spreads.
- OOS: persisted through Aug 2022 but "less pronounced". Effect first appeared mid/late 2020 (April 2020 Bittrex to Dec 2020 others) - i.e. a young, algo-driven micro-structure artifact; t-stat >9 in 2021.
- 10 bps sensitivity: 0.58 bps per minute cannot pay 10 bps round trip; the paper's net result depends on exchange fee tiers/maker-like execution and very high trade counts. Not implementable at Kraken taker cost. FAILS.
- Verdict: microstructure curiosity, not for this account.

### C4. CME gap fill (Friday close -> Sunday open)
- Claims: ~65-90% of gaps "eventually fill"; ~85% within two weeks per various trader blogs (Bitget, Phemex, etc.). No academic paper found; all sources are exchange-academy/marketing pages.
- Decisive fact: CME Bitcoin futures/options moved to near-continuous trading on Globex from 28-29 May 2026 (two-hour maintenance pause 03:00-05:00 UTC Saturday), which ends weekend gaps (CoinDesk, 28 May 2026). The signal no longer exists going forward; only ~3 legacy gaps stayed open (~$80k, $78.5k, just under $70k).
- "Eventually fills" is not a tradeable stat (unbounded holding, survivorship in the denominator, path risk through stops).
- Verdict: DEAD structurally. Do not pursue.

### C5. Weekend / Monday / day-of-week effects
- Evidence: earlier daily-data studies find a Monday effect in the first subperiod that disappears in the second (ScienceDirect refs surfaced by search: S0275531918307827, S1544612317307894; abstracts not retrieved). SSRN 6776934 (17 May 2026, via substack summary; preprint) argues daily sampling creates phantom seasonality; on hourly data only Sunday 23:00-00:00 UTC survives (US-hours re-entry). No effect size, costs or multiple-testing correction retrieved.
- Concretum Group (practitioner, gross of fees): long-short BTC intraday trend ensemble, 2018-2025, Sharpe ~1.6 vs 0.8 for vol-targeted long-only, effect stronger post mid-2020; net costs not specified, out-of-sample not addressed -> cannot use.
- Verdict: LOW; note the operator already trades Thursday shorts; the literature is unstable across sub-periods.

### C6. Intraday/overnight lagged momentum vs reversal on BTC and ETH (12-hour sessions) - best fit for the mandate
- Source: "On the Performance of Lagged Momentum and Reversal Strategies Across Daytime and Overnight Sessions in Bitcoin and Ethereum Cryptocurrencies", J. Risk Financial Manag. 19(9):692, MDPI, 2026 (peer-reviewed, MDPI; full text 403, only abstract/snippet read).
- Data: hourly Kraken, 2016-2025. Day split into 12-hour sessions; 25 ordered combinations of positions. Selected rules: Reversal/Reversal at 08:00 UTC for BTC; Long/Reversal at 05:00 UTC for ETH. BTC = conditional reversal in both sessions; ETH = positive overnight drift + daytime reversal.
- Effect size: NOT RETRIEVED. Result stated: selected rules give higher terminal wealth, better drawdown and Sharpe than buy-and-hold on the realized path BUT differences are NOT statistically significant in paired bootstrap.
- Trades: ~2 per day on the session structure (daily) -> >>1-2/wk.
- OOS: none stated; rules are picked from 25 combinations x start hours on the same sample -> heavy selection; "ETH Long/Reversal" includes a long drift leg (bull-market exposure).
- Costs: not retrieved. 12-hour holds make 10 bps a meaningful but not fatal cost if edge per trade is >20 bps (unknown).
- Verdict: MEDIUM interest as a hypothesis, LOW as evidence; authors themselves say not significant. This is a ~12h-hold, Kraken-native data set, which fits the venue.

### C7. Post large-move (1-day abnormal return) continuation/reversal
- Springer FMPM 2020 "Momentum effects in the cryptocurrency market after one-day abnormal returns" surfaced in search; page redirected (paywall/login), so NOTHING retrieved beyond the title. Wen et al. (C1) confirm conditioning on jumps changes predictability sign. Wiley 2025 "Trading Games: Beating Passive Strategies in the Bullish Crypto Market" (Palazzi, J Futures Markets) title suggests a bull-market benchmark caveat; page 403, not read.
- Verdict: unverified; a lead, not a candidate.

### C8. Volatility-compression/expansion breakouts
- Only found: arXiv 2602.11708 (Feb 2026, preprint) adaptive trend-following, Sharpe 2.41 (70/30 alloc), 40.5% ann, Jan 2022-Dec 2024, with 4 bps taker fee + slippage + funding; and a GitHub repo of momentum research. These are multi-day trend systems (overlap with the operator's closed "fixed TSMOM with vol scaling"), unreplicated, from a 3-year window that includes the 2023-24 bull run. Not read in detail (search-snippet only).
- Verdict: not a distinct candidate; skip.

## Claims

| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| First half-hour BTC return positively predicts last half-hour; stronger in high-volume/volatility sessions; gains especially in downturns | https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/ | Univ. Birmingham / Financial Review (Wiley) | 2022-05 | 2026-09-21 | high (abstract only) | quantitative (no effect size) |
| Intraday BTC predictability shows both momentum and reversal; sign changes with large jumps, FOMC, liquidity, COVID; sample Mar 2013-May 2020 | https://ideas.repec.org/a/eee/ecofin/v62y2022ics1062940822000833.html | RePEc / Elsevier (Wen, Bouri, Xu, Zhao) | 2022 | 2026-09-21 | high | decay/regime |
| Turn-of-candle: +0.58 bps/min at :00/:15/:30/:45; 7 exchanges to Dec 2021; persists to Aug 2022 "less pronounced"; emerged mid/late 2020 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/ | PMC (peer-reviewed) | 2023 | 2026-09-21 | med (secondary AI summary of page) | effect-size/decay |
| Long BTC 21:00-23:00 UTC, Gemini Oct 2015-Jun 2023, ~40.6% ann, MDD -22.7%, rough 2022-23, costs not stated | https://quantpedia.com/the-seasonality-of-bitcoin/ | Quantpedia (practitioner) | ~2023-24 | 2026-09-21 | med | quantitative |
| Hourly Gemini data show time-of-day effects Oct 2015-Feb 2022; strongest 22:00/23:00 UTC, Friday | https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin (via search snippet) | Quantpedia | n/a | 2026-09-21 | low-med (snippet) | quantitative |
| CME BTC futures/options trade near-continuously from 28-29 May 2026 (2h Saturday pause), ending weekend gaps | https://www.coindesk.com/markets/2026/05/28/bitcoin-s-famous-cme-gaps-are-about-to-disappear-though-three-remain-unresolved | CoinDesk | 2026-05-28 | 2026-09-21 | high | decay/structural |
| ~65-90% of CME gaps eventually fill (blog claims, no rigorous study, unbounded time) | https://phemex.com/academy/cme-futures-gap (search snippet) | Phemex Academy | 2026 | 2026-09-21 | low | quantitative |
| Day-of-week pattern collapses on hourly data; only Sunday 23:00-00:00 UTC burst remains (SSRN 6776934, preprint) | https://mlquants.substack.com/p/are-day-of-the-week-effects-in-cryptocurrencies | Substack summary of SSRN | 2026-05-17 | 2026-09-21 | med-low (secondary) | replication/decay |
| Monday effect present in first subperiod, disappears in second (daily data) | https://www.sciencedirect.com/science/article/abs/pii/S0275531918307827 (search snippet) | Elsevier | 2019 | 2026-09-21 | low (snippet) | decay |
| Kraken hourly 2016-2025: BTC reversal/reversal at 08:00 UTC, ETH long/reversal at 05:00 UTC; beats B&H on Sharpe/drawdown but NOT significantly in paired bootstrap | https://www.mdpi.com/1911-8074/19/9/692 | MDPI J. Risk Fin. Manag. | 2026 | 2026-09-21 | med (abstract via search snippet) | quantitative |
| Concretum: BTC intraday trend long-short ensemble Sharpe ~1.6 vs 0.8, 2018-2025, gross of fees | https://concretumgroup.com/seasonality-in-bitcoin-intraday-trend-trading/ | Concretum Group (practitioner) | 2025 | 2026-09-21 | low (gross of fees, no OOS) | quantitative |
| Adaptive crypto trend Sharpe 2.41, 2022-2024, 4 bps taker fee + slippage + funding modelled | https://arxiv.org/pdf/2602.11708 (search snippet) | arXiv preprint | 2026-02 | 2026-09-21 | low | quantitative |
| Technical trading rules' mean Sharpe fell from 0.66 in-sample to 0.06 OOS; do not survive modest costs (snippet, source paper not identified - may not be crypto) | https://arxiv.org/pdf/2602.10785 (search snippet) | arXiv preprint | 2026-02 | 2026-09-21 | low | decay |

## Leads worth chasing
1. MDPI JRFM 19(9):692 (2026), Kraken hourly BTC/ETH session strategies: obtain full text (via the operator's own browser); check per-trade mean/SD, costs, and whether hours 05/08 UTC hold in 2024-2026. Kraken-native data, ~daily trades, Kraken perpetuals-compatible. Highest relevance, but a non-significant result by the authors' own test.
2. SSRN 6776934 (May 2026): read the paper directly; the Sunday 23:00-00:00 UTC burst is a single narrow, mechanistic (US-trader re-entry) window, weekly frequency = ~52 trades/yr; check bps size vs 10 bps cost and whether it holds post-2024.
3. Springer FMPM 2020 one-day abnormal returns paper: read abstract elsewhere (Google Scholar/ideas.repec) for direction and holding period (multi-day holds are cost-tolerant).
4. Wen et al. 2022: FOMC and jump-conditioned momentum/reversal (SSRN 4135239) - the conditional structure could yield a low-frequency (event-based) rule; data end 2020 so needs fresh testing.
5. Palazzi 2025 J. Futures Markets "Trading Games": read for the bull-market benchmark handling.

## What I looked for and could not find
- Any independent replication (different authors, different data) of BTC intraday TSMOM (Shen et al.) or of hour-of-day seasonality.
- Any post-2023 / post-ETF (Jan 2024) out-of-sample test of intraday BTC/ETH seasonality or momentum at cost-aware level. The only recent-window evidence is a non-significant 2026 MDPI paper and a 2026 SSRN preprint that argues effects vanish under hourly sampling.
- Effect-size numbers (mean/SD/win rate per trade) for Shen et al., MDPI 2026, Wen et al.; full texts were blocked.
- Options-expiry (Deribit monthly) and ETF-flow timing academic evidence: not searched to depth given the call budget; nothing retrieved.
- Volatility compression-then-expansion breakout with rigorous BTC/ETH evidence: only unreplicated arXiv trend-following (overlaps closed cluster).
- Any academic study of CME gap fills (only marketing blogs), now moot because CME gaps ended 2026-05-29.
- Anything showing ETH-specific short-horizon edges apart from the MDPI paper.
