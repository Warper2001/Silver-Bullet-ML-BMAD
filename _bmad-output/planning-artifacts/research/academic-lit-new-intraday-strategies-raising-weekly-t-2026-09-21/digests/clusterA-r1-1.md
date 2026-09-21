# Cluster A digest, round 1: daily / multi-day equity-index signals

Accessed 2026-09-21. Web research only. About 15 tool calls. Only two sources were read in full (McConnell & Xu 2008; the Della Corte et al. draft). Everything else is a search snippet or a secondary page, and each such claim is labelled that way.

**Bottom line: no candidate in this cluster clears both bars, credible independent evidence and at least 1-2 trades per week.**
- The best-documented effect (turn of the month) trades about 12 times a year and shows decay in the one recent source I read.
- The IBS / RSI(2) family has only practitioner evidence and about 9-20 trades a year.
- The one candidate with roughly daily frequency (overnight-intraday reversal) is an unpublished 2015 draft that I read only to its abstract and introduction.

## Candidates

### C1. Turn-of-the-month (TOM) long (McConnell & Xu 2008, Financial Analysts Journal 64(2), peer-reviewed). Read in full.
1. **Rule.** Long from the close of the last trading day of the month (Day -1) through the close of Day +3 (the third trading day of the next month). Otherwise flat or in T-bills.
2. **Sample and market.** CRSP US value-weighted (VW) and equal-weighted (EW) indices. 1926-2005, split into 1926-86 and 1987-2005. Also 31 of 35 countries. This is cash equities, not futures.
3. **Effect size** (Table 1, panel A, 1987-2005, VW).
   - Mean 0.15% per day over the 4 TOM days, versus -0.00% on all other days (excluding days -10 to -2 and +4 to +10, per the table note).
   - A window is about 0.6% gross, which is about 7.2% a year at 12 windows.
   - Difference t = 3.78. 66% of TOM days are positive.
   - EW: 0.25% per day versus 0.05% on other days.
   - SD per trade: not reported by the authors. I did not compute it.
   - Trades per year: about 12.
4. **Out of sample or independent.**
   - The 1987-2005 result is out of sample relative to Lakonishok & Smidt (1988, 1897-1986), which the authors say found the same pattern (0.473% over four days).
   - The authors contradict Maberly & Waggoner (2000, S&P 500 futures 1982-99), who reported that the effect vanished after 1990.
   - Same authors and same data family, so this is not fully independent.
5. **Decay.**
   - Split of 1987-2005 into halves: the VW difference was 0.17 points (t=3.63) for 1987-mid-1996 and 0.14 points (t=2.00) for mid-1996-2005. It declined and is borderline in the second half.
   - The sample ends in 2005, so it says nothing about 2006 onward.
   - A secondary, non-peer-reviewed post (Quantseeker, 2025) says that for the classical window in US ETFs, "none of these differences are statistically significant" and the effect "largely disappeared in the past decade". This is one practitioner Substack. I did not verify its data or methods.
6. **Costs.** Not modelled in the paper. Quantseeker used 5 bps one-way and found US CAGR falls below buy-and-hold. Gross 0.6% per window against about 10 bps round-trip is not cost-bound in itself.
7. **Extreme days.** Dropping the top 1% most extreme observations left the 1926-2005 means and t-stats "nearly unchanged" (VW 0.16% versus 0.01%). This is a reported and useful robustness check.
8. **Long-only?** Yes. The effect is well framed against an always-long baseline (other days about 0), so it is timing, not bull-market beta. But it is a 12-a-year signal.

**Verdict.** Fails the frequency bar (about 0.25 trades a week). Decay evidence is mixed. It is not a candidate on its own. It could be an overlay or filter only if the operator accepted that it cannot be validated at the operator's trade counts.

### C2. Maberly & Waggoner (2000), "Closing the Question on the Continuation of TOM Effects: Evidence from the S&P 500 Index Futures Contract" (SSRN 244085)
1. Rule: TOM windows in S&P 500 futures, 1982-99.
2. The SSRN page returned 403. I know its content only through McConnell & Xu's description and a search snippet.
3. Reported claim: the effect disappeared after 1990.
4. It was contradicted by McConnell & Xu on CRSP through 2005. It is the only source in this run using futures directly, so the conflict is unresolved for the operator's instrument.
5. Verdict: no usable numbers. Treat as evidence that TOM's futures persistence is contested.

### C3. RSI(2) / IBS mean reversion on the S&P 500 or SPY (practitioner sources only)
1. **Rule.**
   - Connors RSI(2): long SPY when RSI(2) is very low, typically with a close above the 200-day moving average, and exit on an RSI or moving-average condition.
   - IBS = (close - low) / (high - low). Common version from a Quantified Strategies search snippet: IBS < 0.25 and RSI(21) < 45, go long at the close. This is a snippet only.
   - The Alvarez IBS page (read) is on S&P 500 stocks, not the index: RSI(2) < 2.5, above the 200-day moving average, 126-day return > 0, buy the next open, exit when RSI(2) > 50. IBS < 10 improved average P&L by 58% (34% of trades kept), and IBS < 25 by 21%. Win rate 71% at low IBS versus 57% at high IBS. No dates, costs, out-of-sample test or buy-and-hold comparison are given.
2. **Sample and market.**
   - Quantified Strategies (search snippet, not verified, page blocked by a bot check): 293 trades on the S&P 500 since 1993 (about 9 a year), 0.5% average per trade, 75% win rate, "a few big losing periods".
   - A Substack snippet claims parameters were optimized on 2010-2018 and tested on 2019-2024, with about 70% wins on longs. I did not read it.
3. **Effect size.** About 50 bps gross per trade, 75% wins (secondary). SD per trade is not available.
4. **Out of sample or independent.** None that I found. Every source is a practitioner or vendor site with a commercial interest (Quantified Strategies, Alvarez, Connors). No peer-reviewed test of RSI(2) or IBS on index futures surfaced.
5. **Decay.** A secondary snippet says it "continues to perform after 2010" but "annualised return is poor... would easily be beaten by buy and hold". I could not check this.
6. **Costs.** Not addressed in the pages I read.
7. **Extreme days.** Not reported. The "few big losing periods" wording suggests left-tail dependence, which fits a 75% win-rate profile.
8. **Long-only?** Yes. Time in market is low, so a long-bias-only baseline comparison matters. None found.

**Contrary academic evidence.** The Della Corte et al. draft (C4) states, citing Grossman & Miller (1988), that short-term reversal "is absent or trivial in futures market". That is a citation in an unpublished draft, not a test I read.

**Verdict.** Not credibly evidenced. The trade count (about 9-20 a year) is far below the operator's target.

### C4. Overnight-intraday reversal (Della Corte, Kosowski & Wang, "Market Closure and Short-Term Reversal", Nov 2015 draft, marked "incomplete, do not distribute"). Preprint. Read the abstract and introduction only (4 pages).
1. **Rule.** Daily long (short) positions in assets with low (high) past overnight returns. The abstract does not spell out how it is applied to index futures.
2. **Sample and market.** Abstract claims robustness to international stocks and to equity index, interest rate, commodity and currency futures. I did not see the sample dates or the equity-index-only results.
3. **Effect size.** Only "five times larger than a conventional short-term reversal strategy" for US stocks. No bps, SD or win rate seen.
4. **Out of sample or independent.** None. It is a 2015 draft. I could not confirm whether it was ever published.
5. **Decay.** Unknown. The paper itself says monthly reversal "failed to deliver positive return in recent decade" (its introduction).
6. **Costs.** Unknown. A daily strategy has about 250 round trips a year. One secondary snippet (hmaquant Substack, not verified) says even 1 bp per execution costs about 5% a year on a buy-close/sell-open loop.
7. **Extreme days.** Unknown.
8. **Long-only?** No, it is long-short. That helps against the always-long objection.

**Verdict.** The only candidate that meets the trade-frequency bar, but no usable numbers were retrieved. Needs a full read, and a check for a published version, before any further weight. The operator has already closed always-long overnight holds. This is the reversal variant, a different object, but it draws on the same overnight-versus-intraday decomposition.

### C5. Pre-holiday effect
- Only a title was retrieved: "Pre-holiday effects: International evidence on the decline and reversal of a stock market anomaly". Search results and snippets say calendar effects diminished from the late 1980s. I did not read the paper.
- About 8-9 holidays a year, so about 0.2 trades a week. It fails the frequency bar regardless.
- Verdict: dropped. Evidence shape is decay.

### C6. VIX-conditioned reversal
- Della Corte et al. cite Nagel (2012) as showing that short-term reversal returns are "highly predictable by the VIX index" (cross-sectional stock reversal, liquidity provision). I did not read Nagel.
- That is a stock cross-section result. Nothing found tests a VIX filter on index-level daily reversal.
- Verdict: lead only, not a candidate.

### C7. Overnight vs intraday return decomposition (long overnight)
- Search snippets say essentially all S&P 500 ETF return since 1993 sits in the overnight leg and intraday is about zero (hmaquant Substack, unverified). This is the "always-long overnight" position the operator has closed.
- A daily buy-close/sell-open loop is cost-sensitive (see C4). Excluded per instructions.

### C8. RSI(14) < 30 buy on SPY (CXO Advisory)
- Rule: buy when daily RSI(14) < 30 with a 126-day hold, SPY 1993-2023, frictions ignored.
- Results are paywalled, so nothing usable. The hold length is also far outside the 1-5 day mandate. Listed only to show the search was done.

## Claims

| claim | source URL | publisher | pub_date | accessed=2026-09-21 | confidence | class |
|---|---|---|---|---|---|---|
| US VW TOM (Day -1..+3) mean 0.15%/day vs -0.00% other days, 1987-2005, diff t=3.78 | https://business.purdue.edu/faculty/mcconnell/publications/Equity-Returns-at-the-Turn-of-the-Month.pdf | Financial Analysts Journal (CFA Institute) | 2008-03 | accessed 2026-09-21 | high | quantitative/effect-size |
| TOM diff. falls from 0.17 pts (t=3.63, 1987-mid-96) to 0.14 pts (t=2.00, mid-96-2005) | same | FAJ | 2008-03 | accessed 2026-09-21 | high | decay |
| Dropping top 1% extreme returns leaves TOM means and t-stats nearly unchanged (1926-2005) | same | FAJ | 2008-03 | accessed 2026-09-21 | high | quantitative |
| McConnell & Xu reject Maberly & Waggoner's "disappeared after 1990" for 1987-2005 | same | FAJ | 2008-03 | accessed 2026-09-21 | high | replication |
| Maberly & Waggoner (2000) claim TOM in S&P futures vanished after 1990 (abstract page 403-blocked; known via search snippet and McConnell & Xu) | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=244085 | SSRN | 2000 | accessed 2026-09-21 | low | decay |
| Classical TOM in 40+ ETFs: not significant for most US equities, largely gone in the past decade; ~12 trades/yr; 5 bps one-way costs | https://www.quantseeker.com/p/turn-of-the-month-strategies-do-they | Quantseeker (Substack) | unknown (2025 per content, unverified) | accessed 2026-09-21 | low-med | decay |
| Quantpedia: TOM buy SPY 1 day before month-end, sell day 3; 7.2%/yr, Sharpe 1.04, 1926-2005 (ends before the decay period) | https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes | Quantpedia | undated | accessed 2026-09-21 | med | quantitative |
| IBS on S&P 500 stocks (not index) with RSI(2)<2.5: IBS<10 raises avg P&L 58%; win rates 71% to 57%; no dates, costs, or OOS | https://alvarezquanttrading.com/blog/internal-bar-strength-for-mean-reversion/ | Alvarez Quant Trading | undated | accessed 2026-09-21 | med (for what it says) | effect-size |
| Connors RSI(2) on S&P 500: 293 trades since 1993, 0.5%/trade, 75% win (snippet; page blocked) | https://www.quantifiedstrategies.com/rsi-2-strategy/ | QuantifiedStrategies | undated | accessed 2026-09-21 | low | quantitative |
| IBS+RSI optimized 2010-18, tested 2019-24, ~70% long win rate (search snippet only, not read) | https://quantifiedstrategies.substack.com/p/s-and-p-500-mean-reversion-using-19a | QuantifiedStrategies Substack | ~2024 | accessed 2026-09-21 | low | replication |
| Short-term reversal "absent or trivial in futures market" (citing Grossman & Miller 1988); monthly reversal failed in recent decade | https://www.cicfconf.org/sites/default/files/paper_357.pdf | Della Corte, Kosowski, Wang (draft) | 2015-11 | accessed 2026-09-21 | low-med | decay |
| Overnight-intraday reversal earns 5x a conventional short-term reversal in US stocks; robust to index/rate/commodity/FX futures (abstract; no numbers) | https://www.cicfconf.org/sites/default/files/paper_357.pdf | Della Corte, Kosowski, Wang (preprint, "incomplete") | 2015-11 | accessed 2026-09-21 | low-med | effect-size |
| Short-term reversal profit predictable by VIX (Nagel 2012, as cited in the Della Corte draft) | https://www.cicfconf.org/sites/default/files/paper_357.pdf | Della Corte et al. (citing Nagel) | 2015-11 | accessed 2026-09-21 | low | quantitative |
| Overnight leg holds nearly all S&P ETF return since 1993; buy-close/sell-open ~504 trades/yr, 1 bp costs ~5%/yr (search snippet) | https://hmaquant.substack.com/p/overnight-vs-intraday-returns-the | hmaquant (Substack) | unknown | accessed 2026-09-21 | low | quantitative/decay |
| Pre-holiday effect has "decline and reversal" internationally (title only) | https://www.researchgate.net/publication/222245676_Pre-holiday_effects_International_evidence_on_the_decline_and_reversal_of_a_stock_market_anomaly | ResearchGate (paper listing) | unknown | accessed 2026-09-21 | low | decay |

## Leads worth chasing

1. **Full read of the Della Corte, Kosowski & Wang paper** (check for a published version). The questions: what are the equity-index-futures-only results, the sample dates, the net Sharpe after costs, and the post-2010 subperiod? It is the only frequency-compatible candidate found.
2. **Post-2005 TOM tests on S&P/ES futures with a futures-specific data cut**, since the futures literature (Maberly-Waggoner) conflicts with the CRSP literature. A power calculation should come first: about 12 events a year at about 0.6% gross with index-level daily volatility will make a two-year test weak. Given the operator's trade-count constraint, a first pass would be a power-gate of TOM as a filter or overlay on existing strategies rather than a standalone strategy.
3. **Nagel (2012) "Evaporating Liquidity"** for the VIX-conditioned reversal, and whether anyone has tested it at index level. Not read.
4. **Peer-reviewed data-snooping treatment of technical rules on index data.** Sullivan-Timmermann-White (1999) and Hsu-Taylor-Wang (2016) are known to me from memory only. Neither was retrieved this run, so they are unverified beliefs.

## What I looked for and could not find

- **A peer-reviewed or independently replicated test of IBS or RSI(2) on index futures with post-2015 data.** Only vendor and blog material surfaced. The one out-of-sample claim (2010-18 in-sample, 2019-24 out-of-sample) is a search snippet from a vendor Substack, unread.
- **Any source giving SD per trade, or ex-top-N-days results, for IBS/RSI(2)/pre-holiday rules.**
- **Post-2020 evidence for TOM in futures.** The only post-2015 evidence is one Substack across ETFs.
- **Full text of Maberly & Waggoner, the Pre-holiday decline paper, and the CXO RSI(14) results.** Blocked or paywalled.
- **Trades-per-week fit.** Nothing rule-based and evidenced sits in the 1-2 trades per week zone. TOM about 12 a year, RSI(2) about 9 a year, pre-holiday about 9 a year. The only daily-frequency candidate is the unread overnight-intraday reversal.
- The sources are mostly older and only partly checked. No conclusion should be drawn about current (2025-2026) decay from this run.
