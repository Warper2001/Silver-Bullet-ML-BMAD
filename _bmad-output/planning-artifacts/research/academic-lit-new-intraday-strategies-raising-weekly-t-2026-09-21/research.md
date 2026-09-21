---
title: 'academic-lit research: two new strategies to raise weekly trade count with a credible edge'
type: 'academic-lit'
topic: 'two new strategies to raise weekly trade count with a credible edge'
decision: 'Pick 2 new candidate strategies (plus runner-up) that best fit the requirements frame; next step is power gate, not a seal'
source: 'native run'
status: complete
claims_verified: 6
claims_unverified: 4
claims_overturned: 0
preset: 'standard'
validation: 'high'
created: '2026-09-21'
updated: '2026-09-21'
---

# academic-lit research: two new strategies to raise weekly trade count with a credible edge

**Decision this research serves:** Pick 2 new candidate strategies (plus a runner-up) that best fit the requirements frame; the next step for any pick is an outcome-blind power gate, not a seal.

## Executive summary

**No candidate found clears the requirements frame.** Six families were screened (daily index reversal, weekly scheduled events in non-index futures, short-horizon crypto) and three were read in depth from primary text. None combines external out-of-sample or independent evidence (G1), positive net-of-cost results (G3), power-feasibility within about two years (G4), and at least 1–2 trades per week (G6). Trade frequency is not what limits these ideas. Per-trade effect divided by per-trade SD (d) is about 0.03–0.08 in every case where it could be computed, so detection needs decades of trades whatever the weekly rate.

The two least-bad candidates are below. Neither is a recommended strategy. They are chosen as the cheapest to test decisively, and the matrix totals (Section 4) rank failures, not passes:

1. **Crude-oil EIA-Wednesday intraday momentum on MCL** (Wen et al., Energy Journal 2023). It trades about 0.85 times a week and is orthogonal to the MNQ book (matrix 57, the highest of the finalists). But the paper trades a USO ETF, has t = 1.88, no costs and no out-of-sample test, and by my arithmetic needs about 1,000 trades (≈23 years) for 80% power at its own gross effect.
2. **Overnight-intraday reversal on index futures** (Della Corte–Kosowski–Wang draft; matrix 53). It trades about 5 times a week. It is the only candidate whose failure mode could be tested on data already held: delayed-entry sensitivity. The source result is gross, unpublished, and built on a same-open entry the authors' own stock test shows collapses within a minute. Correlation with your live opening-gap fade is probably high (my inference, untested).

The crypto session rules (matrix 54) are not picked: their full text was unreadable and the authors' own bootstrap finds no significant edge.

**Runner-up / cut:** the FX fixing pattern (Krohn–Mueller–Whelan) is cut on G3. The authors' own Table IX shows only EUR in the Europe window is positive at full quoted spread, at +1.5%/yr (Sharpe 0.05). Everything else needs an assumed 50% spread cut.

**Red team:** a skeptic hunting for counterexamples found none that pass all four tests. Its best near-misses are the noise-band intraday-momentum family your live MIM-NB bot already runs, and the only replication found reports that edge compressing to Sharpe ~0 since 2025 (one unreviewed source; see Section 6). It did not search most other families, so the negative is only partly stress-tested.

**Biggest caveat:** several full texts were blocked (JoF, MDPI, SSRN, ScienceDirect), so the crypto verdict rests on abstracts. The MDPI Kraken paper's own bootstrap finds no significant difference from buy-and-hold.

**Recommendation:** pre-register nothing from this recon. If you want to keep going, run a two-day outcome-blind cost and power gate on the two candidates above. The most likely result is UNDERPOWERED on both, which is a valid verdict that would close both cheaply.

## 1. Screen: what the frame removed

Hard gates and status for each family screened (round 1, three assistants, digests in `digests/`). "Pass/fail" is against the frame the plan gate agreed.

| candidate | G1 external OOS / replication | G3 cost survives | G4 power ≤ ~2y | G6 ≥1–2 trades/wk | cut or kept |
|---|---|---|---|---|---|
| Turn-of-month long, index | mixed: FAJ 1987–2005 t=3.78, but second half t=2.00 [1]; a 2025 Substack finds it gone in ETFs [2] | not modelled [1] | fail (~12 trades/yr) | **fail** (~0.25/wk) | cut on G6 |
| RSI(2) / IBS index reversion | practitioner-only, no OOS [3] | not addressed | fail (~9 trades/yr) | **fail** | cut |
| Overnight-intraday reversal, index futures | preprint draft; no journal version found [4] | **unknown**: gross only [4] | unknown | pass (~5/wk) | **kept (finalist)** |
| EIA-Wednesday crude intraday momentum | ETF, in-sample, t=1.88 [5] | **unknown**: no costs; MCL spread not retrieved | **fail** (~23y, derived) | pass (~0.85/wk) | **kept (finalist)** |
| FX fixing "W-shape" | JoF 2024; sample ends 2019; no replication found [7] | **fail** at full spread (EUR +1.5%/yr, Sharpe 0.05) [7] | fail | pass (~daily) | cut on G3 |
| Crypto 12h session rules (Kraken) | authors' own bootstrap: not significant vs buy-and-hold [10] | unknown | unknown | pass | cut (abstract only) |
| Crypto hour-of-day 21–23 UTC | in-sample-selected hours [11] | **fail** (~6 bp gross vs ~10 bp cost) [11] | – | pass | cut |
| Crypto turn-of-candle | young microstructure effect, persisted to 2022 "less pronounced" [13] | **fail** (0.58 bp/min) [13] | – | pass | cut |
| CME weekend-gap fill | blogs only | – | – | – | **dead structurally**: CME BTC trades near-24/7 from 2026-05-28 [12] |

Not carried forward, and the reasons are in the digests: pre-holiday (decaying, ~9/yr), gold PM fix (post-2014 reform, no significant returns after publication), commodity-ETF half-hour momentum (OOS R² 0.2–0.4% [6], no costs), Treasury auction pressure (magnitude unread, full-size contracts too large for a $2K trailing limit).

## 2. Finalists: what the primary text actually says

### 2.1 EIA-Wednesday crude intraday momentum: Wen, Indriawan, Lien, Xu, *The Energy Journal* 44(5), 2023 [5]

Read from the accepted manuscript (Adelaide/CORE), not an abstract.

- **The instrument is the USO ETF, not futures.** It uses 1-minute data from 2006-04-10 to 2019-07-31 and 591 Wednesday-10:30 ET EIA days. The signal is the 10:30–11:00 return, and the trade is long or short USO from 15:30 to 16:00 [5]. Confidence: high.
- **Regression:** the EIA-day coefficient is 0.038–0.042, with adjusted R² of 3.1% and 8.15%. **No out-of-sample test exists** [5].
- **Trading rule:** t = 1.88, significant only at the 10% level, with no costs charged. The printed "Sharpe 19.54" divides an annualised mean by a daily SD. I decoded this and it reproduces the paper's own t-statistics [5]. Per trade this is about +1.6 bp with an SD of about 21 bp (derived), about 0.5 event-only gross Sharpe per year.
- **Decay and concentration:** the effect is absent in 2014–16 and trends down. It holds only in the high-volatility half of days (N=347) and the first half of the month. Windows shorter than 15 minutes fail (Table 7). There is no top-days-removed test, and the rule's kurtosis is 10 [5].
- **Independent replication:** none found. The only other same-topic work is from overlapping authors, and three citing papers exist that were not read [5].
- **Confidence:** paper facts are high. The per-trade figures are derived and medium.

### 2.2 FX fixing "W-shape": Krohn, Mueller, Whelan, *Journal of Finance* 79(1), 2024 [7]

Read from the 2021 Bank of Canada working paper (JoF page returned 403), so final-version numbers are unverified.

- **Sample and costs:** 1999–2019, G9 vs USD, indicative quotes. At full quoted spread only EUR in the Europe window is positive (+1.52%/yr, Sharpe 0.05); Sharpe 0.12–1.03 needs an assumed 50% spread cut. The authors conclude returns in small windows are "usually more than offset by transaction costs" [7].
- **Decay:** the 2014–19 dollar-portfolio effect is about a third to a half of earlier levels, and the post-London leg is insignificant (t 1.62). The authors cite anecdotes that arbitrage capital now trades it [7].
- **Futures mapping:** CME futures show the pattern gross only, and the post-London effect is about a third of the spot figure. No spread, hours or volume for M6E at the relevant times was retrieved [7]. The "Sharpe 0.5–0.7" seen in round 1 comes from the superseded 2019 draft, not the 2021 paper [7].
- **Independent replication:** none found.

### 2.3 Overnight-intraday reversal: Della Corte, Kosowski, Wang, Nov 2015 draft [4]

Read in full. It is marked "incomplete", and no journal version was found. A later SSRN version by a different author set was blocked (403).

- **Rule:** cross-sectional and dollar-neutral across 5 CME index futures, 1982–2014. Signal and entry both use the same day's open price [4].
- **Reported result:** gross, 0.252%/day, t = 13.2, Sharpe 4.08. The 2007–14 subperiod is stronger (0.364%/day). Nothing after 2014 was tested [4].
- **Threats, all from the paper itself:**
  - In its US-stock test, delaying entry by one minute cut the return from 0.36 to 0.11%/day and by 15 minutes to 0.04%/day.
  - The index-futures lag-1 overnight-return t-statistic is below 2.
  - There is no futures delayed-entry test and no futures cost test [4].
- **Independent replication:** only a website backtest (low confidence, unverified costs) [16].
- **Always-long-overnight artifact:** the cross-sectional demeaning cancels a uniform drift, but the draft never tests this directly [4].

### 2.4 Crypto (abstract-level only)

The MDPI *JRFM* 19(9):692 (2026) paper on Kraken hourly data, 2016–2025, picks its best rules from 25 combinations on one sample. Its own paired bootstrap finds no significant gain over buy-and-hold [10]. The Sunday 23:00–00:00 UTC burst (SSRN 6776934) was seen through a Substack summary and gave no effect size [15]. Neither was read in full. The older BTC intraday-momentum literature (first half-hour predicts last) has no effect size retrieved and no post-2020 data, and its own authors find the sign changes with jumps, FOMC days and liquidity [8][9]. Confidence: low.

## 3. Cost and capacity

- **MCL:** 100 barrels, $1.00 per tick, cash-settled, about 23 hours a day [14]. Confidence: medium (CME spec page seen only in search snippets). At the derived 1.6 bp gross mean, the edge is about one tick at an assumed $60 price (mine, unsourced) against a per-trade SD of about $13. **Spread, commission and slippage for MCL in the 15:30–16:00 ET window were not retrieved.** Minimum plausible costs look about the size of the gross edge.
- **M6E:** €12,500, $1.25 tick [7]. Confidence: medium (CME page fetch timed out). The gross EUR Europe edge is about 6 bp per day (derived), and costs at quoted spread consume nearly all of it.
- **Index futures (overnight reversal):** micro-Nasdaq costs about $2.24 per round trip according to your own cost card. That is about 0.5 bp of notional (derived, index level not retrieved). Cost is small, but execution at the open is the binding constraint, and the authors' own delayed-entry result shows the effect concentrating in the first minute [4].

## 4. Weighted matrix (re-weightable)

Scores are 0–5, weights from the plan gate (evidence 30, trades/week 25, power 20, orthogonality 15, effort 10). **Scores are my judgment from the digests, and the totals rank failures, not passes.** All six fail at least one hard gate.

| candidate | evidence (30) | trades/wk (25) | power (20) | orthogonality (15) | effort (10) | total /100 |
|---|---|---|---|---|---|---|
| Overnight-intraday reversal | 1 → 6 | 5 → 25 | 2 → 8 | 2 → 6 | 4 → 8 | **53** |
| EIA crude momentum | 2 → 12 | 4 → 20 | 1 → 4 | 5 → 15 | 3 → 6 | **57** |
| FX fix (EUR, Europe window) | 2 → 12 | 5 → 25 | 1 → 4 | 5 → 15 | 3 → 6 | 62 |
| Turn-of-month | 3 → 18 | 0 → 0 | 1 → 4 | 3 → 9 | 5 → 10 | 41 |
| Crypto 12h session rules | 1 → 6 | 5 → 25 | 2 → 8 | 3 → 9 | 3 → 6 | 54 |

Two scoring notes. First, the FX fix scores highest on raw points, but it was cut on G3 before the matrix was applied. A hard gate outranks the weighted total. Second, orthogonality for the overnight reversal is scored low on project-context reasoning, not evidence: it is the same overnight-move-then-open-reversal family as your live opening-gap fade, so correlation is likely high. That was not tested.

**Why the picks are not simply the two top totals:** the requirement was "best probability of an edge that adds trades per week", and the least-bad ranking among failures should follow which failure is cheapest to close. EIA crude has the best orthogonality and a mechanical direction rule. The overnight reversal has the only untested failure mode (delayed entry) that your existing data can answer. Both are low-probability candidates.

## 5. Cross-dimension insights

- **The wall is power, not trade frequency.** For 80% power at a one-sided 5% test, N ≈ (2.49/d)². The EIA rule's d ≈ 0.078 gives about 1,020 trades (≈23 years at 44/yr). The FX fix at ~1–3 bp/day gross against a ~46 bp daily SD gives d ≈ 0.03–0.07, so multi-decade samples. A rule at d = 0.25 would be detectable in about 100 trades, or roughly one year at 2 trades a week. Nothing found in the literature was that large after costs. (Arithmetic is mine and outcome-blind. It uses the papers' in-sample effects, which are upward-biased by selection.)
- **The strongest-looking headline numbers were the least tradable.** The Sharpe 4.08 overnight reversal depends on entry at the same open used to form the signal. The "Sharpe 19.54" EIA figure is a units artifact. The FX "Sharpe 0.5–0.7" comes from a superseded draft. In each case the primary text quietly undercuts the headline.
- **The strongest independent evidence found belongs to a family you already run.** The best-replicated result in the whole search (Baltussen 2021 across 60+ futures) is intraday momentum, the MIM-NB mechanism, not a new source of trades [17]. New independent edges are scarcer than the frame assumed.
- **Decay is the norm.** EIA is absent 2014–16 and trending down. The FX effect fell to about half. Turn-of-month weakened across its own sample halves. The CME weekend gap ended structurally on 2026-05-28.

## 6. Contrary evidence (red-team pass)

A fresh-context skeptic was given the three main conclusions and no supporting evidence, and asked to find counterexamples (about 26 tool calls). **Result: no counterexample passes all four tests** (independent replication or out-of-sample, net of cost, survival after ~2020, at least 1–2 trades per week). Three near-misses each fail one:

- **Baltussen, Da, Lammers, Martens, *JFE* 2021** [17]: last-30-minute return predicted by the rest of the day across 60+ futures, 1974–May 2020, with similar pooled results in 1974–99 and 2000–20. About 1 trade a day. The main table excludes costs; a positive net Sharpe on S&P futures at a one-tick cost is stated but no figure was read. **Fails survival:** the sample ends May 2020, and the effect was weak in Feb–May 2020. Confidence: high (PDF read).
- **Zarattini–Aziz–Barbon "Beat the Market"** [18]: noise-area intraday momentum on SPY, 2007–2024, net Sharpe 1.33 (preprint). A single-author GitHub replication reports Sharpe 1.11 over Jul 2020–Jul 2026, then "edge compressed since 2025, recent Sharpe ~0" on SPY and ES [19]. **Fails survival after 2025** and is unreviewed. Confidence: medium-low.
- **Rosa, *J. Futures Markets* 2022** [20]: out-of-sample predictability of overnight return for the last half-hour disappears. That supports the conclusion rather than challenging it (abstract via search snippet).

Also supportive: an arXiv study of 14 OHLCV signal families on MNQ 5-minute bars, 2021–2025, found none passing its five criteria, with 11 of 14 lacking gross edge over a 2-point friction [21] (preprint; abstract only).

**Conclusions 2 and 3 were not challenged by anything retrievable.** No CL/MCL replication of the EIA rule and no peer-reviewed post-2015 index-futures evidence for turn-of-month, RSI(2)/IBS or pre-holiday was found. That is absence of retrieved evidence, not confirmation. A snippet claiming turn-of-month persistence through Q3 2024 could not be verified and is not counted.

**Limit of the red team:** it did not search overnight/intraday decomposition, MOC-imbalance rules, VIX or bond lead-lag to equity index, public order-flow imbalance, FOMC/PEAD futures drift, 1–5 day index trend, or BTC/ETH rules. Conclusion 1 is therefore only partly stress-tested; the untouched space is unsearched, not cleared.

**Project note (my inference, checked against the `mim_nb_live.py` docstring):** the two near-misses that survive best are noise-band intraday momentum, which is the family your live MIM-NB bot runs (noise bands, HH:00/HH:30 entry checks, EOD flatten). The independent evidence for that family is real up to 2020–2024, and the only replication found reports it compressing to Sharpe ~0 from 2025 [19]. That is one unreviewed source, but it bears on a live bot whose ledger is at N=28 with PF 0.848. It is not a new candidate.

## 7. Recommendations

1. **Do not pre-register or change any parameter from this recon.** Confidence: high. Nothing here supports it, and the policy requires a power gate first.
2. **If continuing, run a cost-and-power gate on two candidates, in this order.** Confidence: medium on the ordering, since both are low-probability.
   - **EIA crude / MCL:** pull MCL 1-minute bars around Wednesday 10:30 ET, measure the real spread and cost in the 15:30–16:00 window, and compute the power gate from the paper's d ≈ 0.078 haircut for decay. Expected verdict: UNDERPOWERED. That closes it in a day. Feeds the research queue.
   - **Overnight reversal / index futures:** the single decisive check is the delayed-entry curve on your front-month MNQ/MES bars. That means signal at the open, entry at +1, +5 and +15 minutes, and costs from your own cost card. If the edge drops as it did in the stock test, close it. Also record how correlated the signal is with GAP-1 first, because the family overlap is likely.
3. **Check the decay report against the live MIM-NB ledger before adding anything new.** Confidence: low-medium, since the source is a single unreviewed GitHub replication [19] and MIM-NB's N=28 cannot confirm or refute it. It is a cheap read-only comparison of the ledger with the report's post-2025 window, not an action on the live bot.
4. **Do not pursue** FX fix, turn-of-month, RSI(2)/IBS, hour-of-day and turn-of-candle crypto, or CME-gap fills. Confidence: high for the FX fix and crypto microstructure items (measured against cost); medium for turn-of-month.
5. **Cheap unblock worth doing before any of this:** get the full text of the JoF 2024 FX paper and the MDPI Kraken paper through your own browser. Both were 403 for the research tools. Only worth it if you want to reopen either.

**Follow-up 2026-09-21 — EIA crude power gate run: UNDERPOWERED.** About 1,010 trades needed at the paper's own point estimate (19–23 years at 44–52 events a year); a 2-year leash can only see an effect 3.1–3.4x the paper's. Gross edge is about one MCL tick, so the cost flag is also unfavourable (MCL costs unsourced). Details: `_bmad-output/diagnostics_eia_crude_power_gate_20260921/REPORT.md`. Recommendation 2's crude branch is closed; the overnight-reversal branch is still open.

**Follow-up 2026-09-21 — overnight-reversal power gate run: UNDERPOWERED** (2 MNQ vs 3 MES spread, 992 days 2021–2024, signal-free). At the paper's own 1-minute entry retention (0.31) a cost-free test needs 4.0 years and the net-of-cost test 53; a 2-year leash needs 0.65 retention. Only the paper-as-printed (same-open, untradeable) effect is powered. Correction to Section 4: the GAP-1 overlap I suspected applies to a directional version; the spread version is market-neutral and belongs to the pair-divergence family already closed. Details: `_bmad-output/diagnostics_overnight_reversal_power_gate_20260921/REPORT.md`. Both finalists are now closed by power gates.

**Follow-up 2026-09-21 — MIM-NB ledger vs the decay report [19]: uninformative.** N=29, -$20, PF 0.996, +0.31 bp/trade (95% CI -27 to +28) against the report's +2.6 bp; daily Sharpe -0.02 (CI -3.9 to +3.7). Detecting the report's effect needs about 5,270 trades, so the live ledger can neither confirm nor refute it. Details: `_bmad-output/diagnostics_mim_nb_vs_decay_report_20260921/REPORT.md`.

**Follow-up 2026-09-21 — sealed MIM-NB engine on 2025 bars: in-sample, cannot address decay.** 2025 is the sealed dev window (correction to the earlier "unseen data" suggestion). Engine reproduces the sealed trades exactly; S250 PF 1.50, daily Sharpe +1.30 (CI includes 0), PF 0.90 without the 5 best days. Only unseen history is MNQ front-month 2021–2024. Details: `_bmad-output/diagnostics_mim_nb_2025_rerun_20260921/REPORT.md`.

**Follow-up 2026-09-21 — power gate on MNQ 2021–2024 (MIM-NB decay test): UNDETERMINED.** About 350–430 unseen trades; powered only if the true effect is at least the 2025 in-sample estimate. Details: `_bmad-output/diagnostics_mim_nb_2021_2024_power_gate_20260921/REPORT.md`.

**Decision 2026-09-21 (Alex): the 2021–2024 test is skipped.** The window stays unspent; the MIM-NB decay question is left open.

## 8. Open questions

- Does any independent replication of the Wen et al. EIA effect on CL/MCL futures exist, or any post-2019 evidence? Route: read the three citing papers via Semantic Scholar/Google Scholar, or the SSRN Wen–Gong–Ma–Xu paper.
- Does the overnight-reversal edge survive a delayed entry on index futures, and after 2014? Route: your own data (the gating test above).
- What are MCL and M6E spreads at 15:30–16:00 and 11:00 ET? Route: CME fee schedule and observed quotes.
- Are there candidate families outside the three clusters that qualify? Route: the red-team pass hunted for counterexamples in a fixed set of families; a Deepen run could widen it.

## 9. Source appendix

| n | claim / finding it supports | publisher | pub date | accessed | confidence |
|---|---|---|---|---|---|
| 1 | Turn-of-month 0.15%/day, t=3.78; decay across halves; ex-top-1% robust | [Financial Analysts Journal (McConnell & Xu)](https://business.purdue.edu/faculty/mcconnell/publications/Equity-Returns-at-the-Turn-of-the-Month.pdf) | 2008-03 | 2026-09-21 | high |
| 2 | Turn-of-month largely gone in US ETFs | [Quantseeker (Substack)](https://www.quantseeker.com/p/turn-of-the-month-strategies-do-they) | 2025 (unverified) | 2026-09-21 | low |
| 3 | IBS/RSI(2): practitioner-only evidence, no dates or OOS | [Alvarez Quant Trading](https://alvarezquanttrading.com/blog/internal-bar-strength-for-mean-reversion/) | undated | 2026-09-21 | medium |
| 4 | Overnight-intraday reversal on 5 index futures; delayed-entry collapse in stocks; gross only | [Della Corte, Kosowski, Wang (CICF conference draft)](https://www.cicfconf.org/sites/default/files/paper_357.pdf) | 2015-11 | 2026-09-21 | high (as reported) |
| 5 | EIA-day crude intraday momentum: USO 2006–2019, t=1.88, no costs, no OOS, decline | [The Energy Journal (accepted manuscript, via CORE)](https://core.ac.uk/download/612285763.pdf) | 2023-09 | 2026-09-21 | high |
| 6 | Commodity-ETF half-hour momentum OOS R² 0.2–0.4%, no costs | [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC7480318/) | ~2020 | 2026-09-21 | medium |
| 7 | FX fix W-shape: net of spread ≈ 0; decay; futures gross only | [Bank of Canada Staff WP 2021-48 (preprint of JoF 2024)](https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf) | 2021-10 | 2026-09-21 | high |
| 8 | BTC first-half-hour momentum (abstract only) | [Birmingham / Financial Review](https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/) | 2022-05 | 2026-09-21 | medium |
| 9 | BTC intraday sign changes with jumps/FOMC; sample to 2020 | [RePEc / Elsevier (Wen, Bouri, Xu, Zhao)](https://ideas.repec.org/a/eee/ecofin/v62y2022ics1062940822000833.html) | 2022 | 2026-09-21 | high |
| 10 | Kraken 12h session rules not significant vs buy-and-hold (abstract only) | [MDPI J. Risk Financial Manag. 19(9):692](https://www.mdpi.com/1911-8074/19/9/692) | 2026 | 2026-09-21 | medium |
| 11 | BTC hour-of-day: ~6 bp gross over 21–23 UTC; rough 2022–23 | [Quantpedia](https://quantpedia.com/the-seasonality-of-bitcoin/) | ~2023–24 | 2026-09-21 | medium |
| 12 | CME BTC trading near-24/7 from 2026-05-28 ends weekend gaps | [CoinDesk](https://www.coindesk.com/markets/2026/05/28/bitcoin-s-famous-cme-gaps-are-about-to-disappear-though-three-remain-unresolved) | 2026-05-28 | 2026-09-21 | high |
| 13 | Turn-of-candle 0.58 bp/min; emerged mid-2020; less pronounced by 2022 | [PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/) | 2023 | 2026-09-21 | medium |
| 14 | MCL 100 bbl, $1 tick, ~23h (search-snippet spec) | [CME Group education page](https://www.cmegroup.com/education/courses/basic-principles-of-micro-wti-crude-oil-futures/micro-wti-crude-oil-futures-overview) | n/a | 2026-09-21 | medium |
| 15 | Sunday 23:00 UTC burst; day-of-week effects collapse on hourly data | [ML Quants Substack summarising SSRN 6776934](https://mlquants.substack.com/p/are-day-of-the-week-effects-in-cryptocurrencies) | 2026-05 | 2026-09-21 | low |
| 17 | Intraday momentum across 60+ futures, 1974–May 2020; sample halves similar; costs not in main table | [Journal of Financial Economics (Baltussen et al., author copy)](https://academicweb.nd.edu/~zda/intramom.pdf) | 2021 | 2026-09-21 | high |
| 18 | Noise-area intraday momentum SPY 2007–2024 net Sharpe 1.33 (preprint) | [SSRN 4824172 / Swiss Finance Institute WP 24-97](https://ssrn.com/abstract=4824172) | 2024-05 | 2026-09-21 | medium (snippet) |
| 19 | Independent replication Sharpe 1.11 (2020–26), edge compressed since 2025, recent Sharpe ~0 | [GitHub: giovannibrusco/zarattini-2024-momentum-spy](https://github.com/giovannibrusco/zarattini-2024-momentum-spy) | 2026 | 2026-09-21 | low-medium (single, unreviewed) |
| 20 | Overnight→last-half-hour predictability disappears out of sample | [J. Futures Markets (Rosa), via RePEc](https://econpapers.repec.org/RePEc:wly:jfutmk:v:42:y:2022:i:12:p:2218-2234) | 2022 | 2026-09-21 | medium (abstract snippet) |
| 21 | 14 OHLCV signal families on MNQ 5-min 2021–2025: none pass | [arXiv 2605.04004](https://arxiv.org/abs/2605.04004) | 2026-05 | 2026-09-21 | medium (abstract) |
| 16 | Website backtest of index-futures overnight reversal (Sharpe 2.09), costs unknown | [QuantReturns](https://quantreturns.com/strategy-review/overnight-mean-reversion/) | ~2025 | 2026-09-21 | low |

## 10. Staleness map

Computed with `recon_kit.py staleness` at a 12-month window for effect-size, decay and replication claims (the academic-lit bar for state-of-the-art claims; seminal work has none). Five of the eight tracked claims are past that window (the four older papers and the Baltussen sample, which ends May 2020). That means **their post-publication behaviour has not been re-checked**, not that the papers are superseded. The claims that will age fastest and matter most:

| claim | class | published | re-check by |
|---|---|---|---|
| Noise-area intraday momentum replication: Sharpe ~0 since 2025 [19] | decay | 2026-07 | 2027-07 |
| Kraken 12h session rules not significant vs buy-and-hold [10] | effect-size | 2026-09 | 2027-09 |
| CME BTC trades near-24/7, weekend gaps gone [12] | decay | 2026-05 | 2027-05 |
| EIA-day crude rule (USO 2006–2019) [5] | effect-size | 2023-09 | overdue: re-check for any replication or post-2019 evidence |
| FX fix net-of-cost result [7] | effect-size | 2021-10 | overdue: read the JoF final version and any post-2019 follow-up |

The earliest re-check date the tool returns is 2009-03-01 (turn-of-month, 2008), which only shows how old that source is. **The practical first re-check is the [19] decay report against the MIM-NB ledger, and a Refresh of this run after 2027-05.** Refresh or Deepen this run folder to pick these up.
