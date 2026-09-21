# Digest fxfix-r2-1: Krohn, Mueller, Whelan (FX fixings "W-shaped" pattern) - depth pass

Accessed 2026-09-21. Primary text read: Bank of Canada Staff Working Paper 2021-48 (last updated 2021-10-06, PDF read in full for Sections I-V, Tables I-IX; preprint, NOT the JoF final). Also read: the July 2019 draft (INSEAD-hosted PDF, earlier version). The JoF 79(1) 541-578 (Feb 2024) page returned HTTP 403, so the published version's numbers are NOT verified; all numbers below are from the 2021 WP unless marked "2019 draft". Units: paper returns are ANNUALIZED percent (mean = daily bps x 252 / 100 as far as I can tell); daily-bps conversions are my arithmetic and marked (calc).

## Findings

**1. Sample, pairs, data.** Jan 1999 - Dec 2019 (21 yrs), tick data, G9 vs USD: AUD, CAD, EUR, JPY, NZD, NOK, SEK, CHF, GBP (WP p.1, p.10). Main data = Refinitiv Tick History INDICATIVE bid/ask quotes (bank-to-client aggregator; cannot see traded prices or volume). Refinitiv Matching (RM, inter-dealer, traded prices) from Jun 2006; CME FX futures from Jan 2006; ICE DX futures from Jan 1999 (p.10-11). Fix times: Tokyo 9:55 local (~8:55pm ET), ECB fix 2:15pm CET (~8:15am ET), London WM/R 4:00pm London (11:00am ET). Sample ends 2019: nothing after Dec 2019 is tested.

**2. Trading-strategy results.** The paper's own summary (p.5): ignoring costs, strategies "yield significant returns"; "most of the trading profits disappear when transaction costs are incorporated." Table IX (RTH indicative quotes, EUR/GBP/JPY, 1999-2019, long USD pre-fix, flip to short USD post-fix; annualized avg % / Sharpe vs T-bill):
- Tokyo window, EUR/GBP/JPY, full spread (100%): pre+post -5.14 / -6.92 / -6.54 (SR -1.66 / -2.19 / -1.42). Half spread (50%): +2.65 / +0.51 / +2.73 (SR +0.39 / -0.18 / +0.29). Zero cost: +10.44 / +7.94 / +12.00 (SR 2.45 / 1.84 / 1.99).
- Europe window (long USD 02:00-08:15 ET, short USD 11:00-17:00 ET), full spread: EUR +1.52 (SR 0.05), GBP -1.51 (SR -0.38), JPY -18.23 (SR -2.86). Half spread: EUR +8.64 (SR 1.03), GBP +5.47 (SR 0.60), JPY -9.27 (SR -1.54). Zero cost: EUR +15.75 (SR 2.01), GBP +12.45 (SR 1.57), JPY -0.31.
- Authors: half-spread SRs "between 0.12 and 1.03" for those who trade at better-than-quoted spreads. The "Sharpe 0.5-0.7" in the prior screen matches the 2019 DRAFT (its half-spread claim), not the 2021 WP. 2019 draft also states: EUR around London fix, "conservative" costs, $1 -> more than triple by 2018, SR 0.65 (draft text, no table retrieved).
- Gross per-trade scale (calc): EUR Europe window 15.76%/252 = ~6.3 bps/day, daily SD = 7.24%/sqrt(252) = ~46 bps (calc; Table VII annualized SD 7.24%); EUR Tokyo 10.44% -> ~4.1 bps/day, SD 3.79% -> ~24 bps (calc). Fraction of positive days: EUR Tokyo 0.57, EUR Europe 0.56, GBP Europe 0.55 (Table VII). Kurtosis: EUR 8.7 (Tokyo) / 6.9 (Europe); GBP Tokyo 38.3 (fat tails); JPY Europe mean -0.31 (t -0.21).
- Ranking that matters for a small account: the ONLY cost-robust row (positive at full quoted spread) is EUR Europe window, +1.52%/yr, SR 0.05 = zero. Everything else needs the authors' assumed 50%-of-quoted-spread execution.

**3. "Transaction Costs" section (Sec. V.E, p.33-36).** Method: quoted RTH bid/ask, buy at ask/sell at bid on every leg. Authors note quoted indicative spreads overstate big-trader costs; cite Cespa et al. (2021) suggesting cutting them by up to 75%; so they report 100%/50%/0%. Conclusion (p.35-36, verbatim gist): "while there is strong intraday predictability around the fixings, it is not obvious that this can be exploited by the average trader. First, returns from trading in a relatively small window around the fix are usually more than offset by transaction costs." Cumulative 50%-spread total return indices (Fig. 7): Tokyo strategy negative early, flat ~2007, reversed to ~2013, "negative returns for the EUR and GBP and flat for the JPY" since; "almost doubled" 1999-2019 for JPY/EUR, flat for GBP. London strategy: $1 -> $3.13 (GBP) / $6.08 (EUR); JPY lost heavily. Only EUR/GBP work for Europe window. The paper argues the pattern is compensation for dealer inventory risk, i.e. NOT a market inefficiency (2019 draft: "do not represent a market inefficiency").

**4. Post-2015 / most recent subsample (Table II, dollar portfolio, annualized %, t-stat).** pre-Tokyo / post-Tokyo / pre-ECB / post-London:
- 1999-04: -4.02 (-5.0) / 6.29 (6.7) / -8.57 (-4.8) / 7.05 (4.6)
- 2004-09: -6.01 / 6.96 / -4.68 / 4.54 (t 2.5)
- 2009-14: -9.05 / 6.50 / -0.92 (t -0.40) / 5.20 (t 2.5)
- 2014-19: -1.59 (-2.6) / 2.08 (2.3) / -3.14 (-2.3) / 2.16 (t 1.62, NOT significant)
So 2014-19 is roughly 1/3 to 1/2 of earlier magnitude and post-London is insignificant. Text (p.29-30): Tokyo-fix reversal returns "around 5% or below since 2013," first negative year 2018; Europe-fix reversal dropped "below 5% during the last two years"; "downward trend during the most recent period"; authors cite trader anecdotes that arbitrage capital now trades it, and low returns coincide with low VIX. Reversal size scales with lagged VIX (Table VIII: VIX coef 0.24 Tokyo, 0.14 Europe; ~4 bps/day Tokyo, ~2.5 bps/day London at median VIX 18). No test after 2019. Note: the 2015 WM/R window lengthening (1 min -> 5 min, 15 Feb 2015, fn 8) is described but the paper does not report a pre/post-Feb-2015 split in the text I read (2014-19 bucket straddles it).

**5. Month-end / quarter-end dependence.** Paper says (p.30-31, Sec. V.C) the reversals are "consistently positive in all days of the week, weeks of the month as well as months of the year," "not driven by an end-of-month equity hedging channel" (Melvin & Prins 2015), not by 3rd-week option hedging, and "we also do not find a stronger effect at the end of each quarter"; also not driven by announcement days. Supporting tables are in an Online Appendix that I did NOT retrieve; only the text claim was read. So no reliance on month-end days: trade frequency is roughly daily (about 252 sessions/yr per window, up to 2 windows; each window = long leg + short leg = ~4 order events per day round trip, paper p.33). NOT RETRIEVED: any independent month-end split with trade counts. An unattributed search-result snippet claimed a post-reform reversal persists "in both end-of-month and intra-month days ... insufficient sample size"; I could not trace it to a document; treat as unverified.

**6. Independent replications / critiques.** NOT RETRIEVED for the reversal pattern itself. Related but not replications: Evans, O'Neill, Rime, Saakvitne, "Fixing the Fix?" (FCA Occasional Paper 46 / Georgetown WP, Oct 2018): the 2015 window lengthening improved benchmark price efficiency and robustness but raised tracking error, quoted spreads and price impact rose, HFTs traded more aggressively in the fix (abstract-level via search summary; PDF not read). Benenchia, Galati, Lepone, Pacific-Basin Finance J. 93 (2025), pre-registered re-evaluation of WMR 4pm fix: abstract read (repec), no mention of reversals or Krohn et al. Melvin & Prins (2015 J. Financial Markets, equity hedging at London fix): earlier mechanism paper, not read. No paper found that replicates the W-pattern out of sample after 2019, nor one arguing against it.

**7. Tradability on CME FX futures.**
- Evidence the pattern exists in futures (Table III, dollar portfolio = EUR+GBP+JPY, 2006-2019, annualized % (t)): CME futures pre-Tokyo -4.06 (-8.9), post-Tokyo +5.61 (10.0), pre-ECB -5.06 (-5.8), E-L 0.36, post-London +1.88 (2.6); ICE DX futures -2.09/+2.53/-2.68/+2.49; RM traded VWAPs -1.70/+3.88/-6.87/+4.18 (t 4-6). CME post-London effect is about 1/3 of the spot number (1.88 vs 6.69 forwards). These are gross mid/VWAP returns, not net of costs. The paper also says futures ORDER FLOW is unrelated to the reversals (dealers act in the inter-dealer market), so futures are a place the price pattern is reflected, not the source.
- 2019 draft: CME futures at FULL quoted spread: EUR ECB-fix trade SR 0.61; pound and yen negative. That row is NOT in the 2021 WP Table IX (which is RTH only); I do not know whether it survived into JoF.
- Contract facts (CME search-result summary of cmegroup.com; page fetch timed out): Micro EUR/USD (M6E) = EUR 12,500 (1/10 of 6E), tick $1.25/contract. Not retrieved: trading hours page, actual bid/ask spread by time of day, volume at 8:55pm ET (Tokyo) / 8:15am ET (ECB) / 11am ET (London), commissions. So "liquid at those times" is NOT evidenced this run for M6E specifically; the paper's own volume figures (p.14-15 in text; RM/CME notional volumes for EUR/JPY futures reach ~34 and 12 bn USD at peaks) show volumes spike at fixes but I did not extract the M6E-specific numbers.
- Rough scale (calc, assumes EURUSD ~1.15, unverified): 1 bp on M6E ~ $1.4; gross EUR Europe-window edge ~6 bps/day ~ $9 per micro per day vs a $1.25 tick and 4 order events/day (2 legs, each entered and exited). The authors' full-spread result of ~+1.5%/yr on ~1.5 bps/day-equivalent (calc) implies costs consume essentially all the gross edge at quoted spreads. Futures spreads may be tighter than RTH indicative (paper's own point) but that was not measured here.

**Per-currency effect sizes (Table I; annualized %).** Pre-ECB / post-London: EUR -8.87/+6.87, SEK -7.73/+6.55, GBP -5.78/+6.67, CHF -6.43/+5.65, NOK -3.51/+5.40, CAD -1.91/+3.89, AUD -1.03/+4.86 (pre-E t -0.77), NZD -0.86/+4.55, JPY -2.61/**-2.92** (yen goes the wrong way after London). Pre-Tokyo / post-Tokyo: EUR -4.54/+5.89, GBP -4.75/+3.19, JPY -4.06/+7.94, AUD -7.15/+4.66, NZD -8.53/+5.77. Close-to-close returns are not significant for any currency but AUD/NZD/CHF (avg ~1%/yr for the portfolio): the pattern is intraday-only. A EUR-only Europe-window version is the strongest; GBP is second; JPY only works around Tokyo and fails at London. These are full-sample averages, 1999-2019, and by-subperiod per-currency values were not retrieved.

## Claims

| claim | source URL | publisher | pub_date | accessed | confidence | class |
|---|---|---|---|---|---|---|
| Sample Jan 1999-Dec 2019, G9 vs USD, RTH indicative quotes; RM traded data from Jun 2006; CME futures from Jan 2006 | https://www.bankofcanada.ca/wp-content/uploads/2021/10/swp2021-48.pdf | Bank of Canada (Staff WP 2021-48, preprint) | 2021-10-06 | 2026-09-21 | high | primary, preprint |
| Table IX: EUR Europe-window at 100% quoted spread +1.52%/yr, SR 0.05; at 50% spread +8.64%, SR 1.03; GBP 50%: +5.47%, SR 0.60; JPY negative | same | Bank of Canada | 2021-10-06 | 2026-09-21 | high (table read) | primary, preprint |
| Authors state returns in small windows "usually more than offset by transaction costs"; not exploitable by average trader | same | Bank of Canada | 2021-10-06 | 2026-09-21 | high | primary, preprint |
| Dollar portfolio 2014-19: pre-Tokyo -1.59, post-Tokyo +2.08, pre-ECB -3.14, post-London +2.16 (t 1.62) vs ~5-9% earlier | same | Bank of Canada | 2021-10-06 | 2026-09-21 | high | primary, preprint |
| Authors admit downward trend in recent period; Tokyo reversal <=5% since 2013; Europe <5% in last 2 yrs; arbitrage capital anecdotally trading it | same | Bank of Canada | 2021-10-06 | 2026-09-21 | high | primary, preprint |
| Reversal not concentrated on month-end/quarter-end/announcement days (text claim; appendix not read) | same | Bank of Canada | 2021-10-06 | 2026-09-21 | medium | primary, preprint |
| Pattern present in CME FX futures (EUR/GBP/JPY, 2006-19) pre-Tokyo -4.06, post-Tokyo +5.61, pre-ECB -5.06, post-London +1.88 (gross) | same | Bank of Canada | 2021-10-06 | 2026-09-21 | high | primary, preprint |
| 2019 draft: half-spread Sharpe 0.5-0.7; CME futures full-spread EUR ECB-fix SR 0.61; EUR London-fix SR 0.65 conservative costs | https://sites.insead.edu/facultyresearch/research/file.cfm?fid=66802 | INSEAD-hosted author draft | ~2019-07-23 (PDF metadata) | 2026-09-21 | medium (superseded draft; text-extracted) | primary, superseded draft |
| Published as J. Finance 79(1) 541-578, Feb 2024; abstract: USD appreciates into fixes, depreciates after; W-shape; 21 years, top nine currencies | https://ideas.repec.org/a/bla/jfinan/v79y2024i1p541-578.html | RePEc/Wiley (search summary; Wiley 403) | 2024-02 | 2026-09-21 | medium (final text unread) | primary metadata, peer-reviewed |
| M6E = EUR 12,500, tick $1.25 | https://www.cmegroup.com/trading/fx/e-micros/e-micro-euro_contract_specifications.html | CME Group (via search summary; page fetch timed out) | n/a | 2026-09-21 | medium | primary, unread page |
| 2015 window lengthening improved robustness but raised tracking error/spreads/price impact | https://www.fca.org.uk/publications/occasional-papers/occasional-paper-no-46-fixing-fix-assessing-effectiveness-4pm-fix | UK FCA (Evans et al.) | 2018-10 | 2026-09-21 | low-medium (search summary only) | regulator occasional paper |
| WMR 4pm fix re-evaluation after 2015 (no reversal analysis in abstract) | https://ideas.repec.org/a/eee/pacfin/v93y2025ics0927538x24004049.html | Pacific-Basin Finance J. (Benenchia, Galati, Lepone) | 2025 | 2026-09-21 | high (abstract) | peer-reviewed, not a replication |

## Inputs a power test would need (mean, SD, trades/yr, cost)

| input | value available | source | what is missing |
|---|---|---|---|
| Gross mean, EUR Europe window | 15.76%/yr ~ 6.3 bps/day (calc) full sample; dollar portfolio 2014-19 pre-ECB only ~3.1%/yr ~1.2 bps/day (calc) | Table VII, Table II | EUR-only post-2015 or post-2019 estimate; JoF numbers |
| SD | EUR Europe 7.24%/yr ~ 46 bps/day (calc); EUR Tokyo 3.79%/yr ~ 24 bps/day (calc) | Table VII | SD by subperiod; per-trade (not per-window) dispersion; futures-based SD |
| Trades/yr | ~252 days x 1-2 windows, each window a long and a short leg (4 order events per day) | Sec. V.E logic | Any filter (e.g. VIX>x) changes N and was not tested out of sample |
| Cost | Table IX: full quoted spread leaves ~0 (EUR +1.5%/yr); 50% spread leaves +8.6%/yr EUR | Table IX | Real M6E spread at 2am/8:15am/11am ET, commissions/fees, slippage; futures cost row exists only in 2019 draft |
| Sharpe | net 0.05 (100% spread) to 1.03 (50%) EUR Europe; SR is vs T-bill, 1999-2019 | Table IX | Post-2019 SR; net-of-cost SR by subperiod (Fig. 7 shows Tokyo net flat/negative since ~2013) |
| Stability | 2014-19 effect roughly halves; post-London insignificant | Table II | Any 2020-2026 data; independent replication |
Power implication (calc, informal, not a test): if the true net edge is closer to the post-2013 level (~1-3 bps/day gross, ~0 net at quoted spreads) against a ~46 bps daily SD, per-trade t-ratio is ~0.03-0.07 per sqrt(day), i.e. detection would need multi-decade samples. That supports "UNDERPOWERED/not worth testing" unless a much bigger, conditional effect is identified.

## Contrary evidence found

- Authors' own conclusion: net of quoted spreads the strategy is ~zero or negative; only positive under an assumed 50% spread cut (from a different paper's suggestion for large OTC traders, not a micro-futures retail account).
- Recent-period decay: 2014-19 magnitudes ~1/3-1/2 of earlier; post-London leg insignificant; Tokyo strategy net-of-cost negative for EUR/GBP since ~2013 (Fig. 7 text); reversal linked to VIX so quiet regimes give little.
- Yen fails at the London fix (wrong sign); AUD/NZD pre-ECB insignificant. Pattern is pair-specific, so EUR/GBP only.
- Futures show a much weaker post-London effect (CME +1.88% vs forwards +6.69%).
- Explanation is dealer inventory compensation (not an inefficiency): a rewarded risk premium that can vanish when dealers' constraints loosen or capital arrives.
- Sample ends 2019; no post-2019 out-of-sample evidence.
- Working-paper numbers changed between the 2019 draft (Sharpe 0.5-0.7 at half spread) and the 2021 WP (0.12-1.03), a sign of specification sensitivity.

## What I could not find

- JoF 2024 final text (Wiley 403): any changes to Table IX, the futures row, or post-2019 data are unknown.
- Online Appendix (calendar/month-end splits, announcement-day splits, other base currencies).
- Any independent replication, or any citing paper critically testing the W-pattern after 2019 (two searches, none found).
- Any 2020-2026 data on the pattern.
- M6E/6E trading hours, spread and volume at 8:55pm/8:15am/11am ET, commissions; CME page fetch timed out twice.
- Per-currency values by subperiod; Sharpe of a EUR-only strategy after 2015; number of trades in month-end-only version.
- Pre/post Feb-2015 WM/R window split of the reversal (text only describes the change).
