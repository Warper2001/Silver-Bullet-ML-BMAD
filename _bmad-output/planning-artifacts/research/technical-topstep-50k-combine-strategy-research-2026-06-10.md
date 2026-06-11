---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - '_bmad-output/preregistration_s26_htf.md'
  - 'memory: project_s26_combine_verdict, project_combine_strategy_dead_end, project_btc_crypto_experiments'
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Most profitable strategy class satisfying Topstep 50K combine requirements (broad survey, ranked by combine fit, validatable instruments only)'
research_goals: 'Identify and rank documented futures strategy classes by probability of passing the Topstep 50K combine (+$3,000 target, -$1,000 DLL, -$2,000 trailing MLL), constrained to instruments with local validation data (MNQ/MES/ES, GC, MBT), judged by net-of-cost edge density per trade and Monte Carlo pass probability — not gross PF'
user_name: 'Alex'
date: '2026-06-10'
web_research_enabled: true
source_verification: true
---

# Edge Density Beats Expectancy Elegance: A Ranked Survey of Strategy Classes for the Topstep 50K Combine

**Date:** 2026-06-10 (completed 2026-06-11)
**Author:** Alex
**Research Type:** technical

---

## Executive Summary

Seventeen consecutive strategy failures in this project share one autopsy finding: the gross edge per trade was smaller than the cost to collect it, or the variance profile breached the trailing drawdown before the edge materialized. This research inverts the usual question. Instead of "which strategy is most profitable," it asks "what does the Topstep 50K rule structure *permit*," derives the constraint stack (verified rules, true costs, loss-streak math), and then scores documented strategy classes against it.

The survey produced one standout. **Market Intraday Momentum (MIM-Classic)** — first half-hour return predicts last half-hour return, Gao/Han/Li/Zhou, *Journal of Financial Economics* 2018 — is peer-reviewed, profitable after costs in the literature, structurally compatible with every combine rule (one trade/day, 30-minute hold, exits 8 minutes before the auto-flatten, naturally consistency-rule compliant), clears the edge-density yardstick by ~5× on paper (~$12 gross/contract/trade vs $2.24 true cost on MNQ), and is **the only major documented anomaly never tested in this project**. The strongest published class (multi-day trend following) is eliminated by the no-overnight rule, not by evidence; ORB and 1-min scalping are eliminated by our own graveyard.

**Key Technical Findings:**

- **The combine selects for win-rate density, not expectancy.** At WR 35–40%, 10-loss streaks are ~82% likely over 400 trades, forcing per-trade risk ≤ ~$130 against the $2,000 MLL; a WR≥55% profile can risk $200–300 and pass in 3–4 weeks.
- **True costs are far better than assumed on index micros:** MNQ ≈ $2.24/contract realistic round-trip (vs $7.84 MBT — confirming the S26 kill).
- **Two rules we never modeled change the Monte Carlo:** the MLL ratchets at end-of-day only (friendlier than assumed), and the consistency rule (best day < 50% of target) penalizes lumpy strategies (stricter than assumed).
- **No overnight holds (flat 3:10 PM CT)** structurally eliminates the academically strongest class (TSMOM) and caps every holding period at one session.
- **Base rate honesty:** ~5–10% pass evaluations; ~7% ever get a payout. The combine is priced against impatience; MC-gated entry is the counter.

**Technical Recommendations:**

1. Pre-register and Gate-0 test **MIM-Classic on MNQ** (dev 2025, one-shot OOS 2026, gates net of $2.24/contract).
2. Update the combine MC harness: EOD-ratchet MLL, DLL day-deactivation, consistency-rule check. Require pass% ≥ 50% before any combine entry.
3. Test **MIM-Noise-Bands** (Zarattini 2024) only after Candidate 1's verdict; **HTF mean reversion** only if both fail.
4. Treat **pre-FOMC drift** as an overlay on a passing base strategy, never standalone (~1.7 events/month cannot carry a combine).
5. Do not enter any combine until a candidate passes all three gates; calibrate the slippage assumption from the live S26 bot's fill data.

## Table of Contents

1. Technical Research Scope Confirmation
2. Technology Stack Analysis — Combine Parameters, Instruments & Cost Structure
3. Integration Patterns — Strategy Profile × Combine Rule Interaction
4. Architectural Patterns — Strategy Class Taxonomy, Scored
5. Implementation Approaches — Shortlist Specifications, MNQ-Translated
6. Strategic Synthesis, Roadmap & Risk Assessment
7. Methodology and Source Verification

---

## Research Overview

**Methodology:** Web-verified survey (official Topstep help-center sources for rules and fees; peer-reviewed papers for strategy evidence; practitioner sources flagged as lower confidence), combined with derived constraint math and this project's internal evidence base (17 pre-registered strategy failures, 2 live experiments, 1 validated dormant edge). Every candidate scored against the house yardstick established 2026-06-10: net-of-cost edge per contract per trade and combine Monte Carlo pass probability — gross profit factor is explicitly rejected as a ranking metric.

---

<!-- Content will be appended sequentially through research workflow steps -->

## Technical Research Scope Confirmation

**Research Topic:** Most profitable strategy class satisfying Topstep 50K combine requirements (broad survey, ranked, validatable instruments only)
**Research Goals:** Rank documented futures strategy classes by probability of passing the Topstep 50K combine (+$3,000 target, -$1,000 DLL, -$2,000 trailing MLL), constrained to locally validatable instruments (MNQ/MES/ES, GC, MBT); judge every candidate by net-of-cost edge density per trade and Monte Carlo pass probability, not gross PF.

**Technical Research Scope:**

- Architecture Analysis — strategy class taxonomy (trend/momentum, HTF mean reversion, event-driven, volatility regime, seasonality, spread/carry) vs trailing-drawdown compatibility
- Implementation Approaches — documented entry/exit specs with published, independently corroborated evidence
- Technology Stack — instrument selection within validatable set; tick values, ATR, RT costs; edge-density yardstick (net $/contract/trade vs ~$4–6 RT)
- Integration Patterns — combine-rule interaction (frequency, WR profile, loss clustering vs DLL/MLL math); sizing schemes
- Performance Considerations — expected N/month, realistic net PF range, projected combine MC pass probability per shortlisted class

**Research Methodology:**

- Current web data with rigorous source verification; vendor performance claims excluded without independent corroboration
- Multi-source validation for critical claims; confidence level framework
- House yardstick applied throughout: net-of-cost edge per contract + combine Monte Carlo — gross PF is meaningless
- Prior internal evidence carried in: 16 dead strategy attempts incl. S26 (all timeframes), HCVWAP, ORB, PBC; BTC-CARRY validated/dormant; S25 live pending (~2026-07-23)

**Scope Confirmed:** 2026-06-10

## Technology Stack Analysis (Domain: Combine Parameters, Instruments & Cost Structure)

### Verified Topstep 50K Combine Parameters (current as of 2026-06-10)

| Parameter | Value | Notes |
|---|---|---|
| Profit target | **+$3,000** | from $50,000 starting balance |
| Daily Loss Limit (DLL) | **-$1,000** | hitting it deactivates the *day*, not the account |
| Maximum Loss Limit (MLL) | **-$2,000 trailing** | starts at $48,000; **ratchets up at end-of-day only** (monitored real-time for breach); **locks permanently at $50,000** once reached |
| Consistency rule | **best day < 50% of profit target** | ⚠️ newly confirmed constraint — penalizes lumpy/event-driven equity curves; minimum ~2-3 distributed winning days |
| Max position | 5 contracts / **50 micros** | combine scaling cap |
| Cost | $49/month + $149 activation | unlimited time, min 2 trading days |

_Sources: [Topstep Help — Trading Combine Parameters](https://help.topstep.com/en/articles/8284197-trading-combine-parameters), [Topstep Help — Maximum Loss Limit](https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit), [Topstep Help — Daily Loss Limit](https://help.topstep.com/en/articles/8284207-what-is-the-daily-loss-limit-and-what-happens-if-i-exceed-it)_

**Modeling correction for our Monte Carlo harness:** our 2026-06-10 S26 simulation used per-trade trailing MLL (more conservative than reality) and ignored the consistency rule. Future MCs should model: EOD-ratchet MLL with real-time breach, DLL as day-deactivation (not account kill), and the <50% best-day consistency check. The S26 0%-pass verdict stands a fortiori.

### Verified Cost Structure (TopstepX, micro contracts)

| Instrument | Commission+fees RT | Tick value | +1 tick slip/side | **Realistic RT/contract** | Point value |
|---|---|---|---|---|---|
| MNQ | $1.24 | $0.50 | $1.00 | **≈ $2.24** | $2.00/pt |
| MES | $1.24 | $1.25 | $2.50 | **≈ $3.74** | $5.00/pt |
| MGC | $1.74 | $1.00 | $2.00 | **≈ $3.74** | $10.00/pt |
| MBT | $2.84 | $2.50 | $5.00 | **≈ $7.84** | $0.10/pt |

_Source: [TopstepX Commissions and Fees](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees) (CME non-member rates as of 2025-07-23; commissions $0.25/side micros)_

**Edge-density yardstick (house rule, updated with true costs):** a candidate must generate net positive expectancy after the Realistic RT column, with cost ≤ 25% of average gross win. MNQ is the cheapest venue per contract; MBT is the most expensive — the S26 BTC verdict worsens under true costs ($7.84 > our $6 estimate; edge was $1.09).

### Base Rates (industry reality check)

- 5–10% of traders pass a prop evaluation on first attempt; ~14% eventually pass; **only ~7% ever receive a payout**; 1–3% remain funded long-term (FPFX data, 300k+ accounts).
- Implication: the combine is priced against impatience. Our pre-registered, MC-gated approach is the correct counter; attempt entry should wait for a candidate with MC pass% ≥ 50%.

_Sources: [QuantVPS Prop Firm Statistics 2026](https://www.quantvps.com/blog/prop-firm-statistics), [Damn Prop Firms — Pass Rates Reality Check](https://damnpropfirms.com/trading-guides/prop-firm-evaluation-pass-rates-statistics-reality-check/)_

### Local Validation Infrastructure (constraint: research only what we can test)

| Asset | Data on disk | Granularity | Span |
|---|---|---|---|
| MNQ | mnq_1min_2025.csv / mnq_1min_2026_ytd.csv | 1-min | 2025-01 → 2026-05 |
| BTC (MBT proxy) | kraken/PF_XBTUSD_1min.csv | 1-min | 2024-11 → 2026-05 |
| ETH | kraken/PF_ETHUSD_1min.csv | 1-min | (downloaded) |
| GC | via download_gc_1min.py (TradeStation) | 1-min | on demand |
| ES/MES | via TradeStation API | 1-min | on demand |
| Macro | M2SL, DTWEXBGS, MVRV | daily | on disk |

Execution: TradeStation paper API (live polling bots), combine MC harness (2026-06-10 session), pre-registration seal workflow.

_Confidence: HIGH on combine parameters and fee schedule (official help-center sources, cross-checked). MEDIUM on slippage estimates (1 tick/side assumption; live S26 bot fill data can calibrate this)._

## Integration Patterns Analysis (Domain: Strategy Profile × Combine Rule Interaction)

### Structural Executability Filter (hard rules)

| Rule | Constraint on strategy class |
|---|---|
| **No overnight holds** — flat by 3:10 PM CT, auto-flatten from 3:08 PM CT; no session-to-session carry | Classic multi-day TSMOM/trend following **not executable**. Max hold = one session (5:00 PM CT → 3:10 PM CT next day, ~22h). Daily-bar signals are usable only if expressed as session-long holds. |
| Session window 5:00 PM CT → 3:10 PM CT | Overnight/Globex session edges (e.g., overnight equity drift) ARE partially capturable inside one trading day |
| DLL -$1,000 deactivates the day | Caps trades/day × per-trade risk; a runaway losing day costs time, not the account |
| MLL -$2,000, EOD ratchet, locks at $50,000 | Worst-case cumulative loss tolerance; binding constraint for low-WR classes (see math below) |
| Consistency rule: best day < 50% of profit target | Need ≥3 meaningfully distributed winning days; lumpy event-driven equity curves must size DOWN on event days |

_Sources: [Topstep — when/what can I trade](https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade), [Topstep — trade desk & flattening](https://www.topstep.com/blog/trade-desk-trade-flattening)_

### Loss-Clustering Math vs the $2,000 MLL (derived)

Streak tolerance = $2,000 / per-trade risk r (worst case, no ratchet credit):

| Per-trade risk r | Streak tolerance | Viable for WR profile |
|---|---|---|
| $100 | 20 losses | WR 35–40% (streaks of 10+ are ~82% likely over 400 trades at WR 36%) |
| $133 | 15 losses | WR ~40% floor |
| $200 | 10 losses | WR ≥ 50% |
| $300–400 | 5–6 losses | WR ≥ 55–60% only |

Expectancy throughput to +$3,000 (e = r·(WR·b − (1−WR)), b = avg win/avg loss):

| Profile | r | e/trade | Trades to target | At 3 trades/day |
|---|---|---|---|---|
| WR 38%, b=2.0 (our dead strategies' shape) | $130 | ≈ $18 | ~165 | ~55 days |
| WR 50%, b=1.5 | $200 | ≈ $50 | ~60 | ~20 days |
| WR 60%, b=1.0 | $300 | ≈ $60 | ~50 | ~17 days |

**Integration conclusion (derived + corroborated):** the trailing-DD structure imposes an asymmetric penalty on low-WR/high-RR strategies — variance hits the MLL before the edge materializes. Practitioner consensus matches: 1:1–1.5:1 RR at ≥55% WR outlasts 3:1 RR at 35% WR under trailing drawdown even at equal expectancy. Our own S26 MC is the internal confirmation: WR 38.7% b≈1.8 passed only 42% even with zero costs. **The combine selects for win-rate density, not expectancy elegance.**

_Sources: [Risk Reward Ratio in Trading (proptradingvibes)](https://proptradingvibes.com/blog/risk-reward-ratio-trading), [Trailing Drawdown Survival Guide (CrossTrade)](https://crosstrade.io/learn/risk-management/trailing-drawdown-survival-guide), [Losing-streak probability (backtestbase)](https://www.backtestbase.com/education/losing-streak-calculator-trading)_

### Position Sizing Schemes

- **Fixed fractional risk, MLL-anchored:** r ≤ MLL / required streak tolerance (table above). For a WR≥55% candidate: r ≈ $200–300 (1–1.5 MES-sized stops or 2–4 MNQ points × contracts).
- **Risk-constrained Kelly:** the formal version of our MC gate — choose size s.t. Prob(equity_min < MLL) < β. Our Monte Carlo harness already computes exactly this; β ≤ 0.3 at the chosen size is the deploy criterion. Full Kelly produces 40–60% drawdowns — never applicable here; ¼ Kelly or below maps to the $100–300 range above.
- **Consistency-rule throttle:** cap planned daily profit at ~$1,200 (40% of target) by halting entries after threshold — keeps the best-day ratio compliant without sacrificing pass speed materially.
- **Static-to-locked transition:** once balance ≥ $52,000 intraday → MLL locked at $50,000 after that EOD; risk can step UP after lock (the endgame is friendlier than the opening).

_Sources: [Risk-constrained Kelly (QuantInsti)](https://blog.quantinsti.com/risk-constrained-kelly-criterion/), [Risk of Ruin (CrossTrade)](https://crosstrade.io/learn/risk-management/risk-of-ruin)_

_Confidence: HIGH on rule mechanics (official sources). HIGH on derived math (arithmetic). MEDIUM on practitioner WR-profile consensus (corroborated by our internal MC, but selection-biased sources)._

## Architectural Patterns and Design (Domain: Strategy Class Taxonomy, Scored)

Scoring axes: published evidence quality, WR-profile fit (combine selects for win-rate density), executability (no overnight holds; flat 3:10 PM CT), cost fit on cheapest viable venue, consistency-rule fit, and **internal prior** (our 17 documented failures count as evidence).

### Class A — Intraday Momentum / MIM (first half-hour predicts last half-hour)

- **Published evidence: STRONG.** Gao, Han, Li & Zhou "Market Intraday Momentum" — first 30-min return predicts last 30-min return; ~6.5% annualized after transaction costs on SPY; robust across measures and the post-2001 era. Zarattini, Aziz & Barbon (SSRN 4824172, 2024) "Beat the Market: An Effective Intraday Momentum Strategy for the SPY" — recent out-of-sample-era replication with strong results. Robustness corroborated on KOSPI with explicit cost measures.
- **Combine fit:** trades once/day, defined-risk, intra-RTH (no flat-rule conflict), WR profile moderate-high, naturally distributed P&L (consistency-friendly). MNQ = cheapest venue ($2.24 RT vs $5/pt point value).
- **Internal prior: NEVER TESTED.** Not one of our 17 failures. The only major documented anomaly absent from our graveyard.
- _Sources: [Market Intraday Momentum (Gao et al.)](https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/ee7dac49-530b-4950-b5d0-e0b5eee08f2e.pdf), [Beat the Market — SPY intraday momentum (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172), [KOSPI MIM with trading costs (MDPI)](https://www.mdpi.com/1911-8074/15/11/523)_

### Class B — Opening Range Breakout

- **Published evidence: MIXED.** Zarattini & Aziz ORB results are for stocks-in-play with leverage, not index futures; index-futures ORB studies show significance only on NASDAQ futures, none on S&P/DJIA/HSI/TAIEX in full periods.
- **Internal prior: DEAD ×4** (SORM, ORBM-1/2/3 — "ORB structurally incompatible with combine math").
- **Verdict: eliminated.** External evidence does not overturn four internal failures on the actual instrument.
- _Sources: [Zarattini & Aziz ORB (Semantic Scholar)](https://www.semanticscholar.org/paper/4d55f526cc56f08662cb8976796cd3b719ef6d2b), [ORB on index futures (ResearchGate)](https://www.researchgate.net/publication/331076454)_

### Class C — Overnight Drift (early-morning long window, session-expressible)

- **Published evidence: REAL ANOMALY, COST-FRAGILE.** Equity premium earned overnight (Cooper et al. 2008); drift concentrated ~1:30–3:30 AM ET (Boyarchenko/Larsen/Whelan, NY Fed SR917) — pre-cost Sharpe ~1.3, **post-cost ~0.3** (Alpha Architect: "trading costs wipe out the overnight return anomaly").
- **Combine fit:** executable inside the Topstep session (window is mid-session); but per-window expected move on MNQ is small → edge density near the cost floor — the S26 failure shape.
- **Verdict: research-only; do not shortlist alone.** Possible filter/overlay for Class A.
- _Sources: [NY Fed SR917 — The Overnight Drift](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr917.pdf), [Alpha Architect — costs wipe out overnight anomaly](https://alphaarchitect.com/trading-costs-wipe-out-the-overnight-return-anomaly/)_

### Class D — HTF Intraday Mean Reversion (15-min/1-h band or RSI-style, MES/MNQ)

- **Published evidence: MEDIUM (practitioner-heavy).** Strong daily-bar evidence (IBS effect, RSI-2 family) but the daily forms require overnight holds → not executable; intraday adaptations documented mainly by practitioners (QuantifiedStrategies, StatOasis, UngerAcademy ES systems), little peer review.
- **Combine fit: best WR-profile match** (60–70% WR, modest RR) per the integration math — IF an intraday form has real edge at 15m+.
- **Internal prior: CAUTION.** VWAP mean reversion failed on 1-min MNQ (extensions were continuation, not exhaustion). Untested at 15m+ with band/close-location logic. The S26-HTF lesson cuts both ways: edge was scale-invariant there; mean reversion may differ because the 1-min failure was a *structure* finding (momentum at 1-min), and 15m structure can invert.
- _Sources: [IBS effect paper (NAAIM)](https://www.naaim.org/wp-content/uploads/2014/04/00V_Alexander_Pagonidis_The-IBS-Effect-Mean-Reversion-in-Equity-ETFs-1.pdf), [QuantifiedStrategies — S&P MR with IBS/RSI](https://www.quantifiedstrategies.com/sp-500-mean-reversion-using-ibs-and-rsi/)_

### Class E — Event-Driven Drift (pre-FOMC, post-CPI)

- **Published evidence: STRONG for pre-FOMC** (Lucca & Moench: +49bp S&P in 24h pre-announcement; ECB WP1901 broader pre-announcement drift ~4–5× normal-day returns). Weaker/absent for other macro prints in equities; gold event behavior documented mostly as volatility, not drift.
- **Combine fit: POOR ALONE** — ~8 FOMC + 12 CPI events/year ≈ 1.7 trades/month (target needs ~50–165 trades); consistency rule caps event-day P&L. **GOOD as portfolio component** layered on a base strategy.
- **Internal prior:** GC CPI prospective test pre-registered, event window live now (2026-06-11).
- _Sources: [Lucca & Moench — Pre-FOMC Announcement Drift (NY Fed)](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf), [ECB WP1901 — price drift before US macro news](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1901.en.pdf)_

### Class F — Multi-Day TSMOM / Trend Following

- **Published evidence: STRONGEST in the survey** (Moskowitz/Ooi/Pedersen; 58/58 futures positive; century of evidence) — **and not executable**: requires multi-day holds, Topstep mandates flat by 3:10 PM CT daily. Re-entry each session destroys the structure (daily cost × holding period).
- **Verdict: eliminated by structural rule**, not by evidence.
- _Sources: [Time Series Momentum (Moskowitz et al., NYU)](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf), [A Century of Evidence on Trend-Following (Hurst et al.)](https://fairmodel.econ.yale.edu/ec439/hurst.pdf)_

### Class G — 1-Min Scalping / Microstructure

- **Published evidence:** practitioner consensus favors it for trailing-DD survival (high WR, small risk) — but our **internal prior is 17 dead strategies**, all 1-min, all with edge below the cost floor. Internal evidence overrides practitioner folklore.
- **Verdict: eliminated.** The 1-min timeframe's edge density on retail-accessible signals is below the cost floor. This is the project's most expensively-earned fact.

### Scoring Summary

| Class | Evidence | WR-profile fit | Executable | Cost fit | Internal prior | **Rank** |
|---|---|---|---|---|---|---|
| A — Intraday momentum (MIM) | Strong (peer-reviewed ×2 + replication) | Good | Yes | Good (MNQ) | Untested ✓ | **1** |
| D — HTF intraday mean reversion | Medium (practitioner) | **Best** | Yes | Good (MES/MNQ) | Mixed | **2** |
| E — Event drift (pre-FOMC) | Strong | Good | Yes (≤22h session hold) | Good | Test live | **3 (component)** |
| C — Overnight drift window | Real but cost-fragile | Moderate | Yes | Poor | Untested | research-only |
| B — ORB | Mixed | Poor | Yes | OK | **Dead ×4** | eliminated |
| F — Multi-day TSMOM | Strongest | Good | **No** | — | — | eliminated (rule) |
| G — 1-min scalping | Folklore | Good | Yes | **Below floor** | **Dead ×17** | eliminated |

_Confidence: HIGH on Class A and F evidence (peer-reviewed, replicated). MEDIUM on Class D (practitioner sources). HIGH on eliminations (internal evidence is ours and extensive)._

## Implementation Approaches (Domain: Shortlist Specifications, MNQ-Translated)

### Candidate 1 — "MIM-Classic" (Gao, Han, Li & Zhou, JFE 2018)

**Published rule (frozen-translatable):**
- Predictor: first half-hour return r₁ = (P₁₀:₀₀ET − P_prev 16:00ET close) / P_prev close (optionally + penultimate half-hour return r₁₂ as second predictor)
- Trade: at 15:30 ET enter long if r₁ > 0, short if r₁ < 0; exit 16:00 ET market close
- Published: statistically/economically significant 1993–2013 SPY incl. after costs; ~6.6%/yr unlevered timing return; stronger on high-volatility, high-volume, and macro-news days

**MNQ translation & cost check (derived):**
- Exit 16:00 ET = 3:00 PM CT → inside the 3:08 PM CT auto-flatten with 8 min margin ✓
- Expected gross edge ≈ 2.6 bps/trade avg on ~$48k MNQ notional ≈ **$12/contract/trade vs $2.24 cost (cost fraction ~19%)** — clears the edge-density yardstick by ~5×, the first candidate in this project's history to do so on paper
- Frequency 1/day ≈ 21/month; WR (published) mid-50s%; 30-min hold caps per-trade risk (≈ 0.3–0.5% MNQ move ≈ $30–50/contract unstopped; optional stop at 0.5× first-half-hour range)
- Consistency rule: naturally compliant (no single day dominates)
- Local validation: fully testable on mnq_1min_2025/2026 (≈ 357 trading days → N≈357)

_Sources: [Market Intraday Momentum (SSRN 2440866)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866), [JFE published version](https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351), [QuantConnect replication](https://www.quantconnect.com/learning/articles/investment-strategy-library/intraday-etf-momentum)_

### Candidate 2 — "MIM-Noise-Bands" (Zarattini, Aziz & Barbon, SFI 24-97, 2024)

**Published rule (frozen-translatable):**
- Noise boundaries each minute: Open_today × (1 ± μ₁₄(t)), where μ₁₄(t) = 14-day average of |move from open to minute t|; upper bound shifted up by prior overnight gap-down, lower bound down by gap-up
- At each HH:00/HH:30 check: price above upper → long; below lower → short
- Exit: dynamic trailing stop (band/VWAP-based), else market close
- Published: SPY 2007–2024, +1,985% total **net of costs**, 19.6% ann., Sharpe 1.33

**MNQ translation & caveats (derived):**
- Lower WR than MIM-Classic (breakout family — expect strings of small stop-outs, big-day capture); combine WR-profile fit only moderate
- ⚠️ Cousin-risk: related to our 4 dead ORB variants (fixed open-range, 1-min). Differences that justify one pre-registered test: noise-scaled adaptive bands (not fixed range), all-day half-hourly checks (not open window only), trailing exit (not bracket). If tested, it counts as the *one* allowed ORB-family revisit with new economic rationale
- Local validation: fully testable on MNQ 1-min data

_Sources: [Beat the Market (SSRN 4824172)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172), [Concretum Group summary](https://concretumgroup.com/beat-the-market-an-effective-intraday-momentum-strategy-for-sp500-etf-spy/), [QuantMacro paper review](https://quantmacro.substack.com/p/paper-review-an-effective-intraday)_

### Candidate 3 — "HTF-MR" (15-min Bollinger/close-location mean reversion, MES or MNQ)

**Practitioner rule family (must be frozen before testing):**
- 15-min bars RTH; fade 2σ Bollinger (20) touches back to midline; regime filter mandatory (e.g., ADX floor or NoTrend gate); ~60% WR / ~1:1 RR profile claimed
- **No peer-reviewed net-of-cost evidence found** at this timeframe — evidence grade is the weakest of the three
- Best WR-profile fit for MLL math if real; highest risk of being our 18th corpse
- Position: test third, only if Candidates 1–2 both fail Gate 0

_Sources: [CrossTrade Bollinger MR](https://crosstrade.io/learn/trading-strategies/bollinger-mean-reversion), [QuantifiedStrategies MR overview](https://www.quantifiedstrategies.com/mean-reversion-strategies/)_

### Adoption Pipeline (house methodology, unchanged)

1. **Pre-register** Candidate 1 (MIM-Classic): frozen spec, dev = 2025, OOS = 2026 YTD, gates net of $2.24/contract
2. **Gate 0** → dev 2025: N ≥ 200, net PF ≥ 1.10, net expectancy > $0, cost ≤ 25% avg gross win
3. **Gate 1** → one-shot OOS 2026
4. **Gate 2** → combine MC **with corrected rules**: EOD-ratchet MLL, DLL day-deactivation, consistency-rule check; pass ≥ 50% at ≤ 10 MNQ
5. Candidate 2 only on Candidate 1 verdict; Candidate 3 only if both fail
6. Event-drift overlay (Class E) evaluated as enhancement after a base candidate passes, never as a standalone rescue

_Confidence: HIGH on published specs (two independent sources each for Candidates 1–2). MEDIUM on MNQ transferability (papers are SPY; index exposure identical, microstructure differs). LOW on Candidate 3 evidence._

## Strategic Synthesis, Roadmap & Risk Assessment

### Implementation Roadmap

| Phase | Action | Effort | Gate |
|---|---|---|---|
| 1 | Pre-register MIM-Classic (frozen spec, dev=2025, OOS=2026, costs $2.24/ct) | 1 session | seal commit before any test |
| 2 | Gate 0 dev backtest on mnq_1min_2025 | 1 session | N≥200, net PF≥1.10, net exp>0, cost≤25% avg gross win |
| 3 | One-shot OOS on mnq_1min_2026_ytd | minutes | net PF≥1.05, N≥80 |
| 4 | Combine MC with corrected rules (EOD-ratchet MLL, DLL day-deactivation, consistency check) | 1 session | pass% ≥ 50% at ≤10 MNQ, pass>blow |
| 5 | On pass: deployment pre-registration + paper bot; on fail: Candidate 2 (Noise-Bands), then Candidate 3 (HTF-MR) | — | per house Epic 8 workflow |

### Risk Assessment

- **Transferability risk (main):** published results are SPY 1993–2013 / 2007–2024; MNQ microstructure and the post-publication era may have arbitraged the effect. Mitigation: our 2025–26 data IS the post-publication test; Gate 0 answers this directly.
- **Crowding/decay:** MIM is widely known since 2018. The QuantConnect replication and 2024 Zarattini results suggest persistence, but expect a thinner edge than the papers report.
- **Sample-size risk:** ~250 trades/year for MIM-Classic is adequate for Gate 0 but CI on net PF will be wide; the MC gate (not the PF point estimate) is the deploy decision.
- **Methodology risk:** three candidates = multiple comparisons. Protections: fixed test order, one-shot OOS, global stop after Candidate 3.
- **Live calibration gap:** $2.24 RT assumes 1-tick slippage; thin last-half-hour liquidity could widen this. The S26 live bot's fill data provides a free empirical slippage estimate before deployment.

### Future Outlook

If MIM-Classic passes all gates, natural extensions (each requiring its own pre-registration): volatility/volume conditioning (the papers show the effect concentrates on high-vol days), the pre-FOMC overlay, and MES portfolio sizing for smoother MC profiles. If all three candidates fail, the honest conclusion is that this instrument set offers no documented intraday edge above the cost floor at our data granularity — and the correct move is waiting for the S25 live verdict (~2026-07-23) and the BTC-CARRY funding trigger rather than buying combine attempts.

## Methodology and Source Verification

All combine rules and fees: official Topstep help-center pages (HIGH confidence, fetched 2026-06-10/11). Strategy evidence: peer-reviewed journals (JFE, NY Fed staff reports, ECB working papers, SSRN/SFI preprints) cross-checked against independent replications where available. Practitioner sources (prop-firm blogs, strategy sites) used only for rule-interaction consensus and explicitly flagged. Internal evidence: 17 pre-registered failures in this repository's audit trail, the 2026-06-10 S26 walk-forward + combine Monte Carlo, and the sealed S26-HTF Gate 0 (commits 4d43200, 3766d9f). Full source URLs are cited inline per section.

