# Innovation Strategy: Alex's MNQ Trading Program — "POTUS Alpha" Edge Assessment

**Date:** 2026-07-03
**Strategist:** Alex
**Strategic Focus:** Is there a real, capturable edge in Trump/Truth Social-driven market moves for a solo retail-latency operator — and if so, what form must it take to survive this program's validation gates?

> **Session mode:** Run autonomously (YOLO) at Alex's request ("I don't have time to view and make these type of assessments"). All checkpoints collapsed. Every material claim below comes from live web research performed 2026-07-03 (three parallel research sweeps: second-term event record, academic/latency literature, regime + competitive landscape); items that could not be verified are flagged inline.

---

## 🎯 Strategic Context

### Current Situation

**The "company":** a one-person systematic futures trading program, currently running a real-money Topstep 50K combine (acct 23884932) with two validated, effectively uncorrelated MNQ edges — YANK (bearish FVG + H1 sweep + ML filter, 2ct) and MIM-NB (1ct) — plus GAP-1 panic-open fade in TS SIM paper (STRONG_EDGE backtest, PF 1.761, N=117, promotion review due now), and a Thursday crypto short entering live paper. Infrastructure: TradeStation + ProjectX REST polling at **60-second bar granularity**, Python research stack, hash-chained logging, and a hard-won pre-registration methodology (Gate 0 → OOS → prospective N≥30) that exists precisely because this program spent months manufacturing fake edges before learning to kill them.

**The proposed idea:** exploit the observation that President Trump's Truth Social posts and public statements repeatedly move markets — sometimes hours before the "official" announcement — and that people who watched the right feed (e.g., oil positioning around the Iran conflicts) captured enormous moves. Alex asks: can I capture the "time delta between news coming out and the market moving"?

### Strategic Challenge

Three questions, in order:

1. **Does the phenomenon exist?** (Do Trump posts move markets in a way that is large, repeated, and directionally readable?)
2. **Is any of it left on the table at Alex's latency?** (60-second polling, no colocation, no feed contracts, one human asleep 8 hours a day.)
3. **Can whatever survives (1) and (2) pass this program's validation gates** — backtestable pre-registerable rule, adequate N, costs-in net PF > 1 — within a sample the calendar can deliver?

Graveyard warning loaded up front: the MNQ event-fade scout (2026-06) found a real-looking FOMC-fade lead and was **PARKED anyway** because ~34 events/year makes validation take too long. Any "Trump post" strategy faces related event-frequency arithmetic — plus an adversary (the poster) who is himself adaptive and now personally trades ahead of his own news.

---

## 📊 MARKET ANALYSIS

*(Frameworks applied: PESTLE-style event record, Market Timing Assessment, Competitive Positioning Map, Five Forces. Unflinching clarity about who already occupies this space must precede any innovation talk — this is the most crowded, most surveilled information channel in world markets.)*

### Market Landscape

**1. The phenomenon is real, large, and ongoing — Question 1 answers YES.**

Verified event record:

- **2025-04-09:** Trump posts "THIS IS A GREAT TIME TO BUY!!! DJT" at 9:37 AM ET; at 1:18 PM ET he announces the 90-day tariff pause. S&P 500 **+9.52%** (best day since Oct 2008), Nasdaq **+12.16%**, ~$4T of market cap regained. DJT stock +22.7%. A block of 5,105 SPY 0DTE $504 calls bought 18 minutes before the announcement turned ~$2.1M into >$30M within 24h (Unusual Whales estimate, unaudited). Six senators demanded an SEC probe; **no enforcement outcome as of July 2026**.
- **2025 tariff cycle:** Liberation Day (Apr 2) erased ~$4T over four sessions; Powell "termination" posts (Apr 17–21) knocked the Dow −971 in a day and the walk-back rallied it back; EU 50% tariff post (May 23) sank futures, its delay (May 25) produced S&P +2.1%; China 100% tariff post (Oct 10) erased ~$2T in one day, and a conciliatory Sunday post ("President Xi just had a bad moment") reversed it.
- **2026 Iran cycle:** posts became the *dominant* intraday driver. A 7:05 AM ET post halting planned strikes (Mar 23) moved S&P futures +2.5% instantly and WTI −6%; an April post postponing strikes produced a **$3 trillion S&P market-cap swing in 56 minutes** (Axios); a June "cancelled the scheduled strikes" post dropped WTI 2.6% in minutes; CNBC counted **30+ separate "Iran deal is close" signals, each still moving markets**.
- **June 2025 12-Day War oil round-trip** (the episode Alex cited): crude +7-13% intraday on the June 13 Israeli strikes, ~$80 peak after US strikes on Fordow, then Trump's verbatim June 23 post — "EVERYONE, KEEP OIL PRICES DOWN. I'M WATCHING!" — followed by his ceasefire announcement; WTI −7% then −6% on consecutive days, ending *below* pre-war price within 8 trading days.

**2. "How tall is the market" vs the last administrations — the regime is categorically different from Trump I.**

| Dimension | Trump I start (2017) | Now (July 2026) |
|---|---|---|
| S&P 500 level | ~2,260 | **7,483** (Jul 2) — ~3.3× |
| Shiller CAPE | ~28–33 through 2017–19 | **40.4** — 2nd highest in 140+ yrs (only Dec 1999 higher) |
| Forward P/E | ~17–18 | ~20.4 (vs 10-yr avg 19.0) — earnings expectations doing the heavy lifting |
| Buffett indicator (mkt cap/GDP) | ~130–140% | **~235%** — 2nd highest ever recorded |
| Concentration | top-10 ≈ 20%, FAANG ~13% | **Mag7 ≈ 33–34%, top-10 ≈ 40%+**; AI/tech ≈ half of index cap |
| Volatility | 2017 = calmest year in decades (VIX avg ~11; 52 of history's 68 sub-10 closes) | Episodic-explosive: VIX 60 (Apr 2025), war shock (Mar 2026, S&P −8%, Brent >$119), but mean-reverts to ~16–17 within weeks; VIX ~16.4 in June 2026 |

Interpretation for strategy: the market is **taller (valuation), narrower (concentration), and twitchier (episodic vol)** than in Trump I. A single post can move more dollars faster than at any point in history — April 2025's VIX round-trip took 5 days up and 14 days down. High CAPE + record concentration means shock *amplitude* is bigger; fast mean-reversion means shock *persistence* is shorter. Both properties matter for what's actually harvestable (see below). *(Unverified: exact counts of >1% daily moves by year; better computed from your own 1-min data than from the web.)*

### Competitive Dynamics

*(Five Forces, compressed to the two that decide everything: rivalry and the resource you don't have.)*

The "time delta" Alex wants to capture is a **three-tier latency market**, and every tier is occupied:

- **Tier 0 — before the post (the actual winners):** The documented life-changing profits came from positioning *before* the news existed. CFTC (confirmed probe, Bloomberg 2026-04-15) and DOJ (Forbes 2026-05-07) are investigating ~$500M and ~$950M futures bets placed ~15 minutes before Trump's March 23 and April 7, 2026 Iran posts (~$2.6B combined profit across four trades). A QUT academic study of 1,341 posts over 73 days found **15 anomalous pre-post positioning events**, one ~$920M oil bet placed 4+ hours early. Israel criminally charged two people (one IDF reservist with clearances) for Polymarket bets on the June 2025 strikes. **This tier is information access — leaks — not strategy. It is not replicable, and attempting to replicate its returns from public data is chasing a mirage. This is the honest answer to "how do I do the same": you don't; the people who did are under federal investigation or indicted.**
- **Tier 1 — the first seconds (machines):** Machine-readable news is priced within **~5 milliseconds to 5 seconds** (Chordia et al.; Fed IFDP 1233: RavenPack tags stories in ~300ms and the response is front-loaded into the first 5 seconds — "a window in which only machines can trade"). JPMorgan's Volfefe work documented HFTs retooling to parse Trump posts **by 2019**. Second-term confirmation: the Feb 1, 2025 tariff post moved SPY futures −1.8% within ~90 seconds. Truth Social has **no public API** — institutional players run sub-second scrapers; the retail alert stack (TruthPing, SentryDock, TrumpBot, TruthSignal — a whole cottage industry, $29/mo) delivers in 5–45 seconds, and its own builders report "by the time all that happens, the market has already moved."
- **Tier 2 — minutes to days after (the residual):** This is the only tier where evidence of *systematically* capturable structure exists, and it's the only tier reachable at 60-second polling. The literature: initial impulses on *non-informational* posts revert (first-term company-tweets averaged ~25 bps day-0 and lost significance within 5 days; false-positive machine reactions correct in ~2 minutes; Chan 2003 no-news spikes revert), while *confirmed policy actions* drift (post-FOMC "monetary momentum" runs ~15 days; PEAD analog). The **TACO trade** (coined by FT's Robert Armstrong, May 2025 — threat→selloff→retreat→rally) was the year's dominant pattern: Dow Jones Market Data attributed **9 of the top-10 S&P sessions (Jan 2025–Apr 2026) to TACO-style de-escalations**. But it is now named, mainstream (Trump was asked about it at a presser), and visibly decaying — threat selloffs got shallower through late 2025 as it was priced in, and Fortune argued the muted reactions were emboldening actual tariff implementation, i.e., the pattern erodes its own base rate.

**Supplier power footnote:** the entire signal supply is one adaptive man who now (a) knows the market watches him, (b) per Mediaite's disclosure reporting, made 327 stock purchases the day before his own tariff pause, and (c) has demonstrated willingness to post the *opposite* of subsequent action. A strategy whose signal generator is an adversary aware of the strategy class has non-stationarity as a design feature.

### Market Opportunities

What survives contact with the evidence — ranked by fit to a 60s-latency, validation-gated solo program:

1. **Aftermath structure, not the news itself.** Detect the *market's own reaction* (an anomalous 1-minute impulse bar cluster outside scheduled macro windows) and trade the ensuing drift/reversion at minutes-to-hours horizon. This needs no Truth Social feed, no NLP, no latency race — the 60s bar feed IS the detector, and 18 months of owned 1-min MNQ data can backtest it today. The regime evidence (bigger amplitude, fast mean-reversion) and the literature (no-news spikes revert; policy-confirmed moves drift) both say the aftermath has structure.
2. **Event-risk throttling of existing edges.** The cheapest use of this whole research: YANK/MIM/GAP-1 already trade through post-driven chop. Prediction markets (Kalshi+Polymarket, now ~$21B/mo volume) price geopolitical event risk in real time and were at 99% before the June 2025 strikes. A daily "policy-shock risk" state (prediction-market odds moving + scheduled deadlines like tariff effective dates) could gate position size on the *existing* portfolio — defense, not offense.
3. **Explicitly NOT opportunities:** copy-trading products (Autopilot's Pelosi tracker holds ~$400M copying disclosures that lag 45 days — a marketing product, not an edge); post-alert subscriptions (structurally late); politically-branded funds (Azoria's ETFs were voted shut; MAGA ETF lags); anything requiring discretionary real-time reading.

### Critical Insights

1. **The premise contains its own refutation.** The reason the April 9 options buyer and the March 2026 oil shorts made fortunes is that they acted *before* the public information existed. The "time delta between news coming out and the market moving" is now 90 seconds or less for the impulse; the profitable delta was between *decision* and *announcement* — a delta only leaks can cross. Federal probes, not brokerage accounts, are where that trail ends.
2. **Retail post-chasing is a documented loser's game with a vendor ecosystem monetizing the fantasy.** Zero documented cases of persistent retail profitability at seconds-to-minutes news latency; day-trader base rates run 97–99% losers; the alert vendors' own build logs admit the move precedes execution.
3. **What's actually left is boring and familiar:** conditional drift/reversion at 1-minute-bar resolution — exactly the class of edge this program already knows how to find, gate, and kill. The innovation is *not* a news strategy; it's a **shock-aftermath strategy that is agnostic to what caused the shock**.
4. **The regime comparison matters for the existing book, not just the new idea:** a market at CAPE 40 with 33% Mag7 concentration and a president who moves $3T in 56 minutes is a fat-tail regime for ALL MNQ strategies. The defensive application (opportunity 2) may be worth more than the offensive one.

---

## 💼 BUSINESS MODEL ANALYSIS

*(Frameworks: Value Proposition Canvas, Cost Structure Innovation — assessed against the current program rather than a hypothetical startup.)*

### Current Business Model

Value creation = statistical edges on MNQ microstructure patterns, harvested at 1-minute resolution with strict risk gates; value capture = funded-account evaluation (Topstep combine → funded payouts) rather than personal capital at risk. Key resources: the validation pipeline itself (prereg discipline, backtest engines, live shadow/parity logging), ~18 months of accumulated 1-min data across MNQ/metals/crypto, and running live infrastructure with auto-roll, floor monitoring, and SIM mirroring. Cost structure: near-zero marginal cost per strategy; the scarce resources are **calendar time for prospective validation** and **Alex's attention**.

### Value Proposition Assessment

The program's honest, demonstrated competence is: *finding small, structural, high-frequency-enough edges in bar data and refusing to deploy anything that can't prove itself.* It has **no demonstrated competence** in: discretionary news interpretation, low-latency execution, event-driven options structures, or NLP signal extraction. Any Trump-news strategy depending on those is a capability acquisition, not an extension — and the market analysis shows the capability tiers that pay (Tier 0/1) are respectively illegal-to-access and capital-prohibitive.

### Revenue and Cost Structure

Current live exposure is bounded by combine risk math (trailing MLL, floor-monitor halts). A news-reaction strategy has a cost problem this program has already quantified: MNQ round-trip ~$6/ct against realized signal capture that historically ran ~2% of gross move (edge-headroom screen, 2026-06-15). Event strategies concentrate P&L into a handful of fat-tail days — the profile MIM-NB's Monte Carlo showed is an acceptable *option* but a fragile *income engine*.

### Business Model Weaknesses (relevant to this idea)

1. **Latency floor:** 60-second polling structurally excludes the first move after any public post. Not tunable with effort — an infrastructure class difference.
2. **Validation clock vs event frequency:** market-moving posts of tradeable magnitude cluster in bursts (30+ Iran signals in months) but the *strategy-grade* event set is smaller and non-stationary by construction (one man's posting behavior, actively gamed).
3. **Single-operator attention:** Alex explicitly does not have time to watch feeds. Anything requiring discretionary real-time reading is dead on arrival — fully mechanical or it doesn't exist.

---

## ⚡ DISRUPTION OPPORTUNITIES

*(Frameworks: Jobs to be Done, Disruptive Innovation Theory. What makes disruption different from incremental innovation here: the disruptive move is to REFUSE the arms race everyone else is fighting — latency — and compete on a dimension the latency players ignore: the aftermath.)*

### Disruption Vectors

- **The latency race is a sustaining-innovation trap:** every dollar spent on faster Truth Social scraping competes head-on with sub-second institutional infrastructure on their terms. The disruptive position is the segment they *abandon*: minutes-to-hours-later structure, too slow for HFT to care, too systematic for discretionary traders to harvest consistently.
- **Shock-agnosticism as moat:** a detector keyed to the market's own footprint (impulse bar signature) rather than the news source is immune to the source changing (Trump stops posting; the next administration; a different shock generator). It converts a personality-dependent trade into a structural one.
- **Defense as disruption:** nobody sells "stand down when the president is about to move $3T"; the prediction-market state signal is public, cheap, and unused at retail as a *risk throttle* rather than a *bet*.

### Unmet Customer Jobs

The customer is Alex's own portfolio. Jobs not currently done: (1) protect YANK/MIM/GAP-1 from policy-shock days they were never validated on; (2) monetize post-shock dislocations without watching any feed; (3) convert "I saw the news too late" regret into a rule that doesn't need the news at all.

### Technology Enablers

Already owned: 1-min bar feeds (TS + ProjectX parity-logged), 18 months of 1-min MNQ history including the entire second-term shock record (Apr 2025 tariffs, Oct 2025 China post, Mar–Jun 2026 Iran cycle) — a natural event-study dataset sitting in `data/processed/`. Cheap to add: Kalshi/Polymarket public APIs for event-odds state; the economic calendar filter already exists conceptually in S28 (news calendar filter, queued).

### Strategic White Space

**Post-shock microstructure on index futures at the 15–240 minute horizon, conditioned on shock type (no-scheduled-news impulse vs scheduled-macro impulse), sized by combine math.** The academic literature says structure exists there (drift after informational shocks, reversion after non-informational ones); the HFT crowd has extracted the first seconds and left; the discretionary crowd (TACO traders) operates at daily horizon on narrative, not mechanically. Nobody in the retail product landscape occupies this shelf.

---

## 🚀 INNOVATION OPPORTUNITIES

*(Frameworks: Three Horizons, Innovation Ambition Matrix. Multiple paths explored before committing — but filtered hard by the program's real constraints.)*

### Innovation Initiatives (candidate set, pre-filter)

1. **SHOCK-AFTERMATH (H2, adjacent):** mechanical impulse-bar detector on MNQ 1-min (e.g., k-sigma 1-min range/volume vs rolling baseline, outside scheduled macro release windows) → trade drift or fade at fixed horizon. Backtestable TODAY on owned data. → *survives filter.*
2. **EVENT-RISK THROTTLE (H1, core enhancement):** daily policy-shock risk state from prediction-market odds + tariff/deadline calendar → position-size multiplier or stand-down flag for existing bots. → *survives filter.*
3. **TRUTH SOCIAL SCRAPER + NLP FOLLOW-BOT (H3, transformational):** rejected — competes at Tier 1 against sub-second infrastructure with a 60s feed; vendor ecosystem's own evidence says the move is gone; no backtest possible without a post archive + tick data. → *killed.*
4. **TACO SYSTEMATIC (buy threat-dips, sell walk-back rallies at daily horizon):** the pattern that "9 of the top-10 S&P sessions" rode — but named, crowded, decaying, with N≈10–15 clean cycles total and a self-eroding base rate. Unpre-registerable as a fresh edge; its *content* is partially absorbed into #1 (informational vs non-informational shock classification). → *killed as standalone.*
5. **OIL/ENERGY SIDECAR (trade CL/MCL around geopolitical shocks):** the June 2025 and 2026 oil round-trips were the fattest moves in the record — but new instrument, new cost structure, event N tiny, and the program's metals experience (fat ticks, slippage FAILs) argues against. → *parked, revisit only if #1 validates on MNQ.*
6. **PREDICTION-MARKET DIRECT BETS:** Kalshi/Kalshi-style event positions as the expression vehicle. Regulatorily live, but no infrastructure, no validation dataset, and Polymarket's insider record shows the same Tier-0 problem. → *killed.*

### Business Model Innovation

None required — that is the point. Options 1–2 monetize through the existing combine/funded-account vehicle, reuse the existing prereg pipeline, and cost approximately zero marginal infrastructure. The innovation is a *refusal* to change business model in pursuit of a glamour trade.

### Value Chain Opportunities

The one genuine value-chain addition: a **shock-event dataset** built from owned 1-min bars (timestamped impulse events 2025-01 → present, labeled with what caused them post-hoc). This asset compounds: it feeds #1's Gate 0, S28's calendar filter, and any future event-conditioned research. Build it once, own it.

### Partnership and Ecosystem Plays

Not applicable at this scale, with one small exception: Kalshi/Polymarket public data APIs as free "partners" for the risk-state signal. No paid vendor subscriptions justified by the evidence (the alert products sell access to a window that doesn't pay).

---

## 🎲 STRATEGIC OPTIONS

### Option A: Chase the Post (Latency Play)

Build a Truth Social scraper + LLM classifier + auto-order path; trade MNQ in the direction of the post within ~30–60 seconds.

**Pros:** Directly answers the original ask; emotionally satisfying; scraper is buildable in a weekend; the moves are genuinely enormous.

**Cons:** Structurally lost before it starts — machines price posts in ≤5 seconds and even 90-second impulses complete before a 60s poller can act; the documented winners acted *before* posts (leaks, now under CFTC/DOJ probe) — that alpha is criminal, not clever; no backtest possible (no tick-aligned post archive), so it can never pass Gate 0; vendor ecosystem's own builders admit the window is gone at retail latency; signal source is one adaptive adversary; zero documented retail profitability in this class, against 97–99% day-trader loss base rates.

### Option B: Trade the Aftermath (Shock-Agnostic Impulse Strategy)

Mechanical detector on owned 1-min MNQ data: anomalous impulse bars (range/volume k-sigma vs rolling baseline) outside scheduled macro windows → pre-registered rule for fade or follow at 15–240 min horizon, possibly conditioned on impulse direction, time-of-day, and whether a scheduled event explains it. Gate 0 on 2025–2026 history (which contains the entire tariff/Iran shock record), then standard OOS → prospective pipeline. Trades MNQ, sizes by combine math, needs no feed, no NLP, no watching.

**Pros:** Only option reachable at 60s latency; backtestable *today* on owned data — full Gate 0 verdict cheap and fast; literature-supported (no-news reversion, informational drift, monetary-momentum analogs); shock-agnostic — survives Trump leaving the stage; produces the reusable shock-event dataset as a byproduct; fits every existing pipeline convention (prereg, YAML config, floor-monitor coexistence).

**Cons:** It is NOT the glamour trade Alex asked about — expected value per event is bps-scale structure, not 2,100% option wins; event N may be thin after conditioning (the FOMC-scout failure mode) — Gate 0 may simply return FAIL; shock regimes are non-stationary (a 2025-shock-fitted rule may die in a calm 2027); overlaps partially with GAP-1's panic-fade turf — correlation with the existing book must be checked before any deployment.

### Option C: Event-Risk Throttle (Defensive Overlay)

No new trading strategy. Build a daily "policy-shock risk state" (prediction-market odds velocity on geopolitical/tariff markets + deadline calendar) that throttles size or stands down YANK/MIM/GAP-1 on flagged days. Validate by replaying existing bots' trade history against the flag: did flagged days have worse realized PF?

**Pros:** Cheapest possible build; protects the REAL money (combine account) in exactly the regime documented above (CAPE 40, Mag7 33%, $3T/56min swings); testable retrospectively against months of owned live trade logs; no new market risk taken; complements rather than competes with the existing book; aligns with S28 (news calendar filter) already in the queue.

**Cons:** Generates zero new revenue — pure defense; risk of over-filtering (YANK/MIM validation didn't exclude shock days, so excluding them changes the validated system — requires its own prereg discipline); prediction-market odds are themselves contaminated by insiders (that's *why* they lead), which is fine for defense but means the flag will sometimes fire on false alarms; adds an operational dependency on external APIs.

---

## 🏆 RECOMMENDED STRATEGY

### Strategic Direction

**Reject A. Run B as a scout-tier Gate 0 experiment. Adopt C as the near-term deliverable.** And explicitly: do not let any of this displace the two commitments already on the calendar — the GAP-1 promotion review (due now, with the direction-split check) and the S25 decision rule. This idea enters the research queue behind them, not ahead of them.

Why this direction: the evidence is unambiguous that the phenomenon Alex observed is real — and equally unambiguous about *where the money was made*: before the posts (leaks; under federal investigation) and in the first seconds (machines). Neither tier is accessible, one is illegal. The honest residual is aftermath structure at bar resolution — which happens to be the exact game this program already plays well. What makes me confident: Option B's Gate 0 costs almost nothing and its dataset already sits on disk. What scares me: Option B's most likely Gate 0 outcome is FAIL or MARGINAL (the FOMC-scout precedent), and the psychological pull of the glamour version (A) if B disappoints — the April 9 options story is a lottery-winner story, and lottery-winner stories are how this program's graveyard got populated.

**Direct answers to Alex's original questions, on the record:**
- *"How can I do the same as the oil traders?"* — You can't, and shouldn't want to: the documented pre-post positioners are the subject of CFTC/DOJ probes and Israeli criminal charges. The *legal* version of "knowing first" is prediction-market odds — and their best use at your scale is defense (Option C), not offense.
- *"There's a time delta between news and the market moving."* — There was; it is now ≤90 seconds for the impulse and ≤5 seconds for the machine-readable core. Your exploitable delta is *after* the impulse: minutes-to-days drift/reversion, which is Option B.
- *"How tall is the market / regime vs last administration?"* — S&P ~3.3× the Jan-2017 level; CAPE 40.4 (2nd highest ever); Buffett indicator ~235%; Mag7 ~33% of the index vs ~13% FAANG in 2017; vol regime episodic-explosive (VIX 60 spikes, war shocks) with fast reversion, vs 2017's uniform calm. Practical meaning: bigger shock amplitude, shorter shock persistence, and more of your existing book's risk concentrated in policy-shock days — which is the argument for Option C.

### Key Hypotheses to Validate

1. **H-B1 (Gate 0):** Anomalous no-scheduled-news impulse bars on MNQ 1-min exhibit exploitable drift or reversion at 15–240 min horizon, net of $6/ct RT costs, on 2025–2026 data — with the rule (fade vs follow, horizon, threshold) chosen by sweep on the IS split ONLY (derive-don't-assert) and confirmed on the held-out split.
2. **H-B2:** Any such edge is not merely GAP-1 in disguise — correlation of event days and P&L against GAP-1's backtest must be measured before deployment consideration.
3. **H-C1:** Flagged high-policy-risk days (prediction-market state) had materially worse realized PF for YANK/MIM live/backtest history than unflagged days. If not — C dies quietly and cheaply.
4. **H-Kill conditions:** B dies if Gate 0 net PF < the program's standard bar, if N(events) after conditioning < the point where MC gives a stable verdict, or if the edge concentrates in fewer than ~3 fat days (the MIM-NB fragility profile without MIM-NB's validation).

### Critical Success Factors

- **Prereg before peeking:** the shock-event dataset build and the IS/OOS split must be sealed before any rule sweep — this idea is maximally exposed to narrative-driven overfitting (everyone "knows" the April 9 story).
- **Mechanical definition of "shock"** that references only bar data and the scheduled-release calendar — never the news content. The moment a Trump-specific classifier enters the rule, the edge inherits his non-stationarity.
- **Attention budget honesty:** both B and C must run unattended end-to-end. Any design requiring Alex to read a post is a design failure per his own constraint.
- **Combine-math compatibility:** any deployment sizing must clear the floor-monitor / trailing-MLL arithmetic that governs the real account.

---

## 📋 EXECUTION ROADMAP

*(Phased by dependency and decision gates, not calendar promises.)*

### Phase 1: Immediate Impact

- Finish the two commitments already due: GAP-1 promotion review (incl. the pre-registered direction-split check) and continued S25 accrual. **This strategy's first deliverable is not being a distraction.**
- Build the **shock-event dataset**: scan owned 1-min MNQ (and stored quotes) 2025-01→present for k-sigma impulse bars; label each with scheduled-release explanation (yes/no) post-hoc; store as a versioned CSV in `data/`. No trading logic yet — pure asset build.
- Draft and **seal the Option B prereg** (hypothesis H-B1, IS/OOS split, sweep space, cost model, kill conditions) via `prereg_seal.py` before any rule evaluation.
- Run the **Option C retrospective**: replay existing trade logs against a simple risk-flag reconstruction; verdict on H-C1.

### Phase 2: Foundation Building

- Execute Option B Gate 0 per the sealed prereg: sweep on IS, confirm on OOS split, publish verdict doc (PASS / MARGINAL / FAIL) alongside the existing verdict corpus.
- If H-C1 passed: implement the live risk-state flag (Kalshi/Polymarket odds poller + deadline calendar), wire as a *logging-only* shadow first (the program's shadow-first migration pattern), then as a size throttle behind its own prereg.
- Check H-B2 (GAP-1 overlap) using both backtests' event calendars.

### Phase 3: Scale & Optimization

- Only if B passes Gate 0 + OOS: paper-deploy on TS SIM (the GAP-1 route: SIM first, realized-capture judged against modeled), prospective N accrual under a sealed decision rule before any combine wiring.
- Only if B validates on MNQ: revisit the parked oil sidecar question with the same detector logic on CL/MCL data — as a *new* prereg, never a transplant (the dollar-threshold-transplant lesson from S26 is on file).
- Fold the shock-event dataset into S28 (news calendar filter) research when the S25 queue unblocks.

---

## 📈 SUCCESS METRICS

### Leading Indicators

- Shock-event dataset built, event counts by month/type published (is N even viable? — this number alone may kill B before any sweep).
- Prereg seals committed *before* first sweep runs (process compliance is itself the leading indicator this program lives or dies by).
- Option C retrospective delta: flagged-day PF vs unflagged-day PF on existing book.

### Lagging Indicators

- Option B Gate 0 / OOS verdicts (net PF vs program bar, MC pass rate, fat-day concentration).
- If deployed to SIM: realized capture vs modeled ledger (the GAP-1 judgment standard), prospective PF under sealed decision rule.
- Portfolio-level: change in worst-day / max-giveback statistics on the combine account after any throttle deployment.

### Decision Gates

1. **Dataset gate:** N(qualifying no-news shocks) below MC-stability threshold → B dies at zero cost.
2. **Gate 0:** standard program bar, sealed in the prereg → FAIL kills, MARGINAL parks.
3. **OOS gate** via `oos_checkpoint.py` before any holdout access.
4. **SIM gate:** realized-vs-modeled capture ratio must not repeat the ~2% capture failure documented in the graveyard.
5. **Combine gate:** fresh prereg + floor-monitor math + 3-bot correlation check (the exact gate already defined for GAP-1 promotion).

---

## ⚠️ RISKS AND MITIGATION

### Key Risks

1. **Narrative seduction / graveyard recidivism:** the April 9 story is a lottery narrative; fitting rules to 2025's most famous days is the iteration-loop failure pattern with better production values.
2. **Non-stationary signal source:** the shock generator is one man who adapts, is himself accused of trading his own news, and leaves office; a Trump-fitted edge has a hard expiry.
3. **Thin/concentrated N:** conditioning may leave too few events for inference (FOMC-scout failure mode), or an "edge" that is 3 fat days in a costume (MIM-NB fragility profile).
4. **Overlap cannibalism:** B may be GAP-1 or MIM-NB wearing a mask — deploying correlated strategies into one trailing-MLL account raises blow risk nonlinearly (the doubling analysis on file: 17%→33%).
5. **Legal/venue-adjacent risk:** none for B/C as designed (public data, own account) — but any drift toward "get the information earlier" ideas re-enters the territory currently occupied by federal probes. Bright line: no strategy may depend on information timing advantages over the public record.
6. **Defensive overlay corrupting validated systems:** C changes the effective validated config of YANK/MIM if applied carelessly (silent-direction-gate lesson, S26).

### Mitigation Strategies

1. Prereg-before-peek, derive-don't-assert, one-knob-at-a-time — all already codified in this program; apply without exception. The IS split must exclude the "famous" days from rule *selection* narratives by letting the sweep, not the story, choose parameters.
2. Shock-agnostic mechanical definition (bar signature + release calendar only) — the mitigation IS the design; Trump-specific features are banned from the rule.
3. Dataset gate before any modeling; MC stability requirement; fat-day concentration check copied from the MIM-NB reconciliation standard.
4. H-B2 correlation check mandatory pre-deployment; portfolio MC including the new leg before any combine wiring.
5. Bright-line rule stated in the prereg itself: public-data-only, no expression via prediction-market bets on government action.
6. C ships shadow-first (logging-only), then behind its own prereg as a sized throttle with explicit fire/no-fire audit trail — the ProjectX-migration playbook reused.

---

_Generated using BMAD Creative Intelligence Suite — Innovation Strategy Workflow (autonomous YOLO run, research-grounded, 2026-07-03)_
