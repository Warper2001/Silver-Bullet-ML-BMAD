---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments: []
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Best crypto-native strategy for the Kraken Pro perpetual futures environment'
research_goals: 'Identify untried crypto-native strategy families and recommend the single best candidate for our venue (Kraken Pro Futures) and methodology discipline'
user_name: 'Alex'
date: '2026-06-20'
web_research_enabled: true
source_verification: true
---

# Research Report: technical

**Date:** 2026-06-20
**Author:** Alex
**Research Type:** technical

---

## Research Overview

This report identifies the single best crypto-native trading strategy to pursue for the Kraken Pro perpetual-futures environment, given our data (BTC+ETH 1-min + BTC funding history, 18 months, single-regime bear), venue constraints (60s poll, ~10 bps RT cost), and methodology discipline. Methodology: venue-mechanics research (web-verified against Kraken official docs + a live `/instruments` query) → parallel evidence review of 13 crypto-native strategy families across three independent research threads → screening on three gates (mechanism distinct from our dead strategies × 60s-capturable × ≥30 validatable events/yr) plus a net-of-cost survivability screen.

**Key finding:** the venue prunes the field to two survivors. It structurally favors funding/basis edges and rules out liquidation-cascade (no liquidation feed) and options/vol (no options on Kraken) strategies. The recommended candidate is **ETH session-momentum amplitude (S1)** — not the best-evidenced option, but the only one our power-starved methodology can validate cleanly and fast (~250 events/yr), and one that is orthogonal to our incumbent BTC-CARRY sleeve. Runner-up is **ETH funding-extreme reversal (S2)** (best evidence, but fat-tail/low-N and carry-correlated). The Gate-0 condition before any capital is reproducing S1's edge at our real 10 bps cost on our own ETH data.

See the **Executive Summary & Ranked Recommendation** section at the end for the full ranking, kill-criteria, and next step.

---

<!-- Content will be appended sequentially through research workflow steps -->

## Technical Research Scope Confirmation

**Research Topic:** Best crypto-native strategy for the Kraken Pro perpetual futures environment
**Research Goals:** Identify untried crypto-native strategy families and recommend the single best candidate for our venue (Kraken Pro Futures) and methodology discipline

**Scope (adapted to trading-strategy domain):** venue mechanics → candidate strategy families (crypto-native) → edge mechanism & evidence → fit to our constraints (60s poll, single-regime data, cost floor) → methodology fit (OOS power, pre-registration feasibility).

**Guardrails:** DEAD families stay dead (FRRF, TSMOM/-RF, BTC-ETH cointegration, SMC/FVG transplant). BTC-CARRY is the incumbent winner (not a competitor). Any candidate that cannot generate ≥30 independent OOS events/yr is flagged power-starved up front.

**Scope Confirmed:** 2026-06-20

---

## Venue Mechanics — Kraken Pro Futures (the "stack")

> Research date 2026-06-20. All numeric claims web-verified; sources inline. Our data file `data/kraken/PF_XBTUSD_1min.csv` = the **linear, USD/multi-collateral-margined** global-venue contract (`PF_` prefix), NOT the inverse coin-margined `PI_` contract.

### Fees (cost floor)
Tiered by 30-day volume; same schedule for all perps (no BTC vs ETH difference).
- **Base tier: 2.0 bps maker / 5.0 bps taker.** Taker round-trip ≈ **10 bps** if crossing the spread — this is the dominant cost term and worse than the ~1–2 bps maker tiers Binance-based research assumes.
- Maker turns negative (rebate) only at ≥$250M/30d volume — irrelevant for us.
- _Source: https://support.kraken.com/articles/360048917612-fee-schedule_ (re-verify live; promos + a stale "7.5bps taker/−2bps maker" figure create base-tier ambiguity)

### Funding mechanism (THE differentiator)
- **Settles every 1 hour**, accrues **continuously** as unrealized P&L, realizes hourly. Cannot dodge by flattening before a mark.
- Realization multiplier n=8; **cap ±0.5%/hour** on linear PF_ (widened from ±0.25% in 2025).
- Contrast: Binance/Bybit settle at discrete **8h** timestamps (00/08/16 UTC), capped ~±0.05–0.075%/8h. Kraken charges 8× more often with **far higher annualized headroom** → extreme-funding regimes behave very differently here.
- Historical funding queryable via `historicalfundingrates` endpoint.
- _Sources: https://support.kraken.com/articles/4844359082772-linear-multi-collateral-derivatives-contract-specifications ; https://docs.kraken.com/api/docs/futures-api/trading/historical-data/_

### Contract specs (linear PF_)
| | PF_XBTUSD | PF_ETHUSD |
|---|---|---|
| Type | Linear, USD/multi-collateral | Linear |
| Min order | 0.0001 BTC | 0.001 ETH |
| Tick | 1 USD | 0.1 USD |
| Max leverage | 50×–100× (notional/region-tiered; confirm live via `marginLevels`) | same |
| Initial margin | from 2%, tiered up by notional | same |
- _Source: same linear-contract spec page above_

### Liquidation engine
- Mark price = order-book mid **bounded to a band around the CME CF index** (anti-wick-hunt). Liquidations trigger on **mark**, not last.
- Process: IOC at near-zero-equity price → assignment to LPs → ADL → insurance fund.
- **No confirmed public liquidation-print feed** (no Binance-style `forceOrder` stream found; confidence LOW that one exists). **Implication: a liquidation-cascade strategy would have to INFER cascades from trades + OI deltas, not read a liquidation tape.**
- _Sources: https://www.kraken.com/learn/last-price-mark-price-futures ; https://support.kraken.com/articles/4402283092244-liquidation-faq-derivatives_

### API & market data
- REST public market-data endpoints are **unbudgeted/free** — a 60s poller hitting `/tickers` is essentially free and returns **funding rate, mark, index, last, open interest, best bid/ask** in one call. **OI is first-class.**
- `/history` token bucket is restrictive (100 tokens / 100 per 10min) — slow for bulk backfill.
- WebSocket v2: L2 depth + L3 (individual orders) + trades + ticker; FIX 4.4 available. 100 conns, 100 req/s/conn.
- No published latency figure; at 60s poll cadence latency is a non-issue except at order submission.
- _Sources: https://docs.kraken.com/api/docs/guides/futures-rate-limits/ ; https://docs.kraken.com/api/docs/futures-api/trading/historical-data/_

### Instruments
- Global venue: dozens of `PF_`/`PI_` perps + dated `FI_`/`FF_` futures across majors and altcoins (query `/instruments` for live count).
- **No crypto options** on Kraken (confidence MEDIUM-HIGH — none surfaced). → options/vol family is **off-venue**.
- A separate **US CFTC-regulated venue (Bitnomial, launched 2026-06-15)** has 9 markets and **8h funding / daily cash settlement** — DIFFERENT rules. Our `PF_` data is the **global** venue; don't conflate.

### Cross-exchange context
- Kraken is a **second-tier** perp venue (Binance ~36% share, Bybit/OKX ~21% each; Kraken not top-3). **Thinner book → wider spreads, more slippage on size.** A strategy tuned on Binance fill assumptions will overestimate fills.
- Quirks that break a Binance-developed strategy: hourly continuous funding; ±0.5%/hr cap; PF_ vs PI_ tick/margin trap; higher base taker fee; no liquidation feed; two regulatory venues.

**Venue verdict for strategy selection:** the venue is *structurally biased toward funding/basis strategies* (hourly funding, high cap, free OI+funding API) and *structurally hostile* to (a) liquidation-cascade strategies (no feed), (b) options/vol strategies (no options), and (c) any high-churn taker strategy (10 bps RT cost floor at 60s cadence).

---

## Strategy-Family Edge Analysis

Thirteen crypto-native candidate families researched across three parallel evidence reviews (funding/basis, OI/liquidation, ETH/cross-asset). Each graded on three gates: **(G1) mechanism distinct** from our dead strategies (FRRF, TSMOM/-RF, BTC-ETH cointegration, SMC transplant); **(G2) capturable at 60s poll cadence**; **(G3) ≥30 independent OOS events/yr** so it is validatable on our 18 months of single-regime bear data. Net-of-cost survivability at ~10 bps RT is the fourth, fatal screen.

### Survivors (pass all three gates)

#### S1 — ETH session-momentum amplitude *(best frequency; orthogonal to carry)*
- **Mechanism:** ETH runs ~1.5–2× BTC realized vol. A session-conditioned momentum pattern (one regional session's return predicts the next session's direction) produces a per-trade spread (~45 bps in ETH) large enough to clear costs, while the *identical pattern in BTC (~13 bps) loses money*. Session-conditioned amplitude, **not** continuous trend → mechanically distinct from our dead TSMOM.
- **Evidence:** ETH≈2× BTC vol + regional-open intraday seasonality is **peer-reviewed** (ScienceDirect S1059056024006506; Applied Economics 10.1080/00036846.2023.2212970; Shen/Urquhart/Wang 2022 *Financial Review* — BTC+Kraken session momentum). The *tradeable* session-momentum result is **one practitioner backtest** (Coinmonks: ETH EU→US +257%/4.7yr, Sharpe 0.808, MaxDD −64%, costed at **7 bps**).
- **Gates:** G1 ✅ distinct · G2 ✅ (~250 decisions/yr, one/day) · G3 ✅ easily (~375 events over 18mo)
- **Risk:** the backtest costed 7 bps < our 10 bps, and −64% DD. Genuine risk it dies net-of-cost — the same gross-edge-<-cost wall that killed our stat-arb and event-fade. But because it generates ~250 events/yr it will produce a **fast, statistically trustworthy** yes/no on our starved data.

#### S2 — ETH funding-extreme reversal *(best evidence; convergent pick)*
- **Mechanism:** Extreme positive funding = overcrowded leveraged longs → contrarian short the perp into the extreme expecting reversion via liquidation (symmetric on deeply negative funding). Takes directional risk — **distinct from our delta-neutral carry**.
- **Evidence (the convergent result — two independent subagents ranked this #1 on evidence):** peer-reviewed mechanism in **BIS WP 1087 "Crypto Carry" (Schmeling, Schrimpf, Todorov 2023 → *Management Science*):** *high crypto carry predicts future price crashes; +10% standardized carry → ~22% of OI in sell-liquidations over the next month.* ETH funding runs hotter than BTC (BitMEX Q3-2025: +floor 87.5% vs 78% of the time). **Caveat:** the BIS paper validates the *mechanism* (crowding→liquidation), not a tradeable reversal Sharpe — the reversal *rule itself* is practitioner-anecdotal (no clean ETH Sharpe/win-rate published). Directional R² of funding→forward return is tiny (~0.003, Fulgur).
- **Gates:** G1 ✅ distinct · G2 ✅ (oversamples — funding in `/tickers` every poll) · G3 ⚠️ marginal — ~30–40 events/yr only at a *loose* threshold; the genuine edge concentrates in the extremes you'd dilute to reach that count.
- **Risk:** (i) the **dilution tension** — validatable (loose) version and profitable (extreme) version may be different strategies; this is exactly the fat-tail-concentration trap our memory flags (MIM-NB "edge = 3 fat-tail days", ml_proba "fat-tail mirage"). (ii) **Correlated to our incumbent** — both funding-driven, both die when funding compresses (carry already went negative in 2025, and our 18-mo sample sits in that bear). Lower orthogonality value than S1.

### Rejected families (do not re-litigate)

| Family | Gate failed | Why |
|---|---|---|
| **Dated-futures basis / term structure** | Buildability | Subagent queried Kraken `/instruments` live: **0 dated `FI_/FF_` contracts** exist. Real dated-basis liquidity is on CME/Deribit, not Kraken. |
| **Funding/basis momentum** (trend-follow funding) | Evidence | Causality runs **price→funding**; funding is a *trailing* byproduct (Presto: forward R²≈0). |
| **Liquidation-cascade fade** | Evidence (G1 contaminated) | Most rigorous test (Binance 587 sym): "+299%/Sharpe 3.58" is **leveraged BTC beta**, alpha p=0.182 n.s., negative vol-controlled. Survives only on thin alts (ETH maybe, **BTC no**). |
| **Liquidation-cascade follow** | G2 + evidence | Latency-disadvantaged at 60s (you react after the print); literature shows reversion, not continuation. |
| **OI-divergence (standalone)** | Evidence | Folklore — no return/Sharpe/t-stat in any source; serious factor papers (Liu-Tsyvinski-Wu *JF* 2022) find size/momentum/reversal survive, not OI. Fold OI-delta into S2 as confirmation instead. |
| **BTC→ETH lead-lag** | G2 | Edge lives at **sub-20ms** (HFT-only); at 60s ETH has already reacted (Cohen's h≈−0.019). **Confirms our stat-arb precedent: any 60s lead-lag backtest edge is a bar-close artifact.** |
| **ETH/BTC ratio regime / alt-season** | G3 | ~1–2 independent events/yr (alt-season fired once all of 2025) — fails ≥30 by ~10×. Same fatal-timeline blocker that parked our MNQ event-fade scout. |
| **Cross-sectional momentum** | G3 + cost | Our ~5–10 perp universe is far below the breadth XS ranking needs (edge is in the tails of hundreds of coins); every head-to-head finds XS weaker than TSMOM and dying after costs (Starkiller: +69%/yr IS → −2.4%/yr OOS). |
| **Vol / VRP without options** | Structural | VRP capture is **structurally impossible** perp-only — variance replication needs an options strip for gamma; a linear perp has zero convexity (Carr & Wu 2009). Vol-targeting is an overlay, not a standalone edge. |
| **ETH on-chain (gas/network/DEX)** | G2 | Daily/swing horizon, not 60s; gas does **not** Granger-cause ETH price (MDPI JRFM 16/10/431). |

### Cross-cutting findings
1. **Convergence:** two of three independent reviews landed on the *fade-extreme-funding-gated-by-OI* mechanism (S2) as best-evidenced — anchored to the same BIS paper. Independent convergence is a positive signal on the *mechanism*, tempered by the absence of a published tradeable Sharpe.
2. **Regime fragility is systemic:** the entire carry/funding/basis complex **decayed to negative in 2025** (Borri/Tsyvinski Sharpe 6.45→4.06→negative). Our 18-month sample sits squarely in that bear regime — the window where these edges *invert*. Any funding-family candidate (S2) must be validated regime-aware, not pooled.
3. **The cost wall recurs:** session-momentum on BTC and generic intraday momentum break even at 3–10 bps — at/below our 10 bps floor. This is the identical wall that killed stat-arb and event-fade. S1 survives only because ETH's higher amplitude lifts the gross spread above the wall — that margin is the entire thesis and must be the first thing tested.
4. **Orthogonality:** S1 (price/vol-amplitude) is genuinely orthogonal to our incumbent BTC-CARRY (funding-harvest); S2 is *correlated* to it (both funding-driven). For portfolio value, S1 > S2.

---

## Build Architecture & Validation Methodology (chosen candidate)

> Applying BMAD pre-registration discipline (Epic 8 weekly workflow, Gate 0, derive-don't-assert, regime-aware OOS). The decision rule and thresholds below are *structural* — actual threshold values are **derived by sweep**, never hand-asserted, per `[[feedback_derive_dont_assert_one_knob]]`.

### Recommended primary: S1 — ETH session-momentum amplitude

**Rationale (why S1 over the better-evidenced S2):** this project's recurring failure mode is OOS-power-starvation and fat-tail mirages (`[[project_mim_nb_expectations_reconciled]]`, `[[project_ml_proba_ordinal_prereg]]`, `[[feedback_iteration_loop_pattern]]`). The highest-value property for the *next* candidate is the power to return a **clean, fast, trustworthy verdict**. S1's ~250 events/yr (~375 over our 18mo) delivers that; S2's ~30–40 fat-tail-concentrated events reproduces the exact trap that keeps biting us. S1 is also orthogonal to our incumbent carry sleeve (portfolio value), where S2 is correlated.

**Honest prior:** S1's only tradeable evidence is one un-sealed practitioner backtest costed at **7 bps < our 10 bps**, with a −64% drawdown. Expected outcome is a real chance of Gate-0 FAIL at true cost. That is acceptable and *desirable* — it fails cleanly and fast, which is what our process needs.

#### System architecture (reuses existing infra)
```
Kraken /tickers REST  (poll 60s; free, unbudgeted)
  → ETH 1-min bar build  (already have data/kraken/PF_ETHUSD_1min.csv)
  → session-state machine (tag bar by regional session: Asia / EU / US open-close in ET)
  → signal: prior-session ETH return sign + amplitude gate
  → entry (limit @ next-session open) / exit (session close or time-stop)
  → execution mirror pattern from btc_carry_executor.py (paper-first, separate service)
```
- Data: **already on disk** (ETH 1-min Nov-2024→present). No new acquisition needed for backtest.
- Reuse: streaming/exec scaffolding from `btc_carry_executor.py`; pre-reg tooling `prereg_seal.py`; validation harness pattern from the MNQ `oos_checkpoint.py`.

#### Gate 0 — the cost-wall reproduction (FIRST, before anything else)
The entire thesis is "ETH amplitude lifts gross spread above the 10 bps wall." Test that and nothing else first:
1. Reproduce the EU→US session-momentum rule on our ETH 1-min data.
2. Cost it at **real 10 bps RT taker** (and a maker-limit variant at 2 bps if fills are realistic).
3. **Kill criterion:** if net per-trade expectancy ≤ 0 at 10 bps, STOP — do not tune, do not subset. (This is the wall that killed `[[project_stat_arb_live_no_edge]]` and `[[project_mnq_event_fade_scout_20260615]]`.)
4. **Derive** the amplitude gate by sweep (one knob), don't assert it.

#### Pre-registration & OOS (only if Gate 0 passes)
- Seal config with `prereg_seal.py` BEFORE any holdout access (commit the seal doc first).
- **Regime-aware OOS, not pooled:** our 18mo is single-regime bear. Split by regime tag; a candidate that only works in trending sub-windows is flagged, not celebrated.
- Decision rule (structural): net PF and per-trade expectancy > 0 at 10 bps across ≥2 regime slices AND N≥30 independent OOS sessions per slice. Exact PF bar derived from the random-null, not hand-set.
- Deploy paper-first on a **separate systemd service** (isolation-by-construction, per `[[project_ts_sim_order_mirror]]`), never commingled with the live combine bots or the carry executor.

### Documented runner-up: S2 — ETH funding-extreme reversal
Pre-register only if S1 fails AND you accept the fat-tail/low-N risk. Build notes: threshold = **rolling percentile** (not absolute), gated by OI-falling or price-rejection confirmation; sweep the percentile (one knob); validate regime-aware because the carry/funding complex went negative in 2025. Its peer-reviewed mechanism (BIS WP1087) is the strongest in the set — but the *tradeable rule* is unproven and the edge concentrates in exactly the fat tails our methodology distrusts.

---

## Executive Summary & Ranked Recommendation

**Topic:** the single best crypto-native strategy to pursue for the Kraken Pro perpetual-futures environment, given our venue, data, and methodology constraints.

**Finding:** the venue and our constraints prune 13 candidate families to **two survivors**. The venue structurally favors funding/basis edges (hourly funding, ±0.5%/hr cap, free OI+funding API) and structurally rules out liquidation-cascade (no feed) and options/vol (no options) strategies. Of the survivors, the right pick is **not** the best-evidenced one — it's the one our broken methodology can actually *validate*.

**Recommendation (ranked):**
1. **PRIMARY — S1: ETH session-momentum amplitude.** Only candidate clearing ≥30 events/yr with real margin (~250/yr → trustworthy, fast verdict on starved data), orthogonal to our carry sleeve, mechanically distinct from dead TSMOM. **Gate 0 = reproduce at real 10 bps cost; kill on failure, no tuning.** Data already on disk.
2. **RUNNER-UP — S2: ETH funding-extreme reversal.** Best evidence (peer-reviewed BIS mechanism, convergent pick of 2/3 reviews), but marginal frequency, fat-tail dilution trap, and correlated to incumbent carry. Pre-register only if S1 fails.
3. **DO NOT PURSUE:** dated-futures basis (0 contracts on Kraken), funding-momentum (price→funding), liquidation-cascade (leveraged-beta mirage / no feed), OI-divergence (folklore), BTC→ETH lead-lag (sub-20ms, dead at 60s — confirms our stat-arb post-mortem), ETH/BTC ratio (~1–2 events/yr), cross-sectional momentum (universe too narrow), perp-only vol/VRP (structurally impossible), on-chain (daily horizon).

**Key risks carried forward:** (1) the funding/carry complex decayed to *negative* in 2025 — our 18mo bear sample may not contain a live funding edge at all; (2) S1's net-of-cost survival is unproven at our 10 bps and is the whole thesis; (3) every tradeable claim in the survivor set rests on *one* practitioner source — independent reproduction at our cost is the non-negotiable Gate-0 condition.

**Next step:** stand up the S1 Gate-0 cost-wall backtest on `data/kraken/PF_ETHUSD_1min.csv`. One script, one knob (amplitude gate), one kill criterion. No pre-registration spend until Gate 0 clears.

**Research completion date:** 2026-06-20 · Sources: web-verified inline (BIS WP1087, JF/JFM/Mgmt-Sci papers, Kraken official docs + live `/instruments` query, practitioner backtests graded by tier). Confidence: HIGH on venue facts and family rejections; MEDIUM on S1/S2 tradeable edge (single-source, requires reproduction).
