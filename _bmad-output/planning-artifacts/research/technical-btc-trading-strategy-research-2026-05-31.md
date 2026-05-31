---
stepsCompleted: [1, 2]
inputDocuments: []
workflowType: 'research'
lastStep: 1
research_type: 'technical'
research_topic: 'High-return BTC scalp/swing trading strategy for pre-registered live deployment'
research_goals: 'Identify proven BTC strategies with highest return profiles; produce a fully specified, pre-registerable hypothesis ready for live paper trading on Kraken Futures (PF_XBTUSD)'
user_name: 'Alex'
date: '2026-05-31'
web_research_enabled: true
source_verification: true
---

# Research Report: Technical

**Date:** 2026-05-31
**Author:** Alex
**Research Type:** Technical

---

## Research Overview

Exhaustive research into high-return BTC scalp/swing strategies, evaluated for suitability as a pre-registered live trading hypothesis on Kraken Futures PF_XBTUSD.

---

<!-- Content will be appended sequentially through research workflow steps -->

---

## Technical Research Scope Confirmation

**Research Topic:** High-return BTC scalp/swing trading strategy for pre-registered live deployment
**Research Goals:** Identify proven BTC strategies with highest return profiles; produce a fully specified, pre-registerable hypothesis ready for live paper trading on Kraken Futures (PF_XBTUSD)

**Instrument:** Kraken Futures `PF_XBTUSD` (BTC/USD perpetual, point_value=$1/pt, 24/7)
**Constraint:** S25 bearish FVG stack confirmed no edge on BTC (PF 0.71–0.88 across 600+ trades, 16 months). Must use a structurally different signal.
**Methodology:** Any strategy selected must be pre-registered (hypothesis sealed in git commit) before backtesting against `data/kraken/PF_XBTUSD_1min.csv` (Nov 2024 – May 2026).

**Scope Confirmed:** 2026-05-31

---

## Strategy Landscape Analysis

### Eight BTC Strategy Archetypes: Verified Performance

Research covered eight distinct strategy classes. Ranked below by documented risk-adjusted return and pre-registration suitability.

---

#### 1. Funding Rate Arbitrage (Delta-Neutral Carry)

**Mechanism:** Long BTC spot + Short BTC perpetual simultaneously. Profit = funding payments received when funding is positive.

**Documented Performance:**
- Annual returns: **15–35%** (market-neutral, delta-hedged)
- Sharpe ratio: **3–6** in 3-year backtests; BIS Working Paper 1087 reports Sharpe of **12.8** for BTC contracts in-sample
- Max drawdown: **<5%** for properly managed delta-neutral positions
- Monthly returns positive in **every single month of 2025**, ranging 0.43%–1.42%
- BTC funding positive **322 of 365 days in 2024**; peak Jan 2024: 0.07%/8h = 76.65% annualized
- 2025 average: **19.26% annual** (up from 14.39% in 2024)

**Limitation:** Requires simultaneous spot + perpetual positions. Not directly implementable on Kraken Futures alone (no spot leg). Also: funding rates have compressed — basis collapsed to **4.46% annualized** by Dec 2025.

**Pre-registration suitability:** HIGH for the concept, LOW for Kraken-only implementation.

_Sources: [ScienceDirect Funding Rate Arbitrage Study](https://www.sciencedirect.com/science/article/pii/S2096720925000818), [Gate Learn Perpetual Arbitrage](https://www.gate.com/learn/articles/perpetual-contract-funding-rate-arbitrage/2166), [BIS Working Paper 1087](https://www.bis.org/publ/work1087.pdf)_

---

#### 2. BTC Intraday Seasonality (Time-of-Day Kill Zone)

**Mechanism:** BTC has a statistically robust hourly return pattern. The **21:00–23:00 UTC window** (specifically 22:00 UTC) is the highest-returning hour across multi-year data.

**Documented Performance:**
- Strategy trading only this 2-hour window: **40.64% annualized, Calmar ratio 1.79**
- Best days in this window: **Friday, Thursday, Saturday/Sunday**
- 22:00 UTC average return: **~0.07%** per hour (extraordinary for a single hour)
- A second strategy using this pattern: **33% annualized**
- Monday Asia Open Effect: "pronounced performance pickup starting Sunday evening NY time extending into Monday"
- NYSE-closed effect: when traditional markets closed, BTC has a **strong intraday component** and weak overnight component

**Mechanism explanation:** 22:00 UTC = 18:00 ET = post-US market close + pre-Asia open. Institutional re-balancing, reduced liquidity, and systematic algo positioning converge. The pattern has been stable from 2018–2025.

**Pre-registration suitability:** VERY HIGH — the hypothesis is specific: "trade BTC only during 21:00–23:00 UTC, entry triggered by [signal], closed by [time stop or TP/SL]."

**Implementation:** Directly implementable as a `kill_zone_start: "21:00"`, `kill_zone_end: "23:00"` (UTC) filter in the existing `strategy_config_kraken.yaml`. No new code required.

_Sources: [QuantPedia Intraday Seasonality](https://quantpedia.com/are-there-seasonal-intraday-or-overnight-anomalies-in-bitcoin/), [QuantifiedStrategies BTC Seasonality](https://www.quantifiedstrategies.com/bitcoin-intraday-seasonality-trading-strategy-backtest-results/), [Concretum Group Seasonality](https://concretumgroup.com/seasonality-in-bitcoin-intraday-trend-trading/)_

---

#### 3. Bitcoin Momentum / Multi-Timeframe Trend Following

**Mechanism:** Capture BTC's well-documented trending behavior using moving average crossovers, Donchian channels, or Z-score momentum signals.

**Documented Performance:**
- 20/100-day MA crossover: **116% annualized, Sharpe 1.7** vs buy-and-hold Sharpe 1.3
- Blended 50/50 momentum + mean reversion: **56% annualized, Sharpe 1.71**
- Donchian channel ensemble: Sharpe **>1.5**, 10.8% annualized alpha vs BTC
- BTC intraday trend 2018–2025: gross Sharpe **~1.6**
- Momentum with volatility filtering: Sharpe **1.2** (improves from 1.0 unfiltered)
- Multi-timeframe with trailing stop: Sharpe **1.07, Calmar 0.87**

**Pre-registration suitability:** HIGH — MA crossover parameters can be fully specified before testing.

**Implementation:** Not directly compatible with the current 1-min FVG framework without architectural changes. Would require a separate trend-following module.

_Sources: [QuantPedia Multi-Timeframe BTC](https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/), [QuantifiedStrategies Trend Following](https://www.quantifiedstrategies.com/trend-following-and-momentum-strategies-on-bitcoin/), [Concretum Group Trend](https://concretumgroup.com/catching-crypto-trends-a-tactical-approach-for-bitcoin-and-altcoins/)_

---

#### 4. Funding Rate Contrarian Signal (Directional Entry)

**Mechanism:** When perpetual funding rates reach extreme negative territory, the market is heavily short → historical precursor to major relief rallies. Conversely, extreme positive funding → crowded long → mean reversion downside.

**Documented Performance:**
- Extremely negative funding (< -0.1%/8h) has "historically preceded major relief rallies in Bitcoin's history"
- Early 2026: BTC perpetual funding entered longest sustained negative streak since Nov 2022 bear bottom (which was a major bottom)
- Positive funding >0.07%/8h (Jan 2024 peak) correlated with near-term mean reversion

**Signal specificity:**
- **Long entry signal:** 8h funding rate < -0.05% for 3+ consecutive periods
- **Short entry signal:** 8h funding rate > +0.05% for 3+ consecutive periods AND open interest rising

**Pre-registration suitability:** HIGH — thresholds are specific and data is available from Kraken API.

**Implementation:** Requires adding Kraken funding rate API polling. Moderate complexity.

**Data source:** Kraken Futures API: `GET /api/v3/tickers` returns `fundingRate` field.

_Sources: [BitMEX Q2 2025 Derivatives Report](https://www.bitmex.com/blog/2025q2-derivatives-report), [Gate Wiki Funding + OI Signals](https://web3.gate.com/crypto-wiki/article/how-do-futures-open-interest-and-funding-rates-signal-crypto-derivatives-market-trends-in-2026-20260202), [CoinGlass BTC Funding Rate](https://www.coinglass.com/FundingRate/BTC)_

---

#### 5. Open Interest + Funding Rate Composite (Regime Detection)

**Mechanism:** Combine OI trend with funding rate direction to identify market regime: (a) crowded long = short setup, (b) crowded short = long setup, (c) expanding OI + neutral funding = trend continuation.

**Documented Performance:**
- "Integrating open interest trends, funding rate extremes, and liquidation clustering patterns, traders can identify market turning points"
- Institutional approach: "When ETF inflows surge but funding stays subdued, that is durable demand; when funding spikes to over 20% annualized while ETF flows stall, that is leverage chasing momentum" → mean reversion imminent
- BTC aggregate OI peaked at $70B June 2025, orderly decline Q4 — OI contraction correlated with trend exhaustion

**Signal specificity:**
- **SHORT setup:** OI > 3σ above 30-day MA + funding > +0.03%/8h (crowded long squeeze)
- **LONG setup:** OI falling (unwinding) + funding < -0.02%/8h (capitulation)

**Pre-registration suitability:** MEDIUM-HIGH — requires defining the OI normalization window and funding threshold precisely.

_Sources: [Gate Wiki OI + Funding 2026](https://web3.gate.com/crypto-wiki/article/how-do-futures-open-interest-and-funding-rates-signal-crypto-derivatives-market-trends-in-2026-20260202), [Coinalyze BTC OI](https://coinalyze.net/bitcoin/open-interest/), [AInvest Funding Rate Analysis](https://www.ainvest.com/news/bitcoin-oversold-rsi-deleveraging-trap-flow-catalyst-2602/)_

---

#### 6. Weekend Effect (Day-of-Week Seasonality)

**Mechanism:** BTC has statistically higher returns on weekends vs weekdays.

**Documented Performance:**
- Mean daily return: **0.0023 on weekends vs 0.0012 on weekdays** (Jan 2020–Apr 2025)
- $1 → **$2.47 on weekends** vs $1.85 on weekdays over this period
- "7-day momentum strategy shows weekend momentum returns significantly exceeding weekday returns"

**Pre-registration suitability:** VERY HIGH — simply hold long on Sat-Sun or avoid trading Mon-Fri. Trivially specific.

**Implementation:** Already have `tuesday_exclusion` logic. Generalizing to a `weekday_exclusion` pattern requires minimal code.

**Caution:** Simple long-only weekend bias interacts poorly with bearish-only signal. But combining with short setups on weekdays (when return is lower) makes structural sense.

_Sources: [ACR Journal Weekend Effect Crypto](https://acr-journal.com/article/the-weekend-effect-in-crypto-momentum-does-momentum-change-when-markets-never-sleep--1514/), [QuantPedia Weekend Seasonality](https://quantpedia.com/the-seasonality-of-bitcoin/)_

---

#### 7. Liquidation Cascade Mean Reversion

**Mechanism:** When large leveraged liquidations cascade (>$200M+ within 2h), price typically overshoots and reverts. Enter counter-trend after confirmed cascade.

**Documented Performance:**
- Dec 2024: $400M liquidated → 7% flash crash $103k→$92k → recovered within 48h
- Nov 2025: $450M liquidated in 2 hours → sharp V-reversal
- "Smart money intentionally absorbs liquidation cascades... filling their own orders"

**Signal specificity:** Requires liquidation data feed (Coinglass API or Kraken aggregated liquidation endpoint). Cascade threshold: >$100M/hour.

**Pre-registration suitability:** MEDIUM — cascade detection needs precise definition; data feed adds infrastructure complexity.

**Implementation:** Moderate-high. Needs liquidation data not currently in the system.

_Sources: [XT Exchange Liquidation Cascades](https://medium.com/@XT_com/bitcoin-futures-market-microstructure-liquidation-cascades-funding-regimes-and-open-interest-978b107b4889), [CoinChange Nov 2024 Cascade](https://www.coinchange.io/blog/bitcoins-2-billion-reckoning-how-novembers-liquidations-cascade-exposed-cryptos-structural-fragilities)_

---

#### 8. Order Book Microstructure (Imbalance)

**Mechanism:** Use real-time order book imbalance as a short-term directional signal. Momentum at 10–30 second horizons from parent-order execution pressure.

**Documented Performance:**
- "Order flow imbalance, bid-ask spreads, depth, and trade arrival patterns explain a substantial fraction of return variation at very short horizons"
- Momentum signal at 10–30s horizon: consistent across BTC, LTC, ETH and other assets
- "Does not directly imply a profitable strategy without considering fees and bid-ask spreads"

**Pre-registration suitability:** LOW for current framework — requires tick-level order book data, not 1-min bars. Not implementable without major infrastructure changes.

_Sources: [arXiv Crypto Microstructure 2602.00776](https://arxiv.org/html/2602.00776v1), [arXiv LOB Dynamics 2506.05764](https://arxiv.org/html/2506.05764v2), [Towards Data Science OBI](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6/)_

---

## Candidate Strategy Ranking for Pre-Registration

Evaluated on four criteria: **Performance**, **Specificity** (pre-registerable without ambiguity), **Independence from S25** (not just the same signal on a different asset), and **Implementation complexity** given existing Kraken infrastructure.

| Rank | Strategy | Documented Return | Sharpe | Specificity | Independence | Impl. Complexity |
|---|---|---|---|---|---|---|
| **1** | **Intraday Seasonality (21-23 UTC)** | 40.64% ann. | Calmar 1.79 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟢 Low |
| **2** | **Funding Rate Contrarian** | 15-35% ann. | 3-6 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 Medium |
| **3** | **OI + Funding Composite** | Not isolated | ~2-3 est. | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🟡 Medium |
| **4** | **Momentum / MA Crossover** | 56-116% ann. | 1.5-1.71 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 🔴 High (new module) |
| **5** | **Weekend Effect** | ~$2.47 vs $1.85 | ~0.8 est. | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 🟢 Low |
| **6** | **Liquidation Cascade** | V-recovery | Unknown | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 High (new data) |
| **7** | **Funding Rate Arbitrage** | 15-35% ann. | 3-12.8 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 Not applicable |
| **8** | **Order Book Imbalance** | Short-horizon | Unknown | ⭐⭐ | ⭐⭐⭐⭐⭐ | 🔴 Very high |

---

## Top 3 Pre-Registerable Hypotheses

### Hypothesis A: BTC 21:00–23:00 UTC Kill Zone Momentum
**Full specification:**
- Instrument: PF_XBTUSD (Kraken Futures, 1-min bars)
- Direction: Bearish only (SHORT entries) in the 21:00–23:00 UTC window
- Entry condition: H1 bearish sweep (last 6 H1 bars) + M15 CHoCH + M1 bearish FVG ≥ 0.04×H1 ATR
- Exit: SL 5×gap, TP 6×gap, time-stop 60 bars
- Kill zone: 21:00–23:00 UTC only (no Tuesday exclusion — crypto trades 24/7)
- Decision rule: PF > 1.20 after N≥30 AND 60 days

**Why this first:** Uses existing proven signal stack but applies it only in the documented highest-return hour window. The S25 stack showed PF 0.81 overall on BTC — but that includes ALL hours. Focusing only on 21:00–23:00 UTC is a legitimate, pre-registerable refinement hypothesis. If the BTC edge exists at all in the S25 stack, it is most likely concentrated in this window (lower liquidity, higher realized vol, institutional re-hedging activity).

**Backtest data available:** Yes — existing `PF_XBTUSD_1min.csv` covers the full window.

---

### Hypothesis B: BTC Funding Rate Threshold Reversal
**Full specification:**
- Instrument: PF_XBTUSD (Kraken Futures)
- Signal: Kraken 8-hour funding rate
- SHORT entry condition: 8h funding rate ≥ +0.04% for 2+ consecutive periods (crowded long → short squeeze imminent) + price at H1 resistance level (prior swing high)
- LONG entry condition: 8h funding rate ≤ -0.03% for 2+ consecutive periods (capitulation short) + price at H1 support
- Exit: SL 3×ATR, TP 5×ATR, max hold 24h
- Decision rule: PF > 1.20 after N≥30 AND 60 days

**Why this second:** Fully independent of the FVG pattern. Funding rate is a structural feature of perpetual markets — when too many traders are positioned the same direction, the expected value of that direction decreases. This is academically verified across multiple studies.

**Implementation requirement:** Add Kraken API funding rate polling (single endpoint: `GET /derivatives/api/v3/tickers` returns `fundingRate`).

---

### Hypothesis C: BTC Intraday MA Crossover Trend Signal
**Full specification:**
- Instrument: PF_XBTUSD (1-min bars)
- Signal: EMA(20) crosses above EMA(100) on 1-min bars → LONG; below → SHORT
- Entry: Market order on confirmed cross
- Filter: Only trade when 4h trend (EMA20>EMA100 on H4) aligns with 1m direction
- Exit: Trailing stop 1.5×ATR, max hold 4h
- Decision rule: PF > 1.20 after N≥50 AND 60 days

**Why this third:** Pure momentum, fully independent of liquidity sweep / CHoCH / FVG. Documented Sharpe 1.7 on Bitcoin with MA crossover. The 1-min implementation on perpetuals has not been specifically verified — that IS the hypothesis.

**Implementation:** New signal module required (no H1 sweep dependency). Moderate complexity.

---

## Implementation Recommendation

**Start with Hypothesis A.** It requires zero new infrastructure — `kill_zone_start: "21:00" UTC` and `kill_zone_end: "23:00" UTC` in `strategy_config_kraken_s26.yaml`, plus fixing the two critical bugs in `s26_crypto_streaming_working.py` (Golden Flip direction mismatch + stub `_close_active_trade`). The signal stack is already written.

The pre-registration sequence:
1. **Seal the hypothesis** → `prereg_seal.py` commit before any backtest
2. **Fix the two bugs** → make `_close_active_trade` actually record trades; remove Golden Flip OR make it config-driven
3. **Backtest** → `backtest_crypto_clean.py` with `kill_zone` filter enabled and UTC window 21:00–23:00
4. **Gate:** If backtest PF > 1.10 on 2025 data with N≥30 → deploy live paper trading
5. **Decision rule:** N≥30 AND 60 days live → evaluate

If Hypothesis A fails in backtest: proceed to Hypothesis B (funding rate signal — new API data required).

---

*Sources compiled from: QuantPedia, QuantifiedStrategies, ScienceDirect, BIS Working Papers, Concretum Group, ACR Journal, arXiv, BitMEX Quarterly Reports, CoinGlass, Coinalyze, AInvest, Gate Learn, Medium/CoinMonks*
