# Digest — dimension: execution & fill realism — round 1, assistant 1

Accessed 2026-09-08.

## Findings

- **hftbacktest models queue position and latency explicitly**, using L2/L3 tick data — its own description: "accounts for limit orders, queue positions, and latencies, utilizing full tick data for trades and order books (Level-2 and Level-3)... with real-world crypto trading examples for Binance and Bybit." MIT, Python ≥3.11 / Rust ≥1.90, PyPI v2.4.4 dated 2025-12-10. — github.com/nkaz001/hftbacktest + pypi.org/project/hftbacktest — high — class(capability). **No CME/futures adapter or example found** — usability for CME is architecturally plausible (format-driven) but **unverified in practice**.
- **nautilus_trader's current FillModel is a simple probability toggle, not a queue simulator — the maintainers say so.** Issue #2194 (opened by Stefan Simik, 2025-01-08) states current fill simulation handles only "very basic order-fill scenarios," cannot simulate "competition for liquidity," has **"no queue position simulation capability"**, and exposes two knobs: `prob_fill_on_limit` (default 1.0) and `prob_slippage`. A proposal to add `.process_market_fills`/`.process_limit_fills` hooks was open at that date; **ship status as of Sept 2026 unconfirmed.** — high — class(capability,absence)
- nautilus_trader's higher-fidelity simulation **requires L2/L3 order-book data, not 1-minute bars**: "For the most realistic simulations, higher granularity data sources such as L2 or L3 order book data are recommended when available" — implying bar-only backtests fall back to the probabilistic model. — nautilustrader.io/docs concepts index — medium (dedicated fill-model subpages 404'd) — class(capability)
- **LEAN's default slippage model for futures is zero, and stop fills use bar-close price.** "The default brokerage model uses the NullSlippageModel to model zero slippage for all securities" unless overridden; `FutureFillModel` fills stop orders "at the close price (of the same bar or next bar depending on data), plus slippage." — quantconnect.com/docs reality-modeling/trade-fills — medium — class(capability). *Same limitation this project already has.*
- **backtrader** adds a volume-cap "Filler" (partial-fill approximation); **vanilla vectorbt has none** — characterized as "implicitly assumes you can reach your target position at each bar's price," no queue position, partial fill, or intrabar stop. — backtrader.com/docu/filler.html (high, primary) + a Medium post for vectorbt (**low-medium, single secondary source, not vendor docs — a lead, not settled**) — class(capability)
- **No open-source tool models variable/bursty latency inside a P&L fill simulation.** hftbacktest exposes a latency *interface* ("provided models or your own custom model") but whether the provided models go beyond fixed/empirically-replayed latency is **unverified**. Two adjacent-but-different tools: **ABIDES** (arXiv:1906.12010 lineage) models "variable electronic network latency and agent computation delays" but is an agent-based market simulator, **not a historical-replay P&L backtester**; **HFTPerformance** benchmarks *your own* tick-to-trade latency under synthetic bursty data — it measures your infrastructure, it does not model exchange-side latency inside a fill simulation. — medium — class(**absence**)
- **Literature quantifying a bar-close-backtest-vs-live-fill gap is thin and mostly marketing register.** Hits were practitioner blogs (LuxAlgo, eialgosinc, thortradecopier, enlightenedstocktrading, fortraders) with qualitative claims ("signal generated at bar close may not reach the exchange for 50–500ms") and no disclosed methodology. Downgraded; not used as evidence of a quantified gap. — low — class(evidence, reported thin)
- **One credible dated MNQ-specific academic source exists but addresses friction cost, not fill simulation.** arXiv:2605.04004, "Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures," Mathias Mesfin, submitted 2026-05-05, revised 2026-07-13: across 14 signal families / 947 trading days on 5-min MNQ bars, "the maximum gross return before transaction costs ranged from roughly 0.07 to 1.50 points per trade, well below the assumed two-point round-trip friction cost." — medium (single arXiv preprint, peer-review status unverified) — class(evidence). **Do not conflate with fill-simulation realism.**
- **Databento** sells CME Globex MDP 3.0 MBO and MBP-10 historically, usage-based $/GB; exact rate requires their estimator. Standard/usage tiers get MBO/MBP-10 for only the "Last 1 month"; Plus/Unlimited get "16+ years." **Exact $/GB not retrieved.** — databento.com/pricing (undated page) — medium — class(pricing)
- **CME's own real-time non-professional market-data fees, January 2026:** market-depth $5/month per exchange or $15/month bundled across CME/CBOT/NYMEX/COMEX; top-of-book $1/month per exchange or $3/month bundled. — cmegroup.com January-2026 market data fee list (PDF) — high — class(pricing). **This is real-time distribution licensing, not historical backtest data — do not use it to answer "what does backtest data cost."**

## Fill-model comparison table

| engine | fill resolution | latency model | queue position | data required | license | verdict |
|---|---|---|---|---|---|---|
| **hftbacktest** | tick / order-book, event-driven | feed + order latency, pluggable (provided models' specifics unverified) | **yes, explicit** | L2 MBP or L3 MBO full tick | MIT | most execution-realistic OSS engine found; crypto-proven, **no confirmed CME usage** |
| nautilus_trader | advertises ns/order-book; default FillModel is a 2-param probability toggle | framework-wide config exists; not validated as bursty | **no** (maintainers, #2194, Jan 2025) | L1 bars work but degrade to probabilistic; L2/L3 for advertised fidelity | (not confirmed this run) | strong architecture, but **bar-data fill realism is not meaningfully better than backtrader/LEAN today** |
| LEAN | bar-close (stops at bar close + slippage) | none by default (NullSlippageModel = 0) | no | OHLCV | Apache-2.0 (not re-confirmed) | **same bar-close overstatement this project already has**, by default |
| backtrader | bar-close + volume-cap Filler | none found | no | OHLCV + volume | GPLv3 | marginal improvement; no latency/queue |
| vectorbt (OSS) | bar-close/open, vectorized | none | no | OHLCV | Apache-2.0 (not re-verified) | fastest, least realistic (secondary source, low conf.) |
| ABIDES (adjacent) | continuous double auction, ns | **yes — variable network latency + agent compute delay** | implicit via matching engine | synthetic/agent order flow, not historical replay | not verified | **not a fit** for "replay my MNQ history with realistic fills" |
| vectorbtpro, backtesting.py | not verified this run | — | — | — | — | insufficient evidence |

## Leads worth chasing
- Did nautilus #2194's `process_market_fills`/`process_limit_fills` hooks ship by Sept 2026? Check RELEASES.md / recent tags. **Determines whether the queue gap has closed since Jan 2025.**
- Exact Databento $/GB for CME MDP 3.0 MBO via their estimator (needs an account).
- Read hftbacktest's latency-model source — are "provided models" stochastic/bursty or fixed-constant only? **Single most decision-relevant unresolved question.**
- Has anyone wired hftbacktest to CME MBO (Databento DBN → hftbacktest book format)?
- vectorbtpro's own docs on richer order simulation.

## What I looked for and could not find
- **No OSS project modelling bursty/variable latency inside a P&L fill simulator.** Looks like a genuine gap, not a search miss — it lines up with this team's own bespoke tool failing on exactly that axis.
- No methodologically-disclosed write-up quantifying bar-close backtest P&L vs live P&L holding strategy and costs fixed. Every hit was marketing-register.
- No confirmed case of nautilus_trader, hftbacktest, or LEAN run end-to-end against **CME MDP 3.0 MBO for MNQ or a similar micro future** — all concrete usage evidence was crypto or asset-class-agnostic marketing.
- Exact historical-MBO $/GB from Databento; no competing vendors (Polygon, dxFeed, LSEG Tick History, Exegy) checked.
