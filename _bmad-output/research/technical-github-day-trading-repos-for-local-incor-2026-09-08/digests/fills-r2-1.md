# Digest — dimension: execution & fill realism + coverage sweep — round 2

Accessed 2026-09-08. **Three round-1 findings overturned.**

## Findings

- **OVERTURNS round 1.** nautilus_trader issue #2194 "Enhanced order-fill simulation in backtesting" opened 2025-01-08, **closed 2025-10-30**. — github.com/nautechsystems/nautilus_trader/issues/2194 — high — class(capability)
- **Shipped since #2194** per RELEASES.md: Python `FillModel`/`FeeModel` subclass support, **`CompetitionAwareFillModel`**, **`VolumeSensitiveFillModel`**, `BestPriceFillModel`, `fill_limit_inside_spread`, **L1 quote-based queue-position tracking**, **L3 per-order-delta queue position (#4370)**, queue-position fill gating improvements. — raw RELEASES.md on `develop` — high — class(version,capability)
- **But the architecture stays closed.** Open feature request #3943 (2026-04-28, still OPEN): "FillModelAny is a closed enum and the matching engine dispatches through it statically... custom fill behaviour can't be supplied from outside the Rust core." — high — class(capability)
- nautilus_trader repo: pushed 2026-09-08, **28,617 stars**, **LGPL-3.0**, latest **v2.0.0rc4 (2026-09-02)** alongside maintained v1.231.0 (2026-08-02). — GitHub API — high — class(version)
- **DECISIVE, from source.** `hftbacktest/src/backtest/models/latency.rs` defines trait `LatencyModel` with exactly two concrete types: **`ConstantLatency`** (fixed) and **`IntpOrderLatency`** (linear interpolation over recorded historical req_ts/exch_ts/resp_ts triples). **No stochastic/generative jitter model exists.** — raw source read — high — class(capability)
- hftbacktest repo: pushed **2025-12-23** (~8.5 months stale), 4,630 stars, MIT, Rust. — GitHub API — high — class(version)
- **OVERTURNS round 1's "no CME usage confirmed."** hftbacktest ships an official `hftbacktest.data.utils.databento` module and a **"Level-3 Backtesting" tutorial building a queue-position-aware L3 backtest from Databento CME Market-By-Order data**, worked example **BTCM4 (CME-listed Bitcoin futures, June 2024 expiry)**, comparing L3 vs L2 queue-position accuracy. — hftbacktest.readthedocs.io — high — class(capability). *Caveat: a crypto-linked CME future, not an equity-index/rates/commodity future.*
- Search for new (2025-26) queue-aware or intrabar **futures-specific** fill simulators returned only hftbacktest and nautilus_trader, plus one tiny unverified options-only project (FlashAlpha-lab, spreads not futures). — low (single source, unverified) — class(absence)
- **awesome-systematic-trading tags dead frameworks inline**: zipline "dormant since 2024-02", backtrader "dormant since 2024-08", quanttrader "dormant since 2024-06", gobacktest "archived". — raw README — high — class(list-curation)
- awesome-quant's Trading & Backtesting section contains numerous very-low-star, generic-AI-flavoured entries (lesson-book, TradeSight, PRISM-INSIGHT) — consistent with round 1's content-farm density warning. — medium — class(quality signal)
- **orderflow-metrics** (twowaymind): MIT, dependency-free TS+Python, includes execution cost / price impact, market impact, Kyle's lambda, implementation shortfall, execution scheduling. Pushed 2026-09-07, **only 9 stars**. — README + GitHub API — medium (brand-new, unproven) — class(capability)
- **PyBroker** (edtechre/pybroker) licence field returns **NOASSERTION**, despite both curated lists carrying it. Pushed 2026-09-07; 1-min intraday walkforward documented; futures unconfirmed. — GitHub API — high — class(license)
- **bt** and **ffn** (pmorissette) READMEs contain **zero** "futures"/"intraday" hits — daily-frequency portfolio analytics by design. Both MIT, both pushed 2026-09-07/08. — medium — class(capability)
- **lumibot**: "Backtesting and Trading Library for Stocks, Options, Crypto, **Futures**, FOREX and More!", licence **GPL-3.0**, pushed 2026-09-07. — README + API — high — class(capability,license)
- **Blankly is dead**: pushed **2024-12-30**, no commits in ~21 months — yet still listed unqualified in awesome-quant. — GitHub API — high — class(absence)
- **QuantRocket moonshot**: Apache-2.0, pushed 2026-07-30, tightly coupled to the QuantRocket commercial platform/data subscription rather than usable as a standalone pip library. — medium — class(capability,license)

## Part A verdict: has anything closed the fill-realism gap?

**Partially, and only on nautilus_trader's side.** #2194 closed Oct 2025 with concrete shipped output — a `get_orderbook_for_fill_simulation` hook, several queue-position-aware `FillModel` variants, and genuine L1/L3 queue-position tracking in the matching engine. That is a real, verifiable upgrade from "very basic" in Jan 2025. **The round-1 "no queue position simulation capability" claim is now STALE and must not be repeated as current.** But the still-open #3943 (Apr 2026) shows fill behaviour is dispatched through a static Rust enum, so a user cannot supply arbitrary custom fill logic from outside the core — only the finite set of built-in variants. **Materially improved, not fully solved.**

**hftbacktest's latency model is decisively not stochastic** — fixed constant or deterministic interpolation replay of recorded latency. This confirms from source that **no open-source tool provides a genuinely stochastic/bursty latency model inside a backtest fill simulator**, as of September 2026. hftbacktest has had no push since Dec 2025, so nothing new has shipped there.

## Part B table

| project | licence | last activity | alive | library or platform | futures/intraday | worth a look |
|---|---|---|---|---|---|---|
| nautilus_trader | LGPL-3.0 | 2026-09-08 | very active | platform | multi-asset incl. futures; best fill realism of the set | **yes** — current best-in-class OSS fill simulation |
| hftbacktest | MIT | 2025-12-23 (~8.5mo) | slowing | library | **queue position + CME MBO via Databento confirmed** | **yes, if you have L2/L3 tick data** |
| PyBroker | **NOASSERTION** | 2026-09-07 | yes | library | 1-min intraday walkforward documented; futures unconfirmed | maybe — check licence text first |
| bt | MIT | 2026-09-07 | yes | library | none — daily algo-tree | daily portfolio backtests only |
| ffn | MIT | 2026-09-08 | yes | analytics only | n/a | post-hoc performance stats only |
| lumibot | GPL-3.0 | 2026-09-07 | yes | library/framework | explicit futures claim | maybe — GPL-3.0 |
| moonshot | Apache-2.0 | 2026-07-30 | slow | tied to QuantRocket platform | needs subscription | **no** — platform lock-in |
| Blankly | LGPL-3.0 | 2024-12-30 | **dead** | hybrid | no futures | **no** |
| orderflow-metrics | MIT | 2026-09-07 | brand-new, 9★ | metrics library, not a backtester | instrument-agnostic impact/cost functions | promising lead, too new to trust |

## What I looked for and could not find
- A genuinely stochastic/variable-latency backtest fill model anywhere in open source — **confirmed absent again**. (Residual gap: nautilus's own `LatencyModel` internals in `BacktestNode` were not source-checked this round.)
- Any post-2025 project doing queue-aware/intrabar fill simulation for **traditional (non-crypto-linked) futures** — none surfaced.
- A mature, standalone Almgren–Chriss-style market-impact / transaction-cost library. orderflow-metrics is the only candidate and is effectively one day old.
- PyBroker's actual LICENSE text behind the NOASSERTION flag.
