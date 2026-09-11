# Digest — dimension: implementation reality + hype filter — round 1, assistant 1

Accessed 2026-09-08.

## Findings

1. **NautilusTrader's public discussion venues contain almost no first-person "I ran this live for N months" accounts.** The two most substantive HN threads are dominated by whether algo trading is viable at all, not framework experience. A quant-industry commenter: "we don't even have some of this implemented in code... the tricky part is always going to be the integration"; another: "the simulator for backtesting, integrating with a broker, etc. is such a small part of it" — practitioners there treat framework choice as **secondary to strategy edge and integration work**. — news.ycombinator.com/item?id=44810552 — ~Aug 2025 — medium — class(experience)
2. **STALE:** NautilusTrader had a documented backtesting/data-import doc gap (docs referencing `import_from_data_loader()`, a function no longer implemented; unclear `write_parquet`/`CSVReader` schemas; conflicting `NAUTILUS_BACKTEST_DIR` vs `NAUTILUS_CATALOG`). — GitHub issue #532, opened 2022-01-04, **closed** — high — class(experience, **superseded**). A 4-year-old closed issue; documentation improvement is now an explicit project priority. **Do not repeat as current.**
3. NautilusTrader still has open 2026 issues touching core backtest/account mechanics — "Instruments not loading automatically on market data subscription" (#3237), "Support account equity calculations" (#3899). — GitHub issues — medium (titles only) — class(experience). Consistent with an actively-developed, **still-maturing** codebase.
4. **Free vectorbt has a quantified memory-blowup problem:** a user reported ~24 GB peak RAM for a 2-parameter (~40,000 combination) EMA-crossover grid over ~39,000 bars, vs ~50 MB in equivalent C++ (~480× worse), extrapolating to ~960 GB for a 4-parameter grid on 3-min bars. No maintainer response visible. — github.com/polakowo/vectorbt/issues/406 — 2022-03-07 — high (concrete numbers, primary) — class(experience). **Currency unverified** — could not confirm whether this is fixed, or fixed only in the paid fork.
5. vectorbtpro (paid) markets chunking specs for Numba-compiled simulation plus an optional Rust engine, specifically to avoid that OOM class. — vectorbt.pro/features/performance — **low (vendor marketing, not independently verified)** — class(hype). Cannot certify the free-tier problem is fixed.
6. Third-party comparison blogs describe backtrader as "effectively unmaintained for several years" and failing to install cleanly on Python 3.10+ without patching. — groundy.com (SEO/content-marketing blog promoting a competitor) — **low, flagged not trusted** — class(hype). *(Note: the landscape digest independently confirmed backtrader's last release as 2023-04-19 from GitHub + PyPI.)*
7. **STALE:** LEAN's vendor-lock-in question is largely already litigated — self-hosting is technically possible but reportedly required Docker/Mono/Linux expertise, and the **data** dependency (Morningstar fundamentals) was the real lock-in, not the engine; self-hosting produced no meaningful cost saving. — quantconnect.com forum, thread ~2016-2017 — medium as experience, **stale** — class(experience). No 2025-2026 refresh found; whether this changed is unverified.
8. **"Verified live track record with broker statements / third-party audit" is essentially absent among high-star GitHub day-trading repos.** Searches surfaced only marketing sites for signal-selling services (MyVeridex, FXNX, VerifiedInvesting, Tickeron) — none are GitHub repos, none independently audited in a sense a quant team would accept, all in overt sales register. **No** high-star OSS repo (freqtrade, Lumibot, nautilus_trader…) surfaced with broker-statement-backed or audited live P&L. — medium-high — class(**absence**)
9. **The "day trading repo" content ecosystem is heavily populated by SEO/content-farm blogs** producing near-identical framework comparison posts (BullAlert, AutoTradeLab, Groundy, BrightCoding, python.financial), mutually recycling the same claims ("NautilusTrader bridges the production gap," "pay the learning-curve tax up front") without primary sources or numbers. — directly observed this run — high confidence **in the pattern**, low in any individual claim inside them — class(hype)
10. **awesome-quant** (wilsonfreitas, 957 commits) is a live, actively maintained curated list covering nautilus_trader, vectorbt, backtesting.py, LEAN, freqtrade, Qlib — organized **by category, not star count**, with functional descriptions rather than ratings. — github.com/wilsonfreitas/awesome-quant — medium — class(reference). A second list, `paperswithbacktest/awesome-systematic-trading`, is claimed to hold "97 production-ready libraries, 40+ institutional-grade strategies" per a 2026-06-29 content-farm summary — **low confidence; "production-ready"/"institutional-grade" is the blog's framing, not the curator's.**

## Pain-point table

| framework | current pain (2026, best evidence) | stale complaints | migration-cost signal |
|---|---|---|---|
| **nautilus_trader** | open 2026 issues on instrument auto-loading and account-equity calc; HN practitioners treat framework maturity as secondary to integration/strategy work — **no one reported a completed live migration** | doc/data-import gaps (#532, 2022, closed) | **no first-hand "we migrated our bespoke system" account found** — a gap in evidence, not a finding of low cost |
| **vectorbt / vectorbtpro** | free tier: severe RAM blowup on multi-param grids (24 GB / 2-param / 40k combos; ~960 GB extrapolated for 4-param) | — (currency unverified, not stale) | vendor markets the fix as the **paid** product ⇒ practical path is free→paid, itself a lock-in-shaped cost |
| **freqtrade** | under-evidenced; only generic tutorial advice surfaced | — | crypto-exchange-specific by design ⇒ poor architectural fit, not a documented cost |
| **backtesting.py** | no issue-tracker/forum evidence retrieved | — | not evidenced |
| **LEAN** | data lock-in (proprietary/Morningstar fundamentals) is the structural point even though the engine self-hosts | Docker/Mono friction, "no cost saving" (2016-17) — **unverified for 2026** | engine portability real; data portability reportedly not (old evidence) |
| **Qlib** | no evidence retrieved | — | not evidenced |

## Leads worth chasing
- Deep-read nautilus issues #3237 and #3899 for current primary-source pain.
- **r/algotrading was not reachable** — WebSearch rejected `reddit.com` in allowed_domains and `site:reddit.com` returned nothing. Meaningful blind spot vs the brief.
- freqtrade's own issue tracker (never fetched) for ccxt/exchange-API-churn complaints.
- Confirm free vectorbt's 2026 commit activity directly.
- Primary check of LEAN's current data-access/self-hosting model (only ~9-year-old evidence exists).
- Dedicated pass on migrating a bespoke codebase onto any of these — Q6 came back essentially empty.

## What I looked for and could not find
- **Any** retrospective (6–12 month) first-person production account, for any of the five named frameworks, with concrete before/after numbers (latency, uptime, bug count, P&L drift vs backtest). None in 20 queries/fetches. The public discourse skews to launch announcements, tutorials and SEO comparison blogs, **not retrospectives — itself evidence for the hype-filter dimension.**
- A single high-star GitHub day-trading/AI-trading repo with a broker-verified or audited live track record (Finding 8 — treated as an absence finding).
- Any account of migrating an existing bespoke Python trading codebase onto nautilus_trader, vectorbtpro, LEAN, freqtrade or Qlib with time/effort figures — **completely absent**.
- Working r/algotrading access.
