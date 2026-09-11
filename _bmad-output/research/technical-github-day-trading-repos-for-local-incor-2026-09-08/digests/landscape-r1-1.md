# Digest — dimension: landscape & maturity / ecosystem health — round 1, assistant 1

Accessed 2026-09-08.

## Findings

- **NautilusTrader v1.231.0, released 2026-08-02.** — pypi.org/project/nautilus-trader — PyPI — high — class(version)
- NautilusTrader is **LGPLv3+** with a contributor CLA; targets a bi-weekly release cadence. — github.com/nautechsystems/nautilus_trader README — high — class(license,ecosystem)
- NautilusTrader requires **Python >=3.12,<3.15** (Rust core, Python bindings). — PyPI — high — class(version). *Note: this project pins `python = "^3.11"`.*
- **Backtrader's last release is 1.9.78.123, 2023-04-19** — none since, confirmed on **both** the GitHub repo page and PyPI release history. GPL-3.0. — high — class(ecosystem) — *two-source bar met for the "stagnant" claim.*
- **vectorbt OSS v1.1.0, 2026-07-05** — active. Licensed **Apache-2.0 with Commons Clause** — a non-OSI "fair-code" licence barring selling a product that is primarily this software. — PyPI + GitHub — high — class(version,license)
- **zipline-reloaded v3.1.1, 2025-07-19**, Apache-2.0, Python >=3.10, maintained by Stefan Jansen / ML4Trading post-Quantopian. ~14 months stale ⇒ outside the 6-month health bar, but not confirmed abandoned. — medium — class(version,ecosystem)
- **backtesting.py v0.6.6, 2026-07-22**, **AGPL-3.0-or-later**. — PyPI — high — class(version,license)
- **QuantConnect LEAN** is Apache-2.0, C# core with Python support, positioned as a **full research-to-live platform, not a component library**. — GitHub + QC docs — medium — class(license,landscape). *The "180+ engineers / 300+ hedge funds" figure is unsourced marketing register and was not relied on.*
- **Microsoft Qlib** is MIT. GitHub showed a tagged release "v0.9.0 (2022-12-09)" despite 2,000+ subsequent commits — the "high stars, stale packaged release, unclear commit health" pattern. — **low** — class(ecosystem) — flagged as a lead, not a conclusion.
- Qlib docs reference 1-minute intraday support but examples and market coverage are equities-centric (CN/US), **not futures-native**. — low — class(landscape)
- **pysystemtrade** migrated to a new GitHub org (`pst-group`) in Jan 2026; repo updated as recently as 2026-04-02; Andy Geach maintainer since 2024. — medium — class(ecosystem)
- pysystemtrade is **GPL-3.0** and is explicitly a systematic **futures** framework (Rob Carver's book), a library/framework rather than a turnkey platform. — high — class(license,landscape)
- **freqtrade**: calendar versioning (2026.2/.3/.5), GPL-3.0, ccxt/crypto-exchange only — **not CME futures**. — medium — class(version,license)
- **Hummingbot** v2.13 (~2026-03), Apache-2.0, crypto market-making. — **low** (single-hop aggregator, not fetched from the repo) — class(version,license)
- **Jesse** is MIT and crypto-only; no dated 2026 version retrieved. — low — class(license); version claim **not verified**.
- **Lumibot** (Lumiwealth) actively documented, claims stocks/options/crypto/**futures**/forex backtest-to-live; PyPI listing v4.5.91 surfaced. Licence and true release date **not confirmed**. — low — class(landscape) — lead.

## Comparison table

| repo | version + date | license | last release | futures / 1-min | library or platform | health |
|---|---|---|---|---|---|---|
| nautilus_trader | v1.231.0 (2026-08-02) | **LGPLv3+**, CLA | current, bi-weekly | claims multi-venue incl. futures, ns-resolution ticks/bars (single-source; CME-specific unconfirmed) | platform w/ Rust engine, Python API | **ALIVE, dominant, fast-moving**; needs Py ≥3.12 |
| vectorbt (OSS) | v1.1.0 (2026-07-05) | Apache-2.0 + **Commons Clause** | current | data-agnostic vectorized; works on any OHLCV; no futures-specific machinery confirmed | library | **ALIVE**; non-OSI licence needs a read |
| vectorbtpro | not checked | commercial | — | — | paid library | **GAP** |
| backtrader | 1.9.78.123 (2023-04-19) | GPL-3.0 | none in 3+ yrs | historically yes | library | **STAGNANT / effectively dead** (2 sources) |
| zipline-reloaded | v3.1.1 (2025-07-19) | Apache-2.0 | ~14 mo stale | daily-equities-oriented; minute path exists, not battery-tested per these sources | library | **SLOWING**, not dead |
| backtesting.py | v0.6.6 (2026-07-22) | **AGPL-3.0-or-later** | current | single-asset, any OHLCV incl. 1-min; no futures roll/margin | library | **ALIVE**; highest licence risk here |
| QuantConnect LEAN | current (2026 tag undated this run) | Apache-2.0 | active | yes — multi-asset futures at tick/second/minute | **full platform** | ALIVE; platform-migration cost |
| Qlib | tag v0.9.0 (2022-12-09) vs 2,000+ later commits | MIT | **UNCLEAR** | claims 1-min; equities-centric, not futures-native | ML research hybrid | **AMBIGUOUS** |
| pysystemtrade | updated ~2026-04-02; org moved Jan 2026 | GPL-3.0 | recent | **yes — systematic futures**, IB live | library/framework | **ALIVE** under new ownership |
| freqtrade | 2026.2/.3/.5 | GPL-3.0 | active | **no** — crypto only | bot/platform | alive, **out of scope** |
| jesse | undated | MIT | unverified | crypto only | platform | **out of scope** |
| hummingbot | v2.13 (~2026-03, low conf.) | Apache-2.0 | active (low conf.) | crypto MM | platform | **out of scope** |
| PyBroker | not found | — | — | — | — | **GAP** |
| blankly | not verified | — | — | — | — | **UNVERIFIED** (training-data hint of discontinuation; not confirmed) |
| lumibot | PyPI v4.5.91 seen | not confirmed | not confirmed | **claims futures** | library/framework | **lead worth chasing** |
| moonshot (QuantRocket) | not found | — | — | — | — | **GAP** |
| bt / ffn | not found | — | — | — | — | **GAP** |

## Leads worth chasing
- Qlib's real release/commit health — direct fetch of `/releases` and `/commits/main`.
- vectorbtpro's terms (private repo; needs its own docs/pricing page).
- Lumibot's licence + dated release — notable because it explicitly claims futures support.
- PyBroker, blankly, moonshot, bt/ffn — need direct URL fetches, name-search failed.
- The curated lists themselves (`wilsonfreitas/awesome-quant`, `paperswithbacktest/awesome-systematic-trading`) — never queried; curated-list churn is itself a health signal.
- Whether nautilus_trader's adapters cover **CME** specifically vs generic "futures" framing, and whether it embeds as a pure-Python component or requires the full Rust runtime.

## What I looked for and could not find
- jesse's and hummingbot's current versions to the two-source bar.
- Anything usable on PyBroker, blankly, moonshot/QuantRocket, bt/ffn — complete gaps, not "confirmed dead."
- The awesome-quant / awesome-systematic-trading lists — budget exhausted at 16 tool calls.
- NautilusTrader CME-specific adapter support beyond the repo's own general framing.
- Qlib's true current release cadence.
