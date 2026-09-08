# Digest — dimension: validation & overfitting-control tooling — round 1, assistant 1

Accessed 2026-09-08.

## Findings

- **mlfinlab is NOT open source.** Licensed "all rights reserved," non-commercial research-only, no derivative works; commercial use needs a paid Business/Enterprise licence. — `github.com/hudson-and-thames/mlfinlab` `license.rst` — Hudson & Thames — undated (current file) — high — class(license)
- **mlfinlab's public repo is a bug-tracker front, not the source distribution.** README states the repo exists "for the sole purpose of providing users with an easy way to raise bugs, feature requests, and other issues." — `github.com/hudson-and-thames/mlfinlab` README — Hudson & Thames — undated — high — class(license,version). *Two independent primary pages in the same repo agree ⇒ two-source bar met for the "proprietary" claim.*
- mlfinlab has shipped no new PyPI release in ~12 months. — Snyk Advisor — undated — **medium** (secondary aggregator, not re-verified against PyPI directly) — class(version)
- **mlfinpy** (`github.com/baobach/mlfinpy`) is an independent MIT reimplementation inspired by mlfinlab / López de Prado — *not* an official successor. Latest PyPI release 0.1.2, 2024-10-09, Python >=3.11,<4.0. — pypi.org/project/mlfinpy — high — class(version,license)
- **purgedcv** (`github.com/eslazarev/purged-cross-validation`) implements PurgedKFold, PurgedGroupKFold, WalkForwardSplit, **CombinatorialPurgedCV (CPCV)**, plus Deflated/Probabilistic Sharpe Ratio utilities. MIT. **v0.1.6 released 2026-09-04** — four days before this research. Python 3.10–3.14, 62.5 KB wheel, sklearn/NumPy, optional Polars. — pypi.org/project/purgedcv — high, but **single-sourced (PyPI only)** — class(version,license,method,ecosystem)
- purgedcv self-describes as filling the gap left by mlfinlab going closed-source and by unmaintained predecessors like timeseriescv, calling itself "the maintained open-source standard for honest backtesting validation." — pypi.org/project/purgedcv — **low-medium** (vendor self-description, marketing register) — class(ecosystem)
- **timeseriescv** (`github.com/sam31415/timeseriescv`) implements PurgedWalkForwardCV and CombPurgedKFoldCV, MIT. Only 22 commits total; no confirmable recent-commit date retrieved. — medium (staleness inferred, not dated) — class(version,ecosystem)
- **pypbo** (`github.com/esvhd/pypbo`) implements CSCV-based PBO, PSR, Minimum Track Record Length, **Minimum Backtest Length**, and Deflated Sharpe Ratio. **AGPL-3.0 (copyleft).** 140 stars / 46 forks. Last-commit date not retrievable. — high on licence/scope, **low on maintenance recency** — class(license,method,version)
- **arch** (`bashtage/arch`, Kevin Sheppard) implements **White's Reality Check**, **Hansen's SPA**, and **StepM** (stepwise multiple testing, Romano–Wolf family) returning the set of models superior to a benchmark, plus stationary / circular-block / moving-block bootstraps, in `arch.bootstrap`. — arch.readthedocs.io (v7.2.0 / v8.0.0 docs) — high on method content, **reached via search snippet not direct fetch**; version/licence need a direct confirm — class(method,ecosystem)
- **SharpeBench** (PyPI) *claims* to bundle Deflated Sharpe, PSR, PBO, Reality Check, SPA, Benjamini–Hochberg FDR and pass^k into one package, as a PyO3 binding over a Rust kernel. **UNVERIFIED** — direct PyPI fetch failed; only a search-engine synopsis obtained. — **low** — class(ecosystem)
- **No independent methodological critique of CSCV/PBO or a specific DSR implementation was found** within budget. Search returned only the original Bailey / López de Prado / Borwein / Zhu papers restating the methods' own stated corrections. This is an absence-of-evidence finding, **not a clean bill of health**. — class(critique)

## Package table

| package | implements | license | last release | embeddable | verdict |
|---|---|---|---|---|---|
| mlfinlab | DSR, PBO/CSCV, MinBTL, CPCV, purged/embargoed k-fold, meta-labeling, triple-barrier (reference impl) | **Proprietary, all rights reserved** | PyPI stale ~12+ mo | N/A | **Do not adopt.** Confirmed non-OSS, 2 primary sources. |
| mlfinpy | López de Prado-style data structures, labeling, sampling (method coverage unconfirmed) | MIT | 0.1.2, 2024-10-09 | yes | Candidate; ~2-yr-old single release. Verify completeness + activity. |
| **purgedcv** | PurgedKFold, PurgedGroupKFold, WalkForwardSplit, **CPCV**, DSR/PSR | MIT | **0.1.6, 2026-09-04** | yes, small | **Strongest live candidate for purged/combinatorial CV.** Single-sourced; verify GitHub activity given newness. |
| timeseriescv | PurgedWalkForwardCV, CombPurgedKFoldCV | MIT | undetermined, low activity | yes | Legacy; superseded by purgedcv. |
| pypbo | PBO/CSCV, PSR, MinTRL, **MinBTL**, DSR | **AGPL-3.0** | undetermined | small but copyleft | Exact stats wanted, licence needs sign-off. |
| **arch** | **Reality Check, SPA, StepM**, block bootstraps | OSS (NCSA/MIT-family, not re-verified) | v7.2.0/v8.0.0 documented | yes, focused | **Best-evidenced answer for multiple-testing correction.** Mature, well-known author. |
| SharpeBench | claims DSR+PSR+PBO+RC+SPA+BH-FDR+pass^k, Rust/PyO3 | unverified | unverified | claims small/fast | **Unverified.** Would be the "one library" answer if real. |
| skfolio, PyPortfolioOpt, quantstats, pyfolio-reloaded | portfolio construction / tearsheets, not overfitting control | — | — | — | Not chased (budget); likely low relevance to this dimension (unverified inference). |

## Leads worth chasing
- Confirm **SharpeBench** on GitHub directly — license, maintainer, whether the Rust kernel's math was independently reviewed.
- Verify **arch** licence + latest release by direct primary fetch (two-source bar not yet met).
- Fetch **purgedcv's GitHub** — commit history, contributors, tests, real usage. Is a 4-day-old v0.1.6 a solo weekend project?
- Is arch's **StepM** literally Romano–Wolf (2005) stepdown, or the earlier Hansen/Romano StepM variant? Related, not identical.
- Targeted search for adversarial critique of CSCV's rank logic / DSR's trials-count parameter N.

## What I looked for and could not find
- Second, non-PyPI sources for mlfinpy's and purgedcv's licence/version — both single-sourced ⇒ below the two-source bar.
- Any credible independent critique of DSR / PBO / CSCV. "None found in this pass," not "none exists."
- A working fetch of SharpeBench's PyPI page — license, release date, Python support all unconfirmed.
- Exact last-commit dates for pypbo and timeseriescv — "abandoned vs. quietly stable" unresolved.
- Anything on skfolio / PyPortfolioOpt / finml-utils / quantstats / pyfolio-reloaded for this dimension.
