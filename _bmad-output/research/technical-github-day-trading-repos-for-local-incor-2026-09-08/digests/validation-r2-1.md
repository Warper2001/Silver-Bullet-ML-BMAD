# Digest — dimension: validation & overfitting-control tooling — round 2 (verification pass)

Accessed 2026-09-08. **Two round-1 findings overturned.**

## Findings

1. **OVERTURNS round 1's "four-day-old package" concern.** purgedcv's repo was created **2026-05-15**; v0.1.6 is release #19 in a continuous May→Sept series (0.0.1 → 0.1.6). — GitHub API, `eslazarev/purged-cross-validation` — created 2026-05-15, last push 2026-09-04 — high — class(version)
2. purgedcv contributors: eslazarev 101 commits, github-actions[bot] 19; **no other human contributor ⇒ bus factor 1.** — api.github.com/repos/.../contributors — high — class(credibility)
3. purgedcv has a `tests/` directory, passing CI badge, codecov badge, a `paper/paper.md` (JOSS draft) and `.zenodo.json`. README claims — **self-reported, not independently reproduced** — three validation studies (synthetic leakage collapse, UK smart-meter 6.6% MAE improvement, crypto no-edge check). — medium (page summary, raw test files not inspected) — class(credibility,method)
4. **purgedcv licence = MIT, confirmed on two independent endpoints** (GitHub repo API + PyPI JSON metadata). — high — class(license)
5. **arch**: repo created 2014-08-29, last pushed **2026-08-10**, 1,559 stars, 289 forks, 51 open issues, maintainer Kevin Sheppard. — GitHub API + PyPI JSON — high — class(credibility,version)
6. **arch licence is UNRESOLVED**: GitHub's detector reports `Other (NOASSERTION)`; PyPI's classifier reports `NCSA`. The actual LICENSE file could not be fetched (3 path guesses 404'd). — **low / unresolved** — class(license)
7. `arch/bootstrap/multiple_comparison.py` contains `MultipleComparison`, `MCS`, `StepM`, `SPA`, `RealityCheck`. **`RealityCheck` is implemented as a thin subclass of `SPA`** — White's Reality Check is a special case of Hansen's SPA, consistent with the econometrics literature, not a shortcut or bug. — raw.githubusercontent.com source read — high — class(method)
8. **arch's `StepM` IS the Romano–Wolf (2005) stepdown**, confirmed by the literal docstring citation "Stepwise multiple testing as formalized data snooping. Econometrica, 73(4), 1237-1282." — source read — high — class(method). *Resolves round 1's open question.*
9. **SharpeBench exists and is reachable** (round 1's fetch failure did not replicate): v0.18.4, uploaded **2026-09-04**, author **"General Liquidity, Inc."**, licence "MIT OR Apache-2.0", requires_python >=3.9, a PyO3 binding over a Rust kernel also shipped as a CLI and npm package. — pypi.org/pypi/sharpebench/json — high (existence), **low (credibility)** — class(version,credibility)
10. **No GitHub repository, homepage, or any source location for SharpeBench was found**, and no independent citation, review, or third-party validation of its math surfaced. — medium (absence search, not exhaustive) — class(credibility)
11. **pypbo is not on PyPI at all** — `pypi.org/pypi/pypbo/json` returns 404. — high — class(version)
12. **OVERTURNS round 1's "maintenance unknown".** pypbo repo created 2016-08-28, **last pushed 2026-07-06** (~2 months before access) ⇒ actively maintained. — GitHub API — high — class(version,credibility)
13. pypbo licence = **AGPL-3.0**, confirmed by both the GitHub API licence field and the raw LICENSE file text ("GNU AFFERO GENERAL PUBLIC LICENSE Version 3"). — high — class(license)
14. pypbo contributors: esvhd 65 commits, two others with 1 each ⇒ **bus factor 1**. — high — class(credibility)
15. **`statsmodels.stats.multitest.multipletests` natively supports Benjamini–Hochberg (`fdr_bh`), Holm, Bonferroni, Šidák, Holm–Šidák, Simes–Hochberg, Hommel and further FDR variants.** — statsmodels.org official docs — high — class(method)
16. **`skfolio.model_selection.CombinatorialPurgedCV` is a documented, scikit-learn-native CPCV implementation** (purging + embargo, multi-path output). — skfolio.org official docs — high — class(method)

## Verdict per package

| package | verified licence | verified last release | bus factor | test/correctness signal | safe to depend on? |
|---|---|---|---|---|---|
| **purgedcv** | **MIT (2 sources)** | v0.1.6, 2026-09-04, within a continuous 19-release 4-month history | **1** (solo) | tests/ + CI + codecov exist (summary, not raw-read); 3 validation studies **self-reported**; JOSS/Zenodo scaffolding present but publication unconfirmed | **Conditional** — more credible than round 1 implied, but solo and self-validated. Read the actual test files before making it load-bearing. |
| **arch** | NCSA (PyPI) vs NOASSERTION (GitHub) — **unresolved** | v8.0.0; repo pushed 2026-08-10; 12-year history | Sheppard, but 1,559★/289 forks | **StepM = Romano–Wolf (2005) confirmed from source citation**; SPA = White/Hansen; RealityCheck = documented SPA subclass | **Yes for the statistical tests**, once the licence text is resolved. |
| **SharpeBench** | MIT OR Apache-2.0 (PyPI-stated) | v0.18.4, 2026-09-04 | unknown | **none found** — no repo, no tests, no citations, no source to inspect | **No.** A real package, but a black box from an unestablished publisher at a brand-new version. Wrong tool for adjudicating statistical significance. |
| **pypbo** | **AGPL-3.0 (2 sources)** | **not on PyPI**; GitHub pushed 2026-07-06 | **1** (solo) | not assessed | **Conditional** — actively maintained, but AGPL is a real constraint and it installs only via `pip install git+…`. |
| **statsmodels.stats.multitest** | BSD (not re-verified this run) | present as of statsmodels ≥0.15.0 | established org project | mature, widely used | **Yes** — covers the whole multiple-testing-correction need. |
| **skfolio** | **not verified this run** | docs reference v0.5.1-era module | not assessed | `CombinatorialPurgedCV` documented + sklearn-integrated | **Follow-up needed** — may remove the need for purgedcv for the CV piece entirely. |

## What I looked for and could not find
- **arch's actual LICENSE file text** — three path guesses 404'd; NOASSERTION vs NCSA conflict unresolved.
- **SharpeBench's source code** — no repo, homepage, or public source location anywhere. Its maths cannot be verified at all.
- **Raw inspection of purgedcv's test files**, and whether the JOSS paper was ever submitted or accepted (only that `paper/paper.md` exists in-repo).
- Whether **pypbo installs on modern Python** (no PyPI record to read `requires_python` from; no live install attempted).
- **scipy.stats** support for multiple-testing correction or block bootstrap — not checked.
- **skfolio's own licence, stars, bus factor, maintenance health** — only the CPCV feature was confirmed.
- arch's block-bootstrap class names (only `multiple_comparison.py` was fetched).
