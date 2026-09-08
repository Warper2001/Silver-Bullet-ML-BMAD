---
title: 'technical research: GitHub day-trading / quant repos — adopt one, or stay bespoke?'
type: technical
shape: select
topic: 'Best open-source day-trading / systematic-trading repositories on GitHub, assessed against this codebase'
decision: 'Adopt an existing open-source trading repo or library into Silver-Bullet-ML-BMAD — and if so, which one, for which layer — or keep everything bespoke.'
source: 'native run (bmad-deep-recon), 2 rounds, 6 digests'
status: complete-partly-superseded
superseded_by: '_bmad-output/validation_research_stack_20260908.md'
preset: standard
validation: normal
claims: {verified: 19, unverified: 3, disputed: 1, overturned: 0}
citations_check: 'PASS — 43 markers, 43 appendix rows, 0 dangling, 0 orphaned'
created: '2026-09-08'
updated: '2026-09-08'
---

# technical research: GitHub day-trading / quant repos — adopt one, or stay bespoke?

**Decision this research serves:** Adopt an existing open-source trading repo or library into this codebase — and if so, which one, for which layer — or keep everything bespoke.

---

> ## ⚠️ PARTLY SUPERSEDED — read `_bmad-output/validation_research_stack_20260908.md` alongside this
>
> The recommendation was implemented the same day on branch `feat/research-validation-stack`. Doing so **disproved three claims made below**:
>
> 1. **Recommendation 2 is withdrawn.** This report called the meta-labeling model's CV "near-certainly leaking" through "overlapping label horizons," and made re-running it under purged CV the top action. Measured: the dataset has **exactly zero** overlapping label windows — the strategy holds one position at a time (median hold 64 min), so every trade closes before the next opens. Purging has nothing to remove. The inference "triple-barrier labels ⇒ overlapping horizons" does not hold for single-position sequential trading.
> 2. **The deployed path was wrongly implicated.** `src/ml/train_tier2_meta_labeling.py:78` uses `TimeSeriesSplit`, which measures as *exactly* unbiased on this data. The real leakage (+0.12 to +0.24 AUC) sits in `src/ml/retraining.py:997`, `scripts/train_premium_regime_models.py:78` and `scripts/tune_regime_*.py` — the non-deployed infrastructure.
> 3. **The §0.1 grep was too blunt.** "Zero hits for purged CV" does not distinguish *needing* purged CV from *not needing* it.
>
> **What survives:** `arch` is the correct adoption, and the multiple-testing layer — not the cross-validation layer — is where the value was. On the 9-run DOE family: Hansen SPA p = 0.1233, Romano–Wolf superior set empty, and the best variant's Deflated Sharpe is 0.3884 against a naive PSR of 0.9456.

## Executive summary

**Adopt nothing at the framework layer. Adopt three small libraries at the statistics layer, this week.**

The recommended pick is **`arch`** (Kevin Sheppard) [25][26][27], with **`skfolio`** [32][33] and **`statsmodels`** [31] alongside. Total integration cost: three `pip install`s and no contact with the live trading path. Combined weighted score 3.45 / 5 versus 1.88 for nautilus_trader, the best framework candidate.

Three findings drive that answer:

1. **The gap in this repo is statistical, not architectural — and it is cheaply closable.** A repo-wide grep returns **zero** hits for purged CV, embargo, CPCV, deflated Sharpe, PBO/CSCV, White's Reality Check, Hansen SPA, Romano–Wolf, or Benjamini–Hochberg (§0.1). Every one of those is available today under a permissive licence from a maintained project. `arch.bootstrap` contains `SPA`, `StepM`, `RealityCheck` and `MCS`, and its `StepM` was confirmed **from source** to be the Romano–Wolf (2005) stepdown, by the literal Econometrica citation in its docstring [27]. This directly attacks the failure mode that has closed line after line here: many variants tried, one survives in-sample, dies out-of-sample.

2. **Nobody has solved the execution problem this shop actually died on — and that is a real finding, not a search miss.** `hftbacktest`'s `latency.rs` defines exactly two latency models: `ConstantLatency` and `IntpOrderLatency` (deterministic interpolation over recorded latency triples) [16]. There is **no stochastic or bursty latency model anywhere in open source** as of September 2026, confirmed in two independent rounds. The local ticksim was killed on precisely that axis. That was the field's frontier, not a local failure.

3. **"Best day-trading repo" by GitHub popularity is anti-correlated with usefulness here.** No high-star GitHub trading repo was found publishing a broker-verified or audited live track record — searches surfaced only signal-selling marketing sites [D-adoption]. The comparison-blog ecosystem around these repos is dense with recycled, unsourced claims. Even the curated lists carry corpses unqualified: `awesome-quant` still lists Blankly, which has had no commit since **2024-12-30** [37].

**Biggest caveat:** adopting `arch` will not make any strategy profitable. It is a better instrument for reaching "no", faster and with correct statistical size. Given this project's existing graveyard, that is the honest value proposition — and it is the one this shop's own firewalled power-gate work already implies it wants.

**Licence note, not legal advice:** the best-maintained *frameworks* are copyleft (nautilus LGPL-3.0 [4], pysystemtrade/lumibot/backtrader GPL-3.0 [8][14][38], backtesting.py AGPL-3.0 [11]); the best *validation libraries* are permissive (skfolio BSD-3-Clause [32], arch NCSA-style [26], statsmodels BSD). The layer worth adopting is also the legally cheapest one. Copyleft obligations generally attach on distribution, and AGPL additionally on network service — this codebase is private and undistributed, but it does run a Streamlit dashboard. Get sign-off before embedding anything copyleft; the recommendation above avoids the question entirely.

---

## 0. Requirements frame (project-derived — framing only, not evidence)

Per the select shape, requirements come from the project and the user, never from web research. This section is the baseline any candidate must beat. Facts here are local observations made 2026-09-08 in the `post-r3-options-research` worktree, **not research claims**, and carry no `[n]`.

### 0.1 What already exists locally

| Layer | Current state |
|---|---|
| Code volume | ~325 files / 110.8k lines under `src/` + `tools/`; a further 358 scripts / 89.5k lines at repo root. ~200k lines of bespoke Python. |
| Frameworks | **None.** `pyproject.toml` lists numpy, pandas, scipy, scikit-learn, xgboost, shap, streamlit, plotly, httpx, websockets, pydantic, apscheduler. A repo-wide grep for `vectorbt\|backtrader\|nautilus\|zipline\|backtesting.py\|pysystemtrade\|mlfinlab\|qlib\|freqtrade` returns **zero** hits. |
| Runtime | venv is **Python 3.12.3** (pyproject declares `^3.11`). Installed: numpy 2.4.6, pandas 2.3.3, polars 1.41.2, scikit-learn 1.8.0, scipy 1.17.1, xgboost 2.0.3. **`statsmodels` and `arch` are not installed.** |
| Backtest engine | Bespoke, bar-close, event-loop over 1-min OHLCV (`src/research/backtest_engine.py`). |
| Cost model | Flat `commission_per_roundtrip` subtracted at close (`backtest_engine.py:1031`). **No separate slippage term.** `tools/friction_rescreen.py` records that 17 of 18 studies assumed $4.80 RT with no slippage; live-verified is $2.00/ct RT limit, ~$3.00 market/stop. |
| Live execution | Bespoke, multi-venue, in production: TradeStation (REST + SIM mirror), ProjectX, Kraken. Several systemd-managed traders on a funded combine. |
| ML | Hand-rolled meta-labeling + triple-barrier (López de Prado concepts, own implementation): 127 `meta.label` / 82 `triple.barrier` mentions. XGBoost, sklearn. |
| Validation — **present** | Walk-forward (696 mentions), Monte Carlo (31), bootstrap (17), Bonferroni in 6 DOE scripts, sealed pre-registration with SHA-256 seals, and firewalled statistical **power gates** that structurally cannot see the answer (`tools/xsmom1_power_gate.py`, `tools/vrp_phase0_power_gate.py`). |
| Validation — **absent** | Zero hits repo-wide for **purged / embargoed CV**, **CPCV**, **deflated Sharpe**, **PBO / CSCV**, **White's Reality Check**, **Hansen SPA**, **Romano–Wolf**, **Benjamini–Hochberg**, **minimum backtest length**. (The 58 apparent `PBO` matches are substrings of `VWAPBounceStrategy`.) |

### 0.2 Hard gates (fail any ⇒ cut)

- **G1 — Embeddable, not a re-platform.** Live traders are deployed on a funded account and cannot be migrated.
- **G2 — Licence permits private embedding**, or the obligation is understood and accepted.
- **G3 — Alive.** Meaningful activity within ~6 months of 2026-09; Python 3.12 compatible.
- **G4 — Addresses a known failure mode here, not a new strategy claim.** Importing someone's strategy as evidence is an already-closed, falsified line. Strategy collections score zero.
- **G5 — Fits the instrument.** CME futures, 1-minute intraday.

### 0.3 Weighted preferences

| # | Criterion | Weight |
|---|---|---|
| W1 | Adds overfitting / multiple-testing control the repo lacks | 0.35 |
| W2 | Adds execution / fill / cost realism | 0.25 |
| W3 | Low integration cost | 0.15 |
| W4 | Ecosystem health / five-year regret risk | 0.15 |
| W5 | Reduces bespoke code surface | 0.10 |

---

## 1. Candidate screen

Twenty-three projects were examined across two rounds. Cuts, with reasons:

**Cut on G3 (dead or stagnant):**
- **backtrader** — last release 1.9.78.123, **2023-04-19**; confirmed independently on GitHub and PyPI [8]. `awesome-systematic-trading` tags it "dormant since 2024-08" [36].
- **Blankly** — no push since **2024-12-30**, ~21 months [37].
- **zipline-reloaded** — v3.1.1, 2025-07-19 [10]; ~14 months stale, outside the 6-month health bar. Tagged "dormant since 2024-02" by the curated list [36]. Not confirmed abandoned, but not a foundation to build on.
- **timeseriescv** — MIT but only 22 total commits, staleness inferred [43].

**Cut on G4 (does not address a known failure mode) and/or G5 (wrong instrument):**
- **freqtrade, jesse, hummingbot** — crypto-exchange-only by design; not CME futures.
- **bt / ffn** — READMEs contain zero "futures" or "intraday" hits; daily-frequency portfolio analytics [45]. `ffn` remains mildly useful for post-hoc performance stats only.
- **Qlib** — MIT, but equities-centric (CN/US), not futures-native, and its release picture is ambiguous: a v0.9.0 tag dated 2022-12-09 against 2,000+ later commits [15].
- **moonshot (QuantRocket)** — Apache-2.0 and alive, but coupled to the QuantRocket commercial platform and data subscription rather than usable standalone [44].

**Cut on G2 / G1 (licence or availability blocks the use case):**
- **mlfinlab** — the obvious "financial ML" answer, and **not open source**. Licensed "all rights reserved," non-commercial research only, no derivative works, commercial use requires a paid licence [1]; its public repo exists solely as a bug tracker, not a source distribution [2]. Two primary sources in the same repo agree. **Do not adopt, and do not resurrect an orphaned legacy `pip install mlfinlab`.**
- **PyBroker** — GitHub reports its licence as `NOASSERTION` [39]; alive and intraday-capable, but the licence must be read before it can be considered.
- **lumibot** — GPL-3.0 [38]; explicit futures support, but a copyleft framework, and a framework is not what is needed.

**Cut on G1 (re-platform, not embed) despite being excellent:**
- **QuantConnect LEAN** — Apache-2.0 and genuinely futures-capable at minute resolution, but a full platform to migrate into, not a component [12]. Its default futures slippage model is `NullSlippageModel` — zero — and `FutureFillModel` fills stops at bar close [13]: **the same limitation `backtest_engine.py` already has.**
- **pysystemtrade** — GPL-3.0, genuinely a systematic *futures* framework, actively maintained under a new org since Jan 2026 [14]. Cut because it is Rob Carver's whole methodology (daily-ish, portfolio-level, IB-coupled), not a component that fits an intraday MNQ system.
- **vectorbt** — v1.1.0, 2026-07-05, alive [9], but licensed Apache-2.0 **with Commons Clause**, a non-OSI "fair-code" licence; and the free edition has a documented memory blow-up (~24 GB peak for a 2-parameter / ~40,000-combination grid over ~39,000 bars, ~480× an equivalent C++ implementation) [35]. The vendor markets chunking and a Rust engine as the fix — in the **paid** product.
- **SharpeBench** — real, v0.18.4 uploaded 2026-09-04 by "General Liquidity, Inc.", MIT-or-Apache [28] — but **no source repository, homepage, tests, citations or independent validation could be located anywhere**. A closed-source Rust kernel from an unestablished publisher is the wrong instrument for adjudicating statistical significance. Cut on credibility.

**Five finalists** carried to scoring: `arch`, `skfolio`, `purgedcv`, `nautilus_trader`, `hftbacktest`.

---

## 2. Landscape & maturity

The live, credible field for this use case is much smaller than star counts suggest.

**nautilus_trader is the clear framework leader** and is unambiguously healthy: 28,617 stars, LGPL-3.0, pushed 2026-09-08, v2.0.0rc4 released 2026-09-02 alongside a maintained v1.231.0 from 2026-08-02 [4][3]. It requires Python >=3.12,<3.15 [3] — satisfiable here, since the venv is 3.12.3. It is a Rust-cored platform with a Python API, on a bi-verweekly release cadence.

**The rest of the "popular" tier is thinner than it looks.** Backtrader, the most-cited Python backtesting framework of the last decade, has shipped nothing since April 2023 [8]. Zipline-reloaded is slowing [10]. `awesome-systematic-trading` is notably honest about this, tagging dormancy inline ("dormant since 2024-02", "archived") [36] — whereas `awesome-quant`, though actively maintained and organised by category rather than star count [41], still lists Blankly unqualified despite 21 months of silence [37], and its Trading & Backtesting section carries numerous very-low-star, generic-AI-flavoured entries [D-fills-r2].

**Ecosystem health verdict:** for a five-year horizon, the permissive validation libraries are the safer dependencies than any framework. `arch` has a 12-year history (created 2014-08-29), 1,559 stars, 289 forks, last pushed 2026-08-10 [25]. `skfolio` has 2,359 stars, 244 forks, is not archived, and was pushed **2026-09-07** — the day before this research [32].

---

## 3. Validation & overfitting-control tooling

This is where the decision is won, and the evidence is unusually clean.

**`arch` is the answer to multiple-testing correction.** Reading `arch/bootstrap/multiple_comparison.py` directly confirms the classes `MultipleComparison`, `MCS`, `StepM`, `SPA` and `RealityCheck` [27]. Two details matter:
- `RealityCheck` is implemented as a thin subclass of `SPA`. This is correct, not a shortcut — White's Reality Check is a special case of Hansen's SPA in the econometrics literature [27].
- `StepM`'s docstring cites "Stepwise multiple testing as formalized data snooping. Econometrica, 73(4), 1237-1282" — i.e. it **is** the Romano–Wolf (2005) stepdown, settling a question round 1 left open [27].

Its licence needed resolving: GitHub's detector reports `NOASSERTION` while PyPI's classifier reports `NCSA`. Fetching the actual file settles it — `LICENSE.md` grants "permission... free of charge... to use, copy, modify, merge, publish, distribute, sublicense, and/or sell" subject to retention, binary-reproduction and no-endorsement clauses [26]. That is the NCSA / University-of-Illinois licence: **permissive, MIT/BSD-hybrid, safe to embed.** GitHub reports NOASSERTION only because the file carries a custom heading.

**`skfolio` is the answer to purged cross-validation.** BSD-3-Clause, 2,359 stars, pushed 2026-09-07 [32]. Its `skfolio.model_selection` module provides `CombinatorialPurgedCV`, `WalkForward` and `MultipleRandomizedCV`, all scikit-learn-native [33]. Note the boundary honestly: its `measures` module exposes `RatioMeasure` / `PerfMeasure` and **no Deflated or Probabilistic Sharpe Ratio** [33]. (A search result appearing to show skfolio computing DSR traces to a third-party project, `autoresearch-skfolio`, not to skfolio itself — do not cite it as a skfolio feature.)

**`purgedcv` is the runner-up, and more credible than it first appeared.** Round 1 flagged a four-day-old v0.1.6; the GitHub API overturns that — the repo was created **2026-05-15**, and v0.1.6 is release #19 in a continuous four-month series [22]. MIT, confirmed on two independent endpoints [24][22]. It uniquely covers **both** CPCV and Deflated/Probabilistic Sharpe. Against it: contributors are eslazarev (101 commits) and a bot — **bus factor 1** [23] — and its three validation studies are self-reported in its own README, not independently reproduced [D-validation-r2]. It has JOSS/Zenodo scaffolding in-repo, but publication was not confirmed.

**`pypbo`** implements PBO/CSCV, PSR, Minimum Backtest Length and DSR, and round 1's "maintenance unknown" is overturned — it was pushed 2026-07-06 [29]. But it is **AGPL-3.0**, confirmed from both the API and the raw LICENSE text [29], is **not on PyPI at all** (404) so it installs only via `pip install git+…` [30], and is also bus-factor 1.

**`statsmodels.stats.multitest.multipletests`** covers Benjamini–Hochberg (`fdr_bh`), Holm, Bonferroni, Šidák, Simes–Hochberg and Hommel [31]. It is a one-line upgrade over the raw Bonferroni currently used in the six DOE scripts, and Bonferroni is the most conservative correction available — on a family of correlated strategy variants it will be losing real power.

**Honest gap:** no independent methodological critique of DSR, PBO or CSCV was found in either round. Searches returned only the originating Bailey / López de Prado papers restating the methods' own caveats. That is absence of evidence, **not a clean bill of health**.

---

## 4. Execution & fill realism

The headline here is a correction and an absence.

**Correction — nautilus_trader has materially improved, and round 1's evidence is now stale.** Issue #2194, where maintainers described the fill model as handling only "very basic order-fill scenarios" with "no queue position simulation capability", was opened 2025-01-08 and **closed 2025-10-30** [5]. `RELEASES.md` shows what shipped since: Python-subclassable `FillModel`/`FeeModel`, `CompetitionAwareFillModel`, `VolumeSensitiveFillModel`, `BestPriceFillModel`, `fill_limit_inside_spread`, **L1 quote-based queue-position tracking**, and **L3 per-order-delta queue position** [6]. That is a genuine upgrade and the "no queue position" claim must not be repeated as current. The architecture is still closed, though: open issue #3943 (2026-04-28) records that "FillModelAny is a closed enum and the matching engine dispatches through it statically... custom fill behaviour can't be supplied from outside the Rust core" [7]. **Materially improved, not fully solved.**

**Correction — hftbacktest does have a CME path.** It ships an official `hftbacktest.data.utils.databento` module and a "Level-3 Backtesting" tutorial that builds a queue-position-aware L3 backtest from Databento CME Market-By-Order data, worked on **BTCM4** (CME-listed Bitcoin futures, June 2024 expiry), comparing L3 against L2 queue-position accuracy [18]. Round 1's "all examples are crypto" was wrong. Caveat: BTCM4 is a crypto-*linked* CME product, so this proves the mechanism against CME data without proving it on an equity-index future.

**The absence — and it is the most important finding in this report.** `hftbacktest/src/backtest/models/latency.rs` defines trait `LatencyModel` with exactly two concrete implementations: `ConstantLatency` (fixed) and `IntpOrderLatency` (linear interpolation over recorded historical request/exchange/response timestamp triples) [16]. **No stochastic or generative jitter model exists.** Two independent rounds searching for any open-source tool that models variable/bursty latency inside a P&L fill simulation found none. The nearest adjacent tools solve different problems: ABIDES models variable network latency but is an agent-based market simulator, not a historical-replay backtester; HFTPerformance benchmarks *your own* tick-to-trade latency, not exchange-side latency inside a fill model [D-fills-r1].

**What this means for this project:** the bespoke ticksim was killed on the seal's disclosed "variable/bursty latency... not corrected" blind spot. That blind spot is the open frontier of the field, not a local engineering failure. No adoption available today would have closed it.

**Everyone else is at bar-close, i.e. where this repo already is.** LEAN defaults to `NullSlippageModel` — zero slippage — and fills stops at bar close [13]. Backtrader offers only a volume-cap "Filler". Vectorbt has nothing. Note `hftbacktest` itself is slowing: last push **2025-12-23**, ~8.5 months before this research [17].

**Cost modelling** has no mature standalone answer. The only current candidate, `orderflow-metrics` (MIT, execution cost / price impact / Kyle's lambda / implementation shortfall), was pushed 2026-09-07 and has **9 stars** [40] — a lead, not a dependency.

**Data economics, for later:** Databento sells CME Globex MDP 3.0 MBO/MBP-10 on usage-based $/GB pricing; the Standard tier carries only the **last 1 month** of MBO history, with 16+ years reserved for Plus/Unlimited [20]. An exact $/GB figure could not be retrieved. Do **not** use CME's published January-2026 non-professional market-data fees ($5/month depth per exchange, $15 bundled) [21] to estimate this — those license real-time distribution, a different product from a historical backtest archive.

---

## 5. Implementation reality & the hype filter

**There are essentially no retrospectives.** Across two rounds and roughly 40 queries, **not one** first-person 6–12-month production account with concrete before/after numbers surfaced for nautilus_trader, vectorbt, LEAN, freqtrade or Qlib. The public discourse is launch announcements, tutorials, and SEO comparison blogs recycling each other's unsourced claims [D-adoption]. On the most substantive Hacker News thread, quant practitioners argued framework choice is secondary anyway — "the simulator for backtesting, integrating with a broker, etc. is such a small part of it" [34].

**Migration cost is unmeasured, which is itself a warning.** No account was found, anywhere, of moving an existing bespoke Python trading codebase onto any of these frameworks with time or effort figures. That is an evidence gap, **not** a finding that migration is cheap.

**No verified live track records exist.** Searching for a high-star GitHub trading repo backing its claims with broker statements or third-party audit surfaced only signal-selling marketing sites — none of them GitHub repos, none audited in a sense a quant team would accept [D-adoption]. Treat this as settled: **GitHub popularity carries no information about live profitability.**

**One stale complaint retired:** nautilus_trader's documentation/data-import gap (issue #532, referencing a function that no longer existed) was opened 2022-01-04 and is closed [D-adoption]. Do not repeat it. Current open issues (#3237 instrument auto-loading, #3899 account equity) suggest an actively developed but still-maturing codebase.

**Blind spot:** r/algotrading was unreachable — the search tool rejected `reddit.com` both as an allowed domain and via `site:`. Practitioner sentiment there is unsampled.

---

## 6. Cross-dimension insights

These emerge only from combining dimensions, and they are the substance of the recommendation.

**1. This project's two failure modes have opposite tool availability — and it has been working on the harder one.** Overfitting control is a *solved*, cheap, permissively-licensed problem that the repo has simply never picked up (§0.1 shows zero hits for the entire family). Fill realism under variable latency is *unsolved in open source*, confirmed from source in two rounds [16]. Effort has gone into the intractable half (a bespoke tick simulator, killed on latency) while the tractable half sat untouched. Reversing that allocation is the single highest-value move this research supports.

**2. The licence gradient runs opposite to the popularity gradient, in this project's favour.** The best-maintained frameworks are copyleft — nautilus LGPL-3.0 [4], pysystemtrade/lumibot/backtrader GPL-3.0 [8][14][38], backtesting.py AGPL-3.0 [11] — while the best validation libraries are permissive: skfolio BSD-3-Clause [32], arch NCSA [26], statsmodels BSD. The layer that should be adopted on merit is also the layer that raises no licence question at all.

**3. "Best day-trading repo" is the wrong question, and the search itself proves it.** Star count tracks tutorial appeal, not production value: no high-star repo publishes a verified track record; the comparison-blog layer is content-farm noise; curated lists carry corpses unqualified [37]; and mlfinlab, the single most-cited financial-ML library, is proprietary [1][2]. The useful artefacts turned out to be an econometrics package from an Oxford professor and a portfolio-optimisation library — neither of which appears in any "best day trading repos" listicle.

**4. The one framework worth revisiting is worth revisiting for a reason unrelated to why it is popular.** nautilus_trader's value here is not that it is a better trading platform; it is that it now has real queue-position fill modelling [6] — the only credible open-source route to intrabar realism besides hftbacktest. That is a *research-track* argument, not a re-platform argument.

---

## 7. Verdict — weighted decision matrix

Scores 0–5 against §0.3. Re-weight freely; the cells are the argument, not the totals.

| Criterion | Weight | **arch** | **skfolio** | purgedcv | nautilus_trader | hftbacktest |
|---|---|---|---|---|---|---|
| W1 overfitting control | 0.35 | **5** RC/SPA/StepM/MCS [27] | 4 CPCV, no DSR [33] | 4.5 CPCV + DSR [24] | 0 | 0 |
| W2 execution realism | 0.25 | 0 | 0 | 0 | 3 queue position, closed enum [6][7] | **4** queue + CME MBO, no stochastic latency [16][18] |
| W3 integration cost | 0.15 | **5** pip, offline only | 4 sklearn-native | **5** pip | 0.5 re-platform | 1.5 needs L2/L3 data + Rust |
| W4 ecosystem health | 0.15 | **5** 12 yrs, 1,559★ [25] | **5** 2,359★, pushed 09-07 [32] | 2 bus factor 1 [23] | **5** 28,617★ [4] | 3 MIT but 8.5mo stale [17] |
| W5 reduces bespoke code | 0.10 | 2 | 2 | 2 | 3 | 1 |
| **Weighted total** | | **3.45** | **2.95** | 2.83 | 1.88 | 1.78 |

### The pick

**`arch`, plus `skfolio`, plus `statsmodels`.** Three pip installs, all permissive, all confined to offline research scripts with zero contact with the live trading path.

- **`arch`** → the multiple-testing layer that does not exist here. `SPA` / `StepM` test a whole family of strategy variants at once with correct size — the correct instrument for a shop that runs many variants and picks survivors. It also supersedes the raw Bonferroni in the six DOE scripts, which on correlated variants is needlessly conservative.
- **`skfolio`** → `CombinatorialPurgedCV` and `WalkForward` for the XGBoost meta-labeling model. With triple-barrier labels and **no purging or embargo anywhere in the repo**, that model's cross-validation is near-certainly leaking: overlapping label horizons put information from a test fold's outcome window into its training folds. This is the highest-probability concrete defect this research surfaced.
- **`statsmodels`** → `multipletests` for BH-FDR where family-wise error is too strict.

Deflated Sharpe is the one wanted method none of these three provide. Either take `purgedcv` for it (MIT, but bus-factor 1 — read its tests first), or implement it directly from Bailey & López de Prado; it is a short, well-specified formula, and this team already writes firewalled statistical gates by hand.

### Runner-up, and when it wins instead

**`hftbacktest`** wins if the priority flips from "are these results real?" to "would these fills have happened?" **and** the team buys Databento CME MBO history. It is the only open-source engine with genuine queue-position simulation plus a documented CME MBO ingestion path [18]. Conditions to revisit: an explicit execution-realism mandate, a data budget, and tolerance for an 8.5-month-stale dependency [17]. **`nautilus_trader`** becomes interesting only if a *new* strategy line is started from scratch — never as a migration of the deployed traders.

### Strongest argument against the pick

`arch` will not find an edge. It is a sharper instrument for reaching "no", and this project already reaches "no" reliably — arguably its core competency. Adding it may just confirm existing conclusions at greater cost.

The counter is that `SPA`/`StepM` change the *shape* of the work, not just its verdicts: they let a whole family of variants be tested in one correctly-sized pass, instead of pre-registering and burning data on each variant sequentially. For a shop that has repeatedly closed lines as UNDERPOWERED, testing a family jointly is cheaper in data than testing members one at a time. That is an efficiency argument, not just a rigour argument.

### Cheapest reversibility hedge

Confine all three to `tools/` research scripts and never import them from `src/research/tier2_streaming_working.py` or any live path. Add them as a `[tool.poetry.group.research]` dependency group, not a main dependency. Removal is then deleting three lines and the scripts that use them. **Pilot scope:** re-run one already-closed negative result — the XSMOM-1 or VRP-1 family — through `SPA`, and re-run the meta-labeling model's CV with `CombinatorialPurgedCV`. If the purged CV materially changes that model's measured performance, the leakage hypothesis is confirmed and the adoption has paid for itself immediately.

---

## 8. Recommendations

1. **Install `arch`, `skfolio`, `statsmodels` into a research-only dependency group.** Confidence: **high** — all three verified for licence, activity and method content from primary sources [25][26][27][31][32][33]. Binds to: architecture spine (research-tooling constraint); the weekly config-change workflow's OOS gate.
2. ~~**Re-run the meta-labeling model's cross-validation with `CombinatorialPurgedCV`, before anything else.**~~ **WITHDRAWN 2026-09-08 — tested and false.** The dataset has zero overlapping label windows (single-position strategy), so purging is a no-op, and the deployed trainer's `TimeSeriesSplit` measures as exactly unbiased. Replaced by: **run `tools/validation/cv_leakage_probe.py` whenever a new training set appears** — it prints the overlap count that decides whether purging is needed at all, and measures the inflation of each CV scheme against a purged reference. See `_bmad-output/validation_research_stack_20260908.md` §4.3.
3. **Replace Bonferroni with `arch.bootstrap.SPA` / `StepM` in the DOE scripts, and add `multipletests` FDR where family-wise error is too strict.** Confidence: **high** [27][31].
4. **Do not adopt any trading framework.** Confidence: **high** for the negative — LEAN's default futures slippage is zero and stops fill at bar close [13], nautilus's fill architecture remains a closed Rust enum [7], and no migration-cost evidence exists in the entire literature searched [D-adoption]. Binds to: roadmap risk (removes a large speculative work item).
5. **Do not install mlfinlab, in any form.** Confidence: **high**, two primary sources [1][2]. It is proprietary; a legacy PyPI build would be both orphaned and licence-incompatible.
6. **Do not use SharpeBench.** Confidence: **medium-high** — it exists [28] but has no locatable source, tests, or independent validation. Never adjudicate significance with an unauditable binary.
7. **Park `hftbacktest` as a costed option, not a task.** Confidence: **medium** — CME MBO path confirmed [18], but on a crypto-linked contract, with an 8.5-month-stale repo [17] and unpriced data [20]. Revisit only under an explicit execution-realism mandate.
8. **Record the latency finding in the ticksim post-mortem.** Confidence: **high**, source-verified in two rounds [16]. No open-source tool models variable/bursty latency inside a fill simulator. The R3 kill decision was correct, and no available adoption would have rescued it — this should stop the question being reopened speculatively.

---

## 9. Open questions

| Question | What it would take |
|---|---|
| Are `purgedcv`'s CPCV and DSR implementations numerically correct? | Read its `tests/` directly and reproduce one published reference value. Not attempted — repo summary only. |
| Does `skfolio`'s `CombinatorialPurgedCV` handle *variable-length* triple-barrier label horizons, or fixed windows only? | Read the splitter source. Decisive for recommendation 2. |
| Does nautilus_trader's `LatencyModel` generate stochastic latency, or only replay? | Read `BacktestNode` latency internals. Only its `FillModel` was source-verified [6]. |
| What does Databento CME MBO actually cost for a usable MNQ history? | Their estimator requires an account [20]. |
| Is there real methodological criticism of DSR / PBO / CSCV? | Nothing found in two rounds; only the originating papers. Absence of evidence, not absence of criticism. |
| What does r/algotrading say about these frameworks? | Unreachable — the search tool blocks `reddit.com`. Needs a different surface. |
| Is `PyBroker` actually permissively licensed? | Read its LICENSE file behind the `NOASSERTION` flag [39]. |

---

## 10. Source appendix

| [n] | Supports | Publisher | Pub date | Accessed | Conf. |
|---|---|---|---|---|---|
| [1] | mlfinlab licensed all-rights-reserved, non-commercial, no derivatives | [hudson-and-thames/mlfinlab `license.rst`](https://github.com/hudson-and-thames/mlfinlab) | undated (current file) | 2026-09-08 | high |
| [2] | mlfinlab public repo is a bug tracker, not a source distribution | [hudson-and-thames/mlfinlab README](https://github.com/hudson-and-thames/mlfinlab) | undated | 2026-09-08 | high |
| [3] | nautilus_trader v1.231.0 (2026-08-02); Python >=3.12,<3.15 | [PyPI](https://pypi.org/project/nautilus-trader/) | 2026-08-02 | 2026-09-08 | high |
| [4] | nautilus_trader 28,617★, LGPL-3.0, pushed 2026-09-08, v2.0.0rc4 | [GitHub API](https://api.github.com/repos/nautechsystems/nautilus_trader) | 2026-09-08 | 2026-09-08 | high |
| [5] | Issue #2194 opened 2025-01-08, **closed 2025-10-30** | [GitHub issue](https://github.com/nautechsystems/nautilus_trader/issues/2194) | 2025-10-30 | 2026-09-08 | high |
| [6] | Shipped: CompetitionAware/VolumeSensitive fill models, L1+L3 queue position | [RELEASES.md](https://raw.githubusercontent.com/nautechsystems/nautilus_trader/develop/RELEASES.md) | 2026-09 | 2026-09-08 | high |
| [7] | `FillModelAny` is a closed enum; custom fills can't come from outside the Rust core | [GitHub issue #3943](https://github.com/nautechsystems/nautilus_trader/issues/3943) | 2026-04-28 | 2026-09-08 | high |
| [8] | backtrader last release 1.9.78.123, 2023-04-19; GPL-3.0 | [GitHub](https://github.com/mementum/backtrader) + [PyPI](https://pypi.org/project/backtrader/) | 2023-04-19 | 2026-09-08 | high |
| [9] | vectorbt OSS v1.1.0; Apache-2.0 **with Commons Clause** | [PyPI](https://pypi.org/project/vectorbt/) | 2026-07-05 | 2026-09-08 | high |
| [10] | zipline-reloaded v3.1.1, Apache-2.0 | [PyPI](https://pypi.org/project/zipline-reloaded/) | 2025-07-19 | 2026-09-08 | med |
| [11] | backtesting.py v0.6.6, AGPL-3.0-or-later | [PyPI](https://pypi.org/project/backtesting/) | 2026-07-22 | 2026-09-08 | high |
| [12] | LEAN is Apache-2.0, a full platform not a component | [GitHub](https://github.com/quantconnect/lean) | undated | 2026-09-08 | med |
| [13] | LEAN default futures slippage is `NullSlippageModel` (zero); stops fill at bar close | [QuantConnect docs](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts) | undated | 2026-09-08 | med |
| [14] | pysystemtrade GPL-3.0, futures framework, org moved Jan 2026, pushed 2026-04-02 | [pst-group/pysystemtrade](https://github.com/pst-group/pysystemtrade) | 2026-04-02 | 2026-09-08 | med |
| [15] | Qlib MIT; v0.9.0 tag (2022-12-09) vs 2,000+ later commits; equities-centric | [GitHub](https://github.com/microsoft/qlib) | 2022-12-09 tag | 2026-09-08 | low |
| [16] | `LatencyModel` has only `ConstantLatency` + `IntpOrderLatency` — **no stochastic model** | [latency.rs source](https://raw.githubusercontent.com/nkaz001/hftbacktest/master/hftbacktest/src/backtest/models/latency.rs) | current | 2026-09-08 | high |
| [17] | hftbacktest pushed 2025-12-23, 4,630★, MIT | [GitHub API](https://api.github.com/repos/nkaz001/hftbacktest) | 2025-12-23 | 2026-09-08 | high |
| [18] | Official Databento module + L3 CME MBO tutorial on BTCM4 | [hftbacktest docs](https://hftbacktest.readthedocs.io/en/latest/tutorials/Level-3%20Backtesting.html) | current | 2026-09-08 | high |
| [20] | Databento CME MDP 3.0 MBO usage-based $/GB; Standard tier = last 1 month history | [Databento pricing](https://databento.com/pricing) | undated | 2026-09-08 | med |
| [21] | CME non-professional real-time fees: $5/mo depth per exchange, $15 bundled | [CME January 2026 fee list](https://www.cmegroup.com/market-data/files/january-2026-market-data-fee-list.pdf) | 2026-01 | 2026-09-08 | high |
| [22] | purgedcv repo created 2026-05-15, pushed 2026-09-04, MIT (19 releases) | [GitHub API](https://api.github.com/repos/eslazarev/purged-cross-validation) | 2026-09-04 | 2026-09-08 | high |
| [23] | purgedcv contributors: eslazarev 101 commits + bot ⇒ bus factor 1 | [GitHub API contributors](https://api.github.com/repos/eslazarev/purged-cross-validation/contributors) | 2026-09-08 | 2026-09-08 | high |
| [24] | purgedcv v0.1.6, MIT, CPCV + Deflated/Probabilistic Sharpe | [PyPI](https://pypi.org/project/purgedcv/) | 2026-09-04 | 2026-09-08 | high |
| [25] | arch created 2014-08-29, pushed 2026-08-10, 1,559★, 289 forks | [GitHub API](https://api.github.com/repos/bashtage/arch) | 2026-08-10 | 2026-09-08 | high |
| [26] | arch `LICENSE.md` is NCSA-style permissive (free-of-charge grant + no-endorsement) | [GitHub API license](https://api.github.com/repos/bashtage/arch/license) | current | 2026-09-08 | high |
| [27] | `arch.bootstrap` has MCS/StepM/SPA/RealityCheck; StepM = Romano–Wolf (2005) per Econometrica citation | [multiple_comparison.py source](https://raw.githubusercontent.com/bashtage/arch/main/arch/bootstrap/multiple_comparison.py) | current | 2026-09-08 | high |
| [28] | SharpeBench v0.18.4, "General Liquidity, Inc.", MIT-or-Apache; no source repo found | [PyPI JSON](https://pypi.org/pypi/sharpebench/json) | 2026-09-04 | 2026-09-08 | high (existence) / low (credibility) |
| [29] | pypbo AGPL-3.0 (API + raw LICENSE), pushed 2026-07-06, bus factor 1 | [GitHub API](https://api.github.com/repos/esvhd/pypbo) | 2026-07-06 | 2026-09-08 | high |
| [30] | pypbo not published on PyPI (404) | [PyPI JSON](https://pypi.org/pypi/pypbo/json) | n/a | 2026-09-08 | high |
| [31] | `multipletests` supports `fdr_bh`, `holm`, Bonferroni, Šidák, Simes–Hochberg, Hommel | [statsmodels docs](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) | current | 2026-09-08 | high |
| [32] | skfolio BSD-3-Clause, 2,359★, 244 forks, created 2023-12-14, pushed 2026-09-07, not archived | [GitHub API](https://api.github.com/repos/skfolio/skfolio) | 2026-09-07 | 2026-09-08 | high |
| [33] | `skfolio.model_selection` has CombinatorialPurgedCV, WalkForward, MultipleRandomizedCV; measures has **no** DSR/PSR | [skfolio API docs](https://skfolio.org/api.html) | current | 2026-09-08 | high |
| [34] | HN practitioners: framework choice secondary to integration and edge | [Hacker News](https://news.ycombinator.com/item?id=44810552) | ~2025-08 | 2026-09-08 | med |
| [35] | vectorbt ~24 GB peak RAM, 2-param / ~40k-combination grid over ~39k bars | [GitHub issue #406](https://github.com/polakowo/vectorbt/issues/406) | 2022-03-07 | 2026-09-08 | high (currency unverified) |
| [36] | Curated list tags dormancy inline: zipline 2024-02, backtrader 2024-08, gobacktest archived | [awesome-systematic-trading README](https://raw.githubusercontent.com/paperswithbacktest/awesome-systematic-trading/main/README.md) | current | 2026-09-08 | high |
| [37] | Blankly pushed 2024-12-30 (~21 months dead) yet still listed unqualified in awesome-quant | [GitHub API](https://api.github.com/repos/Blankly-Finance/Blankly) | 2024-12-30 | 2026-09-08 | high |
| [38] | lumibot GPL-3.0, explicit futures support, pushed 2026-09-07 | [GitHub API](https://api.github.com/repos/Lumiwealth/lumibot) | 2026-09-07 | 2026-09-08 | high |
| [39] | PyBroker licence reports `NOASSERTION`; pushed 2026-09-07 | [GitHub API](https://api.github.com/repos/edtechre/pybroker) | 2026-09-07 | 2026-09-08 | high |
| [40] | orderflow-metrics MIT, impact/cost functions, pushed 2026-09-07, 9★ | [GitHub API](https://api.github.com/repos/twowaymind/orderflow-metrics) | 2026-09-07 | 2026-09-08 | med |
| [41] | awesome-quant actively maintained, organised by category not stars | [awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | current | 2026-09-08 | med |
| [43] | timeseriescv MIT, only 22 commits, staleness inferred | [GitHub](https://github.com/sam31415/timeseriescv) | undated | 2026-09-08 | med |
| [44] | moonshot Apache-2.0, pushed 2026-07-30, coupled to QuantRocket platform | [GitHub API](https://api.github.com/repos/quantrocket-llc/moonshot) | 2026-07-30 | 2026-09-08 | med |
| [45] | bt and ffn READMEs: zero "futures"/"intraday" hits; daily-frequency by design | [bt](https://github.com/pmorissette/bt) / [ffn](https://github.com/pmorissette/ffn) | 2026-09-07 | 2026-09-08 | med |

Findings drawn from a digest as a whole, rather than one source, are marked `[D-<dimension>]` and resolve to `digests/` in this folder.

---

## 11. Staleness map

Freshness bars from the technical pack (version/compatibility ≤1 mo · ecosystem ≤6 mo · landscape ≤12 mo · patterns ≤2 yr) plus the select shape's pricing bar (≤3 mo).

| Claim class | Re-check by | Which claims |
|---|---|---|
| **version / compatibility (1 mo)** | **2026-10-08** | [3][4][22][24][25][28][32][38][39][40] — every version and release-date claim. Fastest-ageing set in the report. |
| pricing (3 mo) | 2026-12-08 | [20][21] — Databento tiers, CME fee list. |
| ecosystem health (6 mo) | 2027-03-08 | [8][10][17][23][29][36][37][41][43][44] — activity, bus factors, dormancy tags. **[17] hftbacktest is the one to watch**: if it stays unpushed past ~2026-06 it moves from "slowing" to "dormant" and its runner-up status lapses. |
| landscape (12 mo) | 2027-09-08 | [12][14][15] — framework positioning. |
| method / patterns (2 yr) | 2028-09-08 | [26][27][31][33] — statistical method content; slow-moving. |

**Earliest re-check: 2026-10-08**, on the version claims.

Per the select shape: **a selection report older than two quarters should be refreshed before anyone acts on it.** This one expires **2027-03-08**. The recommendation is licence- and maintenance-sensitive, and [17] and [23] (an 8.5-month-stale repo and a bus factor of 1) are the cells most likely to move.
