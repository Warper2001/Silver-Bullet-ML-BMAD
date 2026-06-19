# Pre-Registration: DIX × GEX Swing-Horizon Interaction (SPX)

**Sealed:** 2026-06-19
**Researcher:** Alex
**Status:** PRE-REGISTRATION (sealed before the response variable — DIX/GEX → forward SPX return — has been examined)

---

## 1. Strategy / Hypothesis Description

**Name:** DIX-GEX-SWING — Dark-pool accumulation conditioned on dealer-gamma fragility, swing horizon.

**Origin:** Distilled in a BMAD party-mode roundtable (2026-06-19) from the GEX technical-research report (`_bmad-output/planning-artifacts/research/technical-institutional-options-flow-...md`) and the failed intraday-filter backtests (`spx-gamma-regime-mnq-filter-backtest-2026-06-19.md`).

**Core diagnosis being acted on:** DIX and GEX are *low-frequency* signals (daily, published after the cash close). Prior work tested them as **same-day intraday filters** on fast MNQ strategies and found nothing (best p≈0.075, year-confounded, opposite-signed across strategies). That is a **timescale mismatch**, not necessarily an absence of signal. This test moves the signal to its native swing horizon.

**Mechanistic thesis (directional prior):** The original SqueezeMetrics DIX thesis is that high DIX = dark-pool institutional accumulation → positive forward SPX returns over days/weeks. The *new, untested* claim is the **interaction**: the forward payoff to DIX accumulation is **larger when dealer gamma is more negative** (short-gamma / fragile regime, where dealers amplify moves) — institutions absorbing supply while dealers are positioned to chase = "compressed spring."

**Instrument for the test:** SPX cash index (the `price` column of the SqueezeMetrics file). DIX is built on SPX/SPY dark-pool flow, so the predictability test is run **SPX-native** to avoid an untested basis assumption. Deployment vehicle (if it ever passes) would be MES/ES; the index→futures translation is explicitly a *separate, later* question and is NOT part of this test.

**Data:** `dix_gex_seal_data_20260619.csv` (frozen snapshot, committed with this doc). Daily SPX: `date, price, dix, gex`. 3,806 sessions, 2011-05-02 → 2026-06-18. Source: SqueezeMetrics `https://squeezemetrics.com/monitor/static/DIX.csv`.

---

## 2. Frozen Parameters

All parameters set before the forward-return relationship has been examined.

### Signal construction (causal, no look-ahead)

| Parameter | Value | Rationale |
|---|---|---|
| `signal_date` | day **D** | `dix_D`, `gex_D` as published after D's cash close |
| `entry` | close of **D+1** | 1-day lag guarantees the signal is executable (DIX published post-close) |
| `forward_horizon` | **10 trading days** (ONLY) | One horizon, declared. Return = `price_{D+11}/price_{D+1} − 1` |
| `standardization` | trailing **252**-session z-score, min 60 | Causal (uses only data ≤ D). `DIX_z`, `GEX_z` |
| `FRAG` | **−GEX_z** | "Fragility": higher = more negative gamma = more dealer amplification |

### Primary test — continuous interaction regression (full sample)

```
R_10d  =  β0  +  β1·DIX_z  +  β2·FRAG  +  β3·(DIX_z × FRAG)  +  ε
```

- **Directional prior (one-tailed):** `β3 > 0` — the DIX payoff rises as fragility rises.
- **Standard errors:** stationary **block bootstrap**, block length = **10** (= horizon), 10,000 resamples. This is mandatory, not optional — daily-sampled 10-day forward returns overlap, so naive OLS SEs are fiction. Report one-tailed bootstrap p for β3 and its CI.
- **Effective N reported as** the count of *non-overlapping* 10-day blocks (~370 over full sample), not the row count.

### Secondary — "coiled-spring" cell (DESCRIPTIVE ONLY, not a decision gate)

| Parameter | Value |
|---|---|
| cell definition | `DIX_D > 80th pct` AND `GEX_D < 20th pct` (each, trailing 252) |
| reported | mean 10-day forward SPX return of **non-overlapping** episodes in the cell vs unconditional mean; block-bootstrap CI; count of independent episodes |

> This cell is reported for interpretability **only**. It does NOT enter the decision rule, so it adds **zero** to the multiple-comparisons family. (This is the explicit resolution of the Carson↔Mary debate: the slope is the test; the cell is the picture.)

### Economic translation

| Parameter | Value |
|---|---|
| `cost_per_side` | 1.5 bps (MES/ES placeholder; negligible vs a 10-day SPX swing) |
| economic effect metric | implied excess 10-day return for the top-fragility/top-DIX corner from the fitted surface |

### Sample period + sealed holdout

| Split | Range | Use |
|---|---|---|
| **In-sample (derive/confirm)** | 2011-05-02 → 2019-12-31 | fit regression, confirm β3 sign + significance |
| **Sealed holdout (single look)** | 2020-01-01 → 2026-06-18 | one look only; directional consistency check |

No standardization-window or threshold re-tuning after the holdout is opened. One look.

---

## 3. Decision Rule

This is **one** pre-registered hypothesis (one interaction coefficient, one horizon, one-tailed). Because exactly one test is committed, the conventional α=0.05 bar is honest — the 12-cell fishing expedition (4 quadrants × 3 horizons) is explicitly **forbidden**; that is what would have demanded Bonferroni.

| Outcome | Condition | Verdict |
|---|---|---|
| **PASS** | In-sample `β3 > 0` with one-tailed block-bootstrap **p < 0.05** **AND** implied corner excess return **≥ +0.8%** over 10d **AND** holdout `β3` **same sign with bootstrap CI lower bound > 0** | Real, orthogonal swing signal → build an SPX/ES swing-overlay spec |
| **FAIL** | In-sample `β3 ≤ 0`, OR in-sample one-tailed p ≥ 0.05, OR implied corner effect < +0.8% | DIX×GEX interaction dead. **No resurrection by sub-slicing, second horizon, or second cell.** |
| **AMBIGUOUS** | In-sample PASS but holdout flips sign or CI crosses zero | "Suggestive, not validated." Do not deploy. May only be revisited with **prospective** out-of-sample data, never by re-cutting this tape. |

The **+0.8% economic floor** exists so a statistically-cute-but-uneconomic blip does not count as a win (the prior power analysis put the minimum detectable effect at ~d≈0.45 ≈ +0.8–1.2% per episode at this N).

---

## 4. Sample-Size Note (honest N)

Full sample ≈ 3,806 daily rows, but 10-day forward returns overlap → ~**370 independent blocks**. The descriptive "coiled-spring" cell will contain only ~**30–60 independent episodes** in-sample — which is *why* it is descriptive-only and the **continuous slope is the primary test**: the slope uses the entire sample and degrades gracefully instead of shattering into a low-N cell. The holdout (2020+) yields ~15 independent cell-episodes, underpowered to confirm alone, so the holdout's job is **directional consistency**, not independent significance.

---

## 5. Integrity

This document and the frozen data snapshot are committed to git **before** the analysis script `backtest_dix_gex_swing.py` is created or run.

**What HAS already been examined (full disclosure):**
- Same-day *intraday* conditioning of MNQ YANK / MIM-NB **per-trade PnL** on same-day GEX/DIX terciles (a *different* response variable and timescale than this test).
- The frequency of negative-GEX days (~4.6% of 2025+ sessions) and descriptive moments of GEX for 2025+.
- The raw DIX/GEX/price columns have been loaded.

**What has NOT been examined — the response variable of THIS test:**
- Any relationship between DIX, GEX, or their interaction and **forward multi-day SPX returns**. The 10-day forward-return regression has never been run on this or any subset of this data. The in-sample years (2011–2019) have not been touched at all.

**Parameters NOT tuned by data.** The 10-day horizon, 252-session standardization, 80th/20th percentile cell, 2σ-style thresholds, and the `β3 > 0` directional prior are drawn from the SqueezeMetrics literature, the party-mode mechanistic argument, and standard practice — not from iterating on backtest output.

**Frozen data snapshot SHA-256:** `46994107673eb8f47b8ea09f2e073d07fbbfcee801491682403b1c9ab454ba2a`  (`_bmad-output/dix_gex_seal_data_20260619.csv`)
**Git HEAD at seal:** `8cb489b78219db1fbec0276425cad9949ae5860e`
**Required new code:** `backtest_dix_gex_swing.py` (to be created AFTER this commit)
**Existing data:** none beyond the frozen snapshot (signal and returns both come from it)

---

## 6. Sources

- SqueezeMetrics, *The Implied Order Book / DIX & GEX* white paper (2017) — original DIX→forward-return thesis.
- GEX technical-research report (this project, 2026-06-19): GEX is a weak *directional* edge, redundant with VIX; mechanism real; best used as a regime/vol term.
- Intraday-filter backtest (this project, 2026-06-19): GEX/DIX intraday gates on YANK/MIM-NB — no deployable edge; opposite-signed across strategies; motivates the timescale pivot.
- Party-mode roundtable (Dr. Quinn, Victor, Carson, Mary), 2026-06-19 — converged on this single-test design.
