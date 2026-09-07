# Pre-registration — XSMOM-1: wide-universe cross-sectional commodity momentum

**Seal ID:** XSMOM-1
**Date:** 2026-09-07
**Status:** sealed before any aligned signal→return statistic is computed.

## Why this seal exists, and why it is a new one

Eleven consecutive strategy threads in this project have closed with zero passes. A diagnosis raised in review is that several may have been **underpowered rather than genuinely null** — a screen that cannot detect a plausible effect returns "no edge" every time, and a run of such returns can look like evidence when it is only evidence of small samples.

This seal is the direct successor to **TSMOM-1**, whose own verdict document set the terms:

> "A future revisit of this general family (cross-market trend-following) would need either a genuinely different, larger, more diverse universe (the original academic papers used 58-67 markets spanning currencies/rates/equities/commodities, not 5 instruments dominated by precious metals) **or a materially different construction (vol-scaled sizing, cross-sectional rather than time-series)** — both out of scope for a re-sweep of this seal."

XSMOM-1 satisfies both conditions: a 12-root, 4-sector universe (vs 5 precious-metals-dominated roots) and a **cross-sectional** construction. Per TSMOM-1's stopping rule this is a **new pre-registration with a new ID**, not a re-sweep, and TSMOM-1/TSC-1 remain closed on their own terms.

**This seal's primary deliverable is a power verdict, not a strategy verdict.** The power gate in §4 runs first, is firewalled from the answer by construction, and is empowered to terminate the study before any strategy return is computed.

## 1. Universe — frozen

12 CME/CBOT/NYMEX/COMEX roots across 4 sectors:

| sector | roots |
|---|---|
| Energy | CL, NG, RB, HO |
| Industrial metal | HG |
| Grains & oilseeds | ZC, ZW, ZS, ZM, ZL |
| Livestock | LE, HE |

Gold and silver are **deliberately excluded** (inherited from the sibling carry spec's reasoning: a metals-heavy result would not replicate the population the literature is built on — and metals dominance is precisely what compromised TSMOM-1).

**No instrument is added or dropped after this seal on the basis of performance.** The only permitted exclusion is the pre-declared, non-performance-based cost rule in §6, whose disposition is also fixed here in advance.

## 2. Data

`data/commodity_curve/coverage-pilot-2025-20260906-v3/contract_bars.csv` — daily bars, 40,376 rows, 214 contracts, **calendar 2025 only: 251 trading days → 51 usable complete weeks.**
Companions: `contracts.csv` (expiry, point value), `coverage-preflight-20260906/symbol-details.json` (point value, tick increment).

Front contract = highest open interest with > 7 days to expiry. Returns are **same-contract log returns only**; roll days contribute zero return (no synthetic cross-contract jump), following the convention frozen in TSC-1/TSMOM-1.

**Known limitation, stated up front:** the span is 0.89 years. Deeper history (2000–2005 for some roots) is fetchable but is **not acquired for this seal**. Whether 0.89 years can support the test is exactly what §4 decides — it is not assumed either way.

## 3. Hypothesis, statistic, and the single frozen parameter

**Hypothesis (H1):** roots that have outperformed their peers over the trailing window continue to outperform over the following week — cross-sectional momentum.

**Statistic.** For each rebalance week *t* with all 12 roots present:
- `s_{i,t}` = trailing **k**-week cumulative same-contract log return, known at the Friday close of week *t−1*
- `f_{i,t}` = week-*t* same-contract log return
- `IC_t = Spearman(s_{·,t}, f_{·,t})` across the 12 roots
- **Θ̂ = mean(IC_t)**

Chosen over a portfolio Sharpe or a top-3/bottom-3 spread because: ranks absorb the universe's 3.6× volatility spread (LE ≈ 15% annualised, NG ≈ 54%) without a vol-estimation step this span cannot support; IC isolates ranking quality from sizing, caps and costs; and — decisively — sd(IC_t) is estimable from **mismatched pairings alone**, which is what makes the §4 firewall possible. A top-3/bottom-3 spread on N=12 discards half the cross-section and measures ~15% noisier for the same span.

**Frozen lookback: k = 8 weeks.** Not swept. For the avoidance of any later "we should have used a different k", the alternatives are disclosed here in advance: k=4 gives 47 evaluable weeks and sd(IC)=0.345; k=12 gives 39 weeks and sd(IC)=0.352 but is **structurally disqualified** — it admits only 14 valid null draws, so an exact randomisation test has minimum attainable one-sided p = 1/15 = 0.067 and *cannot* produce p<0.05 regardless of the data. k=8 gives 43 weeks, 26 valid draws (min p = 0.037), and sits inside the 1–12 month band the momentum literature documents.

**Direction is pre-declared: H1 is Θ > 0, tested one-sided at α = 0.05.** A negative Θ̂ is reported as "fail to reject", **never** as a significant reversal. Sealing the sign is worth ~20% in required span and may not be reversed after the fact.

## 4. THE POWER GATE — runs first, firewalled, and may terminate the study

### 4.1 Why it is firewalled by construction

The gate is credible only if it is *structurally impossible* to see the answer while deciding whether to look. `tools/xsmom1_power_gate.py` therefore computes IC **only under mismatched pairings** — block-bootstrap resampled week indices, or circular shifts with |s| > k. It asserts on every draw that the identity pairing is never evaluated. The aligned Θ̂ never exists as a value in that program. Everything the gate needs (sd(IC_t), SE, valid-draw count, span) is obtainable from mismatched pairings alone.

### 4.2 Standard-error estimators

- **PRIMARY — stationary block bootstrap** of whole weekly *return cross-sections*, re-paired against the fixed signal panel: geometric blocks, mean length 4 weeks (p = 0.25), **B = 10,000**, **seed 20260907**. Whole cross-sections move as units, so cross-sectional dependence is preserved exactly; blocking preserves serial dependence.
- **Cross-check A — iid weeks:** SE = sd(IC_t)/√T. Anti-conservative (ignores IC autocorrelation from overlapping lookbacks).
- **Cross-check C — circular shift.** **Valid shifts are s ∈ [k+1, T−k−1] only.** Shifts with s ≤ k re-create signal/return overlap and are not null draws — measured, this understated SE by 24% (0.256 vs an iid benchmark of 0.302), which would have silently halved the MDE. This trap is encoded as an assertion.

All three are reported as a sensitivity band. **The verdict must not depend on estimator choice**; if it does, that is itself a finding.

### 4.3 Pre-declared plausible effect

Anchored on the cross-sectional commodity momentum literature — Erb & Harvey (2006) ~0.6–0.75 gross SR on a similar-sized universe; Miffre & Rallis (2007) ~0.5–0.7; Fuertes/Miffre/Rallis (2010) momentum leg ~0.4–0.6; post-financialisation re-estimates (Bhardwaj/Gorton/Rouwenhorst; Kang/Rouwenhorst/Tang) ~0.15–0.35 — shaded down for three declared reasons: 12 roots vs 24–32 in most papers; a **weekly** horizon the literature does not support (documented commodity momentum is 1–12 months, and weekly commodity returns lean toward reversal); and post-2010 decay evidence.

| | gross SR | mean IC |
|---|---|---|
| Pessimistic | 0.25 | 0.0121 |
| **Central (declared Θ_plaus)** | **0.50** | **0.0243** |
| Optimistic | 0.75 | 0.0364 |

The central value is set at the **top** of the post-2008 range and the middle of the full-history range — deliberately generous, so that an UNDERPOWERED verdict cannot be dismissed as a rigged prior.

Conversion, measured on this panel: `SR_annualised = mean(IC)/sd(IC) × √52 = mean(IC) × 20.6`.

### 4.4 Decision rule

Let SE_B be the primary bootstrap standard error, `MDE₈₀ = (1.645 + 0.842) × SE_B`, and achieved power `π = Φ(Θ_plaus/SE_B − 1.645)`.

| condition | verdict | pre-committed action |
|---|---|---|
| π ≥ 0.80 | **POWERED** | proceed to the single evaluation run |
| 0.50 ≤ π < 0.80 | **MARGINALLY POWERED** | proceed; a null is reportable only as "no effect of magnitude ≥ MDE₈₀", never as "no effect" |
| π < 0.50 | **UNDERPOWERED** | **terminal.** No strategy returns, no P&L, no charts. The dataset is left unspent for a future adequately-powered seal. The power statement itself is the finding. |

Auxiliary hard stops, all pre-outcome:
- **P2 — randomisation resolution:** valid null draws ≥ 39 (so minimum attainable p ≤ 0.025).
- **P3 — holdout constructibility:** if a holdout's own MDE exceeds 2× the optimistic plausible SR, no holdout is constructed and the study is single-sample, which alone downgrades any PASS to INCONCLUSIVE.
- **P4 — span floor:** required span = ((1.645+0.842)×inflation / SR)². At the measured inflation of 1.21 and SR_optimistic 0.75 this floor is **16.0 years**.

### 4.5 Anti-gaming

Fixed here, in the same commit as the universe, k, costs and Θ_plaus, and **before** SE is computed:

1. **No re-running the gate at a different k after an UNDERPOWERED verdict.** k=8 is frozen; the k=4/k=12 figures are disclosed in §3 precisely so that switching cannot later be presented as new information.
2. **No switching between one- and two-sided** after the fact.
3. **No substituting cross-check A for the primary bootstrap** to shrink the SE.
4. **No sub-universe** (e.g. "energy only") to raise the effect — sub-universes have lower breadth and worse power; any such move is a post-hoc rescue.
5. **No moving to daily rebalance to "get more observations"** — measured, this changes detectable SR from 2.62 to 2.51, i.e. nothing. Span binds, not sampling frequency.
6. **No promoting a pooled instrument-week framing.** A cross-sectionally-neutral book is one bet per week; mean-IC and portfolio-Sharpe are the same test up to the ×20.6 factor. Pooling 552 instrument-weeks does not create 552 independent observations of a mean.
7. **Θ_plaus may not be revised downward after seeing the MDE.** A gate whose target moves is not a gate.
8. Program separation is enforced mechanically: the gate emits `power_verdict.json` carrying input and script SHA-256; `backtest_xsmom1.py` refuses to run unless that file exists, its input hash matches the data it is about to read, and the verdict is POWERED or MARGINALLY POWERED.

**UNDERPOWERED is a first-class recorded outcome**, in the same verdict enum as the rest — not a failure to produce one. Its deliverable is a power statement plus a numeric data requirement.

## 5. Construction (evaluated ONLY if the gate permits)

Rank-weighted, cross-sectionally demeaned, inverse-weekly-volatility sized, Σ|w| = 2; weekly Friday-close rebalance; minimum breadth floor of **≥ 9 roots across ≥ 3 sectors** (inherited from the sibling carry spec) or the week is skipped; instrument and sector risk caps; integer-contract rounding with a hysteresis band (no trade unless the target contract count changes by ≥ 1).

**Controls — load-bearing, not decorative.** Reported alongside any result, following TSMOM-1 §5: an **always-long** baseline and a **full-sample mean-sign** baseline (look-ahead by construction, non-tradeable). TSMOM-1's actual finding was that its signal *underperformed both* — every one of its 5 instruments rose over 2021-2025, so the baselines were identical and the signal's sign-flipping destroyed value. **Any PASS that cannot beat its own look-ahead-cheating control is reported as such, not presented as evidence of predictability.**

## 6. Costs — all figures ESTIMATES

**No fill data exists for any full-size CME/CBOT/NYMEX/COMEX contract in this project.** Every prior measured figure (MHG $4.00, PL $34.00, MNQ $2.24) is a micro or small contract on a different fee schedule and **does not transfer**. Any PASS is provisional on real fills.

`cost_RT = 2 × $2.50 fees + spread_ticks_RT × tick_value`, with spread tiers assigned from measured 2025 liquidity (front-contract median OI, daily volume, notional ADV) — not from memory.

| root | tick value | ticks RT | **cost RT** | cost as % of median weekly σ |
|---|---|---|---|---|
| HG | $12.50 | 1.0 | **$17.50** (estimate) | 0.27% |
| HO | $4.20 | 1.5 | **$11.30** (estimate) | 0.28% |
| RB | $4.20 | 1.5 | **$11.30** (estimate) | 0.39% |
| NG | $10.00 | 1.0 | **$15.00** (estimate) | 0.55% |
| CL | $10.00 | 1.0 | **$15.00** (estimate) | 0.58% |
| LE | $10.00 | 1.5 | **$20.00** (estimate) | 1.12% |
| ZL | $6.00 | 1.5 | **$14.00** (estimate) | 1.27% |
| ZS | $12.50 | 1.0 | **$17.50** (estimate) | 1.51% |
| HE | $10.00 | 2.0 | **$25.00** (estimate) | 2.69% |
| ZM | $10.00 | 1.5 | **$20.00** (estimate) | 2.81% |
| ZW | $12.50 | 1.5 | **$23.75** (estimate) | 3.15% |
| ZC | $12.50 | 1.0 | **$17.50** (estimate) | 3.43% |

Note the broker-root mapping trap: ZC/ZW/ZS/ZM/ZL/LE/HE are carried as C/W/S/SM/BO/LC/LH in `symbol-details.json`, and ZC/ZW/ZS use fractional (eighths) price display.

**Frozen stresses, decision-bearing:** doubled cost (fees *and* spread); uniform 2.0-tick tier; one extra tick on the four least liquid roots (HE, ZW, ZM, ZL); roll cost of one round turn per front-contract switch on any nonzero position (2025 counts: CL/HO/RB 13, NG 9, HE 8, LE 7, others 6), with the 5-business-day roll stressed at 5 and 15 days. Per the sibling carry spec's precedence, **a nonpositive doubled-cost net P&L is a FAIL, not INCONCLUSIVE.**

**Pre-declared cost exclusion rule, and its disposition — both fixed now.** The rule: exclude any root whose estimated RT cost exceeds 2.5% of its median weekly dollar volatility (which would remove ZC, ZW, ZM, HE). **This seal declares the rule and pre-commits to NOT applying it**, for a measured reason: excluding those four cuts cost drag 0.084 → 0.034 Sharpe points but cuts breadth 12 → 8, scaling Θ_plaus by √(8/12); net 0.376 vs 0.416 — *worse*. Instead: k=8 (halves turnover vs k=4), the hysteresis band, and cost-proportional weight damping `1/(1 + λ·cost_RT_i/σ_wk_i)` with **λ frozen at 1.0**.

Measured cost drag at k=8, rank-weighted: **0.084 Sharpe points** (0.168 doubled) ≈ **17% of Θ_plaus**. Material, not fatal — recorded so that a future reader does not mistake cost for the binding constraint.

## 7. Executable-feasibility gate — pre-outcome

Every root here is **full-size**; there are no micros in this universe. Minimum viable capital, from the binding root (HG, weekly σ ≈ $6,514/contract): a top-3/bottom-3 equal-weight book needs annualised portfolio σ ≈ $115k → **≈ $1.15M at a 10% vol target**; the full rank-weighted book's integer floor pushes this to **≈ $5–10M**.

If the capital required to hold the frozen book with integer contracts exceeds available capital, the result is labelled **`theoretical`** in the carry spec's three-level taxonomy and **cannot be labelled `executable` regardless of P&L**. This is knowable before a single return is computed and is recorded here as such.

## 8. Verdict enum and stopping rule

**HOLD / FAIL / INCONCLUSIVE / PASS-RESEARCH / UNDERPOWERED.** PASS-RESEARCH authorises no deployment, no live orders, no data purchase and no risk allocation.

**Stopping rule.** A FAIL or UNDERPOWERED verdict is **terminal for this seal**: no re-sweep, no new lookback grid, no added or dropped instruments, no switch to a time-series construction, no change of statistic, no sub-universe. A materially different variant needs its own new pre-registration with a new ID.

## 9. Access declaration

The 2025 panel is used by the power gate for **second moments only** — sd(IC_t) and the cross-sectional covariance structure, computed exclusively under mismatched pairings. **The aligned signal→return pairing is never computed** unless the gate returns POWERED or MARGINALLY POWERED. On an UNDERPOWERED verdict the 2025 panel therefore remains **unspent** as an evaluation sample. This exposure is recorded here per the sibling carry spec's rule that prior exposure to a period's prices must be declared.
