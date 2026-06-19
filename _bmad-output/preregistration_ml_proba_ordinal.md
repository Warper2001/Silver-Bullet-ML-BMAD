# Pre-Registration: ML-Probability Ordinal-Information Hypothesis — Prospective Test

**Generated:** 2026-06-19
**Experiment ID:** ml-proba-ordinal-v1
**Pre-registration commit:** (populate after `git commit`)
**Prospective cutoff:** entry date **≥ 2026-06-20** (strictly after the seal)

---

## ⚠️ Data-Observation Disclosure (CRITICAL — read before interpreting any result)

This test was designed **after** observing, in a party-mode review on 2026-06-19, that
`trader-s26`'s four trades on 2026-06-15 carried unusually high `ml_proba` (0.63–0.81 vs
the bot's typical 0.56–0.67) and 3 of 4 won. That single Monday (N=4, one day) is the
post-hoc seed of this hypothesis. **It has ZERO independent validity.**

Worse, the in-sample relationship across **all 22** de-duped s26 trades with `ml_proba`
(see [[project_trades_db_idempotency]] for the de-dup) is *seductive but hollow*:

| In-sample slice | N | WR | E$/trade | sum PnL |
|---|---|---|---|---|
| `ml_proba ≥ 0.63` | 10 | 70% | **+$156.3** | +$1,563 |
| `ml_proba < 0.63` | 12 | 50% | +$8.4 | +$101 |
| bottom tercile | 7 | 57% | +$55.5 | +$389 |
| mid tercile | 7 | 43% | +$0.1 | +$0.5 |
| top tercile | 8 | 75% | +$159.4 | +$1,275 |

**The binary +$156 vs +$8 split looks like an edge. It is not.** The rank correlation
is **Spearman ρ = +0.060, two-sided p = 0.791** (one-sided p(ρ>0) = 0.396) — statistically
indistinguishable from zero — and the terciles are **non-monotonic** (the mid tercile is
the worst). The apparent lift is a **fat-tail artifact**: a few large winners happen to sit
in the high-proba group. A rank test correctly discounts this, which is exactly why the
PRIMARY metric below is Spearman ρ, not the binary cut.

**This pre-registration is the only mechanism that creates genuine validation.** If the
prospective rank correlation (on trades entered AFTER this commit) is significantly
positive, that is independent evidence that `ml_proba` carries usable ordinal information.
If it is not, the 2026-06-15 cluster was variance/regime and `ml_proba` is not a
selectivity dial beyond its current binary gate.

---

## Hypothesis

**H₁ (alternative):** Within an ML-gated bot, a trade's `ml_proba` carries *ordinal*
information about realized PnL — higher-probability trades earn more, by rank — measured
prospectively. Specifically: Spearman ρ(ml_proba, net_pnl) > 0 with one-sided p < 0.05
over N ≥ 30 prospective trades, **and** top-tercile mean net PnL > bottom-tercile mean.

**H₀ (null):** The in-sample appearance is a fat-tail artifact (ρ ≈ 0, p = 0.79).
Prospective ρ is not significantly positive; `ml_proba` does not rank-order PnL.

---

## Frozen Analysis Spec (do NOT adjust after any prospective trade)

| Element | Rule |
|---|---|
| Unit of analysis | **Per model**, not pooled — `ml_proba` scales differ across models |
| Primary model | `trader-s26` (the model that produced the observation) |
| Secondary model | `trader-yank` (Tier2 meta-labeling; reported in parallel, own scale; does NOT affect the s26 primary verdict) |
| Pair definition | (`ml_proba`, realized `net_pnl`) for each **completed** trade |
| In-scope trades | entry timestamp **strictly after the seal commit** → operationally entry date **≥ 2026-06-20 UTC** |
| Data source | `data/trades.db` `trades` table (now idempotent; see hashes), filtered by `trader_id` and cutoff |
| Primary metric | **Spearman rank correlation ρ(ml_proba, net_pnl)** — rank-based, robust to fat tails (this is the load-bearing choice) |
| Confirmatory metric | top-tercile mean net PnL > bottom-tercile mean net PnL |
| Descriptive only | the post-hoc `ml_proba ≥ 0.63` binary split — reported, **never** a decision input |
| Population | **ALL** in-scope trades. No restricting to winning days/regimes/direction; no excluding any trade; no day/subgroup carve-outs |

The threshold value `0.63` is **deliberately NOT asserted as a deploy threshold.** Per
[[feedback_derive_dont_assert_one_knob]], if this test passes, the *next* step is a
**separate** config-change pre-registration that DERIVES an optimal threshold on a
train/holdout split — this test only asks whether ordinal information exists at all.

---

## Prospective Tracking Protocol

The bots already log `ml_proba` and `pnl` per trade. No new instrumentation is needed.

To read the running verdict at any time:
```bash
.venv/bin/python analyze_ml_proba_hypothesis.py --trader trader-s26 --cutoff 2026-06-20
.venv/bin/python analyze_ml_proba_hypothesis.py --trader trader-yank --cutoff 2026-06-20
```

### No adjustments allowed (any violation voids this pre-registration)
- Changing the cutoff, the primary metric, the tercile definition, or the `0.63` reference.
- Restricting to a direction, day-of-week, regime, or "winning days" after seeing trades
  (this project's documented cardinal sin — see [[feedback_iteration_loop_pattern]]).
- Excluding trades retroactively, or pooling s26+yank to rescue a per-model FAIL.
- Repeated peeking that drives the decision — exactly **one** interim look is permitted
  (the early-stop at N≥15); the final verdict is taken once, at N=30.

---

## Decision Rules (Pre-committed, Immutable After Seal)

Applied **per model**; primary verdict is `trader-s26`.

### Early stopping (bearish, single interim look)
After **N ≥ 15** prospective trades: if Spearman ρ ≤ 0 → **HALT, declare FAIL.** No
ordinal information; continued tracking wastes time.

### Final verdict (at N = 30 prospective trades)
| Criterion | Threshold | Action if not met |
|---|---|---|
| N prospective trades | ≥ 30 | Continue tracking (PENDING) |
| Spearman ρ(ml_proba, net_pnl) | > 0, one-sided p < 0.05 | FAIL |
| top-tercile E$ > bottom-tercile E$ | True | FAIL |

**If both met → PASS.** `ml_proba` carries ordinal information. Action: open a **separate**
config-change pre-registration to *derive* an optimal threshold (train/holdout), governing
the live ML-gated bots. This document does **not** itself change any config.

**If not met → FAIL.** The 2026-06-15 cluster was variance/regime. Record the result,
update memory, and do not re-test on a different s26 subgroup.

---

## Frozen Parameters Snapshot

```yaml
# ML-Probability Ordinal Hypothesis — Prospective Test (FROZEN)
# Pre-registration: _bmad-output/preregistration_ml_proba_ordinal.md
# DO NOT MODIFY after the pre-registration commit.
version: "ml-proba-ordinal-v1"

primary_model: trader-s26
secondary_model: trader-yank          # parallel, own proba scale, not pooled
data_source: data/trades.db           # trades table, idempotent (UNIQUE natural-key index)
prospective_cutoff: "2026-06-20"      # entry date >= cutoff counts (strictly after seal)

primary_metric: spearman_rho          # rank-based; robust to fat tails (load-bearing)
confirmatory: top_tercile_E$ > bottom_tercile_E$
descriptive_only_cut: 0.63            # reported, NEVER a decision input; NOT a deploy threshold

n_min: 30
interim_look_n: 15                    # single early-stop look; HALT/FAIL if rho <= 0
final_pf_alpha: 0.05                  # one-sided p(rho>0) < 0.05
```

---

## Integrity Hashes

| Hash | File | Value |
|---|---|---|
| (a) analyzer SHA-256 | `analyze_ml_proba_hypothesis.py` | `89ce9f10a9041c94ba3209d454dfbce6251b1f13c05d96248ac00ce1871d22ad` |
| (b) Git HEAD at seal time | — | `da78f5a1223434e4955559152cfbce62b2ff0031` |

*Hash (a): freezes the verdict logic (metric, decision rule, cutoff handling).*
*Hash (b): any s26/yank trade dated on or before the seal is in-sample and excluded from the prospective count.*

---

## Scope Constraint

Knowledge-generating only. This pre-registration changes **no** trading config. YANK exits
remain FROZEN (see [[project_yank_sl_tp_ml_grid_20260613]]); s26 remains a dead-strategy
data collector (see [[project_s26_combine_verdict]]). A PASS authorizes only a *future,
separate* threshold-derivation pre-registration — not any live change here.
