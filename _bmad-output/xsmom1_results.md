# XSMOM-1 — Verdict: UNDERPOWERED (terminal)

**Date:** 2026-09-07
**Pre-registration:** `_bmad-output/preregistration_xsmom1_cross_sectional_momentum.md` — sealed commit `1ff3735`, **before** the gate was written or run.
**Scripts:** `tools/xsmom1_power_gate.py`, `tools/xsmom1_feasibility.py`, `tools/backtest_xsmom1.py`
**Machine-readable:** `_bmad-output/power_verdict.json`, `_bmad-output/xsmom1_feasibility.json`

## Verdict

**UNDERPOWERED — terminal by the seal's stopping rule.** No strategy returns were computed. The 2025 panel remains **unspent** as an evaluation sample.

Three independent hard stops fired, any one of which is disqualifying:

| check | result |
|---|---|
| Achieved power π vs Θ_plaus | **10.6%** (needs ≥ 50% to proceed at all) |
| P2 — randomisation resolution | **26 valid null draws < 39** required |
| P4 — span floor | **0.82 years < 13.8** required |

## The power result

Panel: 12 roots, 43 evaluable weeks, span 0.82 years, k=8 frozen.

| quantity | value |
|---|---|
| sd(IC_t), from mismatched pairings | 0.3574 |
| SE — iid weeks (cross-check A) | 0.0545 |
| **SE — stationary block bootstrap (PRIMARY)** | **0.0611** |
| SE — circular shift, 26 valid (cross-check C) | 0.0545 |
| **MDE₈₀ (one-sided)** | **0.1519 mean IC → gross SR 3.13** |
| Θ_plaus (declared in seal) | 0.0243 mean IC → gross SR 0.50 |

The three standard-error estimators agree closely, so **the verdict cannot be moved by choosing a favourable null** — which the seal required as a gate property in its own right.

Power across the entire declared range, and beyond it:

| effect | achieved power |
|---|---|
| pessimistic, gross SR 0.25 | 7.4% |
| **central, gross SR 0.50 (declared)** | **10.6%** |
| optimistic, gross SR 0.75 | 14.7% |
| SR 1.00 (beyond the literature) | 19.8% |
| SR 2.00 (absurd) | 47.8% |

**At the declared effect the test fires 10.6% of the time against a 5% false-positive rate.** It is barely more likely to detect a real effect than to hallucinate one. Even an absurd SR 2.00 would be missed more often than caught. A null from this design would have carried essentially no information — and that is the point of running the gate first.

## Independent replication

The design analysis computed these quantities separately, before this implementation existed. The agreement is close: SE 0.0611 vs 0.0609, MDE 0.152 vs 0.151, power 10.6% vs 10.5%, valid draws 26 vs 26. Two independent routes to the same numbers.

## What it would take — the actionable output

Years of data needed for 80% power at the measured SE inflation:

| assumed gross SR | years required |
|---|---|
| 0.30 | 86.3 |
| 0.40 | 48.5 |
| **0.50 (declared central)** | **31.1** |
| 0.60 | 21.6 |
| 0.75 | 13.8 |
| 1.00 | 7.8 |

The fetchable history (~2000–2005 onward for most roots, ~20–25 years) reaches 80% power only at **SR ≥ 0.60** — the optimistic end of the literature. At the declared central 0.50 it would still be marginal.

**Width and length are complements.** Going from 12 to ~28 roots raises the plausible effect by ≈√(28/12) = 1.53×, which at 25 years brings 80%-power coverage down to roughly SR 0.39. The TSMOM-1 post-mortem asked for both a wider universe *and* a longer one; this quantifies why neither alone suffices.

## The second, independent blocker: capital

Pre-outcome, from contract specs and second moments only:

| | value |
|---|---|
| Binding root | **HG**, $6,412 weekly σ per contract |
| Minimum account — top-3/bottom-3 book at 10% vol target | **$1,387,128** |
| Minimum account — full rank-weighted book | **$5,548,511** |

**All 12 roots are full-size CME/CBOT/NYMEX/COMEX contracts; no micros exist in this universe.** Under the sibling carry spec's taxonomy any result here is **`theoretical`** and could not be labelled `executable` at this shop's account size — regardless of P&L. This is the Option-4 vehicle lesson again: the container, not the signal.

## Cost is *not* the binding constraint — recorded so nobody concludes otherwise

Estimated cost drag at k=8 is **0.084 Sharpe points ≈ 17%** of the declared plausible effect (34% doubled). Material, but far from fatal. The seal's cost-exclusion rule (>2.5% of weekly σ) would have removed **HE, ZM, ZW, ZC** — and the seal pre-committed **not** to apply it, for the measured reason that cutting breadth 12→8 scales the plausible effect by √(8/12)=0.816, costing more than the drag saved (0.376 vs 0.416). That pre-commitment held.

All cost figures are **estimates**. No fill data exists for any full-size contract in this project; the prior measured figures (MHG $4.00, PL $34.00, MNQ $2.24) are micros on a different fee schedule.

## Verification performed

1. **Firewall holds.** `ic_under_pairing` has exactly two call sites, both mismatched (circular shift with |s|>k; bootstrap-resampled indices). Calling it with the identity pairing raises `FIREWALL VIOLATION` — tested directly.
2. **Ordering provable.** Seal committed `1ff3735` before the gate existed; `git log` preserves the order.
3. **Refusal works.** `tools/backtest_xsmom1.py` exits **non-zero** on the UNDERPOWERED verdict without reading a single return — tested deliberately.
4. **Estimator robustness.** Verdict identical across all three SE estimators.
5. **No live-system contact.** Nothing under `src/research/`, no config, no services.

## Access declaration (seal §9)

The 2025 panel was used for **second moments only** — sd(IC_t) and cross-sectional covariance, computed exclusively under mismatched pairings. **The aligned signal→return pairing was never computed.** The panel therefore remains unspent as an evaluation sample for a future, adequately-powered seal. This exposure is recorded per the carry spec's prior-exposure rule.

## Disposition

**XSMOM-1 closed, UNDERPOWERED, terminal.** Per seal §8: no re-sweep of k, no new lookback grid, no added or dropped instruments, no switch to time-series construction, no sub-universe, no faster rebalance. Any variant needs a new seal with a new ID.

**The wider finding.** This is the first test in this project to establish *before spending data* that its design could not have detected a plausible effect. Eleven prior threads closed with "no edge"; this one closes with "this design could not have seen an edge if one existed, and here is the sample it would take." Those are different statements, and the machinery to tell them apart now exists and is reusable.

**Recommended next step is a data decision, not a strategy decision:** either acquire ~20–25 years across a materially wider universe (both, per the table above), or accept that cross-sectional commodity momentum is not testable at this shop's data and capital scale and stop paying attention to it. The capital blocker argues for the latter unless the account picture changes.
