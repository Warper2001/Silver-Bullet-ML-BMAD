# Pre-Commitment: Cross-Pair Divergence-Fade Survey (Gate 0)

**Date:** 2026-06-12
**Status:** PRE-COMMITTED — this document is committed to git BEFORE any Gate 0
simulation is executed. Frozen config: `pair_survey_config.yaml` (same commit).
Harness: `study_pair_divergence_survey.py` (same commit).

## Motivation

The MNQ/MES divergence-fade short (prereg `preregistration_stat_arb_short_combine.md`)
is the first strategy in 18 attempts to pass Gate 0 → Gate 1 → Gate 2 and is live
on combine account 23884932. This survey asks whether the same mechanism — fade a
5-bar beta-hedged divergence of a high-beta leg against its anchor — exists in
other pairs, applying every methodological lesson from the 17 failures and the
one success.

## Lessons encoded (binding rules)

1. **Cost floor first.** Grid is specified in dollar stops per traded contract;
   the WR gate is the per-pair commission-adjusted breakeven
   `BE_WR = (stop$ + comm) / (2 × stop$)`. Ranking metric is **net edge density**
   (avg net P&L × freq = $/day/contract), never gross PF.
2. **Both directions, always.** MNQ/MES had a decisive asymmetry (only
   short-the-outperformer worked). Every pair is simulated and gated in both
   directions independently; no direction is pre-selected.
3. **Pre-commit before results.** Pair list, test order, grid, gates, session,
   and decision rule are frozen in this document and the YAML before the first
   simulation. Any deviation requires a new pre-commit.
4. **Holdout discipline.** Dev window is hard-cut at 2026-02-28 inside the
   harness. All freshly downloaded data ≥ 2026-03-01 is sealed
   (chmod 444 + ACCESS_LOG) BEFORE the Gate 0 run. Gate 2, if any pair
   qualifies, is a separate one-shot prereg against those sealed slices.
5. **Roll artifacts.** Metals legs roll non-synchronously (SI vs GC vs HG vs PL
   calendars differ) — stitched-contract jumps create artificial divergence
   spikes. Entries are skipped on any roll date of either leg (dates frozen in
   YAML). The MNQ/ES control keeps an empty roll list to reproduce the
   validated template bit-for-bit.

## Pair universe and pre-committed test order

| # | Pair (A−B) | Traded leg (micro econ) | PV/pt | Comm RT | $40 stop in ticks | A-priori note |
|---|---|---|---|---|---|---|
| 1 | SI–GC | SIL ($1,000/pt, tick .005=$5) | $1,000 | $3.74 | 8 | The headline ask; silver = high-beta leg |
| 2 | RTY–ES | M2K ($5/pt, tick .10=$0.50) | $5 | $1.24 | 80 | Closest analog to validated edge; cheapest cost floor |
| 3 | YM–ES | MYM ($0.50/pt, tick 1=$0.50) | $0.50 | $1.24 | 80 | Tight coupling; freq gate is the risk |
| 4 | HG–GC | MHG ($2,500/pt, tick .0005=$1.25) | $2,500 | $3.74 | 32 | Weakest coupling; expect worst-month fail |
| 5 | PL–GC | PL full ($50/pt, tick .10=$5) | $50 | $4.00 | 8 | **INFORMATIONAL ONLY** — no micro exists; a Gate 0 pass does NOT qualify for Gate 1 without a separate liquidity/slippage study |
| 6 | MNQ–ES | MNQ ($2/pt) | $2 | $4.80 | — | **CONTROL** — must reproduce `study_stat_arb_short_only.py` |

Leg A = traded leg (analog of MNQ). Signal bars are full-size contracts
(SI/HG/PL/RTY/YM/GC/ES); micro economics applied at simulation time — the same
pattern as the existing GC-bars/MGC-economics prereg.

## Frozen specification

- **Signal:** rolling 60-bar OLS beta on 1-min price changes (ffill, clip [0,10]);
  divergence = 5-bar cum ΔA − β × 5-bar cum ΔB. Identical to template.
- **Session:** RTH 09:30–15:55 ET for ALL pairs (one variable changes vs the
  validated result: the pair — not the session). One labeled-exploratory
  sensitivity for metals pairs (08:30–13:25 ET, COMEX pit-aligned) reported in
  an appendix; it cannot promote a pair by itself.
- **Trade:** fade leg A only. Short A when div > +THRESH; long A when
  div < −THRESH. TP = full divergence reversion (1×), stop = STOP_MULT × div
  beyond entry, HOLD_MAX 30 bars, force-exit 15:55 ET, one trade at a time,
  skip entry if stop $ > $150/contract, skip entries on frozen roll dates.
- **Grid:** THRESH_USD ∈ {30, 40, 50, 60} per traded contract × STOP_MULT ∈
  {1.0, 2.0}. Converted to leg-A points as `THRESH_USD / point_value`.
- **Primary spec (frozen):** THRESH_USD = $40, STOP_MULT = 1.0× — exactly the
  validated MNQ 20pt × $2/pt primary.
- **Dev window:** 2025-05-01 → 2026-02-28, hard-coded in the harness.

## Gate 0 criteria (per pair-direction, evaluated at primary spec)

1. WR ≥ per-pair breakeven WR = (40 + comm) / 80
2. Avg net P&L > $0/trade
3. Frequency ≥ 1.0 trades/day
4. Median stop ≤ $150/contract
5. Worst-month WR ≥ 35%

## Decision rule — what qualifies for Gate 1 prereg consideration

A pair-direction qualifies iff ALL of:

- (i) primary spec passes all five Gate 0 criteria;
- (ii) ≥ 50% of the same-direction grid cells (8 cells) have positive avg net
  P&L (robustness, anti-cherry-pick);
- (iii) measured WR also clears the slippage-stressed breakeven
  `BE_WR_stress = (40 + comm + slip_stress) / 80` with slip_stress = 1 tick
  per side (frozen per pair in YAML);
- (iv) the pair is not flagged informational-only (PL–GC).

Qualifying pair-directions are ranked by **net edge density** ($/day/contract).
Qualification authorizes only the WRITING of a Gate 1 pre-registration; any
Gate 1/Gate 2 run and any deployment decision are separate, user-approved steps.
A failed pair is closed — no parameter sweeps beyond the frozen grid.

## Multiplicity disclosure

6 pairs × 2 directions × 8 cells = 96 looks. Mitigations: single frozen primary
spec per pair-direction (the grid is context, not selection), robustness clause
(ii), control pair excluded from selection, and the fact that Gate 0 is only a
filter feeding the existing sealed-OOS pipeline — nothing deploys from this
survey directly.

## Data provenance

- New downloads (2026-06-12, TradeStation v3 barcharts, `download_survey_1min.py`):
  SI, HG, PL, RTY, YM 1-min, 2025-05-01 → 2026-06-12, front-month stitched at
  the roll dates listed in the YAML.
- Existing: GC (`gc_1min_2025_2026.csv`), ES (`es_1min_2025_2026.csv`),
  MNQ (2025 + 2026 YTD CSVs).
- Before the Gate 0 run: rows ≥ 2026-03-01 from every NEW instrument plus GC
  are copied to `data/sealed_holdout/{root}_1min_holdout_20260301_plus.csv`
  and protected via `protect_holdout.py --init` (chmod 444 + ACCESS_LOG).
  Existing MNQ/ES seals are untouched.
