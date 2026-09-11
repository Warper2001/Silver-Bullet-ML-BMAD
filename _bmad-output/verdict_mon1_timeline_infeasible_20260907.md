# MON-1 (Monday seasonality accrual) — NOT SET UP: TIMELINE-INFEASIBLE, 2026-09-07

**Trigger:** Alex, "set up the prospective accruals" — one of the two candidates named at the close
of the fan-out work.
**Outcome:** a power/timeline check was run **before** building any infrastructure. **MON-1 needs
7.7–9.0 years to reach a verdict. It is not being set up.** No tracker, no service, no timer.

---

## 1. What MON-1 was

`verdict_option6_calendar_seasonality_20260906.md` swept MNQ day-of-week seasonality across 1,300
sessions and closed FAIL under a Bonferroni-corrected bar (7 tests → 99.29th percentile). The
Monday cell was the near-miss: **+$56.49/day against a required +$58.97**, missing by $2.48.
`project_post_r3_options_pass_20260906` recorded it honestly as *"not established at this N, not
refuted"*, and noted that re-testing on the same data would be circular — the data that generated
the hypothesis is the only data. Prospective accrual was the correct remedy in principle.

## 2. Why it is not viable in practice

Measured from `mnq_1min_2025.csv` + `mnq_1min_2026_ytd.csv`, RTH 09:30–16:00 ET, 1 contract
($2/pt). **Only the volatility was computed — no Monday mean was re-derived**; the effect size is
taken from the sealed Option-6 verdict.

| | value |
|---|---|
| MNQ day-session σ, all days (N=341) | **$526.97** |
| MNQ day-session σ, Mondays (N=69) | **$403.13** |
| effect to detect (Monday vs zero) | +$56.49 |
| effect to detect (Monday minus baseline) | +$52.31 |

At 80% power, α = 0.05 two-sided:

| test | Mondays needed | **years** (52/yr) |
|---|---|---|
| Monday vs zero | 400 | **7.7** |
| Monday vs the all-days baseline | 466 | **9.0** |

**Signal-to-noise is 56 / 403 ≈ 0.14.** That is the whole story: the effect is small relative to
MNQ's daily dispersion, and only time — not finer sampling — buys power for a test about a mean.

**And 7.7 years is the optimistic figure.** The +$56.49 is an *in-sample* estimate that won a
7-cell search. Winner's-curse means the true effect is very likely smaller, which makes the
required span longer, not shorter.

## 3. Decision

**MON-1 is closed as timeline-infeasible** — the same disposition, for the same reason, that
`project_xsmom1_power_gate_20260907` and `verdict_vrp1_phase0_underpowered_20260907.md` reached:
**a design that cannot reach a verdict is closed, not run.** Building an eight-year accrual would
not be research; it would be a monument that invites someone to peek at partial results and read a
running mean as progress.

This is **not** a claim that Monday seasonality is absent. Nothing was tested. It is a claim about
resolving power: at this effect size and this dispersion, no accrual anyone here will wait for can
answer the question.

## 4. What would change the answer

Only a genuinely larger effect or a lower-variance construct — e.g. a Monday signal expressed on a
*risk-normalised* basis, or conditioned so the per-observation σ falls materially. Any such thing
is a **different hypothesis** and needs its own seal and its own power gate. **Do not re-open
MON-1 by lowering the bar or the power requirement.**

**Reversible on request:** if Alex wants the accrual built regardless — it costs one timer and
places no orders — it is roughly twenty minutes of work following the
`gap_velocity_prospective_tracker.py` pattern. The recommendation is not to, and the reason is
recorded above.
