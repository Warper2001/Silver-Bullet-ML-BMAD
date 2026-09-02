---
title: 'Cold-start book tolerance for the §A8.2 Part A replay path'
type: 'bugfix'
created: '2026-09-02'
status: 'done'
review_loop_iteration: 1
baseline_commit: 'efd379db5d4c3cb0c5e30ce3e9ce25decc026cf6'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `run_parity_gate` aborts ~70 s in. `sim.simulate` raises `BookInconsistency` on a
crossed market in the first Part A leg — instrument **42004936 (MNQM6)**, bid 29483.75 ≥ ask
29411.25, **290 ticks** deep, persisted 52.06 ms ≥ `config.MAX_TRANSIENT_CROSS_NS`. The ±90-min
windows carry no UTC-midnight book snapshot, so every window's book is reconstructed cold:
pre-window resting orders are never `ADD`ed, their `CANCEL`/`MODIFY` arrive as unseen-ref
no-ops, and stale levels sit in the book forever. `integrity.preflight_integrity` already
tolerates exactly this (it flags, never raises); the Part A sim does not, and one ghost aborts
the whole gate.

**Approach:** Teach `book._check_cross` to distinguish a *market* cross from a *stale-book*
cross by width. A cross wider than a derived bound is one side being a pre-window ghost:
count it, never arm the persistence timer, never `_fail`. Crosses within the bound keep the
existing 50 ms timer and stay fatal. Surface the counts through the manifest and the amendment
stub so a tolerated window is visible, never silent.

## Boundaries & Constraints

**Always:** the seal's 50 ms `MAX_TRANSIENT_CROSS_NS` is unchanged for real (bounded-width)
crosses. A ts-regression stays fatal on every path. Every tolerated cross is counted and
rendered in the stub — a reader must be able to judge whether the tolerance was warranted.
The width bound is **derived from the measurement below**, not hand-set (memory:
derive-don't-assert).

**Ask First:** the two questions settled below; and if implementation shows the width filter
alone does not clear the abort on the pre-roll windows, HALT rather than widening the bound
until it passes.

**Never:** changing `MAX_TRANSIENT_CROSS_NS`, `PART_A_MIN_N`, or the MAE / p90 / bias
tolerances. Making Part B or the standalone `simulate` CLI tolerant. Auto-excluding a flagged
window from Part A (record, never drop — AD-13).

### Measurement behind the bound (cold-folded, whole window)

| window | contract | crossed-BBO events | deepest | tail |
|---|---|---|---|---|
| w03 2026-06-25 | MNQU6 (normal) | 2,323 | **17 ticks** | nothing >20 |
| t01 2026-06-12 pm | MNQM6 | 1,153 | 49 ticks | nothing >50 |
| w00 2026-06-11 | MNQM6 pre-roll | 10,409 | 281 ticks | 1,777 >200 |
| t00 2026-06-12 am | MNQM6 pre-roll | 19,986 | **484 ticks** | 10,145 >200 |

A clean front-month book never crosses beyond ~17 ticks. The pre-roll MNQM6 windows (62–83%
front-month share, near expiry) reach 484 with over half their crosses beyond 200 ticks.

### Settled design questions

**(a) Width bound = `STALE_CROSS_MAX_TICKS = 50`.** Roughly 3× the widest cross ever observed
in a clean book (17), and below the 51–200 / >200 bands that carry the ghost population. A
new seal-bound `config` constant, documented as a **tolerance** parameter — it gates only
whether a cross is *timed*, and can never change a fill price, a verdict, or a P&L figure.

**(b) The warmup grace is dropped from scope.** The earlier draft proposed also downgrading
unseen-ref `BookInconsistency`. Investigation showed `apply_event` does **not** raise on an
unseen `C`/`M` — it increments `Book.unseen_cm_count` and continues. The cross-persistence
`_fail` is the only thing that aborted. One mechanism, not two.

**Known residual risk, accepted deliberately:** w00 still has 1,349 crosses in the 21–50 band
and t00 has 672, all of which remain inside the bound and therefore still armed. The bound is
set from clean-book evidence, not tuned until the run passes. If those still abort, that is a
real signal about the pre-roll windows — surface it, do not widen the bound.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| normal transient cross | bid ≥ ask, ≤ bound, clears < 50 ms | timer armed, clears, no raise | N/A |
| real persistent cross | bid ≥ ask, ≤ bound, ≥ 50 ms | `BookInconsistency` (unchanged) | raise |
| stale ghost cross | bid ≥ ask, > bound wide | counted in `stale_cross_count`, timer NOT armed, no raise | N/A |
| ghost then real | wide cross open, later a narrow one | wide ignored; narrow arms its own timer | N/A |
| no cross | bid < ask | nothing counted, timer reset | N/A |
| ts regression | `ts_event` < last folded | `BookInconsistency` (unchanged, fatal) | raise |

</frozen-after-approval>

## Code Map

- `src/ticksim/book.py:594` `_check_cross` — arms/advances the timer, calls `_fail` at
  `:624`; `Book` (`:286`) gains `stale_cross_count`; `_fail` at `:125`
- `src/ticksim/config.py:161` `MAX_TRANSIENT_CROSS_NS` — add `STALE_CROSS_MAX_TICKS` beside it
- `src/ticksim/sim.py:242` `Manifest` — surface `stale_cross_count` alongside `unseen_cm_count`
- `src/ticksim/parity/integrity.py:70` — the tolerant reference model (`_WARMUP_NS`,
  `warmup_unknown_ref`, its own non-raising cross state machine)
- `src/ticksim/parity/gate_cli.py` — stub already renders per-window integrity; add the count
- `tests/unit/test_ticksim_book.py` — existing strict-raise tests must stay green

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/config.py` — add `STALE_CROSS_MAX_TICKS = 50`, documented as a tolerance
  parameter (not decision-bearing), with the measurement rationale.
- [x] `src/ticksim/book.py` — `Book.stale_cross_count`; `_check_cross` skips the timer and
  counts when the cross width exceeds the bound; narrow crosses unchanged.
- [x] `src/ticksim/sim.py` — carry `stale_cross_count` into the `Manifest`.
- [x] `src/ticksim/parity/gate_cli.py` — surface the per-window stale-cross count in the stub.
- [x] `tests/unit/test_ticksim_book.py` — cover every I/O-matrix row.

**Acceptance Criteria:**
- Given a cold-reconstructed pre-roll window with a >bound ghost cross, when the gate runs,
  then it does not abort and the count appears in the stub.
- Given a ≤bound cross persisting ≥ 50 ms, when folded, then `BookInconsistency` still raises.
- Given the real 39-window map, when the gate runs, then it completes to a verdict (or fails
  for a reason that is not a stale ghost cross).

## Spec Change Log

### Review round 1 (2026-09-02)

All three reviewer subagents again terminated on an API session rate limit before
reporting; the pass was done inline. One `patch`, no `intent_gap`, no `bad_spec`.

**R1-1 (patch) — the new constant's docstring overstated its own safety.** It read
"can never change a fill price, a verdict or a P&L figure, so it is not seal-bound".
The first half is true, the conclusion is not: this is the guard deciding whether a
run against a given book is *admissible*. Widen it far enough and a genuinely corrupt
book stops aborting, Part A scores fills against it, and the MAE/verdict is wrong --
so it can change a verdict indirectly, by admitting a run that should have stopped.
An overstated "only a tolerance" invites exactly the future loosening this slice was
written to avoid. Reworded to require the same change discipline as a seal-bound
constant: re-derive from a fresh measurement, never widen to make a failing run pass.

**Two in-scope deviations, both justified and disclosed by the implementer:**
`check_invariants` also needed the width test (it raises on "crossed with no active
timer", which a tolerated ghost is by construction, so the end-of-run check would have
aborted precisely the runs this fix rescues); and `integrity.py` gained the per-window
`stale_cross_count` because `gate_cli` holds `IntegrityReport`s and no book, leaving no
other route to the count the stub must show.

**Semantics worth stating:** `stale_cross_count` counts **episodes**, not crossed-BBO
events (the derivation table above is in events). One long-lived ghost is one count.

**The accepted residual risk did not materialise.** Independent re-verification, cold-
folding whole real windows through `apply_event` (the actual abort path):

| window | contract | stale episodes | max *timed* cross | abort | check_invariants |
|---|---|---|---|---|---|
| w00 2026-06-11 | MNQM6 pre-roll | 35 | 20.29 ms | NONE | PASS |
| t00 2026-06-12 | MNQM6 pre-roll | 9 | 7.01 ms | NONE | PASS |
| t01 2026-06-12 | MNQM6 | 0 | -- | NONE | PASS |
| w03 2026-06-25 | MNQU6 clean | **0** | -- | NONE | PASS |

The deepest still-armed (21-50 tick) cross lasted 20.29 ms, comfortably inside the
50 ms seal, so the bound did not need widening and none was applied. The tolerance is
**inert on a clean book** -- w03 counts zero episodes -- which is the evidence that 50
is not too loose. 886 ticksim tests pass, `mypy --strict` clean, black clean in scope.


### 2026-09-02 — re-derived before approval

Parked draft rewritten. The original proposed two mechanisms; investigation removed one
(unseen refs never raise) and replaced the asserted width bound with the measured table above.
The reproduced abort also moved: it is instrument **42004936 (MNQM6, pre-roll)**, not the
earlier unreproducible w03 case.

## Verification

**Commands:**
- `PYTHONPATH=. .venv/bin/python -m pytest tests/ -k ticksim -q` — 868 prior tests stay green
  (run 2026-09-02: **886 passed**, 13 skipped — 868 prior + 18 new)
- `.venv/bin/python -m mypy --strict src/ticksim` — clean, no override (run 2026-09-02: clean)
- Real gate run against the 39-window map — completes past the pre-roll windows *(parent
  session's run; not executed here)*

**Real-window cold folds** (whole window, front-month-filtered + ts-clipped, `apply_event`
straight through — the abort reproduction path):

| window | instrument | events | stale episodes | max narrow cross | abort |
|---|---|---|---|---|---|
| w00 2026-06-11 pre-roll | 42004936 | 23,642,631 | 35 | 20.29 ms | none |
| t00 2026-06-12 am pre-roll | 42004936 | 19,084,234 | 9 | 7.01 ms | none |
| t01 2026-06-12 pm | 42004936 | 4,516,759 | 0 | 0 | none |
| w03 2026-06-25 (clean MNQU6) | 42004800 | 14,495,933 | 0 | 0 | none |

`check_invariants()` passes at the end of all four. The accepted residual risk (21–50-tick
crosses stay armed) did **not** materialise: the deepest *timed* cross on either pre-roll
window lasted 20.3 ms, well inside the seal's 50 ms. The clean windows count zero stale
episodes, so the tolerance is inert where the book is healthy. Counts are episodes, not
crossed-BBO events — the §"Measurement" table above counts events.
