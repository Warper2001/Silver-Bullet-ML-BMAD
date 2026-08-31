---
title: 'ticksim parity/part_b.py — the synthetic invariant battery runner (run_part_b)'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 1
baseline_commit: '2bcd40d'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<!-- SPLIT (2026-08-31, planning): the deterministic ≥1000-order *generator*
     (`generate_synthetic_orders` — random ts/side/kind/size, realistic limit
     prices from a single BBO-sampling pass) is deferred to part_b slice 2.
     THIS slice is the battery runner: given a synthetic OrderIntent list + a
     window source, simulate, run every invariant, and produce the Part B
     verdict. -->

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Prereg §A8.2 Part B runs ≥1000 synthetic orders through the simulator and requires **all six invariants to hold 100%** (any violation = Part B FAIL, regardless of Part A). `parity/invariants.py` implements them as pure post-hoc `check_*` functions; three sub-parts are `fills.py` / `OrderTracker` / `sim` *construction guarantees* (invariant 5's liquidity half, invariant 4's queue time-series monotonicity, invariant 6's AD-20 merge ordering) with unit-test homes. Nothing yet runs the battery against real sim outputs at scale.

**Approach:** add `src/ticksim/parity/part_b.py` — `run_part_b(intents, source, *, config=PRIMARY) -> PartBResult`: one `sim.simulate` over the whole synthetic intent list, then for every `(OrderIntent, OrderOutcome)` pair run `invariants.check_order`, collecting every `InvariantViolation` into a structured result. `verdict == "PASS"` iff zero violations **and** `n_orders >= PART_B_MIN_ORDERS`. No book replay — invariant 5's liquidity half is a `fills.py` construction guarantee, treated exactly as `invariants.py` already treats invariant 4's series and invariant 6's ordering (human decision, loopback 1 — see the Spec Change Log). The synthetic-order generator is slice 2.

This slice also lands the shared **`parity/_bookwalk.py`** (`BookReplay` — the bounded read-only book replay lifted out of `part_a_runner._touch_at`, which is rewritten as a thin wrapper). `part_b` does not use it; the extraction stands because it removes a duplicated fail-closed guard and the slice-2 generator will need BBO sampling.

## Boundaries & Constraints

**Always:**
- **`run_part_b(intents: Sequence[OrderIntent], source: BookEventSource, *, config: SimConfig = PRIMARY) -> PartBResult`.** One `sim.simulate(source, intents, config, valid_intervals)` call — `valid_intervals` is the single half-open `[max(0, lo − PART_B_WINDOW_PAD_NS), hi + PART_B_WINDOW_PAD_NS)` spanning every `intent.submit_ts_ns` (`PART_B_WINDOW_PAD_NS` a module constant, default 5 min ns; a keyword `pad_ns` overrides it, mirroring `run_part_a`). `source` is a single-instrument re-iterable L3 stream (front-month filtering is the caller's job — `sim` raises `IntentLogError` on a multi-instrument stream). `source` is consumed exactly once (no book replay).
- **`intents` are all fresh submits.** Every `intent.action` must be `IntentAction.SUBMIT` — a `replace` / `cancel` reuses an `order_id` and Part B's synthetic battery is standalone orders only. Any non-`SUBMIT` → `PartBError` naming the order_id.
- **Pair up.** Join each `OrderOutcome` to its `OrderIntent` on `order_id`. `simulate` returns exactly one outcome per submitted intent; a missing, duplicate, or foreign `order_id`, or `len(outcomes) != len(intents)`, → `PartBError` (a `sim` bug — the battery cannot certify an incomplete run). `_pair_outcomes` raises on any mismatch, so downstream every intent has exactly one outcome (no `n_outcomes` field — it would always equal `n_orders`; round-2 review).
- **Per-order invariants.** For each pair call `invariants.check_order(intent, outcome)` — covers invariant 1 (no price improvement vs the arrival touch), 2 (never through the limit), 3 (fill ts ≥ arrival), 4 (queue-position *endpoint* structural check), 5a (cumulative partials ≤ size), 6 (fill-ts causal trace), plus the `time_to_fill` / `adverse_selection` consistency checks. Catch the `InvariantViolation`; if its message contains `"join mismatch"` re-raise as `PartBError` (a mispaired intent/outcome makes the whole "all six hold" verdict meaningless); otherwise record a `Violation(order_id, label, message)` and **continue** the sweep — the result reports *all* violating orders, not just the first. `label` is `"1"`…`"6"` / `"time_to_fill"` / `"adverse_selection"` parsed from the message prefix, `"unknown"` if unrecognised; a `test_ticksim_parity_part_b.py` test must trip a **real** `check_*` (not a fake message) and assert the label so the `invariants.py` ↔ `part_b` message coupling is pinned.
- **`sim` raising `InvariantViolation`.** `simulate` itself can raise `InvariantViolation` (a simulator invariant broke mid-run). Catch it around the `simulate` call and record it as `Violation("", "sim", message)` with `verdict == "FAIL"` — a sim-raised breach IS a Part B failure, not a crash. `IntentLogError` / `BookInconsistency` / `ValueError` / `OrderStateError` from `simulate` still propagate (structural faults, not invariant breaches).
- **Invariant 4 series, invariant 5 liquidity, invariant 6 merge-ordering** are **construction guarantees**, not re-verified here: inv-4 series → `OrderTracker` (`decrement_queue_ahead` floors at 0 / never increments; `set_queue_position` once-only), `tests/unit/test_ticksim_orders.py`; inv-5 liquidity → `fills.py` (`_walk_book` emits only for levels with size > 0; a passive fill only once cumulative trade volume at the price exceeds queue-ahead), `tests/unit/test_ticksim_fills.py`; inv-6 merge ordering → `sim` (monotonic clock + `(ts_event, class_rank, sequence, source_index)` stable merge), `tests/unit/test_ticksim_sim.py`. `PART_B_COVERAGE_NOTE` (a module constant string on `PartBResult`) states verbatim which of the six are post-hoc-verified here and which are construction-guaranteed with their test home — so a gate reader knows exactly what the ≥1000-order battery certifies. The note also records that `check_fill_latency` compares against `outcome.arrival_ts_ns` (the *original* arrival for a priority-preserving `replace`), a documented `invariants.py` refinement of the literal prereg "submit + latency" wording.
- **Verdict.** `PartBResult(n_orders, n_fill_events, violations: tuple[Violation, ...], verdict, reason, coverage_note)`. `n_fill_events = Σ len(outcome.fills)` (fill *events*, not contracts — the name says so). `verdict == "PASS"` iff `not violations and n_orders >= PART_B_MIN_ORDERS and n_orders > 0`. Any violation, or `n_orders == 0`, or `n_orders < PART_B_MIN_ORDERS` → `"FAIL"` with `reason` naming the cause(s) and, on a violation FAIL, a per-label count breakdown (sorted). `violations` is sorted by `(order_id, invariant)` for deterministic FAIL reports. `Violation(order_id: str, invariant: str, message: str)`.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. New edges (parity-sibling resolver, already in place): `PERMITTED_INTERNAL_EDGES["_bookwalk"] = {"book", "events"}`; `PERMITTED_INTERNAL_EDGES["part_b"] = {"sim", "orders", "config", "invariants", "events"}` (`events` only for the `BookEventSource` type annotation); `part_a_runner`'s set gains `"_bookwalk"`.

**Ask First:**
- (resolved, loopback 1, 2026-08-31) invariant-5 liquidity is a construction guarantee — no book cross-check in `part_b`. The `_bookwalk.py` extraction + `part_a_runner._touch_at` rewrite still ship (independently valuable). See the Spec Change Log.

**Never:**
- Generating the synthetic orders (that is slice 2 — `run_part_b` takes `intents`).
- Re-implementing any `check_*` from `invariants.py`, or the invariant definitions; a book/`OrderTracker`/`sim`-internal re-derivation of invariants 4-series, 5-liquidity, or 6-ordering.
- Evaluating Part A, combining the two into the §A8.2 gate verdict, or writing the frozen SHA — that is `gate.py`.
- Front-month instrument filtering, `.dbn.zst` path resolution, any network / Tranche-1 pull.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| all invariants hold, ≥1000 orders | a clean synthetic list (market + passive + marketable-limit, both sides) + a well-formed source | `verdict == "PASS"`, `violations == ()` | N/A |
| a real `check_*` breach (doctored outcome) | e.g. a marketable outcome carrying `queue_rank_at_submit` → real invariant 4 | `Violation(order_id, "4", <real message>)` recorded; sweep continues; `verdict == "FAIL"` | N/A |
| several orders each breaking a different invariant | doctored outcomes | all in `violations` (sorted); per-label count in `reason`; `verdict == "FAIL"` | N/A |
| intent/outcome kind disagree | `check_order` raises "join mismatch" | — | `PartBError` |
| `sim.simulate` raises `InvariantViolation` | a simulator invariant broke mid-run | `Violation("", "sim", message)`, `verdict == "FAIL"` — not a crash | N/A |
| fewer than `PART_B_MIN_ORDERS` | 500 clean orders, all invariants hold | `verdict == "FAIL"`, `reason` names the order-count shortfall | N/A |
| `len(outcomes) != len(intents)` / missing / duplicate / foreign outcome | a dropped or extra outcome | — | `PartBError` |
| duplicate `order_id` in `intents` | two intents share an id | — | `PartBError` |
| a non-`SUBMIT` intent | a `replace` / `cancel` record in `intents` | — | `PartBError` naming the order_id |
| multi-instrument source | unfiltered parent stream | — | `IntentLogError` (from `sim`) |
| empty `intents` | `[]` | `verdict == "FAIL"` (order-count); `source` still consumed once by `sim` | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/part_b.py` — NEW. `run_part_b(intents, source, *, config=PRIMARY, pad_ns=PART_B_WINDOW_PAD_NS) -> PartBResult`; frozen dataclasses `Violation(order_id, invariant, message)`, `PartBResult(n_orders, n_outcomes, n_fill_events, violations, verdict, reason, coverage_note)`; `PartBError`; `PART_B_WINDOW_PAD_NS`; `PART_B_COVERAGE_NOTE`; private `_pair_outcomes`, `_invariant_label`, `_verdict`.
- `src/ticksim/parity/invariants.py:323` — `check_order(intent, outcome)` (8 checks fixed order, first breach raises `InvariantViolation`). Message prefixes `_invariant_label` parses: `"invariant N (…"`, `"time_to_fill consistency …"`, `"adverse_selection structural …"`, `"intent/outcome join mismatch …"`. `InvariantViolation` re-exported from `..sim`.
- `src/ticksim/parity/_bookwalk.py` — NEW (this slice). `BookReplay(source)` + `BookWalkError`. `part_b` does **not** import it.
- `src/ticksim/parity/part_a_runner.py` — `_touch_at` rewritten as a `BookReplay` wrapper (behaviour-preserving; `test_touch_at_*` unchanged); `BookWalkError` translated to `PartAError`.
- `src/ticksim/sim.py:709` — `simulate(source, intent_log, config, valid_intervals, *, degraded_days=()) -> (list[OrderOutcome], Manifest)`; raises `IntentLogError` (multi-instrument / non-replayable), `InvariantViolation`, `BookInconsistency`, `ValueError`, `OrderStateError`.
- `src/ticksim/orders.py` — `OrderIntent` (`action`, `order_id`, `submit_ts_ns`, …), `IntentAction.SUBMIT`, `OrderOutcome` (`order_id`, `trade_id`, `fills: tuple[Fill,...]`, `side`, `kind`), `Fill`.
- `src/ticksim/config.py` — `PRIMARY`, `SimConfig`, `PART_B_MIN_ORDERS=1000`.
- `tests/unit/test_ticksim_imports.py:39` — `"_bookwalk"`, `"part_b"` rows; `"part_a_runner"` gains `"_bookwalk"`.
- `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-16 (six invariants defined once), AD-7 (`parity → events` already widened), AD-27 (`PART_B_MIN_ORDERS`).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/_bookwalk.py` — `BookReplay(source)` + `BookWalkError`, `__all__`. Bounded read-only `Book` replay, `advance_to(ts_ns)` (non-decreasing), fail-closed guards; wrap a non-`StopIteration` iterator exception as `BookWalkError`.
- [x] `src/ticksim/parity/part_a_runner.py` — `_touch_at` rewritten on `BookReplay` (behaviour-preserving); edge gains `_bookwalk`. Add a test feeding a mis-ordered / multi-instrument window source through `_touch_at` → `PartAError`.
- [x] `src/ticksim/parity/part_b.py` — `run_part_b(intents, source, *, config=PRIMARY, pad_ns=PART_B_WINDOW_PAD_NS)`, `Violation`/`PartBResult(n_orders, n_fill_events, violations, verdict, reason, coverage_note)`/`PartBError`, `PART_B_WINDOW_PAD_NS`, `PART_B_COVERAGE_NOTE`, `_validate_intents` / `_pair_outcomes` / `_invariant_label` / `_verdict`, `__all__`. **No book replay.** `pad_ns < 0` / `order_id == ""` / non-`SUBMIT` / dup id → `PartBError`; `n_orders == 0` always FAILs; `sim`-raised `InvariantViolation` → `Violation("", "sim", …)` + FAIL; a non-`InvariantViolation` from `check_order` → `PartBError` naming the order.
- [x] `src/ticksim/parity/__init__.py` — allowed-edges docstring for `_bookwalk` + `part_b`; `book` dropped from `part_a_runner`'s.
- [x] `tests/unit/test_ticksim_imports.py` — `"_bookwalk"` + `"part_b"` rows; `"part_a_runner"` gains `_bookwalk`, drops `book`.
- [x] `tests/unit/test_ticksim_parity_bookwalk.py` — `BookReplay`: fold ≤ts and stop, inclusive bound, non-decreasing enforcement, multi-instrument → `BookWalkError`, empty-before-first-ts, non-`StopIteration` iterator error wrapped, `_broken` poisoned-instance guard.
- [x] `tests/unit/test_ticksim_parity_part_b.py` — ~34 tests, all real-pipeline: kind-mix PASS + a genuine 1000-order varied-batch (market / marketable_limit / passive_limit, both sides, sizes 1–5) clean PASS; **real** `check_*` breaches via doctored outcomes → parsed labels `"1"`/`"2"`/`"3"`/`"4"`/`"5"`/`"6"`/`"time_to_fill"`/`"adverse_selection"` with the sweep continuing; real kind-mismatch → `PartBError`; monkeypatched `simulate` raising `InvariantViolation` → `Violation("", "sim", …)` + FAIL; `n_orders < PART_B_MIN_ORDERS` and `n_orders == 0` → FAIL; non-`SUBMIT` / `""` / dup-id / negative-`pad_ns` → `PartBError`; missing / duplicate / count-mismatch outcome → `PartBError`; multi-instrument → `IntentLogError`; empty `intents` (source still consumed — noted as a `sim` contract dependency); `violations` sorted; `n_fill_events` counts partials separately; epoch-clamp regression.
- [x] `tests/unit/test_ticksim_parity_run_part_a.py` — 3 new: `_touch_at` fed a mis-ordered source / a multi-instrument source / a mid-stream iterator error → `PartAError` (through the `_bookwalk` `BookWalkError` wrap).

**Acceptance Criteria:**
- Given a genuine clean synthetic list with `n_orders >= PART_B_MIN_ORDERS` (real 1000+), when `run_part_b` runs, then `verdict == "PASS"`, `violations == ()`, and `coverage_note is PART_B_COVERAGE_NOTE`.
- Given a doctored `OrderOutcome` that genuinely trips `invariants.check_order` (real invariant 4: a marketable outcome carrying `queue_rank_at_submit`), then `violations` has one `Violation` with `invariant == "4"` and the real message, `verdict == "FAIL"`, and other clean orders in the batch are still swept.
- Given a monkeypatched `simulate` that raises `InvariantViolation("invariant 6 …")`, then `run_part_b` returns `verdict == "FAIL"` with a `Violation("", "sim", …)` — no exception escapes.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `part_b.py` imports only `{sim, orders, config, invariants, events}` from `src.ticksim`; the import-graph test passes.

## Spec Change Log

**Loopback 1 — 2026-08-31 — `intent_gap` (blind-hunter + edge-case-hunter converged; verification-gap confirmed the coverage gaps).** The frozen invariant-5 book cross-check was unsound:

| # | Was (iteration 0) | Now | Why |
|---|---|---|---|
| 1 | `run_part_b` does a **single amortised `BookReplay` pass** and records a `"5-liquidity"` `Violation` for any marketable fill with no crossable resting level at/through its price | **no book replay in `part_b`** — invariant 5's liquidity half is a `fills.py` construction guarantee (`_walk_book` emits only for size > 0; passive fill only past the trade-volume threshold), verified in `test_ticksim_fills.py`, listed in `PART_B_COVERAGE_NOTE` exactly as invariant 4's series and invariant 6's ordering already were | `advance_to(fill.ts_ns)` is **inclusive** so it folds the fill's own same-ts book consumption (C/M events) → the replay book holds *less* than `sim` saw → spurious `"5-liquidity"` FAIL on a **correct** sim. The replay is also coarser than `sim`'s `(ts_event, class_rank, sequence, source_index)` merge order, checks a level *exists* not that enough size rested, and does nothing meaningful for `passive_limit` fills. Human chose "drop it — construction guarantee". |
| 2 | `verdict` PASS requires `n_outcomes == n_orders`; a "count mismatch → FAIL" verdict path | `_pair_outcomes` raises `PartBError` on any missing / duplicate / foreign / count-mismatch outcome; the downstream verdict path assumes `n_outcomes == n_orders` (dead branch removed) | The FAIL branch was unreachable — `_pair_outcomes` already raised first. Resolved by making the `PartBError` contract explicit. |
| 3 | (unspecified) `sim.simulate` raising `InvariantViolation` | caught, recorded as `Violation("", "sim", message)`, `verdict == "FAIL"` | A sim-raised invariant breach IS a Part B failure per §A8.2 ("any violation = FAIL"), not a crash. |
| 4 | (unspecified) non-`SUBMIT` intents | `PartBError` naming the order_id | A `replace` / `cancel` reuses `order_id` → collided in the pairing map and raised a misleading "duplicate order_id". Part B's synthetic battery is standalone submits only. |
| 5 | `n_fills`; `Violation(order_id, invariant, message)` unsorted | `n_fill_events` (partials count separately — the name says so); `violations` sorted by `(order_id, invariant)` | Legibility of the FAIL report; a gate reader mis-reads "N fills" as contracts. |
| 6 | (unspecified) `invariants.py` ↔ `part_b` message-string coupling | a test must trip a **real** `check_*` and assert the parsed label; `PART_B_COVERAGE_NOTE` records the `check_fill_latency` vs literal-prereg `replace` refinement | The `_invariant_label` regex and the `"join mismatch"` escalation were only tested against hand-written fake messages — a wording refactor in `invariants.py` would silently mislabel every Part B violation. |

**KEEP:** the per-order sweep that continues past the first violation (all failing orders in one run); `check_order` as the single invariant definition site (no re-implementation); the `_bookwalk.py` extraction + `part_a_runner._touch_at` rewrite (already implemented + tested — independently valuable, slice-2 generator needs BBO sampling); the `intents`-order → deterministic result shape.

**Round 2 — 2026-08-31 — patch round (no re-derivation).** All findings patch/defer. Frozen reconciliation: `PartBResult.n_outcomes` dropped — `_pair_outcomes` raises on any mismatch so it was provably always `== n_orders` (blind-hunter: "redundant and misleading on the sim-raised path"). `n_orders == 0` now always FAILs (a zero-order battery verifies nothing regardless of a misconfigured floor). ~16 patches: dead-branch removal (`_invariant_label` join-mismatch, `_pair_outcomes` `None`), guards (`pad_ns >= 0`, `order_id != ""`, non-`InvariantViolation` from `check_order` → `PartBError`), `BookReplay._broken` poison flag, `part_a_runner` `book` edge dropped (the `_touch_at` rewrite no longer imports it), `PART_B_COVERAGE_NOTE` label-token alignment, and real-`check_*`-breach label tests for `"1"`/`"2"`/`"3"`/`"6"`/`"adverse_selection"` (were fake-string only) + a genuine varied 1000-order PASS.

## Design Notes

**What Part B certifies.** Invariants 1, 2, 3, 5a, 6-trace and the `time_to_fill` / `adverse_selection` consistency checks are fully post-hoc from `(intent, outcome)` — `check_order` decides them, and `run_part_b` runs that against **real sim outputs at ≥1000-order scale** (invariants.py's own unit tests use hand-built outcomes). Invariant 4's *endpoint* is in `check_order`. Invariant 4's *series*, invariant 5's *liquidity*, and invariant 6's *merge ordering* are construction guarantees of `OrderTracker` / `fills.py` / `sim` with named `tests/unit/` homes — `PART_B_COVERAGE_NOTE` states this verbatim so a gate reader knows the battery is a scaled post-hoc check, not a re-derivation of the sim's internals.

**Why the sweep continues past the first violation.** The seal wants "every one of these must hold, 100%" — a gate operator debugging a FAIL needs *all* the failing orders in one run, not a fix-one-rerun loop. `check_order` raises on first breach *per order*; `run_part_b` catches per order and moves on.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_part_b.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity tests/unit/test_ticksim_parity_part_b.py` — expected: clean.

## Suggested Review Order

**Entry point — the sweep**

- `run_part_b`: validate intents → one `simulate` (catching a sim-raised `InvariantViolation`) → pair → per-order `check_order` sweep → `_verdict`.
  [`part_b.py:158`](../../src/ticksim/parity/part_b.py#L158)

- `_invariant_label`: the `invariants.py` message → single-token label parse (pinned by real-breach tests, not fake strings).
  [`part_b.py:330`](../../src/ticksim/parity/part_b.py#L330)

- `_verdict`: PASS iff no violations and `n_orders >= PART_B_MIN_ORDERS` (and `n_orders > 0`); sorted `violations`, per-label FAIL breakdown.
  [`part_b.py:345`](../../src/ticksim/parity/part_b.py#L345)

**Pairing + input contract**

- `_validate_intents` (all `SUBMIT`, unique non-empty `order_id`) and `_pair_outcomes` (`PartBError` on any count / dup / foreign mismatch).
  [`part_b.py:265`](../../src/ticksim/parity/part_b.py#L265)

**What the battery certifies**

- `PART_B_COVERAGE_NOTE`: which labels are scaled post-hoc checks vs `fills.py` / `OrderTracker` / `sim` construction guarantees — the honest scope statement `gate.py` must render.
  [`part_b.py:61`](../../src/ticksim/parity/part_b.py#L61)

**Shared primitive**

- `BookReplay`: bounded read-only book replay, non-decreasing `advance_to`, fail-closed guards, `_broken` poison flag. Not used by `part_b`; `part_a_runner._touch_at` is now a wrapper and the slice-2 generator will sample BBO through it.
  [`_bookwalk.py:55`](../../src/ticksim/parity/_bookwalk.py#L55)

**Peripherals**

- Import edges: `part_b → {sim, orders, config, invariants, events}`, `_bookwalk → {book, events}`, `part_a_runner` drops `book`.
  [`test_ticksim_imports.py:39`](../../tests/unit/test_ticksim_imports.py#L39)

- part_b tests (~34, all real-pipeline); `_bookwalk` tests.
  [`test_ticksim_parity_part_b.py:1`](../../tests/unit/test_ticksim_parity_part_b.py#L1)
