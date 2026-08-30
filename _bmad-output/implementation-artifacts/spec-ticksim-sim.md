---
title: 'ticksim sim.py — the SimRun orchestration loop (AD-20 tick loop, AD-21/22 seam, manifest)'
type: 'feature'
created: '2026-08-29'
status: 'done'
review_loop_iteration: 1
baseline_commit: '09b405a'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/` has every leaf (config, book, events, orders/OrderTracker, fills), but nothing runs the simulation. The parity gate and every H1 study need one pure entry point that consumes a book-event stream + an order-intent log and produces the `OrderOutcome` log + a reproducibility manifest.

**Approach:** add `src/ticksim/sim.py` — `simulate(book_event_source, intent_log, config, valid_intervals) -> (list[OrderOutcome], Manifest)`, a thin wrapper over a `SimRun` class that owns the AD-20 tick loop, drives the AD-21/AD-22 book→order seam, applies the AD-13 session mask, and accumulates the manifest. **`adverse_selection` is out of this slice** (carved off — AD-28 deferred-check queue is a follow-up; every `OrderOutcome.adverse_selection` is `False` here).

## Boundaries & Constraints

**Always:**
- `simulate(...)` is pure (AD-2, AD-5): consumes only a `BookEventSource` and an iterable of `OrderIntent` (AD-23) — **never imports or calls strategy code**. Same `(source events, intent log, config, valid_intervals)` ⇒ **byte-identical `OrderOutcome` log** (AD-11); the manifest is exempt. Only `SimRun` mutates the book / tracker.
- **Determinism (AD-1/AD-11):** one monotonic `int64` `ts_event` ns clock; **no** `time.time()` / `datetime.now()` / `random` unseeded — the sole entropy is `random.Random(config.seed)` (unused this slice but constructed and recorded); every iteration over a `dict`/`set` is explicitly sorted.
- **Intent-log validation (AD-2, AD-13a)** up front → `IntentLogError`: `submit_ts_ns` non-decreasing; every `CANCEL`/`REPLACE` (`replaces_order_id`) references an already-seen `order_id`; no duplicate `SUBMIT` `order_id`; every `submit_ts_ns` inside the union of `valid_intervals`.
- **The tick loop (AD-20).** Merge time-ordered inputs — the `BookEventSource` (class_rank 0), the intent log at each record's `submit_ts_ns`, every `valid_interval` boundary, every pending deferred intent-effect ts, **and every pending order-arrival ts** (`submit_ts + latency`, and the fresh arrival ts of a price-change `replace`). A monotonic clock guard rejects any `now_ns` that regresses (`InvariantViolation`, AD-1). Per distinct `ts_event` **T**:
  1. **Book deltas at T** — for each `BookEvent` with `ts_event == T`: if action is `C`/`M`, `resting_before = book.order_by_id(iid, ev.order_id)`; `book.apply_event(book, ev)`; then `queue_model.observe_book_event(tracker, ev, resting_before)` (`resting_before=None` for `A`/`T`/`F`/`R`). sim is the **sole driver** of this seam (AD-21).
  2. **Intent records at `submit_ts_ns == T`** — `SUBMIT` → `tracker.submit(intent, config.latency_ns, T)`; `CANCEL`/`REPLACE` → enqueue a deferred effect at `T + config.latency_ns` (one latency hop, AD-8).
  3. **Pending intent effects with `effect_ts <= T`** — `CANCEL` → `tracker.cancel(oid, T)`; `REPLACE` → `tracker.replace(intent, config.latency_ns, T)` (a price-change replace re-enters IN_FLIGHT with a fresh arrival; it is re-activated by a later step 4).
  4. **Arrivals** — `activated = tracker.activate_arrivals(T)`; for each id, `snap = tracker.snapshot(oid)`: if `snap.kind == PASSIVE_LIMIT` call `queue_model.queue_ahead_size(book, iid, side, snap.limit_px_dbn, snap.arrival_ts_ns)` **exactly once** and `tracker.set_queue_position(oid, ahead, ahead)`; for **every** activated id `tracker.set_arrival_bbo(oid, *book.snapshot_bbo(iid))`. Because arrival ts is a wake, this snapshots at the arrival tick (AD-22 / AD-16 inv. 1), not at the next later book event.
  5. **Fills** — if `T` is inside the mask: `for fe in fills.decide(book, tracker, T, config): cascaded = tracker.apply_fill(fe, T)` (`_step_fills` also asserts `T` is in-mask → `InvariantViolation`, so a future refactor cannot decide a fill outside the mask, AD-13c). `oco_cascade_cancel_count` accumulates `len(cascaded)`.
- **Session mask (AD-13).** `valid_intervals` is canonicalized on construction (`_normalize_intervals`): non-empty, sorted, `start < end`, and **merged** — overlapping *or contiguous* windows become one, so there is no internal seam where `expire_all` (at an interval `end`) would kill an order still inside the mask at an adjacent window's `start`. Malformed input → `ValueError`. Half-open `[start, end)`. Book events outside the mask are still folded (the book stays continuous) but steps 2, 4, 5 are skipped there; **step 3 drains regardless** so a latency hop past an interval end cannot wedge the loop (its target is `EXPIRED` by then and the effect is dropped). At each interval's `end_ns`, `tracker.expire_all(end_ns)`. Databento-`degraded` days are a manifest field, **not** auto-excluded.
- **`iid` (instrument id)** is the sole instrument in the book (H1 is single-instrument MNQ front month; `OrderIntent`/`OrderTracker` carry none — spine Deferred). The check is **per book event** (`event.instrument_id != self._iid` ⇒ `IntentLogError`), so a second instrument that only ever appears in `T`/`C`/`R` records is still caught. An arrival before the first book event uses `iid = 0` — the book is genuinely empty so `queue_ahead_size` returns 0 and `snapshot_bbo` returns `(None, None)`, both correct.
- **Output (AD-12/AD-24).** `outcomes = tracker.finalize()` (submit-ordered). `Manifest` (a frozen dataclass, `to_dict() -> dict` JSON-safe): `config` (`config.model_dump(mode="json")` — enums → str), `seed`, `valid_intervals` (merged), `degraded_days` (sorted + de-duped; a bare `str` → `TypeError`), `unseen_cm_count` / `overcancel_count` / `max_transient_cross_ns` / `last_ts_ns` (from `book`), `event_count`, `intent_count`, `oco_cascade_cancel_count`, `outcome_schema_version`, `python_version`, `databento_version`, `sortedcontainers_version`, `sibling_run_id` (`None`; set by the study layer). **No monetary field ever enters `OrderOutcome`** (AD-24) — sim does no P&L.
- **Invariant checks.** `book.check_invariants()` at each interval boundary (and run end); a `BookInconsistency` propagates. `IntentLogError` and `InvariantViolation` are defined **here** (`parity/invariants.py`, a later slice, will import `InvariantViolation` from `sim`).
- `PERMITTED_INTERNAL_EDGES["sim"] == {"config","book","orders","events","fills"}` unchanged; relative imports. `mypy --strict src/ticksim` clean, no override; `black`-88; no `assert` for anything the parity verdict depends on.

**Ask First:**
- The AD-28 `adverse_selection` predicate (endpoint vs. window, exact "same-side quote move through our price" direction) — **deferred to the follow-up slice**; if planning this slice forces a decision, HALT.

**Never:**
- Importing `report` / `parity` / `cli` / `databento`; calling `merge_streams` (it merges `BookEvent` streams only — multi-file sourcing is the caller's job; sim takes one `BookEventSource`).
- Editing config / book / events / orders / fills (flag a missing hook, don't bolt on).
- A second replay pass (AD-14: exactly two `SimRun`s per study, both driven by the caller).
- Implementing the AD-28 deferred-adverse queue, the 3-way P&L (`report.py`), or Part A/B (`parity/`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Marketable entry fills at arrival | 1 `SUBMIT` marketable BUY; book has asks | `OrderOutcome` `FILLED`, `fills` non-empty, `arrival_ts_ns == submit_ts + latency` | N/A |
| Passive fill after queue clears | `SUBMIT` passive BUY @P; later trades at P exceed queue-ahead | `OrderOutcome` `FILLED` at P; `queue_ahead_size_at_submit` set once | N/A |
| Order never fills, interval ends | passive order, no qualifying volume before `end_ns` | `OrderOutcome` `EXPIRED` at `end_ns` | N/A |
| Cancel takes a latency hop | `SUBMIT` @T0, `CANCEL` @T1 | order `CANCELLED` at `T1 + latency` (not `T1`) | N/A |
| Bracket: exit fills → entry+sibling cancel | OCO group, TP fills | TP `FILLED`; entry + SL `CANCELLED` same T; all 3 in `outcomes` | N/A |
| Book event outside mask | trade at a `ts_event` between intervals | folded into the book; no arrival/decide/fill at that T | N/A |
| Non-decreasing submit_ts violated | intent log with a `submit_ts_ns` regression | — | `IntentLogError` |
| Cancel of unknown order | `CANCEL` for an `order_id` never submitted | — | `IntentLogError` |
| submit_ts outside the mask | `SUBMIT` `submit_ts_ns` in no interval | — | `IntentLogError` |
| >1 instrument in the book | source yields events for two `instrument_id`s (even via `T`/`C` only) | — | `IntentLogError` |
| Price-change `REPLACE` | `SUBMIT` passive @P0, `REPLACE` @P1 at `T` | takes effect at `T + latency`; re-flights + re-activates then; `arrival_ts_ns` refreshed; queue position re-taken at P1 | N/A |
| Adjacent / overlapping intervals | `valid_intervals = [(0,500),(500,1000)]` | merged to `[(0,1000)]`; no order expired at the internal seam | N/A |
| Malformed intervals | empty, or `start >= end` | — | `ValueError` |
| Negative `config.latency_ns` | `SimConfig(latency_ns=-1)` | — | `ValueError` |
| Duplicate `SUBMIT` order_id | two `SUBMIT`s with the same `order_id` | — | `IntentLogError` |
| Deferred `CANCEL` after expiry | `CANCEL` whose `+latency` lands after `expire_all` | silently dropped; order stays `EXPIRED` | N/A |
| Determinism | run `simulate` twice (same process; and two interpreters, different `PYTHONHASHSEED`) | `[o.model_dump_json() …]` byte-identical | N/A |
| Manifest surfaces book counters | run with pre-window `C`/`M` records | `manifest.unseen_cm_count > 0`, `max_transient_cross_ns` recorded | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/events.py` — `BookEvent` (`action:MboAction, side:MboSide, order_id, price_dbn, size, ts_event, sequence, instrument_id`; structurally a `book.MboRecord`); `BookEventSource` Protocol (`class_rank: int`, re-iterable `__iter__ -> Iterator[BookEvent]`); `merge_streams` (**not used here** — `BookEvent`-only). `MboAction` (`A/C/M/T/F/R/N`).
- `src/ticksim/book.py` — `Book` (`.instruments` dict, `.unseen_cm_count`, `.overcancel_count`, `.max_transient_cross_ns`, `.last_ts_ns`), `apply_event(book, record)` (own monotonic-ts guard → `BookInconsistency`), `order_by_id(iid, oid) -> RestingOrder|None`, `snapshot_bbo(iid) -> (bid|None, ask|None)`, `check_invariants()`, `BookSide` (BID/ASK), `BookInconsistency`.
- `src/ticksim/orders.py` — `OrderTracker`: `submit(intent, latency_ns, now_ns)`, `activate_arrivals(now_ns) -> list[str]`, `cancel(oid, now_ns)`, `replace(intent, latency_ns, now_ns)`, `expire_all(now_ns) -> list[str]`, `apply_fill(fe, now_ns) -> list[str]` (cascaded ids), `set_queue_position(oid, rank, ahead_size)`, `set_arrival_bbo(oid, bid, ask)`, `snapshot(oid) -> OrderSnapshot`, `working_order_ids()`, `in_flight_order_ids()`, `finalize() -> list[OrderOutcome]`. `OrderIntent` (`action:IntentAction {SUBMIT,CANCEL,REPLACE}, order_id, trade_id, leg, kind, side, size, limit_px_dbn, submit_ts_ns, replaces_order_id, oco_group_id`), `IntentAction`, `Side` (BUY/SELL), `OrderKind`, `OrderOutcome` (`schema_version`), `OrderStateError`.
- `src/ticksim/fills.py` — `decide(book, tracker, clock_ns, config) -> list[FillEvent]`; `queue_model_for(config) -> QueueModel`; `QueueModel.queue_ahead_size(book, iid, BookSide, price_dbn, arrival_ts) -> int`, `.observe_book_event(tracker, record, resting_before)`.
- `src/ticksim/config.py` — `SimConfig` (`frozen`, `.queue_model`, `.latency_ns`, `.seed`, `.model_dump()`), `PRIMARY` / `OPTIMISTIC`, `ADVERSE_SELECTION_WINDOW_NS`, `MAX_TRANSIENT_CROSS_NS`.
- `tests/unit/test_ticksim_imports.py:45` — `"sim"` edge already declared; guard auto-covers `sim.py`.
- spine — AD-1/2/5/11 (pure + deterministic), AD-13 (mask, 3 failures), AD-20 (tick order), AD-21 (seam, sole driver), AD-22 (queue position once), AD-12/24 (outputs, no money).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/sim.py` — new module: `IntentLogError`, `InvariantViolation`, `Manifest` (frozen dataclass + `to_dict`), `SimRun`, `simulate(...)`. `_normalize_intervals` helper. Relative imports, `__all__`.
- [x] `tests/unit/test_ticksim_sim.py` — 26 tests: every I/O-matrix row + the review-1 hardening set + a cross-`PYTHONHASHSEED` subprocess determinism test.
- [x] `tests/unit/test_ticksim_imports.py` — unchanged; `sim` edge auto-covered, green.

**Acceptance Criteria:**
- Given a book-event source + a 3-order bracket intent log, when `simulate` runs, then `outcomes` has 3 submit-ordered `OrderOutcome`s with coherent terminal states and the manifest's `event_count` / `intent_count` match the inputs.
- Given the same inputs twice, when `simulate` runs each, then `[o.model_dump() for o in outcomes]` is identical (AD-11).
- Given an intent whose `submit_ts_ns` regresses, or a `CANCEL` of an unknown order, or a `submit_ts_ns` outside `valid_intervals`, then `IntentLogError` before any outcome is produced.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `sim.py` imports only `config`/`book`/`orders`/`events`/`fills` from `src.ticksim`; no `time`/`datetime` wall-clock call.

## Spec Change Log

### 2026-08-29 review-1 (blind / edge-case / verification-gap) — `bad_spec` loopback

Two real defects the reviewers demonstrated, both traceable to the frozen spec / Design Notes:

1. **Adjacent `valid_intervals` seam.** `[(0,500),(500,1000)]` sorted-but-not-merged → at `T=500` `expire_all` fired (it is an `end`) while `T=500` is also *in* the mask (`[500,1000)`), killing any order live across the seam. Fix: `_normalize_intervals` merges overlapping *and contiguous* windows and rejects empty / `start >= end` (`ValueError`). Frozen block amended.
2. **Arrival snapshot from the wrong tick.** The Design Notes said "arrivals are not wake points … caught at the next wake `>=` arrival_ts" — so `queue_ahead_size` / `snapshot_bbo` (which caps marketable price-improvement, AD-16 inv. 1) were taken from a *later* book state, violating AD-22's "exactly once, at the order's arrival tick". Fix: order-arrival ts (and a price-change `replace`'s fresh arrival ts) is now a wake point. Frozen block + Design Notes amended.

Also patched (reviewer-driven, no design choice): per-event multi-instrument check (catches a 2nd instrument seen only via `T`/`C`/`R`); `config.model_dump(mode="json")` for a genuinely JSON-safe manifest; `config.latency_ns < 0` → `ValueError`; `degraded_days` sorted+de-duped, bare `str` → `TypeError`; single-shot reuse → `RuntimeError` (was `InvariantViolation` — not a parity invariant); a monotonic-clock `InvariantViolation` guard; `_step_fills` asserts in-mask (makes the AD-13c docstring claim real); `oco_cascade_cancel_count` manifest field; `IntentAction` else-branch raises; a `_PROGRESS_EVERY` info log (spine Conventions); `R`-in-mask warning (deferred-work). +13 tests incl. the cross-hash-seed subprocess determinism check.

KEEP: sim owns its own merge (not `merge_streams`); the 5-step per-tick order; `SimRun` single-shot; `resting_before` looked up pre-`apply_event`; step 3 drains regardless of mask.

## Design Notes

**Own the merge; don't call `merge_streams`.** It yields `BookEvent`s only. `SimRun._loop` streams over `min(candidate ts)` where candidates = next buffered book event, next intent `submit_ts_ns`, next `valid_interval` bound, `effects` heap top, `arrival_wakes` heap top. A monotonic `clock` rejects a regressing `now_ns`. Deferred `CANCEL`/`REPLACE` go in `effects` keyed `(effect_ts, seq, intent)` (seq keeps the un-orderable intent out of comparisons); a price-change `replace` pushes its fresh `tracker.arrival_ts_ns(oid)` onto `arrival_wakes`.

**Queue rank == ahead-size.** `queue_ahead_size` returns a total, not a per-order count, so `set_queue_position(oid, ahead, ahead)` — contracts-ahead is the rank proxy (consistent with "every resting order ahead of us"). A per-order rank would need a `queue_ahead_size` change (deferred).

**Class-rank ordering inside a tick** is steps 1→5; that *is* AD-20's `book_delta(0) < order_arrival(1)` made concrete (deferred-fill-apply rank 2 = the carved-off AD-28 queue).

## Suggested Review Order

**The tick loop (AD-20 — `SimRun._loop`)**
- the `min(candidates)` merge + monotonic `clock` guard
- step 1: fold book deltas + per-event multi-instrument check + `observe_book_event` seam (AD-21, sole driver, `resting_before` pre-`apply_event`)
- steps 2–3: `SUBMIT` inline; `CANCEL`/`REPLACE` → `effects` heap at `+latency`; drained regardless of mask (terminal target dropped)
- step 4 (`_step_arrivals`): AD-22 `queue_ahead_size` **once** at the arrival-tick wake + `set_arrival_bbo`
- step 5 (`_step_fills`): `decide` once + `apply_fill`; in-mask assertion (AD-13c)

**Session mask (AD-13 — `_normalize_intervals`)** — merge overlapping+contiguous, reject malformed; `expire_all` at each `end`; `check_invariants` at boundaries + run end

**`Manifest`** — frozen dataclass; `model_dump(mode="json")`; `to_dict()` JSON-safe; exempt from AD-11 byte-identity

**Determinism (AD-11)** — `random.Random(config.seed)` sole entropy, no wall-clock, explicit sorts; the cross-`PYTHONHASHSEED` subprocess test

**Deferred** (`deferred-work.md`): AD-28 adverse_selection queue; a mid-RTH `R` book-clear zeroing `queue_ahead`; a true per-order queue rank.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_sim.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/sim.py tests/unit/test_ticksim_sim.py` — expected: clean.
- `grep -nE 'time\.time|datetime\.now|\bnow\(\)' src/ticksim/sim.py` — expected: no matches (AD-1).
