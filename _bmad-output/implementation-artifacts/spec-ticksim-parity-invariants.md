---
title: 'ticksim parity/invariants.py — the six Part-B invariant assertions'
type: 'feature'
created: '2026-08-30'
status: 'done'
review_loop_iteration: 1
baseline_commit: '4ea173d'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The parity gate's Part B (prereg §A8.2) runs ≥1000 synthetic orders through the simulator and requires **all six invariants** to hold 100%. Spine AD-16 says they must be defined **once**, as pure assertion functions, consumed by both `parity/part_b.py` and `tests/unit/`.

**Approach:** add `src/ticksim/parity/invariants.py` (+ `parity/__init__.py`) — one `check_*(...)` per invariant, each raising `InvariantViolation` (imported from `sim`) on breach and returning `None` on hold, plus a `check_order(intent, outcome)` that runs the per-order set. Everything operates on `(OrderIntent, OrderOutcome)` — no book, no sim run. Invariant 5's *liquidity* half is not a post-hoc `OrderOutcome` property and is handled per **Ask First** below.

> **Review loopback 1 (2026-08-30, `bad_spec`).** Two frozen-block signatures were wrong and are amended below — see the Spec Change Log at the end. Summary: `check_fill_latency` / `check_order` do **not** take `latency_ns` (a priority-preserving size-down `replace` keeps the *original* `arrival_ts_ns` while `outcome.submit_ts_ns` moves to the replace tick, so `submit + latency` recomputation false-fails — invariant 3 is checked against `outcome.arrival_ts_ns` directly); and `check_queue_position` requires the queue fields only when the passive order **demonstrably worked** (has fills or reached `FILLED`), since a `passive_limit` that is `REJECTED` or `EXPIRED` while still `IN_FLIGHT` never reached `WORKING` and legitimately serializes both as `None`. Two consistency checks were also added (`check_time_to_fill`, `check_adverse_selection`) because AD-12 / AD-28 name `parity/invariants.py` a co-owner of those `OrderOutcome` families.

## Boundaries & Constraints

**Always:**
- Every function is **pure**: inputs are frozen `OrderIntent` / `OrderOutcome` (+ scalar params); it raises `InvariantViolation` with a message naming the invariant number, the `order_id`, and the offending values, or returns `None`. No I/O, no mutation, no `book`/`events`/`sim`-run.
- **`check_no_price_improvement(intent, outcome)` — invariant 1.** Only when `outcome.kind in {MARKETABLE, MARKETABLE_LIMIT}` (a passive limit legitimately fills at its limit inside the spread — that is invariant 2, not a breach of 1). For every fill: `side == BUY` and `arrival_best_ask_dbn is not None` ⇒ `fill.px_dbn >= arrival_best_ask_dbn`; `side == SELL` and `arrival_best_bid_dbn is not None` ⇒ `fill.px_dbn <= arrival_best_bid_dbn`.
- **`check_within_limit(intent, outcome)` — invariant 2.** Only when `outcome.kind in {PASSIVE_LIMIT, MARKETABLE_LIMIT}` and `intent.limit_px_dbn is not None`. For every fill: `side == BUY` ⇒ `fill.px_dbn <= intent.limit_px_dbn`; `side == SELL` ⇒ `fill.px_dbn >= intent.limit_px_dbn`.
- **`check_fill_latency(outcome)` — invariant 3.** *(amended, loopback 1 — no `latency_ns` param.)* Every `fill.ts_ns >= outcome.arrival_ts_ns`. `outcome.arrival_ts_ns` is the tick the tracker treated as the order's exchange arrival — `submit_ts_ns + latency_ns` for an un-replaced order, the *original* arrival for a priority-preserving size-down `replace` (whose `outcome.submit_ts_ns` is the replace tick). Recomputing `submit + latency` here would false-fail the replace case, so the check is against `arrival_ts_ns` directly.
- **`check_queue_position(outcome)` — invariant 4 (structural subset).** *(amended, loopback 1.)* For a `passive_limit` outcome that **demonstrably worked** (`outcome.fills` non-empty **or** `terminal_state is FILLED`): **both** `queue_rank_at_submit` and `queue_ahead_size_at_submit` are set and `>= 0`. A `passive_limit` that is `REJECTED`, or `EXPIRED`/`CANCELLED` while still `IN_FLIGHT` (never reached `WORKING`), legitimately serializes both as `None` — not a breach. A `marketable` / `marketable_limit` outcome has **both `None`** always. (The full "non-negative and non-increasing until terminal" *time series* is a live `OrderTracker` guarantee — `decrement_queue_ahead` floors at 0 and never increments; `set_queue_position` is once-only — already asserted by `tests/unit/test_ticksim_orders.py`; this function checks the serialized endpoint only. The spec states this scope; do not rebuild the tracker here.)
- **`check_partials_within_size(intent, outcome)` — invariant 5a.** `sum(f.size for f in outcome.fills) <= intent.size`; and `terminal_state == FILLED` ⇒ the sum `== intent.size`; and a non-`FILLED` terminal (`CANCELLED`/`EXPIRED`/`REJECTED`) may have `fills` summing to `< intent.size` (a legit partial-then-terminal) but **not** `> intent.size`.
- **`check_fill_causality(outcome)` — invariant 6 (post-hoc trace).** `outcome.fills` ts_ns are **non-decreasing**, and every `fill.ts_ns >= outcome.arrival_ts_ns` (redundant with 3 but scoped to causality). `fills.decide` stamps `FillEvent.ts_ns = clock_ns`, so a fill stamped out of order or before arrival is the observable signature of a lookahead / clock bug. The AD-20 "only events with ts ≤ clock" property itself is a sim-loop construction guarantee (monotonic clock + stable merge) covered by `tests/unit/test_ticksim_sim.py`; this function checks the trace it leaves.
- **`check_time_to_fill(outcome)` — `OrderOutcome` consistency (AD-12 co-ownership).** `time_to_fill_ns is not None` **iff** `terminal_state is FILLED`; when set it is `>= 0` and equals `outcome.fills[-1].ts_ns - outcome.arrival_ts_ns`.
- **`check_adverse_selection(outcome)` — `OrderOutcome` consistency (AD-28).** If `outcome.adverse_selection` is `True`: `outcome.kind` is **not** marketable/marketable-limit (a marketable fill crosses the spread by design — never adverse-flagged) and `outcome.fills` is non-empty (an order that never filled cannot be adverse-selected).
- **`check_order(intent, outcome)`** *(amended, loopback 1 — no `latency_ns`.)* First validates the `intent`↔`outcome` join (`order_id` / `kind` / `side` / `trade_id` must match — a mispaired intent/outcome in `part_b.py`'s log-join would otherwise let every check pass on a mismatched pair). Then runs 1, 2, 3, 4, 5a, 6, `check_time_to_fill`, `check_adverse_selection` in that fixed order (each skips itself where its `kind` guard excludes it); first breach raises. `part_b.py` will call this in its ≥1000-order loop; it does **not** cover invariant 5's liquidity half (see Ask First).
- `InvariantViolation` is imported `from ..sim import InvariantViolation` (defined there per the sim-slice reconciliation). `MNQ_TICK_DBN` (if a message wants ticks) `from ...` no — `from ..config import MNQ_TICK_DBN` only if needed for a message; keep messages in dbn otherwise.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. `PERMITTED_INTERNAL_EDGES` gains an `"invariants"` row (see Code Map). `parity/__init__.py` is an empty package marker.

**Ask First:**
- **Invariant 5's liquidity half** ("no fill occurs when the book has no liquidity at or through the order's price") has no post-hoc `OrderOutcome` form — it needs the book state at each fill tick, which would drag a time-indexed book accessor into this pure module. **Proposed resolution (confirm at checkpoint):** it is a `fills.py` construction guarantee — `_walk_book` emits a `FillEvent` only for a level with `level_size > 0`; a passive fill only when `cum_trade_vol_since_arrival − queue_ahead > 0` (real trade volume reached the price) — and is verified by `tests/unit/test_ticksim_fills.py`. `parity/part_b.py` (a later slice) MAY add its own book cross-check as it has the book; `invariants.py` does not own it. If you want it enforced in `invariants.py` via an injected accessor instead, say so.

**Never:**
- Importing `book` / `events` / `fills` / `report` / `databento`; running a `SimRun`; touching the `OrderTracker`.
- A `check_*` that mutates its inputs or returns anything other than `None`.
- Re-deriving invariant 4's time-series monotonicity (an `OrderTracker` concern) or invariant 6's merge-ordering (a `sim` concern).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Inv 1 holds | marketable BUY, fills all `>= arrival_best_ask_dbn` | `check_no_price_improvement` → `None` | N/A |
| Inv 1 breach | marketable BUY, a fill `< arrival_best_ask_dbn` | — | `InvariantViolation` (names "1", order_id, px vs ask) |
| Inv 1 skipped | `passive_limit` BUY filling below the arrival ask | `check_no_price_improvement` → `None` (kind-guarded) | N/A |
| Inv 1, no arrival quote | marketable BUY, `arrival_best_ask_dbn is None` | → `None` (nothing to compare) | N/A |
| Inv 2 holds / breach | passive SELL, fill `>= limit` / a fill `< limit` | `None` / `InvariantViolation` ("2") | — |
| Inv 3 holds / breach | all `fill.ts_ns >= arrival_ts_ns` / one before arrival | `None` / `InvariantViolation` ("3") | — |
| Inv 3 size-down replace | `submit_ts_ns` = replace tick, `arrival_ts_ns` = original, fill between | `None` (checked vs `arrival_ts_ns`, not `submit + latency`) | — |
| Inv 4 passive OK / marketable OK | worked passive with both queue fields `>= 0` / marketable with both `None` | `None` | — |
| Inv 4 in-flight passive | `passive_limit` REJECTED/EXPIRED/CANCELLED, no fills, both queue fields `None` | `None` (never reached WORKING) | — |
| Inv 4 breach | filled `passive_limit` with a queue field `None`, or a marketable with a set queue field | — | `InvariantViolation` ("4") |
| Inv 5a holds / partial-then-expired | `Σ fills == size` on FILLED / `Σ fills < size` on EXPIRED | `None` | — |
| Inv 5a breach | `Σ fills > size`; `Σ fills != size` on FILLED; `Σ fills == size` on non-FILLED; any fills on REJECTED | — | `InvariantViolation` ("5") |
| Inv 6 holds / breach | fills ts non-decreasing & `>= arrival` / a fill ts out of order | `None` / `InvariantViolation` ("6") | — |
| `time_to_fill` holds / breach | present iff FILLED, `>= 0`, `== last_fill.ts_ns - arrival_ts_ns` / any mismatch | `None` / `InvariantViolation` ("time_to_fill") | — |
| `adverse_selection` holds / breach | flag on a filled passive / flag on a marketable or an unfilled order | `None` / `InvariantViolation` ("adverse_selection") | — |
| `check_order` all-pass | a coherent marketable and a coherent passive outcome | `None` for each | N/A |
| `check_order` join mismatch | `intent.order_id != outcome.order_id` (or kind/side/trade_id) | — | `InvariantViolation` ("join mismatch") |
| `check_order` per-invariant breach | an outcome that breaks exactly one of 1/2/3/4/5/6 | raises naming that invariant — proves the call is wired | `InvariantViolation` |
| No fills | a `CANCELLED` / `EXPIRED` outcome with `fills == ()` | every `check_*` → `None` (nothing to check) | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/orders.py` — `OrderOutcome` (`kind, side, submit_ts_ns, arrival_ts_ns, terminal_state, fills: tuple[Fill,...], queue_rank_at_submit, queue_ahead_size_at_submit, arrival_best_bid_dbn, arrival_best_ask_dbn`), `OrderIntent` (`order_id, size, limit_px_dbn`), `Fill` (`px_dbn, size, ts_ns`), `Leg`, `Side` (BUY/SELL), `OrderKind` (MARKETABLE/MARKETABLE_LIMIT/PASSIVE_LIMIT), `TerminalState` (FILLED/…).
- `src/ticksim/sim.py:137` — `InvariantViolation(Exception)`, in `__all__`. Imported `from ..sim import InvariantViolation`.
- `src/ticksim/config.py` — `MNQ_TICK_DBN` (only if a message needs ticks). Parity thresholds (`PARITY_*`, `PART_B_MIN_ORDERS`) belong to `part_b.py` / `gate.py`, **not** here.
- `_bmad-output/planning-artifacts/architecture/…/ARCHITECTURE-SPINE.md` — AD-16 (invariants defined once), AD-7 (`parity → sim, report, book, config` — this file adds `orders`; see Design Notes).
- `tests/unit/test_ticksim_imports.py:39` — add `"invariants": {"sim", "orders"}` (`config` too if a message uses `MNQ_TICK_DBN`). `parity/__init__.py` auto-skipped (`path.stem == "__init__"`).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/__init__.py` — empty package marker (module docstring only).
- [x] `src/ticksim/parity/invariants.py` — the six `check_*` + `check_time_to_fill` + `check_adverse_selection` + `check_order`, `__all__`. Relative imports.
- [x] `tests/unit/test_ticksim_imports.py` — `"invariants": {"sim", "orders"}` edge row.
- [x] `tests/unit/test_ticksim_parity_invariants.py` — 46 tests: holds + raises per invariant, kind-guard skips, the `time_to_fill` / `adverse_selection` consistency cases, the `check_order` join-mismatch case, and a parametrised per-invariant `check_order`-raises case, from hand-built `OrderIntent` / `OrderOutcome`.

**Acceptance Criteria:**
- Given a marketable BUY whose fills all price at or above `arrival_best_ask_dbn`, then `check_no_price_improvement` returns `None`; given one fill a tick below, then `InvariantViolation` naming invariant 1 and the `order_id`. ✅
- Given a `passive_limit` `OrderOutcome` that filled with `queue_ahead_size_at_submit is None`, then `check_queue_position` raises `InvariantViolation`; given the same fields on a `REJECTED` passive that never worked, then `None`. ✅
- Given `mypy --strict src/ticksim`, then zero errors, no override; `invariants.py` imports only `sim` + `orders` from `src.ticksim`; the import-graph test passes with the new `"invariants"` row. ✅
- Given the full ticksim unit suite, then green (433 passed, 6 skipped). ✅

**Suggested Review Order:**
1. `check_fill_latency` / `check_queue_position` amended scoping — confirm the size-down-replace and in-flight-passive carve-outs are correct against `orders.py` (`OrderTracker` replace convention AD-23, `TerminalState`) and don't mask a real breach.
2. `check_order` join validation + fixed evaluation order — a mispaired `(intent, outcome)` must raise before any invariant runs.
3. `check_no_price_improvement` "unverifiable ⇒ raise" (not skip) when the arrival quote on the crossed side is `None` — is failing-closed right for Part B, vs the frozen block's original "→ `None` (nothing to compare)"?
4. `check_within_limit` passive-limit exact-price rule (`fill.px_dbn == limit`, not `<=`/`>=`) — matches `fills.decide` emitting `px_dbn == limit` for a resting fill.
5. `check_time_to_fill` / `check_adverse_selection` — are these in scope for this slice, or should they move to a `part_b` cross-check?

## Design Notes

**`parity → orders` edge.** AD-7's rule text lists `parity → sim, report, book, config`; `invariants.py` also needs `orders` for the `OrderIntent` / `OrderOutcome` / `Fill` / enum types (and the imports test's own docstring uses `from ..orders import Fill` as the canonical parity example). Add `orders` to the `"invariants"` row and an inline note to AD-7. It imports **nothing** from `book`/`events`/`report` — narrower than AD-7's list, which is fine.

**Why invariant 4 / 6 are partial here.** AD-16 says the invariants are defined once — but two of them are fundamentally live properties (4's monotonic decrement, 6's merge ordering) whose *only* honest post-hoc form is a trace check on the serialized `OrderOutcome`. The spec pins exactly what each `check_*` verifies and names where the live property is tested (`test_ticksim_orders.py`, `test_ticksim_sim.py`), so Part B + the unit tests are checking the same thing, not "subtly different things".

## Spec Change Log

**Loopback 1 — 2026-08-30 — `bad_spec` (all three reviewers converged on the first two):**

| # | Frozen-block text (iteration 0) | Amended to | Why |
|---|---|---|---|
| 1 | `check_fill_latency(outcome, latency_ns)`; asserts `arrival_ts_ns == submit_ts_ns + latency_ns` | `check_fill_latency(outcome)` — no `latency_ns`; asserts only `fill.ts_ns >= outcome.arrival_ts_ns` | A priority-preserving size-down `replace` (AD-23) keeps the **original** `arrival_ts_ns` while `outcome.submit_ts_ns` moves to the replace tick. `submit + latency` recomputation false-fails every such order. `arrival_ts_ns` is already the authoritative arrival the tracker used. |
| 2 | Inv 4: a `passive_limit` outcome **always** has both queue fields set | queue fields required only when the passive order **worked** (`fills` non-empty or `terminal_state is FILLED`) | A `passive_limit` that is `REJECTED`, or `EXPIRED`/`CANCELLED` while still `IN_FLIGHT`, never reached `WORKING`, so `set_queue_position` was never called and both fields are legitimately `None`. The original text would fail Part B on every never-worked passive order. |
| 3 | `check_order(intent, outcome, *, latency_ns)` | `check_order(intent, outcome)` + an `intent`↔`outcome` join check (`order_id`/`kind`/`side`/`trade_id`) before any invariant | Follows from change 1 (no `latency_ns` anywhere). The join check is new: `part_b.py` pairs an intent log with an outcome log, and a mispaired row would let all six checks pass on unrelated data. |
| 4 | Inv 1, arrival quote `None` on the crossed side → `None` (nothing to compare) | → **raise** `InvariantViolation` ("unverifiable") | Deliberate fail-closed for a parity gate: a marketable order that filled with no bounding touch recorded is a data/sim defect Part B must surface, not silently pass. Reviewer-endorsed; flagged in Suggested Review Order #3 for a second look. |
| 5 | (not in frozen block) | added `check_time_to_fill`, `check_adverse_selection` | AD-12 names `parity/invariants.py` a co-owner of the `terminal_state == FILLED ⇔ fills` family; AD-28 defines the `adverse_selection` structural cases. Both are pure `OrderOutcome` consistency checks that belong at the single definition site. |

Also: `check_within_limit` for a `PASSIVE_LIMIT` now asserts `fill.px_dbn == limit` exactly (a resting fill prices *at* its limit — `fills.decide` emits exactly that), not the `<=`/`>=` half-plane, which is the `MARKETABLE_LIMIT` (book-walk) rule. This sharpens the frozen text rather than contradicting it.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_invariants.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity/ tests/unit/test_ticksim_parity_invariants.py` — expected: clean.
