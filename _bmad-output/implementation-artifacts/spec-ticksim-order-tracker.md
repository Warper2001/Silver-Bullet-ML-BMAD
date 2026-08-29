---
title: 'ticksim OrderTracker — lifecycle state machine + OCO groups'
type: 'feature'
created: '2026-08-29'
status: 'done'
review_loop_iteration: 0
baseline_commit: 'a0fe8c80674590800db99b07c6fb16342016eedc'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/` has the frozen `orders` schemas, `book.py`, and `events.py`, but nothing owns an order's life. The fill engine and the sim loop both need one authority on the state machine `intent → in_flight → working → {filled|cancelled|rejected|expired}`, the per-order counters the queue model reads, and the `OrderOutcome` emitted at the end.

**Approach:** add `OrderTracker` (a plain mutable class, like `book._InstrumentBook`) and `OrderStateError` to `src/ticksim/orders.py`. It is the sole authority (spine AD-8): every `OrderOutcome` field is derived from a transition it performed. It stays a leaf — imports nothing from `src.ticksim`; `latency_ns` is passed in as an `int`.

## Boundaries & Constraints

**Always:**
- One `OrderTracker` owns the machine. Live states: `IN_FLIGHT`, `WORKING`; terminals map 1:1 to `TerminalState` (`FILLED/CANCELLED/REJECTED/EXPIRED`). Transitions only via the tracker's methods; any illegal transition raises `OrderStateError`.
- `submit(intent, latency_ns, now_ns)` — `intent.action == SUBMIT`; creates the order `IN_FLIGHT` with `arrival_ts_ns = intent.submit_ts_ns + latency_ns`; registers it in the `intent.oco_group_id` group if set. Duplicate `order_id` → `OrderStateError`.
- `activate_arrivals(now_ns)` — every `IN_FLIGHT` order with `arrival_ts_ns <= now_ns` → `WORKING`.
- `apply_fill(fill_event, now_ns)` — order must be `WORKING`; `filled_qty += size` (append the `Fill`); `filled_qty > order size` → `OrderStateError`; on `filled_qty == size` → `FILLED`, set `time_to_fill_ns = now_ns - arrival_ts_ns`. **Leg-aware OCO cascade (spine AD-25, amended 2026-08-29):** if the just-filled order is an **`EXIT`** leg, cancel every other live member of its OCO group at `now_ns` (bookkeeping, no new intent) and return their ids; an **`ENTRY`**-leg fill cascades nothing (the exits stay live so the position can be closed and Part A can replay the real exit). Returns `list[str]` — the cascaded ids, `[]` otherwise.
- `cancel(order_id, now_ns)` / `replace(intent, latency_ns, now_ns)` from `IN_FLIGHT` or `WORKING`. `replace`: a **size decrease at the same price** stays `WORKING`, keeps `add_ts_ns` + queue counters (priority preserved); **any price change** → back to `IN_FLIGHT` with a fresh `arrival_ts_ns` and queue counters cleared to `None` (priority lost, spine AD-8).
- `expire_all(now_ns)` — every `IN_FLIGHT` and `WORKING` order → `EXPIRED` at `now_ns` (spine AD-13(b); sim calls it at a `valid_interval` end, `[start, end)`).
- `reject(order_id, now_ns, reason)` from `IN_FLIGHT`.
- sim-only setters, each callable **once** while the order is live: `set_queue_position(order_id, rank, ahead_size)`, `set_arrival_bbo(order_id, bid_dbn, ask_dbn)` (both at the arrival tick); `set_adverse_selection(order_id, value)` callable only on a `FILLED`, not-yet-finalized order (spine AD-28).
- fills-only counter mutators (spine AD-21/22 — the *rules* live in fills.py; only the guarded state lives here): `add_trade_volume(order_id, qty)` (→ `cum_trade_vol_since_arrival`), `decrement_queue_ahead(order_id, qty)` (`queue_ahead` floored at 0). Read: `snapshot(order_id)` returns an immutable view of a working order's counters + fields for `fills.decide`.
- `finalize() → list[OrderOutcome]` — every order must be terminal (else `OrderStateError`); builds one `OrderOutcome` per order, every field from tracked transitions; `fills` as a `tuple[Fill, ...]`. Ordering: submit order.
- Int-only, `ts_event` ns clock (spine AD-10/AD-1). `OrderTracker` a plain class; no `float`.
- `orders.py` imports only stdlib + pydantic; `PERMITTED_INTERNAL_EDGES["orders"]` stays `set()`. `mypy --strict` clean (no override); `black`-88.

**Ask First:**
- Whether a price-change `replace` re-incurs full `latency_ns` (spec assumes **yes** — the replace message travels; it returns to `IN_FLIGHT`).

**Never:**
- Implementing `observe_book_event`'s decrement *rules* (which `T`/`C`/`M` move `queue_ahead`) — that is `fills.py` (spine AD-21). Only the mutable counters + guarded mutators live here.
- Implementing the 1-second `adverse_selection` deferred check — that is `sim.py` (spine AD-28). Only `set_adverse_selection` lives here.
- Any `import` from `src.ticksim` or `databento`; any `SimConfig` reference (take `latency_ns: int`).
- Generating a new `OrderIntent` on an OCO cascade (spine AD-2/AD-25).
- Touching the existing frozen schemas / enums.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Submit → arrival | `submit(i, lat=250, now=0)`, `activate_arrivals(now=250)` | order `WORKING`, `arrival_ts_ns == i.submit_ts_ns + 250` | N/A |
| Not yet arrived | above, `activate_arrivals(now=100)` | order still `IN_FLIGHT` | N/A |
| Partial then full fill | `WORKING` size 4; `apply_fill(sz2)`, `apply_fill(sz2)` | after 2nd: `FILLED`, `len(fills)==2`, `time_to_fill_ns` set | N/A |
| Over-fill | `WORKING` size 3; `apply_fill(sz4)` | — | `OrderStateError` |
| Fill a non-working order | `IN_FLIGHT`; `apply_fill(...)` | — | `OrderStateError` |
| OCO cascade — exit leg fills | group {entry, tp, sl} all live; `apply_fill` fills `tp` (an `EXIT` leg) | `tp` `FILLED`; `entry`, `sl` → `CANCELLED` at the same `now_ns`; `apply_fill` returns `["entry", "sl"]` (sorted) | N/A |
| OCO cascade — entry leg fills | group {entry, tp, sl} all live; `apply_fill` fully fills `entry` (an `ENTRY` leg) | `entry` `FILLED`; `tp`, `sl` stay `WORKING`; `apply_fill` returns `[]` | N/A |
| OCO cascade — both exits cross same tick | `tp` and `sl` both fully fill in one `fills.decide` batch | first `apply_fill` → that leg `FILLED` + sibling `CANCELLED`; second `apply_fill` on the cancelled sibling returns `[]` (voided, not an error) | N/A |
| Cancel working | `WORKING`; `cancel(now)` | `CANCELLED`, `terminal` at `now` | N/A |
| Replace, size down same price | `WORKING` size 5 @ P; `replace(size 3 @ P)` | stays `WORKING`, size 3, `add_ts_ns`/`queue_*` unchanged | N/A |
| Replace, price change | `WORKING` @ P; `replace(@ P')` | → `IN_FLIGHT`, new `arrival_ts_ns`, `queue_rank/ahead_at_submit` → `None` | N/A |
| Expire at interval end | one `WORKING`, one `IN_FLIGHT`; `expire_all(now)` | both → `EXPIRED` at `now` | N/A |
| Duplicate submit | `submit` an `order_id` already tracked | — | `OrderStateError` |
| Finalize with a live order | any order still `WORKING` | — | `OrderStateError` |
| Finalize | all terminal | `list[OrderOutcome]`, one per order, submit-ordered, all fields populated | N/A |
| Set queue position twice | `set_queue_position` called a 2nd time | — | `OrderStateError` |
| Adverse on a non-filled order | `set_adverse_selection` on a `CANCELLED` order | — | `OrderStateError` |

</frozen-after-approval>

## Code Map

- `src/ticksim/orders.py:83` -- `OrderIntent` (`action, order_id, trade_id, leg, kind, side, size, limit_px_dbn, submit_ts_ns, replaces_order_id, oco_group_id`) — the tracker input; `:245` `OrderOutcome` — the output (every field listed there must be filled); `:186` `FillEvent` (`order_id, px_dbn, size, ts_ns`) — the fill input; `:70` `TerminalState`; `:220` `Fill` (`px_dbn, size, ts_ns`).
- `src/ticksim/book.py` -- `_InstrumentBook` / `_PriceLevel`: the "plain mutable class alongside frozen models" pattern to match; `BookInconsistency` / `_fail` helper: the typed-exception + raise-helper style.
- `tests/unit/test_ticksim_orders.py` -- existing schema tests; add `OrderTracker` classes here.
- `tests/unit/test_ticksim_imports.py:41` -- `PERMITTED_INTERNAL_EDGES["orders"] = set()` stays.
- `_bmad-output/planning-artifacts/architecture/…/ARCHITECTURE-SPINE.md` -- AD-8 (sole authority), AD-13(b) (expire), AD-25 (OCO), AD-22 (queue counters — state only), AD-19 (one fill method), AD-28 (adverse setter only), AD-12 (OrderOutcome fields).

## Tasks & Acceptance

**Execution:**
- [ ] `src/ticksim/orders.py` -- append: `OrderStateError(Exception)`; `LiveState` enum (`IN_FLIGHT`, `WORKING`); an internal `_TrackedOrder` (plain dataclass — intent, state, `arrival_ts_ns`, `filled_qty`, `fills: list[Fill]`, `add_ts_ns | None`, `queue_rank_at_submit | None`, `queue_ahead_size_at_submit | None`, live `queue_ahead: int`, `cum_trade_vol_since_arrival: int`, `arrival_best_bid_dbn | None`, `arrival_best_ask_dbn | None`, `adverse_selection: bool`, `terminal_state | None`); `OrderTracker` with the methods in Boundaries, an OCO registry (`dict[str, set[str]]`), and read helpers (`snapshot(order_id)`, `live_order_ids()`, `working_order_ids()`, `in_flight_order_ids()`). Extend `__all__`.
- [ ] `tests/unit/test_ticksim_orders.py` -- an `OrderTracker` test class covering every I/O-matrix row + a full bracket lifecycle (submit 3, arrive, fill entry, then fill tp → sl cancelled) end to end asserting the three `OrderOutcome`s.
- [ ] `tests/unit/test_ticksim_imports.py` -- no change needed (edge already `set()`); confirm the test still passes with the new symbols.

**Acceptance Criteria:**
- Given a bracket OCO group where `tp` fills, when `apply_fill` runs, then `entry` and `sl` are `CANCELLED` at the identical `now_ns` and `finalize` yields three coherent `OrderOutcome`s.
- Given any illegal transition (fill an `IN_FLIGHT` order, over-fill, double `set_queue_position`, finalize while live), then `OrderStateError`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `orders.py` still imports nothing from `src.ticksim`.
- Given a `WORKING` order, when `snapshot(order_id)` is called, then it returns an immutable view carrying `queue_ahead`, `cum_trade_vol_since_arrival`, `queue_ahead_size_at_submit`, `add_ts_ns`, `size`, `side`, `limit_px_dbn`.

## Design Notes

`_TrackedOrder` mirrors `book`'s plain-class style; `OrderOutcome` is built only in `finalize`. `snapshot` returns a frozen dataclass (not the mutable `_TrackedOrder`) so `fills.decide` cannot mutate tracker state (spine AD-5 "pure function"). A price-change `replace` clears `queue_*_at_submit` to `None` and sets `state = IN_FLIGHT` with `arrival_ts_ns = intent.submit_ts_ns + latency_ns`; `activate_arrivals` will re-work it. OCO cascade: on a fill reaching `FILLED`, iterate the group's other ids and `cancel(id, now_ns)` each that is still live — reuse the `cancel` path so its guards apply.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_orders.py tests/unit/test_ticksim_imports.py -q` -- expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` -- expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` -- expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/orders.py tests/unit/test_ticksim_orders.py` -- expected: clean.

## Spec Change Log

### 2026-08-29 (later) — AD-25 leg-aware cascade [HUMAN-AUTHORIZED frozen-block amendment]

Alex chose "leg-aware cascade" for the AD-25 question surfaced by the review pass
(below). Frozen-block edits, authorized:
- Boundaries `apply_fill` bullet: the OCO cascade now fires **only on an `EXIT`-leg
  fill** (cancels the other live group members + returns their ids); an `ENTRY`-leg
  fill cascades nothing.
- I/O matrix: the old single "OCO cascade" row is replaced by three — exit-leg
  fills (cascades), entry-leg fills (no cascade), and both-exits-cross-same-tick
  (second fill voided).

Known-bad state this avoids: a fully-filled bracket ENTRY cancelling its own TP/SL
(naked position) and Part A (AD-16/17) being unable to replay a real exit fill.
Code: `_cascade_oco` gains a `filled.intent.leg is Leg.EXIT` guard; `+1` test
(`test_entry_leg_fill_cascades_nothing`), `test_replace_of_oco_member_*` reworked to
a TP/SL pair. Spine AD-25 amended in the same pass (its Rule now carries the
leg-aware wording).

KEEP: `_cascade_oco`'s sorted iteration + reuse of `cancel` + `oco_cancelled_at`
tagging for the same-tick void — the leg guard is a filter in front of that, nothing
else changed.

### 2026-08-29 — review pass (blind-hunter / edge-case-hunter / verification-gap), no frozen-block change

The three reviewers ran against the OrderTracker diff. No `bad_spec` / `intent_gap`
loopback — the frozen block held. Findings triaged as follows.

**Patched inline (within spec intent, no frozen-block edit):**
- `_finalized` seal on every transition (was only on `set_adverse_selection`) — a
  post-`finalize()` transition would be silently absent from the emitted list.
- Monotonic-clock guard (`_advance_clock` / `_last_now_ns`) on the clock-bearing
  transitions — a backwards `now_ns` produced a negative `time_to_fill_ns` that
  blew up `finalize()` with a Pydantic `ge=0` error instead of an `OrderStateError`.
- `replace()` identity guard — a replace may only change `limit_px_dbn` / `size`;
  changing `side/kind/leg/trade_id/oco_group_id/order_id` now raises (else a
  malformed replace silently corrupts the `OrderOutcome` and diverges `_oco_groups`).
- `replace()` size below `filled_qty` now raises (else the order can never reach
  `FILLED` and every later `apply_fill` trips the over-fill guard).
- `apply_fill()` returns `list[str]` — the OCO-cascade-cancelled ids — so `sim.py`
  can record them; `_cascade_oco` returns the same. Empty on a partial / no group.
- Same-tick OCO crossing: if both legs of an OCO fully fill in one `fills.decide`
  batch, the fill on the leg the cascade already cancelled *this tick* is voided
  (returns `[]`) rather than raising.
- `finalize()` second-call guard; `_finalized` set only after every `OrderOutcome`
  is built (a mid-loop validation error no longer half-seals the tracker).
- `set_queue_position` / `set_arrival_bbo` now require `WORKING` (not merely live) —
  consistent with the counter mutators and their arrival-tick semantics.
- `set_adverse_selection` is now genuinely once-callable (added the guard).
- `OrderSnapshot` gains `kind` (the queue model branches marketable vs passive on it).
- New `reject_reason(order_id)` reader — `reject()` captured a reason that had no
  path out (`OrderOutcome` has no such field per AD-12).
- 15 tests added: finalized seal, backwards clock, replace identity / size-up /
  size-below-filled / OCO-member, cascade return value, same-tick OCO crossing,
  `reject_reason`, return-value ordering, once-setter, negative-input guards,
  `side/leg/kind` on a `BUY` outcome. `test_snapshot_is_frozen` tightened to
  `dataclasses.FrozenInstanceError`.

**Surfaced, then RESOLVED by the AD-25 leg-aware amendment above:**
- **AD-25 bracket semantics.** As originally frozen, any group member reaching
  `FILLED` cancelled every other member, so a fully-filled bracket ENTRY cancelled
  its own TP/SL and Part A (AD-16/17) could not replay a real exit fill. Alex chose
  the leg-aware cascade; frozen block + spine AD-25 amended, code + tests updated
  (see the later 2026-08-29 entry). No longer open.

## Suggested Review Order

**The state machine (AD-8: sole authority; every OrderOutcome field derived from a transition)**

- `_TrackedOrder` — the mutable per-order record; `OrderTracker` is the only writer
  [`orders.py`](../../src/ticksim/orders.py) — `class _TrackedOrder`
- `submit` → `activate_arrivals` → `apply_fill` (partials accumulate; `filled_qty`
  never exceeds size) → `FILLED`; plus `cancel` / `reject` / `replace` / `expire_all`
- `_advance_clock` / `_require_open` — the two guards every transition funnels through

**OCO / bracket (AD-25)**

- `_cascade_oco` — **leg-aware**: on an `EXIT`-leg `FILLED`, cancels the other live
  group members at the same `now_ns` and returns their ids; an `ENTRY`-leg fill
  returns `[]`. `oco_cancelled_at` tags cancelled legs so a same-tick stale fill on
  a losing exit is voided.

**replace priority rules (AD-8)**

- size-down at same price → stays `WORKING`, keeps `add_ts_ns` + queue counters
- any price change, or a size *increase* → back to `IN_FLIGHT`, fresh
  `arrival_ts_ns`, counters cleared; identity fields may not change

**The sim-facing seam (AD-12, AD-19, AD-20, AD-22, AD-28)**

- once-setters: `set_queue_position`, `set_arrival_bbo`, `set_adverse_selection`
- fills-only mutators: `add_trade_volume`, `decrement_queue_ahead` (floored at 0)
- `snapshot(order_id)` → frozen `OrderSnapshot` for `fills.decide`
- `finalize()` → `list[OrderOutcome]`, submit-ordered, every order terminal

**Leaf constraint (AD-7)** — `orders.py` still imports nothing from `src.ticksim`;
`PERMITTED_INTERNAL_EDGES["orders"] == set()` unchanged; `mypy --strict` clean.
