---
title: 'ticksim fills.py — fill-decision engine + the two queue models'
type: 'feature'
created: '2026-08-29'
status: 'done'
review_loop_iteration: 1
baseline_commit: '84b1fb3'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/` has config, book, events, and orders/OrderTracker, but nothing decides *whether an order fills*. `sim.py` (next slice) needs one pure authority that, given the current book + tracker state, returns this-tick fills — and the two seal-frozen queue models (§2.1 primary back-of-queue, secondary time-priority) that drive it.

**Approach:** add `src/ticksim/fills.py` — `decide(book, tracker, clock_ns, config) -> list[FillEvent]` (AD-19, holds no state), a `QueueModel` interface with `BackOfQueueModel` / `TimePriorityModel` (AD-22), and each model's `observe_book_event(...)` seam hook (AD-21). Leaf-ish: imports only `book`, `orders`, `config` + stdlib. One small `book.py` query addition (`resting_orders`/strict `queue_ahead_size`) for the time-priority variant and the marketable walk.

## Boundaries & Constraints

**Always:**
- `decide(book, tracker, clock_ns, config)` iterates `tracker.working_order_ids()`; per order reads `tracker.snapshot(oid)`; emits `FillEvent(order_id, px_dbn, size, ts_ns=clock_ns)` — **this-tick incremental** (AD-19), never cumulative. Returns `[]` when nothing fills. No mutation of `book` or `tracker` inside `decide` (AD-5 pure).
- **Passive limit** (`kind == PASSIVE_LIMIT`): AD-22's formula is *cumulative* — `cumulative = clamp(cum_trade_vol_since_arrival − queue_ahead, 0, size)` is total fill entitlement to date. `decide` returns the **this-tick delta** (AD-19: a `FillEvent` is never cumulative): `fill_qty = cumulative − filled_qty`, emit one `FillEvent` at the limit price when `fill_qty > 0`. `queue_ahead` / `cum_trade_vol_since_arrival` are the live tracker counters maintained by `observe_book_event`. *(review-1 fix: the earlier `clamp(… , 0, size − filled_qty)` form re-emitted a fill every tick — see Spec Change Log.)*
- **Marketable / marketable-limit**: `_walk_book(book, snap, clock_ns)` consumes the opposite side from the touch, best price first, `min(remaining, level_size)` per level, one `FillEvent` per level; `marketable_limit` stops when the level price is beyond `snap.limit_px_dbn` (`> limit` for a BUY, `< limit` for a SELL). A fill never prices better than the arrival touch (`snap.arrival_best_ask_dbn` for a BUY / `arrival_best_bid_dbn` for a SELL — AD-16 inv. 1). **The book is walked at most once per order** (only while `filled_qty == 0`): the book is never depleted (no own-impact), so a re-walk would re-consume the same displayed size. A partially-filled marketable order is IOC-like — its remainder stays working but inert until the AD-13 mask expires it. **No own-order market impact** — the walk does not mutate `book`.
- `observe_book_event(self, tracker, record, resting_before)` (a `QueueModel` method, called by `sim.py` immediately after each `apply_event`, AD-21). `record: book.MboRecord`; `resting_before: book.RestingOrder | None` = the order at `record.order_id` **before** the event (sim looks it up pre-`apply_event`). Enumerated rules, nothing else moves the counters:
  - `record.action == "T"` and `record.ts_event > O.arrival_ts_ns` (strict, AD-20) and the trade price is **at or through** working passive order `O`'s limit on `O`'s side (BUY: `trade_px <= O.limit`; SELL: `trade_px >= O.limit`) → `tracker.add_trade_volume(O.id, record.size)`.
  - `record.action == "C"`, a same-price `"M"` with `new_size < resting_before.size`, **or** a price-changing `"M"` — of a resting order at a working passive order `O`'s side + price (`resting_before.price_dbn`) and `self.counts_resting_order(resting_before.add_ts_ns, resting_before.sequence, O)` → `tracker.decrement_queue_ahead(O.id, removed_size)`. `removed_size`: `C` → `resting_before.size` (every GLBX `C` is a full cancel); same-price size-down `M` → `resting_before.size − new_size`; **price-changing `M` → `resting_before.size`** (the order left `O`'s level — `book.apply_event` does remove-old + add-new).
  - every other action (`A`, size-up same-price `M`, `R`, `N`, `F`) → no-op. `R` (book clear) is a halt/reset event, excluded by the AD-13 mask, not modelled here (prereg §2.2).
- `QueueModel` interface: `queue_ahead_size(book, instrument_id, side, price_dbn, our_arrival_ts_ns) -> int` (AD-22 — the arrival-tick formula, called **once** by sim) and `counts_resting_order(add_ts_ns, sequence, snap) -> bool`. `BackOfQueueModel`: counts every resting order at the price (`add_ts_ns <= arrival`). `TimePriorityModel`: only `add_ts_ns < arrival` (ties → our order last).
- `queue_model_for(config) -> QueueModel` maps `config.queue_model` (`QueueModel` enum) → a fresh model instance; `decide` and `sim.py` both use it.
- Int-only, `ts_event` ns clock, DBN 1e-9 prices (AD-10/AD-1). Plain classes for the models (match `book._InstrumentBook` / `OrderTracker`); `FillEvent` is imported from `orders`, never redefined.
- `PERMITTED_INTERNAL_EDGES["fills"] == {"book", "orders", "config"}` unchanged; relative imports (`from .book import …`). `mypy --strict src/ticksim` clean, no override; `black`-88.

**Ask First:**
- Marketable-limit remainder handling — **resolved at review-1: the book is walked once; a partial marketable order is IOC-like** (remainder inert). A resting-limit conversion would need an OrderTracker change and is out of this slice.

**Never:**
- Importing from `events`, `sim`, `report`, `parity`, or `databento`.
- Mutating `book` or `OrderTracker` inside `decide` (only `observe_book_event` touches the tracker, via its two guarded mutators).
- Implementing the AD-28 1-second adverse-selection deferred check (that is `sim.py`), the AD-20 tick loop, or the 3-way P&L (`report.py`).
- Redefining `FillEvent` / any frozen schema.
- A trade also decrementing `queue_ahead` (Alex decision 2026-08-29: trades feed `cum_trade_vol` only; AD-21's "T decrements queue_ahead" clause is corrected in the spine in this slice).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Passive, queue not cleared | working BUY limit @P size 5; `cum_trade_vol=8`, `queue_ahead=10` | `decide` → no `FillEvent` for it (`8−10<0`) | N/A |
| Passive, partial | as above but `cum_trade_vol=12`, `queue_ahead=10` | one `FillEvent(size=2, px=P, ts=clock_ns)` | N/A |
| Passive, already part-filled | size 5, `filled_qty=3`, `cum_trade_vol=20`, `queue_ahead=0` | `FillEvent(size=2)` (remaining only, capped) | N/A |
| `observe` trade at price, before arrival | `T` at P, `ts_event == O.arrival_ts_ns` | **not** counted (strict `>`) | N/A |
| `observe` trade through price | working BUY @100; `T` at 99, size 4 | `add_trade_volume(O, 4)` | N/A |
| `observe` cancel ahead | working passive O @P; `C` of a resting order @P with `add_ts <= O.arrival` size 6 | `decrement_queue_ahead(O, 6)` | N/A |
| `observe` cancel not ahead (time-priority) | same but resting `add_ts == O.arrival`, model = `TimePriorityModel` | **no** decrement (strict `<`) | N/A |
| `observe` size-up M | same-price `M` raising a resting order's size | no-op | N/A |
| `observe` price-changing M ahead | working passive O @P; `M` moves a resting order @P (`add_ts <= O.arrival`, size 6) to another price | `decrement_queue_ahead(O, 6)` — full size (it left our level) | N/A |
| Marketable partially fills, remainder inert | marketable BUY size 5; asks 30000:2 only | tick 1: `FillEvent(2 @30000)`; a later `decide` returns `[]` for it even if new asks appear | N/A |
| Multi-instrument book | `decide` on a book with >1 instrument (any order kind) | — | `ValueError` (tracker carries no instrument id) |
| Marketable BUY, book deep | working marketable BUY size 3; asks 30000: 5 | one `FillEvent(size=3, px=30000)` | N/A |
| Marketable BUY walks levels | size 5; asks 30000:2, 30025:10 | `FillEvent(2 @30000)`, `FillEvent(3 @30025)` | N/A |
| Marketable-limit BUY stops at limit | limit 30000, size 5; asks 30000:2, 30025:10 | `FillEvent(2 @30000)` only; remainder 3 stays working | N/A |
| Marketable, empty book side | working marketable BUY; no asks for instrument | `[]`; order stays working | N/A |
| `decide` with no working orders | tracker has only terminal / in-flight orders | `[]` | N/A |
| Unknown `config.queue_model` | enum value with no model | — | `ValueError` from `queue_model_for` |

</frozen-after-approval>

## Code Map

- `src/ticksim/orders.py` — `FillEvent(order_id:str, px_dbn:int, size:int, ts_ns:int)` (import, don't redefine); `OrderTracker.working_order_ids()`, `.snapshot(oid) -> OrderSnapshot`, `.add_trade_volume(oid, qty)`, `.decrement_queue_ahead(oid, qty)`. `OrderSnapshot` fields: `order_id, side, kind, size, limit_px_dbn, arrival_ts_ns, add_ts_ns, filled_qty, queue_rank_at_submit, queue_ahead_size_at_submit, queue_ahead, cum_trade_vol_since_arrival` **+ `arrival_best_bid_dbn, arrival_best_ask_dbn` (added review-1 for the AD-16 inv. 1 marketable-touch cap)**. Enums `Side` (BUY/SELL), `OrderKind` (MARKETABLE/MARKETABLE_LIMIT/PASSIVE_LIMIT).
- `src/ticksim/book.py` — `Book` queries: `best_bid_dbn(iid)`, `best_ask_dbn(iid)`, `snapshot_bbo(iid)`, `size_at_price(iid, BookSide, px)`, `queue_ahead_size(iid, BookSide, px, arrival_ts)` (**currently `<=` only** — add a `strict: bool=False` kw for `TimePriorityModel`), `order_by_id(iid, order_id) -> RestingOrder|None`. `BookSide` (BID/ASK), `RestingOrder(instrument_id, side, price_dbn, size, add_ts_ns, sequence)`, `MboRecord` Protocol (`action:str, side:str, order_id:int, price_dbn:int, size:int, ts_event:int, sequence:int, instrument_id:int`). **No public level iterator** — add `Book.resting_levels(iid, BookSide) -> list[tuple[int,int]]` (price, total_size; best first) for `_walk_book`.
- `src/ticksim/config.py:22` — `QueueModel` enum (`BACK_OF_QUEUE`, `TIME_PRIORITY`); `SimConfig.queue_model`, `.latency_ns`; `MNQ_TICK_DBN = 250_000_000`.
- `tests/unit/test_ticksim_imports.py:44` — `"fills": {"book","orders","config"}` already declared; the guard auto-covers `fills.py` once it exists. No change.
- `_bmad-output/planning-artifacts/architecture/…/ARCHITECTURE-SPINE.md` — AD-19 (`FillEvent` / `decide`), AD-20 (tick order, strict `>` for trade-after-arrival), AD-21 (seam — **its "T decrements queue_ahead" clause is corrected here**), AD-22 (one queue interface + the fill formula), AD-5 (pure engine).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/book.py` — added `Book.resting_levels(instrument_id, side) -> list[tuple[int,int]]` (best-first) and a `strict: bool = False` kw on `queue_ahead_size`. No behaviour change to existing callers.
- [x] `src/ticksim/orders.py` — `OrderSnapshot` + `snapshot()` gain `arrival_best_bid_dbn` / `arrival_best_ask_dbn` (review-1; the marketable walk caps at the arrival touch). `_TrackedOrder` already held them.
- [x] `src/ticksim/fills.py` — new module: `QueueModel` ABC, `BackOfQueueModel`, `TimePriorityModel`, `queue_model_for(config)`, `_walk_book`, `decide`. Relative imports.
- [x] `_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md` — AD-21 Rule amended (T → `cum_trade_vol`, not `queue_ahead`; `resting_before` param; price-changing `M` removes full size). Human-authorized (Alex decision 2026-08-29).
- [x] `tests/unit/test_ticksim_fills.py` — new file, 47 tests: a test per I/O-matrix row + the ACs + the review-1 hardening set.

**Acceptance Criteria:**
- Given `config.queue_model == BACK_OF_QUEUE` vs `TIME_PRIORITY`, when `queue_model_for(config).queue_ahead_size(...)` is called on a level with one resting order stamped exactly at our arrival ts, then back-of-queue counts it and time-priority does not.
- Given a working passive order whose `cum_trade_vol_since_arrival` exceeds `queue_ahead` by K with M contracts left, when `decide` runs, then exactly one `FillEvent` of `min(K, M)` at the order's limit price and `ts_ns == clock_ns`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `fills.py` imports only `book`/`orders`/`config` from `src.ticksim`.
- Given the full ticksim suite, then green (existing 265 + the new fills tests); `book.py`'s existing tests unchanged and passing.

## Spec Change Log

### 2026-08-29 review-1 (blind-hunter / edge-case-hunter / verification-gap) — `bad_spec` loopback

All three reviewers independently demonstrated that the frozen passive-fill formula
**re-emits a fill every tick**. `clamp(cum_trade_vol_since_arrival − queue_ahead, 0,
size − filled_qty)` computes a *this-tick* value from *cumulative* inputs
(`cum_trade_vol` only grows, `queue_ahead` only shrinks), so an order that filled 3
last tick with no new trade volume emits `FillEvent(3)` again — filling its whole
size off a sliver of genuinely queue-cleared volume, violating AD-19 ("never
cumulative") and AD-16 invariant 5.

**Frozen block amended** (the correction is forced by AD-19 + AD-22, no design
choice): the formula is `cumulative = clamp(cum_trade_vol_since_arrival −
queue_ahead, 0, size)` and `decide` emits `cumulative − filled_qty`. Known-bad state
avoided: whole-order fills off partial eligible volume once `sim.py` wires this in.

Also amended in the same pass (all reviewer-driven, none a design choice):
- **Marketable walk-once.** The frozen text said the remainder "re-walks on later
  ticks". The book is never depleted (no own-impact), so a re-walk re-consumes the
  same displayed size. Now: the book is walked only while `filled_qty == 0`; a
  partial marketable order is IOC-like (remainder inert until the AD-13 mask). The
  "Ask First" note is resolved to this.
- **Marketable touch cap (AD-16 inv. 1).** A later-tick walk into liquidity that
  appeared at a better price could beat the arrival touch. `OrderSnapshot` gains
  `arrival_best_bid_dbn` / `arrival_best_ask_dbn`; the walk clamps each fill price
  to the arrival touch.
- **Price-changing `M` ahead of us.** The frozen matrix only covered size-down `M`.
  `book.apply_event` implements a price-move `M` as remove-old + add-new, so an
  order ahead of us that moves price has left our level — decrement `queue_ahead`
  by its full size.
- **Multi-instrument guard hoisted.** `_sole_instrument_id(book)` is now checked at
  the top of `decide` unconditionally (was only reached via the marketable path);
  a passive-only tick on a >1-instrument book now raises `ValueError` too. Matrix
  row added.
- Dropped an unused `logger`; documented the side-agnostic trade-volume accrual and
  the `R` (book-clear) no-op (AD-13 mask territory, prereg §2.2).

KEEP: the `QueueModel` ABC shape (`queue_ahead_size` + `counts_resting_order` +
one shared `observe_book_event`); the `_walk_book` best-first level consumption;
`resting_before` threaded from `sim.py`.

## Design Notes

Fill formula (Alex decision + AD-22, review-1): `queue_ahead` (live) is reduced only
by cancels/mods of orders ahead of us; trade volume accumulates separately.
`cumulative = clamp(cum_trade_vol − queue_ahead, 0, size)` is the entitlement to
date; `decide` returns `cumulative − filled_qty`. A through-print lands as ordinary
`cum_trade_vol` and naturally fills us.

`observe_book_event` takes `resting_before` because after `apply_event` a cancelled
order is gone; `sim.py` looks it up via `order_by_id(...)` before folding and passes
it in (`None` for a `T`).

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_fills.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/fills.py src/ticksim/book.py tests/unit/test_ticksim_fills.py` — expected: clean.

Result: **311 ticksim tests pass** (47 in `test_ticksim_fills.py`), `mypy --strict src/ticksim` clean (6 files), `black` clean.

## Suggested Review Order

**The fill decision (AD-19: `decide` is the sole pure authority)**

- `decide` — validates the queue model + single-instrument precondition up front,
  then per working order: `_passive_fill` or `_walk_book`. Mutates nothing.
- `_passive_fill` — cumulative entitlement `clamp(cum_trade_vol − queue_ahead, 0,
  size)` minus `filled_qty` (review-1: the this-tick delta, AD-19)
- `_walk_book` — best-first level consumption; **walk-once** (`filled_qty == 0`);
  each fill price clamped to the arrival touch (AD-16 inv. 1)

**The book-event seam (AD-21: `sim.py` drives it after every `apply_event`)**

- `QueueModel.observe_book_event(tracker, record, resting_before)` — `T` →
  `add_trade_volume` (side-agnostic, strict `ts > arrival`); `C` / size-down `M` /
  price-changing `M` of an order ahead → `decrement_queue_ahead`
- `counts_resting_order` — the one knob: `<=` (back-of-queue) vs `<` (time-priority)

**The two models (AD-22: one interface, one formula)**

- `BackOfQueueModel` / `TimePriorityModel` — differ only in `counts_resting_order`
  and the `strict=` they pass to `book.queue_ahead_size`
- `queue_model_for(config)` — enum → fresh instance, `ValueError` on unmapped

**book.py additions** — `resting_levels` (best-first level list) + `strict=` kw on
`queue_ahead_size`; no behaviour change to existing callers.

**Deferred** (in `deferred-work.md`): a `passive_limit` intent priced through the
book; two same-side marketable orders in one tick (each sees full displayed size);
a `marketable_limit` non-marketable at arrival; the AD-21 `sequence` tie-break
(moot — our order has none).
