---
title: 'ticksim book — L3 order book + apply_event'
type: 'feature'
created: '2026-08-29'
status: 'done'
review_loop_iteration: 0
baseline_commit: '4f8fc346bb83d5e9ea1247d542fe037433fab7eb'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/` has the contract layer (`config.py`, `orders.py` schemas) but no book. Every downstream module — the fill engine, the queue models, the parity invariant checks — needs one venue-faithful L3 book that folds the MBO stream and answers queries at the sim clock.

**Approach:** `src/ticksim/book.py` — a passive L3 data structure plus the single `apply_event(book, record)` function that is the sole authority on MBO→book-state transitions (spine AD-9). Real venue orders only; our orders never enter it (spine AD-3). Best-bid/ask in O(log n) via `sortedcontainers` (spine AD-4). No event loop — `sim.py` drives that later (spine AD-20).

## Boundaries & Constraints

**Always:**
- One `apply_event(book, record)` is the *only* code that mutates the book (spine AD-9). It consumes MBO actions `A C M T F R` (add / cancel / modify / trade / fill / clear); `N` (flag-only) is a documented no-op.
- Resting order stored as `(instrument_id, side, price_dbn, size, add_ts_ns, sequence)` (spine AD-3). Book keyed by `instrument_id`; sub-books auto-created on first event.
- `apply_event` tolerates a `C`/`M` for an unseen `order_id` — no-op, `book.unseen_cm_count += 1` (Amendment 9 §A9.2: ~0.3 % of `C/M`). `T`, `F`, `N` are unconditional no-ops that never look up the order (Change Log 2026-08-29).
- `apply_event` tolerates a transient crossed market (`best_bid_dbn >= best_ask_dbn`) for less than `config.MAX_TRANSIENT_CROSS_NS`; a cross persisting longer raises `BookInconsistency` (a typed exception defined in `book.py`).
- Int-only (spine AD-10): `price_dbn`, `ts`, `size`, `sequence` are `int`. Clock is `ts_event` (spine AD-1) — `ts_recv` is never read.
- `book.py` imports only `src.ticksim.config`, stdlib, `databento`/`databento_dbn`, `sortedcontainers` (spine AD-7). **Not** `orders.py` — the book holds raw tuples, not `OrderIntent`/`Fill`.
- `mypy --strict` clean (no `[[tool.mypy.overrides]]` for `src.ticksim`); `black`-88; tests `tests/unit/test_ticksim_book.py`.

**Ask First:**
- ~~Whether `F` reduces a hit resting order~~ — RESOLVED against the fixture (Change Log 2026-08-29): on GLBX, `F` is informational (no book delta); the trailing `C`/`M` carries the post-fill size and performs the reduction. `T`, `F`, `N` are all no-ops in `apply_event`.
- Any query name/signature differing from AD-22 (`queue_ahead_size(book, side, price_dbn, our_arrival_ts_ns)`).

**Never:**
- Inserting our own orders into the book (spine AD-3) — no synthetic order ids.
- An O(n) scan (`max(dict)`, full price walk) for best bid/ask (spine AD-4).
- Driving an event loop, reading a DBN file, or `import`ing `databento` for anything but record *types* — folding is `sim.py`'s job (spine AD-20).
- Storing an L2 aggregate as a second structure (spine AD-3) — derive it.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Add then best-bid | `A` bid 100 @ id1, `A` bid 101 @ id2 | `best_bid_dbn == 101` | N/A |
| Cancel top | above, then `C` id2 | `best_bid_dbn == 100`; `order_by_id(id2) is None` | N/A |
| Modify size in place | `A` bid 100 sz 5 id1, `M` id1 same price sz 3 | `size_at_price(BID,100) == 3`; `add_ts_ns` unchanged | N/A |
| Modify price | `A` bid 100 id1, `M` id1 price 99 | order now at 99 with the `M` record's `add_ts_ns`/`sequence` | N/A |
| Fill is informational | `A` ask 100 sz 4 id1, `F` id1 sz 3 | `size_at_price(ASK,100) == 4` (unchanged) — `F` does not touch the book on GLBX | N/A |
| Trailing C reduces | `A` ask 100 sz 4 id1, `F` id1 sz 3, `C` id1 sz 1 | after `C`: `size_at_price(ASK,100) == 3` (the `C` carries the post-fill size and performs the reduction) | N/A |
| Unseen cancel | fresh book, `C` id999 | no-op; `book.unseen_cm_count == 1` | N/A |
| Trade record | any book, `T` … | book depth unchanged; no error | N/A |
| Clear | populated book, `R` for iid | that iid's orders + levels wiped | N/A |
| `N` action | any book, `N` | no-op, no error | N/A |
| Transient cross | cross for < `MAX_TRANSIENT_CROSS_NS` then resolves | no error; cross duration surfaced | N/A |
| Persistent cross | `best_bid >= best_ask` for `>= MAX_TRANSIENT_CROSS_NS` | — | raise `BookInconsistency` |
| queue_ahead | `A` bid 100 sz 2 @ ts=10, `A` bid 100 sz 3 @ ts=20 | `queue_ahead_size(BID,100, our_arrival_ts_ns=15) == 2` | N/A |
| Empty side | fresh book | `best_ask_dbn is None`; `size_at_price(...) == 0` | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/config.py:161` -- `MAX_TRANSIENT_CROSS_NS` (import it). Read-only.
- `src/ticksim/orders.py` -- do **not** import; the book is vendor-tuple-level, schemas are for the sim/consumer boundary.
- `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` -- integration fixture: `db.DBNStore.from_file(...)` yields `databento_dbn.MBOMsg` with `.action` (`Action` enum: `ADD='A'`, `CANCEL='C'`, `MODIFY='M'`, `TRADE='T'`, `FILL='F'`, `CLEAR='R'`, `NONE='N'`), `.side` (`Side` enum: `BID='B'`, `ASK='A'`, `NONE='N'`), `.order_id`, `.price` (int DBN 1e-9), `.size`, `.ts_event`, `.sequence`, `.instrument_id`, `.flags`. Front month `instrument_id == 42004800`.
- `tests/unit/test_ticksim_imports.py:42` -- `PERMITTED_INTERNAL_EDGES["book"]` currently `set()`; change to `{"config"}`.
- `_bmad-output/planning-artifacts/architecture/…/ARCHITECTURE-SPINE.md` -- AD-3 (book shape), AD-9 (`apply_event` sole authority + tolerances), AD-4 (O(log n)), AD-22 (`queue_ahead_size`), AD-10 (int), AD-20/AD-1 (passive, `ts_event`).

## Tasks & Acceptance

**Execution:**
- [ ] `src/ticksim/book.py` -- `BookInconsistency(Exception)`; a structural `Protocol` `MboRecord` (attributes above) so `apply_event` takes both `databento_dbn.MBOMsg` now and `events.py`'s normalized type later; a local `BookSide` enum (`BID`/`ASK`) so stored tuples carry no vendor type; `RestingOrder` (frozen dataclass or `NamedTuple`, the AD-3 tuple); a per-instrument `_InstrumentBook` (orders `dict[int, RestingOrder]`; `bids`/`asks` as `SortedDict[price_dbn -> _PriceLevel]`; `_PriceLevel` = total size + orders ordered by `(add_ts_ns, sequence)`); a top-level `Book` (`dict[instrument_id -> _InstrumentBook]`, `unseen_cm_count: int`, transient-cross state); `apply_event(book, record)` handling `A/C/M/T/F/R/N` per the matrix; queries `best_bid_dbn(iid)`, `best_ask_dbn(iid)`, `size_at_price(iid, side, price_dbn)`, `order_by_id(iid, order_id)`, `queue_ahead_size(iid, side, price_dbn, our_arrival_ts_ns)` (strict `add_ts_ns < our_arrival_ts_ns`), `snapshot_bbo(iid) -> tuple[int | None, int | None]`.
- [ ] `tests/unit/test_ticksim_book.py` -- every I/O-matrix row via small hand-built `MboRecord` stubs; plus one integration test that folds the first ~200k records of the fixture for `instrument_id == 42004800` and asserts: `ts_event` non-decreasing across the fold, no `BookInconsistency` raised, final `best_bid_dbn < best_ask_dbn`, spread within a few ticks, `unseen_cm_count / total < 0.01`.
- [ ] `tests/unit/test_ticksim_imports.py` -- set `PERMITTED_INTERNAL_EDGES["book"] = {"config"}`; the existing isolation + edge tests then cover `book.py`.

**Acceptance Criteria:**
- Given a hand-built A/C/M/F sequence, when folded, then all book queries match the matrix exactly.
- Given the fixture's first ~200k front-month records, when folded, then no exception, `ts_event` monotonic, and the reconstructed BBO is sane (bid < ask, few-tick spread).
- Given `mypy --strict src/ticksim`, then zero errors, no override.
- Given the import-graph test, then `book.py` imports only `config` from `src.ticksim`.

## Spec Change Log

- **2026-08-29 — frozen I/O-matrix rows 5 & 6 corrected (human-authorized).** Trigger: implementation verified `F` semantics against `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` per the Ask-First item — order `6878505582524` folds `F sz1 -> C sz1`, and `databento_dbn` documents `Action.FILL` as not affecting the book. The rows assumed `F` reduces the resting order; on GLBX it is informational and the trailing `C`/`M` performs the reduction. Amended: I/O rows 5 & 6, the Ask-First item, Design Notes, AND the Boundaries bullet (F removed from the unseen-`order_id` counter set — an unconditional no-op is never looked up). Known-bad avoided: treating `F` as a reducer double-counts every fill and ~2.5x's `unseen_cm_count`. KEEP: `apply_event` folding `T`/`F`/`N` identically as no-ops; the fixture-anchored integration test.

## Design Notes

`queue_ahead_size` may be O(orders-at-that-price) — that set is small; only best-bid/ask must be O(log n). `T` (trade) does not mutate the resting book — the hit order is reduced by its own `F`/`C`/`M` (verify vs fixture, Ask-First). `M` with an unchanged price keeps `add_ts_ns` (queue priority preserved, spine AD-8 spirit); `M` with a price change takes the `M` record's `ts_event`/`sequence` as the new key. Best-bid/ask return `None` on an empty side. The transient-cross timer resets whenever the book un-crosses; surface `max_transient_cross_ns` and `unseen_cm_count` as plain `Book` attributes for `sim.py` to fold into the run manifest later.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_book.py tests/unit/test_ticksim_imports.py -q` -- expected: all pass.
- `.venv/bin/python -m mypy --strict src/ticksim` -- expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/book.py tests/unit/test_ticksim_book.py` -- expected: clean.

## Suggested Review Order

**The MBO→book contract (AD-9: the sole authority)**

- Entry point — `apply_event`: the one function that mutates the book; monotonic-ts guard first, then dispatch by action
  [`book.py:407`](../../src/ticksim/book.py#L407)
- `A/C/M/T/F/R/N` handling, incl. the resolved `F`/`T`/`N` no-op and `C`-is-always-full-cancel (fixture-cited)
  [`book.py:441`](../../src/ticksim/book.py#L441)
- `_check_cross` — transient-cross timer; raises `BookInconsistency` past `MAX_TRANSIENT_CROSS_NS`; skips recompute on no-ops
  [`book.py:549`](../../src/ticksim/book.py#L549)

**Malformed-input determinism (the review's main theme)**

- `UNDEF_ORDER_SIZE` sentinel + the `_fail` helper that logs every `BookInconsistency`
  [`book.py:105`](../../src/ticksim/book.py#L105)
- `_InstrumentBook` — dup-`A`, side-flip `M`, missing-level → `BookInconsistency` not `KeyError`
  [`book.py:198`](../../src/ticksim/book.py#L198)
- `check_invariants` — `total_size == Σ member sizes` per level; the fold's self-audit
  [`book.py:363`](../../src/ticksim/book.py#L363)

**Queries**

- `queue_ahead_size` — `add_ts_ns <= arrival` (AD-22 "our order is always last"); `sequence` is for `fills.py`, not here
  [`book.py:334`](../../src/ticksim/book.py#L334)
- `Book` — `unseen_cm_count` / `overcancel_count` / `last_ts_ns` as manifest attributes
  [`book.py:273`](../../src/ticksim/book.py#L273)

**Verification**

- Real-fixture fold: ~600k front-month records, `check_invariants` at the end, `@pytest.mark.integration`, `TICKSIM_REQUIRE_FIXTURE` → fail-not-skip
  [`test_ticksim_book.py:1`](../../tests/unit/test_ticksim_book.py#L1)
- `str(Action.*)` / `str(Side.*)` pins — one place a `databento_dbn` `__str__` change fails
  [`test_ticksim_book.py:1`](../../tests/unit/test_ticksim_book.py#L1)

**Peripherals**

- `PERMITTED_INTERNAL_EDGES["book"] = {"config"}` — turns on the isolation guard for `book.py`
  [`test_ticksim_imports.py:42`](../../tests/unit/test_ticksim_imports.py#L42)
- `integration` marker registered
  [`pyproject.toml`](../../pyproject.toml)
