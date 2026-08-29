---
title: 'ticksim events — BookEventSource, DbnMboSource, merge_streams'
type: 'feature'
created: '2026-08-29'
status: 'done'
review_loop_iteration: 0
baseline_commit: '8a7c54cc7d67459ead0a61461ac241599e2cfc9b'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/` has `config`, the `orders` schemas, and `book.py` (folded by `apply_event`), but nothing to feed the fold. The simulator needs (a) a vendor-agnostic event source protocol, (b) one impl that streams a Databento `.dbn.zst` and normalizes each MBO record, and (c) the stable k-way merge that puts events in the one canonical order the sim loop consumes.

**Approach:** `src/ticksim/events.py` — `BookEventSource` (`Protocol`), a frozen `BookEvent` normalized record (satisfies `book.MboRecord`), `DbnMboSource` (wraps `databento.DBNStore`, streaming, yields `BookEvent`s), and `merge_streams(*sources)` — a **stable** k-way merge keyed `(ts_event, class_rank, sequence, source_index)` (spine AD-20). The vendor boundary lives entirely here; nothing downstream sees a `databento` type.

## Boundaries & Constraints

**Always:**
- `BookEvent` is a frozen dataclass (match `book.RestingOrder` style): `action: MboAction`, `side: MboSide`, `order_id: int`, `price_dbn: int`, `size: int`, `ts_event: int`, `sequence: int`, `instrument_id: int` — all int, clock is `ts_event` (spine AD-1, AD-10). `MboAction`/`MboSide` are `enum.StrEnum` (values the single-char codes `A C M T F R N` / `B A N`) so `str(x)` yields the code and `book.apply_event` folds a `BookEvent` unchanged.
- `BookEventSource` (`Protocol`): iterable yielding `BookEvent` in non-decreasing `ts_event`, and a `class_rank: int` attribute. `DbnMboSource.class_rank == 0` (book delta).
- `DbnMboSource(path)` streams — never materializes the file; `DBNStore` iteration is already lazy and transparently handles `.zst`.
- `merge_streams(*sources)` is a **stable** merge: order key `(ev.ts_event, src.class_rank, ev.sequence, source_index)` where `source_index` is the argument position — equal `(ts_event, sequence)` across sources preserves argument order. Heap-based, O(total · log k).
- `events.py` imports only `.book`, `.orders`, stdlib, `databento`; **relative imports** (`from .book import …`) — `mypy --strict src/ticksim` duplicate-module-errors on the absolute form and no override is allowed.
- `mypy --strict` clean (no `[[tool.mypy.overrides]]` for `src.ticksim`); `black`-88; tests `tests/unit/test_ticksim_events.py`.

**Ask First:**
- Whether to also relax `book.MboRecord.action`/`.side` from `databento_dbn.Action`/`Side` to `str` and drop `book.py`'s `from databento_dbn import Action, Side` — the spec assumes **yes** (a `BookEvent` carrying `MboAction` must satisfy the Protocol; and confining the vendor enum to `DbnMboSource` is the AD-18 intent). This touches `book.py` + its test `_Rec` builder.

**Never:**
- Exposing a `databento` / `databento_dbn` type on any `events.py` public name (`BookEvent`, `BookEventSource`, `merge_streams`, `MboAction`) — the whole point of the module (spine AD-18).
- Reading `ts_recv` (spine AD-1) or `flags`.
- Driving a fold / calling `apply_event` — `merge_streams` yields; `sim.py` consumes (spine AD-20).
- Filtering, deduping, or reordering within a single source — `DbnMboSource` yields every record in file order; `R`/`T`/`F`/`N` pass through (book.py decides what they do).
- Emitting `order_arrival` / `deferred_fill_apply` events (`class_rank` 1 / 2) — those are `sim.py`'s; `merge_streams` must *accept* heterogeneous `class_rank`s but this slice only produces rank 0.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Normalize a record | `MBOMsg` action=ADD side=BID price=P size=3 | `BookEvent(action=MboAction.ADD, side=MboSide.BID, price_dbn=P, size=3, …)` | N/A |
| StrEnum → code | `str(MboAction.MODIFY)` | `"M"` | N/A |
| BookEvent folds in book | `book.apply_event(bk, ev)` for an ADD `BookEvent` | book mutates exactly as with a raw `MBOMsg` | N/A |
| Single source order | one `DbnMboSource` | events yielded in file order, `ts_event` non-decreasing | N/A |
| Two sources, distinct ts | src A ts 10,30; src B ts 20 | merged: 10(A),20(B),30(A) | N/A |
| Tie on ts_event | src A (rank0) ev seq 5 @ ts100; src B (rank0) ev seq 3 @ ts100 | B before A (lower `sequence`) | N/A |
| Tie on ts_event AND sequence | src A ev seq 5 @ ts100; src B ev seq 5 @ ts100 | A before B (argument order — stable) | N/A |
| class_rank ordering | book-delta (rank0) and a stub rank1 source, same ts & seq | rank0 before rank1 | N/A |
| Empty source | `DbnMboSource` over a 0-record slice / `merge_streams()` | yields nothing, no error | N/A |
| Missing file | `DbnMboSource("/nope.dbn.zst")` | — | raise (surface the DBNStore error / `FileNotFoundError`) |

</frozen-after-approval>

## Code Map

- `src/ticksim/book.py:130` -- `MboRecord` Protocol (`action, side, order_id, price, size, ts_event, sequence, instrument_id`); `BookEvent` must satisfy it. `str(record.action)` normalization at the top of `apply_event`. `book.py:77` `from databento_dbn import Action, Side` — the import to remove (Ask-First).
- `src/ticksim/book.py:124` -- `BookSide` enum + `_SIDE_BID/_SIDE_ASK` str constants: the pattern `MboSide` mirrors.
- `src/ticksim/orders.py` -- available per the permitted edge; `events.py` likely needs nothing from it yet (order-arrival events are `sim.py`'s).
- `tests/unit/test_ticksim_imports.py:43` -- `PERMITTED_INTERNAL_EDGES["events"] = {"book", "orders"}` already declared; also check the third-party allow-list note (deferred item).
- `tests/unit/test_ticksim_book.py` -- `_Rec` stub builder: if `MboRecord` is relaxed to `str`, update it to pass single-char codes (or import `events.MboAction`).
- `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` -- integration fixture; front-month `instrument_id == 42004800`.
- `_bmad-output/planning-artifacts/architecture/…/ARCHITECTURE-SPINE.md` -- AD-18 (protocol + vendor confinement), AD-20 (canonical order + stable merge), AD-1 (`ts_event`), AD-10 (int).

## Tasks & Acceptance

**Execution:**
- [ ] `src/ticksim/events.py` -- `MboAction(StrEnum)`, `MboSide(StrEnum)`; `BookEvent` frozen dataclass (fields above); `BookEventSource(Protocol)` (`class_rank: int`; `__iter__ -> Iterator[BookEvent]`); `DbnMboSource` (`__init__(path)`, `class_rank = 0`, `__iter__` streams `DBNStore.from_file(path)` mapping each `MBOMsg` → `BookEvent` via `str(msg.action)` / `str(msg.side)` → the StrEnums; raise on an unknown code); `merge_streams(*sources: BookEventSource) -> Iterator[BookEvent]` — `heapq`-based stable k-way merge on `(ev.ts_event, src.class_rank, ev.sequence, source_index)`; `__all__`.
- [ ] `src/ticksim/book.py` -- (Ask-First) `MboRecord.action`/`.side` → `str`; drop `from databento_dbn import Action, Side`; keep all runtime logic (already string-normalized).
- [ ] `tests/unit/test_ticksim_events.py` -- every I/O-matrix row via hand-built `BookEvent`s and tiny stub sources; a `BookEvent`-satisfies-`MboRecord` structural test; a `book.apply_event(bk, BookEvent(...))` equivalence test; `@pytest.mark.integration` test folding the first ~200k front-month records of the fixture through `DbnMboSource` → `book.apply_event`, asserting `ts_event` monotonic, no `BookInconsistency`, `check_invariants()` clean.
- [ ] `tests/unit/test_ticksim_book.py` -- update `_Rec` builder for the relaxed Protocol (if the Ask-First is taken); full ticksim suite stays green.

**Acceptance Criteria:**
- Given a hand-built `BookEvent`, when passed to `book.apply_event`, then the book mutates identically to the equivalent raw `MBOMsg`.
- Given N `DbnMboSource`s, when `merge_streams` iterates, then output is globally `(ts_event, class_rank, sequence)`-ordered with argument-order as the final stable tie-break.
- Given `mypy --strict src/ticksim`, then zero errors, no override; no `databento` type on an `events.py` public name.
- Given the import-graph test, then `events.py` imports only `book` (and optionally `orders`) from `src.ticksim`.

## Design Notes

`MboAction`/`MboSide` as `enum.StrEnum` (Python 3.11+, confirmed available) is what makes a `BookEvent` both type-safe and drop-in for `book.apply_event` — `str(MboAction.ADD) == "A"`. `merge_streams` uses `heapq.merge`-style lazy iteration: push `(key, source_index, event, iterator)` tuples; the 4-tuple key with `source_index` guarantees stability without comparing events. `DbnMboSource` holds only the `DBNStore` handle + the live iterator — one record in flight. `class_rank` on the Protocol (not the event) because it is a property of the *stream*, not the record.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_events.py tests/unit/test_ticksim_book.py tests/unit/test_ticksim_imports.py -q` -- expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` -- expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` -- expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/events.py tests/unit/test_ticksim_events.py` -- expected: clean.

## Suggested Review Order

**The vendor boundary (AD-18: no databento type past this module)**

- `BookEvent` — frozen, slotted, all-int; the normalized record everything downstream sees
  [`events.py:94`](../../src/ticksim/events.py#L94)
- `DbnMboSource` — the only place `databento` is touched; `__iter__` re-opens a fresh handle each call (H1 grid re-reads)
  [`events.py:140`](../../src/ticksim/events.py#L140)
- `TestVendorConfinement` — asserts no databento type on `__all__` / annotations / public signatures
  [`test_ticksim_events.py:1`](../../tests/unit/test_ticksim_events.py#L1)

**The canonical order (AD-20: stable `(ts_event, class_rank, sequence)` merge)**

- `merge_streams` — eager validate (isinstance + duplicate-source) then a lazy `_merge` under `ExitStack`
  [`events.py:230`](../../src/ticksim/events.py#L230)
- the per-source contract guard: `(ts_event, sequence)` non-decreasing or `ValueError`
  [`events.py:230`](../../src/ticksim/events.py#L230)

**Normalization + sentinels**

- `MboAction`/`MboSide` `StrEnum` — `str(x)` yields the code, so `book.apply_event` folds a `BookEvent` unchanged
  [`events.py:62`](../../src/ticksim/events.py#L62)
- `UNDEF_PRICE` on an ADD/MODIFY → `ValueError` (mirrors book's `UNDEF_ORDER_SIZE`)
  [`events.py:46`](../../src/ticksim/events.py#L46)

**book.py adjustment**

- `MboRecord.price` → `price_dbn`; book.py now imports nothing from `databento`
  [`book.py:130`](../../src/ticksim/book.py#L130)

**Verification**

- committed tiny fixture + generator — the real `DBNStore.from_file` path now runs in the ordinary unit suite
  [`generate_mnq_mbo_tiny.py`](../../tests/fixtures/generate_mnq_mbo_tiny.py)
- `PERMITTED_INTERNAL_EDGES["events"]` guard (already declared) now covers `events.py`
  [`test_ticksim_imports.py:43`](../../tests/unit/test_ticksim_imports.py#L43)
