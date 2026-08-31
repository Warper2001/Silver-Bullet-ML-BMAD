---
title: 'ticksim parity/part_a_runner.py — the Part A MBO-window runner (run_part_a)'
type: 'feature'
created: '2026-08-30'
status: 'done'
review_loop_iteration: 0
baseline_commit: 'dd8a52d'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `parity/part_a.py` (slice 1) is the pure Part A core — it reconstructs `OrderIntent` logs and grades `OrderOutcome`s it is *handed*, but nothing wires it to `sim.simulate` over the real ±90-min MBO windows, and a `leg_unfilled` miss comes back with `signed_error_ticks=None` (magnitude "supplied by the runner slice"). Part A cannot actually run.

**Approach:** add `src/ticksim/parity/part_a_runner.py` — `run_part_a(trades, source_for, *, config=PRIMARY)`: for each `ReconstructedTrade`, get its window `BookEventSource`, `sim.simulate` the trade's intents over that window, filter the outcomes to the trade, `compare_fills`, then resolve every `leg_unfilled` miss to a defined magnitude from the window book's touch at the real fill ts + one tick of adverse slip (spine AD-17), and finally `part_a.aggregate` the full error set → the `PartAResult` verdict. Widen the AD-7 `parity` import edge to include `events` (and `part_a`).

## Boundaries & Constraints

**Always:**
- **`run_part_a(trades: Sequence[ReconstructedTrade], source_for: Callable[[ReconstructedTrade], BookEventSource], *, config: SimConfig = PRIMARY) -> PartAResult`.** One `sim.simulate` call per trade. `source_for` returns a **single-instrument** re-iterable L3 source for that trade's window (front-month filtering is the caller's / window-loader's job — `run_part_a` lets `sim` raise `IntentLogError` if the stream is multi-instrument). `config` defaults to `PRIMARY`; the verdict is always the PRIMARY run (Part A §A8.2 does not grade the OPTIMISTIC model — see Design Notes).
- **Per-trade window interval.** `valid_intervals = [(lo − PART_A_WINDOW_PAD_NS, hi + PART_A_WINDOW_PAD_NS)]` where `lo`/`hi` span every `intent.submit_ts_ns` and every `RealFill.ts_ns` of the trade. `PART_A_WINDOW_PAD_NS` is a module constant (default 5 min in ns) so a boundary order is not expired by the AD-13 mask. A single interval, half-open, per the sim contract.
- **Simulate + scope.** `outcomes, _ = simulate(source, trade.intents, config, valid_intervals)`; the `Manifest` is discarded — `run_part_a` returns a bare `PartAResult` (CHECKPOINT 1; provenance is a `gate.py` / `cli.py` concern). Every returned `OrderOutcome` must have `trade_id == trade.trade_id` — this slice passes exactly one trade's intents, so **assert** it (`all(o.trade_id == trade.trade_id ...)`) rather than filter; a foreign `trade_id` is a `sim` bug worth surfacing, not something to silently drop.
- **Grade** with `part_a.compare_fills(scoped_outcomes, trade)`.
- **Resolve `leg_unfilled` misses (spine AD-17).** For each `FillError` with `signed_error_ticks is None` (`miss_reason == "leg_unfilled"`): read the window book's touch at `err.real_ts_ns` via `_touch_at(source, err.real_ts_ns)` → `(best_bid_dbn, best_ask_dbn)`; the sim exit price is the side the order would have crossed **plus one `MNQ_TICK_DBN` of adverse slip** — a `BUY` exit pays `best_ask_dbn + MNQ_TICK_DBN`, a `SELL` exit receives `best_bid_dbn − MNQ_TICK_DBN`; if that touch side is `None`, raise `PartAError` (an un-priceable miss — the window book is incomplete at `exit_ts`, a data fault worth surfacing). Rebuild the `FillError` via `dataclasses.replace` with `sim_vwap_dbn` and `signed_error_ticks` filled in (sign per the same convention: positive = sim worse for the trader); `miss_reason` stays `"leg_unfilled"` so a resolved-from-touch grade is still identifiable, and `aggregate`'s unresolved count (which keys on `signed_error_ticks is None`) now sees it as resolved.
- **`_touch_at(source, ts_ns) -> tuple[int | None, int | None]`.** A bounded read-only book replay: `book = Book()`; iterate a fresh pass of `source`; `for ev in source: if ev.ts_event > ts_ns: break; apply_event(book, ev)`; return `book.snapshot_bbo(instrument_id)` where `instrument_id` is taken from the events walked (all share one; captured from the first). No fills, no tracker, no mutation of anything outside the local `Book`.
- **Aggregate.** `part_a.aggregate(all_resolved_errors)` — one call over the concatenated per-trade error lists → the returned `PartAResult` **is** `run_part_a`'s return value. Do not re-implement any of `aggregate`'s math or the verdict rule, and do not wrap it in a new result type.
- **AD-7.** Widen the spine's `parity` edge to `parity → sim, report, book, orders, config, events` (inline dated note); `part_a_runner.py` imports `sim`, `events`, `book`, `orders`, `config`, and its sibling `part_a`. Add `"part_a_runner"` to `PERMITTED_INTERNAL_EDGES` and teach the import-graph resolver that a `src.ticksim.parity.<sibling>` target is the sibling stem (so `from .part_a import …` is checkable as `part_a`, not `parity`).
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports.

**Ask First:**
- (resolved at CHECKPOINT 1, 2026-08-30) PRIMARY-only run; `_touch_at` stays a simple per-miss source re-walk. `run_part_a` returns a bare `PartAResult` (no manifests — provenance is a `gate.py` / `cli.py` concern).

**Never:**
- Loading `.dbn.zst` paths, resolving which window file belongs to which trade, or front-month instrument filtering — that is the caller / a window-loader / `cli.py`. `run_part_a` takes `source_for`.
- Re-deriving `compare_fills` / `aggregate` / the verdict rule / the sign convention — all live in `part_a.py`.
- Feeding a real fill price into `sim` (AD-17). Evaluating Part B, the §6 H1 verdict, or writing the frozen SHA (`gate.py`). Any network call or Tranche-1 pull.
- Mutating the `Book` the sim owns, or sharing a `Book` between `_touch_at` calls.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| trade with both legs filled in sim | reconstructed trade + a window source that fills entry & exit | `PartAResult` with 2 `FillError`s, both `signed_error_ticks` set | N/A |
| exit leg unfilled in sim | sim exit `terminal_state != FILLED`, book has a touch at `exit_ts` | miss resolved: `sim_vwap_dbn = touch ± tick`, `signed_error_ticks` set, `miss_reason` still `"leg_unfilled"` | N/A |
| exit leg unfilled, book has no touch that side at `exit_ts` | `_touch_at` returns `None` for the crossed side | — | `PartAError` (un-priceable miss) |
| multi-instrument window source | `source_for` returns an unfiltered parent-symbol stream | — | `IntentLogError` (propagated from `sim`) |
| window source not re-iterable | `source_for` returns a one-shot iterator | — | `RuntimeError` (propagated from `sim` / second `_touch_at` walk) |
| several trades, one has a miss | 3 trades, trade 2's exit unfilled | one `aggregate` call over all 6 errors; verdict reflects the resolved miss | N/A |
| all bounds met, N≥28 | enough trades, small errors | `PartAResult.verdict == "PASS"` | N/A |
| `source_for` raises | e.g. missing window file | — | propagates unchanged |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/part_a_runner.py` — NEW. `run_part_a(...) -> PartAResult` + `_touch_at(...)` + `PART_A_WINDOW_PAD_NS`. Re-exports nothing from `part_a`.
- `src/ticksim/parity/part_a.py` — `ReconstructedTrade`, `RealFill`, `FillError` (has `real_ts_ns`, `sim_terminal_state`, `miss_reason`), `PartAResult`, `PartAError`, `compare_fills`, `aggregate`.
- `src/ticksim/sim.py:709` — `simulate(book_event_source, intent_log, config, valid_intervals, *, degraded_days=()) -> tuple[list[OrderOutcome], Manifest]`; raises `IntentLogError` (multi-instrument stream / non-replayable log), `InvariantViolation`.
- `src/ticksim/events.py:122` — `BookEventSource` Protocol (re-iterable, yields `BookEvent` with `ts_event`, `instrument_id`); `DbnMboSource(path)`. `BookEvent` folds through `book.apply_event` unchanged.
- `src/ticksim/book.py:286` — `Book()`; `apply_event(book, record)`; `book.snapshot_bbo(instrument_id) -> (int|None, int|None)`; `best_bid_dbn` / `best_ask_dbn`.
- `src/ticksim/config.py` — `PRIMARY`, `OPTIMISTIC`, `SimConfig`, `MNQ_TICK_DBN=250_000_000`.
- `tests/unit/test_ticksim_imports.py:39` — add `"part_a_runner": {"sim", "events", "book", "orders", "config", "part_a"}`; improve `_internal_targets` so `parity.<sibling>` → `<sibling>`.
- `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` — the one available window (2026-06-22, straddles yank ids 2900–2902); the integration test's source.
- `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-7 (widen `parity → events`), AD-17 (the unfilled-exit miss rule), AD-18 (`BookEventSource`), AD-13 (mask / `valid_intervals`).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/part_a_runner.py` — `run_part_a` (+ `pad_ns` kwarg), `_touch_at` (single-instrument + monotonic-ts guards), `_resolve_leg_unfilled`, `_require_reiterable`, `PART_A_WINDOW_PAD_NS`, `__all__`, relative imports.
- [x] `src/ticksim/parity/__init__.py` — extended the allowed-edges docstring for `events` + `part_a_runner`.
- [x] `src/ticksim/parity/part_a.py` — `PartAError` docstring extended for the runner's window-book-data-fault use.
- [x] `tests/unit/test_ticksim_imports.py` — `"part_a_runner"` row; `_internal_targets` takes `importer_package` and rewrites `parity.<sibling>` → `<sibling>` **only for importers inside `src.ticksim.parity`**; 3 resolver unit tests.
- [x] `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-7 graph edge `parity --> events`, rule-line list updated, inline note.
- [x] `tests/unit/test_ticksim_parity_run_part_a.py` — 33 tests: both-legs-filled offsets + N-floor FAIL; exit-unfilled resolved from touch (BUY + SELL exit); entry-leg unfilled miss; un-priceable miss → `PartAError`; multi-trade single `aggregate` call; `simulate` once per trade; `config=OPTIMISTIC` forwarded; multi-instrument → `IntentLogError`; empty `trades` (no `source_for` call); duplicate `trade_id`; `broker_fill` fidelity path; window-pad / `pad_ns=0` / negative-epoch floor; `_touch_at` boundedness / inclusive-lower / empty; non-re-iterable + `source_for` error propagation.
- [x] `tests/integration/test_ticksim_parity_run_part_a_integration.py` — `@pytest.mark.integration` (renamed off the unit basename), skips without the window fixture; reconstructs the 3 yank 2026-06-22 trades, a front-month-filtered + ≥30-min-lead-clipped `DbnMboSource` source, asserts a `PartAResult` with ≥3 finite `FillError`s; a second test proves an unfiltered multi-instrument source → `IntentLogError`.

**Acceptance Criteria:**
- Given a reconstructed trade and an in-memory source that fills both legs a known offset from the real fills, when `run_part_a` runs, then the `PartAResult` has 2 `FillError`s with `signed_error_ticks` equal to that offset (sign per the convention) and `verdict == "FAIL"` on `n < 28`.
- Given a trade whose sim exit is unfilled and a source whose book at `real_ts_ns` has `best_ask_dbn = A`, when the exit side is `BUY`, then the resolved `FillError.sim_vwap_dbn == A + MNQ_TICK_DBN`.
- Given `_touch_at(source, T)`, then no `BookEvent` with `ts_event > T` has been applied to the returned BBO.
- Given `mypy --strict src/ticksim`, then zero errors, no override; the import-graph test passes with the new `"part_a_runner"` row and the `parity.<sibling>` resolver fix.

## Spec Change Log

**Review round 1 — 2026-08-30 — patch round (no code re-derivation).** Reviewer trio (blind-hunter, edge-case-hunter, verification-gap). All findings triaged patch or defer — no `intent_gap`/`bad_spec` loopback. One frozen-text reconciliation: the "Simulate + scope" bullet still said "Pass `_manifest` through into the result for provenance" — a leftover contradicting the **CHECKPOINT-1 decision the human already made** (bare `PartAResult`, no manifests). The code was already correct (manifest discarded); the frozen text is aligned to the CHECKPOINT-1 answer here, not re-negotiated. The scoping filter is also tightened from a silent `[o for o in outcomes if …]` to an `assert` (a foreign `trade_id` is a `sim` bug, not something to drop). Patches applied to `part_a_runner.py` + tests: `_touch_at` single-instrument + monotonic-ts fail-closed guards, entry-vs-exit-neutral miss messages, `pad_ns` keyword (default still `PART_A_WINDOW_PAD_NS`), `max(0, …)` interval floor, duplicate-`trade_id` guard, unresolved-non-`leg_unfilled` guard, dropped the inert `+ 0.0`, `Raises` docstring completion, `PartAError` docstring extended (window-book-data-fault case), import-graph resolver scoped to parity-internal importers. New running unit tests: multi-instrument → `IntentLogError`, `config=OPTIMISTIC` forwarded, `simulate` once per trade, empty `trades` (no `source_for` call), duplicate `trade_id`, `broker_fill`-fidelity trade, entry-leg miss, negative-epoch interval floor.

*Not applied:* an up-front `_require_reiterable` guard (reviewer suggestion) — the frozen I/O matrix's contract is **lazy** detection ("propagated from sim / second `_touch_at` walk"), and a no-miss run over a one-shot source is genuinely fine (sim consumes it once, no re-walk needed). Kept lazy; `RuntimeError` still surfaces on any run that needs `_touch_at`.

## Design Notes

**PRIMARY-only.** The seal (§2.1) makes the OPTIMISTIC model deliberately generous (time-priority queue, 50 ms latency) — "reported, never decision-bearing". §A8.2 Part A grades fill-price fidelity of the *decision-bearing* model. Running Part A against OPTIMISTIC would test whether a knowingly-optimistic model matches reality, which is not a meaningful gate. A caller may still call `run_part_a(config=OPTIMISTIC)` for a diagnostic stat line.

**Why `run_part_a` doesn't own window-file resolution.** The 28 windows (`~/.claude/jobs/960bda86/tmp/parity_windows.py`) are not yet purchased (only the 2026-06-22 test window exists). Keeping `source_for` an injected callable lets this slice ship + be unit-tested now with in-memory sources, and lets `cli.py` / a window-loader own the `.dbn.zst` path map, front-month `instrument_id` filtering (`MNQ.FUT` parent is 96 % front month, 4 % spread — Amendment 9), and the degraded-day flag when the real data lands.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_run_part_a.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m pytest tests/integration/test_ticksim_parity_run_part_a.py -q -m integration` — expected: passes against the test window.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity tests/unit/test_ticksim_parity_run_part_a.py` — expected: clean.

## Suggested Review Order

**Entry point — the per-trade loop**

- `run_part_a`: one `simulate` per trade, foreign-`trade_id` assert, the concatenate-then-one-`aggregate` shape, `trades == []` → n=0 FAIL.
  [`part_a_runner.py:69`](../../src/ticksim/parity/part_a_runner.py#L69)

**AD-17 — resolving a `leg_unfilled` miss**

- `_resolve_leg_unfilled`: touch ± one tick adverse slip, sign convention mirrors `compare_fills`, `miss_reason` kept, non-`leg_unfilled` unresolved error rejected.
  [`part_a_runner.py:165`](../../src/ticksim/parity/part_a_runner.py#L165)

- `_touch_at`: bounded read-only book replay, single-instrument + monotonic-ts fail-closed guards, `(None, None)` when nothing precedes `ts_ns`.
  [`part_a_runner.py:233`](../../src/ticksim/parity/part_a_runner.py#L233)

**Window + source guards**

- `_window_span` (spans intent + real-fill stamps) and the `max(0, lo - pad_ns)` interval floor; `pad_ns` keyword.
  [`part_a_runner.py:155`](../../src/ticksim/parity/part_a_runner.py#L155)

- `_require_reiterable`: catch a generator before `sim` silently consumes it.
  [`part_a_runner.py:144`](../../src/ticksim/parity/part_a_runner.py#L144)

**Import graph**

- `_internal_targets` now scoped by importer package — `parity.<sibling>` → `<sibling>` only from inside `src.ticksim.parity`.
  [`test_ticksim_imports.py:106`](../../tests/unit/test_ticksim_imports.py#L106)

**Peripherals**

- AD-7 widening: graph edge, rule line, inline note.
  [`ARCHITECTURE-SPINE.md`](../planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md)

- 33 unit tests; integration test (skips without the DBN fixture).
  [`test_ticksim_parity_run_part_a.py:1`](../../tests/unit/test_ticksim_parity_run_part_a.py#L1)
