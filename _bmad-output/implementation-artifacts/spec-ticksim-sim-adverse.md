---
title: 'ticksim sim.py — AD-28 adverse_selection deferred-check queue'
type: 'feature'
created: '2026-08-30'
status: 'done'
review_loop_iteration: 1
baseline_commit: '18a58a4'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/implementation-artifacts/spec-ticksim-sim.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `src/ticksim/sim.py` ships every `OrderOutcome.adverse_selection` as `False` — the AD-28 deferred-check was carved off the sim.py slice. Every study needs the marker: a passive fill followed within 1 s by the market moving against it (prereg §2.1).

**Approach:** add the AD-28 bounded deferred-check queue to `SimRun` — a **step 6** in the tick loop (AD-20 `deferred_fill_apply`, class_rank 2). No second replay (AD-14). The predicate is Alex-pinned (2026-08-29): **any point in the 1 s window**, and **same-side quote moves away** (BUY fill @P adverse iff best bid later `< P`; SELL fill @P adverse iff best ask later `> P`).

## Boundaries & Constraints

**Always:**
- **Enqueue** (`_step_fills`): for each `FillEvent` from `fills.decide`, **before** `tracker.apply_fill` (the order is still `WORKING`), read `snap = tracker.snapshot(fe.order_id)`. If `snap.kind is PASSIVE_LIMIT`, append an `_AdverseCheck(order_id=fe.order_id, price_dbn=fe.px_dbn, side=snap.side, deadline_ns=fe.ts_ns + config.ADVERSE_SELECTION_WINDOW_NS, hit=False)` to `self._adverse_checks` (a plain `list`, push order — deterministic, AD-11). `MARKETABLE` / `MARKETABLE_LIMIT` fills enqueue nothing (§2.1 adverse-selection is passive-only).
- **Evaluate + seal** — a new `_step_adverse(now_ns)` called every tick **after** step 5, in *and* out of the mask (the book is continuous; a fill's 1 s window may cross an interval boundary), and once more at run end (with `now_ns = book.last_ts_ns` or a sentinel `≥` every deadline):
  - *Evaluate:* for every not-yet-sealed check with `fill_ts < now_ns <= deadline_ns` (strictly after the fill tick), latch `check.hit = True` if — `side is BUY` and `book.best_bid_dbn(iid)` is not `None` and `< check.price_dbn`; or `side is SELL` and `book.best_ask_dbn(iid)` is not `None` and `> check.price_dbn`. A `None` touch never triggers (a quote that is not there did not "move through" a price). `fill_ts = deadline_ns - config.ADVERSE_SELECTION_WINDOW_NS`.
  - *Seal:* for every check with `deadline_ns <= now_ns` (or at run end): if `check.hit`, `tracker.set_adverse_selection(check.order_id, True)`; remove it from the list. A not-hit check needs no call (`adverse_selection` defaults `False`). `self._adverse_fill_count += 1` per sealed check that was `hit`.
- Deadlines are **not** wake points. `_step_adverse(now_ns, *, evaluate)` — the **evaluate** branch runs only when a book delta was folded at this tick (`evaluate=True`); the BBO moves only on a book event, so a bare arrival / interval-bound wake must **not** latch `hit` (else the marker would depend on unrelated timing, AD-11). The **seal** branch runs unconditionally. A check that gets no *book* tick inside its window seals `hit=False` — correct (the market did not move against the fill after it landed).
- `iid` = the sole book instrument (`self._iid`, or `0` before any book event, matching `_step_arrivals`).
- Run-end sealing happens **before** `self.book.check_invariants()` / `self.tracker.finalize()` in `SimRun.run` (AD-28: mutable "until run end, then serialized").
- `Manifest` gains `adverse_fill_count: int` (checks that latched `hit`) — parallel to `oco_cascade_cancel_count`; added to `to_dict()`.
- `PERMITTED_INTERNAL_EDGES["sim"]` unchanged (`{config, book, orders, events, fills}`). `mypy --strict src/ticksim` clean, no override. `black`-88. No wall-clock. No `assert` for parity-relevant logic.

**Never:**
- Editing `config` / `book` / `orders` / `fills` / `events` (`OrderOutcome.adverse_selection` and `OrderTracker.set_adverse_selection` already exist; `set_adverse_selection` already guards "FILLED, not-yet-finalized").
- A second `SimRun` / replay pass (AD-14).
- Marking a `MARKETABLE` / `MARKETABLE_LIMIT` fill adverse.
- Look-ahead in `fills.decide` (this is entirely `sim`-side, after the fill).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Passive BUY, bid drops in-window | passive BUY fills @P at `t`; a later `C` drops best bid to `< P` at `t + 0.5s` | `outcome.adverse_selection is True`; `manifest.adverse_fill_count == 1` | N/A |
| Passive BUY, bid drops after the window | best bid drops `< P` only at `t + 1.5s` | `adverse_selection is False` | N/A |
| Transient dip that reverts | best bid `< P` at `t + 0.3s`, back `>= P` by `t + 0.6s` | `adverse_selection is True` (any-point) | N/A |
| Passive SELL, ask rises in-window | passive SELL fills @P; best ask `> P` within 1s | `adverse_selection is True` | N/A |
| Quote side empties | passive BUY fills @P; best bid becomes `None` in-window (no other move) | `adverse_selection is False` (a `None` touch never triggers) | N/A |
| Marketable fill, adverse move | marketable BUY fills; best bid later `< fill px` | `adverse_selection is False` — never marked | N/A |
| Window crosses interval end | fill at `t` near an interval `end`; adverse move at `t + 0.4s`, past `end` | still evaluated (book continuous); `adverse_selection is True` | N/A |
| Run ends before the window closes | fill in the last 0.5s of the run; adverse move before run end | sealed at run end with the latched `hit` | N/A |
| Fill tick itself is adverse | best bid already `< P` at the exact fill tick, never after | `adverse_selection is False` (strict `now_ns > fill_ts`) | N/A |
| Non-book wake in the window | bid `< P` for the whole window, but the only in-window wake is an order arrival (no book delta) | `adverse_selection is False` — the evaluate branch skips a non-book tick (AD-11) | N/A |
| Seal inside the loop | a real book event lands past the deadline | check seals in `_loop` (not the run-end sentinel); a still-open later check is not re-counted | N/A |
| Marketable-limit fill, adverse move | `MARKETABLE_LIMIT` fills; a later bid far below fill px | `adverse_selection is False` — never marked (passive-only) | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/sim.py` — `SimRun.__init__` (add `self._adverse_checks: list[_AdverseCheck] = []`, `self._adverse_fill_count = 0`); `_step_fills` (enqueue, before `apply_fill`); `_loop` (call `self._step_adverse(now_ns)` after the `if in_mask:` fills block, every tick); `run()` (seal remaining before `check_invariants` / `finalize`); `_build_manifest` (`adverse_fill_count=`). `Manifest` dataclass + `to_dict()`.
- `src/ticksim/book.py:312` — `Book.best_bid_dbn(iid) -> int | None`, `:317` `best_ask_dbn(iid) -> int | None`.
- `src/ticksim/orders.py` — `OrderTracker.snapshot(oid) -> OrderSnapshot` (`.kind`, `.side`); `.set_adverse_selection(oid, value)` (works on a FILLED not-yet-finalized order; raises otherwise). `FillEvent` (`order_id, px_dbn, size, ts_ns`). `Side` (BUY/SELL), `OrderKind`.
- `src/ticksim/config.py` — `ADVERSE_SELECTION_WINDOW_NS = 1_000_000_000`.
- spine — AD-28 (bounded deferred check, not a replay), AD-20 (class_rank 2), AD-11 (deterministic iteration), prereg §2.1 (adverse-selection row).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/sim.py` — `_AdverseCheck` (`@dataclass`, mutable `hit`), enqueue in `_step_fills` (guarded: `WORKING` pre-fill, `PASSIVE_LIMIT`, `FILLED` post-fill), `_step_adverse(now_ns, *, evaluate)`, the run-end seal, `Manifest.adverse_fill_count` + `to_dict`. Module docstring updated. `_AdverseCheck` stays private.
- [x] `tests/unit/test_ticksim_sim.py` — 14 AD-28 tests: every matrix row + `adverse_fill_count` + the review-1 set.

## Suggested Review Order

**Enqueue** — `_step_fills`: `snap` read while the order is still `WORKING` (a same-batch OCO-cascade can already have cancelled a losing leg → `live_state` guard); enqueue only when `snap.kind is PASSIVE_LIMIT` **and** the order is now `FILLED` (`set_adverse_selection` contract).

**Step 6** — `_step_adverse(now_ns, *, evaluate)`:
- `evaluate` only when a book delta folded this tick (`self._event_count > events_before`) — the AD-11 fix
- latch: `fill_ts < now_ns <= deadline_ns` and same-side quote away (`best_bid < P` / `best_ask > P`; `None` never)
- seal: `deadline_ns <= now_ns` → `set_adverse_selection` iff `hit`; list rebuilt only on a seal (`_adverse_checks[0].deadline_ns > now_ns` short-circuit)

**Run end** — `run()` calls `_step_adverse(_max_deadline + 1, evaluate=False)` **before** `check_invariants` / `finalize` (AD-28 mutability).

**Manifest** — `adverse_fill_count` (orders, not events).

**Known biases** — see the Spec Change Log; all conservative (under-flag), all documented, marker is a non-decision-bearing diagnostic.

**Acceptance Criteria:**
- Given a passive BUY that fills @P and a book event within `ADVERSE_SELECTION_WINDOW_NS` that drops the best bid below P, when `simulate` runs, then the order's `OrderOutcome.adverse_selection is True` and `manifest.adverse_fill_count == 1`.
- Given the same but the drop occurs after the window, then `adverse_selection is False`.
- Given a marketable fill with any subsequent move, then `adverse_selection is False`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; the AD-11 run-twice + cross-`PYTHONHASHSEED` tests still pass (deterministic check-list iteration).

## Spec Change Log

### 2026-08-30 review-1 (blind / edge-case / verification-gap) — `bad_spec` on the wake dependency

**Defect:** the spec said "`hit` is latched on whatever ticks fall in the window … no in-window BBO state is missed." Reviewers showed the evaluate step also fired on **non-book wakes** (an order arrival, an interval bound) that happen to land in a check's window, sampling the current BBO regardless of whether a book event moved it — so the marker became dependent on unrelated timing (AD-11 violation). **Fix:** `_step_adverse(now_ns, *, evaluate)` — the evaluate branch runs only when a book delta was folded this tick; seal always. Frozen block amended + a matrix row.

Also patched (reviewer-driven): the `survivors` list is rebuilt only when a check actually seals (`_adverse_checks[0].deadline_ns > now_ns` short-circuit — push order is non-decreasing deadline), not every tick over ~22.5M ticks; the "class_rank 2" mislabel corrected (step 6 is AD-20's "deferred fill application", not one of the three merged-stream classes); `run()` `Raises:` + module docstring note the AD-28 `set_adverse_selection` path; `Manifest.adverse_fill_count` gets a field comment (it counts *orders*, unlike `oco_cascade_cancel_count` which counts events). +4 tests (genuine run-end seal, in-loop seal + no-recount, non-book-wake no-latch, `MARKETABLE_LIMIT`); the cross-`PYTHONHASHSEED` determinism test now actually latches two hits sealing in one tick.

**Spine AD-28 amended** in the same pass (inline dated note): its "book state 1 s *after* the fill" wording is superseded by Alex's 2026-08-30 pin — any point in the window, same-side quote moves away, evaluated on book ticks.

### Known conservative biases (documented, not defects — the marker is a diagnostic, "not penalised beyond the fill price itself" per §2.1)

- **Under-flags the "same sweep" case.** If the trade that fills a passive order *also* drives the same-side quote through P and no *later* book event lands in the window, the order seals non-adverse (the fill tick itself is excluded, strict `>`). Dense MNQ MBO makes this rare; direction of the bias is toward under-counting.
- **Same-side only.** An opposite-side collapse (passive BUY @P, best ask craters to 98 within 1 s, one lone bid stays at P) is **not** flagged — Alex's pin. The marker measures "the quote we joined moved away", not "the fill looks bad from either side".
- **Multi-partial passive fills** anchor the 1 s window (and `price_dbn`) to the *completing* fill only — `set_adverse_selection` requires a `FILLED` order. Adverse moves between the first partial and completion, and the filled quantity of a passive order that partial-fills then EXPIRES/CANCELS, are not measured. Deferred (`deferred-work.md`); rare for 1–5-lot H1 which fills in one shot.
- **Mid-window `R` (book clear)** empties the same side → `None` quote → does not latch (the pinned `None`-never-triggers rule). An `R` is a halt/reset, expected outside the AD-13 mask.
- **Intra-tick transient** — step 1 folds all same-`ts` deltas atomically before step 6 reads the BBO, so a bid that dips below P and recovers within one `ts` is invisible (AD-20 atomic same-`ts` fold).

## Design Notes

`_step_adverse` is O(open checks) per tick (open checks ≤ passive fills in the last 1 s, tiny for H1) and rebuilds its list only on a seal. The run-end seal is `self._step_adverse(self._max_deadline + 1, evaluate=False)` — `_max_deadline` tracked at enqueue; `+ 1` is past every remaining deadline. The interpretation of §2.1's "quote move *through* our price" is: the observed BBO *state* `best_bid_dbn < P` (BUY) / `best_ask_dbn > P` (SELL) at an in-window book tick strictly after the fill; strict `<` / `>` (a quote resting exactly at P is not adverse); same-side only — all per Alex's 2026-08-30 pin.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_sim.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/sim.py tests/unit/test_ticksim_sim.py` — expected: clean.
