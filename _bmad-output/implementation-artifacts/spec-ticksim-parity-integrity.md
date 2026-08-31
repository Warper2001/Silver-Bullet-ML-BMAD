---
title: 'ticksim parity/integrity.py — the §5 MBO-window integrity preflight'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 1
baseline_commit: 'fc42b9f'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Prereg §1 / §A9.3 require an **integrity check before use** on every contract-window file — monotonic non-decreasing timestamps; no *persistent* crossed market (> `MAX_TRANSIENT_CROSS_NS`); trade prices within the session's own low/high; the reconstruction must consume `A/C/M/T/F`; report the exact pass rate; flag `degraded` days (2026-05-24, 2026-07-30) without dropping them. Spine AD-26 says `gate.py`'s amendment stub carries this integrity report (currently a `pending` placeholder). Nothing produces it.

**Approach:** add `src/ticksim/parity/integrity.py` — `preflight_integrity(source, *, degraded_days=()) -> IntegrityReport`: one read-only pass over a window `BookEventSource`, folding a `Book` to watch the BBO, counting ts regressions / transient vs persistent crosses / out-of-session-range trades / `BookInconsistency`s / missing action classes. Plus `format_integrity(report) -> str` producing the Markdown block `gate.build_amendment_stub` drops into its `integrity:` slot. **Verdict-reporting, not verdict-bearing** — a `FLAGGED` report tells the analyst the data is suspect; whether the parity-gate CLI refuses on it is that slice's decision.

## Boundaries & Constraints

**Always:**
- **`preflight_integrity(source: BookEventSource, *, degraded_days: Sequence[str] = ()) -> IntegrityReport`.** One pass over `source` (re-iterable per AD-18; the function iterates it once). `source` is a **single-instrument** stream (front-month filtering is the caller's job) — a second `instrument_id` is itself an integrity failure and is counted + flags the report (it does **not** raise; the whole point is to survey a possibly-broken file).
- **Fold a `Book`, never raise.** The whole per-event body runs inside `try/except Exception` (a preflight surveys a *possibly-broken* file — a mid-stream `ValueError` from the vendor source, a malformed record, anything) → `malformed_events += 1`, `continue`. `preflight_integrity` only ever raises if `iter(source)` itself fails. A **foreign-instrument** event (its `instrument_id != instrument_id`) is counted (`foreign_instrument_events`) and **skipped** — not folded into the primary `Book`, not added to `actions_seen`. Otherwise `book.apply_event(book, ev)`; a `BookInconsistency` is caught and bucketed (see below); after a non-crossed event read `book.snapshot_bbo(instrument_id)`. At end of stream call `book.check_invariants()` in a `try` → `check_invariants_failed: bool`.
- **BookInconsistency bucketing.** A caught `BookInconsistency` is attributed as: `warmup_unknown_ref` **only** when it is an unknown-order `C`/`M` reference (detected via the `book` counter delta, e.g. `unseen_cm_count`) **and** `0 <= ev.ts_event − first_ts_ns <= _WARMUP_NS`; a caught inconsistency for the event that is *itself* ts-regressed (see below) is not counted (it is the regression, already flagged); a caught inconsistency while a cross is currently open (the BBO-watch state machine says so) is not counted (it is the cross, already flagged); **everything else** → `book_inconsistencies` (flags).
- **Timestamp monotonicity + gaps.** Track `max_ts_seen` (the running maximum, **not** just the immediate predecessor — `book.apply_event` guards on its own running max, so comparing to the predecessor mis-attributes a cascade of shuffled timestamps). `ev.ts_event < max_ts_seen` → `ts_regressions += 1` (+ capped `(max_ts_seen, ev.ts_event)` examples) and this event is "regressed". `first_ts_ns` = the first event's ts; `last_ts_ns` = the last's; `duration_ns = last − first`. Track `largest_gap_ns = max forward jump between consecutive events`; `gaps_over_threshold = count of forward jumps > _MAX_GAP_NS` (an unrecorded halt / missing data — §1) — a non-zero count flags.
- **Crossed markets.** After folding a non-foreign event, if `bid is not None and ask is not None and bid >= ask` the book is crossed. One open-cross variable; when it resolves (`bid < ask` or a side `None`) `duration = resolve_ts − start_ts`: `duration < 0` (a ts regression opened inside the cross) **or** `duration > MAX_TRANSIENT_CROSS_NS` → `persistent_cross_count += 1` + a capped `(start, resolve, duration)` entry; else `transient_cross_count += 1`. `bbo_cross_rate = (transient_cross_count + persistent_cross_count) / n_events` (the §A9.3 "0.014 %" figure; `0.0` when `n_events == 0`). A cross still open at stream end → `unresolved_cross_at_end = True` (its own flag), **not** folded into either count.
- **Trade-price checks.** For an `ev.action == TRADE` (price `p`): update `session_low_dbn` / `session_high_dbn`; `n_trades += 1`. And — the off-book-print check (CHECKPOINT 1): if the book BBO at that instant is two-sided (`bid`, `ask` both non-`None`) and `p` is more than `_TRADE_BBO_TOLERANCE_TICKS * MNQ_TICK_DBN` above `ask` or below `bid` → `trades_off_book += 1`. A one-sided or empty book at that instant is not counted (nothing to compare against). (Off-book prints get a count, not an example list — the `IntegrityReport` field list is authoritative.)
- **Action coverage.** `actions_seen: frozenset[str]` collects `str(ev.action)` for **primary-instrument** events only (`"A"`/`"C"`/`"M"`/`"T"`/`"F"`). `missing_actions = tuple(sorted({"A","C","M","T","F"} − actions_seen))`. A missing `T`/`F` means the fill logic was never exercised by this file → flags.
- **Warm-up references.** `warmup_unknown_ref` = unknown-order `C`/`M` references inside the first `_WARMUP_NS` — bucketed per "BookInconsistency bucketing" above; expected (§A9.2 ~0.3 %) and **never flags**. The same unknown-ref *after* `_WARMUP_NS` → `book_inconsistencies` (flags).
- **`IntegrityReport`** is a frozen dataclass: `n_events: int`, `malformed_events: int`, `instrument_id: int | None`, `foreign_instrument_events: int`, `first_ts_ns: int | None`, `last_ts_ns: int | None`, `duration_ns: int`, `largest_gap_ns: int`, `gaps_over_threshold: int`, `ts_regressions: int`, `ts_regression_examples: tuple[tuple[int,int],...]`, `transient_cross_count: int`, `persistent_cross_count: int`, `persistent_crosses: tuple[tuple[int,int,int],...]`, `bbo_cross_rate: float`, `unresolved_cross_at_end: bool`, `n_trades: int`, `session_low_dbn: int | None`, `session_high_dbn: int | None`, `trades_off_book: int`, `actions_seen: frozenset[str]`, `missing_actions: tuple[str,...]`, `book_inconsistencies: int`, `warmup_unknown_ref: int`, `check_invariants_failed: bool`, `degraded_days: tuple[str,...]`, `verdict: Literal["OK","FLAGGED"]`, `flags: tuple[str,...]`. `verdict == "FLAGGED"` iff any of: `ts_regressions > 0`, `persistent_cross_count > 0`, `unresolved_cross_at_end`, `foreign_instrument_events > 0`, `missing_actions != ()`, `book_inconsistencies > 0`, `trades_off_book > 0`, `malformed_events > 0`, `gaps_over_threshold > 0`, `check_invariants_failed`, `n_events == 0`. Transient crosses, `degraded_days`, and `warmup_unknown_ref` **never** flag (all expected — §A9.2/§A9.3). `flags` is a fixed-order human list of the reasons (empty iff `verdict == "OK"`).
- **`format_integrity(report: IntegrityReport) -> str`** — a fixed-template Markdown block (no `#`/`##` headings — it nests inside the gate stub's section): `integrity: OK` or `integrity: FLAGGED (<reasons joined by ", ">)` — the header is derived from `report.flags` (a `FLAGGED` report always has ≥1 flag, an `OK` report has none, so `"FLAGGED ()"` is unrepresentable); the window span (`first_ts_ns .. last_ts_ns`, `duration_ns`); one bullet per counter incl. `bbo_cross_rate` as a percentage and `session_low/high` labelled `(informational)`; the degraded-day note (`str(d)` each — tolerate a non-`str`); and, on `FLAGGED`, the capped regression / persistent-cross examples. Deterministic; ASCII-only (`--`, no em-dash); byte-identical across calls.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. `PERMITTED_INTERNAL_EDGES["integrity"] = {"events", "book", "config"}`. Stdlib: `dataclasses`, `typing` only — no wall-clock, no `logging`.
- **Do not re-implement `book`'s cross timer.** The BBO-watch state machine here is for *classifying* a cross's duration from the observed BBO; it must not diverge from `book`'s own `cross_start_ns` semantics on an `R` (clear) — a clear that resets the book also clears the module's open-cross variable (recompute the BBO after every folded event, including a clear).

**Ask First:**
- (resolved at CHECKPOINT 1, 2026-08-31) the trade-price check is the off-book-print check (`trades_off_book` when a trade prints > `_TRADE_BBO_TOLERANCE_TICKS` outside the book BBO; flags the report); `session_low`/`session_high`/`n_trades` are also reported. `preflight_integrity` is **diagnostic only** — it returns a report with `verdict: "OK"|"FLAGGED"` and never raises on a data problem; whether a `FLAGGED` report blocks the gate is the parity-gate CLI slice's CHECKPOINT.

**Never:**
- Raising on a data problem (the whole point is to survey a broken file — `preflight_integrity` returns a report, always, unless `source` itself is un-iterable).
- Excluding a degraded day, a halt window, or any event (§1 / AD-13 — record, never drop).
- Importing `sim` / `part_a` / `part_b` / `gate` / `report` / `databento`, running `simulate`, or front-month filtering.
- `datetime.now` / `time.time` / network / a second source pass.
- Re-implementing `book.apply_event`'s own structural checks — this module *drives* it and counts what it raises.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| clean RTH window | monotonic ts, all 5 actions, only transient crosses | `verdict == "OK"`, `missing_actions == ()`, `persistent_cross_count == 0` | N/A |
| single ts regression | one `ev.ts_event < max_ts_seen` | `ts_regressions == 1`, example recorded, `verdict == "FLAGGED"` | N/A |
| cascade of shuffled ts | `100, 200, 50, 60` | `ts_regressions == 2` (50<200, 60<200), `book_inconsistencies == 0` — the apply_event raises for the shuffled events are not miscounted | caught, not counted |
| transient cross | `bid >= ask` for 20 ms then resolves | `transient_cross_count == 1`, `persistent_cross_count == 0`, still `OK` | N/A |
| persistent cross, multi-event | `bid >= ask` across ~10 events over 300 ms | `persistent_cross_count == 1`, `persistent_crosses[0][2] >= 300_000_000`, `book_inconsistencies == 0`, `flags == ("persistent cross",)` | the ≥50 ms `apply_event` raises are the cross — not counted |
| cross open at stream end | crossed on the last event | `unresolved_cross_at_end == True`, `verdict == "FLAGGED"`, **not** in transient/persistent count | N/A |
| missing action class | no `T` events (from the primary instrument) | `missing_actions == ("T",)`, `verdict == "FLAGGED"` | N/A |
| over-cancel mid-window | `C`/`M` for an order the book has (not unknown-ref), past warm-up | `book_inconsistencies >= 1`, `verdict == "FLAGGED"` | caught, counted |
| warm-up unknown ref | unknown-order `C`/`M` within `_WARMUP_NS` of the first event | `warmup_unknown_ref >= 1`, `book_inconsistencies == 0`, `verdict == "OK"` | caught, bucketed |
| second instrument | an event with a different `instrument_id` | `foreign_instrument_events >= 1`, event **skipped** (not folded into the book, not in `actions_seen`), `verdict == "FLAGGED"` | counted, not raised |
| source raises mid-iteration | `_normalize` `ValueError` on event 3 | `malformed_events == 1`, scan continues, `verdict == "FLAGGED"` | caught, counted, **not** re-raised |
| big inter-event gap | a 10-min forward jump between two events | `gaps_over_threshold >= 1`, `largest_gap_ns >= 600e9`, `verdict == "FLAGGED"` | N/A |
| degraded day supplied | `degraded_days=("2026-07-30",)`, otherwise clean | recorded + `format_integrity` note; `verdict == "OK"` | N/A |
| empty source | no events | `n_events == 0`, `first_ts_ns is None`, `bbo_cross_rate == 0.0`, `verdict == "FLAGGED"` (`"no events"`) | N/A |
| `format_integrity` | OK and FLAGGED reports | `integrity: OK` / `integrity: FLAGGED (<reasons>)` (header from `report.flags`), window span, per-counter bullets, `bbo_cross_rate` as %, degraded-day note; ASCII-only, byte-identical across calls | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/integrity.py` — NEW. `preflight_integrity`, `format_integrity`, `IntegrityReport`, module constants `_MAX_EXAMPLES` / `_WARMUP_NS` / `_TRADE_BBO_TOLERANCE_TICKS` / `_MAX_GAP_NS`, `__all__`. `book.check_invariants()` called once at end-of-stream.
- `src/ticksim/parity/_bookwalk.py` — `BookReplay` is *not* reused (this needs per-event inspection, not a seek-to-ts); a fresh `Book` + `book.apply_event` loop, same pattern as `_bookwalk` internally.
- `src/ticksim/book.py` — `Book()`, `apply_event(book, record) -> None` (raises `BookInconsistency`), `book.snapshot_bbo(iid) -> (int|None, int|None)`. `BookInconsistency`.
- `src/ticksim/events.py` — `BookEventSource` Protocol; `BookEvent` (`action: MboAction`, `side`, `price_dbn`, `ts_event`, `instrument_id`); `MboAction` (`ADD="A"`, `CANCEL="C"`, `MODIFY="M"`, `TRADE="T"`, `FILL="F"` — `str(MboAction.TRADE) == "T"`).
- `src/ticksim/config.py` — `MAX_TRANSIENT_CROSS_NS = 50_000_000`, `MNQ_TICK_DBN = 250_000_000`.
- `src/ticksim/parity/gate.py` — `build_amendment_stub(..., integrity: str | None = None, ...)` — the consumer of `format_integrity`'s output (not imported here — decoupled by the `str` type).
- `tests/unit/test_ticksim_imports.py:39` — add `"integrity"` row.
- `_bmad-output/preregistration_tick_data_infrastructure.md` — §1 "Integrity check before use", §A9.3 (the refined check: no persistent cross, must consume A/C/M/T/F), §A8.3 note (window 17 = 2026-07-30 degraded). `…/ARCHITECTURE-SPINE.md` — AD-9 (`book.apply_event` sole MBO authority), AD-20, AD-26 (the stub carries this).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/integrity.py` — `preflight_integrity`, `format_integrity`, `IntegrityReport`, constants, `__all__`.
- [x] `src/ticksim/parity/__init__.py` — allowed-edges docstring for `integrity`.
- [x] `tests/unit/test_ticksim_imports.py` — add the `"integrity"` edge row.
- [x] `tests/unit/test_ticksim_parity_integrity.py` — with hand-built `BookEvent` lists: every I/O-matrix row; `format_integrity` output for an OK and a FLAGGED report (assert the reason list, the counters, ASCII-only, determinism across two calls); `verdict` computed correctly for each flag condition in isolation.
- [x] `tests/integration/test_ticksim_parity_integrity.py` — `@pytest.mark.integration`, skips without the fixture: `preflight_integrity` over the front-month-filtered 2026-06-22 test window → assert `verdict == "OK"` (or a documented known-flag), `missing_actions == ()`, `transient_cross_count` roughly matches Amendment 9's 0.014 % figure, `n_events` ≈ 22.5M (front-month subset).

**Acceptance Criteria:**
- Given a hand-built window with monotonic ts, at least one of each `A/C/M/T/F`, and only sub-50-ms crosses, when `preflight_integrity` runs, then `verdict == "OK"` and `missing_actions == ()`.
- Given a window with a 200-ms crossed market, then `persistent_cross_count == 1`, `persistent_crosses[0][2] >= 200_000_000`, `verdict == "FLAGGED"`, and `"persistent cross"` appears in `flags`.
- Given `format_integrity` on the same report twice, then the two strings are byte-identical and contain no non-ASCII character.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `integrity.py` imports only `{events, book, config}` from `src.ticksim`; the import-graph test passes.

## Spec Change Log

**Review round 1 — 2026-08-31 — patch round (no code re-derivation).** Reviewer trio converged on one real bug + several §1-completeness gaps in the frozen `IntegrityReport` field list.

| # | Was | Now | Why |
|---|---|---|---|
| 1 | any caught `BookInconsistency` not on the ts-regressed event → `book_inconsistencies` | a caught `BookInconsistency` **while a cross is open** (BBO-watch says so) is *not* counted — it is the cross | `book.apply_event` calls `_fail()` on **every** event that lands while a cross has persisted ≥ `MAX_TRANSIENT_CROSS_NS`; a 300 ms lock over 10 events was reported as `persistent_cross_count=1` **and** `book_inconsistencies=9`, double-flagging. |
| 2 | `regressed` computed vs the immediate predecessor `prev_ts` | vs `max_ts_seen` (running max) | `book.apply_event` guards on its own running max; after one regression, later sub-watermark events had `regressed=False` and were miscounted as book inconsistencies. |
| 3 | "catches `BookInconsistency`, never raises" | whole per-event body in `try/except Exception` → `malformed_events`, continue | `DbnMboSource` / `events._normalize` raise plain `ValueError` mid-stream (unknown action code, undefined price) — those propagated, breaking the "never raises" contract. |
| 4 | (missing) §1 "report the exact pass rate" | `bbo_cross_rate: float` (the §A9.3 0.014 % figure) | §1 requires it; the report had raw counts only. |
| 5 | (missing) which window? halt detection? | `first_ts_ns` / `last_ts_ns` / `duration_ns` / `largest_gap_ns` / `gaps_over_threshold` (`> _MAX_GAP_NS` flags) | §1 explicitly cares about halts / missing data; monotonic-ts alone passes a window with a multi-hour hole. |
| 6 | foreign-instrument event still `apply_event`'d + added to `actions_seen` | foreign event counted + **skipped** | another instrument's book damage / crossed market was attributed to the primary window. |
| 7 | `warmup_unknown_ref` absorbed *all* caught inconsistencies in the first 60 s | only unknown-order `C`/`M` refs (via the `book` counter delta) | a dup-`A` / side-flip-`M` / over-cancel 30 s in was silently bucketed benign. |
| 8 | cross open at EOF force-classified by `duration` (0 → transient) | `unresolved_cross_at_end: bool` — its own flag | a file ending mid-cross always passed. |
| 9 | `format_integrity` header could render `FLAGGED ()` | header derived from `report.flags` (FLAGGED ⇒ ≥1 flag) | internal-consistency; a test-built inconsistent report can't produce a contradictory block. |
| 10 | (missing) | `check_invariants_failed: bool` (`book.check_invariants()` once at EOF) | catches level-size disagreement the event pass misses (Code Map already cited it as "cheap enough"). |

Also: `str(d)` for each `degraded_day` in `format_integrity` (tolerate a non-`str`); trade with `UNDEF_PRICE` / `<= 0` skipped before the session-range update; `session_low/high` labelled `(informational)`. New tests per the amended matrix (multi-event persistent cross → `book_inconsistencies == 0`; ts cascade; mid-iteration source `ValueError`; big gap; unresolved-cross-at-end; foreign-event-skipped; `format_integrity` window span + rate + FLAGGED example rendering).

## Design Notes

**Why not `BookReplay`.** `_bookwalk.BookReplay.advance_to(ts)` folds *forward to a cutoff* and is built for point queries. The integrity scan needs to inspect the BBO **after every single event** (to catch a cross that opens and closes between two candidate timestamps), so it runs its own `for ev in source: apply_event(...)` loop. It shares the "catch `BookInconsistency`, keep going" and single-instrument discipline but not the class.

**The cross state machine.** One open-cross variable (`_cross_open_ts: int | None`). Set it the first event the book goes crossed; on the first event it un-crosses (or a side goes `None`), classify `resolve_ts − start_ts` and clear. This correctly handles a cross that deepens (stays open) and back-to-back crosses (each classified separately).

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_integrity.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity tests/unit/test_ticksim_parity_integrity.py` — expected: clean.

## Suggested Review Order

**The pass**

- `preflight_integrity`: `iter(source)` is the only propagating call; every per-event body is double-guarded (`next()` + body) -> `malformed_events`, continue. Foreign-instrument event counted + skipped.
  [`integrity.py:136`](../../src/ticksim/parity/integrity.py#L136)

- BookInconsistency bucketing (loopback-1 #1/#2/#7): `unseen_delta` -> warmup vs book_inconsistencies by ts; `overcancel_delta` -> always book_inconsistencies; the catch-all fallback fires only when `not regressed and not cross_was_open` (the persistent-cross `_fail()` storm and the regression raise are already flagged once).
  [`integrity.py:234`](../../src/ticksim/parity/integrity.py#L234)

- ts-regression vs `max_ts_seen` (running max, not predecessor); inter-event gap -> `largest_gap_ns` / `gaps_over_threshold`.
  [`integrity.py:215`](../../src/ticksim/parity/integrity.py#L215)

- cross state machine: `duration < 0 or > MAX_TRANSIENT_CROSS_NS` -> persistent; open at EOF -> `unresolved_cross_at_end`.
  [`integrity.py:270`](../../src/ticksim/parity/integrity.py#L270)

**Report + format**

- `IntegrityReport` frozen field list + the `verdict`/`flags` derivation; `book.check_invariants()` once at EOF.
  [`integrity.py:92`](../../src/ticksim/parity/integrity.py#L92)

- `format_integrity`: header from `report.flags`, window span, `bbo_cross_rate` as %, `session_low/high (informational)`, `str(d)` degraded days, ASCII-only + deterministic.
  [`integrity.py:382`](../../src/ticksim/parity/integrity.py#L382)

**Peripherals**

- Import edge `integrity -> {events, book, config}`; 25 unit tests covering every amended matrix row.
  [`test_ticksim_parity_integrity.py:1`](../../tests/unit/test_ticksim_parity_integrity.py#L1)
