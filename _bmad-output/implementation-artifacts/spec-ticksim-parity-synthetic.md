---
title: 'ticksim parity/synthetic.py — the Part B synthetic-order generator'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 1
baseline_commit: 'da7472b'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Prereg §A8.2 Part B needs "**≥ 1000 synthetic orders** (mix of marketable and passive limit, both sides, sizes 1–5) at random timestamps across the Tranche 1 data." `run_part_b` (already built) takes a pre-made `OrderIntent` list; nothing produces one, and a hand-written 1000-line JSONL is not reproducible.

**Approach:** add `src/ticksim/parity/synthetic.py` — `generate_synthetic_orders(source, lo_ns, hi_ns, *, n=PART_B_MIN_ORDERS, seed=0) -> list[OrderIntent]`: deterministically (from `random.Random(seed)` — the sole entropy, AD-11) draw `n` standalone `SUBMIT` orders with random `(ts, side, kind, size)` in the window, and for the limit kinds pick a realistic `limit_px_dbn` from the book's BBO at the order's timestamp via **one** `parity._bookwalk.BookReplay` forward pass. The result is submit-ts-sorted and feeds straight into `run_part_b`.

## Boundaries & Constraints

**Always:**
- **`generate_synthetic_orders(source: BookEventSource, lo_ns: int, hi_ns: int, *, n: int = PART_B_MIN_ORDERS, seed: int = 0) -> list[OrderIntent]`.** `source` is a **single-instrument re-iterable** L3 stream (front-month filtering is the caller's job — the one `BookReplay` pass fails closed on a multi-instrument stream). `0 <= lo_ns < hi_ns` and `n >= 1` → else `SyntheticError`. Determinism (AD-11): the same `(source events, lo_ns, hi_ns, n, seed)` returns a **byte-identical** list — `random.Random(seed)` is the only randomness; no wall-clock, no set iteration order dependence.
- **Draw.** Generate candidate tuples with the `rng` in a fixed field order per candidate — `ts = rng.randrange(lo_ns, hi_ns)`, `side = rng.choice([BUY, SELL])`, `kind = rng.choices([MARKETABLE, MARKETABLE_LIMIT, PASSIVE_LIMIT], weights=_KIND_WEIGHTS)[0]`, `size = rng.randint(1, 5)`, and (only for a limit kind) `offset_ticks = rng.randint(0, _MAX_OFFSET_TICKS)`. Over-generate to `ceil(n * _OVERGEN_FACTOR)` candidates so limit orders that can't be priced (see below) can be dropped and still leave `n`.
- **Sort + one BBO pass.** Sort candidates by `ts` (stable — ties keep draw order). Walk `source` **once** through `BookReplay`, `replay.advance_to(ts)` in `ts` order; at each candidate read `bid, ask = replay.book.snapshot_bbo(replay.instrument_id)`.
- **Price the limit.** For `MARKETABLE_LIMIT`: a `BUY` limit is `ask + offset_ticks * MNQ_TICK_DBN`, a `SELL` limit is `bid − offset_ticks * MNQ_TICK_DBN`. For `PASSIVE_LIMIT`: a `BUY` limit is `bid − offset_ticks * MNQ_TICK_DBN`, a `SELL` limit is `ask + offset_ticks * MNQ_TICK_DBN`. A `MARKETABLE` order carries `limit_px_dbn=None`. The candidate is **dropped** (not repriced) when: the needed touch side is `None`; the book is crossed at that ts (`bid >= ask`, both non-`None` — a transient CME cross); or the computed limit `<= 0`. A `MARKETABLE` candidate is always keepable. An `OrderKind` this function doesn't handle → `SyntheticError`. *(The BBO is read at `submit_ts_ns`, not `arrival_ts_ns = submit + latency`; by fill time the market has moved, so the marketable-vs-passive labelling is nominal — `run_part_b`'s invariant checks compare against the actual arrival state regardless. Noted in Design Notes.)*
- **Select + emit.** Price **all** `ceil(n * _OVERGEN_FACTOR)` candidates in the one pass, collecting the priceable ones (still `ts`-sorted). If fewer than `n` are priceable → `SyntheticError` naming the counts (marketable / limit-kept / dropped) — the window is too sparse or `_OVERGEN_FACTOR` too low. Otherwise pick **`n` evenly-spaced indices** across the priceable list — `priceable[round(k * (len - 1) / (n - 1))]` for `k in range(n)` (`n == 1` → index 0) — so the emitted orders **span the whole `[lo_ns, hi_ns)` window**, not just its earliest fraction. Each emitted intent: `action=SUBMIT`, `order_id=f"{_ID_PREFIX}{i:0{width}d}"` (`i` = emit index 0..n−1, zero-padded to `n`'s width), `trade_id = order_id`, `leg=Leg.ENTRY`, `kind`, `side`, `size`, `limit_px_dbn`, `submit_ts_ns=ts`, `oco_group_id=None`, `replaces_order_id=None`. Return sorted by `submit_ts_ns` (the even-index pick over a sorted list already is; a stable re-sort is the cheap guarantee).
- **`_KINDS` (tuple), `_KIND_WEIGHTS` (tuple, `(1, 1, 1)`), `_MAX_OFFSET_TICKS`, `_OVERGEN_FACTOR` (`2.0`), `_ID_PREFIX`** are module constants, each commented as a **non-seal-bound** tuning knob (the seal fixes only "mix … both sides, sizes 1–5" and `n ≥ 1000`). `_OVERGEN_FACTOR > 1.0` is asserted at import. Emit a `logging.getLogger(__name__).debug` line with the marketable / limit-kept / dropped breakdown so `_OVERGEN_FACTOR` can be calibrated against a real window.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. `PERMITTED_INTERNAL_EDGES["synthetic"] = {"orders", "config", "_bookwalk", "events"}`; a test asserts the module's source contains no `databento` reference.

**Ask First:**
- (resolved at CHECKPOINT 1, 2026-08-31) over-generate + drop for un-priceable limits; all three `OrderKind`s at ~equal `_KIND_WEIGHTS`; BBO + `offset_ticks * MNQ_TICK_DBN` stays on the tick grid — no snapping. *(Round-1 review amended the keep-strategy from "first n in ts order" to "price all, subsample n evenly" — see the Spec Change Log.)*

**Never:**
- Any randomness other than `random.Random(seed)` — no `secrets`, no `os.urandom`, no `time`-seeded default `random` (AD-11).
- Importing `sim` / `part_b` / `part_a` / `report` / `databento`, or calling `simulate` — the generator produces intents and nothing else.
- Emitting a `replace` / `cancel` intent, an OCO group, or a multi-leg trade — Part B's battery is standalone `SUBMIT`s (`run_part_b` rejects anything else).
- Front-month instrument filtering, `.dbn.zst` path resolution, any network.
- Producing fewer than `n` orders silently (→ `SyntheticError`).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| happy path | a dense window source, `n=1000`, `seed=0` | exactly 1000 `OrderIntent`s, submit-ts-sorted, all `SUBMIT`/`ENTRY`, kinds mixed, sides mixed, sizes 1–5, limit kinds carry `limit_px_dbn` | N/A |
| determinism | same `(source, lo, hi, n, seed)` called twice | two byte-identical lists (compare `model_dump_json()` line by line) | N/A |
| different seed | `seed=1` vs `seed=0` | different list | N/A |
| `MARKETABLE_LIMIT` BUY price | offset 2, ask `A` | `limit_px_dbn == A + 2 * MNQ_TICK_DBN` | N/A |
| `PASSIVE_LIMIT` SELL price | offset 1, ask `A` | `limit_px_dbn == A + 1 * MNQ_TICK_DBN` | N/A |
| thin book at a ts | `snapshot_bbo` `(None, ask)` for a `SELL` passive (needs ask → ok) / `(bid, None)` for a `BUY` marketable-limit (needs ask → drop) | the un-priceable candidate is dropped; a `MARKETABLE` at the same ts is kept | N/A |
| crossed book at a ts | `bid >= ask` (both non-`None`) | limit candidate dropped (transient CME cross — not a valid price) | N/A |
| computed limit `<= 0` | pathological low bid + large offset | candidate dropped (would fail `OrderIntent`'s `gt=0` validator) | N/A |
| window too sparse / thin-book | `< n` of `ceil(n·2.0)` candidates priceable | — | `SyntheticError` (counts: marketable / limit-kept / dropped) |
| emitted-order span | dense window, low droppage | the `n` emitted `submit_ts_ns` span the whole `[lo_ns, hi_ns)` (evenly subsampled), not the earliest fraction | N/A |
| bad bounds | `lo_ns >= hi_ns` or `lo_ns < 0` or `n < 1` or `hi_ns - lo_ns < n` | — | `SyntheticError` |
| multi-instrument / mis-ordered source | unfiltered parent stream | — | `BookWalkError` (from `BookReplay`) |
| empty source | no events | every limit candidate drops; ~1/3 of `ceil(n·2)` are `MARKETABLE` ≈ `0.67n < n` → `SyntheticError` (a window that can't price `n` orders is a data problem, not something to paper over with an all-marketable battery) | `SyntheticError` |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/synthetic.py` — NEW. `generate_synthetic_orders(...)`, `SyntheticError`, `_KIND_WEIGHTS` / `_MAX_OFFSET_TICKS` / `_OVERGEN_FACTOR` / `_ID_PREFIX` constants, `__all__`. A private `_Candidate` NamedTuple/dataclass for the draw→resolve pipeline.
- `src/ticksim/parity/_bookwalk.py` — `BookReplay(source)`: `.advance_to(ts_ns)` (non-decreasing), `.book` (`Book`), `.instrument_id` (`int | None`); `BookWalkError` on a multi-instrument / mis-ordered stream. `Book.snapshot_bbo(iid) -> (int | None, int | None)`.
- `src/ticksim/orders.py:124` — `OrderIntent` (pydantic: `action`, `order_id`, `trade_id`, `leg`, `kind`, `size`, `limit_px_dbn`, `submit_ts_ns`, `oco_group_id`, `replaces_order_id`); validator: `kind != MARKETABLE ⟹ limit_px_dbn is not None`. `IntentAction.SUBMIT`, `Leg.ENTRY`, `OrderKind.{MARKETABLE,MARKETABLE_LIMIT,PASSIVE_LIMIT}`, `Side.{BUY,SELL}`.
- `src/ticksim/config.py` — `PART_B_MIN_ORDERS = 1000`, `MNQ_TICK_DBN = 250_000_000`.
- `src/ticksim/events.py` — `BookEventSource` Protocol (re-iterable).
- `src/ticksim/parity/part_b.py` — `run_part_b(intents, source, …)` is the consumer (not imported here — decoupled by the `list[OrderIntent]` type).
- `tests/unit/test_ticksim_imports.py:39` — add `"synthetic"` row.
- `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-11 (`random.Random(seed)` sole entropy, determinism), AD-16 / AD-27 (Part B, `PART_B_MIN_ORDERS`), §A8.2 (the "≥1000 synthetic … mix … sizes 1–5" wording).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/synthetic.py` — `generate_synthetic_orders`, `SyntheticError`, the constants, `__all__`.
- [x] `src/ticksim/parity/__init__.py` — allowed-edges docstring for `synthetic`.
- [x] `tests/unit/test_ticksim_imports.py` — add the `"synthetic"` edge row.
- [x] `tests/unit/test_ticksim_parity_synthetic.py` (32 tests): — with a hand-built in-memory `BookEventSource` (a deep static two-sided book): a 1000-order run (assert count, all `SUBMIT`/`ENTRY`, submit-ts non-decreasing, both sides present, all sizes 1–5 present, all three kinds present, limit kinds have `limit_px_dbn` and marketables have `None`, every intent round-trips `model_validate_json`); byte-identical determinism across two calls; a different `seed` → a different list; the four limit-price formulas (marketable-limit BUY/SELL, passive-limit BUY/SELL at a known BBO); an un-priceable candidate dropped when the book has one side only; `SyntheticError` on bad bounds and on a too-sparse window (a source that empties partway → not enough limit candidates *and* cap `n` high enough that even all-marketable can't cover — or assert the all-marketable fallback); multi-instrument source → `BookWalkError`; feed the output straight into `run_part_b` (small `n`, monkeypatched `PART_B_MIN_ORDERS`) and assert `verdict == "PASS"`.
- [x] `tests/integration/test_ticksim_parity_synthetic.py` — `@pytest.mark.integration`, skips without the fixture: `generate_synthetic_orders` over the 2026-06-22 test window (front-month filtered), `n=1000`, then `run_part_b(orders, <same window>)` → `verdict == "PASS"`, `violations == ()` (the real end-to-end Part B smoke test).

**Acceptance Criteria:**
- Given a dense two-sided book source, `n=1000`, `seed=0`, when `generate_synthetic_orders` runs, then it returns 1000 `OrderIntent`s, submit-ts non-decreasing, with `>= 1` order of each `OrderKind`, both `Side`s, and every size in `1..5`.
- Given the same arguments twice, then `[o.model_dump_json() for o in a] == [... for b]`.
- Given a `MARKETABLE_LIMIT` `BUY` candidate with `offset_ticks = 3` at a tick where the ask is `A`, then its `limit_px_dbn == A + 3 * MNQ_TICK_DBN`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `synthetic.py` imports only `{orders, config, _bookwalk, events}` from `src.ticksim`; the import-graph test passes.

## Spec Change Log

**Review round 1 — 2026-08-31 — patch round (no code re-derivation).** Reviewer trio. Two frozen reconciliations (internal inconsistencies — the frozen block contradicted itself):

| # | Was | Now | Why |
|---|---|---|---|
| 1 | **Emit:** "keep candidates in the sorted order until `n` valid" | **Select + emit:** price *all* `ceil(n·_OVERGEN_FACTOR)` candidates, then pick `n` **evenly-spaced indices** across the priceable list | Keeping the first `n` priceable candidates by `ts` returns only the earliest ~1/`_OVERGEN_FACTOR` of the window when droppage is low — contradicting the frozen Intent's "at random timestamps **across** the Tranche 1 data". Even-index subsampling over the ts-sorted priceable list spans the whole window and stays deterministic. |
| 2 | matrix: "empty source → all `n` marketable" | matrix: empty / too-sparse source → `SyntheticError` | Unreachable: with `_KIND_WEIGHTS = (1,1,1)` only ~1/3 of candidates are `MARKETABLE` ≈ `0.67n < n`, so an empty book raises. A window that can't price `n` orders is a data problem to surface, not to paper over with an all-marketable battery (which wouldn't exercise the passive / marketable-limit fill paths at scale — Part B's whole point). |

**Patches:** `_price_limit` drops a candidate on a crossed book (`bid >= ask`) or a computed limit `<= 0` (avoids `OrderIntent`'s `gt=0` `ValidationError`); an unhandled `OrderKind` → `SyntheticError`; `_OVERGEN_FACTOR` bumped `1.5 → 2.0` and asserted `> 1.0` at import; `_KINDS` is a `tuple`; a `logger.debug` breakdown (marketable / limit-kept / dropped); `hi_ns - lo_ns < n` → `SyntheticError`; docstring `Raises:` gains `BookInconsistency` (pass-through from `apply_event`); the "draws `n` orders" summary corrected to "draws `ceil(n·2)` candidates, emits `n`". New tests: a partial-two-sided-book run at `n` a few hundred still returns `n` (droppage near the overgen limit); book deltas interleaved among candidate timestamps (BBO tracked forward); emitted `submit_ts_ns` span the full window; crossed-book drop; a golden snapshot of the first ~5 emitted intents for `seed=0` (pins the rng sequence so a conditional-offset-draw refactor is caught); a stronger different-seed assertion (high fraction of differing rows); an `n=3000` run; a source-scan test for no `databento` import.

## Design Notes

**Why over-generate rather than re-draw.** `BookReplay.advance_to` is monotonic — it cannot seek back to price a re-drawn earlier `ts`. Over-generating `ceil(n·_OVERGEN_FACTOR)` candidates and dropping the un-priceable ones keeps everything to one sorted pass and one `rng` sequence, so determinism is trivial to reason about. `_OVERGEN_FACTOR = 2.0` absorbs up to ~50 % of limit candidates dropping (≈1/3 of all candidates) before `SyntheticError`; bump it if a real Tranche-1 window's warm-up gaps prove worse (the `logger.debug` breakdown tells you the drop rate).

**Nominal vs fill-time kind.** The BBO is read at `submit_ts_ns`; the sim fills at `arrival_ts_ns = submit_ts_ns + latency_ns` (250 ms later, `PRIMARY`), by which point a `MARKETABLE_LIMIT` at `offset=0` may be resting and a `PASSIVE_LIMIT` may be marketable. This is spec-compliant ("the order's timestamp") and harmless for Part B: `run_part_b`'s `check_order` evaluates every invariant against the *actual* arrival state, not the generator's label. Advancing the pass to `submit_ts_ns + latency_ns` would make the labels literal but couple the generator to a latency value — not worth it.

**Determinism discipline.** Every `rng` call happens in a fixed order: all `ceil(n·2)` candidates are drawn up front — each consuming exactly `ts`, `side`, `kind`, `size`, `offset` (the `offset` draw happens even for a `MARKETABLE`, where it is discarded, so a `kind` outcome never desyncs the stream) — *then* the book pass resolves them, *then* the even-index subsample. The emit-index `i` (not the draw or priceable index) names the order, so a dropped candidate leaves no gap in `order_id`s.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_synthetic.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity tests/unit/test_ticksim_parity_synthetic.py` — expected: clean.

## Suggested Review Order

**The pipeline**

- `generate_synthetic_orders`: bounds checks -> draw all `ceil(n*2)` candidates (fixed rng call order) -> ts-sort -> one `BookReplay` pass pricing every candidate -> `SyntheticError` if `<n` priceable -> even-index subsample of `n` -> emit.
  [`synthetic.py:119`](../../src/ticksim/parity/synthetic.py#L119)

- `_evenly_spaced_indices(length, n)`: `round(k*(length-1)/(n-1))` -> spans the whole priceable list; the frozen-reconciliation-#1 fix for window truncation.
  [`synthetic.py:255`](../../src/ticksim/parity/synthetic.py#L255)

- `_price_limit`: the four limit formulas + the three drop conditions (`None` touch side / crossed book / limit `<=0`); unknown `OrderKind` -> `SyntheticError`.
  [`synthetic.py:268`](../../src/ticksim/parity/synthetic.py#L268)

**Constants + determinism**

- `_KINDS`/`_KIND_WEIGHTS` (tuples), `_OVERGEN_FACTOR = 2.0` (asserted `>1.0`), `_MAX_OFFSET_TICKS`; the `logger.debug` drop-rate breakdown.
  [`synthetic.py:64`](../../src/ticksim/parity/synthetic.py#L64)

**Peripherals**

- Import edge `synthetic -> {orders, config, _bookwalk, events}`; the no-`databento` AST guard; the `run_part_b` round-trip test.
  [`test_ticksim_parity_synthetic.py:1`](../../tests/unit/test_ticksim_parity_synthetic.py#L1)
