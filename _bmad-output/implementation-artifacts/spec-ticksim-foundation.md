---
title: 'ticksim foundation — deps + config + frozen schemas'
type: 'feature'
created: '2026-08-29'
status: 'done'
review_loop_iteration: 0
baseline_commit: 'f4642d21393882d739cd7d8004400d0e6275defd'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/project-context.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The TICK-INFRA fill simulator (`src/ticksim/`, architecture spine, 28 ADs) has no code yet. Every later module binds to a contract layer that does not exist: the frozen `SimConfig` + seal-bound presets/constants, and the frozen Pydantic schemas that are the simulator↔consumer interface. `databento` is only pip-installed in `.venv` (not in `pyproject.toml`), so `poetry install` would remove it.

**Approach:** Ship the pure contract layer only — no behaviour. Add the two runtime deps; create `src/ticksim/` with `config.py` (frozen `SimConfig`, `PRIMARY`/`OPTIMISTIC` presets, named seal-cited constants) and `orders.py` schemas (`OrderIntent`, `FillEvent`, `Fill`, `OrderOutcome`) as frozen Pydantic v2 models. Full unit tests + the import-graph guard test.

## Boundaries & Constraints

**Always:**
- Every model is Pydantic v2 with `model_config = ConfigDict(frozen=True)`; every numeric field is `int` (ns time, DBN 1e-9 fixed-point price, sizes, counters) — spine AD-10, AD-12, AD-19, AD-23.
- Every `PRIMARY`/`OPTIMISTIC` field and every named constant carries a comment citing its prereg section (§2.1 / §2.3 / Amendment 8) — AD-15, AD-27.
- Preset values match `_bmad-output/preregistration_tick_data_infrastructure.md` §2.1 exactly: PRIMARY = back-of-queue, 250 ms latency, `$0.72` exch+reg + `$0.58` commission, no own-impact; OPTIMISTIC = time-priority, 50 ms.
- `mypy --strict` clean with **no** `[[tool.mypy.overrides]]` entry for `src.ticksim`; `black` (line 88) clean.
- Tests: `tests/unit/test_ticksim_<module>.py`, classes `Test*`, functions `test_*`.

**Ask First:**
- Any field name, type, or enum value that differs from the spine's AD-12 / AD-19 / AD-23 wording.
- Whether `latency_ns` applies the seal's "250 ms round trip" as the full submit→arrival delay (this spec assumes yes — the conservative reading).

**Never:**
- Any behaviour: no `OrderBook`, no `OrderTracker`, no `apply_event`, no fill logic, no queue-model classes. `orders.py` gets **schemas only** — `OrderTracker` and OCO groups are a later spec.
- Any import from `src.data`, `src.research`, `src.detection`, `src.execution`, `src.ml`, `src.risk`, `src.monitoring`, `src.dashboard` (AD-4).
- Any `float` field on any of the four schemas.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Frozen config | `PRIMARY.model_copy(update=...)` or attribute set | mutation rejected | `ValidationError` / `TypeError` |
| Float into int field | `Fill(px_dbn=1.5, ...)` | rejected — no float coercion | `ValidationError` (strict int) |
| Replace intent well-formed | `OrderIntent(action="replace", order_id="x", replaces_order_id="x", ...)` | valid | N/A |
| Replace intent missing target | `OrderIntent(action="replace", replaces_order_id=None, ...)` | rejected | `ValidationError` |
| Cancel intent with price/size | `OrderIntent(action="cancel", limit_px_dbn=..., ...)` | valid but price/size ignorable — documented, not enforced here | N/A |
| Preset drift | any `PRIMARY` field ≠ §2.1 | test fails | assertion in `test_ticksim_config.py` |
| OrderOutcome numeric audit | serialize any `OrderOutcome` to JSON | every leaf numeric is `int` | assertion in `test_ticksim_orders.py` |

</frozen-after-approval>

## Code Map

- `pyproject.toml` -- `[tool.poetry.dependencies]` gets `databento` + `sortedcontainers`; `[[tool.mypy.overrides]]` block exists for legacy modules — do **not** add `src.ticksim`.
- `src/data/models.py` -- style reference only: `BaseModel`, `Field(..., description=...)`, `@field_validator` + `@classmethod`. Read-only.
- `_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md` -- the binding contract. AD-12 (OrderOutcome fields), AD-19 (FillEvent), AD-23 (OrderIntent fields + replace convention), AD-15/24/27 (constants), AD-4/7 (isolation + import graph), AD-10 (int-only).
- `_bmad-output/preregistration_tick_data_infrastructure.md` §2.1 table -- authoritative preset values.
- `src/ticksim/` -- new package (does not exist).

## Tasks & Acceptance

**Execution:**
- [x] `pyproject.toml` -- add `databento = "^0.85.0"` and `sortedcontainers = "^2.4.0"` under `[tool.poetry.dependencies]`; run `poetry lock --no-update` then `poetry install` -- makes `databento` a declared dep (Amendment 9 §A9.5) and adds the O(log n) book structure dep (AD-4).
- [x] `src/ticksim/__init__.py` -- empty package marker.
- [x] `src/ticksim/config.py` -- `QueueModel` enum (`BACK_OF_QUEUE`, `TIME_PRIORITY`); frozen `SimConfig` (`queue_model: QueueModel`, `latency_ns: int`, `exch_reg_fee_dbn`/`commission_dbn` or `_usd_cents: int`, `seed: int`, `own_impact: bool = False`); `PRIMARY` and `OPTIMISTIC` module constants; named constants `DOLLARS_PER_INDEX_POINT = 2`, `MNQ_TICK_DBN = 250_000_000`, `PARITY_MAE_MAX_TICKS`, `PARITY_P90_MAX_TICKS`, `PARITY_SIGNED_BIAS_MAX_TICKS`, `PART_A_MIN_N`, `PART_B_MIN_ORDERS`, `MAX_TRANSIENT_CROSS_NS` (values per AD-27), each seal-cited -- AD-15/24/27.
- [x] `src/ticksim/orders.py` -- frozen models `OrderIntent` (AD-23 fields incl. `schema_version`, `action` enum, `leg` enum, `kind` enum, `replaces_order_id`, `oco_group_id`; validator: `action == replace` ⇒ `replaces_order_id` set), `FillEvent` (AD-19: `order_id, px_dbn, size, ts_ns` — nothing else), `Fill` (`px_dbn, size, ts_ns`), `OrderOutcome` (AD-12 fields incl. `schema_version`, `trade_id`, `leg`, `kind`, `terminal_state` enum, `fills: list[Fill]`, queue + arrival + `adverse_selection` fields). Enums for `Side`, `OrderKind`, `IntentAction`, `Leg`, `TerminalState`. **No `OrderTracker`.**
- [x] `tests/unit/test_ticksim_config.py` -- assert `PRIMARY`/`OPTIMISTIC` field-by-field vs §2.1; assert every constant's value vs AD-27; assert `SimConfig` frozen (mutation raises).
- [x] `tests/unit/test_ticksim_orders.py` -- the I/O matrix rows for the four schemas; the "every OrderOutcome numeric is int" JSON audit (AD-10); replace-validator both directions.
- [x] `tests/unit/test_ticksim_imports.py` -- walk `src/ticksim/*.py` ASTs; assert no `import`/`from` targets `src.data|research|detection|execution|ml|risk|monitoring|dashboard`; assert `config.py` and `orders.py` import nothing from `src.ticksim` (AD-4, AD-7). Designed to grow as modules land.

**Acceptance Criteria:**
- Given a fresh `poetry install`, when `python -c "import databento, sortedcontainers"` runs, then it succeeds.
- Given `mypy --strict src/ticksim`, when run, then zero errors and no override was added.
- Given the spine's AD-12/19/23 field lists, when `orders.py` is diffed against them, then every field is present with the AD's type and no extra behavioural field.
- Given `import src.ticksim.config` and `import src.ticksim.orders`, when the import-graph test runs, then neither pulls in any other `src.*` package.

## Design Notes

`SimConfig.queue_model` is an **enum**, not a strategy object — `config.py` must stay a leaf (AD-7); `fills.py` later maps `QueueModel → BackOfQueueModel()/TimePriorityModel()`. Money is stored as `int` (USD cents or DBN units) per AD-10; `report.py` is the only consumer and converts (AD-24) — pick cents, document it. `latency_ns` for `PRIMARY` = `250_000_000` (the seal's "250 ms round trip" applied as the full submit→arrival delay — conservative; flagged Ask-First). `schema_version` starts at `1` on `OrderIntent` and `OrderOutcome`.

## Verification

**Commands:**
- `poetry lock --no-update && poetry install` -- expected: resolves, `databento`/`sortedcontainers` installed.
- `.venv/bin/python -m pytest tests/unit/test_ticksim_config.py tests/unit/test_ticksim_orders.py tests/unit/test_ticksim_imports.py -q` -- expected: all pass.
- `poetry run mypy src/ticksim` -- expected: `Success: no issues`.
- `poetry run black --check src/ticksim tests/unit/test_ticksim_*.py` -- expected: clean.

## Suggested Review Order

**Design intent — the contract the whole simulator binds to**

- Enum, not a strategy object, so `config.py` stays a spine leaf (AD-7)
  [`config.py:22`](../../src/ticksim/config.py#L22)
- Frozen run config; every field pinned to prereg §2.1, each seal-cited (AD-15)
  [`config.py:33`](../../src/ticksim/config.py#L33)
- The two seal-bound presets — changing a value is a pre-registration violation
  [`config.py:102`](../../src/ticksim/config.py#L102)

**Seal-bound constants + the one tick→DBN conversion point**

- Parity thresholds / cross tolerance / N floors as named constants (AD-27)
  [`config.py:141`](../../src/ticksim/config.py#L141)
- Adverse-selection window promoted from prose to a named constant (AD-28)
  [`config.py:167`](../../src/ticksim/config.py#L167)
- `ticks_to_dbn` — single place parity code converts a tick tolerance
  [`config.py:175`](../../src/ticksim/config.py#L175)

**The four wire schemas (frozen, `extra="forbid"`, int-only)**

- `OrderIntent` — the JSONL log the simulator consumes (AD-23)
  [`orders.py:83`](../../src/ticksim/orders.py#L83)
- The one cross-field validator: replace-reuses-order_id + limit-needs-price (AD-23)
  [`orders.py:162`](../../src/ticksim/orders.py#L162)
- `OrderOutcome` — the fills contract; consumers join to the intent log (AD-12)
  [`orders.py:245`](../../src/ticksim/orders.py#L245)
- `FillEvent` — this-tick incremental, exactly four fields (AD-19)
  [`orders.py:186`](../../src/ticksim/orders.py#L186)

**Verification — the guards that must survive as modules land**

- `_resolve_imports` — relative + `from src import x` resolution (the AD-4/AD-7 guard)
  [`test_ticksim_imports.py:60`](../../tests/unit/test_ticksim_imports.py#L60)
- Resolver self-tests — the guard's own guard
  [`test_ticksim_imports.py:117`](../../tests/unit/test_ticksim_imports.py#L117)
- Preset / constant drift detectors vs prereg §2.1 + AD-27
  [`test_ticksim_config.py:1`](../../tests/unit/test_ticksim_config.py#L1)

**Peripherals**

- deps added ahead of their consumers (databento: Amendment 9 §A9.5; sortedcontainers: book.py next)
  [`pyproject.toml:29`](../../pyproject.toml#L29)
- `py.typed` — PEP 561 marker so mypy consumers get real types
  [`py.typed`](../../src/ticksim/py.typed)
