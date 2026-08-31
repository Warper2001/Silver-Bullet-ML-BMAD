---
title: 'ticksim cli.py — the `simulate` subcommand + front-month filter'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 0
baseline_commit: 'd7f5daf'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<!-- SPLIT (2026-08-31, planning): `cli.py` has two entry points (AD-6:
     `simulate`, `parity-gate`). THIS slice is `simulate` + the shared
     re-iterable front-month `instrument_id` filter. The `parity-gate`
     subcommand (reconstruct trades -> run_part_a + run_part_b -> gate) is
     deferred to cli.py slice 2 — it needs the §5 integrity preflight (gate
     slice 6) and the synthetic-order generator (slice 7) to run a complete
     gate. -->

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `sim.simulate` is a library function; there is no command-line way to run the simulator over a real `.dbn.zst` window + a JSONL intent log and get an `OrderOutcome` log back. Spine AD-6 names `cli.py` (`simulate`, `parity-gate`) the entry-point module.

**Approach:** add `src/ticksim/cli.py` with an `argparse` sub-command dispatcher and the `simulate` sub-command: read an intent-log JSONL file, open the DBN window, filter it to one instrument, run `sim.simulate` under one config (`PRIMARY` or `OPTIMISTIC`), write the `OrderOutcome` JSONL + the run `Manifest` JSON. Also lands the shared **`FrontMonthSource`** — a re-iterable `BookEventSource` wrapper that yields only one `instrument_id`'s events (the `MNQ.FUT` parent is ~96 % front month, ~4 % spread — Amendment 9). The 3-way P&L `report` sub-command and the `parity-gate` sub-command are later slices.

## Boundaries & Constraints

**Always:**
- **`main(argv: Sequence[str] | None = None) -> int`** — `argparse` with `subparsers` (`dest="command"`). Unknown / missing command → print usage to stderr, return `2`. Each sub-command handler returns an `int` exit code (`0` ok, `1` a handled error with a message on stderr, `2` a usage error). `if __name__ == "__main__": raise SystemExit(main())`.
- **`simulate` sub-command args:** `--dbn PATH` (the `.dbn.zst` window, required), `--intents PATH` (JSONL `OrderIntent` log, required), `--config {primary,optimistic}` (default `primary`), `--instrument-id INT` (optional — the front-month id; omitted → auto-detect, below), `--interval START_NS END_NS` (repeatable; omitted → one interval spanning the intent log ± a 5-min pad, mirroring `run_part_a`), `--out-outcomes PATH` (JSONL, required), `--out-manifest PATH` (JSON, required), `--degraded-day YYYY-MM-DD` (repeatable, recorded in the manifest — never auto-excluded, AD-13).
- **Intent log read.** One `OrderIntent` per non-blank line via `OrderIntent.model_validate_json(line)`; a malformed line → exit `1` naming the line number. Empty file → exit `1` ("no intents"). The list is passed to `simulate` as-is — `sim` validates causal replayability and raises `IntentLogError`, which the handler catches → exit `1` with the message.
- **`FrontMonthSource(inner: BookEventSource, instrument_id: int)`** — a class (re-iterable per AD-18): `__iter__` yields `ev for ev in inner if ev.instrument_id == instrument_id`; `class_rank` forwarded from `inner`. No other filtering, no mutation.
- **Auto-detect the instrument.** When `--instrument-id` is omitted, `detect_front_month(source) -> int` does **one** pass counting `ev.instrument_id`, returns the modal id, and the handler logs `detected front-month instrument_id=<id> (<pct>% of <n> events)` to stderr. An empty stream → exit `1`. A tie → exit `1` (ambiguous — the analyst must pass `--instrument-id`).
- **Run.** `outcomes, manifest = simulate(FrontMonthSource(DbnMboSource(dbn), iid), intents, cfg, valid_intervals, degraded_days=days)` where `cfg` is `config.PRIMARY` / `config.OPTIMISTIC`. Propagated `IntentLogError` / `InvariantViolation` / `BookInconsistency` / `ValueError` → caught, printed to stderr, exit `1` (a simulator fault is an analyst-facing condition here, not a crash). `OrderStateError` is a bug — let it propagate.
- **Write.** `--out-outcomes`: one `outcome.model_dump_json()` per line, in the order `simulate` returned them. `--out-manifest`: `json.dumps(manifest.to_dict(), indent=2, sort_keys=True)`. Parent dirs created if missing. An existing file is **overwritten** (a re-run is idempotent by design — AD-11). On any write error → exit `1`.
- **Summary to stdout** — a short human block: config name, N intents, N outcomes, N fill events, terminal-state counts, `manifest` seed + oco/adverse counts, the two output paths. Machine-readable enough to `grep`, not JSON.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports (`from .sim import …`). `PERMITTED_INTERNAL_EDGES["cli"] = {"sim", "events", "orders", "config", "book"}` (`book` for the `BookInconsistency` type — round-1 review; `report` re-added with the `report` sub-command slice). Stdlib: `argparse`, `json`, `sys`, `pathlib`, `collections`, `logging`, `math`, `re`, `os`, `typing` — no `datetime` / `time` in `src/ticksim` beyond passing a `--degraded-day` token through as a plain `str`.

**Ask First:**
- (resolved at CHECKPOINT 1, 2026-08-31) `simulate` is one run / one config — the 3-way P&L `report` is a separate sub-command in a later slice; `report` is **not** in this slice's import edge. `--interval` omitted → auto-span the intent log's `submit_ts_ns` range ± `--pad-minutes` (default 5); `--interval START_NS END_NS` repeatable overrides.

**Never:**
- The `parity-gate` sub-command, reconstruction, `run_part_a` / `run_part_b` / `gate` calls (slice 2).
- Importing `databento` directly (only `events` may — AD-18) or `parity/*` (slice 2).
- Auto-excluding a degraded day, a halt window, or any bar (AD-13 — record, never drop).
- `datetime.now` / `time.time` / a second `subprocess` / network (AD-1/AD-4/AD-11).
- Writing anywhere but the two `--out-*` paths the analyst gave.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| happy path, explicit iid | valid DBN + intents + `--instrument-id` | outcome JSONL + manifest JSON written; stdout summary; exit `0` | N/A |
| iid auto-detect | `--instrument-id` omitted, one dominant id | detects it, logs `<pct>%`, runs; exit `0` | N/A |
| iid auto-detect tie | two ids equally common | — | exit `1` "ambiguous — pass --instrument-id" |
| empty DBN stream | file has no events | — | exit `1` |
| malformed intent line | line 7 not valid JSON / fails the schema | — | exit `1` naming line 7 |
| empty intent file | 0 non-blank lines | — | exit `1` "no intents" |
| non-causal intent log | `cancel` before `submit` | — | exit `1` (from `IntentLogError`) |
| simulator invariant breach | `sim` raises `InvariantViolation` | — | exit `1` with the message |
| multi-instrument after filter | `FrontMonthSource` still yields 2 ids (impossible — guard) | — | `IntentLogError` from `sim` → exit `1` |
| unknown sub-command | `ticksim frobnicate` | usage to stderr | exit `2` |
| missing required arg | no `--out-manifest` | argparse usage | exit `2` |
| out path in a missing dir | `--out-outcomes /nope/x.jsonl` | parent dir created, file written | exit `0` (or `1` if truly unwritable) |
| `--config optimistic` | valid inputs | runs under `config.OPTIMISTIC`; summary says so | exit `0` |

</frozen-after-approval>

## Code Map

- `src/ticksim/cli.py` — NEW. `main`, `_cmd_simulate`, `FrontMonthSource`, `detect_front_month`, `_read_intents`, `_span_interval`, `__all__`. `argparse` dispatcher.
- `src/ticksim/sim.py:709` — `simulate(book_event_source, intent_log, config, valid_intervals, *, degraded_days=()) -> (list[OrderOutcome], Manifest)`; `IntentLogError`, `InvariantViolation`. `Manifest.to_dict()`.
- `src/ticksim/events.py:140` — `DbnMboSource(path)` (re-iterable, `class_rank = 0`); `BookEventSource` Protocol (`class_rank`, `__iter__`); `BookEvent` (`instrument_id`, `ts_event`, …).
- `src/ticksim/orders.py:124` — `OrderIntent` (pydantic — `model_validate_json`, `submit_ts_ns`); `OrderOutcome` (`model_dump_json`, `terminal_state`, `fills`).
- `src/ticksim/config.py` — `PRIMARY`, `OPTIMISTIC` (`SimConfig`).
- `data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst` — the integration test's DBN window.
- `tests/unit/test_ticksim_imports.py:39` — add `"cli"` row.
- `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-6 (`cli.py` entry points), AD-13 (mask / degraded days), AD-18 (`BookEventSource` re-iterable), AD-11 (idempotent re-run).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/cli.py` — `main` + `_cmd_simulate` + `_run_simulate` + `FrontMonthSource` + `detect_front_month` + `_read_intents` + `_span_interval` + `_atomic_write_pair` + `_validate_simulate_args`, `__all__`. `-v/--verbose`, `--pad-minutes`, output-path + interval + pad validation.
- [x] `src/ticksim/__main__.py` + `pyproject.toml [tool.poetry.scripts] ticksim` — `python -m src.ticksim` / `ticksim` entry point.
- [x] `tests/unit/test_ticksim_imports.py` — `"cli": {"sim", "events", "orders", "config", "book"}` + `"__main__": {"cli"}` rows.
- [x] `tests/unit/test_ticksim_cli.py` — 74 tests: — `FrontMonthSource` re-iterability + filtering; `detect_front_month` modal / empty / tie; `_read_intents` happy / malformed-line / empty; `_span_interval` pad math; `main(["simulate", …])` end-to-end with a hand-built in-memory monkeypatched `DbnMboSource` (list of `BookEvent`) writing to `tmp_path` — assert the outcome JSONL round-trips via `OrderOutcome.model_validate_json`, the manifest JSON parses, exit `0`; every I/O-matrix error row → the stated exit code + a stderr message.
- [x] `tests/integration/test_ticksim_cli.py` — `@pytest.mark.integration`, skips without the fixture: `main(["simulate", "--dbn", <test window>, "--intents", <a hand-built 2-order JSONL>, "--instrument-id", <front month>, "--out-outcomes", …, "--out-manifest", …])` → exit `0`, a non-empty outcome log.

**Acceptance Criteria:**
- Given a valid DBN window, a 2-line intent JSONL, and `--instrument-id`, when `main(["simulate", …])` runs, then exit `0`, the `--out-outcomes` file has 2 lines each parsing as an `OrderOutcome`, and the `--out-manifest` file is valid JSON with a `seed` key.
- Given `--instrument-id` omitted and a stream with a 96/4 id split, then `detect_front_month` returns the 96 % id and stderr logs the percentage.
- Given a `cancel`-before-`submit` intent log, then exit `1` and stderr contains the `IntentLogError` message.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `cli.py` imports only its permitted edge set; `python -m src.ticksim.cli simulate --help` exits `0`.

## Spec Change Log

**Review round 1 — 2026-08-31 — patch round (no code re-derivation).** All findings patch. One frozen reconciliation: `PERMITTED_INTERNAL_EDGES["cli"]` gains `"book"` — the round-0 spec's over-broad `except Exception` around `simulate()` (needed because `book.BookInconsistency` wasn't importable) was masking real cli↔sim wiring bugs; importing `book` to catch `BookInconsistency` **explicitly** and letting `TypeError` / `AttributeError` / `KeyError` propagate (like `OrderStateError` already does) is the correct fix. Stdlib list widened (`logging`, `math`, `re`, `os`). ~18 patches: `--pad-minutes` validated finite/`>= 0` and the auto-span end is `+ pad_ns + 1` ns (a boundary intent must be strictly inside the half-open mask — `pad 0` was silently rejected); explicit `--interval` bounds + negative `--instrument-id` → `parser.error` (exit 2); atomic write (both files to `.tmp` → `os.replace` after both succeed); output paths must differ from each other and from `--dbn` / `--intents`; `encoding="utf-8"` on every read/write; `UnicodeDecodeError` on `--intents` → exit 1; `-v/--verbose` → `logging.basicConfig(INFO, stream=stderr)`, default `WARNING`; a non-`YYYY-MM-DD` `--degraded-day` → stderr warning (still recorded); `detect_front_month` / `FrontMonthSource` reject a one-shot `inner` (`iter(x) is x`); `BrokenPipeError` from the summary → clean exit; `src/ticksim/__main__.py` (+ a `pyproject` `ticksim` script entry) so `python -m src.ticksim` works. New tests: `OrderStateError` propagates (not exit 1), auto-span/`--pad-minutes` effect on `valid_intervals`, summary-line field assertions, multi-instrument-after-filter row, idempotent-manifest fields, output-path-collision, atomic-write partial-failure.

## Design Notes

**`FrontMonthSource` is a class, not a generator.** AD-18 requires a `BookEventSource` be re-iterable (the merge is pull-based; downstream replays sources). A bare `(ev for ev in inner if …)` generator is one-shot and would break `sim`'s internal re-iteration and any `_bookwalk` reuse. The class re-creates the filtered iterator on each `__iter__`.

**Why exit codes, not exceptions, for simulator faults.** `cli.py` is the human boundary. An `IntentLogError` from a hand-written intent log is a user error, not a bug — it deserves a one-line stderr message and exit `1`, not a traceback. `OrderStateError` (an illegal tracker transition) *is* a bug and propagates.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_cli.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim tests/unit/test_ticksim_cli.py` — expected: clean.
- `PYTHONPATH=. .venv/bin/python -m src.ticksim.cli simulate --help` — expected: exit `0`, usage text.

## Suggested Review Order

**Entry point + dispatch**

- `main`: argparse subparsers, `-v/--verbose` → `src.ticksim` package-logger to stderr (deviation from `logging.basicConfig` — safe under repeated in-process calls / pytest capture), exit-code contract.
  [`cli.py:558`](../../src/ticksim/cli.py#L558)

- `_validate_simulate_args`: `--interval` `[start,end)` shape, `--pad-minutes` finite/`>=0`, `--instrument-id >= 0`, output-path distinctness (from each other and `--dbn`/`--intents`) → `parser.error` (exit 2).
  [`cli.py:506`](../../src/ticksim/cli.py#L506)

**The run**

- `_run_simulate`: `_read_intents` → span/interval → iid resolve → `simulate(FrontMonthSource(DbnMboSource(...), iid), ...)`. Explicit `(IntentLogError, InvariantViolation, BookInconsistency, ValueError)` → exit 1; `OrderStateError` + any other exception propagate (bugs).
  [`cli.py:226`](../../src/ticksim/cli.py#L226)

- `_span_interval`: `hi = max(stamp) + pad_ns + 1` — the `+1` ns keeps the last intent strictly inside the half-open mask even at `--pad-minutes 0`.
  [`cli.py:189`](../../src/ticksim/cli.py#L189)

- `_atomic_write_pair`: both `*.tmp` first, then `os.replace` both; rollback + unlink on any failure — never an orphan outcomes file.
  [`cli.py:317`](../../src/ticksim/cli.py#L317)

**Shared filter**

- `FrontMonthSource` (re-iterable, one-shot-source `TypeError` guard) + `detect_front_month` (modal `instrument_id`, tie/empty → error, second-full-pass cost noted in `--help`).
  [`cli.py:90`](../../src/ticksim/cli.py#L90)

**Peripherals**

- `__main__.py` + `pyproject` `ticksim` script; import edges `cli → {sim, events, orders, config, book}`, `__main__ → {cli}`.
  [`__main__.py:1`](../../src/ticksim/__main__.py#L1)

- 74 unit tests (incl. `OrderStateError`/`KeyError` propagate, atomic-write failure, idempotent manifest); integration test (skips w/o DBN fixture).
  [`test_ticksim_cli.py:1`](../../tests/unit/test_ticksim_cli.py#L1)
