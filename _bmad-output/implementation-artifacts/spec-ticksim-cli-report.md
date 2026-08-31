---
title: 'ticksim cli.py — the `report` subcommand (AD-14 three-way P&L)'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 0
baseline_commit: '61d7e24'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `report.build_report` produces the §2.3 / AD-14 three-way P&L (`primary` / `stressed` / `optimistic`) from a **PRIMARY** run's outcomes+manifest and an **OPTIMISTIC** run's outcomes+manifest, but there's no command-line way to run it — the `cli.py simulate` subcommand deliberately writes one run at a time (its CHECKPOINT-1 split).

**Approach:** add a second `cli.py` subcommand, `report`: read the two `OrderOutcome` JSONL logs + the two `Manifest` JSON files (all four produced by prior `simulate --config primary` / `--config optimistic` runs), call `report.build_report`, write `ThreeWayReport.to_dict()` as pretty JSON, print a grep-friendly stdout summary. Re-add `report` to `cli.py`'s import edge.

## Boundaries & Constraints

**Always:**
- **`report` subcommand args:** `--primary-outcomes PATH` (JSONL, required), `--primary-manifest PATH` (JSON, required), `--optimistic-outcomes PATH` (JSONL, required), `--optimistic-manifest PATH` (JSON, required), `--out PATH` (JSON, required). All five paths resolved and required to be **distinct** (any collision → `parser.error`, exit `2`).
- **Read outcomes.** One `OrderOutcome` per non-blank line via `OrderOutcome.model_validate_json(line)` (reuse the `_read_intents` pattern — `encoding="utf-8"`, catch `(OSError, UnicodeDecodeError)` and a per-line schema/JSON error → `_CliError` naming the file + line number → exit `1`). An empty outcome file → `_CliError` ("no outcomes").
- **Read manifests.** `json.loads(path.read_text(encoding="utf-8"))` → must be a JSON object (`dict`) → else `_CliError`. Passed straight to `build_report` as the `Mapping[str, Any]` (the `Manifest.to_dict()` shape). A read / parse error → `_CliError` → exit `1`.
- **Build.** `report.build_report(primary_outcomes, primary_manifest, optimistic_outcomes, optimistic_manifest) -> ThreeWayReport`. A `ReportError` it raises (malformed manifest, non-PRIMARY/OPTIMISTIC config pair, mixed entry sides, duplicate `order_id`, …) is caught → `_CliError` with the message → exit `1`. Any other exception propagates (a bug).
- **Write.** `--out`: `json.dumps(report.to_dict(), indent=2, sort_keys=True)` via the existing atomic-write helper (`.tmp` → `os.replace`; parent dirs created; overwrite; on failure exit `1`). One file, so a single-path variant of `_atomic_write_pair` (or reuse it with one entry).
- **Summary to stdout** — a short block: N round trips, N incomplete positions, and for each of the three models (`primary` / `stressed` / `optimistic`): `n`, net cents, win rate, profit factor (render `inf` / `None` as `"inf"` / `"n/a"`); a line noting the optimistic model is over the both-completed subset; the `--out` path.
- **`main` dispatch** routes `command == "report"` to `_cmd_report(args)` (mirrors `_cmd_simulate`: catch `_CliError` → stderr + return `1`). `report --help` exits `0`.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. `PERMITTED_INTERNAL_EDGES["cli"] = {"sim", "events", "orders", "config", "book", "report"}` (`report` re-added). No new stdlib beyond what `cli.py` already imports.

**Ask First:**
- (none — the `report` subcommand always rebuilds via `build_report` from the two runs' outcome logs; it never reloads a prior report JSON, so no `ThreeWayReport.from_dict` is needed. `build_report`'s existing `ReportError` taxonomy covers every malformed-input case.)

**Never:**
- The `parity-gate` subcommand, `run_part_a` / `run_part_b` / `gate` / reconstruction / `preflight_integrity` (a later slice).
- Running `sim.simulate` (that's `simulate`) — `report` consumes already-written outcome logs.
- Importing `databento`, `parity/*`. `datetime.now` / network / a second `subprocess`.
- Writing anywhere but `--out`.
- Silently proceeding when the two manifests aren't a PRIMARY+OPTIMISTIC pair — `build_report` raises `ReportError`, which becomes exit `1`.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| happy path | 4 valid files from a real PRIMARY + OPTIMISTIC pair | `--out` JSON written (`round_trips`, `primary`, `stressed`, `optimistic` keys), stdout summary, exit `0` | N/A |
| malformed outcome line | line 5 of `--primary-outcomes` not valid `OrderOutcome` JSON | — | exit `1` naming the file + line 5 |
| empty outcome file | 0 non-blank lines | — | exit `1` ("no outcomes") |
| manifest not an object | `--primary-manifest` is a JSON array | — | exit `1` |
| manifest unreadable | missing file | — | exit `1` |
| not a PRIMARY+OPTIMISTIC pair | two PRIMARY manifests | — | exit `1` (from `ReportError`) |
| mixed entry sides / dup order_id | a corrupt outcome log | — | exit `1` (from `ReportError`) |
| output path collides with an input | `--out` == `--primary-manifest` | — | exit `2` (`parser.error`) |
| `--out` in a missing dir | `/nope/r.json` | parent created, written | exit `0` (or `1` if unwritable) |
| missing required arg | no `--optimistic-manifest` | argparse usage | exit `2` |
| `report --help` | — | usage to stdout | exit `0` |

</frozen-after-approval>

## Code Map

- `src/ticksim/cli.py` — add `_cmd_report(args) -> int`, `_run_report(args) -> int`, `_read_outcomes(path) -> list[OrderOutcome]`, `_read_manifest(path) -> dict[str, object]`, `_atomic_write_one(path, text)` (or a one-entry call into `_atomic_write_pair`), a `report` subparser in `_build_parser`, `_validate_report_args`, and the `command == "report"` branch in `main`. `__all__` unchanged (`report` internals are private).
- `src/ticksim/report.py:400` — `build_report(primary_outcomes, primary_manifest, optimistic_outcomes, optimistic_manifest) -> ThreeWayReport`; `ThreeWayReport.to_dict()`; `ReportError`.
- `src/ticksim/orders.py` — `OrderOutcome` (pydantic — `model_validate_json`).
- `src/ticksim/sim.py:242` — `Manifest.to_dict()` shape (what `--*-manifest` files hold; a plain dict on disk).
- `tests/unit/test_ticksim_imports.py:39` — `"cli"` edge gains `"report"`.
- Existing `cli.py` helpers reused: `_CliError`, `_read_intents` (as the pattern), `_atomic_write_pair`, `_write_tmp`, `_configure_logging`, `_cmd_simulate` (as the dispatch pattern).

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/cli.py` — the `report` subcommand: parser, `_validate_report_args`, `_read_outcomes`, `_read_manifest`, `_run_report`, `_cmd_report`, `_atomic_write_one`, `main` branch.
- [x] `tests/unit/test_ticksim_imports.py` — add `"report"` to the `"cli"` edge set.
- [x] `tests/unit/test_ticksim_cli_report.py` (25 tests) — build a real PRIMARY + OPTIMISTIC pair in `tmp_path` (either via `cli.main(["simulate", …])` twice with a hand-built in-memory `DbnMboSource`, or by hand-constructing `OrderOutcome`s + `Manifest.to_dict()` dicts), then `cli.main(["report", …])` → assert `--out` JSON parses with the four keys, exit `0`; every I/O-matrix error row → its exit code + stderr message; the stdout summary lines.

**Acceptance Criteria:**
- Given four valid files from a PRIMARY + OPTIMISTIC run of the same intent log, when `cli.main(["report", …])` runs, then exit `0` and `--out` is JSON with `round_trips` / `primary` / `stressed` / `optimistic` keys, each model carrying an `n` and a `net_cents`.
- Given two PRIMARY manifests, then exit `1` and stderr contains the `ReportError` message about the config pair.
- Given `--out` equal to `--primary-outcomes`, then exit `2`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `cli.py` imports `report`; `python -m src.ticksim.cli report --help` exits `0`.

## Spec Change Log

**Review round 1 — 2026-08-31 — patch round (no code re-derivation).** All findings patch. One Code-Map reconciliation: `_read_manifest -> dict[str, object]` (stricter than the `Any` originally written; still satisfies `build_report`'s `Mapping[str, Any]`). ~19 patches: `BrokenPipeError` guard in `_cmd_report` (mirror `_cmd_simulate`); `report.to_dict()` sanitized before `json.dumps` — `float('inf')` / `-inf` / `nan` → `null` recursively (a `profit_factor` of `inf` was writing a bare `Infinity` token, invalid for strict JSON parsers); `_validate_report_args` rejects an empty-name `--out` (`Path(args.out).name == ""` → `parser.error`); the stdout summary gains `optimistic-only-completed` and `partially-closed` counts and only prints the optimistic-subset explainer when `optimistic.n != primary.n` (always showing the two n's); `_read_outcomes` + `_read_intents` refactored to a shared `_read_jsonl(path, model_cls, label)`; `_run_report` `-v` logging gains the resolved config/queue-model pair + round-trip / incomplete / partially-closed counts; the `--optimistic-outcomes` help string + module docstring `simulate` fragment cleaned up. New tests: unreadable / non-UTF-8 `--primary-outcomes`; a zero-round-trip pair (entry-only intents) → exit 0, `round trips: 0`, `win_rate=n/a` / `profit_factor=n/a`; `--out` unwritable (monkeypatch `_write_tmp` → `OSError`) → exit 1 + no `.tmp` residue; an input↔input path collision → exit 2; a mixed-entry-side outcome log → exit 1 (`ReportError`); a non-zero `incomplete` (real `OpenPosition`); numeric assertions on `net_cents` and the `stressed = primary − 2·TICK_VALUE_CENTS·matched_size` transform; the `-v` INFO path; a tightened "not a PRIMARY+OPTIMISTIC pair" assertion.

## Design Notes

**Reuse over re-invent.** `report` is a thin file-in / file-out wrapper around `build_report` — every hard part (P&L math, the stress transform, the both-completed optimistic subset, the `ReportError` taxonomy) is already in `report.py` and tested. This slice is argparse plumbing + the outcome/manifest readers, patterned on the `simulate` subcommand's `_read_intents` / `_atomic_write_pair` / `_cmd_simulate`.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_cli_report.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim tests/unit/test_ticksim_cli_report.py` — expected: clean.
- `PYTHONPATH=. .venv/bin/python -m src.ticksim.cli report --help` — expected: exit `0`.

## Suggested Review Order

**Dispatch + validation**

- `_cmd_report` / `main` branch: mirrors `_cmd_simulate` (catch `_CliError` -> exit 1, `BrokenPipeError` -> exit 0). `_validate_report_args`: 5 distinct resolved paths + non-empty `--out` name -> `parser.error` (exit 2).
  [`cli.py:434`](../../src/ticksim/cli.py#L434)

**The run**

- `_run_report`: read 2 outcome logs + 2 manifests -> `build_report` (catch `ReportError` -> `_CliError`) -> `_json_safe` -> atomic write -> summary.
  [`cli.py:449`](../../src/ticksim/cli.py#L449)

- `_json_safe`: recursively maps `inf`/`-inf`/`nan` -> `None` so the JSON file is strict-parseable (a `profit_factor` of `inf` was writing a bare `Infinity` token).
  [`cli.py:499`](../../src/ticksim/cli.py#L499)

- `_print_report_summary`: round-trip / incomplete / partially-closed / optimistic-only counts; per-model `n`/`net_cents`/`win_rate`/`profit_factor`; the subset explainer only when `optimistic.n != primary.n`.
  [`cli.py:592`](../../src/ticksim/cli.py#L592)

**Shared readers**

- `_read_jsonl(path, model_cls, label, empty_noun)` — the dedup of `_read_intents` + `_read_outcomes` (generic over a pydantic model); both public names kept as wrappers.
  [`cli.py:174`](../../src/ticksim/cli.py#L174)

**Peripherals**

- `"cli"` edge gains `report`; 25 tests incl. zero-round-trip (`n/a` path), unwritable `--out`, mixed-entry-side `ReportError`, the `stressed` transform arithmetic.
  [`test_ticksim_cli_report.py:1`](../../tests/unit/test_ticksim_cli_report.py#L1)
