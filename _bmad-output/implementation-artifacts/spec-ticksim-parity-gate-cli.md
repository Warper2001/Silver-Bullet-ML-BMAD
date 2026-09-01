---
title: 'ticksim parity/gate_cli.py + cli.py `parity-gate` — the §A8.2 gate orchestrator'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 2
baseline_commit: '64c058a'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** Every §A8.2 building block is built and merged — `part_a` reconstruction, `part_a_runner.run_part_a`, `synthetic.generate_synthetic_orders`, `part_b.run_part_b`, `integrity.preflight_integrity`, `gate.evaluate` / `gate.build_amendment_stub` — but nothing wires them into the one command the analyst runs to get a verdict + the append-only amendment stub. This is the last capstone.

**Approach:** add `src/ticksim/parity/gate_cli.py` — a pure orchestrator `run_parity_gate(...) -> GateRun` that runs Part A over the reconstructed trades' windows, Part B over a synthetic-order batch, the §5 integrity preflight per window, then `gate.evaluate` + `gate.build_amendment_stub`; and add the thin `cli.py parity-gate` subcommand that does the file/DB I/O (read `data/mim_nb/orders.csv`, `data/trades.db`, a `--windows` JSON) and hands `gate_cli` the pieces. `run_parity_gate` never touches `.dbn.zst` paths or the DB — the CLI injects a `source_for` callable.

## Boundaries & Constraints

**Always:**
- **`run_parity_gate(part_a_trades, windows, synthetic_window, source_for, *, synthetic_seed, synthetic_n, amendment_number, cycle_number, config=None, sha=None, date=None) -> GateRun`** (`gate_cli.py`). `part_a_trades: Sequence[ReconstructedTrade]`; `windows: Mapping[WindowKey, WindowSpec]` where `WindowSpec` is a frozen `(lo_ns: int, hi_ns: int, degraded_days: tuple[str, ...] = ())` (`WindowKey` alone is a `str` and cannot carry the ranges the trade→window routing and Part B `lo_ns`/`hi_ns` need — round-1 reconciliation); `synthetic_window: WindowKey` (one dense window, CHECKPOINT 1b) must be a key in `windows`; `source_for: Callable[[WindowKey], BookEventSource]` returns a **single-instrument re-iterable** source for a window (front-month filtering + `.dbn.zst` open + `[lo_ns, hi_ns)` clipping is the CLI's job). `config: SimConfig | None` (`None` → each runner's own `PRIMARY` default — `gate_cli`'s import edge excludes `config` so `PRIMARY` can't be imported; the CLI always passes a real `SimConfig`). **`amendment_number <= 0` → `GateCliError` at the top of `run_parity_gate`** (before any compute). Steps, in order:
  1. **Part A** — `part_a_runner.run_part_a(part_a_trades, lambda t: source_for(<t's window key>), config=config) -> PartAResult`. (The trade→window mapping: `source_for` is `lambda t: window_source_for(_window_of(t, windows))` where `_window_of` matches the trade's entry ts to a `[lo_ns, hi_ns)` — CHECKPOINT 1d; the CLI builds the `windows` structure.)
  2. **Part B** — `src = source_for(synthetic_window)`; `intents = synthetic.generate_synthetic_orders(src, lo_ns, hi_ns, n=synthetic_n, seed=synthetic_seed)`; `part_b.run_part_b(intents, src, config=config) -> PartBResult`. `lo_ns`/`hi_ns` come from the window key.
  3. **Integrity** — for **every distinct window** touched (Part A windows ∪ `synthetic_window`): `integrity.preflight_integrity(source_for(w), degraded_days=windows[w].degraded_days) -> IntegrityReport`; join each `format_integrity(report)` into one `integrity` string, one `window <key>: …` line per window with newlines and `## ` flattened out (`build_amendment_stub`'s `_reject_template_break` rejects both — round-1). Window identity + OK/FLAGGED status preserved.
  4. **Verdict + stub** — `verdict = gate.evaluate(part_a_result, part_b_result)`; `stub = gate.build_amendment_stub(part_a_result, part_b_result, amendment_number=amendment_number, cycle_number=cycle_number, sha=sha, integrity=<the joined string>, date=date, trader_by_trade_id=<derived>)`.
  - `GateRun(part_a: PartAResult, part_b: PartBResult, integrity_reports: tuple[tuple[str, IntegrityReport], ...], verdict: GateVerdict, stub: str, integrity_flagged: bool)`. `integrity_flagged = any(r.verdict == "FLAGGED" for _, r in integrity_reports)`.
  - **A `FLAGGED` integrity report does not change `verdict`** (CHECKPOINT 1a — AD-26's verdict stays Part A PASS AND Part B PASS); `run_parity_gate` sets `integrity_flagged=True`, the stub integrity section is loud, and the CLI exits `3` on a flagged-but-PASS run.
- **`trader_by_trade_id`** — `_trader_of(trade)`: `"trader-mim-nb"` when `trade_id` starts `"mimnb-"` (CSV `reconstruct_mim_nb`) **or** `"trader-mim-nb-"` (DB-fallback `reconstruct_trades_db_row`), else `"trader-yank"`. Passed to `build_amendment_stub` for its per-trader breakdown.
- **`cli.py parity-gate` args:** `--orders-csv PATH` (default `data/mim_nb/orders.csv`), `--trades-db PATH` (default `data/trades.db`), `--windows PATH` (JSON, required — see Code Map for the schema), `--synthetic-window KEY` (required — a key present in `--windows`), `--synthetic-seed INT` (default 0), `--synthetic-n INT` (default `PART_B_MIN_ORDERS`), `--amendment-number INT` (required), `--cycle-number INT` (required), `--config {primary,optimistic}` (default primary), `--sha SHA` (optional — else `gate.frozen_sha()`), `--date STR` (optional), `--out PATH` (the `.md` stub, required), `-v`.
- **CLI reconstruction.** `csv.DictReader(open(orders_csv, encoding="utf-8"))` → `part_a.reconstruct_mim_nb(rows)`. `sqlite3` with a dict row factory: `SELECT * FROM trades WHERE trader_id='trader-yank' AND timestamp>='2026-06-17'` (CHECKPOINT 1c: the CSV is authoritative for mim-nb — the DB's mim-nb rows are used only if the CSV is absent) → `part_a.reconstruct_trades_db_row(row)` per row. A reconstruction `PartAError` → `_CliError` naming the trade → exit `1`.
- **CLI source_for.** From `--windows` JSON: each window entry gives a `.dbn.zst` path, a front-month `instrument_id`, `lo_ns`/`hi_ns`, and an optional `degraded_days` list. `source_for(key) = <ts-clipped> FrontMonthSource(DbnMboSource(path), iid)`. A missing window file → `_CliError` → exit `1`.
- **CLI write.** The stub `.md` to `--out` via the atomic single-file helper — **a new file, never appended to the seal** (AD-26: the analyst appends). Exit `0` on PASS + no integrity flag; `3` on PASS + integrity flag; `1` on any FAIL / handled error; `2` on a usage error. Stdout: the verdict line, the flag state, the `--out` path.
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. `PERMITTED_INTERNAL_EDGES["gate_cli"] = {"part_a", "part_a_runner", "synthetic", "part_b", "integrity", "gate", "events"}`; `PERMITTED_INTERNAL_EDGES["cli"]` gains `"parity"` (it imports `parity.gate_cli` + `parity.part_a`). Stdlib in `cli.py`: `csv`, `sqlite3` added.

**Ask First:**
- (resolved at CHECKPOINT 1, 2026-08-31, all four): (a) a `FLAGGED` integrity report does **not** change `verdict` (AD-26's two-part rule stands) — `run_parity_gate` sets `integrity_flagged=True`, the stub's integrity section is loud, and the CLI exits `3` on a flagged-but-PASS run. (b) Part B is generated over **one** `--synthetic-window KEY` (a single dense RTH window). (c) mim-nb is reconstructed from **`data/mim_nb/orders.csv`** via `reconstruct_mim_nb`; the DB's mim-nb rows are a fallback only when the CSV is absent; yank is always `reconstruct_trades_db_row` on the DB rows `>= 2026-06-17`. (d) each reconstructed trade is mapped to its window by **matching its entry timestamp** to the `--windows` entry whose `[lo_ns, hi_ns)` contains it (no second mapping file); a trade matching zero or >1 window → `_CliError` → exit `1`.

**Never:**
- Appending to `_bmad-output/preregistration_tick_data_infrastructure.md` (the CLI writes a standalone `.md`; the analyst appends). Deriving the cycle / amendment number in code (AD-26).
- `gate_cli.py` opening a `.dbn.zst`, touching `data/trades.db` / `data/mim_nb/`, or importing `databento` — the CLI injects `source_for` and the reconstructed trades.
- Re-implementing `run_part_a` / `run_part_b` / `evaluate` / `build_amendment_stub` / any invariant — `gate_cli` only sequences them.
- A second `subprocess` (only `gate.frozen_sha` may), `datetime.now`, network.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| clean PASS gate | valid trades + windows, Part A & B both PASS, integrity OK | `GateRun.verdict.verdict == "PASS"`, `integrity_flagged == False`; CLI writes the stub, exit `0` | N/A |
| Part A FAIL | run_part_a returns FAIL | `verdict == "FAIL"`, stub says "miscalibrated"; CLI exit `1` | N/A |
| Part B FAIL | a synthetic order breaks an invariant | `verdict == "FAIL"`, stub says "structurally broken"; CLI exit `1` | N/A |
| integrity FLAGGED, else PASS | a window has a persistent cross | `verdict == "PASS"`, `integrity_flagged == True`, stub integrity section loud; CLI exit `3` | N/A |
| reconstruction error | a malformed `orders.csv` row | — | `_CliError` → exit `1` naming the row |
| missing window file | `--windows` points at an absent `.dbn.zst` | — | `_CliError` → exit `1` |
| `--synthetic-window` not in `--windows` | bad key | — | `parser.error` → exit `2` |
| `gate.frozen_sha()` fails, no `--sha` | run outside a git repo | — | `GateError` → `_CliError` → exit `1` (never a SHA-less stub) |
| `--out` collides with an input | — | — | `parser.error` → exit `2` |
| `run_parity_gate` unit (no CLI) | in-memory sources + hand-built trades | a `GateRun` with all five fields; `gate.build_amendment_stub` called once | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/gate_cli.py` — NEW. `run_parity_gate(...) -> GateRun`, frozen `GateRun`, `GateCliError`, `WindowKey` type alias, `_trader_of(trade) -> str`, `__all__`. Pure — no I/O.
- `src/ticksim/cli.py` — add `_cmd_parity_gate` / `_run_parity_gate` / `_read_windows(path) -> dict` / `_reconstruct_part_a_trades(orders_csv, trades_db) -> list[ReconstructedTrade]` / `_source_for_factory(windows) -> Callable` / `_validate_parity_gate_args`, the `parity-gate` subparser, the `main` branch.
- `--windows` JSON schema: `{"<key>": {"dbn": "<path>", "instrument_id": <int>, "lo_ns": <int>, "hi_ns": <int>, "degraded_days": ["YYYY-MM-DD", ...]}}`. Generated from `~/.claude/jobs/960bda86/tmp/parity_windows.py` output + the download manifests once Tranche 1 lands.
- `src/ticksim/parity/part_a.py` — `reconstruct_mim_nb(rows)`, `reconstruct_trades_db_row(row)`, `ReconstructedTrade` (`trade_id`, `intents`, `real_fills`), `PartAError`, `PartAResult`.
- `src/ticksim/parity/part_a_runner.py:73` — `run_part_a(trades, source_for, *, config=PRIMARY, pad_ns=...) -> PartAResult`.
- `src/ticksim/parity/synthetic.py:119` — `generate_synthetic_orders(source, lo_ns, hi_ns, *, n, seed) -> list[OrderIntent]`; `SyntheticError`.
- `src/ticksim/parity/part_b.py:158` — `run_part_b(intents, source, *, config=PRIMARY, pad_ns=...) -> PartBResult`; `PartBError`.
- `src/ticksim/parity/integrity.py` — `preflight_integrity(source, *, degraded_days=()) -> IntegrityReport`; `format_integrity(report) -> str`.
- `src/ticksim/parity/gate.py:136,386` — `evaluate(part_a, part_b) -> GateVerdict`; `build_amendment_stub(part_a, part_b, *, amendment_number, cycle_number, sha=None, integrity=None, date=None, trader_by_trade_id=None) -> str`; `frozen_sha()`; `GateError`.
- `src/ticksim/cli.py` — existing `FrontMonthSource` / `_CliError` / `_atomic_write_one` / `_configure_logging` / `_validate_*` patterns / the `main` dispatch.
- `tests/unit/test_ticksim_imports.py:39` — `"gate_cli"` row; `"cli"` gains `"parity"`.

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/gate_cli.py` — `run_parity_gate`, `GateRun`, `GateCliError`, `_trader_of`, `__all__`.
- [x] `src/ticksim/parity/__init__.py` — allowed-edges docstring for `gate_cli`.
- [x] `src/ticksim/cli.py` — the `parity-gate` subcommand.
- [x] `tests/unit/test_ticksim_imports.py` — `"gate_cli"` row + `"parity"` in the `"cli"` set.
- [x] `tests/unit/test_ticksim_parity_gate_cli.py` — `run_parity_gate` with hand-built `ReconstructedTrade`s + in-memory sources: a clean PASS `GateRun`; Part A FAIL; Part B FAIL; integrity FLAGGED sets `integrity_flagged` but not `verdict`; the integrity string joins per-window sub-headings; `_trader_of` mim vs yank; `gate.build_amendment_stub` called exactly once (spy).
- [x] `tests/unit/test_ticksim_cli_parity_gate.py` — `cli.main(["parity-gate", …])` with a monkeypatched `cli.DbnMboSource`, a hand-built `--windows` JSON + a tiny `orders.csv` + an in-memory `trades.db` (`sqlite3` `:memory:` dumped to a file, or a real temp DB): a clean run → stub `.md` written, exit `0`; a FAIL → exit `1`; integrity FLAGGED → exit `3`; every error-matrix row.
- [x] `tests/integration/test_ticksim_parity_gate.py` — `@pytest.mark.integration`, skips without the fixture: the real 3-yank-trade Part A + a 1000-order Part B + integrity over `data/tick/_test/…dbn.zst`, → a `GateRun` (not a verdict assertion — N=3).

**Acceptance Criteria:**
- Given hand-built trades whose sim outcomes make Part A and Part B both PASS and clean integrity, when `run_parity_gate` runs, then `GateRun.verdict.verdict == "PASS"`, `integrity_flagged == False`, and `GateRun.stub` contains the frozen SHA line and the `PART_B_COVERAGE_NOTE`.
- Given a window whose source has a persistent cross, when `run_parity_gate` runs and Part A/B otherwise PASS, then `verdict.verdict == "PASS"` but `integrity_flagged == True`, and `cli.main(["parity-gate", …])` exits `3`.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `gate_cli.py` imports only its permitted parity siblings + `events`; the import-graph test passes.

## Spec Change Log

**Review round 1 — 2026-08-31 — patch round (no code re-derivation).** Reviewer trio. One frozen-signature reconciliation + ~22 patches.

**Frozen reconciliation:** `run_parity_gate` gains a `windows: Mapping[WindowKey, WindowSpec]` positional (2nd) — `WindowKey` is a `str` and can't carry the `lo_ns`/`hi_ns` the trade→window routing and Part B need; the frozen text already referenced "the `windows` structure the CLI builds". `config=PRIMARY` → `config: SimConfig | None = None` (the `gate_cli` edge excludes `config`; `None` forwards each runner's own `PRIMARY` default — behaviourally identical). `WindowSpec` (frozen `(lo_ns, hi_ns, degraded_days=())`) is added to the Code Map + `__all__`. `amendment_number <= 0` is now rejected at the top of `run_parity_gate`, not deep in `build_amendment_stub`.

**Patches:**
- `_trader_of` — also match `"trader-mim-nb-"` (the DB-fallback `reconstruct_trades_db_row` prefix), not only `"mimnb-"` (the CSV `reconstruct_mim_nb` prefix). Without it, a CSV-absent run labels every mim-nb trade `trader-yank` in the stub's per-trader table.
- `_join_integrity` — the library artifact must not narrate the CLI's exit codes. State the flag ("integrity FLAGGED on window(s) X — the §A8.2 verdict is unchanged per AD-26; review before relying on this run"), never "the CLI exits 3". Also neutralise `"## "` (not only `"\n"`) before it reaches `build_amendment_stub`'s `_reject_template_break` guard.
- `_cmd_parity_gate` `BrokenPipeError` handler — compute the intended `rc` **before** the summary `print()`s; `return rc` in the handler, not `0` (here the exit code IS the verdict); wrap `os.dup2(...)` in `try/except (ValueError, OSError)`.
- `_db_rows` — `ORDER BY timestamp, id` (deterministic stub across runs); `AND timestamp IS NOT NULL`; the mim-nb DB fallback gets the same `>= 2026-06-17` floor as yank; the fallback path logs at `WARNING` and adds a one-line provenance note into the stub (mim-nb was reconstructed from the DB, no bracket legs, lower fidelity).
- `_reconstruct_part_a_trades` — `orders_csv.open(encoding="utf-8-sig", newline="")` (BOM tolerance).
- `_read_windows` — reject a window key or a `degraded_days` entry containing `"\n"` or `"## "`; range-check `instrument_id >= 0`; parse `--windows` **once** (in validation) and pass the parsed structure through (`_validate_parity_gate_args` and `_read_windows` currently both read+parse, disagree on strictness, and cause the `--synthetic-window` exit-code inconsistency).
- `_window_of` — validate the trade's **full** stamp span (min/max over intents + real fills) ⊆ `[lo_ns, hi_ns)`, not just the entry ts; a trade whose exit leg falls past `hi_ns` → `GateCliError` (else `_ClippedSource` truncates the book under it).
- `_source_for_factory` — memoise the `_ClippedSource` per `WindowKey` so N trades in one window don't trigger N `DbnMboSource` / `DBNStore.from_file` decompression passes. Fix the `run_parity_gate` docstring's "once per Part A window" claim.
- `--out` — refuse to overwrite an existing file unless `--force` (each kill-clock cycle should keep its own amendment stub).
- `_run_parity_gate` — on a FAIL, echo `run.verdict.reason` to stderr (not just `verdict:` / the path).
- `_validate_parity_gate_args` — `--amendment-number <= 0` / `--cycle-number <= 0` → `parser.error` (exit 2, a usage error, not the exit-1 a deep `GateCliError` gives).
- `tests/unit/test_ticksim_imports.py` — add a `TYPE_CHECKING`-block carve-out to `_resolve_imports` (a `from ..config import SimConfig` under `if TYPE_CHECKING:` is a type-only reference, not a runtime dependency edge) so `gate_cli` can type `config: SimConfig | None` instead of `Any`. New resolver unit test.
- Tests: `--config optimistic` forwarded (spy); a real `trader-mim-nb` DB-row fallback run + `_trader_of` labelling; a mixed mim-nb + yank trade set → correct per-trader stub section; a `--windows` entry whose `[lo_ns, hi_ns)` doesn't contain a trade's span → `GateCliError`/exit 1; `_ClippedSource` actually drops an out-of-window event; `degraded_days` from `--windows` reaches the stub; the `BrokenPipeError` path preserves the exit code; `--out` overwrite refused. Test timestamp helpers use integer-ns `divmod`, not `dt.timestamp() * 1e9` (float drift at 1.7e18 ns).

## Design Notes

**Why `gate_cli.py` is a separate module, not `cli.py` code.** `cli.py`'s edge would balloon to every parity sibling. Keeping the sequencing in `parity/gate_cli.py` (which is *allowed* to import its siblings) means `cli.py` only gains `parity` (via `gate_cli` + `part_a` for the reconstruction helpers) and stays a thin I/O shell. `gate_cli` is pure — the CLI injects `source_for` and the reconstructed trades — so it's fully unit-testable with in-memory sources.

**Exit `3` for a flagged-but-passing gate.** A `FLAGGED` integrity report on a window that Part A/B nonetheless passed is exactly the "interim result on possibly-broken data" case §A8.4 worries about. The verdict stays `PASS` per AD-26's literal rule, but a distinct exit code means a wrapper script (or the analyst's eye) can't treat it as a clean pass.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_gate_cli.py tests/unit/test_ticksim_cli_parity_gate.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim tests/unit/test_ticksim_parity_gate_cli.py tests/unit/test_ticksim_cli_parity_gate.py` — expected: clean.
- `PYTHONPATH=. .venv/bin/python -m src.ticksim.cli parity-gate --help` — expected: exit `0`.

## Spec Change Log -- review round 2 (2026-09-01)

Three reviewer subagents (blind-spot / edge-case / verification-gap) were
dispatched and all three died on an API session rate limit before reporting. The
round-2 pass was therefore done inline against the round-1 delta. No
`intent_gap` and no `bad_spec` found -- the frozen block stands as amended in
round 1. Two `patch`-class verification gaps, both fixed:

| # | Class | Finding | Resolution |
|---|---|---|---|
| R2-1 | patch | Round-1 item 13 (`_source_for_factory` per-key memoisation -- the fix that stops N Part A trades in one window each triggering a fresh `DBNStore.from_file` decompression pass) shipped with **no test**. A regression would be silent: correct output, N-fold runtime on a ~0.5 GB/hour window. | Added `test_source_for_factory_opens_each_dbn_once` (spy `DbnMboSource`; 10 `source_for("wA")` calls + 2 `source_for("wB")` -> `opened == [dbn_a, dbn_b]`, identity-checked, plus a re-iterability assert per AD-18) and `test_source_for_factory_missing_dbn_raises_cli_error`. |
| R2-2 | patch | Round-1 item 7 (`utf-8-sig` BOM tolerance on `orders.csv`) shipped with **no test**. | Added `test_orders_csv_with_utf8_bom_is_reconstructed` (a real `\xef\xbb\xbf`-prefixed CSV; asserts no reconstruction fault and that the written stub carries the mim-nb trade). |

**Mutation-verified.** Both new tests were confirmed regression-catching by
reverting their fixes in `cli.py`: the BOM test reproduces the exact original
failure (`empty ts_utc in row {'﻿ts_utc': ...}`) and the memoisation test
fails on identity. Fixes restored; suite re-green.

### Checked and sound (no change needed)

* `_flatten` applies `.replace("\n", " / ")` **before** `.replace("## ", "")`, so a
  `"##\n"` sequence cannot be reassembled into a live `## ` marker after
  flattening -- the ordering is load-bearing and correct.
* `_trade_span` can never see an empty stamp list: `ReconstructedTrade.__post_init__`
  already rejects zero intents and zero `real_fills` (`part_a.py:180-185`).
* `_db_rows`' `ORDER BY timestamp, id` is valid against the real schema
  (`trades` has an `id INTEGER PRIMARY KEY`), and `timestamp` is stored as an
  ISO-8601 string (`'2025-01-20T13:31:00+00:00'`), so the lexical
  `timestamp >= '2026-06-17'` floor orders and filters correctly.
* `_atomic_write_one` creates missing parent dirs and converts any `OSError`
  (including `--out` pointing at a directory under `--force`) into a `_CliError`
  -> exit 1, never a traceback.
* `_ClippedSource` is genuinely re-iterable (`__iter__` builds a fresh generator
  expression each call), so a memoised instance survives the many `source_for`
  calls; `_require_reiterable` guards the wrapped `inner`.
* The `TYPE_CHECKING` carve-out excludes only the `if` body, not an `else`
  fallback, and is covered by three resolver unit tests including the negative
  case.

### Deferred (appended to `deferred-work.md`)

* `_is_type_checking_test` treats **any** `ast.Attribute` whose `.attr` is
  `TYPE_CHECKING` as the guard (so a hypothetical `if somemodule.TYPE_CHECKING:`
  would be honoured). Not reachable in this codebase; tightening it to
  `typing`/`t` bindings needs alias tracking.
* The round-2 reviewer subagents never ran. If the rate limit clears before the
  Tranche-1 purchase, re-run the three-reviewer pass over this slice.
