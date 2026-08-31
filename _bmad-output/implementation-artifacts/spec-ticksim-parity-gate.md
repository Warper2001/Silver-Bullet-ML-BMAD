---
title: 'ticksim parity/gate.py — the two-part verdict + amendment stub (AD-26)'
type: 'feature'
created: '2026-08-31'
status: 'done'
review_loop_iteration: 0
baseline_commit: '57152cd'
context:
  - '{project-root}/_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md'
  - '{project-root}/_bmad-output/preregistration_tick_data_infrastructure.md'
---

<!-- SPLIT (2026-08-31, planning): the §5 integrity preflight scan over the MBO
     windows (ts-monotonic / persistent-cross / A-C-M-T-F-seen / degraded-day
     flag, AD-9/AD-20) is deferred to gate.py slice 2. THIS slice is the AD-26
     output contract: the two-part verdict, the one sanctioned git-SHA
     subprocess, and the fixed-template amendment stub built from a PartAResult
     + a PartBResult. The stub carries an `integrity:` placeholder line slice 2
     fills in. -->

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** `run_part_a` produces a `PartAResult`, `run_part_b` a `PartBResult`, but nothing combines them into the prereg §A8.2 gate verdict or emits the §4 append-only **amendment stub** with the frozen simulator commit SHA. Spine AD-26 pins that output contract to `parity/gate.py`.

**Approach:** add `src/ticksim/parity/gate.py` — `evaluate(part_a, part_b) -> GateVerdict` (verdict = **Part A PASS AND Part B PASS**, per AD-26 / §A8.2), `frozen_sha() -> str` (the **one** sanctioned `subprocess` call — `git rev-parse HEAD` — AD-4/AD-11 otherwise stand), and `build_amendment_stub(part_a, part_b, *, cycle_number, sha=None, integrity=None, trader_by_trade_id=None) -> str` producing the fixed-template Markdown an analyst appends to the seal. The §5 integrity preflight scan is slice 2 (the stub takes an `integrity` string, `None` → a `pending` placeholder line).

## Boundaries & Constraints

**Always:**
- **`evaluate(part_a: PartAResult, part_b: PartBResult) -> GateVerdict`.** `GateVerdict(verdict: Literal["PASS","FAIL"], part_a_pass: bool, part_b_pass: bool, reason: str)`. `part_a_pass = (part_a.verdict == "PASS")`, `part_b_pass = (part_b.verdict == "PASS")`, `verdict == "PASS"` iff **both**. `reason` states which side(s) failed, quoting the failing result's own `reason` verbatim, and — per AD-26 — spells out the asymmetry: a Part A pass + Part B fail is "the fill model is structurally broken"; a Part B pass + Part A fail is "the model runs but is miscalibrated".
- **`frozen_sha() -> str`.** `subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=<repo root>)`, return the stripped 40-char hex. This is the **only** `subprocess` / shell call anywhere in `src/ticksim/` (AD-26). On `CalledProcessError` / `FileNotFoundError` (no git) → `GateError` naming the cause — the amendment MUST carry a real SHA, never a guess. A dirty working tree is **not** an error here (the analyst runs the gate from a committed state by discipline) but `build_amendment_stub` notes `git rev-parse` was taken as-is.
- **`build_amendment_stub(part_a, part_b, *, amendment_number: int, cycle_number: int, sha: str | None = None, integrity: str | None = None, date: str | None = None, trader_by_trade_id: Mapping[str, str] | None = None) -> str`.** A fixed-template Markdown block, append-only in spirit (the function only *returns* the text — it never writes a file). Sections, in this order:
  1. **Header** — `# Amendment {amendment_number} -- Parity gate result (cycle {cycle_number})`; a `_date:` line = `date` if given else `"TBD (fill on append)"` (no wall-clock in `src/ticksim` — AD-1). The rendered document uses `--`, never an em-dash (ASCII-safe, deterministic — round-1 patch).
  2. **Verdict** — `evaluate(part_a, part_b).verdict`, then the `part_a_pass` / `part_b_pass` booleans and the `reason`.
  3. **Frozen SHA** — `sha` if given, else `frozen_sha()`; the line reads `simulator commit: <sha>` (and on a `frozen_sha()` failure the whole call raises `GateError` — never emit a stub without a SHA).
  4. **Part A** — sample `N` (`part_a.stats.n`); a 3-row table: MAE / p90 / signed-bias, each `value` vs its `config` tolerance (`PARITY_MAE_MAX_TICKS` / `PARITY_P90_MAX_TICKS` / `PARITY_SIGNED_BIAS_MAX_TICKS`) with a `PASS`/`FAIL` per row; the `broker_fill`-only stats as a second 3-row table; `part_a.warning` if set; `PART_A_MIN_N` and whether `n` met it.
  5. **Part A per-trader breakdown** — group `part_a.errors` by `trader_by_trade_id[err.trade_id]` when the map is given, else by `err.fidelity` (`broker_fill` ≈ mim-nb, `bar_reconstructed` ≈ yank for this sample — stated as a caveat); per group: count, mean |error| ticks, mean signed error ticks.
  6. **Part B** — `n_orders`, `n_fill_events`, `PART_B_MIN_ORDERS` and whether met; a per-label violation count table sorted by label (from `part_b.violations`), `-- none --` when clean; then `part_b.coverage_note` **verbatim** (the deferred-work item: the gate report MUST surface what the battery does and does not certify).
  7. **Integrity** — `integrity` verbatim if given, else `integrity: pending (gate.py slice 2 -- §5 preflight not yet wired)`.
  8. **Cycle / kill-criterion** — `cycle {cycle_number} of 3`; a fixed sentence that the 15-working-day / 3-cycle kill clock is tracked **out of code** by the analyst (AD-26).
- **No verdict logic outside `evaluate`.** `build_amendment_stub` calls `evaluate`; it does not re-derive PASS/FAIL. The Part-A row-level PASS/FAIL (step 4) re-compares the stat to the `config` tolerance for *display* — for a `PASS` result (both the full and `broker_fill` tables), a row out of tolerance → `GateError` ("part_a display rows are inconsistent with `src.ticksim.config` — different constants or an `aggregate` bug"). Only `PASS` is guarded (a `FAIL`'s cause may be the N-floor / unresolved-miss, invisible in the 3 rows). Also raised: `amendment_number <= 0`; a `sha` / `date` / `integrity` that is a non-40-hex string / contains a newline or `## `; either `verdict` not exactly `"PASS"`/`"FAIL"` (from `evaluate`).
- `mypy --strict src/ticksim` clean, no override; `black`-88; relative imports. `PERMITTED_INTERNAL_EDGES["gate"] = {"config", "part_a", "part_b"}` (parity-sibling resolver already handles `part_a` / `part_b`). Standard library: `subprocess`, `dataclasses`, `typing`, `pathlib`, `collections`, `re` only — **no `datetime` / `time`, no `os.system`, no network** (AD-1/AD-4/AD-11); exactly one `subprocess.run` call in the whole module (a source-guard test enforces this via AST, not text-splitting).

**Ask First:**
- (resolved at CHECKPOINT 1, 2026-08-31) `frozen_sha()` uses a `_find_repo_root()` helper — walk parent dirs from `__file__` to the first containing a `.git` entry; `GateError` if none (no extra subprocess). `build_amendment_stub` takes a **required** `amendment_number: int` param (analyst supplies it, like `cycle_number`); the header renders the real number.

**Never:**
- Writing to the seal file, or any file (the function returns text; the analyst appends).
- A second `subprocess` call, any shell string, `os.system`, network, or `datetime.now` / `time.time` (AD-1/AD-4/AD-11).
- Re-running Part A or Part B, loading `.dbn.zst`, or the §5 integrity scan (slice 2).
- Deriving the cycle number or the kill-criterion clock in code (AD-26 — out of code).

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| both parts PASS | `part_a.verdict == "PASS"`, `part_b.verdict == "PASS"` | `GateVerdict(verdict="PASS", part_a_pass=True, part_b_pass=True, …)` | N/A |
| Part A PASS, Part B FAIL | — | `verdict == "FAIL"`, reason includes "structurally broken" + Part B's `reason` | N/A |
| Part B PASS, Part A FAIL | — | `verdict == "FAIL"`, reason includes "miscalibrated" + Part A's `reason` | N/A |
| both FAIL | — | `verdict == "FAIL"`, reason names both | N/A |
| `frozen_sha()` with git present | run inside the repo | 40-char lowercase hex | N/A |
| `frozen_sha()` no git / not a repo | `git` missing or `cwd` outside a repo | — | `GateError` |
| `build_amendment_stub` clean gate | passing results, `sha` supplied, `integrity` supplied | full Markdown; Part B table `— none —`; coverage note present; integrity line = supplied text | N/A |
| stub, no `sha`, git present | `sha=None` | `frozen_sha()` invoked once; SHA line filled | N/A |
| stub, no `sha`, git absent | `sha=None`, no git | — | `GateError` (never a SHA-less stub) |
| stub, no `integrity` | `integrity=None` | `integrity: pending (…slice 2…)` line | N/A |
| Part-A display rows contradict `part_a.verdict` | `part_a.verdict=="PASS"` but MAE > tolerance | — | `GateError` (part_a built against different constants) |
| `trader_by_trade_id` omitted | `None` | per-trader section groups by `fidelity` with the mim-nb/yank caveat | N/A |
| `cycle_number` out of 1..3 | e.g. 4 | rendered as-is (`cycle 4 of 3`) — the analyst owns the clock | N/A |

</frozen-after-approval>

## Code Map

- `src/ticksim/parity/gate.py` — NEW. `evaluate`, `frozen_sha`, `build_amendment_stub`, `_find_repo_root`, frozen `GateVerdict`, `GateError`, `__all__`. No module-level state.
- `src/ticksim/parity/part_a.py` — `PartAResult(stats: PartAStats(n, mae_ticks, p90_ticks, signed_bias_ticks), broker_fill_stats, verdict: "PASS"|"FAIL", reason, warning: str|None, errors: tuple[FillError(trade_id, order_id, …, fidelity), ...])`.
- `src/ticksim/parity/part_b.py` — `PartBResult(n_orders, n_fill_events, violations: tuple[Violation(order_id, invariant, message), ...], verdict, reason, coverage_note)`; `PART_B_MIN_ORDERS` via `config`.
- `src/ticksim/config.py` — `PARITY_MAE_MAX_TICKS=1.0`, `PARITY_P90_MAX_TICKS=2.0`, `PARITY_SIGNED_BIAS_MAX_TICKS=0.25`, `PART_A_MIN_N=28`, `PART_B_MIN_ORDERS=1000`.
- `_bmad-output/…/ARCHITECTURE-SPINE.md` — AD-26 (this contract, verbatim), AD-4/AD-11 (the one-subprocess carve-out), AD-27 (the seal-bound constants).
- `_bmad-output/preregistration_tick_data_infrastructure.md` — §A8.2 Verdict (the PASS-both rule + the two asymmetry sentences), §4 (kill criterion, out-of-code clock), §A2.3 (on PASS: record SHA, buy Tranche 2).
- `tests/unit/test_ticksim_imports.py:39` — add `"gate": {"config", "part_a", "part_b"}`.

## Tasks & Acceptance

**Execution:**
- [x] `src/ticksim/parity/gate.py` — `evaluate` + `frozen_sha` (`timeout=15`, catches `CalledProcessError`/`FileNotFoundError`/`OSError`/`TimeoutExpired`) + `build_amendment_stub` (`amendment_number` required, caller-`sha`/`date`/`integrity` validated for hex/empty/newline) + `_find_repo_root` + `_stat_table` / `_part_a_rows` / `_per_trader_section` / `_check_caller_field` + `GateVerdict` / `GateError`, `__all__`, relative imports. One `subprocess.run`. `--` throughout the rendered doc.
- [x] `src/ticksim/parity/__init__.py` — allowed-edges docstring for `gate`.
- [x] `tests/unit/test_ticksim_imports.py` — `"gate": {"config", "part_a", "part_b"}`.
- [x] `tests/unit/test_ticksim_parity_gate.py` — 46 tests: `evaluate` all four PASS/FAIL combos + asymmetry wording + verbatim reason quoting + unknown-verdict `GateError`; `frozen_sha()` real HEAD (lowercase 40-hex, deterministic) + `CalledProcessError`/bad-hex via `monkeypatch.setattr(gate.subprocess, "run", …)`; `_find_repo_root` happy + no-`.git` failure; `build_amendment_stub` passing pair (all section headers in order, SHA line, coverage note verbatim, `-- none --` Part B, `pending` integrity when omitted, `unresolved misses` line, min-floor met) + failing pair (Part A row FAIL, sorted Part B per-label table, min-floor NOT met both parts); distinct `stats` vs `broker_fill_stats` tables; empty broker_fill subset → `(empty subset …)`; per-trader grouping by `fidelity` (map omitted) + `<unmapped:…>` (map missing a trade_id) + miss-count column; `GateError` on SHA-less+no-git, `amendment_number <= 0`, display/verdict contradiction (incl. `broker_fill_stats`), newline/`## ` in a caller field; `date=""`/`integrity=""` → placeholder; byte-identical determinism with fixed `sha`/`date`; an AST source-guard (exactly one `subprocess.run`, no `datetime`/`time`/`os.system`/`socket`/`urllib`).

**Acceptance Criteria:**
- Given `part_a.verdict == "PASS"` and `part_b.verdict == "FAIL"`, when `evaluate` runs, then `verdict == "FAIL"` and `reason` contains "structurally broken" and Part B's `reason`.
- Given a passing `PartAResult` + `PartBResult`, a fixed `sha` and `date`, when `build_amendment_stub` runs twice, then the two strings are byte-identical and contain `part_b.coverage_note` verbatim.
- Given `sha=None` and no `git` on `PATH`, when `build_amendment_stub` runs, then `GateError` — no stub is returned.
- Given `mypy --strict src/ticksim`, then zero errors, no override; `gate.py` imports only `{config, part_a, part_b}` from `src.ticksim` and makes exactly one `subprocess` call in the whole module.

## Spec Change Log

**Review round 1 — 2026-08-31 — patch round (no code re-derivation).** Reviewer trio; all findings patch. One frozen-text reconciliation: the "Standard library:" allowlist under-enumerated `collections` (`Counter` / `abc.Mapping`) and `re` (message parsing) — both benign, added. Patches applied to `gate.py` + tests: fixed the malformed `_per_trader_section` Markdown table (literal `|err|` pipes, 5-vs-4 column separator); empty `broker_fill` subset now renders `(empty subset)` / `n/a` rows instead of misleading `0.0000 PASS`; a caller-supplied `sha` gets the same 40-lower-hex validation `frozen_sha()` applies (a truncated/typo SHA must never seal); `date=""` / `integrity=""` coalesce to the placeholder; caller strings (`sha`, `date`, `integrity`) with a newline or `## ` → `GateError` (no Markdown-section forgery, determinism preserved); `evaluate` raises `GateError` if either `verdict` is not exactly `"PASS"`/`"FAIL"`; `frozen_sha()` gains `timeout=15` and catches `(CalledProcessError, FileNotFoundError, OSError, TimeoutExpired)`; the PASS-side contradiction guard also checks `broker_fill_stats` and its message is softened ("inconsistent with `config` — different constants or an `aggregate` bug"); `unresolved_misses` gets its own Part A line; the per-trader table gains a miss-count column; `amendment_number <= 0` → `GateError`; the rendered doc uses `--` throughout (no em-dash — ASCII-safe + deterministic); one `import subprocess` only (`subprocess.CalledProcessError` etc.). New tests: distinct `stats` vs `broker_fill_stats` tables, both min-floor NOT-met branches, per-trader empty / all-miss groups, `frozen_sha` `CalledProcessError` + bad-hex via `monkeypatch.setattr(gate.subprocess, "run", …)`, `_find_repo_root` no-`.git` failure, `<unmapped:…>` trader-map path, the PASS `reason` wording.

## Design Notes

**Why `evaluate` is separate from the stub.** `cli.py` (next slice) will print the verdict to the terminal for a quick check and only build the full stub when the analyst asks. Keeping `evaluate -> GateVerdict` a pure 3-line function lets both paths share the exact PASS-both rule with no duplication.

**The one subprocess.** AD-26 explicitly carves `git rev-parse HEAD` out of the AD-4/AD-11 "no subprocess / no wall-clock" rule because the frozen SHA is the whole point of §4's amendment. A test asserts the module's source contains exactly one `subprocess.` reference so a second one can't creep in.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/test_ticksim_parity_gate.py tests/unit/test_ticksim_imports.py -q` — expected: all pass.
- `.venv/bin/python -m pytest tests/unit/ -k ticksim -q` — expected: full ticksim unit suite green.
- `.venv/bin/python -m mypy --strict src/ticksim` — expected: `Success`.
- `.venv/bin/python -m black --check src/ticksim/parity tests/unit/test_ticksim_parity_gate.py` — expected: clean.

## Suggested Review Order

**The verdict (AD-26 / §A8.2)**

- `evaluate`: PASS iff both parts PASS; the "structurally broken" / "miscalibrated" asymmetry wording; unknown-verdict `GateError`.
  [`gate.py:136`](../../src/ticksim/parity/gate.py#L136)

**The one sanctioned subprocess**

- `frozen_sha`: `git rev-parse HEAD` with `timeout=15`, 40-lower-hex validation, `GateError` on any failure. `_find_repo_root` walk-up (no extra subprocess).
  [`gate.py:200`](../../src/ticksim/parity/gate.py#L200)

**The amendment stub**

- `build_amendment_stub`: required `amendment_number`, caller-field validation (hex / empty / newline), resolves the SHA before emitting any text, calls `evaluate` (no re-derivation), renders `part_b.coverage_note` verbatim, `integrity: pending` placeholder.
  [`gate.py:386`](../../src/ticksim/parity/gate.py#L386)

- `_stat_table` (empty-subset `(empty subset)` rows) + `_part_a_rows` (display PASS/FAIL vs `config` tolerance, contradiction guard) + `_per_trader_section` (fixed 5-col table, `misses` column, `fidelity` fallback grouping).
  [`gate.py:264`](../../src/ticksim/parity/gate.py#L264)

**Peripherals**

- Import edge `gate → {config, part_a, part_b}`; AST source-guard test.
  [`test_ticksim_parity_gate.py:1`](../../tests/unit/test_ticksim_parity_gate.py#L1)
