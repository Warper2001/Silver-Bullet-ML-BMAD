# Spine Review — TICK-INFRA Fill Simulator

**File reviewed:** `_bmad-output/planning-artifacts/architecture/tick-infra-fill-simulator-2026-08-29/ARCHITECTURE-SPINE.md`
**Reviewed against:** good-spine checklist (8 items)
**Inputs it must serve (not reviewed):** `preregistration_tick_data_infrastructure.md` §2 / §3 / Amendment 8 §A8.2 / Amendment 9; `project-context.md`
**Date:** 2026-08-29
**Verdict:** PASS-WITH-FIXES

---

## Summary

A well-constructed spine. 18 ADs, each a genuine divergence-preventer; the module DAG is test-enforced (AD-7), the hot-path numeric model is pinned (AD-10), determinism is a stated testable property (AD-11), and the §2.3 three-way reporting contract (AD-14) and the six Part-B invariants (AD-16) are near-perfect projections of the seal. The operational envelope is covered (offline batch, `cli.py`, outputs to `data/ticksim/runs/<run-id>/`, manifest = reproducibility).

The fixes below are concrete gaps a builder hits when implementing the **parity gate** — the artefact that is this build's own acceptance test (§4 kill criterion). None require an architectural rewrite. Findings 1–5 should be closed before build; 6–9 are cleanup.

---

## Checklist findings

### 1. Fixes the real divergence points, misses none? — MOSTLY, two gaps

Covered well: "now"/clock (AD-1), book mutation authority (AD-3, AD-9), order-lifecycle ownership (AD-8), fill-engine statelessness (AD-5), MBO record semantics (AD-9), numeric representation (AD-10), determinism/RNG (AD-11), output contract (AD-12), module split (AD-6), import graph (AD-7), strategy boundary (AD-2), mask handling (AD-13), preset freeze (AD-15), Part-B single-implementation (AD-16), Part-A method (AD-17), source abstraction (AD-18).

**Missed — F2 (bracket/OCO semantics):** the entire §A8.2 Part A parity sample is bracket-order trades (limit entry + TP-limit + SL-stop). How a bracket is represented in the `OrderIntent` log is the single highest-leverage modelling choice for Part A and **no AD touches it.** It also collides with AD-2 ("no intra-`simulate()` feedback from fills to new intents"): a live broker auto-cancels the sibling exit on fill; the simulator cannot do that without feedback, so `part_a.py` must either (a) model server-side OCO inside the book, or (b) pre-flatten each trade to entry + the exit that actually happened (from `trades.db` `exit_price`/`exit_reason`). Pick one, in an AD.

**Missed — F4 (warm-up tolerance):** Amendment 9 §A9.2 (a design input the spine `binds`) established that at a Mode-C intraday start ~0.3 % of C/M events reference orders added before the window. AD-9 pins `apply_event`'s A/C/M/T/F handling and transient-cross tolerance but never says it must tolerate a modify/cancel for an `order_id` it has never seen. Builders diverge here (raise vs silently drop).

### 2. Every AD's Rule enforceable, and prevents its stated divergence? — MOSTLY

Enforceable: AD-1 (greppable — no `time.time`/`datetime.now`), AD-4/AD-7 (import-graph walk test, explicitly named in AD-7), AD-10 (typed int fields + review), AD-11 (run-twice byte-compare), AD-13 (hard-fail test), AD-16 (shared assertion module, inspection).

**F3 — AD-15 has a twin that is missing.** AD-15 correctly makes the model presets frozen, seal-cited, module-level constants. The **parity thresholds are the same kind of object** — seal-frozen, decision-bearing — and get no equivalent rule: MAE ≤ 1.0 tick, p90 ≤ 2.0 ticks, signed bias ≤ ±0.25 tick, `N ≥ 28`, the `PASS = PartA AND PartB` verdict, and the 3-cycle / 15-working-day kill criterion (§A8.2, §4). Add an AD: these live as named constants in `parity/gate.py` with seal-section comments, and changing a value requires a seal amendment — exactly parallel to AD-15. As written a builder can hardcode `1.0` uncited, or get `0.25` subtly wrong, with nothing to catch it.

AD-15 itself is also only enforced "socially" — worth a unit test asserting `PRIMARY`/`OPTIMISTIC` field values equal the seal literals.

### 3. Could anything under Deferred let two units diverge? — NO

`pyproject.toml databento entry` is deferred as "a build task" while Amendment 9 §A9.5 says it "must be added before the simulator build" — acceptable, and the Stack table already pins `0.85.0` so the value is unambiguous. Multi-instrument is safe (book is per-`instrument_id`, AD-3 + A9.3). Own-impact / variable-latency are seal-§2.2-out-of-scope with a clean seam (new `SimConfig` field + amendment). Grid-cache is held behind the AD-18 protocol. Nothing here can split two units in *this* epic.

### 4. Named tech verified-current and pinned? — YES

| Name | Spine | pyproject | Check |
|---|---|---|---|
| Python | ^3.11 | ^3.11 | ✓ |
| pydantic | ^2.0 | ^2.0.0 | ✓ |
| databento | 0.85.0 (exact) | — (to be added) | ✓ **0.85.0 is the latest on PyPI**; installed in `.venv` and end-to-end tested per Amendment 9. Exact pin is correct for a wire-format-sensitive vendor SDK. |
| pytest | ^7.4 | ^7.4.0 | ✓ (installed is actually 9.0.2 — a repo drift, not the spine's problem) |

**F1 (also a checklist-2 / checklist-7 issue): the one dependency question that matters is left contradictory.** AD-4: "Its only new third-party dependency is `databento`." Structural Seed, book internals: "candidate `sortedcontainers.SortedDict` per side per instrument" — and `sortedcontainers` is **not** in `pyproject.toml` and **not** installed. This is the most performance-critical decision in the build (Amendment 9 named the naïve `max(dict)` as the *measured* bottleneck). Resolve one way: add `sortedcontainers` to AD-4's allow-list and the Stack table, or mandate a specific stdlib structure (`bisect` over a sorted list of price levels, hand-rolled) and state the complexity budget it must hit.

### 5. Ratifies rather than contradicts project-context.md? — MOSTLY

Ratifies: Pydantic models (AD-12, conventions table), mypy strict with no new ignore-override for the new package, black-88, `tests/unit/test_ticksim_*.py` + `tests/integration/`, `test_<module>.py` naming, package-per-capability (`src/ticksim/` is a clean new package), `|` unions / `list[T]`. The deliberate rejection of the repo's async-queue pipeline pattern is correct for an offline batch tool and is made explicit in the Design Paradigm — good.

**F8 (minor):** two silent overrides. (a) project-context mandates America/New_York tz-aware `datetime`; the spine uses `int64` UTC ns everywhere (correct for a DES) but never says it is deliberately not following that rule inside `src/ticksim/`, with ET appearing only at mask construction. One sentence fixes it. (b) The repo convention is a single `src.<pkg>.exceptions` module; AD-6's module list has no `exceptions.py` and the typed exceptions (`MaskViolation`, `BookInconsistency`, `IntentLogError`) have no assigned home (`InvariantViolation` is placed, in `parity/invariants.py`).

### 6. Covers the seal — §2.1 / §2.2 / §2.3 + Amendment 8 gate? — MOSTLY

| Seal item | Covered | Where |
|---|---|---|
| §2.1 back-of-queue / time-priority queue | ✓ | AD-5, AD-15, `fills.py` two models |
| §2.1 250 ms / 50 ms latency | ✓ | AD-15 presets |
| §2.1 passive-fill rule (cum. volume at price since arrival > queue-ahead) | ✓ | AD-5 (names it explicitly) |
| §2.1 marketable book-walk, partials | ✓ (implied) | AD-5 `decide()`, AD-16 invariant 5 |
| §2.1 adverse-selection marker (same-side move through price within 1 s) | ⚠️ partial | AD-12 has the field; the 1 s rule + who computes it is unpinned — **F7** |
| §2.1 cancel/replace: 1 latency hop; priority lost on price change, kept on size-decrease | ✗ | **F7** — the priority-retention rule appears in no AD |
| §2.1 fees $0.72 + commission (default $0.58 RT) | ⚠️ | AD-15 says "configurable commission" but does not cite the $0.58 seal default as the preset default — **F7** |
| §2.1 own-impact = none / ±1-tick stress | ✓ | AD-14 (b) |
| §2.2 not-modelled list | ✓ | "not built"; halts via AD-13 mask; clean seam for later |
| §2.3 three-way reporting | ✓✓ | AD-14 — 2 runs / 3 reports / decision on (a) must-hold-under-(b) / (c) context. Exemplary. |
| §A8.2 Part A (real-fill replay, sample, tolerances, N≥28, verdict) | ⚠️ | method in AD-17; **tolerances/verdict/kill-criterion constants unpinned — F3**; **bracket representation unspecified — F2** |
| §A8.2 Part B (≥1000 synthetic, 6 invariants, 100 %) | ✓✓ | AD-16 — all six listed, single implementation, typed `InvariantViolation`, shared by gate + unit tests |
| §A8.2 combined verdict (A AND B) | ✓ | `gate.py` responsibility (Structural Seed); not in an AD but unambiguous |
| §4 kill criterion (3 cycles / 15 days) | ⚠️ | mapped to `gate.py`; tracking mechanism unstated (acceptable — largely process) — **F3** |
| §5 integrity (ts monotonic, transient-cross tolerance, A/C/M/T/F) | ✓ | AD-9 + `gate.py` preflight; correctly incorporates the Amendment 9 §A9.3 refinement |
| A9.3/A9.4 `instrument_id` keying | ✓ | AD-3, capability map |
| A9.2 Mode-C warm-up tolerance | ✗ | **F4** |

### 7. Every dimension decided / deferred / open — nothing silent? — MOSTLY

Operational envelope is solid: run mode, host, entry points, output location, reproducibility mechanism all stated.

Silent dimensions (beyond F2/F3/F4 above):
- **F5 — `valid_intervals` producer.** AD-13 makes `SimRun` *consume* explicit `valid_intervals` and hard-fail outside them (good), but nothing in the module decomposition *builds* the RTH-minus-CME-maintenance-minus-halts calendar, and no source for the halt/holiday calendar is named. Real work, unassigned.
- **F6 — `OrderIntent` is an unversioned cross-unit contract.** AD-12 makes `OrderOutcome` frozen + versioned + sole consumer contract. `OrderIntent` is produced by *both* `parity/part_a.py` and the future H1 strategy and consumed by `sim.py`, and gets no equivalent frozen/versioned statement — schema drift between the two producers is possible. Its field set (side, type, limit price, submit_ts, cancel-target, replace params) is never enumerated.
- Minor / code-owned: `<run-id>` scheme; logging sink; which module transforms `intent` → latency-delayed `arrival` event (AD-1 mentions the delayed arrivals in the stream; AD-6 does not assign the transform — plausibly `events.py` or `sim.py`).

### 8. Terse build-substrate, or bloated? — TERSE (mild redundancy)

~245 lines for an epic spanning 8 modules + a `parity/` sub-package. States invariants and boundaries, not implementations — not a code mirror. **F9 (cosmetic):** the module decomposition is stated three times (Namespace map table, AD-6, Structural Seed); `[ADOPTED]` tags appear on 4 of 18 ADs with no legend, leaving the reader unsure whether unmarked ADs are less settled (doc status is `draft`). Trim one copy of the module list; add a one-line note on what `[ADOPTED]` means or drop the tags.

---

## Fix list

| # | Severity | Fix |
|---|---|---|
| F1 | blocking | Resolve AD-4 vs Structural Seed: either allow `sortedcontainers` (AD-4 list + Stack table) or mandate a named stdlib book structure + its complexity budget. |
| F2 | blocking | Add an AD deciding how bracket / OCO orders are represented in the `OrderIntent` log for Part A replay, consistent with AD-2's no-feedback rule. |
| F3 | blocking | Add an AD-15-twin: parity tolerances, verdict logic, and kill criterion are seal-bound named constants in `parity/gate.py` with seal citations; change requires an amendment. |
| F4 | blocking | AD-9: state that `apply_event` tolerates modify/cancel for an unseen `order_id` (Mode-C warm-up, per Amendment 9 §A9.2). |
| F5 | should-fix | Assign an owner for constructing `valid_intervals` (session calendar − CME maintenance − halts); name the calendar source. |
| F6 | should-fix | Give `OrderIntent` the AD-12 treatment — frozen, versioned, enumerated fields — since two units produce it. |
| F7 | should-fix | Pin the remaining §2.1 sub-rules in an AD: cancel/replace priority retention, adverse-selection 1 s marker + owner, commission default = seal $0.58. |
| F8 | minor | Note the deliberate int64-UTC-ns override of the repo's NY-tz `datetime` convention; assign a home for ticksim's typed exceptions. |
| F9 | cosmetic | Remove one of the three module-list restatements; add a legend for `[ADOPTED]` or drop the tags. |

## What is strong (keep)

- AD-14 — the three-way reporting contract is an exact, independently-reproducible projection of §2.3.
- AD-16 — six invariants defined once, typed exception, shared gate ↔ unit tests.
- AD-7 — permitted import edges enumerated + a graph-walk test to enforce.
- AD-10 / AD-11 — integer hot path and bit-for-bit determinism, both stated as testable properties.
- AD-3 + AD-9 — single book, single MBO-semantics function, transient-cross tolerance already reconciled with Amendment 9.
- Operational envelope and the `binds` front-matter tying every AD back to a seal section.
