---
title: 'MIM Profit-Giveback Experiment'
type: 'feature'
created: '2026-09-13'
status: 'done'
route: 'dispatch'
review_loop_iteration: 1
baseline_commit: '3ffc82a4a9b81602e954b8a3ac62848f343d3ec9'
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The only surviving MIM PF hypothesis is that large retreat from maximum favorable excursion at an existing half-hour decision mark may predict poor remaining payoff. All history through 2026-08-27 informed this hypothesis, so it cannot validate the rule.

**Approach:** Build a research-only, immutable, firewalled program that first tests power without aligned signal/outcome evaluation, derives one giveback-ratio threshold from an exposed-history sweep only if powered, and evaluates the frozen rule once on later observations.

## Boundaries & Constraints

**Always:** Target prospective PF >= 1.40, net profit >= 90% of baseline, and a one-sided 95% paired-net lower confidence bound above zero. Stop at 500 eligible sessions or 30 months. Use arm A, delay 2, one contract, $2.24 round-trip costs, existing half-hour marks, and original event/re-entry ordering. Treat power failure, development failure, and insufficient prospective coverage as terminal results.

**Never:** Access sealed holdout, invoke brokers or collectors, modify live/configured behavior, report interim efficacy, add another strategy parameter, extend the horizon automatically, or infer deployment authorization.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Power | Bound diagnostic marks and bars | Mismatched-pairing MDE, power verdict, immutable report | Identity pairing is refused |
| Development | POWERED parent | Nine mechanically derived thresholds and deterministic selection | Non-powered parent or no qualifying rule stops |
| Collection | Committed frozen protocol plus future receipt-tagged bars | Immutable append-only observation chain without efficacy | Duplicate, late, pre-freeze, malformed, or changed inputs fail closed |
| Evaluation | Endpoint reached | One final paired verdict and reports | Early or incomplete evaluation is INCONCLUSIVE without efficacy |

</frozen-after-approval>

## Code Map

- `research/mim_robustness/engine.py` -- preserve baseline fill, stop, reversal, EOD, cost, and re-entry ordering.
- `research/mim_comparison/data.py` -- reuse contract validation, timezone normalization, and causal contract selection.
- `research/pf_improvement/marks.py` -- source the verified 6,673 causal decision marks and baseline reconciliation.

## Tasks & Acceptance

**Execution:**
- [x] `research/mim_giveback/` -- add immutable artifacts, overlay simulation, firewall, development, collection, evaluation, verification, CLI, and reports.
- [x] `tests/unit/mim_giveback/` -- cover accounting, firewall, event ordering, immutable chains, data corruption, horizon, and deterministic results.
- [x] `research/mim_giveback/runs/` -- execute and independently verify the first permitted stage.

**Acceptance Criteria:**
- Given the frozen source, when reconstructed, then baseline totals equal 1,323 sessions, 801 trades, $21,889.76 net, and PF 1.2856688 within $1e-8.
- Given an identity/aligned power pairing, when requested, then no statistic is produced.
- Given a prior stop/reversal/EOD event, when an overlay exit is pending, then original ordering wins and each execution side is charged once.
- Given any incomplete or corrupted lineage, when a command runs, then it seals failure evidence and produces no favorable result.
- Given a sealed parent run is committed or moved to another worktree, when consumed, then repository-relative bindings resolve in the current checkout and every content hash still verifies.
- Given prospective input, when collected or evaluated, then the committed protocol bytes and commit time, frozen source/rules/history, strict append order, receipt deadline, session endpoint, and horizon are enforced before efficacy is available.
- Given a power or final evaluation result, when independently verified, then the power construction is regenerated and the final gates are derived again from their recorded outcomes.

## Implementation Notes

- Work is isolated on `research/mim-diagnostics`; `origin/main` commit `303c29a` was merged before edits.
- Definitive power run `20260913T184305-power-de068376a1` returned terminal `UNDERPOWERED`; full independent replay verification run `20260913T184823-verify-c3e57d363c` passed.
- Three complete power runs reproduced the same verdict and per-threshold power table exactly; the definitive run snapshots all imported accounting/data dependencies.
- Downstream inventory run `20260913T185428-inventory-1934561818` correctly emitted sealed failure evidence rather than a threshold sweep; run `20260913T185432-verify-33d96073db` verified that terminal cause.

## Spec Change Log

- **Review loop 1:** The first review found that the implementation did not fully carry the existing immutability, causality, and independent-verification requirements into portable paths and prospective lineage. The acceptance section now names those checks explicitly. Re-derivation preserved the reconciled baseline, non-identity null, terminal gate, one-parameter rule, and absence of aligned candidate returns while avoiding mutable data/rules, retrospective receipts, non-append collections, and self-consistent-only verification.

## Review Triage Log

| ID | Verdict | Route | Evidence |
|---|---|---|---|
| V1 | high | bad_spec, fixed loop 1 | No test exercised the constructed shifted null and zero variance could report full power. `run_power` now validates a complete, nondegenerate grid and a deterministic test checks donor transfer and nonzero deltas. |
| V2 | high | bad_spec, fixed loop 1 | Cross-run repeats and conflicts were untested. Collection now requires a strict timestamp append and tests cover valid append, repeated history, and protocol mismatch. |
| V3 | high | bad_spec, fixed loop 1 | Only the all-true evaluation path was tested. Each gate now has an isolated failing case and semantic verification derives gates and verdict again. |
| V4 | medium | bad_spec, fixed loop 1 | Nested write modes were unchecked. Sealing tests every nested mode; verification rejects writable untracked runs while accepting Git-materialized, hash-identical evidence because Git does not preserve read-only modes. |
| V5 | high | bad_spec, fixed loop 1 | A tracked but locally resealed protocol could pass. Collection now compares the active completion hash to the exact committed Git blob. |
| B1 | false | rejected | The approved power translation explicitly retains 90% of baseline winning dollars; 90% of baseline net remains a separate development/prospective gate. The verdict now records that definition explicitly. |
| B2 | high | bad_spec, fixed loop 1 | Absolute worktree bindings made committed evidence disposable. Repository and operator paths are now encoded portably and resolved in the active checkout. |
| B3 | high | bad_spec, fixed loop 1 | Protocol commit lookup did not prove active bytes matched the commit. The completion blob is now compared byte-for-byte by hash. |
| B4 | high | bad_spec, fixed loop 1 | Artifact creation time alone allowed observations from before protocol commitment. Collection now uses the later of freeze time and Git commit time. |
| B5 | high | bad_spec, fixed loop 1 | Mixed pre-freeze rows were silently filtered. Any pre-freeze or pre-commit row now fails the whole invocation. |
| B6 | high | bad_spec, fixed loop 1 | Late receipt rows were merely made ineligible. Any receipt after its one-minute execution deadline now fails the batch. |
| B7 | high | bad_spec, fixed loop 1 | Prior-chain backfills could reorder history. Every new batch must begin strictly after the prior chain tip. |
| B8 | high | bad_spec, fixed loop 1 | Prospective stages read mutable history absent from lineage. Freeze records its path/hash and collect/evaluate manifests bind and recheck it. |
| B9 | high | bad_spec, fixed loop 1 | An empty selected-contract slice could pass `.all()`. Eligibility now requires exactly 390 observed selected-contract minutes. |
| B10 | high | bad_spec, fixed loop 1 | A 500th eligible session after deadline could trigger efficacy. Evaluation checks that session's final observed timestamp against the frozen deadline. |
| B11 | high | bad_spec, fixed loop 1 | Prospective execution depended on current `CONFIG`. Freeze now records all execution and inference rules, and evaluation uses those frozen values while checking frozen source hashes. |
| B12 | high | bad_spec, fixed loop 1 | Sweep accepted source/data different from the powered inventory. Inventory records both hashes and sweep rejects either mismatch. |
| B13 | high | bad_spec, fixed loop 1 | Prospective code ignored parent command types. Collection now requires `freeze`; evaluation requires `collect`. |
| B14 | high | bad_spec, fixed loop 1 | Power verification reused the stored null. It now rebuilds sessions, donor mappings, simulations, null deltas, power table, and verdict. |
| B15 | high | bad_spec, fixed loop 1 | Any failed run was called semantically verified. Only the demonstrated terminal inventory denial is now accepted as an expected failure. |
| B16 | high | bad_spec, fixed loop 1 | Invocation-resolution errors produced no artifact. The CLI now creates and seals failure evidence and removes partial favorable outputs. |
| E1 | high | bad_spec, fixed loop 1 | Inputs could drift after creation hashing. Successful seal rehashes every input after workflow consumption and refuses changed bytes. |
| E2 | high | bad_spec, fixed loop 1 | Same portability defect as B2; portable bindings and a worktree relocation test cover it. |
| E3 | high | bad_spec, fixed loop 1 | Same committed-protocol defect as B3; the active completion must equal its committed blob. |
| E4 | high | bad_spec, fixed loop 1 | Equivalent timezone spellings bypassed textual duplicate checks. Duplicate keys now use normalized UTC timestamps. |
| E5 | high | bad_spec, fixed loop 1 | Same silent pre-freeze filtering defect as B5; mixed or wholly early input fails. |
| E6 | high | bad_spec, fixed loop 1 | Same late-receipt defect as B6; late input fails before eligibility. |
| E7 | medium | bad_spec, fixed loop 1 | Identical prior observations were silently collapsed. The strict-append rule rejects all repeats. |
| E8 | high | bad_spec, fixed loop 1 | Future-dated receipts could count immediately. Receipt time must be no later than the observed collection time. |
| E9 | high | bad_spec, fixed loop 1 | Same mutable-history defect as B8; frozen history is in both protocol and run inputs. |
| E10 | false | rejected | All accepted future bars arrive within one minute of completion, so a previous session's bars are available before the next session opens; the history prefix is frozen. |
| E11 | false | rejected | Accepted future bars are globally timely and frozen historical bars predate commitment, so the rolling volatility window cannot contain the described late history. |
| E12 | high | bad_spec, fixed loop 1 | Same post-deadline endpoint defect as B10; a dedicated 30-month boundary test now returns `INCONCLUSIVE` without efficacy. |
| E13 | medium | bad_spec, fixed loop 1 | Collection could continue after 500 eligible sessions. Prior endpoints and batches extending beyond the endpoint are rejected. |
| E14 | high | bad_spec, fixed loop 1 | A non-freeze artifact could masquerade as protocol. Manifest command is now enforced. |
| E15 | high | bad_spec, fixed loop 1 | A non-collection artifact could reach evaluation. Manifest command is now enforced. |
| E16 | high | bad_spec, fixed loop 1 | Forged efficacy metrics/gates could pass verification. Paired daily values and all final gates/verdict are now derived again. |
| E17 | high | bad_spec, fixed loop 1 | Requiring read-only modes after Git checkout would invalidate committed evidence. Verification uses modes for untracked runs and content seals for Git-materialized runs. |
| E18 | high | bad_spec, fixed loop 1 | Re-evaluation under changed current rules could differ. Protocol source hashes and all execution/inference rules are frozen and enforced, making repeat evaluation deterministic. |
| E19 | high | bad_spec, fixed loop 1 | Tests omitted the 30-month boundary. A 500th session after deadline is now explicitly tested as no-efficacy `INCONCLUSIVE`. |
| E20 | high | bad_spec, fixed loop 1 | Same shallow power-verification defect as B14; the full shifted simulation is regenerated. |
| E21 | high | bad_spec, fixed loop 1 | Pre-create corruption or mid-workflow failure could leave no evidence or favorable partial files. Both paths now end in sealed failure-only evidence and have focused tests. |

## Design Notes

Power derives the candidate family from causal feature quantiles but estimates its null variability only under non-identity session pairings. A powered result permits aligned development; it does not spend or relabel exposed history as validation.

## Verification

**Commands:**
- `.venv/bin/python -m pytest tests/unit/mim_giveback -q` -- 32 passed.
- `.venv/bin/python -m pytest tests/unit/mim_robustness -q` -- 55 passed; `.venv/bin/python -m pytest tests/unit/pf_improvement -q` -- 17 passed.
- `.venv/bin/python -m research.mim_giveback power ...` followed by `verify --run ...` -- the first permitted run seals, reconciles, and is regenerated independently.
