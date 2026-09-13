---
title: 'Execute the portfolio PF improvement shortlist'
type: 'feature'
created: '2026-09-13'
status: 'done'
route: 'dispatch'
review_loop_iteration: 0
context: []
baseline_commit: '1800bf509b9c362b3395dc17f524c0fee65e2e01'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The operator wants the approved evidence-first sequence most likely to uncover a defensible improvement in pooled closed-trade net PF. Existing records do not yet prove a repairable execution loss, the proposed MIM profit-protection mechanism is untested, and commodity carry remains blocked by execution and data feasibility.

**Approach:** Implement one immutable, read-only research workflow that first reconciles MIM/GAP execution, then advances to a single MIM scheduled-mark profit-giveback feasibility study unless a current economic execution defect is proven, and finally produces a three-venue commodity-carry stop/go packet. Report net profit, sample size, exposure, concentration and drawdown beside PF; preserve all prior verdicts and protocols.

## Boundaries & Constraints

**Always:** Work only in the isolated worktree; read existing records and public official documentation; freeze hashes, source snapshots, runtime, definitions and failures; redact account identities; preserve ambiguous/missing evidence; use arm A, delay2, one contract and $2.24 round-trip costs for MIM; keep the portfolio decay monitor independent.

**Never:** Import or invoke live traders, brokers, collectors, repair tools, parity/backtest entrypoints or strategy simulators; access sealed holdouts; calculate alternative-strategy returns; select a threshold/window/classifier; contact vendors, purchase data, change services/config/sizing/orders, or copy raw account-bearing/DBN evidence into outputs.

## I/O & Edge-Case Matrix

| Scenario | Expected behavior |
|---|---|
| Exact broker order ID | Join first by ID; validate strategy, side and size; calculate broker net only with both legs and complete costs. |
| Missing/ambiguous causality | Keep unmatched candidates and return `INSUFFICIENT_CAUSAL_EVIDENCE`; never estimate recoverable dollars. |
| Current economic defect proven | Return `CURRENT_REPAIR_CANDIDATE`, write a repair specification and stop before later stages. |
| MIM active scheduled mark | Use only completed information through 10:00–15:30 ET; reversal mark belongs to the exiting leg. |
| Stop/open/EOD boundary | Exclude ambiguous stop-minute OHLC and all post-exit prices; retain known fills and actual ordering. |
| Carry venue | Topstep, TradeStation SIM and future self-funded paths receive separate evidence classifications; multiple park reasons may coexist. |
| Corrupt or unsafe input/output | Fail closed, preserve `failure.json`, seal the failed inventory and refuse overwrite/path escape. |

</frozen-after-approval>

## Code Map

- `research/mim_diagnostics/{analysis,artifacts}.py` — reuse accounting, containment, immutable inventory and independent-verification behavior; do not modify its frozen runs.
- `research/mim_lifecycle/analysis.py` — reuse validated held-interval and stop-censoring concepts; map trade IDs by day, contract and entry event, and charge only the incurred $1.12 entry cost at open marks.
- `research/mim_robustness/runs/20260912T151842-run-f8608e71fb` — canonical decisions/trades source; arm A delay2 contains 12 scheduled decisions per session and 801 trades.
- `/root/Silver-Bullet-ML-BMAD/data/{mim_nb,gap_fade}` — mutable operational evidence read from the original checkout and hash-bound without copying raw account IDs.
- `/root/Silver-Bullet-ML-BMAD/{data/commodity_curve,docs/commodity-curve-*,_bmad-output/specs/spec-commodity-curve-carry}` — ignored external evidence; normalize facts and hashes only.
- `tools/portfolio_decay_shadow.py` and its log — report observation timestamp/coverage only; never call, edit or interpret efficacy.

## Tasks & Acceptance

- [x] Add `research/pf_improvement/` CLI with `audit`, `run`, and `verify`; create fresh sealed `runs/` children, manifest-bound defaults, normalized snapshots and standalone Markdown/HTML reports.
- [x] Build exact-first MIM/GAP execution event and round-trip ledgers, chain/coverage diagnostics, configuration-era classifications and the three-state execution gate.
- [x] Build the 6,673 expected active MIM decision-mark rows with causal current P&L, running MFE/MAE, giveback, prior-mark change, coverage and retrospective outcome labels; audit V1/V2, neutral-exit and lifecycle overlap without candidate returns.
- [x] Build the 12-root × 3-venue carry matrix, prepared unsent source questions, official-source evidence inventory and overall `PARK_ACCOUNT`/`PARK_DATA`/`PARK_POWER` or source-clarification verdicts.
- [x] Add independent verification, charts, corruption/redaction/gating fixtures and full-run evidence; update package README/results.

Acceptance: Given the pinned sources, `run` either stops on a proven current repair candidate or completes the later feasibility stages. MIM totals reconcile to 1,323 sessions, 801 trades and $21,889.76 within `1e-8`; 71 CAT_STOP, 723 EOD and seven reversal labels are preserved. Re-running from unchanged inputs yields identical substantive files, and `verify --run` rejects any input, inventory, ledger or privacy drift.

## Implementation Notes

- User explicitly approved the gated three-stage plan and chose all known venue paths. Branch `research/mim-diagnostics` was merged with current `origin/main` at merge commit `1800bf509b9c362b3395dc17f524c0fee65e2e01` before implementation.
- Investigation found GAP supports gross fill comparison but lacks costs/timestamps/quotes; MIM local fill logging requires saved-export ID corroboration and has historical chain breaks. Expected Stage 1 verdict is evidence-driven, not hardcoded.
- Existing carry evidence implies Topstep unavailable for overnight carry, TradeStation SIM validation-only, and self-funded requirements-only; the implementation must recompute classifications from bound evidence.
- Completed run `research/pf_improvement/runs/20260913T022924-run-26b118774d` independently verifies 6,673 marks, 801 trades and the 12 x 3 carry matrix. A second unchanged-input run matched all 30 substantive artifact hashes.
- Verification passed 218 focused and related tests after review fixes. Black passed for the new Python files; Flake8 passed with Black-compatible `E501`, `E203`, and `W503` exclusions. Two existing Pydantic deprecation warnings remain outside this package.

## Review Triage Log

| ID | Layer | Verdict / route | Evidence |
|---|---|---|---|
| V1 | verification-gap | medium / patch | The mark test and verifier prove timestamps and algebra but do not independently prove which held bars feed MFE/MAE; a producer regression could admit pre-entry or post-mark extremes. |
| V2 | verification-gap | medium / patch | `STOP_REQUIRED` is only a subset and the verifier returns immediately, so a repair-candidate run containing later-stage files would still verify. |
| V3 | verification-gap | medium / patch | The required PF, exposure, drawdown and concentration fields are emitted, while verification checks only sessions, trades and net. |
| V4 | verification-gap | medium / patch | Carry verification accepts any status in the enum and does not enforce the three approved venue classifications. |
| B1 | blind-hunter | false / reject | The workflow is deliberately bound to a fixed snapshot whose zero causally decomposable rows require the insufficient-evidence state; the pure gate covers all three states and source drift is refused. |
| B2 | blind-hunter | medium / patch | `classify_gate` does not require an explicit finding that intended behavior is unchanged, although the frozen gate definition does. |
| B3 | blind-hunter | false / reject | No current row is causally decomposable, so the gross difference cannot trigger a repair candidate; incomplete-cost GAP rows are explicitly excluded. |
| B4 | blind-hunter | false / reject | Missing broker coverage is represented by the 20 local-only MIM rows and incomplete GAP fields, forcing `complete_current_coverage=False`; pinned-source omissions cannot silently enter because hashes are fixed. |
| B5 | blind-hunter | low / reject | Calendar eras are descriptive labels for this reviewed fixed snapshot and do not determine P&L; adding deployment-manifest machinery would exceed the negligible present harm. |
| B6 | blind-hunter | false / reject | The seven saved pairs are adjacent flat-to-flat pairs in the pinned export; any account/contract/size/side mismatch raises rather than being silently paired. |
| B7 | blind-hunter | medium / patch | The lineage table is authored prose even though the prior V1/V2, neutral-exit and lifecycle records are pinned; the distinctness verdict should validate those bound records. |
| B8 | blind-hunter | false / reject | The report explicitly describes distributions “at observed marks”; the approved artifact is a scheduled-mark ledger, so duration weighting is the stated estimand rather than an independence claim. |
| B9 | blind-hunter | medium / patch | `daily.csv` drives drawdown and day concentration without a day-level reconciliation to the selected trade ledger. |
| B10 | blind-hunter | medium / patch | Exact final mark presence does not reject a duplicate or missing minute inside a held interval, which can bias excursions and coverage. |
| B11 | blind-hunter | medium / patch | Output-only monotonicity and algebra do not independently establish source-bound MFE/MAE or the permitted held interval. |
| B12 | blind-hunter | medium / patch | Execution verification checks output columns and counts but does not independently rejoin emitted identifiers, prices and costs to the pinned source rows. |
| B13 | blind-hunter | medium / patch | Audit and stopped-run early returns skip semantic checks of their normalized inventory, coverage and gate evidence. |
| B14 | blind-hunter | false / reject | The pinned evidence cannot reach the stop branch; the generic template is not presented as a completed concrete repair and no successful run can currently expose that outcome. |
| B15 | blind-hunter | medium / patch | The carry matrix loads official evidence but does not validate that its classifications follow from the bound venue facts and development evidence. |
| B16 | blind-hunter | medium / patch | Counts and unique-label counts do not prove the exact 12-root by three-venue Cartesian product or the declared park-reason set. |
| B17 | blind-hunter | medium / patch | Several central verdict, evidence, question, lineage and report relationships are only protected by inventory hashes, so producer contradictions can still be sealed. |
| B18 | blind-hunter | low / reject | The normalized, timestamped and hash-frozen official-fact record preserves exactly what was observed and links the public primary pages; archiving mutable page bytes is outside the present metadata packet. |
| B19 | blind-hunter | medium / patch | Raw ProjectX order and fill identifiers are exported even though equality can be preserved with deterministic pseudonyms; privacy scanning covers account IDs only. |
| E1 | edge-case-hunter | low / patch | Sealed files are read-only but the run child directory remains writable, allowing additions that invalidate the seal until detected. |
| E2 | edge-case-hunter | medium / patch | `completion.json` is created before chmod finishes; a chmod failure then suppresses `failure.json` because completion already exists. |
| E3 | edge-case-hunter | false / reject | Every pinned CSV source has a chain column; the saved broker export is separately and explicitly `NOT_APPLICABLE`, and changed inputs are rejected before analysis. |
| E4 | edge-case-hunter | false / reject | The only pinned duplicate business key is the registered 2026-06-25 GAP scar and is classified `KNOWN_SCAR`; no unregistered duplicate reaches this run. |
| E5 | edge-case-hunter | false / reject | All pinned GAP directions are L/S and malformed drift is rejected by the input hash before reconciliation. |
| E6 | edge-case-hunter | false / reject | Pinned GAP IDs are non-null, distinct and unreused; changed evidence cannot enter this manifest-bound workflow. |
| E7 | edge-case-hunter | medium / patch | Held-interval coverage does not assert a complete unique one-minute grid, so missing or duplicate bars could alter excursions. |
| E8 | edge-case-hunter | false / reject | The pinned 801-trade ledger contains zero break-even trades, and source drift is refused. |
| E9 | edge-case-hunter | medium / patch | `verify` does not recompute the gate from its evidence booleans or reject unknown/conflicting states. |
| E10 | edge-case-hunter | medium / patch | The summary verifier does not independently recompute the required PF, exposure, drawdown and concentration values. |
| E11 | edge-case-hunter | medium / patch | The verifier does not enforce every expected root/venue pair exactly once. |

All accepted patch findings were resolved. No findings were deferred.

## Verification

- `.venv/bin/python -m pytest tests/unit/pf_improvement tests/unit/mim_diagnostics tests/unit/mim_lifecycle tests/unit/test_gap_fade_fills.py tests/unit/test_mim_reconcile.py tests/unit/test_mim_nb_reconcile_commingling.py -v`
- `.venv/bin/python -m research.pf_improvement audit`
- `.venv/bin/python -m research.pf_improvement run`
- `.venv/bin/python -m research.pf_improvement verify --run <completed-run>`
