---
title: 'Evidence-backed portfolio PF improvement shortlist'
type: 'chore'
created: '2026-09-12'
status: 'done'
route: 'oneshot'
review_loop_iteration: 0
context: []
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The operator wants work that can raise portfolio net profit factor. The preceding party discussion proposed a ranked shortlist comparing recoverable execution losses, one loss-reduction mechanism, and one economically distinct strategy contribution. MIM historical losses alone do not identify the largest portfolio opportunity, and monitoring is not demonstrated PF uplift.

**Approach:** Produce an evidence-backed shortlist with explicit ranking rationale, potential benefit and damage, falsifiers, prerequisites and a concrete next deliverable for each route. Use pooled closed-trade net PF as the default requested objective, keep live/SIM/paper/unknown and historical evidence separate, and preserve all closed/underpowered verdicts. Reuse the verified unchanged MIM payoff/excursion records and inspect existing execution/candidate evidence plus read-only portfolio ledger provenance. Make no new strategy test, candidate-return computation, power verdict, filter selection, threshold sweep, portfolio allocation, broker request, collector/service change or holdout access. Rank research actions by evidence readiness and decision value, not fabricated PF uplift. Retain the existing prospective A/B, FOMC and other seals unchanged.

</frozen-after-approval>

## Implementation Notes

- Existing clean research/mim-diagnostics worktree; no original source edits, merge or push. Reuse the already-rendered bmad-build oneshot workflow.
- New report research/mim_diagnostics/PF_SHORTLIST.md, compact timestamped evidence and attribution CSV under evidence/, and README link. Reversible documentation/evidence-only footprint; no new callable API.
- Read-only SQLite transaction inventories row provenance/modes/metadata markers without calculating prospective efficacy. Existing diagnostics trades.csv supports sampled excursion descriptions only; no trade receives a new exit or return.
- Independent agents inspect existing execution evidence and distinct strategy feasibility in the current original checkout, preserving exact source hashes and limitations. Their findings will be checked and incorporated rather than treated as authority.
- Verify source inventories, CSV partitions and net PF arithmetic; review documentation independently, resolve actionable findings and commit. No application tests required if no application code changes.
- Investigation found no verified current execution-loss recovery pool: GAP saved SIM difference is adverse by $46.50, while seven MIM saved fill comparisons are favorable by $47.50 before complete commissions. Prior historical fixes must not be sold as remaining uplift.
- Existing MIM trade partitions: 298/339 losers exceeded recorded round-trip cost in sampled favorable excursion; 408/462 winners experienced sampled adverse gross excursion. These are overlapping descriptive states, not an exit rule or proof of separation. Prior neutral-sample exit research is explicitly disclosed.
- Distinct candidate is the already-specified diversified commodity curve carry, HOLD-DATA/unsealed. Send-time mechanics are already complete; provenance, calendars, account and power remain. TSC/TSMOM/COT FAIL and XSMOM/VRP UNDERPOWERED verdicts preserved.
- Wrote report, three evidence JSONs and CSV partition artifact; added README link. Independently verified all 16 partition rows/fields against 801 existing trades, exact PF arithmetic, 20 source hashes and 15 document links. No application code or strategy tests changed.

## Review Triage Log

The oneshot workflow's Blind Hunter review found six actionable documentation gaps. All were checked and patched; none deferred.

| Finding | Verdict / checked evidence | Resolution |
|---|---|---|
| Missing exit described as unmatched existing exit | Low: source says an entry lacks its corresponding exit | Corrected to expected exit fill missing |
| Complete trade paths overstates minute coverage | Medium: stop intervals exclude unavailable post-exit extrema | Describe sampled paths with gaps and require missingness preservation |
| Winner MFE count includes terminal observation | Medium: all winners necessarily exceed costs at exit | Label full-life samples and state terminal-fill/tautology limitation for favorable winners and adverse losers |
| Carry position type and TSC distinction ambiguous | Medium: canonical protocol holds outrights, not calendar spreads | State outright high/low carry positions and cross-sectional versus absolute-deadzone distinction |
| PF trade grouping unspecified across rolls | Medium: splitting campaigns changes PF without changing profit | Require frozen roll/partial-close/quantity/campaign grouping; add general cross-product condition and zero-loss limitation |
| Transformation recipe missing | Medium: hashes and high-level definitions did not fully reproduce reducers | Add PF_REPRODUCTION.md with exact source/group/formula/serialization recipe, metadata reductions/missingness and mutable-observation replay limitation |

## Verification

- No new trading code, strategy returns, power verdict or prospective efficacy calculation. Accounting verified at absolute $1e-8; net PF verified at $1e-12.
- Final linked documents/recipes checked; final report, README, recipe and three underlying evidence artifacts hash-bound in verification JSON.
- Independent reviewer confirmed all six corrections in a bounded follow-up; no unresolved findings.
