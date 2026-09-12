# Portfolio PF improvement: ranked research shortlist

**The first task should be a bounded execution-shortfall reconciliation, followed by one MIM profit-protection feasibility study. Diversified commodity curve carry is the distinct-strategy candidate, currently parked behind data and account feasibility.** This ranks the value of the next evidence-producing action, not expected PF uplift: none of the three has demonstrated an improvement.

The objective is pooled **closed-trade net PF**, with net profit, sample size, exposure and drawdown reported alongside it. The operator's latest instruction prioritizes PF over the previously proposed Sharpe benchmark; that design remains inactive and unchanged. Daily portfolio PF is a different statistic and cannot substitute silently.

## Ranked routes

| Rank | Route and current disposition | Why this next action has value | What could overturn the ranking |
|---|---|---|---|
| 1 | Execution: reconcile existing MIM/GAP intent, fills and complete costs; **MEASURE, no repair selected** | A small, bounded comparison can distinguish genuine economic loss from reporting errors and beneficial fills before engineering effort is spent | If no repeatable, causally repairable adverse shortfall is found, close the repair thesis and move research effort to route 2 |
| 2 | MIM: one profit-protection hypothesis; **DIAGNOSE, no exit rule selected** | The existing recorded sampled paths, with documented intraminute gaps, permit a causal-state feasibility study without purchasing data; losses are substantial, but protecting them may sacrifice the rare winners | Lack of winner/loser separation, duplication of an already-studied exit, or unattainable validation power ends this route before an exit sweep |
| 3 | Diversified commodity curve carry; **HOLD-DATA / conditional feasibility** | It proposes a distinct source of returns and is already specified, but neither tradeability nor PF contribution is established | It moves up only if account implementability and credible causal history can be established without disproportionate acquisition work |

There is no numerical opportunity score, invented uplift percentage or proposed allocation. This is a practical sequencing judgment based on current evidence readiness and bounded next-work cost. No claim is made that MIM is the portfolio's largest realized loss source.

## 1. Execution: establish a repairable loss before proposing a repair

Existing evidence contains one concrete adverse example and a small favorable comparison:

- **GAP, August 6, broker SIM:** modeled gross +$646 versus fill-derived gross +$599.50, a **−$46.50** signed difference. Entry order `965760604` and TP `965760598` support the price reconciliation. It is not yet evidence of avoidable loss: the modeled opening mark may precede a feasible decision and fill. One observation does not establish recurrence. [Incident evidence](/root/Silver-Bullet-ML-BMAD/_bmad-output/ledger_incident_20260806_gap_fade.md)
- **MIM, saved September 10 comparison:** seven matched round trips produced fill-gross −$293.50 versus reference-ledger −$341, a **+$47.50 favorable** difference. Mean absolute difference of $13.64 is not a recoverable loss estimate. One expected exit fill is missing, and separate commissions are missing from the quoted fee text, so this is not broker-net PF. [Saved reconciliation](/root/Silver-Bullet-ML-BMAD/_bmad-output/planning-artifacts/research/academic-lit-mim-nb-live-performance-decision-2026-09-10/imports/local-evidence.md)

Earlier MIM stale-contract, DLL and stop-reconciliation problems appear repaired in inspected current source; this audit did not inspect running process versions. The July ledger correction from −$500 to −$165 improved the record, not realized economic profit. Likewise GAP missing/duplicate ledger rows were reporting incidents. Thursday's startup repair is already documented; its new eligible accrual begins September 17 and must retain its own look rules. YANK execution evidence remains conditional/HOLD_VALIDATION. Do not claim these historical issues as currently available gains.

**Next deliverable:** one frozen, existing-record reconciliation for MIM and GAP, separated by execution venue and account/configuration era. Join decision/intent → order acknowledgments → entry/exit fills → complete fees/commissions. Keep every unmatched or ambiguous record. Decompose signed shortfall into decision-to-arrival movement and arrival-to-fill costs only when the timestamps and executable quotes support that distinction. Report observed adverse differences, favorable differences, record-only errors and known repaired faults separately. No hypothetical better fill and no new broker request is part of this bounded step.

**Potential contribution:** recover an independently demonstrated recurring implementation loss under unchanged intended behavior. **Potential damage:** an allegedly cheaper order could miss winners, incur adverse selection or delay a protective exit. **Falsifier:** costs and causally available quotes explain the differences, or complete joins reveal no recurring repairable adverse mechanism. Stop at insufficient evidence if missing records prevent attribution; do not turn missing data into an assumed repair budget.

## 2. MIM: investigate profit protection without assuming reversals identify losers

The verified historical baseline has **462 winners, 339 losers, net PF 1.2856688**, winning dollars $98,516.12 and losing dollars $76,626.36. This is the unchanged one-contract historical model, not current live portfolio performance.

A fresh descriptive partition of the already-recorded fill/close excursions gives:

| Existing outcome | Count | Full-life sampled favorable excursion exceeding the recorded round-trip cost | Sampled adverse gross excursion |
|---|---:|---:|---:|
| All losing trades | 339 | 298 (87.9%) | 339 |
| EOD losing trades | 261 | 231 (88.5%) | 261 |
| Catastrophe-stop trades | 71 | 61 (85.9%) | 71 |
| Reversal-label trades | 7 | 6 (85.7%) | 7 |
| All winning trades | 462 | 462 | 408 (88.3%) |

The favorable measurement exceeds the existing $2.24 accounting cost; it is not a selected take-profit threshold. A sampled close/fill is not necessarily an executable alternative exit. Adverse excursions are gross price movements, excluding the automatic initial fee debit. All measurements inherit the diagnostics' stop-minute exclusions and timing uncertainty. Terminal exit fills are included: all winners mechanically exceed their costs at exit, so the 462 favorable winner count is not evidence of an earlier protective opportunity. Likewise a losing terminal fill can supply the adverse maximum. Preserve missingness in any later state study; do not fill uncertain intervals with post-exit bars.

**What this establishes:** many losers had some earlier favorable movement, while most winners also endured adverse movement. The coarse states overlap; neither “was profitable” nor “went against the entry” separates outcomes. The two maxima do not reveal the ordering of those movements. These numbers do not establish a profitable protection rule.

The single proposed mechanism is **persistent loss of an established favorable price path**, assessed only at existing completed-minute decision times, as a possible reason to exit before EOD. It remains a hypothesis family, not an executable threshold. The next study must establish whether causally observed path history contains useful separation beyond a simple neutral-sample exit. That simpler exit is already exposed research: published exit behavior reduced historical mean P&L by $4.83/session relative to A. It must not be presented as a new discovery or rerun as a fresh test. [Prior mechanism comparison](/root/Silver-Bullet-ML-BMAD/research/mim_comparison/RESULTS.md)

**Next deliverable:** a fixed descriptive ledger at the baseline's existing half-hour decision marks, containing only information available at each mark: position age, accumulated favorable/adverse excursion, current marked P&L and the relevant frozen decision state. Preserve every opportunity and the subsequent original outcome, including the largest winners. First audit lineage against the earlier V1/V2 and published-neutral-exit work. Then describe path overlap and whether a genuinely distinct hypothesis remains. Do not sweep persistence lengths, choose favorable windows, move exits, fit a classifier or calculate candidate returns in that deliverable.

**Potential contribution:** reduce losing dollars while retaining sufficient winning dollars. **Potential damage:** truncate the rare sessions sustaining total profit, create repeated exit/re-entry costs or replace catastrophic losses with many smaller losses. In the simplified case where only winning dollars C and losing dollars A are removed, PF improves only if `C < baseline_PF * A`, with a positive remaining loss denominator. Net profit improves only if `C < A`. Consequently a PF improvement can still reduce profit; both must be disclosed. These are algebraic conditions, not calibrated trade thresholds or forecast savings.

**Falsifier:** causal path descriptions fail to distinguish outcomes, the proposed mechanism duplicates already exposed research, or a subsequent appropriately powered, preregistered unseen-data test fails net-PF improvement after costs and winner retention are accounted for. No new strategy test occurs here. Any decision rule requires its own derived settings, suitable power gate and committed preregistration before testing; ambiguous evaluation evidence is FAIL.

## 3. Distinct candidate: diversified commodity curve carry

The existing unsealed specification ranks simultaneous nearby/deferred commodity curves monthly, taking long outright futures positions in high-carry roots and short outright futures positions in low-carry roots across sectors. The near/deferred pair measures the signal; this is not a long-near/short-far calendar-spread trade. The proposed mechanism concerns curve/inventory conditions rather than intraday Nasdaq continuation or crypto weekday behavior. Economic distinction is not proof of low correlation, positive expectancy or portfolio PF uplift.

Mechanics fixtures already pass. January–August 2025 staging contains **95,537 settlement-vintage records across 214 instruments and 12 roots**. The subsequent send-time reconstruction is complete; repeating it is not the next task. Original-publication lineage, revisions, effective delivery/session calendars, inactive-contract coverage, executable costs and account constraints remain unresolved. Eight months of development staging is not a powered strategy sample. [Pilot acceptance](/root/Silver-Bullet-ML-BMAD/docs/commodity-curve-pilot-acceptance-20260907.md), [completed send-time audit](/root/Silver-Bullet-ML-BMAD/docs/commodity-curve-send-time-audit-20260907.md)

**Next deliverable:** a stop/go feasibility packet: establish the intended account's product/overnight/integer-sizing constraints; consolidate remaining source questions using the existing unsent inquiry; specify adequate-history and carry-specific power requirements. Recommend parking or a separately authorized source clarification. No vendor message, purchase or new market-data acquisition occurs here. A fractional research result cannot imply an executable two-sided book. [Existing protocol](/root/Silver-Bullet-ML-BMAD/_bmad-output/specs/spec-commodity-curve-carry/experiment-protocol.md), [unsent inquiry](/root/Silver-Bullet-ML-BMAD/docs/commodity-curve-databento-fields-inquiry-20260907.md)

**Potential contribution:** a stronger net trade stream at a feasible allocation. With additive unchanged trade accounting and positive loss denominators, a new component raises pooled-trade PF only when its own net PF exceeds the current aggregate; allocation changes its contribution magnitude. More generally compare `W_new * L_portfolio > W_portfolio * L_new` when the combined loss denominator is positive. A zero-loss finite sample is not proof of a reliable infinite PF. Before any comparison, freeze how rolls, partial closes, contract quantities and position campaigns form closed trades, and assign all costs consistently. Splitting one economic position at rolls can change PF without changing total profit; show any alternative grouping only as a disclosed sensitivity, never select the grouping for a higher PF. **Potential damage:** costs, integer rounding, capital/overnight requirements and common losses can erase the expected benefit. **Falsifier:** the intended account cannot implement the book, causal history/power cannot be established, or a future authorized independent comparison does not improve the declared portfolio objective.

Preserve TSC-1, TSMOM-1 and COT FAIL verdicts, and XSMOM-1/VRP-1 UNDERPOWERED verdicts. This separately documented cross-sectional ranking construction differs from TSC-1's failed absolute-deadzone time-series timing rule; it is not a rescue or resweep of TSC-1. Any hypothetical holdout outcomes printed in older result documents are exposed evidence regardless of a contradictory “holdout not spent” label. No protected data was opened.

## Portfolio accounting boundary

The timestamped SQLite inventory found MIM 23 realtime/live-labelled rows, GAP 26 realtime/SIM rows, and YANK five realtime/unknown-mode rows. One of those YANK rows also has a backfill metadata marker. Other rows include paper, unknown-mode and backfilled histories; Thursday uses a separate ledger. `realtime` alone does not establish real-money execution, experiment eligibility or complete broker-net P&L. No prospective P&L was queried and no current portfolio PF was computed.

A later contribution table must reconcile fills, fees, corrections, duplicate identity, account epochs, strategy versions and each protocol's eligibility before pooling. Keep live, SIM, paper, unknown and backtest cohorts separate. Preserve the existing portfolio shadow monitor and its native-metric rules; this shortlist adds no monitor trigger and changes no allocation. A monitoring tool's correctness or a repaired report is not a measured economic PF improvement.

## Evidence and verification

- [Timestamped provenance and historical snapshot](evidence/pf-opportunity-snapshot-20260912.json): single read-only SQLite transaction, explicit SQL, metadata markers and existing-baseline hashes.
- [Exact transformation recipe](evidence/PF_REPRODUCTION.md): grouping, formulas, parsing/missingness behavior and limits of replaying a mutable database snapshot.
- [Descriptive excursion partitions](evidence/pf-excursion-partitions-20260912.csv): each exit/outcome partition reconciles to the unchanged historical total; ALL rows are margins, not extra trades.
- [Route evidence and 20 independently checked source hashes](evidence/pf-route-evidence-20260912.json): historical execution observations, preserved verdicts and candidate limitations.

Only reports and evidence artifacts were written in the isolated worktree. No strategy return was recalculated under a new rule; no power verdict, parameter, service, order, sizing or deployment changed. Large original source runs remain local, hash-bound evidence. No numerical PF-uplift forecast is supportable from this shortlist.

[Independent verification](evidence/pf-shortlist-verification-20260912.json) recomputed all 16 partition rows from the 801 hash-bound existing trades, checked PF arithmetic and 20 source hashes, and validated document links. This follow-up changes documentation and evidence only; no application test suite or new strategy power script was run.
