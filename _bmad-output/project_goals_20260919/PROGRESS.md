# Six project goals — September 19, 2026

Engineering is implemented in `research/project_goals/`; evidence remains provisional or insufficient where listed. No strategy parameter, frozen comparison code, existing threshold, holdout access, account capital, trader lifecycle or historical journal was changed by this implementation. Deployment belongs to the lead after review and merge.

| Goal | Engineering delivered | Evidence status | Next action/date |
|---|---|---|---|
| 1. Scalable revenue | Broker audit, monthly cashflow/cost/capital sensitivity, attributable portfolio scenarios | Partial broker snapshot net −$1,082.58; no demonstrated earning capacity | Obtain complete statements/invoices before revenue or capacity decisions; review September 21 |
| 2. Returns relative to risk | Synchronized contract marks, actual/1:2/1:1/equal-exposure comparisons, drawdown/recovery, cost sweeps | Ten separately marked sessions; account reset register missing, so continuous equity/Sharpe withheld; no supported allocation change | Complete calendar and independent coverage/start positions, then out-of-sample comparison |
| 3. Trustworthy edge | Exposed standalone and paired power sweeps; committed-registration/freshness/hash gates | Standalone calibration available; paired corrected GAP evidence missing, confirmatory evaluation refused | Build paired corrected calibration; commit it before separate registration; no efficacy on insufficient fresh data |
| 4. Trustworthy execution | Independent persisted price provenance; fail-closed lookup; singleton completion+1s poll wrapper, health and lossless snapshot storage | Frozen checks pass; collector zero eligible sessions; full-session commissioning pending | Lead deploy after review/merge; inspect September 21 complete coverage; eligibility also requires prior-session context (possibly September 22 earliest); retain June 10, 2027 deadline |
| 5. Withdrawals | Configurable phase/payout/floor/cost model, direct $10k case, source/version packet | Account-specific applicability/invoices unresolved; automated Live unavailable; remote-hosting restriction unresolved | Obtain written account/provider confirmation before route decisions |
| 6. Complementary returns/repeatability | GAP versus MIM1/YANK2 paired block evaluation and normalized exposure, reusable schemas/hash inventories | Joint portfolio gate INSUFFICIENT_DATA; GAP operational N=30 sizing unchanged | Collect fresh adequate registered panel; no favorable historical filter adoption |

## Actual available evidence

[Audit](audit-reviewed/report.md), [machine evidence](audit-reviewed/evidence.json), [partial fill pairings](audit-reviewed/broker_roundtrip_parts.csv), [all strategy/ledger comparisons](audit-reviewed/strategy_ledger_reconciliation.json), [unmatched fills](audit-reviewed/unmatched_broker_fills.json).

Fresh broker export: 24 fills, 20 MIM and four YANK. Explicit broker order IDs establish attribution via hash-verified strategy-order/log prefixes. Broker account is 26556101; current units use that ID. MIM gross −$1,372.50 less $12.20 costs = −$1,384.70. YANK gross $307.00 less $4.88 = $302.12. Total net −$1,082.58 bridges a hypothesized $50,000 starting balance to broker balance $48,917.42. This supports a conditional accounting reconstruction, not an independent coverage certification. Original 14-fill export remains distinct and incomplete. All fill IDs are unique; broker order filled quantities agree and net contract quantities are zero. Current broker position/open-order snapshots are empty.

Snapshot cash P&L: August −$202.48; September −$880.10, both incomplete months. Modeled extra slippage and operating expenses are separate from actual fees. Broker execution prices already include actual execution effects; differences from strategy logs are not subtracted again. Inferred comparisons find ten MIM strategy rows and twelve realtime ledger records; 37 other records remain unmatched, including the August 13 EXTERNAL_CLOSE before the first current-account entry. Do not assume date alone assigns that row to this account.

[Conditional marked portfolio](portfolio-reviewed/evidence.json), [minute curves](portfolio-reviewed/marked_equity.csv). Exact-contract ProjectX minute exports cover ten fill sessions through September 16, including 16:01 close labels for actual post-16:00 EOD fills. Zero starting inventory/export completeness are explicitly conditional. The two SIM equity CSV streams observe the same shared account and are never added.

The reviewed report preserves synchronized **per-session** curves and actual / 1:2 / 1:1 / equal-exposure scenarios. It withholds continuous multi-session drawdown, recovery and Sharpe because an independent account reset register is absent. Earlier exploratory stitched figures are superseded and are not the reviewed evidence. The 1:1 figures reweight attributable broker executions; they are not reconciled performance of the shared SIM account. SIM strategy-level fill attribution remains missing.

The timestamp-only calendar has 27 observed dates, including 17 without fills and 211 RTH rows on September 7. Those dates are retained as source evidence, but unknown account epochs prevent assuming flat carry across them. Minute closes omit intraminute excursions. Partial realized cashflow statistics omit unrealized risk and cannot establish capital adequacy, combine passage or payout eligibility. No allocation change is supported.

[Power calibration](power-reviewed/report.md), [sweep CSV](power-reviewed/standalone_sweep.csv). The 115 exposed corrected GAP trades support a standalone trade-level variance calculation. IID diagnostics need 44,605 / 11,152 / 2,788 / 697 future trades to detect $5 / $10 / $20 / $40 effects at two-sided .05 and .8 power. Dependence sweeps are retained. Net-cost decomposition remains unresolved; these are not joint session/Sharpe power. The paired panel is missing; [evaluation gate](evaluation-reviewed/evidence.json) refuses confirmation. `registration-draft.json` is deliberately blocked, not a sealed threshold or approved test.

[Operations](operations.json): frozen source hashes and consumed prefixes/warmup match. Six sessions September 11–18 are excluded; eligible count zero. Commissioning is pending. The timer proposal preserves 120 eligible sessions, 60 seconds and June 10, 2027 deadline with no interim efficacy analysis. Lossless chunk archives retain completed snapshots without duplicating unchanged chunks; authoritative journals and old governing snapshots remain untouched. Capacity is finite and needs measured growth monitoring. The existing portfolio-decay observer remains advisory: no scheduler was found, only September 12/14 log runs; the Thursday Sharpe formula/metric provenance remains unresolved. No new threshold is adopted.

## Runbook and unresolved dependencies

See [package schemas/runbook](../../research/project_goals/README.md), [account questions/version register](../../research/project_goals/account-questions.md), and [blocked registration draft](../../research/project_goals/registration-draft.json).

Before committing these artifacts, preserve calibration-before-registration order. Before deployment, reconcile advancing main, independently review/test, verify flatness, merge rather than copy, then apply the lead's permitted lifecycle actions. This implementation has not deployed the proposed service or restarted any trader.

## Verification receipt

All 131 existing/focused MIM tests pass after updating bare object fixtures with the required active symbol; 71 project-goals tests pass. Frozen comparison suite: 112 tests passed across the broadened run and the corrected poll-window rerun. The broadened run initially had 15 environment-only failures because frozen `poll.sh` hardcodes the worktree's `.venv/bin/python`; a worktree-local `.venv` symlink to the original interpreter environment fixed them, without installing anything or linking production data. All 30 poll-window cases then passed. Existing Pydantic deprecation warnings remain.

New code passes Flake8 undefined/unused-name checks. Both proposed systemd units pass `systemd-analyze verify`. All four CLI commands were smoke-tested against available data/blocked gates; portfolio reconstruction ran on exact-contract API exports with a prefix-verified observed calendar. Frozen collector, adapter and `poll.sh` remain byte-identical. Archive tests cover byte-exact restore, deduplication, corruption and prohibited snapshot paths. No production journal was used by unit fixtures.

## Review corrections and reproducibility

Three independent review layers identified evidence-validation and verification gaps. The implementation was corrected to reject inconsistent source identities, parse order IDs exactly, withhold outputs from invalid fill batches, preserve unknown account epochs, and align power calibration with the registered simultaneous cost/block rule. Missing data never upgrades to usable evidence. The full triage is in [implementation spec](../../six-goals-spec.md).

[Observer provenance audit](observer-audit.json) records the checked scheduler locations, observed invocation dates and metric sources. No observer threshold or native halt authority changed.

[Account demonstration inputs](account-inputs/README.md) are explicitly synthetic software demonstrations, not forecasts. Generated paths compare Combine/XFA rules and self-funded capital without changing strategy size; fees remain published assumptions until invoices arrive. Capital requirements cannot be certified without broker margin and reliable intraday risk evidence.

[Verification receipts](verification/) include the independent 131-test MIM run and a byte-exact archive/restore of a copy of an actual 38.7 MB frozen poll snapshot (about 3.45 MB compressed, roughly one second). Production journals and original snapshots were untouched by that test.

[Account and capital comparison](account-comparison.md) links both synthetic path demonstrations, their assumptions and full component statuses. All review findings were corrected and the final source hashes match the regenerated evidence.
