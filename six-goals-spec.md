---
title: Six project goals implementation
created: '2026-09-19'
status: done
route: dispatch
baseline_commit: '47349c3e50b5d2c2e7f06022a231a5e4a2e2ff36'
context: ['/root/Silver-Bullet-ML-BMAD/AGENTS.md']
---

<frozen-after-approval>
## Intent
Implement all six recommendations explicitly authorized in the user plan. Deliver trustworthy operational and research tools, actual available-data reports, and separate engineering/evidence statuses. Work only in `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/six-goals`. Original checkout `/root/Silver-Bullet-ML-BMAD` supplies read-only data. User authorizes eventual merge and restart in verified flat window; lead handles deployment. No remote push, order endpoints, capital changes, messages to providers, holdout reads, strategy changes, or historical journal rewrites.

## Boundaries
Preserve sigma history and its contamination age-out policy. Frozen comparison sources and poll.sh remain byte-identical. Preserve September 10 freeze, June 10 2027 deadline, 120 eligible sessions, 60-second rule, no interim efficacy analysis. Research input files stay read-only. Unknown attribution/cost/coverage remains unknown, not zero. Two SIM equity streams observe one shared account. Existing GAP-1 operational N=30 sizing rules remain unchanged. Account modeling is provisional; automated Live route unavailable without written confirmation. $10,000 self-funded scenario; $20,000 monthly aspiration cannot be justified by linear extrapolation.

## Edge cases and acceptance
- Given persisted price provenance matching active contract, restore it; given legacy/mismatch/unknown provenance, rederive via contract lookup and fail closed on unavailable value. Cold seed cannot label old-contract price as current. Open-position reconciliation and sigma restore survive.
- Given source/prefix drift, overlapping poll, late bars, roll or failed process, scheduler refuses unsafe progress or reports failure, preserves journals and recovers on next permitted poll. Completion plus one second cadence, singleton lock, heartbeat, latency, eligible count and exclusion reasons; commissioning remains pending until full eligible session observed.
- Given partial/duplicate fills, resets or missing costs, audit retains quantity/account/contract provenance and unmatched records; totals reproducible without inventing joins. Missing marks never imply flat equity.
- Given insufficient power, changed inputs or insufficient fresh sessions, evaluate refuses confirmatory conclusions. GAP comparison is MIM1/YANK2 versus plus GAP1, also normalized to same gross exposure. Require positive standalone net expectancy AND portfolio risk-adjusted improvement under paired session blocks and cost stress. Calibration sweeps precede registration and confirmation.
</frozen-after-approval>

## Code Map
- `src/research/mim_nb_live.py`: `_backfill` trusts legacy st.symbol; `_seed_sigma_from_bars` directly assigns last seed close; `_maybe_roll`, `_close_out_session`, `_save_state`, constructor all need prev_close_symbol maintained. `_prev_close_for_symbol` existing contract-specific read/fetch. Existing provenance tests use object.__new__ fixtures.
- `research/mim_comparison/poll.sh`: existing frozen finite locked poll, absolute state rooted to own repo; adapter + verified 500-row window + frozen collector. DO NOT EDIT. Main runs/20260910T214803-shadow-47055eba62/collector/{freeze.json,journal.sqlite}; adapter runs/20260910-contract-feed; wrapper runs/20260910-contract-feed-poll. Core/adapter/poll/warmup and consumed prefixes currently match, zero eligible sessions, last polling Sept10. Health queries must read sessions/exclusion tables only, no outcome queries.
- `research/mim_comparison/{shadow.py,feed_adapter/adapter.py}` contains frozen validation to mirror/read without altering. `tools/portfolio_decay_shadow.py` advisory existing PF observer; inspect service/timer and metric provenance, do not change thresholds.
- `data/mim_nb/{projectx_fills.json,orders.csv,trades.csv,sessions.csv,state.json}` original repo read-only; snapshot 14 fills Aug13–28 account26556101 MNQU26, not current complete evidence. Strategy CSV lacks account/symbol. Ledger `data/trades.db` mode=ro, realtime only, ISO8601; metadata sometimes lacks quantity/account/symbol. Expose incomplete coverage.
- `data/ts_sim_mirror/{mim,yank}_invvol_equity.csv`: same shared account (ts_utc,equity,buying_power,contracts); not attributable strategy curves. Build synchronized marked equity only from attributable fills plus matching corrected contract marks; otherwise report missing evidence with usable input interface. Compare actual exposure, 1:2, 1:1, normalized exposure; concurrent losses, drawdown/recovery, net Sharpe, cost sweeps, intraminute uncertainty.
- `_bmad-output/diagnostics_gap_fade_gate0_rescore_20260916/corrected_gate0_trades.csv`, diagnostics_gap_fade_splice_20260916/, diagnostics_gap_fade_power_gate_20260913/ contain exposed calibration. `tools/xsmom1_power_gate.py` gate pattern. Never read data/sealed_holdout. If paired corrected sessions unavailable produce UNDERPOWERED/INSUFFICIENT_DATA and collection horizon formula rather than invented results.
- Existing account packet `_bmad-output/planning-artifacts/research/academic-lit-six-project-goals-next-actions-2026-09-18/account-questions.md`. Current official API page https://help.topstep.com/en/articles/11187768-topstepx-api-access verified Sept19: no Live API, no remote order transmission. Lead can supply additional current official rules.

## Tasks
- [x] Fix MIM price provenance plus isolated regression tests: matching/mismatch/legacy/failure/cold/open-position.
- [x] New research-only `research/project_goals/` package commands audit/report/power/evaluate, documented schemas, readonly input hash inventories, CSV evidence JSON status readable reports. Fail closed on missing evidence; useful implemented calculations not empty scaffolds.
- [x] Add operational wrapper around frozen poll and research-only systemd service (completion +1s, singleton, health); preflight hashes/prefixes/warmup and session eligibility. Tests isolated from production journals incl overlap/restart/source drift and latency. Lead installs only after review/merge.
- [x] Reconciliation/economics including actual fees vs modeled slippage, monthly variability drawdown recovery operating cost and capital sensitivity. No false current-account completeness.
- [x] Portfolio marked equity and shared-account observation handling with missing-mark/cost uncertainty. Input schema supports complete future attributable exports.
- [x] GAP power sweep and gated paired evaluation, hashes and committed registration checks, no efficacy run without fresh adequate evidence. Calibration artifacts and registration staged for lead to commit in order.
- [x] Account packet expanded for fees/payouts/cushion/floors/phases and response/version register. Configurable phase/path model including self-funded $10k and sensitivity; provisional rules cited; no advice to transfer funds.
- [x] One progress record linking all six statuses, reports/dependencies/next dates, actual available-data outputs and runbook.
- [x] Targeted tests incl duplicate/partial fill, epoch, missing cost/mark, shared account, report determinism, failed/changed gates and insufficient fresh evidence.

## Implementation Notes
User's explicit implement-all plan supplies scope/approval; no renewed spec gate needed. Implementation agent owns code/tests/docs only in worktree, no merge/systemd changes. May investigate original checkout read-only; run Python via absolute original `.venv/bin/python` or `.venv-research/bin/python`, no installs. Long >30s tasks nohup and monitor. Do not create symlink to production data; imports of live module can create logs locally in worktree. Keep frozen collector tests fixture-isolated. Obtain full canonical git IDs with status and both divergence counts before git operations; avoid unrelated changes. Keep evidence conclusions honest when datasets unavailable.

## Verification
Run focused pytest covering changed code and MIM provenance suites; smoke each CLI on fixtures and available real read-only evidence. Confirm frozen sources identical. Lead independently inspects diff and runs review before merge.

## Review Triage Log

All three layers completed before triage. Findings are concrete defects or missing tests in the implementation; apply local corrections without changing the authorized intent or public command surface. No findings discarded.

| Finding | Verdict | Evidence and route |
|---|---|---|
| B1 | high | patch — Audit suppresses aggregate net on issues but strategy/monthly outputs and canonical fills bypass the guard; patch propagation. |
| B2 | high | patch — Substring test accepts unrelated numeric content; exact order-field parsing required. |
| B3 | high | patch — Source-keyed dictionaries overwrite conflicting hashes/prefix lengths; reject conflicting declarations. |
| B4 | high | patch — None account_epoch and fabricated canonical epoch permit reset crossing; retain unknown provenance and restrict conditional pairing to a session. |
| B5 | high | patch — Evaluation divides alpha over costs and blocks while calibration omits these and stressed turnover; align design and selection gate. |
| B6 | medium | patch — RNG reads seed after guarded block without validation; validate nonnegative integer. |
| B7 | medium | patch — Combine minimum uses row count; count evidenced trading days. |
| B8 | high | patch — All bar contracts receive multiplier2; reject unsupported contract identities. |
| B9 | medium | patch — First bar date owns entire envelope and second same-date contract rejected; partition by session and contract. |
| B10 | medium | patch — CLI broker path omits operating-cost argument for every scenario; pass through. |
| B11 | medium | patch — No child deadline and infinite systemd timeout permit stale RUNNING; add live progress/stale detection and explicit operator recovery. Automatic process killing is prohibited, so do not add forced termination. |
| B12 | medium | patch — Empty portfolio insufficiency is overwritten unconditionally; preserve failure status. |
| E1 | high | patch — Independently reproduced order-ID substring mismatch; same root as B2. |
| E2 | high | patch — Conflicting duplicate first record still produces canonical and strategy net; same root as B1. |
| E3 | medium | patch — Audit/FIFO ISO text ordering differs from chronological order with mixed UTC offsets; sort parsed timestamps. |
| E4 | high | patch — Unvalidated exact non-MNQ contract uses multiplier2; same root as B8. |
| E5 | high | patch — Session-local epoch checks allow stitching reset epochs; validate entire report before partitioning. |
| E6 | medium | patch — Empty report upgraded to conditional; same root as B12. |
| E7 | medium | patch — Operating expense dropped on broker report; same root as B10. |
| E8 | medium | patch — Zero scenario exposure falls back to factor1 even when actual exposure positive; label normalization unknown. |
| E9 | medium | patch — Missing/None/invalid seed escapes validation; same root as B6. |
| E10 | medium | patch — Breach branches leave end_balance equal pre-loss balance; terminal value unknown, preserve last verified balance separately. |
| E11 | medium | patch — Combine uses calendar rows; same root as B7. |
| E12 | medium | patch — Hung child stops polling silently; same root as B11, operator permission required before killing. |
| V1 | high | patch — Preverified regression gap: existing evaluate tests exit before final all-endpoint bounds; add synthetic PASS and endpoint/cost FAIL coverage. |
| V2 | medium | patch — Preverified regression gap: no requested_payout tests reach balance deduction/floor/eligibility reset; add both paths and denials. |
| V3 | medium | patch — Preverified regression gap: fake scheduler children all fail; add successful real-format parse/archive integration with restore checks. |
| P1 | medium | patch — Parent caller check: Standard XFA win_days increments on P&L alone; require evidenced traded day just as Combine/Consistency do. |
| P2 | medium | patch — Parent numerical check: draws>=2 allows a bootstrap grid coarser than the simultaneous tail probability; require enough draws to resolve the registered tail and label Monte Carlo approximation. This derives from the decision rule, not a new strategy threshold. |

## Review resolution

All B/E/V/P patch findings resolved. Parent independently verified 131 MIM tests and 71 project-goals tests; source and test Flake8 F checks and systemd unit validation pass. Frozen 14 files remain byte-identical and operational preflight passes. Reviewed real artifacts preserve unknown epochs and per-session scope; no efficacy conclusion or approved GAP registration. Deployment and future-session/provider evidence are tracked separately in PROGRESS.md.
