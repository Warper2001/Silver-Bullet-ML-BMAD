# Prospective collection readiness — September 12, 2026

**MIM collection is not operationally ready. FOMC scheduling is active, but its event eligibility implementation does not match the sealed protocol. The proposed benchmark remains inactive.** These conclusions concern coverage and correctness; no prospective efficacy was inspected.

Operational observations were taken September 12 at 21:09–21:12 UTC, with any later evidence checks individually timestamped in the [machine-readable evidence](evidence/readiness-20260912.json). They describe that observation window, not an ongoing health guarantee. The separate [fresh diagnostics audit](runs/20260912T211038-audit-847291ba09/feasibility.md) and its hash inventory passed the diagnostics CLI's independent `verify` command.

## MIM: integrity passes, coverage and operation do not

| Check | Observed evidence | Readiness consequence |
|---|---|---|
| Frozen implementation | All nine core source hashes, four adapter source hashes and polling-wrapper hash match their frozen states | Resume must retain these exact versions; no source-drift repair is needed |
| Frozen warmup | SHA256 matches; August 24–27 each has 390 MNQU26 cash-session minutes; August 28 has only 109, ending 11:19 ET | Latest complete warmup grid is August 27; historical integrity does not establish fresh prior-session context |
| Authoritative feed | 78,658 rows, all MNQU26; last event September 10 at 22:10 UTC; committed 193,356,329-byte prefix matches its journal SHA256 | Feed is stale relative to the observation date |
| Durable collector | 21,626 observations, all unavailable as outside the frozen horizon; zero session records and zero eligible sessions | There is no valid prospective bridge from warmup to the next collection day |
| Scheduler | No MIM comparison scheduler found in the inspected systemd/cron inventory; the separate `mim-parity-check` is not this collector | The finite polling wrapper does not run itself; an external scheduler outside the inspected scope cannot be ruled out |
| Input backlog | Consumed raw-bar and runtime-log prefixes match recorded hashes and filesystem identity; roughly 174 KB of bars and 21 MB of logs remain unconsumed | Stable resumable input exists; old backlog cannot become timely prospective evidence |
| Contract transition | No MNQZ26 rows in the identified feed; inherited quarterly-expiry policy excludes MNQU26 on September 18 | Next-contract volume, prior close and minute coverage are unproven |
| Arrival budget | Frozen receipt and durable-collection budget is 60 seconds; earlier bounded poll took 22.339 seconds | One historical timing observation is not a sustained-latency guarantee |

The governing freeze remains **2026-09-10T21:02:24.363841+00:00**, with **120 eligible paired sessions** and deadline **2027-06-10T21:02:24.363841+00:00**. No interim efficacy look, deadline extension or eligibility reset is authorized.

The immediate warmup problem is specific. In [`context_for()`](/root/Silver-Bullet-ML-BMAD/research/mim_comparison/shadow.py:243), contract selection requires a complete **previous weekday's** contract-volume grid. With no timely intervening observations, the next collected session lacks that context. The unchanged collector can retain genuinely timely bars while a session is unavailable; subsequent sessions may qualify after their prerequisites exist. This is not a claim that another fourteen fresh sessions are necessarily required: the frozen code also retains historical warmup. Do not splice late bars into the frozen warmup or manufacture holiday/contract coverage to force eligibility.

Contract identity in the existing feed is inferred from causal request/log context. Its frozen audit passes, but that log does not authenticate the broker response payload's contract. This limitation survives every hash check and matters for any new experiment's provenance standard.

## FOMC: active timer, incomplete correctness assurance

The installed timer was enabled and active/waiting. Its last execution completed successfully at **September 12, 06:45:03 UTC**, and its next scheduled firing was **September 13, 06:45 UTC**. Installed service/timer bytes match repository copies. The ledger does not yet exist, consistent with the first registered event on **September 16**.

The tracker accrues only event dates earlier than its execution date. Under the current timer, first due-event collection is therefore **September 17 at 06:45 UTC**, not on the event afternoon. Successful pre-event invocations exercise the no-due-event path; they establish neither working authentication nor a valid market-data response. No broker request or credential access was attempted by this audit.

Four engineering issues need attention:

1. **The full-window gap requirement is missing.** The [seal's eligibility rule](/root/Silver-Bullet-ML-BMAD/_bmad-output/preregistration_evfade_fomc_prospective.md) requires 14:00–14:33 ET coverage with no gap exceeding two minutes. [`evaluate()`](/root/Silver-Bullet-ML-BMAD/evfade_fomc_prospective_tracker.py:153) reads only 14:00, 14:03 and 14:33. A mapping containing just those three nonconstant endpoint prices passes its presence checks despite a thirty-minute gap. This is a static control-flow finding; no strategy evaluation was run.
2. **Eligibility cannot be reconstructed from the event CSV alone.** Its [field list](/root/Silver-Bullet-ML-BMAD/evfade_fomc_prospective_tracker.py:72) lacks selected contract, bar timestamp convention, raw-source/hash and interval-coverage evidence. Preserve such evidence separately with any future accrued row and every failed acquisition or ineligible-event attempt. Currently a `None` result leaves no event row, so later invocations retry it. Record each attempt timestamp, source/contract, coverage and exclusion reason, and define how later revisions are handled before accrual; never erase an earlier attempt or infer missing provenance from a profitable result.
3. **Stopping is manual rather than enforced by the collector.** `STOP_DATE` is printed but not used in the due-event filter. The sample target is also reported after accrual, rather than bounding that loop. Human evaluation is intentional in the seal, so this does not establish a past violation; the operational runbook must enforce the stopping/sample boundary, or a reviewed engineering change must do so. A catch-up invocation can also append several overdue events and cross N=15. Preserve an immutable first-15 accrued sample in the declared accumulation order and a one-time interim-look record; later retries must not replace that sample. Specify recovery when a backlog crosses the boundary, preserving the original interim/final protocol rather than creating a later or repeated interim look.
4. **The calendar does not span the entire protocol.** Eleven dates end December 8, 2027, while the stopping date is December 31, 2030. Later official schedule publication and reschedules require versioned maintenance under the existing protocol. The current audit did not retrieve a new official schedule or establish historical publication vintages.

Keep K3/M30, the registered costs, N=15 interim, N=30 final and 2030 stopping rule unchanged. The failed impulse-following study and rejected policy-shock throttle remain closed. None of these operational findings is a reason to inspect running returns or amend a hypothesis.

## Remediation sequence and evidence of completion

These are concrete follow-up requirements, not actions taken by this audit. The existing collection state and experimental boundaries must survive each repair.

| Order | Work package | Completion evidence |
|---|---|---|
| 1 | Prepare an isolated FOMC engineering correction before the September 16 event: enforce the existing full-window gap rule; retain successful and unsuccessful attempt evidence and revision handling; specify stopping enforcement and immutable N=15 sample/look records | Synthetic tests reject the three-endpoint-only case and gaps >2 minutes; accept the registered boundary; handle missing endpoints, duplicates, nonfinite values, DST, retries and catch-up across N=15/N=30; prove no orders or interim efficacy output; compare all strategy constants and the seal before/after |
| 2 | Prepare an operational MIM scheduling design around the unchanged `poll.sh`, adapter and original journal | Reviewed unit/runbook with original absolute state paths, non-overlap locking, failure evidence, receipt-to-durable-collection monitoring and measured end-to-end latency under representative load; cadence follows the 60-second budget rather than an arbitrary polling interval |
| 3 | Establish fresh input and September contract-transition readiness | Timely first observations for required candidate contracts, complete prior-day volume and selected-contract prior close, explicit provenance, and verified unavailable-session reasons; no retrospective repair of missed sessions |
| 4 | Verify an entire prospective session operationally | Complete input coverage and all durable half-hour decisions within the frozen budget; eligibility count and exclusion reasons only; no payoff/efficacy summaries |
| 5 | Calibrate the separate benchmark design and commit its activation record if justified | Benchmark-specific hash-bound power artifact, reviewed inference, derived horizon/decision settings and verified independent future start; NOT_ASSESSED or UNDERPOWERED denies activation |

No scheduler was installed, service started/stopped/restarted, collector invoked, live code edited, dataset acquired or ledger repaired. Steps 1–4 require their own reviewed operational work; any deployment follows repository merge and service-authorization rules. Existing missed periods remain unavailable.

## Benchmark design delivered

The [committed design](../../_bmad-output/preregistration_mim_cash_session_benchmark_design.md) defines unchanged A versus one scheduled long MNQ contract from the cash open to the close, with the common eligible-session grid, explicit modeled execution/costs, and no selected time windows.

Its proposed primary comparison is the difference in annualized daily-dollar Sharpe statistics, jointly requiring positive expected net A. Exposure, dollar expectancy and sampled risk remain separately reported. A paired dependence-aware power and interval-coverage calibration must validate that design before it is activated. A better Sharpe statistic would not establish lower intrabar risk, sufficient capital or deployment readiness.

**Design status: inactive. Power status: NOT_ASSESSED.** No benchmark returns, effect-size sweep, strategy power verdict or candidate promotion was produced. A numerical economically worthwhile effect cannot be inferred from exposed profits; the design records the required calibration and economic justification rather than inventing one.

## Reproduction and limits

The [evidence JSON](evidence/readiness-20260912.json) records paths, hashes, source-line references, query/command scope and timestamped observations. It deliberately excludes credentials, raw broker logs and prospective outcome values. Operational queries are read-only; coverage checks do not assert future liveness. File hashes establish observed byte identity, not economic efficacy or external authenticity.

From the isolated diagnostics worktree, recheck the preserved audit:

```sh
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.mim_diagnostics verify --run research/mim_diagnostics/runs/20260912T211038-audit-847291ba09
```

The audit artifact retains its original timestamp. For a later observation, run a fresh diagnostics `audit` and repeat the explicit scheduler, frozen-source, input-prefix, warmup and roll checks in the evidence record. Do not run `poll.sh`, either tracker or `evaluate` merely to reproduce this readiness report. Large audit artifacts are local under ignored `runs/`; the compact evidence and this report are versioned.

[Independent verification](evidence/readiness-verification-20260912.json) rehashed all 26 recorded files and both consumed source prefixes, checked the report/design links, and confirmed the fresh audit's successful verification. These are documentation and evidence changes; no research or collector code changed, so no new strategy tests or application test suite was run for this follow-up.
