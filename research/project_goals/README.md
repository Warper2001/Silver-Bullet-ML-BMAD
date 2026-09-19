# Six project goals evidence tools

This package separates implemented engineering from evidence of an edge, execution reliability, and withdrawable income. It is research-only. It does not change strategy parameters, consult sealed holdout data, send orders, or alter the September 10 comparison freeze. Live changes are made only after reviewed merge by the lead.

## Commands

Run from the worktree using the original interpreter:

```bash
PYTHONPATH=. /root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.project_goals audit --root /root/Silver-Bullet-ML-BMAD --output _bmad-output/new-audit
PYTHONPATH=. /root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.project_goals report --fills fills.csv --marks marks.csv --coverage coverage.json --output _bmad-output/new-report
PYTHONPATH=. /root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.project_goals power --paired paired.csv --standalone exposed-trades.csv --output _bmad-output/new-power
PYTHONPATH=. /root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.project_goals evaluate --data fresh-paired.csv --registration registration.json --power power/evidence.json --calibration exposed-paired.csv --repo . --output _bmad-output/new-evaluation
```

Every output directory must be new. `audit` accepts `--fills` (ProjectX JSON list or read-only envelope) and `--attribution` (explicit order-ID source map); `--account`/`--epoch-start` filter original-snapshot totals without deleting excluded records. Each output contains hashes, CSV evidence where applicable, JSON status and readable Markdown. Inputs stay read-only. A changed live input produces a retry status rather than an asserted stable audit.

## Evidence schemas

`attribution.json`: `orders` maps each broker order ID string to `{strategy,source,line,sha256}`. The source must hash exactly and the cited line must contain that order ID. Strategy CSV rolling `chain` fields are independent file hashes, never join keys. Broker `creationTimestamp`, not strategy signal/submission time, drives fill timing. MIM CSV `FILL` payloads can be unrelated recent fills; use ID-matched broker records as authoritative execution evidence.

`fills.csv`: `id,timestamp,account,epoch,strategy,contract,signed_quantity,price,actual_cost`. Time must contain timezone. Quantity is signed, costs are total actual cash fees plus commission for that fill. Each partial fill remains separate. Deduplicate only identical account/epoch/fill IDs; conflicting records are errors. Strategies use MIM/YANK/GAP names. Missing quantity/account/epoch/contract/cost prevents marked equity. Account epochs must be separate runs.

`marks.csv`: `timestamp,contract,close,multiplier,corrected_contract`. Use exact identified contracts, not an adjusted continuous series. `corrected_contract=true` is an input attestation that must be backed by the export's provenance; it is not proof by itself. A missing mark for an open position is unknown equity. Mark grid includes all strategies synchronously and must have no missing timestamps. Close-only marks cannot establish within-minute excursions or execution ordering.

`coverage.json`: `{start,end,initial_flat:true,complete_export:true,strategies:["MIM","YANK"],baseline_units:{"MIM":1,"YANK":2},grid_seconds:60}`. `baseline_units` specifies actual strategy sizing in the supplied fills so hypothetical 1:2 and 1:1 quantities are scaled correctly. Full exported coverage and verified starting positions must be evidenced independently. Do not set flags merely to make a report run. This strict interface covers a continuous interval; concatenate session results only with verified boundary inventory. Shared SIM balance CSVs are observations of one account, never attributable strategy curves.

`paired.csv`: one row per sorted unique eligible session, including no-trade zero sessions, with `session,mim_net_unit,yank_net_unit,gap_net_unit,mim_exposure_unit,yank_exposure_unit,gap_exposure_unit,mim_turnover_unit,yank_turnover_unit,gap_turnover_unit,corrected_contract,complete_costs`. Net values are per baseline strategy unit after actual modeled preregistered costs; exposures are synchronized gross-notional-time integrals using the same grid; turnover is total contracts filled. Attestations require source evidence and hash inventories. GAP operational N=30 rules are unchanged.

`account-sessions.json`: list of `{session,net_pnl,intraday_min_pnl,traded,requested_payout?,live_callup?}`. Intraday minimum includes unrealized P&L relative to opening equity and cannot exceed `min(0,net_pnl)`. Missing path means unknown breach risk. `--account-config` overrides provisional fields from `accounts.DEFAULT`. Published fees are scenarios, not actual invoices. `direct_capital` defaults to $10,000; 5k/10k/20k sensitivity changes capital without scaling P&L. Margin adequacy is unknown. Live API remains unavailable. The model never transfers funds.

## Evidence gates and registration order

1. Inventory and validate exposed corrected inputs. Run standalone and paired power sweeps before efficacy analysis. Standalone trade-level power is not portfolio/session power. Paired power includes block-based mean and Sharpe-increment design approximations.
2. Review design assumptions, effect-size sweep, costs and joint endpoint power. Select one parameter per experiment. Freeze the horizon and coverage rules before observing fresh outcomes.
3. Commit code and calibration artifacts. Then prepare and separately commit registration with artifact SHA256, computational source hashes and calibration commit. The draft below is deliberately blocked until paired evidence exists; it is not an approved seal.
4. Collect fresh, provenance-backed paired sessions after the registration commit and freshness cutoff. Unknown historical exposure, costs or start positions remain unknown.
5. `evaluate` verifies committed registration, unchanged power/calibration/code, adequate mean and risk endpoint N, fresh dates and exactly the registered horizon before computing any efficacy. It compares MIM1/YANK2 with MIM1/YANK2/GAP1, including equal gross exposure; requires positive standalone net expectancy and positive Sharpe increment at every registered block and cost stress, with simultaneous lower bounds. Undefined estimates refuse conclusions. No deployment follows automatically.

Required registration keys: `status:"registered", registered_at, freshness_after, power_sha256, code_sha256, source_hashes, power_status:"ADEQUATE_BOTH_ENDPOINTS", effect_usd, required_sessions, blocks, cost_stress, alpha, draws, seed, allocations:{MIM:1,YANK:2,GAP:1}`. `source_hashes` is emitted by `power.source_hashes()`. The registered alpha, complete cost list, complete block list, draws, seed and allocations must exactly match the committed calibration `design`; N/effect must select a supported horizon with adequate power for every cost/block/endpoint. Seed is a nonnegative integer. Draws must be at least `ceil(3 * len(cost_stress) * len(blocks) / alpha)` to resolve the simultaneous tail. Power remains a conservative normal/Monte Carlo approximation using the same three paired-bootstrap endpoints and simultaneous decision family. Future data collection provenance must be reviewed independently; booleans cannot certify it.

## Operational scheduler runbook

`systemd/project-goals-shadow.service` and `.timer` are proposed units for lead installation after merge. `OnUnitInactiveSec=1s` schedules from completion, not wall-clock start. An outer singleton lock and the frozen poll's own lock prevent overlap. The service records started/finished heartbeat, elapsed latency, failure and eligibility; it performs no outcome-table queries. Per-bar 60-second eligibility remains solely the frozen collector's rule. Historical late sessions stay excluded. Deadline remains June 10, 2027, with 120 eligible sessions and no interim efficacy review. Zero eligible sessions means commissioning is pending.

Before installation, create the research runs directory, verify frozen sources/prefixes/warmup, check journal permissions and `systemd-analyze verify` units. After starting the research timer, inspect heartbeat and journal. Restart recovery preserves authoritative journals. Failed child processes are recorded; the next permitted timer poll retries. No trader lifecycle action is performed by this package.

Newly completed operational snapshot directories reported by this wrapper's own child are sealed/hash-verified and stored in content-addressed gzip chunks. Identical chunks are shared across snapshots; full reconstructed file hashes are verified before only those new snapshot copies are removed. Original governing snapshots, active journals, feed, consumed prefixes and historical runs are untouched. A crash before cleanup may leave both copies; never delete originals to recover. Capacity remains finite: inspect `storage`, each archive's new compressed bytes and disk growth. The wrapper refuses below a 4 GiB operating reserve; this is infrastructure protection, not a strategy rule. Commissioning cannot promise a horizon until measured session growth supports it.

Restore a snapshot to a **new** isolated directory:

```python
from research.project_goals.archive import restore_chunks
restore_chunks('research/project_goals/runs/shadow-health/archives/<invocation>.chunks.json', '/tmp/restored-snapshot')
```

Verify the manifest hash against the heartbeat before restoration. Archive manipulation does not inspect or summarize outcomes. Check `heartbeat.json` plus systemd active state; a stale RUNNING record means interrupted process, not success. Operational report never certifies trading behavior from an idle-weekend poll. Observe a complete eligible session before commissioning.

The existing portfolio-decay observer remains advisory: no schedule or threshold changes here. Its metric provenance and a full-session operation remain outstanding, so it cannot independently justify a strategy restriction.

## Available broker snapshots and conditional reports

`audit --fills <envelope> --attribution <map> --ledger-export <snapshot>` also emits FIFO partial-fill round trips, unmatched broker records, and strategy/realtime-ledger comparisons. Inferred minute/date/side links are labelled separately from explicit order-ID attribution. Their reported P&L differences are not automatically slippage. Prefix-backed attribution adds `source_prefix_bytes`; later appends are allowed only while the consumed prefix hash and cited line still match.

`report --fills <canonical_fills.csv> --broker-marks <directory>` consumes read-only ProjectX History/retrieveBars envelopes, verifies exact contract requests and complete minute settings, and shifts start labels by one minute as the existing ProjectX adapter does. It stitches per-session cash offsets and requires flat observed quantities at session boundaries. This path is explicitly **CONDITIONAL_DESCRIPTIVE**: API-export completeness and zero initial positions remain assumptions backed by quantity/balance checks, not an independent statement. It does not weaken the strict default `--marks/--coverage` path. The output includes only sessions containing fills, so its annualized Sharpe is a selected-active-session diagnostic, not a complete-calendar estimate. Account breach risk remains uncertain between minute closes.

`--session-calendar observed_calendar.json` adds observed no-fill dates as zero cashflow only under the same complete-export/flat-carry assumptions. The file records source path, consumed byte count and prefix SHA256 plus `{session,first_timestamp,last_timestamp,observed_rows}`. Calendar timestamps come from actual observed bars, never contractless prices; this does not certify holiday coverage. Without it, active-session-only Sharpe remains explicitly labelled.

### Stale child recovery

The wrapper observes the child each second. After 60 seconds it records `OPERATOR_RECOVERY_REQUIRED_CHILD_ALIVE`, PID, process-start identity, elapsed time and a fresh observation timestamp. This is an operational alert, not a command to terminate. The child inherits the singleton lock, so an interrupted wrapper cannot permit overlapping polls. `--health-only --state <state>` inspects the persistent child receipt without signaling any process.

A pending `child.json` blocks a replacement poll, even if the old child has exited, until an operator reviews the recorded process identity, completed artifacts and journal integrity. A child whose late exit is observed preserves its stale/latency alert history, finishes result handling and archival, then clears its receipt and resumes completion-plus-one-second cadence. Manual recovery is reserved for orphaned/interrupted receipts or unhandled results. Do not restart the service or launch a replacement while its child is alive. Process termination requires separate user authorization under repository policy; this package never sends termination signals. After the child has exited and the operator has reviewed the evidence, preserve `child.json` under a dated recovery filename before permitting the next poll. Never remove active journals or authoritative state as recovery.

Epochs are never fabricated: `audit --account/--epoch-start` propagates only explicit caller-supplied account/epoch information, without independently proving a reset. Absent epoch identity remains blank. Strict marked equity rejects unknown/placeholder epochs. The API-export report can still emit **PER_SESSION_CONDITIONAL_UNKNOWN_EPOCH** curves and actual/1:2/1:1/equal-exposure comparisons separately for each session; it does not stitch balances, drawdown or recovery across dates or infer no-fill-day flat carry. These rows remain in `marked_equity.csv` with per-session scope. Multiple actual epochs require separate reports. Monthly expense allocation across unknown epochs is reported as unknown, with the requested expense assumption retained.

API-mark exports may span dates or contain different supported MNQ contracts in separate files. Bars are partitioned by actual close date and contract, and conflicting duplicates are refused. No other product receives the MNQ multiplier implicitly.
