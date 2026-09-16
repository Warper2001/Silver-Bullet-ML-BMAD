---
id: mnq-wick-short-calibration-20260916
status: preregistered-calibration-protocol-only
created: 2026-09-16
original_gate_verdict: POWER_UNDETERMINED
evaluation_allowed: false
strategy_parameters_changed: false
historical_sample_role: calibration-development-only
prospective_sample_role: observation-calibration-only
calibration_executed: false
collector_activated: false
confirmation_authorized: false
---

# MNQ wick-short calibration pre-registration

This separate registration defines two later calibration phases: estimate the
frozen model's reference-price effect and variability on the already exposed
historical sample, then observe quotes and signal availability for four
prospective weeks. Preparing and committing this document does not run either
phase. The original gate remains `POWER_UNDETERMINED`, with
`evaluation_allowed=false`; this document does not turn that gate into permission
to evaluate profitability.

## 1. Scope and evidence boundary

The entire existing sample strictly before `2026-03-01T00:00:00Z` is assigned
exclusively to calibration/development. No portion of it is reserved or later
relabelled as untouched confirmation data. The prospective observation window
is also calibration only. Neither phase becomes confirmation evidence, whether
its results are favorable, unfavorable, or unassessable.

The original registration's prohibition on aligned outcomes continues to govern
the original gate and its artifacts. This separate protocol defines a later
calibration calculation of aligned reference outcomes; it does not amend the
gate code, overwrite its report, or permit a profitability verdict. Execution
of either phase is outside this registration-only task. Any later implementation
must be committed and verified against this contract before it runs.

No holdout access, collector launch, order submission, live-code change,
strategy-parameter change, deployment, or service operation is authorized by
preparing this registration. No new CLI or public API is implemented. Do not
import a live trader or write its trade logs or `data/trades.db` for calibration.

## 2. Bound artifacts and prior exposure

Paths below are relative to the repository root unless absolute. SHA-256 values
identify exact file bytes. Canonical revisions are the existing commits
containing these artifacts, verified against the checkout when drafting this
registration; preserve those files unchanged.

- Registration/specification/gate revision:
  `a04fde464d499bb9ca8f9a663657a984bd28b0f5`.
- Result revision: `8492ae12212d85e0b420240c36fa5742a8cae289`.
- Main checkout used as the drafting base:
  `4fd1368b25f49bb378bd8c25be0a87a8742c10ae`.

| Artifact | Canonical Git revision | SHA-256 |
| --- | --- | --- |
| `_bmad-output/preregistration_mnq_wick_short_power_20260916.md` | `a04fde464d499bb9ca8f9a663657a984bd28b0f5` | `ed3fad3c455100ca4b12a918fa658b7a717f011a4dfdfd0cf292c1e66784daff` |
| `tools/mnq_wick_short_power_gate.py` | `a04fde464d499bb9ca8f9a663657a984bd28b0f5` | `adb16fda09e0324ca30180589faf39e7c8580d46c8032e806e8556125244b0da` |
| `_bmad-output/specs/spec-mnq-wick-short-power/SPEC.md` | `a04fde464d499bb9ca8f9a663657a984bd28b0f5` | `0a93788286199fc420bb72ec1faf7c568722f527e1ea5dd8c9e7a6f434f1384d` |
| `_bmad-output/specs/spec-mnq-wick-short-power/mechanics.md` | `a04fde464d499bb9ca8f9a663657a984bd28b0f5` | `d943106f4bf940992937cd7993ffd516a265d22ed04f467962684cd077d74e72` |
| `_bmad-output/specs/spec-mnq-wick-short-power/data-handling.md` | `a04fde464d499bb9ca8f9a663657a984bd28b0f5` | `35574935330b56c91cfaa9d3a7896f02b08c11f290c71beabb69997ed958526a` |
| `_bmad-output/specs/spec-mnq-wick-short-power/statistical-assumptions.md` | `a04fde464d499bb9ca8f9a663657a984bd28b0f5` | `b1f1f34ec3fb175d05bf53446a5f4adb8c0ced34771f8db33d1ca31cae74de10` |
| `_bmad-output/mnq-wick-short-power-20260916/report.json` | `8492ae12212d85e0b420240c36fa5742a8cae289` | `4b33288c6abfefe2d0a36f93365e7841784e29f7c3630a1a13694b3b40930546` |
| `_bmad-output/mnq-wick-short-power-20260916/report.md` | `8492ae12212d85e0b420240c36fa5742a8cae289` | `4c557946a7fb798a7ef10d2c158cede6e38bb7968e5720a0767c5aba04d3779c` |
| `/root/mnq_historical.json` | External input; hash recorded in result revision `8492ae12212d85e0b420240c36fa5742a8cae289` | `e7aed8ba786436ba80f4b081d3b7a4ee97b06bd3e8cc347c035547ec57dcb924` |

The input is outside Git and has no canonical Git blob/revision of its own.
Its hash above is taken from the committed result, not from a fresh read of
market data for this registration. A later historical run must verify that
whole-file hash and retain only records strictly before the UTC cutoff. The
hash covers the source container, not a claim that every row is pre-cutoff.
A mismatch stops the run for a recorded provenance discrepancy; do not silently
replace the input or broaden the sample.

The already published exposure is:

| Quantity | Existing result |
| --- | ---: |
| RTH dates seen / eligible / excluded | 576 / 515 / 61 |
| Exclusions: incomplete RTH / mixed contract | 23 / 38 |
| Signals / sessions with signals / zero-signal eligible sessions | 585 / 351 / 164 |
| Signals per eligible session | 1.1359223300970873 |
| Records at or after cutoff skipped | 62,277 |
| Circular session shifts | 506, offsets 5 through 510 inclusive |
| Transferred-dispersion SE, median / p10 / p90, dollars per signal | 1.8075052354576118 / 1.6341162806457556 / 2.096199469093597 |
| Conditional normal-approximation net MDE, dollars per signal | 4.494316328231563 |
| Conditional gross movement for costs $1.22 / $2.22 / $3.22 | $5.714316328231563 / $6.714316328231563 / $7.714316328231563 |

These counts and shifted-dispersion calculations have already informed design.
The gate did not pair signals with their own following-bar outcomes. Its shifts
are sensitivity calculations, not additional independent observations, an
uncertainty interval for aligned returns, a future-edge estimate, or validated
power. No new historical outcomes are calculated in this document.

## 3. Phase A: frozen historical calibration

### Eligibility and mechanics

Retain the original input rules: reject malformed or offset-free timestamps,
duplicate retained timestamps, non-finite OHLC and invalid high/low ordering.
Use `America/New_York` session dates with DST-aware conversion from UTC.
Eligible sessions contain exactly the 390 close-stamped RTH minutes from
09:31 through 16:00 inclusive and exactly one contract label. Exclude incomplete
and mixed-contract sessions with reasons; do not fill missing minutes, splice
contracts, or substitute a continuous/back-adjusted series. Short holiday
sessions remain incomplete under these frozen mechanics. Audit calendar/roll
anomalies in the ledger; resolving a discrepancy requires a recorded amendment,
not an outcome-informed eligibility change.

Form 78 right-labelled five-minute OHLC bars. The first contains minute labels
09:31--09:35 and is labelled 09:35; the final contains 15:56--16:00 and is
labelled 16:00. For every bar define:

```text
B = abs(close - open)
U = high - max(open, close)
D = min(open, close) - low
signal = (B > 0) and (U >= 2 * B) and (D <= 0.1 * U)
```

Either candle colour qualifies. Signals are eligible from 09:35 through 15:50
inclusive (zero-based slots 0 through 75). There are no trend, weekday,
red-day, calendar-performance or other discretionary filters. No parameter
searches, stops, targets, discretionary exits, or sizing changes are allowed.

Each signal uses one MNQ contract over its own following bar's open-to-close
reference interval. A 09:35 signal uses the open of the minute labelled 09:36
through the close labelled 09:40. A 15:50 signal ends at 15:55; 15:55 and 16:00
signal bars are excluded. Adjacent signals have adjacent, non-overlapping
holding intervals; do not skip adjacent signals or treat a shared boundary as
two fills at an identical price. These prices are historical references, not
evidence that the signal was available in time to execute at the next open.

### Monetary quantities and reporting

For signal `i`, using the following bar from the same eligible session:

```text
x_i = 2 * (next_open_i - next_close_i)     # gross reference dollars, one MNQ
y_i(c) = x_i - c                         # assumed net reference dollars
c in {1.22, 2.22, 3.22}                  # total round-trip cost per signal
```

The published TopstepX MNQ round-trip cost is $1.22 as checked on 2026-09-16.
The inherited scenarios add $0, $1 or $2 of adverse execution cost to that fee
(zero, two or four total MNQ ticks). They are assumptions about prospective
costs applied to historical reference prices, not reconstructed historical
charges or measured future costs. Keep all three scenarios; do not select the
most favorable. Any later fee change is disclosed separately and does not
silently rewrite these scenarios. [Topstep fee schedule](https://help.topstep.com/en/articles/8284213-topstepx-commissions-and-fees)

Report the gross and each scenario's mean dollars per signal, sample standard
deviation (`N-1` denominator), and empirical outcome distribution. Report total
signals `N`, all eligible sessions `S`, `N/S`, sessions with and without signals,
and the session count distribution. For each eligible session `s`, retain
`n_s`, gross total `X_s = sum(x_i in s)` and scenario total
`Y_s(c) = X_s - c*n_s`. A zero-signal session has zero count and zero totals,
not a missing row. Report aggregate dollars, mean dollars per eligible session
and dispersion of session totals, including those zero-signal sessions. Keep
excluded/incomplete sessions distinct from eligible sessions with zero signals.

Counts should reconcile to the bound gate result on the identical input and
rules. A discrepancy must be explained before calculation proceeds; do not
force counts or drop observations to reproduce a desired result. This phase
reports calibration estimates and uncertainty only: no profitability verdict,
hypothesis-test pass/fail, parameter ranking, or assumed future edge.

### Approximate 95% uncertainty intervals

For a per-signal mean, let `N` be the number of signals, `y_i` be gross or one
cost-scenario outcome, and `mean_y = sum(y_i)/N`. Calculate both groupings:
the original signal's local session date and its `(ISO year, ISO week)`.
For each grouping separately, count only nonempty clusters of observations:

```text
G = number of nonempty clusters for this grouping
R_g = sum(y_i - mean_y for i in cluster g)
SE^2 = G/(G-1) * sum(R_g^2 for g in clusters) / N^2
SE = sqrt(SE^2)
CI = [mean_y - t_(0.975,G-1)*SE, mean_y + t_(0.975,G-1)*SE]
```

Report `N`, `G`, degrees of freedom, SE and both interval endpoints for each
grouping. Report both intervals and their outer envelope
`[min(lower_session, lower_week), max(upper_session, upper_week)]`. This is a
sensitivity envelope, not an independent third confidence procedure. The
finite-cluster correction and Student-t critical values here belong to this
new calibration protocol; do not revise or reuse the old gate's uncorrected SE
as though it implemented this formula.

Apply the same recipe to mean session totals and signal frequency using one
observation per eligible session (`X_s`, `Y_s(c)`, or `n_s`) and replacing `N`
by `S`. In those calculations every eligible session is an observation, even
when its value is zero. Session clustering then has one observation per
cluster; ISO-week clustering groups those session observations. Conversely,
zero-signal sessions are not extra observations or clusters in the per-signal
mean. A cluster whose residual sum is zero is still nonempty and must count.
Constant per-signal cost subtraction shifts mean/interval endpoints by `c`
without changing per-signal dispersion or SE; session totals need separate
calculation because `n_s` varies.

If there are fewer than two observations, fewer than two usable clusters, or
a non-finite, nonpositive or otherwise invalid variance, label that interval
`UNASSESSABLE` with a reason. Never substitute zero uncertainty. Preserve any
assessable companion interval, but mark the two-grouping envelope unassessable
unless both exist. Undefined means/dispersion and interval fields are JSON
`null` with reasons, not zero, NaN or infinity. An observed constant series may
have descriptive sample SD zero; its inferential uncertainty is still
unassessable under this rule.

These approximate intervals assume dependence is adequately represented by
the chosen clusters, independence across those clusters, finite variance and
enough informative clusters for the approximation. Weekly clusters do not
resolve dependence across weeks, changing regimes, feed errors or prior design
exposure. Report cluster sizes/concentration and these limitations. Neither
clustering nor the envelope restores untouched-sample status or guarantees
95% coverage in this market process.

## 4. Phase B: four-week prospective observation

### Activation manifest and fixed calendar window

A separate, trade-free collector will observe TradeStation minute-bar updates
and ProjectX best-bid/best-ask updates. ProjectX documents `GatewayQuote` on
its market-data hub with `bestBid`, `bestAsk`, `timestamp` and `lastUpdated`;
those provider fields must be preserved without assuming they measure routing
or exchange-to-client latency. [ProjectX realtime documentation](https://gateway.docs.projectx.com/docs/realtime/)

Before collection, commit a separate activation manifest that identifies:

1. This registration's eventual Git revision and SHA-256; the collector,
   configuration, dependencies and synthetic-check artifacts with exact code
   revisions/hashes; a unique run ID and absolute output location.
2. The manifest's commit/activation record in UTC, calendar timezone
   `America/New_York`, exact start/end dates and UTC offsets, all intended RTH
   session windows, holidays and early closes, with cited calendar evidence.
3. Exact TradeStation symbols and ProjectX contract IDs mapped to the same MNQ
   expiry, mapping evidence and any planned roll dates/boundaries. Record
   roll evidence before collection; never choose a contract from later returns.
4. Feed entitlement and real-time/delayed/simulated status for both sources,
   subscription/endpoint identity, and evidence for timestamp, completion and
   revision semantics. Store references to credential configuration, never
   credentials or tokens in the manifest or raw records.
5. Clock/receipt instrumentation, reconnect and gap handling, bar revision
   handling, quote validity/ordering rules below, and output schema versions.

The start is 00:00 America/New_York on the first Monday strictly following
the actual manifest commit date in that timezone (a Monday commit starts the
next Monday). The endpoint is 00:00 on the Monday four calendar weeks later;
the interval is start-inclusive and end-exclusive, covering four consecutive
Monday--Sunday weeks. Record both local and UTC instants, handling DST by
calendar arithmetic rather than assuming a fixed UTC-hour duration. If the
manifest is revised before activation, commit the finalized version and derive
the dates from that commit. The manifest's own hash/revision is bound by the
run record after commit, avoiding a self-referential hash.

The schedule becomes fixed on activation. Holidays, missing days, outages and
late startup consume the budget and do not extend it. Report at the fixed
endpoint without stopping early or extending because results look favorable.
Operational interruptions are recorded as missing coverage within the original
window; they do not create replacement weeks. Four weeks is an observation
budget, not a claim of adequate sample size or sufficient power.

### Continuous capture and causal signal availability

Capture quotes and minute-bar updates continuously throughout scheduled RTH,
including all non-signal periods. Early-close/holiday coverage follows the
manifest calendar and is reported, while frozen complete-session eligibility
remains separate. Preserve raw market payloads and append-only event IDs,
provider timestamps, local UTC receipt timestamps, monotonic receipt times,
process/boot identity, and receive order. Record completion indicators and
their evidence, partial bars, duplicates, out-of-order events, revisions,
subscription acknowledgements, disconnect/reconnect intervals, data gaps and
clock synchronization/offset diagnostics. Never join monotonic clocks across
process restarts as though they shared an origin.

For each candidate, record its bar label, component minute IDs/versions, wick
geometry, first receipt at which the candidate could be recognized, first
verified availability from completed constituent bars, nominal next-bar start,
and scheduled holding end (signal label plus five minutes). Keep a merely
labelled or apparently complete bar distinct from verified completion; passage
of its timestamp alone proves neither finality nor timely receipt. Unresolved
completion makes signal timing `UNASSESSABLE` and is counted explicitly.

Record an assessable decision at the first verified recognition time and retain
the time/receive-order fence for that decision. Preserve later revisions as
new events: a revised candidate is not backdated, an invalidated earlier signal
is annotated, and earlier quote selections are not rewritten. If the first
verified recognition is at or after the scheduled holding end, flag the missed
interval; do not manufacture an entry or extend the holding end. Full-session
eligibility is determined after the session; preserve candidate observations
from ultimately incomplete/mixed-contract sessions with their eligibility flag
instead of using future session completeness to rewrite what was known live.

### Quotes at decisions and scheduled exits

At every assessable decision and scheduled exit, record the latest valid quote
state already received for the mapped contract at that instant, plus the quote
event ID and selection fence. Selection uses local receipt order/monotonic time,
not a provider timestamp that might make a late message appear available
earlier. A scheduled exit uses an as-of fence at its scheduled time even if
its callback runs late; record callback delay separately and exclude quotes
received after the fence. If clock mapping cannot establish that fence, label
the exit observation unverifiable. An exit whose candidate is learned later
may be reconstructed only from contemporaneously captured events already
received by that exit fence and must be labelled a retrospective as-of audit.

A valid quote has finite positive bid/ask, `bid <= ask`, verified contract and
field semantics, and a verifiable current connection/ordering state. Preserve
locked quotes (`bid == ask`) as labelled observations. Missing sides, crossed
quotes, disconnects, unknown contract mapping, ambiguous partial updates or
unresolved ordering produce explicit invalid/unverifiable states. Do not reach
behind a current invalid state for an older attractive quote or carry a cached
quote across a disconnect; resumption requires a verified fresh quote state.
Keep all rejected/raw observations with reasons. Never use a later quote
retrospectively, including one with an earlier provider timestamp.

Report local quote receipt age at the selection fence, provider timestamp age
with clock caveats, decision delay from the nominal signal close/next-bar start,
completion delay, and scheduled-exit callback delay. Keep raw differences and
clock anomalies; do not clamp negative clock differences to zero. No hand-set
spread, quote-age or timing cutoff selects trades or converts observations into
a pass/fail result. Age and delay are continuous measurements. Missing,
crossed, disconnected and unverifiable cases have explicit counts and coverage
denominators rather than disappearing from distributions.

### Observation summaries and execution limits

Summarize spread, quote age, completion/decision/exit delay distributions and
coverage for each session/week and the whole fixed window. Include counts,
empirical quantiles/ECDFs and missingness; distinguish quote-update-weighted,
time-weighted (only across verifiably connected intervals) and candidate-event
summaries so busy quote periods do not silently dominate a time-coverage claim.
Separate signal and non-signal observations, and disclose early closes, rolls,
feed differences, revisions and excluded/incomplete sessions without selecting
a favorable subgroup. No quality, economic or power verdict is inferred from
the four-week duration or these descriptive distributions.

If both causal quotes are assessable, an optional one-contract short proxy is
`2 * (bid_at_decision - ask_at_scheduled_exit)`. Entry reference differences use
the observed bid versus the TradeStation next open; exit differences use the
observed ask versus its next close. Label every such comparison a
**quote-based execution proxy**, retaining quote ages, actual decision delay,
contract mapping and differences between the two feeds. This proxy may include
price movement between nominal and observed decision times; it is not pure
slippage. The bid/ask proxy already includes the observed spread; do not add
that spread a second time as a charge. Keep any assumed fee deduction explicit
and separate from historical reference-cost scenarios. Quote observations
cannot establish order-routing latency, fill slippage, executable size/queue
priority or actual account charges. Incomplete pairs remain missing with
reasons, not filled from later quotes or historical OHLC.

## 5. Required future artifacts and synthetic checks

Future outputs must retain this registration's revision/hash, input hashes,
implementation revision/hashes, run timestamps, schema versions, assumptions,
status/reasons and `evaluation_allowed=false`. Store phase A and phase B in
distinct run directories, with separate denominators and sample-role labels;
neither directory is a confirmation dataset. Do not overwrite a prior run.

| Future artifact | Required contents |
| --- | --- |
| Historical eligibility ledger | Every observed RTH date, contract identity, minute coverage, inclusion/exclusion reasons, ISO year/week, eligible zero-signal sessions, counts reconciled to the gate |
| Historical reference-outcome ledger | Stable signal/session/slot IDs, component provenance, geometry, next open/close, gross and all three assumed net dollar outcomes, associated session counts/totals |
| Prospective raw and coverage/timing records | Append-only quotes/bars and versions, receipts and provider times, clocks, gaps/reconnects, calendar/roll coverage, candidate availability, eligibility, decision/exit fences, selected quote IDs, ages/delays and all missingness reasons |
| Separate JSON and Markdown summaries | Historical means/dispersion/frequency/session totals with both clustered intervals and envelope; prospective spreads/timing/coverage and explicitly labelled optional proxies; limitations and unassessable fields |

Before any later implementation consumes market outcomes or activates the
collector, verify it using synthetic fixtures and independent arithmetic:

- Candle equality boundaries `B > 0`, `U >= 2B`, `D <= 0.1U`, both colours,
  doji rejection, missing/non-finite OHLC and invalid ordering.
- Close-stamped minutes, right-labelled five-minute bars, first/last RTH bars,
  DST, incomplete/short sessions, duplicates and adjacent non-overlapping
  intervals; 15:50 accepted, 15:55 rejected, final allowed exit 15:55.
- UTC cutoff equality and offsets, one-contract versus mixed/roll sessions,
  input/hash mismatches, eligibility reasons and zero-signal session totals.
- Independent short-dollar signs, point value, fee subtraction and aggregation;
  unequal cluster sizes, `G/(G-1)`, `N^2`, t critical values, ISO year transitions,
  both intervals/envelope, zero residual-sum clusters, constant-cost invariance,
  session-level denominators, insufficient observations/clusters and invalid or
  zero variance yielding unassessable inference rather than zero uncertainty.
- Causal quote selection with receipt-order ties, late and out-of-order quotes,
  crossed/locked/missing sides, invalid-current-state handling, quote ages,
  partial bars, completion evidence, candidate revisions and delayed recognition.
- Disconnect/reconnect cache invalidation, lost bars, provider/local clock
  anomalies, process restarts, scheduled-exit callback delays and exclusion of
  every quote received after its fence, including retrospectively audited exits.
- Activation on the first subsequent Monday (including Monday commits), four
  calendar weeks across DST, fixed endpoint, holidays/missing days/late starts,
  no replacement weeks, and no early stopping based on observed results.
- Trade-free operation and isolated outputs: no order methods, live trader
  imports, production ledger writes or changes to the original gate artifacts.

This registration's verification is limited to document scope, provenance,
formula arithmetic and cited references. The checks above are obligations for
future code, not a claim that such code or a collector already exists.

## 6. Later design decision and confirmation reservation

Use calibration estimates as inputs to a separate design decision. A favorable
sample mean, confidence endpoint, quote proxy or old MDE must not become an
assumed future edge by substitution. A new confirmation registration must
separately justify its target effect, cost model, dependence assumptions,
decision rule and prospective sample size, and run a new power assessment before
any strategy test. NIST's sample-size methodology makes the detectable effect,
variability, significance level and target power explicit design inputs; the
clustered market-data assumptions here require their own justification.
[NIST sample-size methodology](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm)

Confirmation is reserved for a separate, prospectively registered sample whose
collection/evaluation window begins after calibration, that design decision,
and the new power assessment. No historical calibration row or four-week
observation may be recycled into it. If effect, execution assumptions or power
remain unsupported, retain `POWER_UNDETERMINED` (or an appropriate later
`UNDERPOWERED` assessment) and do not proceed to confirmation. Preparing this
document preserves the existing gate's `POWER_UNDETERMINED` and
`evaluation_allowed=false` without launching any subsequent phase.
