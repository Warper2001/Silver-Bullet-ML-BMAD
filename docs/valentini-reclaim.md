# Gated Globex value-area reclaim research harness

`tools/valentini_reclaim.py` is standalone research code. It imports no trader and
writes no trade ledger. The simulator is tested only on synthetic fixtures. This
release deliberately cannot evaluate market performance or issue `POWERED`.

Run from `/root/Silver-Bullet-ML-BMAD/.claude/worktrees/valentini-globex`, using the
existing interpreter; install nothing:

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/test_valentini_reclaim.py -q
/root/Silver-Bullet-ML-BMAD/.venv/bin/python tools/valentini_reclaim.py audit --csv /absolute/path/bars.csv --csv /absolute/path/other-bars.csv --output /tmp/valentini-audit.json
/root/Silver-Bullet-ML-BMAD/.venv/bin/python tools/valentini_reclaim.py power --audit /tmp/valentini-audit.json --output /tmp/valentini-power.json
/root/Silver-Bullet-ML-BMAD/.venv/bin/python tools/valentini_reclaim.py evaluate --audit /tmp/valentini-audit.json --gate /tmp/valentini-power.json
```

The last command exits 2 with a JSON refusal and produces no ledger. `audit` and
`power` exit 0 when they successfully produce a terminal report; this is not data
admission or strategy approval. File/schema failures exit 2. There is no force or
bypass option. Output directories must already exist. Output may not overwrite
its explicit input or audited market data, including hard-link aliases.

## Metadata audit and evidence boundary

The CSV contract has case-sensitive columns `timestamp,open,high,low,close,volume`;
additional columns are listed but not interpreted. Timestamps must carry an
explicit UTC offset and lie on minute boundaries. The audit does not guess a
naive timestamp's timezone, rename columns, aggregate bars, fill missing minutes,
deduplicate, or remove invalid rows. Prices must be finite positive multiples of
0.25 with valid OHLC ordering. Volume must be finite and nonnegative; actual
zero-volume input bars are legal, fabricated zero-volume bars are not.

The audit streams CSV rows and SHA-256 hashes, records counts, timestamp range,
invalid-row examples, duplicates, reversed order, gaps, schema findings, and file
hashes. Invalid rows remain counted. Timestamp range and cadence findings concern
valid rows; invalid rows prevent admission and are never silently accepted.
Duplicate checks are within each file; cross-file overlap is not certified.
Clock gaps include legitimate overnight/holiday breaks: without an independent
calendar their meaning is unresolved. A set of observed timestamps is retained
for duplicate detection, so memory use grows with unique rows. Files are hashed
before and after parsing and rejected if the hashes differ.

Requested paths, every intermediate symlink target, and resolved paths are checked. Access to
`sealed_holdout`, `.env`, `.access_token`, and `trades.db` is refused. This harness
does not request credentials or connect to brokers.

Every audit explicitly leaves timestamp start/end-label semantics, the dated
session calendar, fixed-minute source provenance, and contract provenance
`UNKNOWN`. A filename is never evidence of suitability. The known file hash
`3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822` is flagged
as dollar-aggregated and unsuitable for fixed-minute inference.

CSV parsing is strict: malformed quoting rejects the audit. JSON objects with
duplicate keys are rejected rather than taking the last value.

`power` rechecks the audit version, current module hash, and every input hash. An
optional `--evidence /absolute/path/evidence.json` accepts a strict JSON object;
`transferable_effect` and `independent_calibration` fields can describe candidate
evidence, but their presence does not certify it. The evidence hash is bound into
the gate. Supply the same `--evidence` to `evaluate` if one was used for `power`.
User-authored booleans, verdict strings, and effect estimates cannot promote the
study. Unknown data admission produces `verdict: DATA_UNSUITABLE`, with a separate
`power_status: POWER_UNDETERMINED`; this release has no evidence validator capable
of removing the admission blockers. It computes neither nulls nor returns on
these inputs and does not call an absent effect `UNDERPOWERED`.

`evaluate` reads only admission artifacts and evidence hashes, checks their
version/code/audit/evidence binding, and refuses before accessing market rows or
executing the simulator. Even a forged `POWERED` file with consistent hashes is
rejected because independent promotion is unsupported. Input hashes are verified
by `power`; `evaluate` has no executable market-data path in this release.

Future promotion needs an independently implemented and tested validator for
source timestamp semantics, complete fixed-minute bars, a dated exchange schedule
including maintenance halts, early closes and DST, contract identity and roll
handling, and hash-bound provenance artifacts. It also needs a transferable effect
estimate from data this choice did not see, a prespecified dependence-aware null,
cost assumptions, MDE/power calibration, and independently verifiable gate
production. A future gate must revalidate all bindings before admitting market
evaluation. These are incomplete capabilities, not evidence against the strategy.

## Pure synthetic engine contract

The Python API accepts typed `Bar`, `Session`, and `Costs` objects and returns
`Simulation(signals, trades)` without persistence. `Session.start` is inclusive,
`Session.end` exclusive; both use independently supplied UTC minute boundaries.
`Session.breaks` optionally supplies sorted nonoverlapping `(start, end)` UTC
intervals of scheduled nontrading time within the same session. This can represent
a dated full Globex session's trading halt without resetting its value profile.
No exchange calendar is inferred from observed bars. The caller defines the
correct complete schedule for each date, including shortened sessions and DST.

The complete session is validated before any signal calculation. There must be
exactly one bar per expected tradable minute in order, with no duplicates, invalid
OHLCV, off-tick prices, missing minutes or extra bars. `simulate_sessions` validates
all supplied sessions before calculating any result; bars outside the schedule,
overlapping sessions and duplicate session names are rejected. There is no
partial-session acceptance. Full Globex means all supplied tradable minutes, with
no RTH entry restriction.

The developing profile allocates each completed bar's volume uniformly over
inclusive tick levels from its low to its high. A zero-volume profile has no
value area. The POC is the highest-volume tick with lower-price ties. From the
POC, the area expands contiguously toward the higher-volume adjacent tick until
it contains at least 70% of accumulated volume; ties expand lower. Empty levels
between traded ticks have zero volume. Current-bar volume never enters the
profile used to assess the current bar's break.

A long setup requires the preceding close inside the pre-break VAL/VAH, current
close strictly below VAL, and current volume strictly below the immediately
preceding minute's volume. Freeze VAL and VAH and track the lowest low from the
break through the reclaim, inclusive. The first later close at or above frozen
VAL confirms only when it is at or below frozen VAH and volume is strictly above
the preceding minute's volume. An overshoot or equal/lower reclaim volume cancels
the setup, even if a later bar would qualify.

Enter one MNQ contract at the next contiguous minute's open plus adverse
slippage. Both raw open and slipped entry must lie strictly above the stop and
below the target. Otherwise cancel the pending entry. The stop is one tick below
the full excursion low; the target is the frozen VAH. A final-minute confirmation
is recorded as a signal but cannot become a trade. A scheduled halt cancels
setups and pending entries and prevents adjacent-volume comparisons across the
halt; it preserves the profile and any already open position. Session boundaries
reset all state, and missing unscheduled minutes reject the session entirely.

Stops are assessed on the entry bar as well as later bars. For an open position,
an opening price at or below stop exits at `open - slippage`; an opening price
strictly above target exits exactly at target. These known opening events precede
later intrabar extrema. If neither exit occurs at the open and stop and target
are both touched intrabar, stop wins. An intrabar stop fills at `stop - slippage`. A target requires high
strictly above target (a touch does not fill) and fills exactly at target, with
no favorable gap improvement. A surviving position exits at the last tradable
bar's close minus slippage. No setup is armed on the same bar as an exit, entry
cancellation or reclaim cancellation; a fresh break can arm on the next bar.

`Costs` requires explicit nonnegative finite `slippage_points` and
`commission_per_side`; there is no empirically asserted default. Commissions are
charged on both fills, including limit targets. `Trade` records signal levels,
times, entry/exit, exit reason, one contract, gross dollars, commission dollars
and net dollars at $2/point. `exit_bar_time` is always the exit bar's minute-start
label. `exit_time` is the bar-open timestamp when the opening price establishes
a stop or target exit. It is null for intrabar stop/target fills because OHLCV
cannot establish an intrabar execution timestamp. For `session_end`, it is the final tradable
minute's close boundary (`exit_bar_time + one minute`), including when a scheduled
halt follows that minute. `entry_time` is the known next-bar opening boundary.
Finite inputs that overflow accumulated profile volume or calculated fills,
commissions or results are rejected. Prices already incorporate slippage, so gross dollars
are after slippage and before commission. The pure engine has no profitability
reporting or file-writing side effects.

## Synthetic null/MDE utility and limitations

`derangements(size, draws, seed)` reproducibly supplies complete permutations
without identity or any fixed point. `null_mde(panel, pairings, effect=...,
alpha=..., power=..., dependence_factors=...)` accepts a square synthetic panel:
cell `[i][j]` represents setup session `i` paired with outcome path session `j`.
It uses only mismatched off-diagonal cells. Matched diagonal cells can be NaN to
verify they are never used; selected off-diagonal cells must be finite. Every
pairing must use each destination exactly once and must have no fixed points.

The helper reports each pairing's mean, their mean and sample standard deviation,
and illustrative one-sided normal-approximation MDEs:
`(z(1-alpha) + z(power)) * null_sd * sqrt(variance_inflation)`. Caller-supplied
variance inflation factors (at least one, each >= 1) expose dependence sensitivity.
Pairing draws may repeat and are not claimed to be independent observations, but
calibration requires at least two distinct derangements and positive finite null
spread. Repeated single pairings, two-session panels (only one derangement), and
zero-spread nulls are rejected as degenerate; they cannot report a zero MDE.
The helper does not reconstruct paths, validate transferability, or estimate real
session dependence. Effect is explicitly supplied or absent; there is no borrowed
empirical effect constant. Its verdict is always `SYNTHETIC_CALIBRATION_ONLY` and
it cannot authorize evaluation. There is no CLI that runs this helper on market
data.

Uniform OHLCV allocation is a disclosed proxy, not actual volume at price. It
smears volume across potentially untraded prices and loses intrabar event order.
Stop-first ambiguity and target trade-through are conservative modeling choices,
not verified execution fidelity. Synthetic correctness establishes implementation
mechanics only, not data suitability, statistical power or trading profitability.
JSON artifacts use sorted keys, no timestamps/random run IDs, explicit code/input
hashes, and reject nonfinite numbers; deterministic fields match for identical
paths, bytes and code.

## Native measurement audit

`tools/valentini_native_audit.py` measures MNQM5 native traded volume at price
against this harness's uniform OHLCV profile. It never invokes the simulator,
replay runners, broker code, or market-evaluation admission. From the isolated
`valentini-native` worktree, using an output directory that does not yet exist:

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python tools/valentini_native_audit.py \
  --source-root /root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907 \
  --reconstruction /root/Silver-Bullet-ML-BMAD-yank-minute/data/yank/native-minute-reviewed-a \
  --calendar docs/valentini-native/calendar-2025-05.json \
  --output-dir /tmp/valentini-native-measurement
```

For a full scan, use `nohup` with an operational log outside the output directory.
The parent directory must exist. The audit rejects existing output paths,
symlink/hardlink aliases and outputs inside supplied input directories. It hashes
all pinned acquisition files before and after the scan, validates definitions and
status, and reconciles six reconstructed data artifacts plus saved counts against
seven frozen core artifact hashes. It deliberately does not load saved replay
results. Acquisition pins must also match the saved report's acquisition list.
A mismatch exits 2 and publishes no result directory. Cached decoder code is
bound to the bytes loaded. Publication holds a directory descriptor opened
without following symlinks, so a later parent swap cannot redirect writes.

The dated [calendar evidence](valentini-native/calendar-sources.md) defines explicit
UTC sessions and breaks for MNQM5 in America/New_York. Other instruments or
timezones and nonzero fractional minute boundaries are rejected. The parser
does not infer a session from observed rows.
Unverified calendar sessions, partial acquisition, unexplained missing minutes,
and calendar/status conflicts are excluded in `sessions.json`. Exact source-bound
exchange transitions at calendar edges can explain nanosecond-late capture status;
there is no arbitrary timing tolerance. The closing boundary also needs coverage
and nontrading evidence; an exact sourced closing transition can explain mixed
capture status. Missing source files during scheduled breaks exclude the session.
Entire inter-session nontrading intervals are not certified. Observed bars outside
the calendar remain visible in `observed-bars.jsonl`.

Eligible sessions compare developing profiles before each bar. Both profiles use
the same preceding bars, preserve profiles across explicit breaks and reset at
session boundaries. A cumulative LAST-event availability watermark excludes an
unavailable snapshot permanently; later boundaries may include the completed
prefix. Native volume is summed in integer contract/tick histograms. The native
70% area uses exact integer arithmetic and the existing lower-price POC/adjacent
expansion tie rules; the proxy uses the existing `Profile` implementation.

`histograms.jsonl` retains sorted native tick-volume pairs per capture minute;
`snapshots.jsonl` reports native and proxy VAL/VAH/POC in ticks, signed differences
(proxy minus native), absolute differences, prior-bar counts and integer volume
denominators. `report.json` summarizes agreement, mean, median and linearly
interpolated quantiles both pooled and per session, with explicit denominators
and snapshot exclusions. Excluded sessions retain zero-comparison summaries.
`provenance.json` binds code, runtime, calendar and input hashes; `manifest.json`
hashes every published artifact. Artifacts exclude timing logs and output paths,
so identical code/input bytes produce identical results in different output
directories. `NO_ELIGIBLE_COMPARISONS` publishes the explanatory artifacts but
exits 3. A completed measurement with comparisons exits 0.

A completed measurement still reports `market_evaluation: NOT_ADMITTED`.
Reconciliation proves agreement with the saved native reconstruction, not
independent exchange-feed completeness, strategy suitability or statistical
power. Calendar evidence, measurement exclusions, future independent testing,
and the existing market-evaluation refusal remain distinct.

The completed pilot's [measurements and recommendation](valentini-native/results.md)
include all six admitted sessions and links to committed evidence.
