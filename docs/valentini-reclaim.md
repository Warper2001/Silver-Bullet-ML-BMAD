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
