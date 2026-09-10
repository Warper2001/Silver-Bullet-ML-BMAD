# Contract evidence adapter

Run a finite poll from the repository root:

```bash
.venv/bin/python -m research.mim_comparison.feed_adapter \
  --bars data/mim_nb/bars_raw.csv --log logs/mim_nb_live.log \
  --log-timezone UTC \
  --state research/mim_comparison/runs/20260910-contract-feed
```

The launcher requires Linux bubblewrap and x86-64 seccomp. It mounts only the
adapter package, Python runtime, read-only source inputs and isolated writable
state. Host home/credentials are hidden, all socket creation is denied, and no
production code is imported. There is no service or deployment step.

Use this state's `feed.csv` as the existing comparison CLI's shadow `--data`,
with a separate existing shadow `--state`. Retain the original historical run,
protocol, timestamp labels and warmup `data/mim_x/mnq_1min_by_contract.csv`.
The adapter never changes the original nine-month freeze. Warmup currently ends
August 28; unavailable prior-session contract coverage can exclude initial
sessions. The supported universe is explicit quarterly MNQ H/M/U/Z contracts.
Unidentified or skipped bars cannot establish eligible flat sessions: the
unchanged shadow collector requires every minute and excludes missing weekdays.

Identity is **log-inferred provenance, not authenticated response payload
identity**. A successful exact one-minute TradeStation GET must complete within
15 seconds before the original receipt, with a unique contract and matching
known TradeStation signal DATA context. Startup invalidates DATA context.
AUTOROLL log transitions update context only causally; request evidence still
governs each bar. Order records never relabel bars. Repeated same-contract log
lines are permitted; different contracts or incompatible contexts are excluded.
Joins are indexed and limited to 1,000 requests per receipt window.

The adapter preserves original OHLCV/event/receipt text. Historical mapping is
never a prospective result. Initial source rows are historical replay; initial
rows recent enough to appear timely are explicitly excluded. Later rows retain
original receipt and actual adapter observation time. The existing shadow
collector separately applies its original receipt and actual wall-clock <=60s
checks. Adapter `timely_at_adapter` is diagnostic, not session eligibility.

Every raw first observation is journaled durably before the next bar. Identity
is the normalized UTC event time, conservatively across all contracts. An
identifiable malformed first row tombstones that event permanently. A row whose
event cannot be identified permanently stops subsequent mapping in that state;
no placeholder contracts are emitted. Partial final lines wait for newline.
The bar byte boundary is taken before the log boundary. Log evidence arriving
on a later poll cannot reinstate an excluded first observation.

Hash-chain policy: exclude each broken-link row, retain its expected/recorded
hash and increment the segment. Later matching links may be mapped, but are
explicitly `unanchored_after_break`; they never regain GENESIS linkage. This
does not repair any source or authenticate a response. Frozen consumed-prefix
hashes and device/inode identities provide local observation stability. Source
replacement, truncation or changed previously observed bytes fail closed.
Mutation detected during consumption permanently stops that state before feed
publication. Sources/config/paths/timezone freeze before attribution; resume
rejects drift. Do not edit adapter code while continuing an existing state.

`journal.sqlite` is the authoritative append-only observation/decision history
(with mutable cursors and incremental indexes). `feed.csv` is append-only;
recovery checks its committed byte prefix and reconciles any torn final append
against journaled canonical bytes. Concurrent CLI invocations fail on a lock.
Each uniquely named `poll-*` contains report, manifest and per-poll raw evidence/
exclusion snapshots. Failed polls also receive sealed diagnostic artifacts;
journal decisions not yet snapshotted are included on the next successful poll.
Artifacts are hash-manifested and read-only, not protected against a privileged
operator. Counts are cumulative and commit in the same transaction as each row.
No polling invocation rescans historical rows in SQL; prefix file hashing is
necessarily linear in consumed input size, and snapshots cover new decisions.

## Finite bounded-reader poll

The full authoritative feed can contain tens of thousands of replay records.
Use the separate operational wrapper to keep newly appended records near the
front of the unchanged shadow reader:

```bash
research/mim_comparison/poll.sh
```

Each invocation runs the frozen adapter, verifies the committed `feed.csv`
prefix against its journal hash under the adapter lock, then scans backwards
for the last 500 complete single-line records. It atomically replaces
`runs/20260910-contract-feed-poll/window.csv` with the original header and those
unchanged records. The adjacent manifest identifies the full source prefix,
window hash, row count and pending partial bytes. The window is a derived
snapshot, never the authoritative feed. No source feed bytes are changed.
Complete uncommitted tails require adapter recovery; incomplete tails remain
pending. Failed verification leaves the previous window intact.

Before calling shadow, the wrapper ensures the isolated collector's
`invalid_rows(event,contract)` lookup has the `mim_feed_event_contract` SQLite
index. This avoids repeatedly scanning large raw exclusion payloads; it changes
only database indexing and preserves all observations and decisions. If the
table does not yet exist, index creation is skipped.

The shadow invocation retains historical run
`20260910T210224-historical-f3950efb68`, the original warmup and labels, and
collector `20260910T214803-shadow-47055eba62/collector`. Existing collector
observations persist across windows. Five hundred rows exceed a 390-minute RTH
session; poll regularly so new records are observed within the original 60-second
limit. Outages, missed rows, alternate-contract coverage and missing warmup
remain unavailable under the unchanged rules. The wrapper does not extend the
original freeze/deadline or make late records timely.

The wrapper holds a separate invocation lock throughout adapter, preparation
and shadow. Its source hash and configuration freeze in its own state before
the first poll; drift fails closed without changing the adapter freeze. It runs
once and exits; no persistent service is installed.

Fixture preparation can run independently without starting either collector:
`poll.sh --prepare-only --feed research/mim_comparison/runs/FIXTURE/feed.csv
--state research/mim_comparison/runs/FIXTURE-WINDOW`. The fixture feed requires
adapter-format committed metadata in its adjacent `journal.sqlite`.
An optional `--collector-journal` exercises only the index preflight on an
isolated fixture database. Every writable path must remain under research runs;
fixture overrides and custom state directories are refused in live mode, so all live wrapper invocations share one lock. Prepare-only defaults to a separate `20260910-contract-feed-poll-prepare` state and refuses the live state before writing; prepare-only configuration is frozen too.

The derived window and manifest are individually atomic and can temporarily disagree after an interrupted publication. Neither is authoritative. The wrapper validates the exact returned manifest and window SHA256 before dispatching shadow; failed publication or validation prevents dispatch. A later successful poll republishes the pair.
