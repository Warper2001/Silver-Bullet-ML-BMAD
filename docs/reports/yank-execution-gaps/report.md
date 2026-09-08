# Incremental frozen execution gaps

HOLD_VALIDATION. 30 scenarios; 11 supported, 19 unassessable; no outcome changes.

| Case | Label | Delay ms | Frozen outcome | Attributed blockers |
|---|---|---:|---|---:|
| case-1 | start | 0 | supported | 0 |
| case-1 | start | 100 | supported | 0 |
| case-1 | start | 500 | supported | 0 |
| case-1 | end | 0 | unassessable | 1 |
| case-1 | end | 100 | supported | 0 |
| case-1 | end | 500 | supported | 0 |
| case-2 | start | 0 | supported | 0 |
| case-2 | start | 100 | supported | 0 |
| case-2 | start | 500 | supported | 0 |
| case-2 | end | 0 | supported | 0 |
| case-2 | end | 100 | supported | 0 |
| case-2 | end | 500 | supported | 0 |
| case-3 | start | 0 | unassessable | 53 |
| case-3 | start | 100 | unassessable | 53 |
| case-3 | start | 500 | unassessable | 53 |
| case-3 | end | 0 | unassessable | 53 |
| case-3 | end | 100 | unassessable | 53 |
| case-3 | end | 500 | unassessable | 53 |
| case-4 | start | 0 | unassessable | 53 |
| case-4 | start | 100 | unassessable | 53 |
| case-4 | start | 500 | unassessable | 53 |
| case-4 | end | 0 | unassessable | 53 |
| case-4 | end | 100 | unassessable | 53 |
| case-4 | end | 500 | unassessable | 53 |
| case-5 | start | 0 | unassessable | 53 |
| case-5 | start | 100 | unassessable | 53 |
| case-5 | start | 500 | unassessable | 53 |
| case-5 | end | 0 | unassessable | 53 |
| case-5 | end | 100 | unassessable | 53 |
| case-5 | end | 500 | unassessable | 53 |

Blocker counts overlap; they are not independent failed scenarios. The JSON retains each native record/event reference, file hash, recovery boundary and full-window intersection.

Case 1 is an event spanning arrival; its later terminator cannot supply a strictly prior completed book. Cases 3–5 combine invalid completed books with observed market-event and scheduled nontrading statuses. Nontrading is observed evidence, not presumed missing acquisition. Invalid book recovery does not validate preceding events. Hypothetical queue position remains inherently unobservable.

Disposition details and exact additional evidence requirements are recorded separately for every blocker in report.json. Neither this report nor any local engineering pass establishes profitability.
