# Wick-short: PARKED (Alex, 2026-09-21)

**Decision:** park. No pre-registration, collector, unit, timer or further calibration is to be started. Reopen only through a new pre-registration.

## Why (Phase A calibration, corrected r2 publication)
Source: `mnq-wick-short-calibration-phase-a-execution-r2-20260916.md`, implementation revision `7ab4c9a20fde3d9aa3358abc4de42d2dc7060229`,
calibration registration revision `3167d5354d2b71af413e9e2404b69efa82df96fc`. The run is independently audited and its own front matter says:
`sample_role: calibration-development-only`, `evaluation_allowed: false`, `confirmation_authorized: false`, `collector_activated: false`,
`original_gate_verdict: POWER_UNDETERMINED`. **It is calibration only and carries no verdict.**

Counts: 576 observed dates, 515 eligible sessions, **585 signals on 351 sessions**, 164 eligible zero-signal sessions (1.136 signals per eligible session).
Gross per-signal sample dispersion: $50.94.

| historical reference outcome | mean $/signal | outer envelope of approximate 95% intervals |
|---|---|---|
| gross | **-4.09** | [-8.45, +0.28] |
| assumed net, $1.22 cost | -5.31 | [-9.67, -0.94] |
| assumed net, $2.22 cost | -6.31 | [-10.67, -1.94] |
| assumed net, $3.22 cost | -7.31 | [-11.67, -2.94] |

Every net interval lies below zero and the gross interval barely reaches zero. The measurements estimate size, variability and frequency in an
already-exposed sample; they are not evidence for or against an executable edge.

## State at parking
- Committed and left as they are: `tools/{mnq_wick_short_power_gate,mnq_wick_short_calibration,audit_mnq_wick_short_source}.py`, their four unit tests, the two
  registrations (`preregistration_mnq_wick_short_{power,calibration}_20260916.md`) and the Phase A result files.
- No systemd unit or timer for wick-short exists (checked 2026-09-21).
- Nothing about it is live, and nothing live depends on it.

## Reopening
Only with a **new pre-registration** that names a fresh, unseen sample and states its power before any outcome is computed. The exposed calibration sample cannot be reused as confirmation, and the Phase A numbers must not be recut into a filter (policy: no restriction chosen on past data without retesting on data it did not see).
