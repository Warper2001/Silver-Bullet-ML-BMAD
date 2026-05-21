---
title: 'S26 Kill-Zone Filtered Live Pre-Registration'
type: 'feature'
created: '2026-05-21'
status: 'in-progress'
baseline_commit: '470681c5861c4fadc1f685b44350b900e12a7ff1'
context:
  - '{project-root}/_bmad-output/preregistration_s25_live_deployment.md'
  - '{project-root}/data/ml_training/s23_meta_labels_2025.csv'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** 59% of H1·M15·M1·g0.25 trades fall outside ICT kill zones and produce PF=0.967 (net losing), dragging total PF from ~1.49 to 1.17 and Sharpe from ~1.09 to ~0.37. Monday alone produces PF=0.51 with zero TP hits.

**Approach:** Pre-register S26 as a prospective subgroup analysis of S25 live trades. Before any S25 live results are reviewed, commit a time-of-day and day-of-week filter rule. At evaluation close, parse `logs/tier2_filter_log.csv` and apply the filter. No new code deployment; S25 runs unchanged.

## Boundaries & Constraints

**Always:**
- Pre-registration document committed to git before any S25 live trade timestamps are examined
- S26 filter rule is fixed at commit time — no adjustment after observing any live trades
- S25 system configuration stays frozen (commit 69972c3)
- S26 filter is applied to `entry_ts` field in `logs/tier2_filter_log.csv`

**Ask First:**
- If S25 evaluation window closes before S26 reaches N_live ≥ 20, whether to extend S26 independently

**Never:**
- Modify S25 Tier2StreamingTrader config for S26
- Use sealed holdout data (`data/sealed_holdout/`)
- Block Wednesday (no ICT theoretical basis; risk of over-fitting in-sample DOW pattern)
- Retroactively change the kill-zone windows after seeing live timestamps

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Normal kill-zone trade | `entry_ts` hour in [10,11,14] ET, dow not Mon/Tue | Included in S26 sample | N/A |
| Outside kill zone | `entry_ts` hour not in [10,11,14] ET | Excluded from S26 sample | N/A |
| Monday trade | `entry_ts` dow == Monday | Excluded from S26 sample | N/A |
| Tuesday trade | `entry_ts` dow == Tuesday | Already blocked by S25; will not appear in log | N/A |
| Empty log at eval time | N_live < 20 after 90 days | `insufficient_live_sample` verdict | Pre-committed: re-evaluate frequency |

</frozen-after-approval>

## Code Map

- `_bmad-output/preregistration_s25_live_deployment.md` -- format template for S26 doc
- `data/ml_training/s23_meta_labels_2025.csv` -- 2025 in-sample labeled trades; `hour_et` and `dow_et` columns used for baseline validation
- `logs/tier2_filter_log.csv` -- S25 live trade log; S26 filter applied to this at evaluation time
- `_bmad-output/preregistration_s26_kz_filtered_live.md` -- PRIMARY DELIVERABLE (create)
- `s26_kz_validate.py` -- validation script to confirm 2025 in-sample baseline (create)

## Tasks & Acceptance

**Execution:**
- [ ] `s26_kz_validate.py` -- CREATE: load `data/ml_training/s23_meta_labels_2025.csv`, apply the S26 filter (`hour_et in [10, 11, 14]` AND `dow_et != 0`), compute N, PF, win rate (TP), win rate (pnl>0), annualized Sharpe, and annual P&L at 1x and 5-contract size. Print a formatted report and write to `data/reports/s26_kz_validate_<timestamp>.txt`.
- [ ] `_bmad-output/preregistration_s26_kz_filtered_live.md` -- CREATE: full pre-registration document using s26_kz_validate.py output as the in-sample reference numbers. Model structure exactly on `preregistration_s25_live_deployment.md`. Must include: architecture table, filter definition, in-sample baseline table, hypothesis, evaluation window, and decision rule table.
- [ ] `s26_kz_validate.py` -- RUN: execute on 2025 data, capture output, verify numbers are internally consistent before writing the pre-registration doc.
- [ ] `_bmad-output/preregistration_s26_kz_filtered_live.md` -- COMMIT: `git add` + `git commit` with message referencing S26 pre-registration. This commit SHA is the pre-registration gate.

**Acceptance Criteria:**
- Given the pre-registration doc is committed to git, when `s26_kz_validate.py` is re-run, then the N and PF values in the doc match the script output within ±1 trade and ±0.01 PF
- Given a future `logs/tier2_filter_log.csv` entry with `entry_ts` at 10:30 AM ET on a Thursday, when S26 filter is applied, then the trade is included in the S26 sample
- Given a future log entry at 09:15 AM ET or any time on Monday, when S26 filter is applied, then the trade is excluded
- Given N_live_filtered ≥ 20 and evaluation window has closed, when S26 PF is computed from the filtered live log, then the pre-committed decision rule (identical in structure to S25's) determines the verdict with no discretion
- Given the pre-registration doc contains a hypothesis with a specific PF threshold, when the commit SHA is recorded in the doc, then any post-registration parameter change is detectable via `git diff`

## Spec Change Log

## Design Notes

**S26 filter definition (exact, implementation-ready):**
```python
import pytz
ET = pytz.timezone("America/New_York")

def is_s26_eligible(entry_ts_utc):
    """True if this trade counts toward S26 evaluation."""
    et = entry_ts_utc.astimezone(ET)
    hour = et.hour
    dow  = et.weekday()          # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    in_kz  = hour in (10, 11, 14)  # 10:00-12:00 ET or 14:00-15:00 ET
    blocked = dow in (0, 1)         # Monday + Tuesday
    return in_kz and not blocked
```

**Why subgroup of S25 (not separate deployment):** S25 already logs all qualifying H1·M15·M1·g0.25 signals. Applying a pre-committed filter to the log at evaluation time is equivalent to running a pre-registered subgroup analysis — valid under Program C provided the rule is committed before any live trade timestamps are examined.

**In-sample reference (S23, 2025):** The `s26_kz_validate.py` script populates the exact numbers. Expected range based on kill-zone analysis: N≈45–55, PF≈1.48–1.60, Sharpe≈1.0–1.4.

**S26 hypothesis threshold:** Set at S25's `live_edge_confirmed` lower bound (PF > 1.1350) — same S12 random baseline. If S26 PF > S25 in-sample KZ PF (≈1.54), a secondary `exceeds_inscope_insample` verdict applies.

**Evaluation window:** Start = date of S26 pre-registration commit. End = when N_live_filtered ≥ 20 AND 60 calendar days elapsed (whichever is later). Maximum = 180 calendar days (S26 trades less frequently than S25 due to filter).

## Verification

**Commands:**
- `.venv/bin/python s26_kz_validate.py` -- expected: prints N≥40, PF≥1.40, exits 0
- `.venv/bin/python -c "import ast; ast.parse(open('s26_kz_validate.py').read()); print('OK')"` -- expected: `OK`
- `git log --oneline -1 -- _bmad-output/preregistration_s26_kz_filtered_live.md` -- expected: shows commit SHA (pre-registration gate sealed)
