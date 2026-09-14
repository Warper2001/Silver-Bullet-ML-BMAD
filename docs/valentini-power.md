# Native reclaim conditional power feasibility gate

`tools/valentini_power_gate.py` uses the committed native measurement's session
ledger and provenance for prospective statistical design. It hashes all six
measurement artifacts but parses only report, sessions and provenance. It does
not decode native data, generate signals or calculate strategy outcomes.

Run from the `valentini-power` worktree **after the reviewed code/tests are
committed**, using a new output directory whose parent already exists:

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python tools/valentini_power_gate.py \
  --audit-dir _bmad-output/valentini-native-20260914/run-final \
  --prereg _bmad-output/preregistration_valentini_native_power_20260914.md \
  --inventory _bmad-output/valentini-power-20260914/inventory.json \
  --output-dir _bmad-output/valentini-power-20260914/run-final
```

The preregistration, inventory and measurement manifest have fixed SHA256 pins.
All measurement artifact hashes and provenance code hashes must agree. The gate
checks inputs again after calculation and publishes canonical `report.json`
and a hash-binding `manifest.json` without overwriting an existing destination.
Invalid inputs, protected paths, aliases and collisions return exit code 2;
a valid terminal report returns 0.

The conditional model is a one-sided t test of positive mean net session profit,
alpha 0.05 and target power 0.80, assuming hypothetical iid normal outcomes with
unknown variance. The report includes the 80% detectable standardized effect
frontier and all prespecified effects 0.1, 0.2, 0.3, 0.5 and 1.0. Minimum integer
session counts use bracketed binary search with an adjacent-count check and a
1,000,000-session cap. Counts below two are explicitly unassessable. A separate
known-variance normal calculation is a cross-check, not the t-test result.

Here `d = hypothetical mean net session dollars / population standard deviation
of net session dollars`. Every effect is hypothetical: no mean, variance or
net-cost estimate exists here.
Profiles and dependent comparisons are not independent trades or a calibrated
effective sample size. Conditional sample counts are neither guaranteed horizons
nor data-purchase authorization. The iid assumption is not a universal bound.

The operational verdict is always `POWER_UNDETERMINED`, with
`evaluation_allowed=false` and `market_evaluation=NOT_ADMITTED`, even when a
hypothetical scenario reaches 80%. Missing calibration is not an UNDERPOWERED
verdict. Proceeding requires representative full native sessions, calibration and
validation roles fixed before outcomes, defensible effect/cost assumptions, a
suitable dependence-aware test and a subsequent preregistration.

The report preserves exclusions, includes scoped inventory evidence and records
registration commit, input/code hashes, runtime versions and selected dependency
module hashes. Raw source hashes are historical audit evidence: this gate does
not repeat raw-native reconciliation. Module hashes cover selected loaded files,
not the entire installed environment.

Synthetic verification (no actual-ledger calculation):

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/test_valentini_power_gate.py -q
```

Tests independently integrate the chi-square representation of noncentral t
power and cover null alpha, monotonicity, MDE inversion, adjacent sample-size
boundaries, malformed inputs, artifact mutation and output protection.

The completed [2026-09-14 feasibility result](valentini-power-results.md) reports all scenarios and immutable execution evidence.
