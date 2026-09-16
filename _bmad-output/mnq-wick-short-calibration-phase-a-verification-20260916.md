---
id: mnq-wick-short-calibration-phase-a-verification-20260916
phase: A
sample_role: calibration-development-only
evaluation_allowed: false
original_gate_verdict: POWER_UNDETERMINED
historical_run_started: false
---

# Phase A implementation verification before historical execution

The implementation and this verification record are committed before the runner
reads the bound historical container. The registration and all eight other
pinned original/calibration artifacts match their frozen SHA-256 values.
The original gate, original mechanics tests, strategy configuration, live code,
services and production ledgers have not been changed.

Interpreter: `/root/Silver-Bullet-ML-BMAD/.venv/bin/python`, Python 3.12.3.
Installed dependencies observed without installation: numpy 2.4.6, scipy 1.17.1,
pytest 9.0.2. The runner records interpreter and installed dependency file
inventory hashes in each completed report.

## Commands and results

All commands ran from the designated Phase A worktree. The common executable
prefix below is `/root/Silver-Bullet-ML-BMAD/.venv/bin/`.

- `python -m pytest tests/unit/test_mnq_wick_short_calibration.py tests/unit/test_mnq_wick_short_power_gate.py -q`: **61 passed** in 1.96 seconds. All fixtures are synthetic; no market data or credentials were used.
- `black --check tools/mnq_wick_short_calibration.py tests/unit/test_mnq_wick_short_calibration.py`: passed.
- `flake8 --max-line-length=88 tools/mnq_wick_short_calibration.py tests/unit/test_mnq_wick_short_calibration.py`: passed; 88 matches the repository Black configuration.
- `mypy --follow-imports=silent --explicit-package-bases --disable-error-code=import-untyped tools/mnq_wick_short_calibration.py tests/unit/test_mnq_wick_short_calibration.py`: passed, two source files. Namespace-package handling avoids duplicate module identities; only third-party untyped-import diagnostics are disabled because SciPy stubs are absent. No repository type-ignore override or package installation was added.

## Contract coverage

The tests exercise valid bullish and bearish OHLC, wick equality boundaries,
dojis, missing/non-finite fields, invalid ordering, offset-free and malformed
timestamps, equivalent-offset duplicates, strict cutoff equality, DST, complete
and short sessions, mixed-contract and overlapping exclusion reasons, right
labels and component minutes, final signal/exit slots, actual adjacent interval
prices, short-dollar signs, all costs and eligible zero-signal totals.

Independent statistical fixtures cover unequal clusters, the finite-cluster
correction, known Student-t critical values at one and two degrees of freedom,
both interval endpoints and their envelope, zero-valued nonempty residual
clusters, per-signal cost invariance, varying session costs/counts, ISO-year
transitions, invalid or constant variance, non-finite observations and arithmetic
overflow. Balanced decimal-valued clusters also yield unassessable inference:
exact rational residual arithmetic prevents floating-point cancellation from
inventing positive variance and uses no hand-set variance tolerance.

Refusal/publication tests cover alternate and sealed input paths, prohibited or
existing output destinations, symlink destinations, input hash mismatches,
malformed input, gate count mismatches before alignment, changed provenance,
canonical pins, uncommitted/staged code, interrupted publication without a
completion marker, and output hash reconciliation. The synthetic CLI disables
old gate execution, shifted outcomes and old SE/publication paths explicitly.

## Before-run implementation bytes

| File | SHA-256 |
| --- | --- |
| `tools/mnq_wick_short_calibration.py` | `21c72585c22c697520269bf54dbf899b908b2e67d5f357363f305d175d60c9b3` |
| `tests/unit/test_mnq_wick_short_calibration.py` | `e79e41b16329bbd2b1ce5fef03fae8cfbf012ac255a7023395c38f3c8c84ec38` |
| `tests/unit/test_mnq_wick_short_power_gate.py` (unchanged) | `385d397521d15ee7f60907ef11e1a384a0f880cf48f725ab8291874a0dc13a85` |
| `pyproject.toml` (unchanged) | `85a7ee721de73a99b4025d59c3da8895467984cfeee70a7efbdd2cb98b5670c5` |

The later result report binds the implementation commit, these file hashes,
canonical registrations/artifacts, input hash, run timestamps and runtime. This
record is pre-execution verification, not a historical result or fresh power
assessment. The parent independently audits the persisted output ledgers and
statistics after execution and performs the final merge.
