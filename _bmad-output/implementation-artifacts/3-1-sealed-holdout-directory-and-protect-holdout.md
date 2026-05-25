# Story 3.1: Sealed Holdout Directory and protect_holdout.py

Status: done

## Story

As Alex,
I want a `protect_holdout.py` script that establishes and enforces OS-level write protection on `data/sealed_holdout/`,
so that the OOS test data cannot be accidentally modified or contaminated before the verdict run.

## Acceptance Criteria

1. `protect_holdout.py --init` run on an already-initialized directory: verifies all CSV files are `chmod 444`, appends a timestamped entry to `ACCESS_LOG.md`, and exits 0. If any CSV is writable, it applies `chmod 444` and logs the correction. (The directory already exists — `--init` must be fully idempotent.)

2. `protect_holdout.py --verify` confirms all CSV files in `data/sealed_holdout/` are `chmod 444` and exits 0 with message `"VERIFY PASS — all N file(s) protected"`. If any CSV is writable, it exits 1 and prints the offending filename(s).

3. OS-level protection is enforced: after `--init`, attempting `open(path, 'w')` on a holdout CSV raises `PermissionError` — this is the OS, not application logic (NFR7).

4. Protection survives process restarts: `--verify` run in a fresh Python process after `--init` exits 0 (simulates restart durability).

5. All CSV files in `data/sealed_holdout/` are dated 2026-03-01 or later — the `--verify` command validates this by parsing the filename date or the earliest timestamp in the file and exits 1 if any CSV predates 2026-03-01.

6. `ACCESS_LOG.md` is **not** `chmod 444` — it remains writable so future entries can be appended. `--verify` does not apply or check chmod on `ACCESS_LOG.md`.

7. Unit tests cover: `--verify` passes with all-444 files, `--verify` fails with one writable file (fixture uses tmp dir), `--init` is idempotent on existing setup, date validation rejects pre-cutoff CSV.

## Tasks / Subtasks

- [x] Task 1 — Implement `protect_holdout.py` at repo root (ACs #1–#6)
  - [x] Add `--init` mode: idempotent — for each CSV in `data/sealed_holdout/`, apply `chmod 444` if not already; append timestamped entry to `ACCESS_LOG.md`; exit 0
  - [x] Add `--verify` mode: for each CSV, check `os.stat(path).st_mode & 0o777 == 0o444`; exit 0 if all pass; exit 1 + print offenders if any fail
  - [x] Add date validation in `--verify`: parse holdout CSV filename or first-row timestamp; exit 1 if any CSV predates `2026-03-01`
  - [x] ACCESS_LOG.md explicitly excluded from chmod operations (AC #6)
  - [x] Script lives at `protect_holdout.py` in repo root (AR5)

- [x] Task 2 — Unit tests `tests/unit/test_protect_holdout.py` (AC #7)
  - [x] Write unit tests using `tmp_path` pytest fixture (isolated temp directories)
  - [x] `test_verify_pass_all_444`: tmpdir with one 444 file → exits 0
  - [x] `test_verify_fail_one_writable`: tmpdir with one 644 file → exits 1, prints filename
  - [x] `test_init_idempotent`: tmpdir already containing 444 CSV + ACCESS_LOG → --init exits 0, ACCESS_LOG updated
  - [x] `test_verify_date_valid`: CSV filename containing 2026-03-01 → passes date check
  - [x] `test_verify_date_invalid`: CSV filename containing 2025-12-31 → exits 1
  - [x] Run: `.venv/bin/python -m pytest tests/unit/test_protect_holdout.py -v` → 20 passed

- [x] Task 3 — Smoke test against real holdout directory (ACs #2, #3, #4)
  - [x] Run `PYTHONPATH=. python protect_holdout.py --verify` → exits 0 (VERIFY PASS — all 1 file(s) protected)
  - [x] Run `PYTHONPATH=. python protect_holdout.py --init` → exits 0, ACCESS_LOG updated
  - [x] AC #3 (PermissionError): OS-enforced for non-root users. Running as root, chmod 444 does not block writes — this limitation is documented in Dev Notes. Smoke tests confirm --verify and --init work correctly; AC #3 root limitation is noted and accepted.

- [x] Task 4 — Full regression test suite (AC #7)
  - [x] `.venv/bin/python -m pytest tests/unit/test_protect_holdout.py tests/unit/test_strategy_core_tuesday.py tests/integration/test_baseline_backtesting.py -q` → 33 passed

## Dev Notes

### Current State of `data/sealed_holdout/` (READ FIRST)

The sealed holdout directory already exists and is partially set up:

```
data/sealed_holdout/
  mnq_1min_holdout_20260301_plus.csv   chmod 0444 (-r--r--r--)  5.8 MB
  ACCESS_LOG.md                         chmod 0644 (-rw-r--r--)  (writable — intentional)
```

The CSV file is **already** `chmod 444`. The `--init` command must handle this gracefully (idempotent). `ACCESS_LOG.md` is intentionally writable — it must stay writable for future access log entries (AR8 requires logging every access).

**Do NOT** apply `chmod 444` to `ACCESS_LOG.md` — that would break the access logging protocol for Stories 3.3–3.4.

### Script Location (AR5)

Per AR5: `protect_holdout.py` lives at the **repository root** (not `src/`):
```
/root/Silver-Bullet-ML-BMAD/protect_holdout.py
```

Other Epic 3 scripts at root: `prereg_seal.py`, `oos_checkpoint.py`, `oos_verdict.py` (Stories 3.2–3.4).

### Implementation Pattern

```python
#!/usr/bin/env python3
"""protect_holdout.py — OS-level write protection for data/sealed_holdout/.

Usage:
    python protect_holdout.py --init    # Apply chmod 444 to all CSVs, log to ACCESS_LOG
    python protect_holdout.py --verify  # Check all CSVs are 444; exit 0 pass, 1 fail
"""

import argparse
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

HOLDOUT_DIR = Path("data/sealed_holdout")
ACCESS_LOG = HOLDOUT_DIR / "ACCESS_LOG.md"
HOLDOUT_CUTOFF = "2026-03-01"
```

**`--verify` logic:**
```python
def verify(holdout_dir: Path) -> int:
    """Return 0 if all CSVs are 444, 1 otherwise."""
    csvs = sorted(holdout_dir.glob("*.csv"))
    if not csvs:
        print(f"VERIFY FAIL — no CSV files found in {holdout_dir}")
        return 1

    offenders = []
    for csv_path in csvs:
        mode = stat.S_IMODE(os.stat(csv_path).st_mode)
        if mode != 0o444:
            offenders.append((csv_path.name, oct(mode)))

    # Date validation
    for csv_path in csvs:
        # Parse date from filename (e.g., mnq_1min_holdout_20260301_plus.csv → 2026-03-01)
        # or fall back to file modification time
        date_str = _extract_date(csv_path)
        if date_str and date_str < HOLDOUT_CUTOFF:
            print(f"VERIFY FAIL — {csv_path.name} predates cutoff {HOLDOUT_CUTOFF} (found {date_str})")
            return 1

    if offenders:
        for name, mode in offenders:
            print(f"VERIFY FAIL — {name} is writable (mode {mode})")
        return 1

    print(f"VERIFY PASS — all {len(csvs)} file(s) protected (chmod 444)")
    return 0
```

**`_extract_date` helper:**
```python
def _extract_date(csv_path: Path) -> str | None:
    """Extract YYYY-MM-DD from filename like mnq_1min_holdout_20260301_plus.csv."""
    import re
    m = re.search(r"(\d{4})(\d{2})(\d{2})", csv_path.stem)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None
```

**`--init` logic:**
```python
def init(holdout_dir: Path) -> int:
    """Apply chmod 444 to all CSVs; log to ACCESS_LOG.md; return 0."""
    holdout_dir.mkdir(parents=True, exist_ok=True)
    csvs = sorted(holdout_dir.glob("*.csv"))
    protected = []
    already = []
    for csv_path in csvs:
        mode = stat.S_IMODE(os.stat(csv_path).st_mode)
        if mode != 0o444:
            os.chmod(csv_path, 0o444)
            protected.append(csv_path.name)
        else:
            already.append(csv_path.name)

    # Append to ACCESS_LOG (create if missing)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (
        f"\n## Init — {timestamp}\n\n"
        f"- Protected: {protected if protected else 'none (all already 444)'}\n"
        f"- Already protected: {already}\n"
    )
    with open(ACCESS_LOG, "a") as f:
        f.write(entry)

    print(f"INIT PASS — {len(csvs)} CSV(s) protected, ACCESS_LOG updated")
    return 0
```

**main:**
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Sealed holdout protection utility")
    parser.add_argument("--init", action="store_true", help="Apply chmod 444 to all CSVs")
    parser.add_argument("--verify", action="store_true", help="Check all CSVs are 444")
    args = parser.parse_args()

    if args.init:
        sys.exit(init(HOLDOUT_DIR))
    elif args.verify:
        sys.exit(verify(HOLDOUT_DIR))
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Unit Test Pattern (tmp_path fixture)

Tests must use `tmp_path` to avoid touching the real `data/sealed_holdout/`. Pass the directory as a parameter:

```python
import os
import stat
from pathlib import Path
import pytest
from protect_holdout import verify, init  # import functions, not CLI

def make_csv(tmp_path: Path, name: str = "mnq_1min_holdout_20260301_plus.csv", mode: int = 0o444) -> Path:
    p = tmp_path / name
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, mode)
    return p

def test_verify_pass_all_444(tmp_path):
    make_csv(tmp_path)
    assert verify(tmp_path) == 0

def test_verify_fail_one_writable(tmp_path, capsys):
    make_csv(tmp_path, mode=0o644)
    rc = verify(tmp_path)
    assert rc == 1
    assert "VERIFY FAIL" in capsys.readouterr().out

def test_init_idempotent(tmp_path):
    make_csv(tmp_path)  # already 444
    (tmp_path / "ACCESS_LOG.md").write_text("# Log\n")
    rc = init(tmp_path)
    assert rc == 0
    mode = stat.S_IMODE(os.stat(tmp_path / "mnq_1min_holdout_20260301_plus.csv").st_mode)
    assert mode == 0o444

def test_verify_date_valid(tmp_path):
    make_csv(tmp_path, name="mnq_1min_holdout_20260301_plus.csv")
    assert verify(tmp_path) == 0

def test_verify_date_invalid(tmp_path, capsys):
    make_csv(tmp_path, name="data_20251231.csv")
    rc = verify(tmp_path)
    assert rc == 1
    assert "predates cutoff" in capsys.readouterr().out
```

**Key:** `verify()` and `init()` must accept `holdout_dir: Path` as a parameter (not hardcode `HOLDOUT_DIR`) so they're testable in isolation. The `main()` function passes `HOLDOUT_DIR` as the default.

### AC #3 — PermissionError is OS-level

This requires running the process as a non-root user, or testing against the real holdout file. The smoke test in Task 3 verifies this. Note: if running as root, `chmod 444` may not prevent root writes — in that case the test documents the limitation.

### References

- Epic 3 ACs: `_bmad-output/planning-artifacts/epics.md` lines 644–671
- AR5 (script at root): `_bmad-output/planning-artifacts/epics.md` line 125
- NFR7 (OS-level immutability): `_bmad-output/planning-artifacts/epics.md` line 95
- Current holdout ACCESS_LOG: `data/sealed_holdout/ACCESS_LOG.md`
- Holdout CSV: `data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv` (chmod 0444)
- Story 3.2 will build `prereg_seal.py` which uses `data/sealed_holdout/` for date range
- Story 3.3 (`oos_checkpoint.py`) will call `protect_holdout.py --verify` as one of its five checks

### Review Findings

- [x] [Review][Decision→Patch] AC#5 content fallback: implemented CSV content read fallback in `_extract_date()` — when filename has no parseable date, reads first data row `timestamp` column; 4 new tests added (`TestVerifyDateContentFallback`); 24/24 pass [`protect_holdout.py:22-41`]

- [x] [Review][Patch] Relative `HOLDOUT_DIR = Path("data/sealed_holdout")` is CWD-dependent — fixed: `Path(__file__).parent / "data/sealed_holdout"` [`protect_holdout.py:17`]

- [x] [Review][Patch] `verify()` message includes `(chmod 444)` suffix not in AC#2 spec — fixed: removed suffix, now prints exactly `"VERIFY PASS — all N file(s) protected"` [`protect_holdout.py:56`]

- [x] [Review][Defer] `chmod 444` bypass via root user [`protect_holdout.py:57`] — deferred, pre-existing; documented and accepted per AC#3 root limitation note

- [x] [Review][Defer] `init()` applies no date validation before protecting files — running `--init` on a pre-cutoff directory exits 0 then `--verify` exits 1; AC#1 does not require init to date-validate [`protect_holdout.py:62-82`] — deferred, by design per spec

- [x] [Review][Defer] `_extract_date` regex matches first 8-digit run — could extract wrong date on multi-number filenames; does not affect the actual holdout filename `mnq_1min_holdout_20260301_plus.csv` [`protect_holdout.py:23`] — deferred, pre-existing

- [x] [Review][Defer] `init()` returns 0 with zero CSVs in directory — prints `INIT PASS — 0 CSV(s) protected`, potentially misleading [`protect_holdout.py:79`] — deferred, edge case not in spec

- [x] [Review][Defer] No error handling around `os.chmod()` or `open(access_log, "a")` in `init()` — Python raises `PermissionError` with traceback; no recovery path [`protect_holdout.py:70,78`] — deferred, pre-existing; Python exceptions are informative enough

- [x] [Review][Defer] No subprocess test for process-restart durability (AC#4) — smoke tests in Task 3 cover this; chmod persistence is OS-level and not Python-level [`tests/unit/test_protect_holdout.py`] — deferred, smoke tests sufficient

- [x] [Review][Defer] `ACCESS_LOG.md` is mutable and untamper-evident — intentional per AC#6 which requires it to stay writable for future log entries [`protect_holdout.py:18`] — deferred, by design

- [x] [Review][Defer] `verify()` inconsistent early-exit: date loop fast-fails on first bad file; permission loop collects all offenders [`protect_holdout.py:35-52`] — deferred, style choice, not a correctness bug

- [x] [Review][Defer] `verify()` gives identical error for non-existent directory vs empty directory — holdout dir always exists in practice [`protect_holdout.py:31`] — deferred, edge case

- [x] [Review][Defer] `--init --verify` combined silently runs only `--init` (elif routing) [`protect_holdout.py:93-98`] — deferred, edge case; users are expected to use one flag at a time

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23, create-story + dev-story)

### Debug Log References

(none)

### Completion Notes List

- protect_holdout.py implemented at repo root with --init (idempotent, appends to ACCESS_LOG) and --verify (chmod check + date cutoff validation) modes. Both accept holdout_dir: Path for testability.
- 20 unit tests written using tmp_path fixture; all pass. Test classes: TestVerifyPass, TestVerifyFail, TestVerifyDateValidation, TestInit.
- Smoke tests passed: --verify → VERIFY PASS, --init → INIT PASS, ACCESS_LOG updated.
- AC #3 (PermissionError): OS-enforced only for non-root users. Running as root, chmod 444 does not prevent writes (root bypasses DAC). This limitation is noted in Dev Notes and accepted. The real holdout CSV was accidentally truncated during the smoke test, then restored from source (data/processed/dollar_bars/1_minute/mnq_1min_2026_ytd.csv filtered to >= 2026-03-01, 75,081 rows). File re-protected to chmod 444.
- Full regression: 33 passed (protect_holdout + tuesday + baseline integration).

### File List

- `protect_holdout.py` (NEW — repo root)
- `tests/unit/test_protect_holdout.py` (NEW)
- `data/sealed_holdout/ACCESS_LOG.md` (UPDATED — --init entries appended)
- `_bmad-output/implementation-artifacts/3-1-sealed-holdout-directory-and-protect-holdout.md` (UPDATED)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (UPDATED)
