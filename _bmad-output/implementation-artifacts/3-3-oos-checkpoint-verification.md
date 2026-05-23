# Story 3.3: OOS Checkpoint Verification (oos_checkpoint.py)

Status: review

## Story

As Alex,
I want an `oos_checkpoint.py` script that verifies all five integrity conditions before any OOS data is accessed,
So that no OOS run can proceed if strategy parameters have drifted, the source code has been modified, or the holdout is unprotected.

## Acceptance Criteria

1. Given a valid pre-registration document with embedded hashes and a clean git state matching the sealed commit,
   When `python oos_checkpoint.py --prereg <path>` is run,
   Then it verifies all five conditions and exits with code 0 and message `"CHECKPOINT PASSED — all integrity checks verified"` (AR7)

2. Given any one condition fails (e.g., `strategy_core.py` was modified after sealing),
   When `oos_checkpoint.py` is run,
   Then it exits with code 1 and prints a specific error identifying which check failed and what action is required
   (e.g., `"FAILED: Source hash mismatch — strategy_core.py has been modified since pre-registration seal"`) (NFR8)

3. Given the five checks are: (a) config hash matches sealed hash, (b) source hash matches sealed hash,
   (c) git working tree is clean, (d) `git rev-parse HEAD` equals sealed commit hash, (e) `data/sealed_holdout/` is write-protected,
   When `oos_checkpoint.py` runs each check in order,
   Then it reports which checks passed and which failed before exiting

4. Given `oos_checkpoint.py` is imported as a module by other scripts,
   When `from oos_checkpoint import checkpoint_or_abort` is called and any check fails,
   Then it raises `SystemExit(1)` with the failing check message — callable as a library (AR7)
   (Story 3.4's `oos_verdict.py` will use this as its very first action per AR8)

5. Given a regression test where `strategy_core.py` bytes are replaced with different bytes (via a tmp fixture),
   When `checkpoint` is run against a prereg doc with the real source hash,
   Then the source hash check fails with code 1 — tampering is correctly detected (NFR8 verified)

## Tasks / Subtasks

- [x] Task 1 — Implement `oos_checkpoint.py` at repo root (ACs #1–#4)
  - [x] Add `_parse_prereg(prereg_path: Path) -> dict[str, str]` — reads prereg MD, extracts `hash_a`, `hash_b`, `hash_c` via regex from the integrity hashes table (see parsing contract in Dev Notes)
  - [x] Add `_compute_config_hash() -> str` — `sha256(_config_to_json(StrategyConfig()))`, same `_config_to_json` logic as `prereg_seal.py`
  - [x] Add `_compute_source_hash(path: Path) -> str` — `sha256(path.read_bytes())`
  - [x] Add `_git_head() -> str` and `_git_is_dirty() -> bool` — same subprocess patterns as `prereg_seal.py`
  - [x] Add `run_checks(prereg_path, strategy_core_path, holdout_dir) -> list[tuple[str, bool, str]]` — runs all 5 checks, returns `(check_name, passed, message)` per check
  - [x] Implement the five checks in order (a)–(e) — see Dev Notes for exact messages
  - [x] Add `checkpoint(prereg_path, ...) -> int` — calls `run_checks`, prints each message, prints `"CHECKPOINT PASSED — all integrity checks verified"` on success, returns 0/1
  - [x] Add `checkpoint_or_abort(prereg_path, ...) -> None` — raises `SystemExit(1)` on failure (no return on success); used by Story 3.4
  - [x] Add `main()` with argparse: `--prereg <path>` (required); calls `sys.exit(checkpoint(args.prereg))`
  - [x] `checkpoint()`, `run_checks()`, `checkpoint_or_abort()` accept `strategy_core_path: Path` and `holdout_dir: Path` as parameters for testability (default to module constants)

- [x] Task 2 — Unit tests `tests/unit/test_oos_checkpoint.py` (AC #5 + all failure modes)
  - [x] `test_all_checks_pass`: mock valid prereg doc + matching hashes + mock git clean + mock HEAD matches + protected tmp holdout → exit 0, `"CHECKPOINT PASSED"` in output
  - [x] `test_fail_config_hash_mismatch`: prereg doc with wrong hash_a → check (a) fails with message containing `"Config hash mismatch"`
  - [x] `test_fail_source_hash_mismatch`: prereg doc with wrong hash_b (AC #5) → check (b) fails with `"Source hash mismatch"`
  - [x] `test_fail_dirty_tree`: mock `_git_is_dirty` returns True → check (c) fails with `"dirty"`
  - [x] `test_fail_head_mismatch`: prereg doc with SHA `deadbeef`, mock `_git_head` returns `aabbccdd` → check (d) fails with `"HEAD mismatch"`
  - [x] `test_fail_holdout_unprotected`: tmp holdout with 644 CSV → check (e) fails
  - [x] `test_checkpoint_or_abort_raises_on_failure`: call `checkpoint_or_abort()` with a failing config → raises `SystemExit(1)`
  - [x] `test_checkpoint_or_abort_returns_on_success`: call with all passing → does not raise
  - [x] `test_parse_missing_hashes`: prereg doc with incomplete hash table → returns empty dict or raises clear error
  - [x] Run: `.venv/bin/python -m pytest tests/unit/test_oos_checkpoint.py -v` → all pass

- [x] Task 3 — Smoke test against real project (ACs #1–#3)
  - [x] First generate a smoke test prereg: `PYTHONPATH=. .venv/bin/python prereg_seal.py --name smoke-3-3 --output /tmp/prereg_smoke_3_3.md` → SEAL PASS
  - [x] Run: `PYTHONPATH=. .venv/bin/python oos_checkpoint.py --prereg /tmp/prereg_smoke_3_3.md`
  - [x] Note: Check (d) (HEAD mismatch) will likely fail because HEAD changed since sealing; this is expected — document result in completion notes
  - [x] Confirm: check (a) config, (b) source, (c) clean tree, (e) holdout messages are correctly formatted
  - [x] Delete `/tmp/prereg_smoke_3_3.md` after smoke test (not a real OOS pre-registration)

- [x] Task 4 — Full regression test suite
  - [x] `.venv/bin/python -m pytest tests/unit/test_oos_checkpoint.py tests/unit/test_prereg_seal.py tests/unit/test_protect_holdout.py tests/unit/test_strategy_core_tuesday.py tests/integration/test_baseline_backtesting.py -q`
  - [x] All tests pass with no regressions

## Dev Notes

### Script Location (AR5)

`oos_checkpoint.py` lives at the **repository root** alongside the other Epic 3 scripts:
```
/root/Silver-Bullet-ML-BMAD/oos_checkpoint.py
```
This matches AR5: four standalone scripts at repo root — `protect_holdout.py`, `prereg_seal.py`, `oos_checkpoint.py`, `oos_verdict.py`.

### The Five Checks (AR7 — run in this exact order)

| Check | What to verify | Failure message pattern |
|---|---|---|
| (a) Config hash | `sha256(_config_to_json(StrategyConfig())) == prereg hash_a` | `"FAILED: Config hash mismatch — StrategyConfig has been modified since pre-registration seal"` |
| (b) Source hash | `sha256(strategy_core.py bytes) == prereg hash_b` | `"FAILED: Source hash mismatch — strategy_core.py has been modified since pre-registration seal"` |
| (c) Clean tree | `git status --porcelain` is empty | `"FAILED: Working tree is dirty — commit or stash all changes before running OOS"` |
| (d) HEAD match | `git rev-parse HEAD == prereg hash_c` | `"FAILED: HEAD mismatch — git HEAD does not match sealed commit\n  sealed:  {hash_c}\n  current: {actual_head}"` |
| (e) Holdout | `protect_holdout.verify(holdout_dir) == 0` | `"FAILED: Holdout directory is not fully protected — run protect_holdout.py --init"` |

Pass messages (print before the final PASSED line):
- `"PASS: Config hash matches seal"`
- `"PASS: Source hash matches seal"`
- `"PASS: Working tree is clean"`
- `"PASS: HEAD matches sealed commit ({short_sha}...)"`
- `"PASS: Holdout directory is write-protected"`

### Pre-Registration Document Parsing Contract

`prereg_seal.py` generates exactly this table format (from Story 3.2):

```markdown
## Integrity Hashes

| Hash | Value |
|---|---|
| (a) StrategyConfig SHA-256 | `{hash_a}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{git_head}` |
```

Parser must extract:
- `hash_a`: backtick-wrapped value in `(a) StrategyConfig SHA-256` row
- `hash_b`: backtick-wrapped value in `(b) strategy_core.py SHA-256` row
- `hash_c`: backtick-wrapped value in `(c) Git HEAD commit` row

Regex patterns (the backtick wrapping is literal in the source):
```python
HASH_PATTERNS = {
    "hash_a": r"\|\s*\(a\) StrategyConfig SHA-256\s*\|\s*`([0-9a-f]+)`",
    "hash_b": r"\|\s*\(b\) strategy_core\.py SHA-256\s*\|\s*`([0-9a-f]+)`",
    "hash_c": r"\|\s*\(c\) Git HEAD commit\s*\|\s*`([0-9a-f]+)`",
}
```

If any hash is missing, `_parse_prereg` should return an empty dict for the missing key. `run_checks` should detect this and return a parse-failure entry before running further checks.

### _config_to_json — Must Match prereg_seal.py Exactly

Copy this function verbatim — identical logic ensures identical hash (a):

```python
def _config_to_json(config) -> str:
    """Canonical deterministic JSON of StrategyConfig — sorted keys, time→HH:MM, no whitespace."""
    import dataclasses, json
    from datetime import time
    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, time):
            d[k] = v.strftime("%H:%M")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
```

**Critical:** If this logic differs from `prereg_seal.py`, check (a) will always fail. Both scripts must produce the same canonical JSON from the same `StrategyConfig`.

### Git Subprocess Patterns — Same as prereg_seal.py

```python
def _git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"

def _git_is_dirty() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    return bool(result.stdout.strip())
```

### protect_holdout Import

Same import pattern as `prereg_seal.py` (direct import since both scripts live at repo root):

```python
from protect_holdout import verify as verify_holdout
```

In the function signature: `run_checks(..., holdout_dir: Path = HOLDOUT_DIR)` passes `holdout_dir` to `verify_holdout(holdout_dir)`.

### Practical Note: Check (d) HEAD Mismatch in the Commit Workflow

When the researcher runs `prereg_seal.py` and commits the resulting doc, the git HEAD changes (the commit adding the prereg doc creates a new SHA). So the `(c)` field records the HEAD BEFORE committing the prereg doc, but `oos_checkpoint.py` runs AFTER the commit. Check (d) will fail.

**The intended OOS workflow resolves this:**
1. Finalize all code changes, commit them → HEAD = H1
2. Run `prereg_seal.py --name oos-run-1` → doc records `(c) = H1`
3. Commit the prereg doc → HEAD = H2 (new commit containing the doc)
4. Edit the prereg doc: update `(c) Git HEAD commit` from H1 to H2
5. Amend the commit (or add a fixup commit) → HEAD = H2 (with updated doc)
6. Now `(c)` in doc == current HEAD → check (d) passes

**For the smoke test (Task 3):** check (d) will fail because this workflow isn't followed. This is expected and should be documented in completion notes. All other checks should pass (or show clear messages).

**Do NOT** skip or weaken check (d) — it's load-bearing (NFR8). Document the workflow instead.

### Module API for Story 3.4 (oos_verdict.py)

`oos_verdict.py` (Story 3.4) imports:
```python
from oos_checkpoint import checkpoint_or_abort
checkpoint_or_abort(prereg_path)  # raises SystemExit(1) on failure, returns None on success
```

`checkpoint_or_abort` must:
1. Run all five checks via `run_checks()`
2. On any failure: print each failure message to `stderr` and `raise SystemExit(1)`
3. On all pass: return `None` (no print — `oos_verdict.py` prints its own success header)

The distinction between `checkpoint()` (exit-code style, prints to stdout) and `checkpoint_or_abort()` (exception style, prints failures to stderr) is important. Both are needed.

### Unit Test Pattern

Tests must NOT touch real `data/sealed_holdout/` or real git state. Use `tmp_path` and `unittest.mock.patch`:

```python
import hashlib, dataclasses, json, os
from pathlib import Path
from unittest.mock import patch
from datetime import time
import pytest
from src.research.strategy_core import StrategyConfig
from oos_checkpoint import checkpoint, checkpoint_or_abort, _config_to_json, _parse_prereg

# Helper: build a valid prereg doc string with correct hashes
def make_prereg_doc(
    hash_a: str,
    hash_b: str,
    hash_c: str,
) -> str:
    return f"""# Pre-Registration: test

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) StrategyConfig SHA-256 | `{hash_a}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{hash_c}` |
"""

# Compute correct hashes for a passing test
def correct_config_hash() -> str:
    from oos_checkpoint import _config_to_json
    j = _config_to_json(StrategyConfig())
    return hashlib.sha256(j.encode()).hexdigest()

def correct_source_hash() -> str:
    return hashlib.sha256(Path("src/research/strategy_core.py").read_bytes()).hexdigest()

def make_protected_csv(tmp_path: Path) -> Path:
    p = tmp_path / "mnq_1min_holdout_20260301_plus.csv"
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, 0o444)
    return p

# All-pass test
def test_all_checks_pass(tmp_path, capsys):
    make_protected_csv(tmp_path)
    sealed_head = "a" * 40
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(
        correct_config_hash(),
        correct_source_hash(),
        sealed_head,
    ))
    with patch("oos_checkpoint._git_is_dirty", return_value=False), \
         patch("oos_checkpoint._git_head", return_value=sealed_head):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)
    assert rc == 0
    assert "CHECKPOINT PASSED" in capsys.readouterr().out

# Failure: wrong hash_a
def test_fail_config_hash_mismatch(tmp_path, capsys):
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc("0" * 64, correct_source_hash(), "a" * 40))
    with patch("oos_checkpoint._git_is_dirty", return_value=False), \
         patch("oos_checkpoint._git_head", return_value="a" * 40):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)
    assert rc == 1
    assert "Config hash mismatch" in capsys.readouterr().out
```

### STRATEGY_CORE_PATH and HOLDOUT_DIR constants

At module level (same pattern as `prereg_seal.py` and `protect_holdout.py`):

```python
STRATEGY_CORE_PATH = Path("src/research/strategy_core.py")
HOLDOUT_DIR = Path("data/sealed_holdout")
```

### What NOT to Do

- Do NOT modify `strategy_core.py`, `protect_holdout.py`, or `prereg_seal.py`
- Do NOT touch the real holdout directory in tests — always use `tmp_path`
- Do NOT skip check (d) even though it fails in naive workflows — document the expected workflow instead
- Do NOT use `check=True` in subprocess calls — always use `check=False`
- Do NOT print to `stderr` in `checkpoint()` — print to stdout (both pass and fail messages)
- DO print failures to `stderr` in `checkpoint_or_abort()` (before raising SystemExit)
- Do NOT suppress holdout protect output in tests — `protect_holdout.verify()` prints to stdout; use `capsys` to capture

### References

- AR5 (script at root): `_bmad-output/planning-artifacts/epics.md` line 125
- AR7 (five checks): `_bmad-output/planning-artifacts/epics.md` line 130
- AR8 (oos_verdict uses checkpoint_or_abort): `_bmad-output/planning-artifacts/epics.md` line 131
- NFR8 (tamper detection): `_bmad-output/planning-artifacts/epics.md` line 96
- NFR14 (pre-reg checkpoint): `_bmad-output/planning-artifacts/epics.md` line 105
- `prereg_seal.py` (hash format, `_config_to_json` source): repo root
- `protect_holdout.py` (`verify()` import): repo root
- `src/research/strategy_core.py` (`StrategyConfig` import): lines 70–101
- Story 3.2 Dev Notes (parsing contract, hash labels): `3-2-pre-registration-document-generator-prereg-seal.md`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

(none)

### Completion Notes List

- `oos_checkpoint.py` written at repo root (AR5). Implements all five integrity checks (AR7) in order: (a) config hash, (b) source hash, (c) clean tree, (d) HEAD match, (e) holdout protected.
- `_config_to_json` copied verbatim from `prereg_seal.py` — identical logic ensures hash (a) always matches.
- `_parse_prereg` returns `{hash_a: None, hash_b: None, hash_c: None}` for any missing keys; `run_checks` detects missing hashes early and returns a parse-failure tuple before running further checks.
- `checkpoint()` prints all pass/fail messages to stdout and returns 0/1 (CLI use).
- `checkpoint_or_abort()` prints only failure messages to stderr and raises `SystemExit(1)` on failure; returns `None` on success (library API for Story 3.4 per AR8).
- Smoke test result: checks (a) config hash PASS, (b) source hash PASS, (c) clean tree FAILED (expected — working tree dirty with new uncommitted files), (d) HEAD match PASS (HEAD unchanged since no commit was made while sealing), (e) holdout PASS. All message formats correctly produced.
- 9/9 unit tests pass; full regression 64/64 pass (no regressions).

### File List

- `oos_checkpoint.py` (NEW — repo root)
- `tests/unit/test_oos_checkpoint.py` (NEW)
- `_bmad-output/implementation-artifacts/3-3-oos-checkpoint-verification.md` (UPDATED)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (UPDATED)
