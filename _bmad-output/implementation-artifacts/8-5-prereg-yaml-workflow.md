# Story 8.5: Pre-Registration YAML Workflow

Status: done

## Story

As Alex (the researcher),
I want the weekly pre-registration workflow to be fully exercised end-to-end with the YAML config,
so that I can confirm the `--config` flag in `prereg_seal.py` and `oos_checkpoint.py` work together and add the missing unit tests.

## Background

Story 8-2 added `--config <yaml>` to `prereg_seal.py` and `oos_checkpoint.py`. Story 8-5 is the closing story for Epic 8:
1. **Add unit tests** for the YAML-hash path in `prereg_seal.py` and `oos_checkpoint.py` (they currently only have tests for the legacy StrategyConfig JSON path)
2. **Document the weekly pre-registration workflow** in `CLAUDE.md`
3. **Mark Epic 8 done** once all 5 stories are in review

The weekly config change workflow (complete):
```bash
# Step 1: Pre-register BEFORE changing anything live
PYTHONPATH=. python prereg_seal.py \
  --name week-23-test \
  --config strategy_config.yaml \
  --output _bmad-output/preregistration_week23.md
git add -f _bmad-output/preregistration_week23.md && git commit -m "pre-register week 23"

# Step 2: Edit strategy_config.yaml (e.g. change min_gap_atr_ratio)
# Step 3: Restart live trader (picks up new YAML automatically)

# Step 4: Before running any OOS test, verify integrity
PYTHONPATH=. python oos_checkpoint.py \
  --prereg _bmad-output/preregistration_week23.md \
  --config strategy_config.yaml

# Step 5: Weekly backtest check
PYTHONPATH=. python tools/weekly_backtest.py --weeks 4
```

## Acceptance Criteria

1. `tests/unit/test_prereg_seal.py` has new tests for the `--config <yaml>` path in `seal()`:
   - `test_seal_yaml_config_hash_a_is_yaml_bytes_sha256`
   - `test_seal_yaml_config_label_in_document`
   - `test_seal_yaml_config_missing_file_returns_1`
   - `test_seal_yaml_config_backward_compat_without_flag`
2. `tests/unit/test_oos_checkpoint.py` has new tests for the YAML-hash verification path:
   - `test_run_checks_yaml_hash_passes_when_yaml_unchanged`
   - `test_run_checks_yaml_hash_fails_when_yaml_modified`
   - `test_hash_pattern_matches_yaml_config_label`
3. `CLAUDE.md` "Methodology Status" section (or a new "Weekly Workflow" section) documents the 5-step weekly config change workflow.
4. `sprint-status.yaml` updated: `epic-8: done`, all 5 stories at `review` (or `done`).
5. All 8 new tests pass: `python -m pytest tests/unit/test_prereg_seal.py tests/unit/test_oos_checkpoint.py -q` — all tests green, no regressions.

## Tasks / Subtasks

- [x] Task 1: Add YAML-path tests to `test_prereg_seal.py` (AC: #1)
  - [x] `test_seal_yaml_config_hash_a_is_yaml_bytes_sha256` — verifies hash_a = sha256(yaml_bytes)
  - [x] `test_seal_yaml_config_label_in_document` — verifies "(a) YAML config SHA-256" appears in doc
  - [x] `test_seal_yaml_config_missing_file_returns_1` — missing yaml → rc=1
  - [x] `test_seal_yaml_config_backward_compat_without_flag` — omitting yaml_path → old "(a) StrategyConfig SHA-256" label preserved
  - [x] Run tests; confirm all pass

- [x] Task 2: Add YAML-hash tests to `test_oos_checkpoint.py` (AC: #2)
  - [x] `test_run_checks_yaml_hash_passes_when_yaml_unchanged` — create prereg with yaml hash, run checks with same yaml → config_hash PASS
  - [x] `test_run_checks_yaml_hash_fails_when_yaml_modified` — create prereg with yaml hash, modify yaml bytes, run checks → config_hash FAIL
  - [x] `test_hash_pattern_matches_yaml_config_label` — verify updated regex in HASH_PATTERNS matches "(a) YAML config SHA-256"
  - [x] Run tests; confirm all pass

- [x] Task 3: Document weekly workflow in `CLAUDE.md` (AC: #3)
  - [x] Add "Weekly Config Change Workflow" subsection under "Methodology Status"
  - [x] Include the 5-step process (pre-register → edit YAML → restart → checkpoint → weekly backtest)

- [x] Task 4: Update `sprint-status.yaml` for Epic 8 completion (AC: #4)
  - [x] Set `epic-8: done` (all 5 stories at review/done)
  - [x] Update `last_updated` timestamp

- [x] Task 5: Final test run (AC: #5)
  - [x] `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_prereg_seal.py tests/unit/test_oos_checkpoint.py -q`
  - [x] All green — 43 passed, 0 failures

## Dev Notes

### `test_prereg_seal.py` — New Test Class

Add `class TestSealYamlConfig` with the 4 new tests. Key pattern:

```python
def test_seal_yaml_config_hash_a_is_yaml_bytes_sha256(self, tmp_path):
    from prereg_seal import seal
    import hashlib

    make_protected_csv(tmp_path)
    make_access_log(tmp_path)

    # Write a YAML file
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("sl_multiplier: 5.0\n")
    expected_hash = hashlib.sha256(yaml_path.read_bytes()).hexdigest()

    output = tmp_path / "prereg.md"
    with patch("prereg_seal._git_head", return_value="abc123"), \
         patch("prereg_seal._git_is_dirty", return_value=False):
        rc = seal(
            StrategyConfig(), output, "test-yaml",
            Path("src/research/strategy_core.py"), tmp_path,
            yaml_path=yaml_path,
        )
    assert rc == 0
    content = output.read_text()
    assert expected_hash in content

def test_seal_yaml_config_label_in_document(self, tmp_path):
    ...
    assert "(a) YAML config SHA-256" in content
    assert "StrategyConfig SHA-256" not in content

def test_seal_yaml_config_missing_file_returns_1(self, tmp_path):
    ...
    rc = seal(..., yaml_path=tmp_path / "nonexistent.yaml")
    assert rc == 1

def test_seal_yaml_config_backward_compat_without_flag(self, tmp_path):
    ...
    rc = seal(..., yaml_path=None)  # or omit yaml_path
    content = output.read_text()
    assert "(a) StrategyConfig SHA-256" in content
```

### `test_oos_checkpoint.py` — New Test Class

Add `class TestYamlHashVerification`:

```python
def make_yaml_prereg_doc(yaml_hash: str, hash_b: str, hash_c: str) -> str:
    return f"""## Integrity Hashes
| Hash | Value |
|---|---|
| (a) YAML config SHA-256 | `{yaml_hash}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{hash_c}` |
"""

def test_run_checks_yaml_hash_passes_when_yaml_unchanged(tmp_path):
    from oos_checkpoint import run_checks, _compute_yaml_hash
    import hashlib

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("sl_multiplier: 5.0\n")
    yaml_hash = hashlib.sha256(yaml_path.read_bytes()).hexdigest()

    prereg = tmp_path / "prereg.md"
    prereg.write_text(make_yaml_prereg_doc(yaml_hash, correct_source_hash(), "abc123"))

    make_protected_csv(tmp_path)
    with patch("oos_checkpoint._git_head", return_value="abc123"), \
         patch("oos_checkpoint._git_is_dirty", return_value=False):
        results = run_checks(prereg, Path("src/research/strategy_core.py"), tmp_path, yaml_path=yaml_path)
    config_result = next(r for r in results if r[0] == "config_hash")
    assert config_result[1] is True

def test_run_checks_yaml_hash_fails_when_yaml_modified(tmp_path):
    # Create prereg with original YAML hash, then modify YAML
    ...
    assert config_result[1] is False

def test_hash_pattern_matches_yaml_config_label(tmp_path):
    import re
    from oos_checkpoint import HASH_PATTERNS
    doc = "| (a) YAML config SHA-256 | `abc123` |"
    m = re.search(HASH_PATTERNS["hash_a"], doc)
    assert m is not None
    assert m.group(1) == "abc123"
```

### `CLAUDE.md` Section to Add

Add under "Methodology Status" (after the current last bullet):

```markdown
### Weekly Config Change Workflow (Epic 8)

1. **Pre-register** (BEFORE any change): `python prereg_seal.py --name week-N --config strategy_config.yaml --output _bmad-output/preregistration_weekN.md && git add -f ... && git commit`
2. **Edit YAML**: change `strategy_config.yaml` (no Python code changes needed)
3. **Restart trader**: live system picks up new YAML automatically
4. **Weekly check**: `python tools/weekly_backtest.py --weeks 4` (requires fresh post-holdout data)
5. **OOS gate** (before any holdout access): `python oos_checkpoint.py --prereg ... --config strategy_config.yaml`
```

### References

- `prereg_seal.py` lines 92–130: `seal()` with `yaml_path` parameter
- `oos_checkpoint.py` lines 32–36: `HASH_PATTERNS` with updated regex
- `tests/unit/test_prereg_seal.py`: existing `TestSeal` class (add new `TestSealYamlConfig`)
- `tests/unit/test_oos_checkpoint.py`: existing test classes (add new `TestYamlHashVerification`)

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- Task 1: Added `class TestSealYamlConfig` (4 tests) to `tests/unit/test_prereg_seal.py`. All 4 cover the `yaml_path` parameter added to `seal()` in Story 8-2: YAML hash verification, label change, missing-file rc=1, and backward-compat (yaml_path=None → legacy label preserved).
- Task 2: Added `class TestYamlHashVerification` (4 tests) and `make_yaml_prereg_doc()` helper to `tests/unit/test_oos_checkpoint.py`. Tests confirm: YAML hash PASS when unchanged, FAIL when modified, and that `HASH_PATTERNS["hash_a"]` regex matches both new YAML label and legacy StrategyConfig label.
- Task 3: Added "Weekly Config Change Workflow (Epic 8)" subsection to CLAUDE.md under "Methodology Status" with the 5-step process.
- Task 4: `sprint-status.yaml` updated — `epic-8: done`, all 5 stories at review, `last_updated` bumped.
- Task 5: Final test run — 43 passed, 0 failures, 2 Pydantic deprecation warnings (pre-existing).

### Review Findings (2026-05-25)

**Patched (0 items):** None required.

**Deferred (1 item):**
- [ ] No test for YAML-file-missing at `oos_checkpoint` verification time — `run_checks()` handles `FileNotFoundError` correctly but there is no test exercising that path. Low risk (production code is safe). [`oos_checkpoint.py:140-144`]

**Dismissed (1 item):**
- `HASH_PATTERNS["hash_a"]` regex too permissive — already logged in deferred-work.md from Story 8-2 review. Duplicate finding; no action.

### File List

- `tests/unit/test_prereg_seal.py` (add TestSealYamlConfig class)
- `tests/unit/test_oos_checkpoint.py` (add TestYamlHashVerification class)
- `CLAUDE.md` (add Weekly Config Change Workflow section)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (epic-8: done)
