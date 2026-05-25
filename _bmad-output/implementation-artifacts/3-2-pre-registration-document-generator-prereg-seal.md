# Story 3.2: Pre-Registration Document Generator (prereg_seal.py)

Status: done

## Story

As Alex,
I want a `prereg_seal.py` script that generates a tamper-evident pre-registration document with all strategy parameters and three cryptographic hashes,
So that no parameter can be changed after the OOS run begins without visible evidence of tampering.

## Acceptance Criteria

1. Given a `StrategyConfig` instance and `strategy_core.py` source file,
   When `python prereg_seal.py --name <experiment-id> --output <path>` is run,
   Then the generated Markdown document contains: all `StrategyConfig` field names and values, holdout date range extracted from `data/sealed_holdout/` CSV filenames, success metrics (PF ≥ 2.0, Sharpe ≥ 1.5, max DD ≤ 10%), stopping rule (halt if PF < 1.1 after first 100 OOS trades), and minimum sample size (N = 200).

2. Given the document generation,
   When `prereg_seal.py` computes hashes,
   Then it records exactly three hashes in the document footer:
   - (a) SHA-256 of canonical `StrategyConfig` serialization (`dataclasses.asdict()` → sorted-key JSON, with `datetime.time` fields converted to `"HH:MM"` strings)
   - (b) SHA-256 of `strategy_core.py` source text
   - (c) `git rev-parse HEAD` commit hash at seal time

3. Given the generated document is committed to git with `git commit`,
   When `git log --oneline` is inspected,
   Then the commit hash and timestamp serve as the tamper-evident seal.

4. Given `prereg_seal.py` is run with uncommitted changes in the working tree,
   When it detects a dirty git state (`git status --porcelain` is non-empty),
   Then it prints `"WARNING: Working tree is dirty — sealed commit will not match HEAD"` but does not abort (exits 0 after writing the document).

5. Given two runs of `prereg_seal.py` with the same `StrategyConfig` and same `strategy_core.py`,
   When the SHA-256 hashes are compared,
   Then hash (a) and hash (b) are identical — canonical serialization is deterministic.

6. Given `protect_holdout.verify()` returns non-zero (holdout unprotected),
   When `prereg_seal.py` is run,
   Then it prints `"ERROR: Holdout not protected — run protect_holdout.py --init first"` and exits 1 without writing the document.

7. Unit tests cover: hash determinism, dirty-tree warning, holdout date extraction, document contains all StrategyConfig fields, holdout verify gate rejects unprotected state.

## Tasks / Subtasks

- [x] Task 1 — Implement `prereg_seal.py` at repo root (ACs #1–#6)
  - [x] Add `seal(config: StrategyConfig, output_path: Path, name: str, strategy_core_path: Path, holdout_dir: Path) -> int` function — core logic, returns 0 on success, 1 on error
  - [x] Pre-flight: call `protect_holdout.verify(holdout_dir)` → exit 1 with error message if non-zero
  - [x] Extract holdout date range: glob `holdout_dir/*.csv`, parse dates from filenames using `_extract_date()` pattern from `protect_holdout.py`
  - [x] Compute hash (a): `dataclasses.asdict(config)` → convert `datetime.time` values to `"HH:MM"` strings → sort keys → `json.dumps(separators=(',', ':'))` → SHA-256 hex
  - [x] Compute hash (b): read `strategy_core_path` bytes → SHA-256 hex
  - [x] Compute hash (c): `subprocess.run(['git', 'rev-parse', 'HEAD'], ...)` → strip whitespace
  - [x] Check dirty tree: `subprocess.run(['git', 'status', '--porcelain'], ...)` → non-empty output → print WARNING (do not exit)
  - [x] Generate Markdown document (see template in Dev Notes)
  - [x] Write document to `output_path` (create parent dirs if needed)
  - [x] Add `main()` with argparse: `--name` (required), `--output` (default: `_bmad-output/preregistration_{name}.md`), `--config-json` (optional: JSON override of StrategyConfig fields)

- [x] Task 2 — Unit tests `tests/unit/test_prereg_seal.py` (AC #7)
  - [x] `test_hash_a_deterministic`: two calls with same config → same hash (a)
  - [x] `test_hash_b_deterministic`: two calls with same source path → same hash (b)
  - [x] `test_dirty_tree_warning`: mock `git status --porcelain` returning non-empty → check WARNING in output, exit 0
  - [x] `test_holdout_verify_gate`: call `seal()` with a tmp holdout dir containing a 644 CSV → verify exit 1
  - [x] `test_document_contains_all_fields`: run `seal()` on tmp dir → read output → assert all StrategyConfig field names present
  - [x] `test_date_extraction`: CSV filename `mnq_1min_holdout_20260301_plus.csv` → date extracted as `"2026-03-01"`
  - [x] Run: `.venv/bin/python -m pytest tests/unit/test_prereg_seal.py -v` → 22 passed

- [x] Task 3 — Smoke test against real project (ACs #1–#5)
  - [x] Run: `PYTHONPATH=. .venv/bin/python prereg_seal.py --name smoke-test-3-2 --output /tmp/prereg_test.md` → SEAL PASS
  - [x] Confirm: document written, all 21 StrategyConfig fields present, three hashes in table, holdout date 2026-03-01
  - [x] Confirm: hash (a) `ff2d1a4e...` and (b) `df5153e5...` identical on second run ✓
  - [x] Deleted /tmp/prereg_test.md (smoke test only)

- [x] Task 4 — Full regression test suite
  - [x] `.venv/bin/python -m pytest tests/unit/test_prereg_seal.py tests/unit/test_protect_holdout.py tests/unit/test_strategy_core_tuesday.py tests/integration/test_baseline_backtesting.py -q` → 55 passed

## Dev Notes

### Script Location (AR5)

`prereg_seal.py` lives at the **repository root** (same as `protect_holdout.py`):
```
/root/Silver-Bullet-ML-BMAD/prereg_seal.py
```

### StrategyConfig Fields (Complete List — Must All Appear in Document)

From `src/research/strategy_core.py` lines 70–101:
```python
@dataclass(frozen=True)
class StrategyConfig:
    sl_multiplier: float = 5.0
    tp_multiplier: float = 6.0
    entry_pct: float = 0.5
    atr_threshold: float = 0.5
    max_gap_dollars: float = 60.0
    max_hold_bars: int = 60
    max_pending_bars: int = 240
    contracts_per_trade: int = 5
    max_daily_loss: float = -750.0
    vol_regime_lookback: int = 120
    vol_regime_threshold: float = 0.75
    min_gap_atr_ratio: float = 0.25
    ml_threshold: float = 0.0
    bearish_only: bool = True
    h1_sweep_lookback: int = 6
    kill_zone_start_et: time = time(9, 30)   # NOT JSON-serializable — convert to "09:30"
    kill_zone_end_et: time = time(11, 0)     # NOT JSON-serializable — convert to "11:00"
    commission_per_roundtrip: float = 4.0
    enable_kill_zone_filter: bool = False
    m15_confirmation: bool = False
    tuesday_exclusion: bool = True
```

**Critical:** `kill_zone_start_et` and `kill_zone_end_et` are `datetime.time` objects. `dataclasses.asdict()` does NOT convert these to JSON-serializable types — you must handle them explicitly:

```python
import dataclasses
import json
from datetime import time

def _config_to_json(config: StrategyConfig) -> str:
    """Canonical deterministic JSON of StrategyConfig (sorted keys, time→HH:MM)."""
    d = dataclasses.asdict(config)
    # Convert datetime.time objects to "HH:MM" strings
    for k, v in d.items():
        if isinstance(v, time):
            d[k] = v.strftime("%H:%M")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))
```

### Holdout Date Range Extraction

Use the same `_extract_date()` pattern as `protect_holdout.py`:
```python
import re
from pathlib import Path

def _extract_holdout_dates(holdout_dir: Path) -> tuple[str, str]:
    """Return (start_date, end_date) from CSV filenames in holdout dir."""
    csvs = sorted(holdout_dir.glob("*.csv"))
    dates = []
    for csv_path in csvs:
        m = re.search(r"(\d{4})(\d{2})(\d{2})", csv_path.stem)
        if m:
            dates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    if not dates:
        return ("unknown", "unknown")
    return (min(dates), max(dates))
```

Note: The actual holdout file is `mnq_1min_holdout_20260301_plus.csv` — start date = `2026-03-01`. End date = same (single file). For now extract from filename. Story 3.4 may need to read actual timestamps.

### Importing protect_holdout (Pre-flight Check)

`prereg_seal.py` must call `protect_holdout.verify()` as a pre-flight. Import directly (same pattern as unit tests):

```python
import sys
sys.path.insert(0, str(Path(__file__).parent))  # ensure repo root on path
from protect_holdout import verify as verify_holdout
```

Or call as subprocess:
```python
import subprocess
result = subprocess.run(
    [sys.executable, str(Path(__file__).parent / "protect_holdout.py"), "--verify"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("ERROR: Holdout not protected — run protect_holdout.py --init first")
    return 1
```

**Recommended:** Import directly (no subprocess overhead, easier to test by passing `holdout_dir` param).

### Git Operations

Use `subprocess.run(..., capture_output=True, text=True, check=False)` — never `check=True`:

```python
def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()

def _git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return bool(result.stdout.strip())
```

### Generated Document Template

The generated Markdown must include all of the following sections:

```markdown
# Pre-Registration: {name}

**Generated:** {YYYY-MM-DD}
**Experiment ID:** {name}

---

## Hypothesis

<!-- Fill in H₁ and H₀ after running prereg_seal.py, BEFORE committing -->

---

## Decision Rule (Pre-committed, Immutable After Seal)

| Criterion | Threshold |
|---|---|
| Profit Factor | PF ≥ 2.0 |
| Sharpe Ratio | Sharpe ≥ 1.5 |
| Max Drawdown | ≤ 10% |
| Minimum Trades | N ≥ 200 |
| Stopping Rule | Halt if PF < 1.1 after first 100 OOS trades |

---

## Configuration Snapshot

| Field | Value |
|---|---|
| sl_multiplier | {value} |
| tp_multiplier | {value} |
| ... (all 21 fields) ... |

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** {start_date}
- **End date:** {end_date}

---

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) StrategyConfig SHA-256 | `{hash_a}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{git_head}` |

*Hash (a): canonical JSON of `dataclasses.asdict(config)` with `time` fields as `"HH:MM"`, sorted keys, no whitespace.*
*Hash (b): SHA-256 of `src/research/strategy_core.py` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
```

### `--config-json` Override (Optional CLI Flag)

Allow the researcher to override specific fields without editing Python:
```
python prereg_seal.py --name my-test \
  --config-json '{"bearish_only": false, "h1_sweep_lookback": 10}'
```
This merges the JSON overrides into the default `StrategyConfig`. Implementation:

```python
import json, dataclasses
from src.research.strategy_core import StrategyConfig

def _build_config(config_json: str | None) -> StrategyConfig:
    if not config_json:
        return StrategyConfig()
    overrides = json.loads(config_json)
    defaults = dataclasses.asdict(StrategyConfig())
    defaults.update(overrides)
    # Re-convert time strings back to time objects
    from datetime import time as time_type
    for k, v in defaults.items():
        field_default = getattr(StrategyConfig, k, None)
        if isinstance(getattr(StrategyConfig(), k), time_type) and isinstance(v, str):
            h, m = v.split(":")
            defaults[k] = time_type(int(h), int(m))
    return StrategyConfig(**defaults)
```

### Unit Test Pattern

Tests must not touch the real `data/sealed_holdout/` or the real git repo state. Use `tmp_path` and mock subprocess:

```python
import os, hashlib, json, dataclasses
from pathlib import Path
from unittest.mock import patch
import pytest
from src.research.strategy_core import StrategyConfig
from prereg_seal import seal, _config_to_json, _extract_holdout_dates

def make_protected_csv(tmp_path: Path, name: str = "mnq_1min_holdout_20260301_plus.csv") -> Path:
    p = tmp_path / name
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, 0o444)
    return p

def test_hash_a_deterministic(tmp_path):
    make_protected_csv(tmp_path)
    config = StrategyConfig()
    json1 = _config_to_json(config)
    json2 = _config_to_json(config)
    assert json1 == json2
    h1 = hashlib.sha256(json1.encode()).hexdigest()
    h2 = hashlib.sha256(json2.encode()).hexdigest()
    assert h1 == h2

def test_document_contains_all_fields(tmp_path, capsys):
    make_protected_csv(tmp_path)
    output_path = tmp_path / "prereg.md"
    with patch("prereg_seal._git_head", return_value="abc123"), \
         patch("prereg_seal._git_is_dirty", return_value=False):
        rc = seal(StrategyConfig(), output_path, "test", 
                  Path("src/research/strategy_core.py"), tmp_path)
    assert rc == 0
    content = output_path.read_text()
    config = StrategyConfig()
    for field in dataclasses.fields(config):
        assert field.name in content, f"Field {field.name!r} missing from document"
```

### What Story 3.3 Needs from This Story

Story 3.3 (`oos_checkpoint.py`) will verify the three hashes from the pre-reg document before any OOS run. To make that possible:
- The document must embed the hashes in a **parseable, consistent format** — use the table format shown above with `|` separators
- The field labels must be exactly `(a) StrategyConfig SHA-256`, `(b) strategy_core.py SHA-256`, `(c) Git HEAD commit`
- Story 3.3 will grep for these labels and extract the hash values

### What NOT to Do

- Do NOT make the script try to commit to git — the researcher does this manually
- Do NOT prompt for interactive input — fully CLI-driven
- Do NOT modify `strategy_core.py` or `protect_holdout.py`
- Do NOT hardcode the success thresholds (PF ≥ 2.0, etc.) anywhere except the generated document — they're research protocol, not code
- Do NOT make the `--output` path mandatory — default to `_bmad-output/preregistration_{name}.md`
- Do NOT write to `data/sealed_holdout/` — that directory is protected; only `protect_holdout.py --init` touches it

### References

- AR5 (script at root): `_bmad-output/planning-artifacts/architecture.md`
- AR6 (three hashes): architecture.md
- NFR8 (tamper detection): architecture.md
- NFR15 (deterministic serialization): architecture.md
- Story 3.1 (prerequisite): `3-1-sealed-holdout-directory-and-protect-holdout.md`
- `protect_holdout.py` (call verify() before sealing): repo root
- `src/research/strategy_core.py` (StrategyConfig source): lines 70–101
- Existing pre-reg format: `_bmad-output/preregistration_s_vol_15m.md`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23, create-story + dev-story)

### Debug Log References

(none)

### Completion Notes List

- `prereg_seal.py` implemented at repo root with `seal()` accepting `holdout_dir: Path` for testability.
- `_config_to_json()`: `dataclasses.asdict()` + `time→"HH:MM"` conversion + `sort_keys=True` + no whitespace → deterministic SHA-256.
- Pre-flight calls `protect_holdout.verify(holdout_dir)` directly (import, not subprocess) → exits 1 with ERROR if unprotected.
- `_git_is_dirty()` / `_git_head()` use `subprocess.run(..., check=False)` — never `check=True`.
- `--config-json` override: merges JSON into StrategyConfig defaults, handles time-string→time re-conversion.
- 22 unit tests pass using `tmp_path` + `unittest.mock.patch` for git functions.
- Smoke test confirmed: all 21 StrategyConfig fields in document, three hashes present, date extraction correct, hash (a)+(b) deterministic across two runs.
- 55/55 regression tests green.

### File List

- `prereg_seal.py` (NEW — repo root)
- `tests/unit/test_prereg_seal.py` (NEW)
- `_bmad-output/implementation-artifacts/3-2-pre-registration-document-generator-prereg-seal.md` (UPDATED)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (UPDATED)
