# Story 8.2: YAML Config Externalization

Status: done

## Story

As Alex (the researcher),
I want to externalize `StrategyConfig` parameters to a `strategy_config.yaml` file at the repo root,
so that weekly config changes can be made by editing YAML (no Python code edits, no source-hash re-seal).

## Background

**Why this matters:** The current workflow to change a strategy parameter requires:
1. Edit `strategy_core.py` (or a constant in `tier2_streaming_working.py`)
2. The `strategy_core.py` source bytes change → `hash_b` in the pre-registration changes
3. A full re-seal of `prereg_seal.py` is required

With YAML externalization, the weekly workflow becomes:
1. Edit `strategy_config.yaml` (no Python touched)
2. `python prereg_seal.py --config strategy_config.yaml --name week-N`
3. `git add -f _bmad-output/preregistration_week-N.md && git commit`
4. Restart the live trader (picks up new YAML)

**Pre-registration note:** S25 is already sealed and deployed. This story does NOT change any strategy parameters — it only moves the current parameter values (exactly `StrategyConfig()` defaults) into `strategy_config.yaml`. The S25 decision rule continues unchanged.

## Acceptance Criteria

1. `strategy_config.yaml` exists at repo root and contains all `StrategyConfig` fields at their current S25-deployed values (same as `StrategyConfig()` defaults).
2. `src/research/config_loader.py` (new file) has a `load_strategy_config(yaml_path: str | Path) -> StrategyConfig` function that reads the YAML and returns a `StrategyConfig`. **NOTE: `from_yaml` must NOT be added to `strategy_core.py` — the AR1 purity contract forbids I/O in that module.**
3. `tier2_streaming_working.py`'s `_build_strategy_config()` reads from `STRATEGY_CONFIG_PATH` env var (if set) or from `strategy_config.yaml` at repo root (if it exists); falls back to `StrategyConfig()` defaults if neither is present.
4. `prereg_seal.py` gains a `--config <yaml_path>` flag. When provided:
   - `hash_a` = SHA-256 of the YAML file bytes (not the StrategyConfig JSON)
   - The document label changes to `(a) YAML config SHA-256`
   - The footnote changes to describe the YAML hash
   - The config table in the document still shows parsed StrategyConfig field values
5. Running `python prereg_seal.py --config strategy_config.yaml --name test-yaml` produces a valid pre-registration document (RC=0) with a YAML hash in hash_a.
6. `oos_checkpoint.py` gains a matching `--config <yaml_path>` flag that verifies the YAML hash against the pre-registration document.
7. All existing tests pass — `python -m pytest tests/unit/test_prereg_seal.py tests/unit/test_oos_checkpoint.py tests/unit/test_strategy_core_detection.py tests/unit/test_strategy_core_exits.py -q` green.
8. New unit tests in `tests/unit/test_config_loader.py` cover: load defaults, load override, missing field uses default, invalid YAML raises, time-field parsing (if applicable).

## Tasks / Subtasks

- [x] Task 1: Create `strategy_config.yaml` (AC: #1)
  - [x] Write YAML at repo root with all StrategyConfig fields at S25 values (StrategyConfig() defaults)
  - [x] Include all fields: sl_multiplier, tp_multiplier, entry_pct, atr_threshold, max_gap_dollars, max_hold_bars, max_pending_bars, contracts_per_trade, max_daily_loss, vol_regime_lookback, vol_regime_threshold, min_gap_atr_ratio, ml_threshold, bearish_only, h1_sweep_lookback, kill_zone_start_et, kill_zone_end_et, commission_per_roundtrip, enable_kill_zone_filter, m15_confirmation, tuesday_exclusion
  - [x] Add comments documenting the S25 pre-registration freeze

- [x] Task 2: Create `src/research/config_loader.py` (AC: #2)
  - [x] Write `load_strategy_config(yaml_path: str | Path) -> StrategyConfig`
  - [x] Load YAML with PyYAML (`yaml.safe_load`)
  - [x] Map YAML fields to StrategyConfig kwargs, handle time fields (e.g. `"09:30"` → `time(9, 30)`)
  - [x] Any YAML field not present falls back to the StrategyConfig default (partial YAML is valid)
  - [x] Unknown YAML keys are silently ignored (forward-compat)

- [x] Task 3: Write unit tests for `config_loader.py` (AC: #8)
  - [x] `test_load_returns_strategyconfig_instance`
  - [x] `test_load_default_yaml_matches_strategyconfig_defaults`
  - [x] `test_load_override_single_field`
  - [x] `test_load_partial_yaml_uses_defaults_for_missing`
  - [x] `test_load_invalid_yaml_raises`
  - [x] `test_load_time_field_string_parsed` (e.g. "09:30" → time(9,30))
  - [x] Run tests — 11/11 pass

- [x] Task 4: Update `tier2_streaming_working.py` `_build_strategy_config()` (AC: #3)
  - [x] Import `load_strategy_config` from `src.research.config_loader`
  - [x] Check `STRATEGY_CONFIG_PATH` env var first; then `strategy_config.yaml` in repo root; then `StrategyConfig()` default
  - [x] Log which config source is being used at startup
  - [x] Do NOT change the TIER2_CONFIG label string or any other constants

- [x] Task 5: Add `--config` flag to `prereg_seal.py` (AC: #4, #5)
  - [x] Add `--config` optional argument to argparse
  - [x] In `seal()`: add `yaml_path` parameter (optional); when not None, compute `hash_a = sha256(yaml_path.read_bytes()).hexdigest()`
  - [x] Change document label from `(a) StrategyConfig SHA-256` → `(a) YAML config SHA-256` when yaml_path is provided
  - [x] Change footnote text for hash_a accordingly
  - [x] Config table still shows StrategyConfig field values (derived from parsed YAML or defaults)
  - [x] When `--config` is not provided, behavior is identical to current (backward-compatible)

- [x] Task 6: Add `--config` flag to `oos_checkpoint.py` (AC: #6)
  - [x] Add `--config` optional argument to argparse
  - [x] When `--config` is provided: read YAML bytes, compute SHA-256, compare against hash_a in pre-registration doc
  - [x] When `--config` is not provided: compare against StrategyConfig JSON hash (current behavior — backward-compatible)
  - [x] Updated HASH_PATTERNS["hash_a"] regex to match both "(a) StrategyConfig SHA-256" and "(a) YAML config SHA-256"

- [x] Task 7: Run full relevant test suite (AC: #7)
  - [x] 94 tests pass (prereg_seal + oos_checkpoint + oos_verdict + protect_holdout + config_loader + kill_zone + tuesday)
  - [x] No regressions

## Dev Notes

### Purity Contract — Do NOT Add I/O to `strategy_core.py`

`strategy_core.py` enforces AR1: **no I/O, no imports from `src.*`, no side effects.** Adding `StrategyConfig.from_yaml()` directly in `strategy_core.py` would violate this contract. The approved approach is a separate `src/research/config_loader.py` module that does the I/O and calls `StrategyConfig(...)`.

### `strategy_config.yaml` Content

The YAML must match `StrategyConfig()` defaults exactly. From `strategy_core.py` lines 81–101:

```yaml
# strategy_config.yaml — S25 pre-registered configuration (DO NOT CHANGE without pre-registration)
# Sealed: 2026-05-21, S25 pre-reg ref: preregistration_s25_live_deployment.md

sl_multiplier: 5.0
tp_multiplier: 6.0
entry_pct: 0.5
atr_threshold: 0.5
max_gap_dollars: 60.0
max_hold_bars: 60
max_pending_bars: 240
contracts_per_trade: 5
max_daily_loss: -750.0
vol_regime_lookback: 120
vol_regime_threshold: 0.75
min_gap_atr_ratio: 0.25
ml_threshold: 0.0
bearish_only: true
h1_sweep_lookback: 6
kill_zone_start_et: "09:30"
kill_zone_end_et: "11:00"
commission_per_roundtrip: 4.0
enable_kill_zone_filter: false
m15_confirmation: false
tuesday_exclusion: true
```

### `config_loader.py` Implementation Pattern

```python
# src/research/config_loader.py
from datetime import time
from pathlib import Path
from typing import Union
import yaml
from src.research.strategy_core import StrategyConfig
import dataclasses

def load_strategy_config(yaml_path: Union[str, Path]) -> StrategyConfig:
    path = Path(yaml_path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return StrategyConfig()
    
    defaults = dataclasses.asdict(StrategyConfig())
    merged = {**defaults}  # start with defaults
    
    for k, v in raw.items():
        if k not in defaults:
            continue  # skip unknown keys
        # Handle time fields
        base_val = defaults[k]
        if isinstance(base_val, time) and isinstance(v, str):
            h, m = v.split(":")
            merged[k] = time(int(h), int(m))
        else:
            merged[k] = v
    
    return StrategyConfig(**merged)
```

### `_build_strategy_config()` in `tier2_streaming_working.py`

Current (line 104–106):
```python
def _build_strategy_config() -> StrategyConfig:
    """Return a StrategyConfig using canonical defaults from strategy_core (single source of truth)."""
    return StrategyConfig()
```

New behavior:
```python
def _build_strategy_config() -> StrategyConfig:
    """Load StrategyConfig from YAML if available; fall back to dataclass defaults."""
    from src.research.config_loader import load_strategy_config
    
    yaml_path_env = os.environ.get("STRATEGY_CONFIG_PATH")
    if yaml_path_env:
        path = Path(yaml_path_env)
        if path.exists():
            logger.info(f"Loading strategy config from env: {path}")
            return load_strategy_config(path)
    
    default_yaml = Path(__file__).parent.parent.parent / "strategy_config.yaml"
    if default_yaml.exists():
        logger.info(f"Loading strategy config from {default_yaml}")
        return load_strategy_config(default_yaml)
    
    logger.info("Using StrategyConfig() dataclass defaults")
    return StrategyConfig()
```

### `prereg_seal.py` Changes

The `seal()` function currently takes `config` (a StrategyConfig) and computes:
```python
config_json = _config_to_json(config)
hash_a = hashlib.sha256(config_json.encode()).hexdigest()
```

New behavior when `yaml_path` is provided:
```python
if yaml_path is not None:
    hash_a = hashlib.sha256(Path(yaml_path).read_bytes()).hexdigest()
    hash_a_label = "(a) YAML config SHA-256"
    hash_a_footnote = f"*Hash (a): SHA-256 of `{yaml_path}` file bytes.*"
else:
    config_json = _config_to_json(config)
    hash_a = hashlib.sha256(config_json.encode()).hexdigest()
    hash_a_label = "(a) StrategyConfig SHA-256"
    hash_a_footnote = "*Hash (a): canonical JSON of `dataclasses.asdict(config)` with `time` fields as `\"HH:MM\"`, sorted keys, no whitespace.*"
```

The `main()` function adds:
```python
parser.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config file; hashes YAML bytes as hash_a")
```

And passes `yaml_path=args.config` to `seal()`.

### `oos_checkpoint.py` Changes

Read `oos_checkpoint.py` before editing to understand how hash_a verification currently works. The verification should:
- When `--config <yaml>` provided: recompute SHA-256 of YAML bytes, compare against hash_a in pre-reg doc
- When not provided: current behavior (recompute from StrategyConfig JSON)

### Existing Tests — What Breaks if Not Careful

`tests/unit/test_prereg_seal.py` calls `seal(StrategyConfig(), ...)` without `yaml_path`. This must continue to work identically. The `yaml_path` parameter must default to `None` and old behavior is preserved when `None`.

`tests/unit/test_oos_checkpoint.py` similarly tests without `--config`. Must remain backward-compatible.

Do NOT break any existing tests. Only add new tests for the new YAML path.

### Testing Patterns

```bash
# Run only the relevant unit tests (fast)
PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_prereg_seal.py tests/unit/test_oos_checkpoint.py tests/unit/test_config_loader.py -v

# Smoke test: load YAML config
PYTHONPATH=. .venv/bin/python -c "
from src.research.config_loader import load_strategy_config
cfg = load_strategy_config('strategy_config.yaml')
print(f'min_gap_atr_ratio={cfg.min_gap_atr_ratio} bearish_only={cfg.bearish_only}')
assert cfg.min_gap_atr_ratio == 0.25
assert cfg.bearish_only is True
print('PASS')
"

# Smoke test: prereg with --config flag
PYTHONPATH=. .venv/bin/python prereg_seal.py \
  --name test-yaml-hash \
  --config strategy_config.yaml \
  --output /tmp/test_prereg.md

# Smoke test: env var override
STRATEGY_CONFIG_PATH=strategy_config.yaml PYTHONPATH=. .venv/bin/python -c "
from src.research.tier2_streaming_working import _build_strategy_config
cfg = _build_strategy_config()
assert cfg.min_gap_atr_ratio == 0.25
print('PASS: config loaded from env var')
"
```

### PyYAML Availability

PyYAML (`import yaml`) is available in the `.venv` — it is a transitive dependency of many packages. Verify with `PYTHONPATH=. .venv/bin/python -c "import yaml; print(yaml.__version__)"` before implementing. If not present, use `python-dotenv` or add `pyyaml` via pip.

### What NOT to Change

- Do NOT change `strategy_core.py` — AR1 purity forbids I/O there
- Do NOT change any StrategyConfig field values (YAML mirrors current defaults)
- Do NOT change the TIER2_CONFIG string in `tier2_streaming_working.py`
- Do NOT modify `bearish_only`, `tuesday_exclusion`, `min_gap_atr_ratio` values — S25 pre-registered

### References

- `src/research/strategy_core.py` lines 70–101: StrategyConfig dataclass (all field names and defaults)
- `prereg_seal.py` lines 92–200: `seal()` function (how hash_a is computed)
- `oos_checkpoint.py`: read before modifying for hash_a verification
- `tests/unit/test_prereg_seal.py`: all existing tests must keep passing
- `tests/unit/test_oos_checkpoint.py`: all existing tests must keep passing

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

### Completion Notes List

- `strategy_config.yaml` created at repo root with all 21 StrategyConfig fields at S25-deployed values (identical to `StrategyConfig()` defaults)
- `src/research/config_loader.py` (new): `load_strategy_config(yaml_path)` — PyYAML safe_load, time field parsing, unknown-key tolerance, partial-YAML defaults
- AR1 purity constraint respected: no I/O added to `strategy_core.py`; config loading in separate module
- `tier2_streaming_working.py` `_build_strategy_config()` updated: reads from `STRATEGY_CONFIG_PATH` env var → `strategy_config.yaml` at repo root → `StrategyConfig()` defaults
- `prereg_seal.py`: `--config <yaml>` flag added; `seal()` accepts `yaml_path: Path | None`; hash_a = SHA-256 of YAML bytes when provided; document labels update accordingly; fully backward-compatible (no `--config` → old behavior unchanged)
- `oos_checkpoint.py`: `--config <yaml>` flag added; `run_checks()` / `checkpoint()` / `checkpoint_or_abort()` accept `yaml_path: Path | None`; `HASH_PATTERNS["hash_a"]` regex updated to match both label variants; `_compute_yaml_hash()` helper added
- Smoke test confirmed: `prereg_seal.py --config strategy_config.yaml --name test-yaml-hash` produces SEAL PASS with `(a) YAML config SHA-256` label
- 94 tests pass; 0 regressions

### Review Findings

- [x] [Review][Patch] `_git_is_dirty` returns `False` when git binary unavailable — should return `True` (assume dirty) to prevent silent false-clean seals [prereg_seal.py, oos_checkpoint.py]
- [x] [Review][Patch] `_compute_yaml_hash` has no error handling — `read_bytes()` crash propagates as unhandled exception through `run_checks` instead of clean FAILED entry [oos_checkpoint.py]
- [x] [Review][Defer] `config_loader.py` time-parser (`split(":")`) vs `prereg_seal._build_config()` (`fromisoformat`) inconsistency — both handle standard "HH:MM" correctly; single-digit-hour edge case only in legacy `--config-json` path — deferred, low priority
- [x] [Review][Defer] HASH_PATTERNS `hash_a` regex is overly broad — alternation `(StrategyConfig|YAML config)` would be safer; low practical risk given controlled doc format — deferred
- [x] [Review][Defer] Misleading FAILED message when `--config` omitted on YAML-workflow prereg doc — user sees "StrategyConfig modified" instead of "missing --config flag"; auto-detection from label is future work — deferred
- [x] [Review][Defer] `yaml.safe_load(path.read_text())` without `encoding="utf-8"` arg in config_loader — low probability on UTF-8 Linux systems — deferred
- [x] [Review][Defer] `load_strategy_config` returns defaults silently on empty YAML file — correct behavior (hash-of-empty-file still round-trips correctly); warning would be nice — deferred

### File List

- `strategy_config.yaml` (new)
- `src/research/config_loader.py` (new)
- `tests/unit/test_config_loader.py` (new)
- `src/research/tier2_streaming_working.py` (updated `_build_strategy_config`)
- `prereg_seal.py` (added `--config` flag, `yaml_path` param in `seal()`)
- `oos_checkpoint.py` (added `--config` flag, `yaml_path` param in `run_checks/checkpoint/checkpoint_or_abort`, updated regex, added `_compute_yaml_hash`)
