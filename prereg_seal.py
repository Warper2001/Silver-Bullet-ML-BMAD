#!/usr/bin/env python3
"""prereg_seal.py — Generate a tamper-evident pre-registration document.

Usage:
    python prereg_seal.py --name <experiment-id> [--output <path>] [--config-json '<json>']

The generated document records all StrategyConfig fields, holdout date range,
success metrics, stopping rule, and three cryptographic hashes (config SHA-256,
strategy_core.py SHA-256, git HEAD). Commit the document to make it tamper-evident.
"""

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, time
from pathlib import Path

HOLDOUT_DIR = Path(__file__).parent / "data/sealed_holdout"
STRATEGY_CORE_PATH = Path(__file__).parent / "src/research/strategy_core.py"


def _config_to_json(config) -> str:
    """Canonical deterministic JSON of StrategyConfig — sorted keys, time→HH:MM, no whitespace."""
    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, time):
            d[k] = v.strftime("%H:%M")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _extract_holdout_dates(holdout_dir: Path) -> tuple[str, str]:
    """Return (start_date, end_date) parsed from CSV filenames in holdout_dir."""
    csvs = sorted(holdout_dir.glob("*.csv"))
    dates = []
    for csv_path in csvs:
        m = re.search(r"(\d{4})(\d{2})(\d{2})", csv_path.stem)
        if m:
            dates.append(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    if not dates:
        return ("unknown", "unknown")
    return (min(dates), max(dates))


def _git_head() -> str:
    """Return git HEAD commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, OSError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, OSError):
        return True  # assume dirty when git is unavailable
    return bool(result.stdout.strip())


def _build_config(config_json: str | None):
    """Build a StrategyConfig from defaults, optionally overriding fields via JSON string."""
    from src.research.strategy_core import StrategyConfig

    if not config_json:
        return StrategyConfig()

    overrides = json.loads(config_json)
    defaults = dataclasses.asdict(StrategyConfig())
    defaults.update(overrides)

    # Re-convert time strings back to time objects
    base = StrategyConfig()
    for k, v in defaults.items():
        if isinstance(getattr(base, k), time) and isinstance(v, str):
            defaults[k] = time.fromisoformat(v)

    return StrategyConfig(**defaults)


def seal(
    config,
    output_path: Path,
    name: str,
    strategy_core_path: Path,
    holdout_dir: Path,
    yaml_path: Path | None = None,
) -> int:
    """Generate and write the pre-registration document. Returns 0 on success, 1 on error.

    When yaml_path is provided, hash_a is the SHA-256 of the YAML file bytes
    (not the StrategyConfig JSON). This supports the weekly YAML-driven workflow.
    """
    from protect_holdout import verify as verify_holdout

    # Pre-flight: holdout must be protected
    if verify_holdout(holdout_dir) != 0:
        print("ERROR: Holdout not protected — run protect_holdout.py --init first")
        return 1

    # Dirty tree warning
    if _git_is_dirty():
        print("WARNING: Working tree is dirty — sealed commit will not match HEAD")

    # Compute hash_a: YAML bytes when --config provided, else StrategyConfig JSON
    if yaml_path is not None:
        try:
            yaml_bytes = yaml_path.read_bytes()
        except FileNotFoundError:
            print(f"ERROR: config YAML not found at {yaml_path}")
            return 1
        hash_a = hashlib.sha256(yaml_bytes).hexdigest()
        hash_a_label = "(a) YAML config SHA-256"
        hash_a_footnote = f"*Hash (a): SHA-256 of `{yaml_path}` file bytes.*"
    else:
        config_json = _config_to_json(config)
        hash_a = hashlib.sha256(config_json.encode()).hexdigest()
        hash_a_label = "(a) StrategyConfig SHA-256"
        hash_a_footnote = "*Hash (a): canonical JSON of `dataclasses.asdict(config)` with `time` fields as `\"HH:MM\"`, sorted keys, no whitespace.*"

    try:
        strategy_core_bytes = strategy_core_path.read_bytes()
    except FileNotFoundError:
        print(f"ERROR: strategy_core.py not found at {strategy_core_path}")
        return 1
    hash_b = hashlib.sha256(strategy_core_bytes).hexdigest()

    git_head = _git_head()

    # Holdout date range
    start_date, end_date = _extract_holdout_dates(holdout_dir)

    # Config table rows
    config_rows = []
    for field in dataclasses.fields(config):
        val = getattr(config, field.name)
        if isinstance(val, time):
            val = val.strftime("%H:%M")
        config_rows.append(f"| {field.name} | {val} |")
    config_table = "\n".join(config_rows)

    today = date.today().isoformat()

    doc = f"""\
# Pre-Registration: {name}

**Generated:** {today}
**Experiment ID:** {name}

---

## Hypothesis

<!-- Fill in H₁ and H₀ after running prereg_seal.py, BEFORE committing -->

**H₁ (alternative):**

**H₀ (null):**

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
{config_table}

---

## Holdout Data Range

- **Directory:** `data/sealed_holdout/`
- **Start date:** {start_date}
- **End date:** {end_date}

---

## Integrity Hashes

| Hash | Value |
|---|---|
| {hash_a_label} | `{hash_a}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{git_head}` |

{hash_a_footnote}
*Hash (b): SHA-256 of `{strategy_core_path}` source bytes.*
*Hash (c): `git rev-parse HEAD` at seal time — commit this document to make it tamper-evident.*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc)
    print(f"SEAL PASS — pre-registration written to {output_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a tamper-evident pre-registration document."
    )
    parser.add_argument("--name", required=True, help="Experiment identifier (e.g. s27-oos-run-1)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: _bmad-output/preregistration_{name}.md)",
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help='JSON string of StrategyConfig field overrides (e.g. \'{"bearish_only": false}\')',
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file; SHA-256 of YAML bytes used as hash_a instead of StrategyConfig JSON",
    )
    args = parser.parse_args()

    output_path = args.output or Path(f"_bmad-output/preregistration_{args.name}.md")

    if args.config is not None:
        from src.research.config_loader import load_strategy_config
        config = load_strategy_config(args.config)
    else:
        config = _build_config(args.config_json)

    sys.exit(
        seal(config, output_path, args.name, STRATEGY_CORE_PATH, HOLDOUT_DIR, yaml_path=args.config)
    )


if __name__ == "__main__":
    main()
