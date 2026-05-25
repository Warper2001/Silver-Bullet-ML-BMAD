#!/usr/bin/env python3
"""oos_checkpoint.py — Pre-OOS integrity gate (AR7).

Verifies five conditions before any sealed-holdout data is accessed:
  (a) StrategyConfig hash matches sealed hash
  (b) strategy_core.py hash matches sealed hash
  (c) Git working tree is clean
  (d) git HEAD matches sealed commit
  (e) data/sealed_holdout/ is write-protected

Usage:
    python oos_checkpoint.py --prereg <path-to-prereg-doc.md>

Module API (AR8 — used by oos_verdict.py):
    from oos_checkpoint import checkpoint_or_abort
    checkpoint_or_abort(prereg_path)  # raises SystemExit(1) on failure
"""

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from datetime import time
from pathlib import Path

STRATEGY_CORE_PATH = Path(__file__).parent / "src/research/strategy_core.py"
HOLDOUT_DIR = Path(__file__).parent / "data/sealed_holdout"

HASH_PATTERNS = {
    # Matches both "(a) StrategyConfig SHA-256" and "(a) YAML config SHA-256"
    "hash_a": r"\|\s*\(a\)[^\|]+SHA-256\s*\|\s*`([0-9a-f]+)`",
    "hash_b": r"\|\s*\(b\) strategy_core\.py SHA-256\s*\|\s*`([0-9a-f]+)`",
    "hash_c": r"\|\s*\(c\) Git HEAD commit\s*\|\s*`([0-9a-f]+)`",
}


# ---------------------------------------------------------------------------
# Internal helpers (identical logic to prereg_seal.py for hash (a))
# ---------------------------------------------------------------------------

def _config_to_json(config) -> str:
    """Canonical deterministic JSON of StrategyConfig — sorted keys, time→HH:MM, no whitespace."""
    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, time):
            d[k] = v.strftime("%H:%M")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _git_head() -> str:
    """Return git HEAD commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, OSError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes to tracked files."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False
        )
    except (FileNotFoundError, OSError):
        return True  # assume dirty when git is unavailable
    return bool(result.stdout.strip())


def _parse_prereg(prereg_path: Path) -> dict:
    """Parse integrity hashes from a prereg_seal.py-generated document.

    Returns dict with keys 'hash_a', 'hash_b', 'hash_c'. Missing keys map to None.
    """
    text = prereg_path.read_text()
    result: dict = {"hash_a": None, "hash_b": None, "hash_c": None}
    for key, pattern in HASH_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            result[key] = m.group(1)
    return result


def _compute_config_hash() -> str:
    from src.research.strategy_core import StrategyConfig

    return hashlib.sha256(_config_to_json(StrategyConfig()).encode()).hexdigest()


def _compute_yaml_hash(yaml_path: Path) -> str:
    """SHA-256 of YAML config file bytes — matches hash_a when prereg_seal used --config."""
    return hashlib.sha256(yaml_path.read_bytes()).hexdigest()


def _compute_source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_checks(
    prereg_path: Path,
    strategy_core_path: Path = STRATEGY_CORE_PATH,
    holdout_dir: Path = HOLDOUT_DIR,
    yaml_path: Path | None = None,
) -> list:
    """Run all five integrity checks.

    Returns list of (check_name, passed: bool, message: str).
    Aborts early with a parse-failure entry if hashes cannot be parsed.

    When yaml_path is provided, hash_a is verified against SHA-256 of YAML bytes
    (matching prereg_seal --config workflow). Otherwise hash_a is verified against
    StrategyConfig JSON hash (legacy workflow).
    """
    from protect_holdout import verify as verify_holdout

    hashes = _parse_prereg(prereg_path)
    if any(v is None for v in hashes.values()):
        missing = [k for k, v in hashes.items() if v is None]
        return [
            (
                "parse",
                False,
                f"FAILED: Could not parse required hashes from prereg doc: {missing}",
            )
        ]

    results = []

    # (a) Config hash — YAML bytes or StrategyConfig JSON depending on workflow
    if yaml_path is not None:
        try:
            actual_a = _compute_yaml_hash(yaml_path)
        except (FileNotFoundError, OSError) as exc:
            results.append(("config_hash", False, f"FAILED: Cannot read YAML config at {yaml_path} — {exc}"))
            actual_a = None
        mismatch_msg = f"FAILED: YAML config hash mismatch — {yaml_path} has been modified since pre-registration seal"
    else:
        actual_a = _compute_config_hash()
        mismatch_msg = "FAILED: Config hash mismatch — StrategyConfig has been modified since pre-registration seal"

    if actual_a is not None and actual_a == hashes["hash_a"]:
        results.append(("config_hash", True, "PASS: Config hash matches seal"))
    elif actual_a is not None:
        results.append(
            (
                "config_hash",
                False,
                mismatch_msg,
            )
        )

    # (b) Source hash
    actual_b = _compute_source_hash(strategy_core_path)
    if actual_b == hashes["hash_b"]:
        results.append(("source_hash", True, "PASS: Source hash matches seal"))
    else:
        results.append(
            (
                "source_hash",
                False,
                "FAILED: Source hash mismatch — strategy_core.py has been modified since pre-registration seal",
            )
        )

    # (c) Clean tree
    if not _git_is_dirty():
        results.append(("clean_tree", True, "PASS: Working tree is clean"))
    else:
        results.append(
            (
                "clean_tree",
                False,
                "FAILED: Working tree is dirty — commit or stash all changes before running OOS",
            )
        )

    # (d) HEAD match
    actual_head = _git_head()
    hash_c = hashes["hash_c"]
    if actual_head == hash_c:
        short = actual_head[:8]
        results.append(("head_match", True, f"PASS: HEAD matches sealed commit ({short}...)"))
    else:
        results.append(
            (
                "head_match",
                False,
                f"FAILED: HEAD mismatch — git HEAD does not match sealed commit\n  sealed:  {hash_c}\n  current: {actual_head}",
            )
        )

    # (e) Holdout protected
    if verify_holdout(holdout_dir) == 0:
        results.append(("holdout_protected", True, "PASS: Holdout directory is write-protected"))
    else:
        results.append(
            (
                "holdout_protected",
                False,
                "FAILED: Holdout directory is not fully protected — run protect_holdout.py --init",
            )
        )

    return results


def checkpoint(
    prereg_path: Path,
    strategy_core_path: Path = STRATEGY_CORE_PATH,
    holdout_dir: Path = HOLDOUT_DIR,
    yaml_path: Path | None = None,
) -> int:
    """CLI-style runner: prints all check results to stdout. Returns 0 (pass) or 1 (fail)."""
    results = run_checks(prereg_path, strategy_core_path, holdout_dir, yaml_path=yaml_path)
    passed = True
    for _name, ok, msg in results:
        print(msg)
        if not ok:
            passed = False
    if passed:
        print("CHECKPOINT PASSED — all integrity checks verified")
        return 0
    return 1


def checkpoint_or_abort(
    prereg_path: Path,
    strategy_core_path: Path = STRATEGY_CORE_PATH,
    holdout_dir: Path = HOLDOUT_DIR,
    yaml_path: Path | None = None,
) -> None:
    """Library API for oos_verdict.py (AR8).

    Raises SystemExit(1) on any failure (prints failures to stderr).
    Returns None on success (caller prints its own header).
    """
    results = run_checks(prereg_path, strategy_core_path, holdout_dir, yaml_path=yaml_path)
    failures = [(name, msg) for name, ok, msg in results if not ok]
    if failures:
        for _name, msg in failures:
            print(msg, file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify pre-OOS integrity checks before accessing sealed holdout data."
    )
    parser.add_argument("--prereg", type=Path, required=True, help="Path to pre-registration document (.md)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file; verifies YAML bytes hash against hash_a (use when prereg was sealed with --config)",
    )
    args = parser.parse_args()
    sys.exit(checkpoint(args.prereg, yaml_path=args.config))


if __name__ == "__main__":
    main()
