"""Unit tests for oos_checkpoint.py (Story 3.3).

Tests never touch real data/sealed_holdout/ or real git state.
All git calls and holdout dirs are mocked via tmp_path + unittest.mock.patch.
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.research.strategy_core import StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_to_json_ref(config) -> str:
    """Reference implementation — must produce identical output to oos_checkpoint._config_to_json."""
    import dataclasses
    import json
    from datetime import time

    d = dataclasses.asdict(config)
    for k, v in d.items():
        if isinstance(v, time):
            d[k] = v.strftime("%H:%M")
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def correct_config_hash() -> str:
    j = _config_to_json_ref(StrategyConfig())
    return hashlib.sha256(j.encode()).hexdigest()


def correct_source_hash() -> str:
    return hashlib.sha256(Path("src/research/strategy_core.py").read_bytes()).hexdigest()


def make_prereg_doc(hash_a: str, hash_b: str, hash_c: str) -> str:
    return f"""# Pre-Registration: test

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) StrategyConfig SHA-256 | `{hash_a}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{hash_c}` |
"""


def make_protected_csv(tmp_path: Path) -> Path:
    p = tmp_path / "mnq_1min_holdout_20260301_plus.csv"
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, 0o444)
    return p


SEALED_HEAD = "a" * 40


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_checks_pass(tmp_path, capsys):
    """All five checks pass → exit code 0 and CHECKPOINT PASSED in stdout."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(correct_config_hash(), correct_source_hash(), SEALED_HEAD))

    from oos_checkpoint import checkpoint

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
    ):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert rc == 0
    out = capsys.readouterr().out
    assert "CHECKPOINT PASSED" in out


def test_fail_config_hash_mismatch(tmp_path, capsys):
    """Wrong hash_a → check (a) fails with 'Config hash mismatch'."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc("0" * 64, correct_source_hash(), SEALED_HEAD))

    from oos_checkpoint import checkpoint

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
    ):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert rc == 1
    out = capsys.readouterr().out
    assert "Config hash mismatch" in out


def test_fail_source_hash_mismatch(tmp_path, capsys):
    """Wrong hash_b → check (b) fails with 'Source hash mismatch' (AC #5)."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(correct_config_hash(), "f" * 64, SEALED_HEAD))

    from oos_checkpoint import checkpoint

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
    ):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert rc == 1
    out = capsys.readouterr().out
    assert "Source hash mismatch" in out


def test_fail_dirty_tree(tmp_path, capsys):
    """Dirty git tree → check (c) fails with 'dirty'."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(correct_config_hash(), correct_source_hash(), SEALED_HEAD))

    from oos_checkpoint import checkpoint

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=True),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
    ):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert rc == 1
    out = capsys.readouterr().out
    assert "dirty" in out.lower()


def test_fail_head_mismatch(tmp_path, capsys):
    """HEAD doesn't match sealed commit → check (d) fails with 'HEAD mismatch'."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(correct_config_hash(), correct_source_hash(), "dead" + "b" * 36))

    from oos_checkpoint import checkpoint

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value="aabb" + "c" * 36),
    ):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert rc == 1
    out = capsys.readouterr().out
    assert "HEAD mismatch" in out


def test_fail_holdout_unprotected(tmp_path, capsys):
    """Holdout CSV with mode 644 → check (e) fails."""
    p = tmp_path / "mnq_1min_holdout_20260301_plus.csv"
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, 0o644)  # writable — not protected

    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(correct_config_hash(), correct_source_hash(), SEALED_HEAD))

    from oos_checkpoint import checkpoint

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
    ):
        rc = checkpoint(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert rc == 1
    out = capsys.readouterr().out
    assert "protected" in out.lower() or "FAILED" in out


def test_checkpoint_or_abort_raises_on_failure(tmp_path):
    """checkpoint_or_abort raises SystemExit(1) when any check fails."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc("0" * 64, correct_source_hash(), SEALED_HEAD))

    from oos_checkpoint import checkpoint_or_abort

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
        pytest.raises(SystemExit) as exc_info,
    ):
        checkpoint_or_abort(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert exc_info.value.code == 1


def test_checkpoint_or_abort_returns_on_success(tmp_path):
    """checkpoint_or_abort returns None (no raise) when all checks pass."""
    make_protected_csv(tmp_path)
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text(make_prereg_doc(correct_config_hash(), correct_source_hash(), SEALED_HEAD))

    from oos_checkpoint import checkpoint_or_abort

    with (
        patch("oos_checkpoint._git_is_dirty", return_value=False),
        patch("oos_checkpoint._git_head", return_value=SEALED_HEAD),
    ):
        result = checkpoint_or_abort(doc_path, Path("src/research/strategy_core.py"), tmp_path)

    assert result is None


def test_parse_missing_hashes(tmp_path):
    """Prereg doc with incomplete hash table → _parse_prereg returns empty dict."""
    doc_path = tmp_path / "prereg.md"
    doc_path.write_text("# Pre-Registration: incomplete\n\nNo hashes here.\n")

    from oos_checkpoint import _parse_prereg

    result = _parse_prereg(doc_path)
    # All three keys should be absent (empty dict or missing keys)
    assert result.get("hash_a") is None
    assert result.get("hash_b") is None
    assert result.get("hash_c") is None
