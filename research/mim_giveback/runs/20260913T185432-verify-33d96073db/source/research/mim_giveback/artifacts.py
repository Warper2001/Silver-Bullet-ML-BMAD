from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

from .config import BASE, CONFIG, OPERATOR_ROOT, ROOT, RUNS


def digest(path: Path | str) -> str:
    path = readable(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def readable(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    if "sealed_holdout" in resolved.parts:
        raise ValueError("Sealed holdout access prohibited")
    return resolved


def portable_path(path: Path | str) -> str:
    """Encode repository paths without binding artifacts to a worktree location."""
    resolved = readable(path)
    for prefix, root in (("repo", ROOT), ("operator", OPERATOR_ROOT)):
        try:
            return f"{prefix}:{resolved.relative_to(root.resolve())}"
        except ValueError:
            pass
    return f"absolute:{resolved}"


def resolve_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("repo:"):
        return readable(ROOT / text.removeprefix("repo:"))
    if text.startswith("operator:"):
        return readable(OPERATOR_ROOT / text.removeprefix("operator:"))
    if text.startswith("absolute:"):
        return readable(text.removeprefix("absolute:"))
    # Backward-compatible reading for artifacts produced before portable bindings.
    return readable(text)


def portable_bindings(value: object) -> object:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {key: portable_bindings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_bindings(item) for item in value]
    return value


def json_write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe_run(path: Path | str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or RUNS.is_symlink():
        raise ValueError("Symlink output prohibited")
    resolved = raw.resolve()
    if resolved.parent != RUNS.resolve():
        raise ValueError("Output must be a fresh direct mim_giveback/runs child")
    return resolved


def source_files() -> list[Path]:
    dependencies = [
        ROOT / "research/mim_comparison/data.py",
        ROOT / "research/mim_comparison/references.py",
        ROOT / "research/mim_robustness/artifacts.py",
        ROOT / "research/mim_robustness/engine.py",
        ROOT / "research/mim_robustness/features.py",
        ROOT / "research/mim_robustness/study.py",
    ]
    return sorted(BASE.glob("*.py")) + [BASE / "definitions.md", *dependencies]


def inventory(path: Path | str) -> dict[str, str]:
    path = readable(path)
    result: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Symlink in immutable inventory")
        if item.is_file() and item.name != "completion.json":
            result[str(item.relative_to(path))] = digest(item)
    return result


def create(
    command: str,
    inputs: list[Path],
    output: Path | None = None,
    bindings: dict[str, object] | None = None,
) -> Path:
    RUNS.mkdir(exist_ok=True)
    path = safe_run(
        output
        or RUNS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + f"-{command}-{uuid.uuid4().hex[:10]}"
        )
    )
    path.mkdir()
    try:
        source = {str(p.relative_to(ROOT)): digest(p) for p in source_files()}
        input_hashes = {portable_path(p): digest(p) for p in inputs}
        manifest = {
            "command": command,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "deployment_authorized": False,
            "sealed_holdout_accessed": False,
            "config": CONFIG,
            "runtime": {
                "python": sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "machine": platform.machine(),
            },
            "inputs": input_hashes,
            "bindings": portable_bindings(bindings or {}),
            "source": source,
        }
        for item in source_files():
            target = path / "source" / item.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
        shutil.copyfile(BASE / "definitions.md", path / "definitions.md")
        json_write(path / "manifest.json", manifest)
    except Exception as exc:
        json_write(
            path / "failure.json", {"type": type(exc).__name__, "error": str(exc)}
        )
        seal(path)
        raise
    return path


def seal(path: Path) -> None:
    path = safe_run(path)
    if (path / "completion.json").exists():
        raise FileExistsError("Run is already sealed")
    # A successful run is accepted only if every external input still has the
    # bytes hashed at creation. This closes the create-to-consume drift window.
    if not (path / "failure.json").exists():
        manifest = json.loads((path / "manifest.json").read_text())
        for input_path, expected in manifest["inputs"].items():
            if digest(resolve_path(input_path)) != expected:
                raise ValueError("Run input drift before seal")
    recorded = {"sha256": inventory(path)}
    for item in path.rglob("*"):
        if item.is_file():
            item.chmod(0o444)
    for directory in sorted(
        (p for p in path.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    json_write(path / "completion.json", recorded)
    (path / "completion.json").chmod(0o444)
    path.chmod(0o555)


def verify_run(path: Path | str) -> dict[str, object]:
    path = safe_run(path)
    if not (path / "completion.json").is_file():
        raise ValueError("Run is incomplete")
    writable = [
        item for item in [path, *path.rglob("*")] if item.stat().st_mode & 0o222
    ]
    if writable and not _is_git_materialized(path):
        raise ValueError("Sealed run contains writable paths")
    recorded = json.loads((path / "completion.json").read_text())["sha256"]
    if inventory(path) != recorded:
        raise ValueError("Run inventory integrity failure")
    manifest = json.loads((path / "manifest.json").read_text())
    for relative, expected in manifest["source"].items():
        snapshot = path / "source" / relative
        if digest(snapshot) != expected:
            raise ValueError("Source snapshot drift")
    for input_path, expected in manifest["inputs"].items():
        if digest(resolve_path(input_path)) != expected:
            raise ValueError("Run input drift")
    return manifest


def _is_git_materialized(path: Path) -> bool:
    """Git restores tracked artifacts writable; hashes remain the seal there."""
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        return False
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        return False
    for command in (
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        ["git", "-C", str(ROOT), "rev-list", "--count", "origin/main..HEAD"],
        ["git", "-C", str(ROOT), "rev-list", "--count", "HEAD..origin/main"],
    ):
        if subprocess.run(command, capture_output=True, check=False).returncode:
            return False
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--error-unmatch",
            *[str(item.relative_to(ROOT)) for item in files],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(relative.parts)


def fail(path: Path, exc: Exception) -> None:
    # Remove partial result files so a failed invocation cannot leave favorable
    # output beside its failure evidence. Creation-time provenance is retained.
    keep = {"manifest.json", "definitions.md"}
    for item in list(path.iterdir()):
        if item.name not in keep and item.name != "source":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    if not (path / "failure.json").exists():
        json_write(
            path / "failure.json",
            {
                "type": type(exc).__name__,
                "error": str(exc),
                "deployment_authorized": False,
            },
        )
    if not (path / "report.md").exists():
        (path / "report.md").write_text(
            "# Failed MIM giveback invocation\n\n" + str(exc) + "\n",
            encoding="utf-8",
        )
    if not (path / "completion.json").exists():
        seal(path)
