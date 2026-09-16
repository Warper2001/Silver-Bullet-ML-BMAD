"""Safely expose the local BMAD runtime to an isolated Git worktree.

The renderer writes generated snapshots below the supplied worktree, so this
tool deliberately does not link ``_bmad/render`` (or any generated output).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

LINKS = ("config.toml", "config.user.toml", "custom", "scripts")


class BootstrapError(ValueError):
    """The requested overlay would not be safe to create."""


def validate_source(source_root: Path) -> Path:
    runtime = source_root.resolve() / "_bmad"
    if not runtime.is_dir() or runtime.is_symlink():
        raise BootstrapError(f"source BMAD runtime is not a directory: {runtime}")
    for name in LINKS:
        entry = runtime / name
        if not entry.exists() or entry.is_symlink():
            raise BootstrapError(f"source runtime entry is missing or linked: {entry}")
    if not (runtime / "config.toml").is_file() or not (runtime / "scripts").is_dir():
        raise BootstrapError("source runtime has an invalid config/scripts layout")
    return runtime


def bootstrap(source_root: Path, worktree_root: Path) -> list[Path]:
    """Create the small runtime overlay, refusing every existing target entry."""
    source = validate_source(source_root)
    worktree = worktree_root.resolve()
    if not (worktree / ".git").exists():
        raise BootstrapError(f"target is not a Git worktree: {worktree}")
    target = worktree / "_bmad"
    if target.exists() and (target.is_symlink() or not target.is_dir()):
        raise BootstrapError(f"target runtime entry exists: {target}")
    conflicts = [
        target / name
        for name in LINKS
        if (target / name).exists() or (target / name).is_symlink()
    ]
    if conflicts:
        raise BootstrapError(f"target runtime entry exists: {conflicts[0]}")
    created_target = not target.exists()
    if not target.exists():
        target.mkdir()
    created = []
    try:
        for name in LINKS:
            destination = target / name
            destination.symlink_to(
                source / name, target_is_directory=(source / name).is_dir()
            )
            created.append(destination)
    except OSError as error:
        for destination in created:
            destination.unlink(missing_ok=True)
        if created_target:
            target.rmdir()
        raise BootstrapError("failed to create runtime links; rolled back") from error
    return created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        links = bootstrap(args.source_root, args.worktree_root)
    except BootstrapError as error:
        print(f"bootstrap refused: {error}", file=sys.stderr)
        return 2
    print("Created BMAD runtime links:")
    print("\n".join(str(path) for path in links))
    print(
        "Render output remains local: "
        f"{args.worktree_root.resolve() / '_bmad' / 'render'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
