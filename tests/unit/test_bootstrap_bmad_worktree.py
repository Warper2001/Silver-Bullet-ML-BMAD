from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/bootstrap_bmad_worktree.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bootstrap_bmad_worktree", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def target(tmp_path: Path) -> Path:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: synthetic\n")
    return worktree


def source_runtime(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    runtime = source / "_bmad"
    runtime.mkdir(parents=True)
    (runtime / "config.toml").write_text("[core]\n")
    (runtime / "config.user.toml").write_text("")
    (runtime / "custom").mkdir()
    (runtime / "scripts").mkdir()
    return source


def test_bootstrap_creates_only_named_links_and_render_stays_local(
    tmp_path: Path,
) -> None:
    module = load_module()
    worktree = target(tmp_path)
    source = source_runtime(tmp_path)
    created = module.bootstrap(source, worktree)

    assert [path.name for path in created] == list(module.LINKS)
    runtime = worktree / "_bmad"
    assert sorted(path.name for path in runtime.iterdir()) == sorted(module.LINKS)
    assert all(path.is_symlink() for path in created)
    assert not (runtime / "render").exists()

    render = runtime / "render" / "synthetic"
    render.mkdir(parents=True)
    assert render.is_relative_to(worktree / "_bmad" / "render")
    assert not render.is_relative_to(source / "_bmad" / "render")


@pytest.mark.parametrize("existing", ["file", "divergent_link"])
def test_bootstrap_refuses_existing_target_entry(tmp_path: Path, existing: str) -> None:
    module = load_module()
    worktree = target(tmp_path)
    runtime = worktree / "_bmad"
    runtime.mkdir()
    entry = runtime / "scripts"
    if existing == "file":
        entry.write_text("do not replace")
    else:
        entry.symlink_to(tmp_path / "somewhere-else")

    with pytest.raises(module.BootstrapError, match="target runtime entry exists"):
        module.bootstrap(source_runtime(tmp_path), worktree)
    assert entry.exists() or entry.is_symlink()


def test_bootstrap_refuses_source_layout_mismatch(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    (source / "_bmad").mkdir(parents=True)
    with pytest.raises(module.BootstrapError, match="missing or linked"):
        module.bootstrap(source, target(tmp_path))


def test_bootstrap_rolls_back_partial_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    worktree = target(tmp_path)
    source = source_runtime(tmp_path)
    original = Path.symlink_to
    calls = 0

    def fail_second(
        self: Path, target: Path, target_is_directory: bool = False
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic link failure")
        original(self, target, target_is_directory=target_is_directory)

    monkeypatch.setattr(Path, "symlink_to", fail_second)
    with pytest.raises(module.BootstrapError, match="rolled back"):
        module.bootstrap(source, worktree)
    assert not (worktree / "_bmad").exists()
