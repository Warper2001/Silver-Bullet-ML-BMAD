"""Exclusive runs, pinned evidence, source/runtime snapshots, and fail-closed sealing."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re

import matplotlib
import sys
import uuid

import numpy as np
import pandas as pd

from research.mim_robustness import artifacts as original
from research.mim_lifecycle import analysis

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
SOURCE_RUN = ROOT / "research/mim_robustness/runs/20260912T151842-run-f8608e71fb"
SOURCE_COMPLETION_HASH = (
    "b747ceff679c3874a6236d71eb960c4a68890aa5267aaa0575070b34b097a9a7"
)
DATA = ROOT / "data/mim_x/mnq_1min_by_contract.csv"
DATA_HASH = "ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea"
SPEC = ROOT / "_bmad-output/implementation-artifacts/spec-mim-nb-trade-lifecycle.md"
PROTOCOL = original.PROTOCOL
CONFIG = dict(
    arm="A",
    delay=2,
    quantity=1,
    dollars_per_point=analysis.POINT_VALUE,
    roundtrip_friction=analysis.FEE,
    checkpoints_minutes=list(analysis.CHECKPOINTS),
    expected_trades=analysis.EXPECTED_TRADES,
    expected_sessions=analysis.EXPECTED_SESSIONS,
    expected_net=analysis.EXPECTED_NET,
    hindsight_top_trade_fraction=analysis.TOP_TRADE_FRACTION,
    hindsight_top_daily_count=analysis.TOP_DAILY_COUNT,
    descriptive_only=True,
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "x") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def source_hashes():
    paths = list(BASE.glob("*.py")) + [
        ROOT / "research/mim_comparison/data.py",
        ROOT / "research/mim_robustness/artifacts.py",
        ROOT / "research/mim_robustness/__init__.py",
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


LOADED = source_hashes()


def runtime():
    return dict(
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        matplotlib=matplotlib.__version__,
        machine=platform.machine(),
    )


def safe_run(path):
    path = Path(path)
    if (
        RUNS.is_symlink()
        or RUNS.resolve() != BASE.resolve() / "runs"
        or path.is_symlink()
        or path.resolve().parent != RUNS.resolve()
    ):
        raise ValueError("Output must be a direct isolated lifecycle runs child")
    return path.resolve()


def inventory(path):
    files = list(Path(path).rglob("*"))
    if any(p.is_symlink() for p in files):
        raise ValueError("Symlink in run inventory")
    return {
        str(p.relative_to(path)): digest(p)
        for p in sorted(files)
        if p.is_file() and p != Path(path) / "completion.json"
    }


def create():
    RUNS.mkdir(exist_ok=True)
    path = safe_run(
        RUNS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-diagnostic-"
            + uuid.uuid4().hex[:12]
        )
    )
    path.mkdir()
    return path


def frozen_spec_hash(path):
    match = re.search(
        r"<frozen-after-approval>(.*?)</frozen-after-approval>",
        Path(path).read_text(),
        flags=re.S,
    )
    if not match:
        raise ValueError("Missing frozen spec contract")
    return hashlib.sha256(match.group(1).encode()).hexdigest()


def bind(path):
    """Snapshot before any numerical analysis; verification runs before and after."""
    path = safe_run(path)
    if source_hashes() != LOADED:
        raise ValueError("Loaded source drift")
    inputs = [
        DATA,
        SOURCE_RUN / "completion.json",
        SOURCE_RUN / "trades.csv",
        SOURCE_RUN / "daily.csv",
        original.BASELINE / "completion.json",
    ]
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        runtime=runtime(),
        config=CONFIG,
        source=LOADED,
        inputs={str(p): digest(p) for p in inputs},
        spec_hash=digest(SPEC),
        frozen_spec_hash=frozen_spec_hash(SPEC),
        protocol_hash=digest(PROTOCOL),
        descriptive_only=True,
        strategy_test=False,
        deployment_authorized=False,
    )
    for rel in LOADED:
        target = path / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "xb") as stream:
            stream.write((ROOT / rel).read_bytes())
    for source, name in ((SPEC, "spec.md"), (PROTOCOL, "source_protocol.md")):
        with open(path / name, "xb") as stream:
            stream.write(source.read_bytes())
    write_json(path / "manifest.json", manifest)
    verify_inputs()
    verify(path)


def verify_inputs():
    if (
        digest(SOURCE_RUN / "completion.json") != SOURCE_COMPLETION_HASH
        or digest(DATA) != DATA_HASH
    ):
        raise ValueError("Pinned source completion/data hash mismatch")
    original.verify(SOURCE_RUN, complete=True)
    original.verify_baseline(DATA, original.BASELINE)


SUCCESS_FILES = frozenset(
    (
        "manifest.json",
        "spec.md",
        "source_protocol.md",
        "paths.csv",
        "lifecycle.csv",
        "landmarks.csv",
        "summary.json",
        "report.md",
        "report.html",
        "excursions.svg",
        "first_positive.svg",
    )
)


def require_success(path):
    if (path / "failure.json").exists():
        raise ValueError("Failed invocation cannot verify as successful")
    missing = sorted(name for name in SUCCESS_FILES if not (path / name).is_file())
    if missing:
        raise ValueError("Missing successful outputs: " + ", ".join(missing))


def verify(path, complete=False):
    path = safe_run(path)
    if complete:
        require_success(path)
        recorded = json.loads((path / "completion.json").read_text())["sha256"]
        if inventory(path) != recorded:
            raise ValueError("Completion inventory mismatch")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["runtime"] != runtime() or manifest["config"] != CONFIG:
        raise ValueError("Runtime/config drift")
    if manifest["source"] != LOADED or source_hashes() != LOADED:
        raise ValueError("Implementation source drift")
    for rel, expected in manifest["source"].items():
        if digest(path / "source" / rel) != expected:
            raise ValueError("Source snapshot drift")
    for name, key, original_path in (
        ("spec.md", "spec_hash", SPEC),
        ("source_protocol.md", "protocol_hash", PROTOCOL),
    ):
        if digest(path / name) != manifest[key] or (
            name != "spec.md" and digest(original_path) != manifest[key]
        ):
            raise ValueError("Spec/protocol drift")
    if (
        frozen_spec_hash(SPEC) != manifest["frozen_spec_hash"]
        or frozen_spec_hash(path / "spec.md") != manifest["frozen_spec_hash"]
    ):
        raise ValueError("Frozen spec intent drift")
    for source, expected in manifest["inputs"].items():
        if digest(source) != expected:
            raise ValueError("Input drift")
    return manifest


def seal(path, failed=False):
    path = safe_run(path)
    if failed:
        if not (path / "failure.json").is_file():
            raise ValueError("Failed sealing requires failure.json")
    else:
        require_success(path)
    write_json(path / "completion.json", dict(sha256=inventory(path)))
    for p in path.rglob("*"):
        if p.is_file():
            p.chmod(0o444)


def fail(path, exc):
    """A failure is sealed evidence, never a partial success report."""
    path = safe_run(path)
    for name in ("report.md", "report.html"):
        (path / name).unlink(missing_ok=True)
    write_json(path / "failure.json", dict(type=type(exc).__name__, error=str(exc)))
    seal(path, failed=True)
