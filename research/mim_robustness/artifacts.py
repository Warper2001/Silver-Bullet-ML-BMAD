"""Exclusive creation, source/input pinning, and inventory verification."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import uuid
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
BASELINE_NAME = "20260910T210224-historical-f3950efb68"
BASELINE = ROOT / "research/mim_comparison/runs" / BASELINE_NAME
DATA = ROOT / "data/mim_x/mnq_1min_by_contract.csv"
DATA_HASH = "ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea"
BASELINE_COMPLETION_HASH = (
    "968d14b1d6c2fe8d31ff4e6a54a705c13c105d64a6f3ae6a8ec9e24448ca469a"
)
PROTOCOL = ROOT / "_bmad-output/specs/spec-mim-nb-sharpe-experiments/protocol.md"
CONFIG = dict(
    labels="end",
    arms=["A", "R", "E", "F", "P"],
    delays=[2, 1],
    costs=[2.24, 3.24, 6.24],
    capital=10000,
    risk_free=0,
    annualization=252,
    window=30,
    expected_sessions=1323,
    threshold=0.30,
    stop_points=250,
    reference_guard=-1000,
    quantity=1,
    draws=20000,
    seed=7,
    blocks=[5, 10, 20],
    minimum_delta_sharpe=0.20,
    profit_retention=0.75,
    drawdown_ratio=0.80,
    adjusted_confidence=0.9875,
    max_undefined_fraction=0.01,
    highcost_rule="positive daily expectancy and deltaSharpe, no worse drawdown",
    reset_rule="post-exit later flat inside-band close, subsequent entry check",
)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def source_hashes():
    paths = list(BASE.glob("*.py")) + [
        ROOT / "research/mim_comparison" / name for name in ("data.py", "references.py")
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


LOADED = source_hashes()


def safe_run(path):
    path = Path(path).resolve()
    if (
        RUNS.is_symlink()
        or RUNS.resolve() != BASE.resolve() / "runs"
        or path.parent != RUNS.resolve()
        or path.is_symlink()
    ):
        raise ValueError("Output must be a direct isolated robustness runs child")
    return path


def inventory(path):
    if any(p.is_symlink() for p in path.rglob("*")):
        raise ValueError("Symlink in immutable inventory")
    return {
        str(p.relative_to(path)): digest(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "completion.json"
    }


def verify_inventory(path):
    path = Path(path)
    recorded = json.loads((path / "completion.json").read_text())["sha256"]
    if recorded != inventory(path):
        raise ValueError("Run inventory integrity failure")


def verify_baseline(data, baseline):
    baseline = Path(baseline).resolve()
    if (
        baseline.name != BASELINE_NAME
        or digest(baseline / "completion.json") != BASELINE_COMPLETION_HASH
    ):
        raise ValueError("Unapproved baseline run")
    verify_inventory(baseline)
    manifest = json.loads((baseline / "manifest.json").read_text())
    for rel, expected in manifest["source"].items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Frozen comparison source drift")
    if digest(data) != DATA_HASH:
        raise ValueError("Historical input hash drift")


def runtime():
    return dict(
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        machine=platform.machine(),
    )


def create(command, inputs, bindings=None):
    RUNS.mkdir(exist_ok=True)
    path = safe_run(
        RUNS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + command
            + "-"
            + uuid.uuid4().hex[:10]
        )
    )
    path.mkdir()
    try:
        current = source_hashes()
        if current != LOADED:
            raise ValueError("Loaded implementation source drift")
        manifest = dict(
            command=command,
            created_at=datetime.now(timezone.utc).isoformat(),
            config=CONFIG,
            runtime=runtime(),
            bindings=bindings or {},
            source=current,
            inputs={str(Path(p).resolve()): digest(p) for p in inputs},
            protocol_hash=digest(PROTOCOL),
            historical_only=True,
            deployment_authorized=False,
        )
        manifest["config_hash"] = hashlib.sha256(
            json.dumps(CONFIG, sort_keys=True).encode()
        ).hexdigest()
        for rel in current:
            snapshot = path / "source" / rel
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with open(snapshot, "xb") as stream:
                stream.write((ROOT / rel).read_bytes())
        with open(path / "protocol.md", "xb") as stream:
            stream.write(PROTOCOL.read_bytes())
        write_json(path / "manifest.json", manifest)
    except Exception as exc:
        write_json(path / "failure.json", dict(error=str(exc), type=type(exc).__name__))
        with open(path / "report.md", "x") as stream:
            stream.write("# Failed research invocation\n\n" + str(exc) + "\n")
        seal(path)
        raise
    return path


def verify(path, complete=False):
    path = safe_run(path)
    if complete:
        verify_inventory(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["runtime"] != runtime():
        raise ValueError("Numerical runtime drift")
    if source_hashes() != LOADED or manifest["source"] != LOADED:
        raise ValueError("Implementation source drift")
    if (
        manifest["config"] != CONFIG
        or manifest["config_hash"]
        != hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("Configuration drift")
    if (
        digest(PROTOCOL) != manifest["protocol_hash"]
        or digest(path / "protocol.md") != manifest["protocol_hash"]
    ):
        raise ValueError("Protocol drift")
    for rel, expected in manifest["source"].items():
        if digest(path / "source" / rel) != expected:
            raise ValueError("Source snapshot drift")
    for input_path, expected in manifest["inputs"].items():
        if digest(input_path) != expected:
            raise ValueError("Input drift")
    return manifest


def seal(path):
    path = safe_run(path)
    write_json(path / "completion.json", dict(sha256=inventory(path)))
    for file in path.rglob("*"):
        if file.is_file():
            file.chmod(0o444)
