"""Fixed offline pilot boundary and atomic, deterministic artifact publication."""

import ctypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import zoneinfo

from .builder import Builder, coverage, read_auxiliary
from .replay import replay, REPLAY_ROOT, MODEL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path("/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907")


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while raw := f.read(4 << 20):
            h.update(raw)
    return h.hexdigest()


def verify_pins(pins):
    for item in pins["files"]:
        path = DATA / item["file"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError("acquisition pin mismatch: " + item["file"])
    for path, expected in pins["sources"].items():
        if sha(REPLAY_ROOT / path) != expected:
            raise ValueError("frozen source pin mismatch: " + path)
    if sha(MODEL) != pins["model_sha256"]:
        raise ValueError("model pin mismatch")
    if "timezone" in pins:
        selected = next(
            (
                Path(root) / "America/New_York"
                for root in zoneinfo.TZPATH
                if (Path(root) / "America/New_York").is_file()
            ),
            None,
        )
        if selected is None or sha(selected) != pins["timezone"]["sha256"]:
            raise ValueError("timezone pin mismatch")
    for name, expected in pins["versions"].items():
        if version(name) != expected:
            raise ValueError("dependency pin mismatch: " + name)


def no_replace(source, target):
    # Linux renameat2 RENAME_NOREPLACE closes the empty-directory collision race.
    libc = ctypes.CDLL(None, use_errno=True)
    code = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if code:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(target))


def publish(output, artifacts):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".yank-native-minute-", dir=output.parent))
    try:
        hashes = {}
        for name, raw in artifacts.items():
            (staging / name).write_bytes(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        (staging / "manifest.json").write_bytes(
            canonical({"artifacts": hashes, "research_status": "HOLD_VALIDATION"})
        )
        no_replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(output):
    output = Path(output).absolute()
    resolved = output.resolve()
    for protected in (
        DATA,
        REPLAY_ROOT,
        MODEL.parent,
        Path("/root/Silver-Bullet-ML-BMAD/data"),
        ROOT / "src",
        ROOT / "tests",
        ROOT / "docs",
    ):
        if resolved.is_relative_to(protected.resolve()):
            raise ValueError("output overlaps protected input/source root")
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    pins = json.loads((HERE / "pins.json").read_text())
    verify_pins(pins)
    code_paths = sorted(HERE.glob("*.py")) + [
        HERE / "pins.json",
        ROOT / "src/cli/check_yank_native_minute.py",
        ROOT / "pyproject.toml",
    ]
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
    builder = Builder()
    statuses, definitions = [], []
    for item in pins["files"]:
        name = item["file"]
        path = DATA / name
        if name.endswith(".definition.dbn.zst"):
            definitions.extend(read_auxiliary(path, name, "definition"))
        elif name.endswith(".status.dbn.zst"):
            statuses.extend(read_auxiliary(path, name, "status"))
        elif name.endswith(".mbo.dbn.zst"):
            builder.read_mbo(path, name)
            print("Read " + path.name, flush=True)
    bars, delayed = builder.finish()
    rows = coverage(builder, statuses)
    for row in rows:
        if row["coverage"] == "NO_TRADE_STATUS_UNKNOWN":
            builder.holds.add("NO_TRADE_STATUS_UNKNOWN")
        if (
            not row["mbo_source_file_present"]
            and row["interval_status"] != "NONTRADING"
        ):
            builder.holds.add("ABSENT_MBO_FILE_WITHOUT_OBSERVED_NONTRADING_STATUS")
    result = replay(bars, pins, builder.holds)
    report = dict(
        status="PASS_DATA_CHECKS" if not builder.holds else "HOLD_DATA_CHECKS",
        research_status="HOLD_VALIDATION",
        replay_outcome=result["outcome"],
        hold_reasons=sorted(set(builder.holds) | set(result.get("reasons", []))),
        range={"start_ns": builder.start, "end_ns": builder.end},
        bar_count=len(bars),
        coverage_minutes=len(rows),
        coverage_counts=dict(
            __import__("collections").Counter(r["coverage"] for r in rows)
        ),
        counts=dict(builder.counts),
        files=builder.files,
        delayed_bar_count=len(delayed),
        source_pins=pins,
        code_sha256=code_hashes,
        runtime={"python": platform.python_version(), "packages": pins["versions"]},
        contract={
            "clock": "ts_recv capture UTC",
            "interval": "[start,end)",
            "label": "start",
            "price_scale": 1_000_000_000,
            "event_availability": "max(interval_end,trade_event_LAST_capture)",
            "ohlc_order": "native file/record order",
            "trade_action": "T excluding SNAPSHOT",
            "volume": "integer contracts; F excluded",
        },
        qualifications=[
            "Diagnostic 2025 development only; HOLD_VALIDATION remains",
            "No pre-purchase warm-up; LR WARMUP allowed until 1950 closes",
            "Volatility minimum 20 positive ATR observations/full 120; ADR up to 20 preceding observed days",
            "H1/M15 updates conservatively on following completed observed minute",
            "Expiry and holding periods count observed bars, not elapsed minutes",
            "No forward filling; empty coverage rows are not strategy bars",
            "Native stream pinned but no full book reconstruction or independent exchange completeness guarantee",
            "Initial status at day start is snapshot evidence, not actual arrival",
            "Next-bar OHLC modeled fills; intrabar order and execution unproved",
            "Historical 4 USD fee per closed trade remains a modeled assumption",
            "No terminal liquidation; frozen dollar-bar history retained",
            "No profitability validation; model training provenance and LR original-run identity unresolved",
        ],
    )
    artifacts = {
        "report.json": canonical(report),
        "bars.jsonl": b"".join(canonical(b) for b in bars),
        "exchange-diagnostic.jsonl": b"".join(
            canonical(builder.exchange[m]) for m in sorted(builder.exchange)
        ),
        "coverage.jsonl": b"".join(canonical(r) for r in rows),
        "status.json": canonical(statuses),
        "definitions.json": canonical(definitions),
        "delayed-events.json": canonical(delayed),
        "replay.json": canonical(result),
    }
    verify_pins(pins)
    if any(sha(ROOT / path) != expected for path, expected in code_hashes.items()):
        raise ValueError("code changed during build")
    publish(output, artifacts)
    return report
