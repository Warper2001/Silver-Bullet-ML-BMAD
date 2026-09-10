"""Exclusive-create run artifacts and protocol hashes, sealed before return computation."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, uuid

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    def convert(item):
        if hasattr(item, "item"):
            return item.item()
        return str(item)

    with open(path, "x") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=convert, allow_nan=False)
        f.write("\n")


# Pin the implementation present when this process loaded the research package.
LOADED_SOURCE_HASHES = {str(p.relative_to(ROOT)): digest(p) for p in BASE.glob("*.py")}


def verify_run(path, data=None):
    manifest = json.loads((path / "manifest.json").read_text())
    current = {str(p.relative_to(ROOT)): digest(p) for p in BASE.glob("*.py")}
    if current != manifest["source"] or current != LOADED_SOURCE_HASHES:
        raise ValueError("Implementation changed during run")
    if (
        data is not None
        and digest(data) != manifest["inputs"][str(Path(data).resolve())]
    ):
        raise ValueError("Historical input changed during run")


def make_run(command, inputs, config):
    RUNS.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    path = RUNS / (
        now.strftime("%Y%m%dT%H%M%S") + "-" + command + "-" + uuid.uuid4().hex[:10]
    )
    path.mkdir()
    protocol = {
        "freeze": now.isoformat(),
        "first_session_rule": "First complete ET session whose 09:30 open is strictly after freeze",
        "eligible_sessions": 120,
        "calendar_month_limit": 9,
        "arms": ["A", "B"],
        "timeliness_seconds": 60,
        "primary_delay": 2,
        "costs": [2.24, 3.24, 6.24],
        "primary_cost": 2.24,
        "bootstrap_draws": 20000,
        "seed": 7,
        "mean_blocks": [5, 10, 20],
        "incremental_threshold_usd": 5,
        "confidence": 0.95,
        "no_interim_efficacy": True,
        "history_exposed": True,
        "deployment_authorized": False,
        "expiry": "quarterly MNQ third Friday, exclude expiry date from 09:30 session",
        "quantity": 1,
        "stop_points": 250,
        "gross_reference_guard_usd": -1000,
        "eod": "16:00 close proxy",
        "provenance_gate": "explicit contract per bar, historical audit and warmup; operational contractless logs ineligible",
    }
    write_json(path / "protocol.json", protocol)
    (path / "source").mkdir()
    source = {}
    for p in sorted(BASE.glob("*.py")):
        snapshot = path / "source" / p.name
        snapshot.write_bytes(p.read_bytes())
        source[str(p.relative_to(ROOT))] = digest(snapshot)
    evidence = {
        str(p.relative_to(ROOT)): digest(p)
        for p in sorted((BASE / "evidence").glob("*"))
        if p.is_file()
    }
    stream = Path(config["data"]).resolve() if command == "shadow" else None
    input_hashes = {
        str(Path(p).resolve()): digest(p)
        for p in inputs
        if stream is None or Path(p).resolve() != stream
    }
    if command == "shadow":
        input_hashes[str(Path(config["warmup"]).resolve())] = digest(config["warmup"])
    if command == "historical" and (ROOT / "data/mim_nb/decisions.csv").exists():
        operational = ROOT / "data/mim_nb/decisions.csv"
        snapshot = path / "operational-decisions-snapshot.csv"
        snapshot.write_bytes(operational.read_bytes())
        input_hashes[str(operational)] = digest(snapshot)
    manifest = {
        "command": command,
        "created_at": now.isoformat(),
        "config": config,
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "inputs": input_hashes,
        "source": source,
        "evidence": evidence,
        "protocol_hash": digest(path / "protocol.json"),
        "stream_input": (
            {
                "path": str(stream),
                "integrity_basis": "durable first-observation journal and invalid-row tombstones; source may append or change",
            }
            if stream
            else None
        ),
        "history_exposed": True,
        "immutable_policy": "exclusive create; completed files chmod read-only; no overwrite",
    }
    write_json(path / "manifest.json", manifest)
    return path, protocol


def seal(path):
    files = [
        p for p in path.iterdir() if p.is_file() and p.name != "completion.json"
    ] + list((path / "source").glob("*.py"))
    hashes = {str(p.relative_to(path)): digest(p) for p in sorted(files)}
    write_json(path / "completion.json", {"sha256": hashes})
    for p in files + [path / "completion.json"]:
        p.chmod(0o444)


def paired_daily(frame):
    import pandas as pd

    if not len(frame):
        return pd.DataFrame(columns=["day", "delay", "cost", "A", "B", "B_minus_A"])
    paired = (
        frame[frame.arm.isin(["A", "B"])]
        .pivot(index=["day", "delay", "cost"], columns="arm", values="net")
        .reset_index()
    )
    paired["B_minus_A"] = paired.B - paired.A
    return paired


def ensure_outputs(path):
    import pandas as pd

    schemas = {
        "daily.csv": [
            "day",
            "contract",
            "arm",
            "delay",
            "cost",
            "net",
            "eligible",
            "exclusion",
        ],
        "ledger.csv": [
            "day",
            "contract",
            "arm",
            "event_timestamp",
            "modeled_fill_timestamp",
            "fill",
            "quantity",
            "costs",
            "reason",
            "eligible",
            "exclusion",
        ],
        "decisions.csv": ["event_timestamp", "arm", "target", "eligible", "exclusion"],
    }
    for name, columns in schemas.items():
        if not (path / name).exists() or (path / name).stat().st_size <= 1:
            if (path / name).exists():
                (
                    path / name
                ).unlink()  # replace only an empty newly created placeholder
            with open(path / name, "x") as out:
                pd.DataFrame(columns=columns).to_csv(out, index=False)
    if not (path / "paired-daily.csv").exists():
        paired_daily(pd.read_csv(path / "daily.csv")).to_csv(
            path / "paired-daily.csv", index=False
        )
    if not (path / "report.md").exists():
        record = next(
            (
                json.loads((path / name).read_text())
                for name in (
                    "status.json",
                    "decision.json",
                    "failure.json",
                    "audit.json",
                )
                if (path / name).exists()
            ),
            {},
        )
        with open(path / "report.md", "x") as out:
            out.write(
                "# Research run status\n\n"
                + json.dumps(record, indent=2)
                + "\n\nNo interim efficacy decision or deployment authorization. Empty ledgers mean unavailable/not applicable.\n"
            )
