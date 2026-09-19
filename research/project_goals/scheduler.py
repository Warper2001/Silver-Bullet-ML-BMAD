"""Operational wrapper only: never reads collector outcomes or modifies its protocol."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import time
from datetime import datetime, timezone
from .common import digest, input_path
from .archive import archive_chunks

RUN = "20260910T214803-shadow-47055eba62/collector"
ADAPTER = "20260910-contract-feed"
POLL = "20260910-contract-feed-poll"


def read_db(path, sql):
    with sqlite3.connect(
        input_path(path).as_uri() + "?mode=ro", uri=True, timeout=1
    ) as db:
        return db.execute(sql).fetchall()


def prefix(path, record, check_identity=False):
    h = hashlib.sha256()
    with input_path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if check_identity and [before.st_dev, before.st_ino] != record["identity"]:
            raise ValueError("consumed source replaced: " + str(path))
        left = record["size"]
        while left:
            chunk = stream.read(min(left, 1048576))
            if not chunk:
                raise ValueError("consumed source truncated: " + str(path))
            h.update(chunk)
            left -= len(chunk)
    if h.hexdigest() != record["hash"]:
        raise ValueError("consumed prefix drift: " + str(path))


def preflight(root):
    root = input_path(root)
    base = root / "research/mim_comparison"
    runs = base / "runs"
    frozen = json.loads((runs / RUN / "freeze.json").read_text())
    adapter = json.loads((runs / ADAPTER / "freeze.json").read_text())
    poll = json.loads((runs / POLL / "freeze.json").read_text())
    if {p.name: digest(p) for p in base.glob("*.py")} != frozen["source"]:
        raise ValueError("frozen collector source drift")
    if {p.name: digest(p) for p in (base / "feed_adapter").glob("*.py")} != adapter[
        "sources"
    ]:
        raise ValueError("frozen adapter source drift")
    if digest(base / "poll.sh") != poll["wrapper_sha256"]:
        raise ValueError("frozen poll source drift")
    if digest(root / "data/mim_x/mnq_1min_by_contract.csv") != frozen["warmup_hash"]:
        raise ValueError("frozen warmup drift")
    if (
        digest(Path(poll["historical_run"]) / "manifest.json")
        != frozen["historical_manifest_hash"]
    ):
        raise ValueError("historical manifest drift")
    if (
        Path(poll["collector"]).resolve() != (runs / RUN).resolve()
        or poll["prepare_only"]
    ):
        raise ValueError("poll points to unexpected collector")
    protocol = frozen["protocol"]
    if (
        protocol["eligible_sessions"] != 120
        or protocol["timeliness_seconds"] != 60
        or not protocol["no_interim_efficacy"]
        or not protocol["freeze"].startswith("2026-09-10")
        or protocol["calendar_month_limit"] != 9
    ):
        raise ValueError("frozen protocol drift")
    meta = {
        key: json.loads(value)
        for key, value in read_db(
            runs / ADAPTER / "journal.sqlite", "SELECT key,value FROM meta"
        )
    }
    for kind in ("bars", "log"):
        prefix(adapter[kind], meta[kind + "_source"], True)
    prefix(runs / ADAPTER / "feed.csv", meta["feed"])
    return dict(
        status="VERIFIED",
        freeze=protocol["freeze"],
        deadline="2027-06-10",
        required_eligible_sessions=120,
        timeliness_seconds=60,
        interim_efficacy=False,
    )


def health(root):
    """Only session eligibility and exclusion tables; deliberately no outcome queries."""
    journal = input_path(root) / "research/mim_comparison/runs" / RUN / "journal.sqlite"
    sessions = read_db(
        journal, "SELECT day,contract,eligible,reason FROM sessions ORDER BY day"
    )
    flags = read_db(journal, "SELECT day,reason FROM session_flags ORDER BY day,reason")
    count = sum(bool(row[2]) for row in sessions)
    return dict(
        eligible_sessions=count,
        required_eligible_sessions=120,
        commissioning=(
            "FULL_ELIGIBLE_SESSION_OBSERVED"
            if count
            else "PENDING_FULL_ELIGIBLE_SESSION"
        ),
        sessions=[
            dict(day=d, contract=c, eligible=e, reason=r) for d, c, e, r in sessions
        ],
        exclusions=[dict(day=d, reason=r) for d, r in flags],
        deadline="2027-06-10",
        no_interim_efficacy=True,
    )


def atomic(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_identity(pid):
    """Read process identity without signals, preventing PID-reuse confusion."""
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        if parts[0] == "Z":
            return None
        return dict(
            pid=pid,
            start_ticks=parts[19],
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        )
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def child_health(state):
    path = Path(state) / "child.json"
    if not path.exists():
        return dict(status="NO_PENDING_CHILD")
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict):
            raise ValueError("invalid receipt")
        started_at = datetime.fromisoformat(record["started_at"])
        if started_at.tzinfo is None:
            raise ValueError("naive receipt time")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(
            status="OPERATOR_RECOVERY_REQUIRED",
            operator_recovery_required=True,
            reason="Unreadable child receipt; identity unknown",
        )
    try:
        beat = json.loads((Path(state) / "heartbeat.json").read_text())
        if not isinstance(beat, dict):
            beat = {}
    except (OSError, ValueError):
        beat = {}
    identity = record.get("identity")
    alive = bool(
        isinstance(identity, dict)
        and identity.get("pid")
        and process_identity(identity["pid"]) == identity
    )
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    supervisor = record.get("supervisor")
    supervised = bool(
        isinstance(supervisor, dict)
        and supervisor.get("pid")
        and process_identity(supervisor["pid"]) == supervisor
        and beat.get("supervisor") == supervisor
        and beat.get("started_at") == record.get("poll_started_at")
    )
    observed = beat.get("last_observed_at", beat.get("started_at"))
    try:
        heartbeat_age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(observed)
        ).total_seconds()
    except (TypeError, ValueError):
        heartbeat_age = float("inf")
    overdue = elapsed >= record.get("stale_after_seconds", 60)
    if supervised and beat.get("status") not in (
        "OPERATOR_RECOVERY_REQUIRED",
        "PREFLIGHT_OR_HEALTH_FAILED",
    ):
        if alive:
            status = (
                "OVERDUE_EXECUTION"
                if overdue or heartbeat_age > 10
                else "SUPERVISED_POLLING"
            )
        else:
            status = "OVERDUE_FINALIZATION" if heartbeat_age > 60 else "FINALIZING"
        recovery = False
        action = "Supervisor owns this poll; observe latency and finalization. No replacement or process killing."
    else:
        status = "ORPHANED_CHILD" if alive else "OPERATOR_RECOVERY_REQUIRED"
        recovery = True
        action = "Review receipt and journals; preserve receipt before manual recovery. Do not replace a living child."
    return dict(
        status=status,
        child_alive=alive,
        supervisor_alive=supervised,
        operator_recovery_required=recovery,
        elapsed_seconds=elapsed,
        child=record,
        action=action,
    )


def run_monitored(
    command, root, output, state, beat, lock_fd, clock, stale_after, popen
):
    # Child inherits the singleton lock, so an outer crash cannot unlock a running poll.
    child = popen(
        command, cwd=root, stdout=output, stderr=subprocess.STDOUT, pass_fds=(lock_fd,)
    )
    started = clock()
    receipt = dict(
        identity=process_identity(child.pid),
        pid=child.pid,
        started_at=datetime.now(timezone.utc).isoformat(),
        stale_after_seconds=stale_after,
        supervisor=process_identity(os.getpid()),
        poll_started_at=beat["started_at"],
    )
    atomic(state / "child.json", receipt)
    stale = False
    while child.poll() is None:
        elapsed = clock() - started
        stale = stale or elapsed >= stale_after
        beat.update(
            status=("OVERDUE_EXECUTION" if stale else "SUPERVISED_POLLING"),
            child_pid=child.pid,
            child_alive=True,
            child_elapsed_seconds=elapsed,
            operator_recovery_required=False,
            last_observed_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic(state / "heartbeat.json", beat)
        try:
            child.wait(timeout=min(1.0, max(0.01, stale_after / 4)))
        except subprocess.TimeoutExpired:
            pass
    beat.update(
        status="FINALIZING",
        child_alive=False,
        operator_recovery_required=False,
        stale_child_observed=stale,
        last_observed_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic(state / "heartbeat.json", beat)
    if stale:
        beat["recovery_note"] = (
            "Late child completion observed; preserve latency alert, finish archival, then resume permitted cadence."
        )
    return subprocess.CompletedProcess(command, child.returncode)


def poll_once(
    root,
    state,
    runner=None,
    clock=time.monotonic,
    *,
    stale_after=60.0,
    popen=subprocess.Popen,
):
    root = input_path(root)
    state = Path(state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    started = clock()
    beat = dict(
        started_at=datetime.now(timezone.utc).isoformat(),
        status="RUNNING",
        supervisor=process_identity(os.getpid()),
    )
    with (state / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status="OVERLAP_REFUSED")
        pending = child_health(state)
        if pending["status"] != "NO_PENDING_CHILD":
            atomic(state / "heartbeat.json", pending)
            return pending
        atomic(state / "heartbeat.json", beat)
        try:
            usage = shutil.disk_usage(root)
            beat["storage"] = dict(
                free_bytes=usage.free,
                reserve_bytes=4 * 1024**3,
                snapshot_growth_risk="New completed snapshots are byte-verified and compressed; historical and authoritative journals remain untouched.",
            )
            if usage.free < 4 * 1024**3:
                raise ValueError(
                    "disk safety reserve reached; poll refused, commissioning/storage maintenance required"
                )
            beat["preflight"] = preflight(root)
            runs = root / "research/mim_comparison/runs"
            existed = set(runs.glob("*-shadow-*")) | set(
                (runs / ADAPTER).glob("poll-*")
            )
            log = state / "poll.log"
            offset = log.stat().st_size if log.exists() else 0
            with log.open("a") as output:
                command = ["bash", str(root / "research/mim_comparison/poll.sh")]
                if (
                    runner is not None
                ):  # deterministic injected runner for isolated tests only
                    result = runner(
                        command,
                        cwd=root,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                else:
                    result = run_monitored(
                        command,
                        root,
                        output,
                        state,
                        beat,
                        lock.fileno(),
                        clock,
                        stale_after,
                        popen,
                    )
            beat["poll_returncode"] = result.returncode
            beat["health"] = health(root)
            beat["archives"] = []
            if result.returncode == 0:
                with log.open() as stream:
                    stream.seek(offset)
                    lines = stream.readlines()
                candidates = []
                for line in lines:
                    line = line.strip()
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict) and "invocation" in item:
                            candidates.append(Path(item["invocation"]))
                    except ValueError:
                        if (
                            line.startswith(str(runs) + "/")
                            and "-shadow-" in Path(line).name
                        ):
                            candidates.append(Path(line))
                if len(candidates) != 2:
                    raise ValueError(
                        "cannot identify both completed poll artifact directories"
                    )
                for directory in candidates:
                    beat["archives"].append(
                        archive_chunks(directory, runs, state / "archives", existed)
                    )
            beat["status"] = "SUCCESS" if result.returncode == 0 else "POLL_FAILED"
            if runner is None:
                (state / "child.json").unlink()
        except Exception as exc:
            beat.update(status="PREFLIGHT_OR_HEALTH_FAILED", error=str(exc))
            if (state / "child.json").exists():
                beat.update(
                    status="OPERATOR_RECOVERY_REQUIRED", operator_recovery_required=True
                )
        beat["elapsed_seconds"] = clock() - started
        beat["latency_budget_exceeded"] = beat["elapsed_seconds"] > 60
        beat["latency_note"] = (
            "End-to-end poll wall time; collector independently enforces actual per-bar 60-second receipt rule."
        )
        beat["finished_at"] = datetime.now(timezone.utc).isoformat()
        beat["next_poll"] = "one second after completion; systemd OnUnitInactiveSec=1s"
        atomic(state / "heartbeat.json", beat)
        return beat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()
    if args.health_only:
        result = health(args.root)
        if args.state:
            result["scheduler"] = child_health(args.state)
        print(json.dumps(result, sort_keys=True))
        return
    if args.state is None:
        parser.error("--state required for poll")
    result = poll_once(args.root, args.state)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
