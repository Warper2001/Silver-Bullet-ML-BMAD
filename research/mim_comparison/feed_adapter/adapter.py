"""Append-only first-observation journal; no production imports or connectivity."""

import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs
import uuid

PACKAGE = Path(__file__).resolve().parent
RUNS = PACKAGE.parent / "runs"
FIELDS = ["ts_utc", "open", "high", "low", "close", "volume", "received_at", "chain"]
OUTPUT = [
    "contract",
    "timestamp",
    *FIELDS[1:7],
    "observed_at",
    "inference",
    "chain_status",
    "request_evidence",
]
LABEL = "log_inferred_not_authenticated_response_identity"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("offset-aware timestamp required")
    return result.astimezone(timezone.utc)


def csv_line(values):
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator="\n").writerow(values)
    return out.getvalue().encode()


def put(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, encoded(value)))


def get(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def connect(state):
    db = sqlite3.connect(state / "journal.sqlite")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS evidence(offset INTEGER PRIMARY KEY,stamp REAL,kind TEXT,payload TEXT);
    CREATE INDEX IF NOT EXISTS evidence_time ON evidence(kind,stamp,offset);
    CREATE TABLE IF NOT EXISTS first_seen(identity TEXT PRIMARY KEY,offset INTEGER);
    CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY,offset INTEGER UNIQUE,raw TEXT,decision TEXT,output BLOB);
    """)
    db.commit()
    return db


def checked_source(db, kind, path):
    """Verify all bytes previously seen, including pending incomplete lines."""
    stream = open(path, "rb")
    stat = os.fstat(stream.fileno())
    old = get(db, kind + "_source")
    size = stat.st_size
    digest = hashlib.sha256()
    if old:
        if [stat.st_dev, stat.st_ino] != old["identity"] or size < old["size"]:
            raise ValueError(kind + " source replaced or truncated")
        remaining = old["size"]
        while remaining:
            block = stream.read(min(1048576, remaining))
            if not block:
                raise ValueError(kind + " source truncated during verification")
            digest.update(block)
            remaining -= len(block)
        if digest.hexdigest() != old["hash"]:
            raise ValueError(kind + " consumed prefix changed")
    remaining = size - (old["size"] if old else 0)
    while remaining:
        block = stream.read(min(1048576, remaining))
        if not block:
            raise ValueError(kind + " source truncated during snapshot")
        digest.update(block)
        remaining -= len(block)
    put(
        db,
        kind + "_source",
        {
            "identity": [stat.st_dev, stat.st_ino],
            "size": size,
            "hash": digest.hexdigest(),
        },
    )
    db.commit()
    stream.seek(get(db, kind + "_offset", 0))
    return stream, size


def recheck_source(db, kind, path):
    frozen = get(db, kind + "_source")
    with open(path, "rb") as stream:
        stat = os.fstat(stream.fileno())
        if [stat.st_dev, stat.st_ino] != frozen["identity"] or stat.st_size < frozen[
            "size"
        ]:
            raise ValueError(kind + " source changed during consumption")
        digest = hashlib.sha256()
        remaining = frozen["size"]
        while remaining:
            block = stream.read(min(1048576, remaining))
            if not block:
                raise ValueError(kind + " source truncated during consumption")
            digest.update(block)
            remaining -= len(block)
        if digest.hexdigest() != frozen["hash"]:
            raise ValueError(kind + " source changed during consumption")


def lines(stream, size):
    while stream.tell() < size:
        offset = stream.tell()
        raw = stream.readline(size - offset)
        if not raw.endswith(b"\n"):
            break
        yield offset, stream.tell(), raw


def parse_log(raw):
    text = raw.decode("utf-8", errors="replace").rstrip()
    match = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})", text)
    if not match:
        return None
    stamp = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").replace(
        tzinfo=timezone.utc
    )
    kind, payload = None, {"text": text, "timestamp": stamp.isoformat()}
    if "MIM-NB LIVE" in text or "AUTOROLL startup:" in text:
        kind = "context"
        payload.update(source=None, contract=None)
    elif "DATA:" in text:
        kind = "context"
        source = re.search(r"DATA: (\w+) \(signal\)", text)
        contract = re.search(
            r"px_contract=CON\.F\.US\.MNQ\.([HMUZ]\d{2})(?:\s|$)", text
        )
        payload.update(
            source=source[1] if source else None,
            contract="MNQ" + contract[1] if contract else None,
        )
    elif "AUTOROLL: front month" in text:
        kind = "roll"
        roll = re.search(
            r"AUTOROLL: front month (MNQ[HMUZ]\d{2}) → (MNQ[HMUZ]\d{2})(?:\s|$)", text
        )
        payload.update(
            old=roll[1] if roll else None, contract=roll[2] if roll else None
        )
    else:
        request = re.search(r'HTTP Request: GET (\S+) "HTTP/[\d.]+ 200(?: OK)?"', text)
        if request:
            url = urlsplit(request[1])
            symbol = re.fullmatch(
                r"/v3/marketdata/barcharts/(MNQ[HMUZ]\d{2})", url.path
            )
            query = parse_qs(url.query)
            if (
                url.scheme == "https"
                and url.netloc == "api.tradestation.com"
                and symbol
                and query.get("interval") == ["1"]
                and query.get("unit") == ["Minute"]
            ):
                kind = "request"
                payload["contract"] = symbol[1]
    return (stamp.timestamp(), kind, payload) if kind else None


def index_log(db, stream, size):
    for offset, end, raw in lines(stream, size):
        parsed = parse_log(raw)
        if parsed:
            stamp, kind, payload = parsed
            payload["byte_offset"] = offset
            db.execute(
                "INSERT INTO evidence VALUES (?,?,?,?)",
                (offset, stamp, kind, encoded(payload)),
            )
        put(db, "log_offset", end)
    db.commit()


def context_at(db, stamp, offset):
    context = db.execute(
        "SELECT stamp,offset,payload FROM evidence WHERE kind='context' AND (stamp<? OR (stamp=? AND offset<=?)) ORDER BY stamp DESC,offset DESC LIMIT 1",
        (stamp, stamp, offset),
    ).fetchone()
    if not context:
        return None, []
    current = json.loads(context[2])
    evidence = [current]
    applied = set()
    for _, _, payload in db.execute(
        "SELECT stamp,offset,payload FROM evidence WHERE kind='roll' AND (stamp>? OR (stamp=? AND offset>?)) AND (stamp<? OR (stamp=? AND offset<=?)) ORDER BY stamp,offset",
        (context[0], context[0], context[1], stamp, stamp, offset),
    ):
        roll = json.loads(payload)
        transition = (roll["timestamp"], roll["old"], roll["contract"])
        if transition in applied:
            evidence.append(roll)
            continue
        applied.add(transition)
        if not roll["old"] or current["contract"] != roll["old"]:
            current = dict(current, source=None, contract=None)
        else:
            current = dict(current, contract=roll["contract"])
        evidence.append(roll)
    return current, evidence


def attribute(db, raw, now):
    event, receipt = timestamp(raw["ts_utc"]), timestamp(raw["received_at"])
    if event.second or event.microsecond:
        raise ValueError("nonminute_event")
    values = [float(raw[k]) for k in FIELDS[1:6]]
    o, h, l, c, v = values
    if (
        not all(math.isfinite(x) for x in values)
        or min(o, h, l, c) <= 0
        or v < 0
        or h < max(o, l, c)
        or l > min(o, h, c)
    ):
        raise ValueError("invalid_ohlcv")
    if receipt > now or event > receipt:
        raise ValueError("future_receipt_or_event")
    requests = db.execute(
        "SELECT stamp,offset,payload FROM evidence WHERE kind='request' AND stamp>=? AND stamp<=? ORDER BY stamp,offset LIMIT 1001",
        (receipt.timestamp() - 15, receipt.timestamp()),
    ).fetchall()
    if len(requests) > 1000:
        raise ValueError("excessive_request_ambiguity")
    if not requests:
        raise ValueError("absent_causal_request")
    symbols = {json.loads(r[2])["contract"] for r in requests}
    if len(symbols) != 1:
        raise ValueError("ambiguous_contract")
    symbol = symbols.pop()
    evidence = []
    for stamp, offset, payload in requests:
        context, transitions = context_at(db, stamp, offset)
        if (
            not context
            or context["source"] != "tradestation"
            or context["contract"] != symbol
        ):
            raise ValueError("unknown_or_contradictory_signal_context")
        evidence.append(dict(json.loads(payload), context=transitions))
    context, transitions = context_at(db, receipt.timestamp(), 2**63 - 1)
    if (
        not context
        or context["source"] != "tradestation"
        or context["contract"] != symbol
    ):
        raise ValueError("receipt_context_changed")
    return (
        symbol,
        evidence,
        (now - event).total_seconds() <= 60 and (receipt - event).total_seconds() <= 60,
    )


def sync_feed(db, state):
    """Committed byte prefix plus pending journal outputs recover a torn append exactly."""
    path = state / "feed.csv"
    old = get(db, "feed", {"id": 0, "size": 0, "hash": sha(b"")})
    digest = hashlib.sha256()
    with open(path, "a+b") as output:
        output.seek(0)
        remaining = old["size"]
        while remaining:
            block = output.read(min(1048576, remaining))
            if not block:
                raise ValueError("feed truncated")
            digest.update(block)
            remaining -= len(block)
        if digest.hexdigest() != old["hash"]:
            raise ValueError("feed canonical prefix changed")
        pending = []
        if not old["size"]:
            pending.append((0, csv_line(OUTPUT)))
        # Only newly committed outputs are queried on resume.
        pending.extend(
            db.execute(
                "SELECT id,output FROM observations WHERE id>? AND output IS NOT NULL ORDER BY id",
                (old["id"],),
            )
        )
        last_id = old["id"]
        size = old["size"]
        for row_id, canonical in pending:
            present = output.read(len(canonical))
            if not canonical.startswith(present):
                raise ValueError("feed uncommitted tail is not canonical")
            if len(present) < len(canonical):
                output.seek(0, 2)
                output.write(canonical[len(present) :])
                output.flush()
            digest.update(canonical)
            size += len(canonical)
            last_id = row_id
        if output.read(1):
            raise ValueError("feed unexpected tail")
        output.flush()
        os.fsync(output.fileno())
    put(db, "feed", {"id": last_id, "size": size, "hash": digest.hexdigest()})
    db.commit()


def markdown_report(invocation, counts, error=None):
    (invocation / "report.md").write_text(
        "# Contract feed poll\n\n"
        + ("Failed: " + str(error) + "\n\n" if error else "Completed.\n\n")
        + f"Mapped: {counts.get('mapped', 0)}. Excluded: {counts.get('excluded', 0)}. "
        + f"Timely at adapter: {counts.get('timely_at_adapter', 0)}.\n\n"
        + "Counts are cumulative observations, not prospective sessions. "
        + "Contract identity is inferred from logs, not authenticated response identity. "
        + "Post-break chain segments remain unanchored. Original shadow timeliness, "
        + "complete-minute coverage and warmup requirements still apply. No deployment authorization.\n"
    )


def seal(invocation):
    # Persist file contents, modes, directory entries, and parent entry before cursor commit.
    for artifact in invocation.iterdir():
        artifact.chmod(0o444)
        with open(artifact, "rb") as stream:
            os.fsync(stream.fileno())
    invocation.chmod(0o555)
    for directory in (invocation, invocation.parent):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def collect(config):
    state = Path(config["state"])
    db = connect(state)
    invocation = state / (
        "poll-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    invocation.mkdir()
    start_id = get(db, "snapshot_id", 0)
    try:
        if get(db, "integrity_failure"):
            raise ValueError("state permanently stopped after source integrity failure")
        bars, bar_size = checked_source(db, "bars", config["bars"])
        if get(db, "initial_bar_size") is None:
            put(db, "initial_bar_size", bar_size)
            db.commit()
        log, log_size = checked_source(db, "log", config["log"])
        with log:
            index_log(db, log, log_size)
        sync_feed(db, state)
        with bars:
            for offset, end, raw_bytes in lines(bars, bar_size):
                raw_text = raw_bytes.decode("utf-8", errors="replace")
                if offset == 0:
                    if next(csv.reader([raw_text])) != FIELDS:
                        raise ValueError("unexpected raw bar header")
                    put(db, "bars_offset", end)
                    db.commit()
                    continue
                now = datetime.now(timezone.utc)
                decision = {
                    "observed_at": now.isoformat(),
                    "offset": offset,
                    "inference": LABEL,
                }
                output = None
                raw = {}
                chain_head = get(db, "chain_head", "GENESIS")
                segment = get(db, "chain_segment", 0)
                try:
                    try:
                        identity = timestamp(
                            raw_text.split(",", 1)[0].strip('"')
                        ).isoformat()
                    except (ValueError, TypeError, IndexError):
                        put(db, "unidentified_corruption", True)
                        raise ValueError(
                            "unidentified_corruption_permanently_stops_mapping"
                        )
                    seen = db.execute(
                        "SELECT offset FROM first_seen WHERE identity=?", (identity,)
                    ).fetchone()
                    if not seen:
                        db.execute(
                            "INSERT INTO first_seen VALUES (?,?)", (identity, offset)
                        )
                    vals = next(csv.reader([raw_text], strict=True))
                    if len(vals) != len(FIELDS):
                        put(db, "chain_segment", segment + 1)
                        raise ValueError("invalid_csv_shape")
                    raw = dict(zip(FIELDS, vals))
                    expected = sha((chain_head + "|" + "|".join(vals[:-1])).encode())[
                        :16
                    ]
                    broken = expected != raw["chain"]
                    if broken:
                        segment += 1
                    put(db, "chain_head", raw["chain"])
                    put(db, "chain_segment", segment)
                    decision.update(
                        chain_expected=expected,
                        chain_recorded=raw["chain"],
                        chain_segment=segment,
                        chain_status=(
                            "genesis_linked"
                            if not segment
                            else "unanchored_after_break"
                        ),
                    )
                    if seen:
                        raise ValueError(
                            "duplicate_or_correction_first_observation_permanent"
                        )
                    if get(db, "unidentified_corruption"):
                        raise ValueError("prior_unidentified_corruption")
                    if broken:
                        raise ValueError("chain_break")
                    symbol, evidence, timely = attribute(db, raw, now)
                    if len(encoded(evidence)) > 131072:
                        raise ValueError("oversized_request_evidence")
                    decision["initial_historical_replay"] = offset < get(
                        db, "initial_bar_size"
                    )
                    if timely and decision["initial_historical_replay"]:
                        raise ValueError("initial_snapshot_replay_not_prospective")
                    decision.update(
                        status="mapped",
                        contract=symbol,
                        evidence=evidence,
                        timely_at_adapter=timely,
                    )
                    output = csv_line(
                        [
                            symbol,
                            raw["ts_utc"],
                            *[raw[k] for k in FIELDS[1:7]],
                            now.isoformat(),
                            LABEL,
                            decision["chain_status"],
                            encoded(evidence),
                        ]
                    )
                except (ValueError, TypeError, OverflowError, csv.Error) as exc:
                    if isinstance(exc, csv.Error):
                        put(db, "chain_segment", segment + 1)
                    decision.update(status="excluded", reason=str(exc))
                    try:
                        receipt_seconds = timestamp(
                            raw.get("received_at", "")
                        ).timestamp()
                        decision["nearby_request_evidence"] = [
                            json.loads(r[0])
                            for r in db.execute(
                                "SELECT payload FROM evidence WHERE kind='request' AND stamp>=? AND stamp<=? ORDER BY stamp,offset LIMIT 1001",
                                (receipt_seconds - 15, receipt_seconds + 15),
                            )
                        ]
                        _, transitions = context_at(db, receipt_seconds, 2**63 - 1)
                        decision["receipt_context_evidence"] = transitions
                    except (ValueError, TypeError, OverflowError):
                        pass
                db.execute(
                    "INSERT INTO observations(offset,raw,decision,output) VALUES (?,?,?,?)",
                    (offset, raw_text, encoded(decision), output),
                )
                counts = get(
                    db, "counts", {"mapped": 0, "excluded": 0, "timely_at_adapter": 0}
                )
                counts[decision["status"]] += 1
                counts["timely_at_adapter"] += int(
                    decision.get("timely_at_adapter", False)
                )
                bucket = (
                    "contracts"
                    if decision["status"] == "mapped"
                    else "exclusion_reasons"
                )
                key = (
                    decision.get("contract")
                    if bucket == "contracts"
                    else decision["reason"]
                )
                counts.setdefault(bucket, {})[key] = (
                    counts.setdefault(bucket, {}).get(key, 0) + 1
                )
                put(db, "counts", counts)
                put(db, "bars_offset", end)
                db.commit()  # First observation and chain decision durable before next bar.
        try:
            for kind in ("bars", "log"):
                recheck_source(db, kind, config[kind])
        except ValueError as exc:
            put(db, "integrity_failure", str(exc))
            db.commit()
            raise
        sync_feed(db, state)
        counts = get(db, "counts", {"mapped": 0, "excluded": 0, "timely_at_adapter": 0})
        with (
            open(invocation / "evidence.jsonl", "x") as accepted,
            open(invocation / "exclusions.jsonl", "x") as excluded,
        ):
            for raw, payload in db.execute(
                "SELECT raw,decision FROM observations WHERE id>? ORDER BY id",
                (start_id,),
            ):
                decision = json.loads(payload)
                (accepted if decision["status"] == "mapped" else excluded).write(
                    encoded(dict(decision, raw=raw)) + "\n"
                )
        put(db, "counts", counts)
        db.commit()
        report = dict(
            counts,
            inference=LABEL,
            chain_policy="exclude broken links; later segments remain unanchored",
            prospective_sessions_claimed=0,
            deployment_authorized=False,
        )
        (invocation / "report.json").write_text(encoded(report) + "\n")
        markdown_report(invocation, counts)
        manifest = {
            "freeze": json.loads((state / "freeze.json").read_text()),
            "inputs": {k: get(db, k + "_source") for k in ("bars", "log")},
            "feed": get(db, "feed"),
            "artifacts": {p.name: sha(p.read_bytes()) for p in invocation.iterdir()},
        }
        (invocation / "manifest.json").write_text(encoded(manifest) + "\n")
        seal(invocation)
        put(
            db,
            "snapshot_id",
            db.execute("SELECT COALESCE(MAX(id),0) FROM observations").fetchone()[0],
        )
        db.commit()
        return dict(report, invocation=str(invocation))
    except Exception as exc:
        # No snapshot cursor has committed: finish this pending invocation as failed.
        invocation.chmod(0o755)
        for artifact in invocation.iterdir():
            artifact.chmod(0o644)
        # Failed invocations retain committed decisions and their frozen binding.
        for filename, mapped in (("evidence.jsonl", True), ("exclusions.jsonl", False)):
            if not (invocation / filename).exists():
                with open(invocation / filename, "x") as out:
                    for raw_text, payload in db.execute(
                        "SELECT raw,decision FROM observations WHERE id>? ORDER BY id",
                        (start_id,),
                    ):
                        decision = json.loads(payload)
                        if (decision["status"] == "mapped") == mapped:
                            out.write(encoded(dict(decision, raw=raw_text)) + "\n")
        (invocation / "report.json").write_text(
            encoded(
                {"error": str(exc), "status": "failed", "counts": get(db, "counts", {})}
            )
            + "\n"
        )
        markdown_report(invocation, get(db, "counts", {}), exc)
        manifest = {
            "error": str(exc),
            "freeze": json.loads((state / "freeze.json").read_text()),
            "inputs": {k: get(db, k + "_source") for k in ("bars", "log")},
            "feed": get(db, "feed"),
            "artifacts": {
                p.name: sha(p.read_bytes())
                for p in invocation.iterdir()
                if p.name != "manifest.json"
            },
        }
        (invocation / "manifest.json").write_text(encoded(manifest) + "\n")
        seal(invocation)
        raise
    finally:
        db.close()


def sandbox_command(bars, log, state, argv=None):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap unavailable; no unsafe fallback")
    cmd = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "PYTHONPATH",
        "/code",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/code",
        "--dir",
        "/code/research/mim_comparison",
    ]
    for directory in ("/usr", "/lib", "/lib64"):
        if Path(directory).exists():
            cmd += ["--ro-bind", directory, directory]
    venv = Path(sys.prefix).resolve()
    if str(venv) not in ("/usr", "/usr/local"):
        cmd += ["--ro-bind", str(venv), str(venv)]
    cmd += [
        "--ro-bind",
        str(PACKAGE),
        "/code/research/mim_comparison/feed_adapter",
        "--ro-bind",
        str(bars),
        "/inputs/bars.csv",
        "--ro-bind",
        str(log),
        "/inputs/log",
        "--bind",
        str(state),
        "/state",
        "--remount-ro",
        "/code",
        "--chdir",
        "/code",
    ]
    return cmd + (
        argv
        or [sys.executable, "-m", "research.mim_comparison.feed_adapter", "--worker"]
    )


def launch(bars, log, state, log_timezone):
    bars, log, state = [Path(p).resolve() for p in (bars, log, state)]
    if not state.is_relative_to(RUNS.resolve()) or state == RUNS.resolve():
        raise ValueError("state must be under isolated research/mim_comparison/runs")
    if log_timezone != "UTC":
        raise ValueError("explicit UTC log timezone required")
    if bars == log or any(p.is_relative_to(state) for p in (bars, log)):
        raise ValueError("inputs must be distinct and outside writable state")
    state.mkdir(parents=True, exist_ok=True)
    with open(state / "lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("adapter invocation already running") from exc
        frozen = {
            "bars": str(bars),
            "log": str(log),
            "log_timezone": log_timezone,
            "sources": {
                p.name: sha(p.read_bytes()) for p in sorted(PACKAGE.glob("*.py"))
            },
            "join_seconds": 15,
            "inference": LABEL,
        }
        freeze = state / "freeze.json"
        if freeze.exists():
            if json.loads(freeze.read_text()) != frozen:
                raise ValueError("frozen adapter source/config drift")
        else:
            with open(freeze, "x") as output:
                output.write(encoded(frozen) + "\n")
                output.flush()
                os.fsync(output.fileno())
        result = subprocess.run(
            sandbox_command(bars, log, state), env={}, text=True, capture_output=True
        )
        if result.returncode:
            raise RuntimeError("sandboxed adapter failed: " + result.stderr[-3000:])
        report = json.loads(result.stdout)
        report["invocation"] = str(
            state / Path(report["invocation"]).relative_to("/state")
        )
        return report
