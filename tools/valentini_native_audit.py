"""Hash-bound native volume measurement only; no market-evaluation admission."""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import secrets
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tools.valentini_reclaim import (  # noqa: E402
    Bar,
    Profile,
    Rejected,
    Session,
    canonical_json,
    protected_path,
    read_json,
    scheduled_minutes,
    sha256_file,
)

BUILDER_PATH = ROOT / "src/research/yank_native_minute/builder.py"
PINS_PATH = ROOT / "src/research/yank_native_minute/pins.json"
MINUTE = 60_000_000_000
TICK_NANOS = 250_000_000
VERSION = "valentini-native-measurement-v1"
CORE_HASHES = {
    "bars.jsonl": "efffa1e196d65aab61865361c719fbf4ec44a14cf29643c557a277e50f9bf1af",
    "coverage.jsonl": "8a014f618087fee2980983119887119fc4c7cd7e53f98ba297d3acecf7122df5",
    "definitions.json": "bf1f6c3b3338be32403cff1be1d48d67e52aa6b6d4940c42d498751610ef3959",
    "delayed-events.json": "b18902ca6b02a7dce28ad759cdd707e57ef30531c0c2f8924279a37cf082cd81",
    "exchange-diagnostic.jsonl": "3e62d01009764a4e919d68bfbe6dad6b51d9321f3b3a506323c6ee72da7f616e",
    "report.json": "10a1676ebaba2e4a9562fd5139f50bb78abaaf4dd036070ec6a9e6169b0bb713",
    "status.json": "f2e506da30092c2305d3717e4dfee1986a7484e59ad947582a24d9527ed24930",
}


def load_builder() -> Any:
    """Load only the bounded decoder; never execute src.research.__init__."""
    name = "_valentini_native_builder"
    source = protected_path(BUILDER_PATH).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if name in sys.modules:
        if getattr(sys.modules[name], "_source_sha256", None) != digest:
            raise Rejected("Cached native decoder source hash mismatch")
        return sys.modules[name]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BUILDER_PATH)
        if spec is None or spec.loader is None:
            raise Rejected("Cannot load native decoder")
        module = importlib.util.module_from_spec(spec)
        exec(compile(source, str(BUILDER_PATH), "exec"), module.__dict__)
        setattr(module, "_source_sha256", digest)
        sys.modules[name] = module
    return sys.modules[name]


def histogram_builder(start: int | None = None, end: int | None = None) -> Any:
    decoder = load_builder()

    class HistogramBuilder(decoder.Builder):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__(
                decoder.START if start is None else start,
                decoder.END if end is None else end,
            )
            self.histograms: dict[int, dict[int, int]] = {}

        def aggregate(
            self, rows: Any, indices: Any, file: str, clock: str, table: Any
        ) -> None:
            super().aggregate(rows, indices, file, clock, table)
            if clock != "recv":
                return
            minutes = rows[clock] // MINUTE * MINUTE
            for minute in np.unique(minutes):
                m = int(minute)
                if not self.start <= m < self.end:
                    continue
                selected = rows[minutes == minute]
                ticks, inverse = np.unique(
                    selected["price"] // TICK_NANOS, return_inverse=True
                )
                sums = np.zeros(len(ticks), dtype=np.uint64)
                np.add.at(sums, inverse, selected["size"])
                histogram = self.histograms.setdefault(m, {})
                for tick, volume in zip(ticks, sums):
                    t = int(tick)
                    histogram[t] = histogram.get(t, 0) + int(volume)

    return HistogramBuilder()


def native_area(volumes: Mapping[int, int]) -> tuple[int, int, int] | None:
    """Exact 70% integer volume; same lower POC/adjacent tie rules as Profile."""
    total = sum(volumes.values())
    if not total:
        return None
    poc = min(volumes, key=lambda tick: (-volumes[tick], tick))
    low = high = poc
    covered = volumes[poc]
    bottom, top = min(volumes), max(volumes)
    while covered * 10 < total * 7 and (low > bottom or high < top):
        left = volumes.get(low - 1, 0) if low > bottom else -1
        right = volumes.get(high + 1, 0) if high < top else -1
        if left >= right:
            low -= 1
            covered += left
        else:
            high += 1
            covered += right
    return low, high, poc


def strict_value(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise Rejected("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value: str) -> Any:
        raise Rejected(f"Nonfinite JSON: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    canonical_json(result)
    return result


def safe_input(raw: str | Path) -> Path:
    requested = Path(raw).expanduser().absolute()
    resolved = protected_path(requested)
    # Reject aliases, including benign ones, so identities remain reviewable.
    if requested != resolved or any(
        p.is_symlink() for p in (requested, *requested.parents)
    ):
        raise Rejected("Symlink input/output alias")
    if resolved.is_file() and resolved.stat().st_nlink != 1:
        raise Rejected("Hardlink input alias")
    return resolved


def child(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise Rejected("Input pin escapes its root")
    path = safe_input(root / relative)
    if not path.is_relative_to(root) or not path.is_file():
        raise Rejected("Missing or escaped pinned input")
    return path


def verify_sources(root: Path, pins: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pin in pins["files"]:
        name = pin["file"]
        if name in hashes:
            raise Rejected("Duplicate acquisition pin")
        path = child(root, name)
        digest = sha256_file(path)
        if path.stat().st_size != pin["bytes"] or digest != pin["sha256"]:
            raise Rejected(f"Acquisition pin mismatch: {name}")
        hashes[name] = digest
    for schema in ("mbo", "definition", "status"):
        if not any(n.endswith(f".{schema}.dbn.zst") for n in hashes):
            raise Rejected(f"Missing pinned {schema} input")
    return hashes


def load_reconstruction(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = child(root, "manifest.json")
    manifest = read_json(manifest_path)
    data: dict[str, Any] = {}
    hashes = {"manifest.json": sha256_file(manifest_path)}
    for name, expected in CORE_HASHES.items():
        path = child(root, name)
        digest = sha256_file(path)
        if digest != expected or manifest["artifacts"].get(name) != digest:
            raise Rejected(f"Saved reconstruction pin mismatch: {name}")
        hashes[name] = digest
        with path.open(encoding="utf-8") as stream:
            data[name] = (
                [strict_value(line) for line in stream]
                if name.endswith(".jsonl")
                else strict_value(stream.read())
            )
    return data, hashes


def minute_ns(value: str) -> int:
    if not isinstance(value, str):
        raise Rejected("Calendar timestamp must be explicit UTC")
    if any(set(digits) != {"0"} for digits in re.findall(r"[.,](\d+)", value)):
        raise Rejected("Calendar boundaries must be minute aligned")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() != timezone.utc.utcoffset(dt):
        raise Rejected("Calendar timestamp must be explicit UTC")
    if dt.second or dt.microsecond:
        raise Rejected("Calendar boundaries must be minute aligned")
    return int(dt.timestamp()) * 1_000_000_000


def utc(ns: int) -> datetime:
    return datetime.fromtimestamp(ns // 1_000_000_000, timezone.utc)


def calendar_sessions(calendar: Mapping[str, Any]) -> list[dict[str, Any]]:
    if calendar.get("schema_version") != 1:
        raise Rejected("Unknown calendar schema")
    if (
        calendar.get("instrument") != "MNQM5"
        or calendar.get("timezone") != "America/New_York"
    ):
        raise Rejected("Unsupported calendar instrument/timezone")
    sources = calendar.get("sources", [])
    ids = {s["id"] for s in sources}
    if (
        not sources
        or len(ids) != len(sources)
        or any(not s.get("url") for s in sources)
    ):
        raise Rejected("Calendar requires unique source citations")
    if not isinstance(calendar.get("unresolved_evidence"), list):
        raise Rejected("Calendar must retain unresolved evidence")

    def cited(row: Mapping[str, Any]) -> None:
        refs = row.get("sources", [])
        if not refs or not isinstance(refs, list) or not set(refs) <= ids:
            raise Rejected("Missing or unknown calendar source citation")

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    previous_end: int | None = None
    for item in calendar["sessions"]:
        cited(item)
        start, end = minute_ns(item["start"]), minute_ns(item["end"])
        if item["name"] in names or (previous_end is not None and start < previous_end):
            raise Rejected("Duplicate or overlapping calendar sessions")
        names.add(item["name"])
        previous_end = end
        if item.get("verification") not in {"VERIFIED", "UNVERIFIED"}:
            raise Rejected("Calendar verification must be explicit")
        if not isinstance(item.get("unresolved_evidence"), list):
            raise Rejected("Session must retain unresolved evidence")
        if item["verification"] == "VERIFIED" and item["unresolved_evidence"]:
            raise Rejected("Verified session has unresolved evidence")
        breaks = []
        for pause in item.get("breaks", []):
            cited(pause)
            breaks.append((minute_ns(pause["start"]), minute_ns(pause["end"])))
        session = Session(
            item["name"],
            utc(start),
            utc(end),
            tuple((utc(a), utc(b)) for a, b in breaks),
        )
        minutes = {
            int(dt.timestamp()) * 1_000_000_000 for dt in scheduled_minutes(session)
        }
        exceptions: dict[int, Any] = {}
        edges = {start, end} | {boundary for pause in breaks for boundary in pause}
        for exc in item.get("boundary_status_exceptions", []):
            cited(exc)
            minute = minute_ns(exc["minute"])
            if (
                minute in exceptions
                or not start <= minute <= end
                or minute not in edges
                or type(exc.get("expected_trading")) is not bool
                or not exc.get("reason")
                or not exc.get("transition_refs")
            ):
                raise Rejected("Malformed boundary status exception")
            if exc["expected_trading"] != (minute in minutes):
                raise Rejected("Boundary exception contradicts calendar")
            exceptions[minute] = exc
        result.append(
            dict(
                name=item["name"],
                start=start,
                end=end,
                minutes=minutes,
                exceptions=exceptions,
                evidence=item,
            )
        )
    if not result:
        raise Rejected("Calendar contains no sessions")
    return result


def status_matches(
    row: Mapping[str, Any], trading: bool, exception: Mapping[str, Any] | None
) -> bool:
    expected = "TRADING" if trading else "NONTRADING"
    if row["interval_status"] == expected:
        return True
    if row["interval_status"] != "MIXED" or exception is None:
        return False
    # An exact evidence reference, not a guessed latency tolerance, admits a
    # boundary capture transition. All exchange events must be at the boundary.
    transitions = row["status_transitions"]
    keys = ("file", "record_index", "ts_recv_ns", "ts_event_ns")
    refs = [{k: transition[k] for k in keys} for transition in transitions]
    return bool(transitions) and (
        refs == exception["transition_refs"]
        and all(t["ts_event_ns"] == row["start_ns"] for t in transitions)
        and all(t["is_trading"] is trading for t in transitions)
        and row["status_at_end"]["is_trading"] is trading
    )


def classify_sessions(
    sessions: Sequence[dict[str, Any]],
    bars: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
    holds: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bar_map = {bar["start_ns"]: bar for bar in bars}
    cover_map = {row["start_ns"]: row for row in coverage}
    ledger: list[dict[str, Any]] = []
    assignments: dict[int, dict[str, Any]] = {}
    for session in sessions:
        reasons: dict[str, list[int]] = collections.defaultdict(list)
        evidence = session["evidence"]
        if evidence["verification"] != "VERIFIED":
            reasons["UNVERIFIED_CALENDAR"] = []
        for hold in holds:
            reasons[f"NATIVE_{hold}"] = []
        for minute in range(session["start"], session["end"], MINUTE):
            expected = minute in session["minutes"]
            row = cover_map.get(minute)
            bar = bar_map.get(minute)
            if row is None:
                reasons["OUTSIDE_ACQUIRED_RANGE"].append(minute)
            else:
                if not row["mbo_source_file_present"]:
                    reasons["MISSING_SOURCE_FILE"].append(minute)
                if not status_matches(row, expected, session["exceptions"].get(minute)):
                    reasons["CALENDAR_STATUS_CONFLICT"].append(minute)
            if expected and bar is None:
                reasons["MISSING_SCHEDULED_MINUTE"].append(minute)
            if not expected and bar is not None:
                reasons["TRADE_DURING_BREAK"].append(minute)
            if bar is not None:
                if bar["incomplete_event"]:
                    reasons["INCOMPLETE_TRADE_EVENT"].append(minute)
                assignments[minute] = dict(session=session["name"], scheduled=expected)
        closing = cover_map.get(session["end"])
        if closing is None:
            reasons["MISSING_CLOSING_EVIDENCE"].append(session["end"])
        elif not status_matches(
            closing, False, session["exceptions"].get(session["end"])
        ):
            reasons["CLOSING_STATUS_CONFLICT"].append(session["end"])
        eligible = not reasons
        ledger.append(
            dict(
                session=session["name"],
                start_ns=session["start"],
                end_ns=session["end"],
                eligible=eligible,
                expected_minutes=len(session["minutes"]),
                observed_minutes=sum(m in bar_map for m in session["minutes"]),
                exclusions=dict(sorted(reasons.items())),
                evidence=evidence,
            )
        )
    eligibility = {row["session"]: row["eligible"] for row in ledger}
    observed = []
    for bar in bars:
        assignment = assignments.get(bar["start_ns"])
        observed.append(
            dict(
                start_ns=bar["start_ns"],
                session=None if assignment is None else assignment["session"],
                classification=(
                    "OUTSIDE_CALENDAR"
                    if assignment is None
                    else (
                        "ELIGIBLE_SESSION"
                        if eligibility[assignment["session"]]
                        else "EXCLUDED_SESSION"
                    )
                ),
            )
        )
    return ledger, observed


def compare_profiles(
    sessions: Sequence[dict[str, Any]],
    ledger: Sequence[dict[str, Any]],
    bars: Sequence[dict[str, Any]],
    histograms: Mapping[int, Mapping[int, int]],
) -> list[dict[str, Any]]:
    eligible = {r["session"] for r in ledger if r["eligible"]}
    bar_map = {bar["start_ns"]: bar for bar in bars}
    snapshots: list[dict[str, Any]] = []
    for session in sessions:
        if session["name"] not in eligible:
            continue
        native: dict[int, int] = {}
        proxy = Profile()
        available = 0
        preceding = 0
        for minute in sorted(session["minutes"]):
            snapshot: dict[str, Any] = dict(
                session=session["name"],
                boundary_ns=minute,
                prior_bar_count=preceding,
                cumulative_volume=sum(native.values()),
                available_ns=available,
            )
            if preceding == 0:
                snapshot["status"] = "EMPTY_PREFIX"
            elif available > minute:
                snapshot["status"] = "UNAVAILABLE_PREFIX"
            else:
                area, proxy_area = native_area(native), proxy.area()
                if area is None or proxy_area is None:
                    snapshot["status"] = "ZERO_VOLUME_PREFIX"
                else:
                    proxy_ticks = tuple(round(price / 0.25) for price in proxy_area)
                    differences = [p - n for n, p in zip(area, proxy_ticks)]
                    snapshot.update(
                        status="COMPARED",
                        native_ticks=list(area),
                        proxy_ticks=list(proxy_ticks),
                        signed_proxy_minus_native_ticks=differences,
                        absolute_ticks=list(map(abs, differences)),
                    )
            snapshots.append(snapshot)
            bar = bar_map[minute]
            for tick, volume in histograms[minute].items():
                native[tick] = native.get(tick, 0) + volume
            prices = [price / 1_000_000_000 for price in bar["ohlcv"][:4]]
            proxy.add(
                Bar(
                    utc(minute),
                    prices[0],
                    prices[1],
                    prices[2],
                    prices[3],
                    bar["ohlcv"][4],
                )
            )
            available = max(available, bar["availability_ns"])
            preceding += 1
    return snapshots


def quantile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(snapshots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    compared = [row for row in snapshots if row["status"] == "COMPARED"]
    result: dict[str, Any] = dict(
        snapshot_status_counts=dict(
            sorted(collections.Counter(row["status"] for row in snapshots).items())
        ),
        comparison_count=len(compared),
        difference_convention="proxy minus native",
        quantile_method="linear interpolation at (N-1)*p",
        levels={},
    )
    for index, name in enumerate(("VAL", "VAH", "POC")):
        signed = [row["signed_proxy_minus_native_ticks"][index] for row in compared]
        absolute = list(map(abs, signed))
        result["levels"][name] = dict(
            denominator=len(signed),
            exact_agreement_count=signed.count(0),
            exact_agreement_fraction=signed.count(0) / len(signed) if signed else None,
            signed_ticks=distribution(signed),
            absolute_ticks=distribution(absolute),
        )
    result["all_levels_exact_count"] = sum(
        not any(row["absolute_ticks"]) for row in compared
    )
    return result


def distribution(values: Sequence[int]) -> dict[str, Any] | None:
    if not values:
        return None
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        quantiles={str(p): quantile(values, p) for p in (0.05, 0.25, 0.75, 0.95)},
    )


def reconcile(
    builder: Any,
    bars: Sequence[dict[str, Any]],
    delayed: Any,
    definitions: Any,
    statuses: Any,
    coverage: Any,
    saved: Mapping[str, Any],
) -> dict[str, Any]:
    actual = {
        "bars.jsonl": list(bars),
        "coverage.jsonl": coverage,
        "definitions.json": definitions,
        "status.json": statuses,
        "delayed-events.json": delayed,
        "exchange-diagnostic.jsonl": [
            builder.exchange[m] for m in sorted(builder.exchange)
        ],
    }
    for name, value in actual.items():
        if value != saved[name]:
            raise Rejected(f"Native reconciliation mismatch: {name}")
    report = saved["report.json"]
    if (
        dict(builder.counts) != report["counts"]
        or builder.files != report["files"]
        or sorted(builder.holds) != report["hold_reasons"]
    ):
        raise Rejected("Native counts/files/holds reconciliation mismatch")
    if set(builder.histograms) != {bar["start_ns"] for bar in bars}:
        raise Rejected("Histogram minute reconciliation mismatch")
    total = 0
    for bar in bars:
        histogram = builder.histograms[bar["start_ns"]]
        if (
            sum(histogram.values()) != bar["ohlcv"][4]
            or min(histogram) * TICK_NANOS != bar["ohlcv"][2]
            or max(histogram) * TICK_NANOS != bar["ohlcv"][1]
        ):
            raise Rejected("Histogram OHLC/volume conservation mismatch")
        total += sum(histogram.values())
    return dict(
        exact=True,
        bar_count=len(bars),
        trade_count=builder.counts["included_T_records"],
        integer_contract_volume=total,
        native_record_count=builder.counts["records"],
        reconciled_artifacts=sorted(actual),
        counts=dict(builder.counts),
        raw_trade_digests="exact per-bar native ordered bytes",
    )


def jsonl(rows: Sequence[Any]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
        for row in rows
    )


def output_path(raw: str | Path, inputs: Sequence[Path]) -> Path:
    path = safe_input(raw)
    if path.exists():
        raise Rejected("Output must be a new directory")
    if not path.parent.is_dir():
        raise Rejected("Output parent directory must already exist")
    if any(path == source or path.is_relative_to(source) for source in inputs):
        raise Rejected("Output collides with protected input")
    return path


def directory_fd(path: Path) -> int:
    """Resolve every directory component without following symlinks."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if part in {".", ".."}:
                raise Rejected("Unsafe output directory component")
            following = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def publish(path: Path, artifacts: Mapping[str, bytes]) -> None:
    """Publish relative to a held, no-symlink directory descriptor."""
    parent_fd = directory_fd(path.absolute().parent)
    stage = ".valentini-native-" + secrets.token_hex(16)
    stage_fd = None
    created = published = False
    files: list[str] = []
    try:
        os.mkdir(stage, mode=0o700, dir_fd=parent_fd)
        created = True
        stage_fd = os.open(
            stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        for name, content in artifacts.items():
            if Path(name).name != name or name in {".", ".."}:
                raise Rejected("Unsafe output artifact name")
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode=0o600,
                dir_fd=stage_fd,
            )
            files.append(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        if rename(parent_fd, os.fsencode(stage), parent_fd, os.fsencode(path.name), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(path))
        published = True
    finally:
        try:
            if stage_fd is not None:
                try:
                    if not published:
                        for name in files:
                            os.unlink(name, dir_fd=stage_fd)
                finally:
                    os.close(stage_fd)
            if created and not published:
                os.rmdir(stage, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)


def run(
    source_root: str | Path,
    reconstruction: str | Path,
    calendar_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_records: int = 65536,
) -> dict[str, Any]:
    if not 1 <= chunk_records <= 1_048_576:
        raise Rejected("Chunk records must be between 1 and 1048576")
    source, saved_root, calendar_file = map(
        safe_input, (source_root, reconstruction, calendar_path)
    )
    output = output_path(
        output_dir, (source, saved_root, calendar_file, ROOT / "src", ROOT / "tools")
    )
    code_paths = [
        Path(__file__),
        BUILDER_PATH,
        PINS_PATH,
        ROOT / "tools/valentini_reclaim.py",
    ]
    code_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths
    }
    calendar_hash = sha256_file(calendar_file)
    calendar = read_json(calendar_file)
    sessions = calendar_sessions(calendar)
    pins = read_json(PINS_PATH)
    before = verify_sources(source, pins)
    saved, saved_hashes = load_reconstruction(saved_root)
    if pins["files"] != saved["report.json"]["source_pins"]["files"]:
        raise Rejected("Acquisition pins differ from saved reconstruction")
    for citation in calendar["sources"]:
        if citation["url"].startswith("local:"):
            name = citation["url"][6:]
            if name not in saved_hashes or citation.get("sha256") != saved_hashes[name]:
                raise Rejected("Calendar local evidence hash mismatch")
    decoder = load_builder()
    builder = histogram_builder()
    definitions: list[Any] = []
    statuses: list[Any] = []
    for name in sorted(before):
        path = child(source, name)
        if name.endswith(".definition.dbn.zst"):
            definitions.extend(decoder.read_auxiliary(path, name, "definition"))
        elif name.endswith(".status.dbn.zst"):
            statuses.extend(decoder.read_auxiliary(path, name, "status"))
        elif name.endswith(".mbo.dbn.zst"):
            print(f"Decoding {name}", file=sys.stderr, flush=True)
            builder.read_mbo(path, name, chunk_records=chunk_records)
    bars, delayed = builder.finish()
    coverage = decoder.coverage(builder, statuses)
    conservation = reconcile(
        builder, bars, delayed, definitions, statuses, coverage, saved
    )
    ledger, observed = classify_sessions(
        sessions, bars, coverage, sorted(builder.holds)
    )
    snapshots = compare_profiles(sessions, ledger, bars, builder.histograms)
    summary = summarize(snapshots)
    report = dict(
        version=VERSION,
        kind="native_volume_profile_measurement",
        measurement_status=(
            "MEASURED_WITH_EXCLUSIONS"
            if summary["comparison_count"]
            else "NO_ELIGIBLE_COMPARISONS"
        ),
        market_evaluation="NOT_ADMITTED",
        conservation=conservation,
        eligible_sessions=sum(row["eligible"] for row in ledger),
        excluded_sessions=sum(not row["eligible"] for row in ledger),
        observed_bar_classifications=dict(
            collections.Counter(row["classification"] for row in observed)
        ),
        delayed_events=delayed,
        profile_summary=summary,
        session_profile_summaries={
            session["name"]: summarize(
                [row for row in snapshots if row["session"] == session["name"]]
            )
            for session in sessions
        },
        limitations=[
            "Reconciliation does not independently prove exchange feed completeness.",
            "Calendar uncertainty excludes sessions; no empty bars are fabricated.",
            "Closing boundaries are corroborated; entire inter-session nontrading "
            "intervals are not certified.",
            "Native capture clock and cumulative LAST availability govern eligibility.",
            "Unavailable snapshots remain excluded, even after later completion.",
            "Measurement does not unlock strategy tests; independent methodology "
            "and data admission remain required.",
        ],
    )
    if before != verify_sources(source, pins):
        raise Rejected("Source changed during audit")
    for name, digest in saved_hashes.items():
        if sha256_file(child(saved_root, name)) != digest:
            raise Rejected("Reconstruction changed during audit")
    if calendar_hash != sha256_file(calendar_file):
        raise Rejected("Calendar changed during audit")
    if any(sha256_file(ROOT / name) != digest for name, digest in code_hashes.items()):
        raise Rejected("Audit code changed during execution")
    provenance = dict(
        version=VERSION,
        code_sha256=code_hashes,
        calendar_sha256=calendar_hash,
        calendar=calendar,
        source_sha256=before,
        reconstruction_sha256=saved_hashes,
        runtime={
            "python": sys.version.split()[0],
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "databento_dbn", "zstandard")
            },
        },
        contract={
            "clock": "native ts_recv",
            "action": "T excluding SNAPSHOT and F",
            "price_unit": "integer ticks",
            "volume_unit": "integer contracts",
            "snapshot": "pre-bar boundary, preceding completed available bars only",
            "native_area": "70% exact integer volume; lower-price POC and adjacent ties",
            "proxy": "existing Profile uniform OHLCV allocation",
        },
    )
    artifacts = {
        "report.json": canonical_json(report).encode(),
        "sessions.json": canonical_json(ledger).encode(),
        "observed-bars.jsonl": jsonl(observed),
        "snapshots.jsonl": jsonl(snapshots),
        "histograms.jsonl": jsonl(
            [
                dict(start_ns=m, tick_volumes=sorted(h.items()))
                for m, h in sorted(builder.histograms.items())
            ]
        ),
        "provenance.json": canonical_json(provenance).encode(),
    }
    artifacts["manifest.json"] = canonical_json(
        {
            "version": VERSION,
            "artifacts": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in sorted(artifacts.items())
            },
        }
    ).encode()
    publish(output, artifacts)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "reconstruction", "calendar", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--chunk-records", type=int, default=65536)
    args = parser.parse_args(argv)
    try:
        report = run(
            args.source_root,
            args.reconstruction,
            args.calendar,
            args.output_dir,
            chunk_records=args.chunk_records,
        )
        print(
            canonical_json(
                {
                    "measurement_status": report["measurement_status"],
                    "comparison_count": report["profile_summary"]["comparison_count"],
                }
            ),
            end="",
        )
        return 3 if report["measurement_status"] == "NO_ELIGIBLE_COMPARISONS" else 0
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(
            canonical_json({"error": str(exc), "market_evaluation": "NOT_ADMITTED"}),
            file=sys.stderr,
            end="",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
