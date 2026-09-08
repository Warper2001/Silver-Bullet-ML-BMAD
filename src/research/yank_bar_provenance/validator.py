"""Verify fixed evidence, reproduce dollar bars, and retain conditional findings."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import (
    Decimal,
    Context,
    InvalidOperation,
    DivisionByZero,
    Overflow,
    ROUND_HALF_EVEN,
    localcontext,
)
from itertools import zip_longest
from pathlib import Path

ARITHMETIC = Context(
    prec=200,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)

RESEARCH = "_bmad-output/planning-artifacts/research/technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/"
RAW = "mnq_historical.json"
CSV = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
EXTRACT = RESEARCH + "imports/provenance-raw-2025.jsonl"
RECON = RESEARCH + "imports/frozen-audit/reconciliation.jsonl"
FIELDS = ("open", "high", "low", "close", "volume", "notional")
BASELINES = dict(
    raw_observations=351628,
    matched_rows=289230,
    multi_record=29520,
    mixed_contract=5583,
    pilot_labels=12697,
    pilot_multi_record=706,
    pilot_mixed_contract=0,
    capture_exact=10631,
    capture_different=2066,
    exchange_exact=10644,
    exchange_different=2053,
    capture_different_multi=692,
    capture_different_single=1374,
    common_differences=2040,
    resolved_by_exchange=26,
    introduced_by_exchange=13,
)
BLOCKERS = [
    "Raw origin is unauthenticated.",
    "Source interval semantics and arrival/availability are unknown.",
    "Mixed raw contracts prevent a suitability upgrade.",
    "Factor 20 and threshold 50000000 reproduce construction, not economic correctness.",
    "Raw MNQM25 and native MNQM5 / instrument 42009475 are separately attributed identifiers.",
]


class EvidenceError(ValueError):
    """Unsupported or divergent evidence; never a suitability success."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def write_json(path, value):
    path.write_text(encode(value), encoding="utf-8")


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EvidenceError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_float=Decimal,
        parse_constant=lambda x: (_ for _ in ()).throw(
            EvidenceError(f"non-finite JSON: {x}")
        ),
        object_pairs_hook=pairs,
    )


def number(value, context):
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise EvidenceError(f"{context}: expected exact numeric value")
    try:
        n = Decimal(value)
    except InvalidOperation as exc:
        raise EvidenceError(f"{context}: invalid number {value!r}") from exc
    if (
        not n.is_finite()
        or len(n.as_tuple().digits) > 40
        or abs(n.as_tuple().exponent) > 40
    ):
        raise EvidenceError(f"{context}: non-finite or unsupported precision: {value}")
    return n


def stamp(value, context, only_2025=True):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d{1,9})?(?:Z|[+-]\d\d:\d\d)", value
    ):
        raise EvidenceError(f"{context}: invalid timestamp {value!r}")
    # datetime has microsecond resolution: reject nonzero sub-microsecond evidence.
    fraction = re.search(r"\.(\d+)", value)
    if fraction and any(x != "0" for x in fraction[1][6:]):
        raise EvidenceError(f"{context}: unsupported sub-microsecond timestamp")
    try:
        t = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError as exc:
        raise EvidenceError(f"{context}: invalid timestamp {value!r}") from exc
    if only_2025 and t.year != 2025:
        raise EvidenceError(f"{context}: outside authorized 2025 scope")
    return t


def iso(t):
    return t.isoformat().replace("+00:00", "Z")


def ohlcv(values, context):
    if len(values) != 5:
        raise EvidenceError(f"{context}: expected OHLCV")
    o, h, l, c, v = [number(x, context) for x in values]
    if not l <= min(o, c) <= max(o, c) <= h:
        raise EvidenceError(f"{context}: incoherent OHLC bounds")
    if v < 0 or v != v.to_integral_value():
        raise EvidenceError(f"{context}: volume must be nonnegative integral")
    return o, h, l, c, v


def jsonlines(path):
    with path.open(encoding="utf-8") as f:
        for line, text in enumerate(f, 1):
            try:
                value = strict_json(text)
                if not isinstance(value, dict):
                    raise EvidenceError("expected object")
                yield line, value
            except (ValueError, TypeError) as exc:
                raise EvidenceError(f"{path.name}:{line}: {exc}") from exc


def raw_2025(path):
    """Bounded flat-object scanner; only timestamp strings of other years are read."""
    buf = []
    start = ordinal = 0
    with path.open(encoding="utf-8") as f:
        for line, text in enumerate(f, 1):
            s = text.strip()
            if s == "{":
                if buf:
                    raise EvidenceError(f"raw:{line}: unsupported nested object")
                buf = [text]
                start = line
            elif buf:
                buf.append(text)
                if sum(map(len, buf)) > 65536:
                    raise EvidenceError(f"raw:{start}: object exceeds supported size")
                if s in ("}", "},"):
                    ordinal += 1
                    content = "".join(buf)
                    labels = re.findall(r'"TimeStamp"\s*:\s*"([^"\n]+)"', content)
                    if len(labels) != 1:
                        raise EvidenceError(f"raw:{start}: expected one timestamp")
                    t = stamp(labels[0], f"raw:{start}", only_2025=False)
                    if t.year == 2025:
                        yield ordinal, {
                            "source_line": start,
                            "bar": strict_json(content.rstrip().rstrip(",")),
                        }
                    buf = []
            elif s not in ("[", "]", ""):
                raise EvidenceError(f"raw:{line}: unsupported outer structure")
    if buf:
        raise EvidenceError(f"raw:{start}: unterminated object")


def verified_sources(root):
    records = []
    seen = set()
    for raw, extracted in zip_longest(raw_2025(root / RAW), jsonlines(root / EXTRACT)):
        if raw is None or extracted is None:
            raise EvidenceError("raw/extract missing or extra source record")
        ordinal, obj = raw
        extract_line, other = extracted
        if obj != other:
            raise EvidenceError(
                f"raw/extract mismatch at extract line {extract_line}, raw ordinal {ordinal}"
            )
        b = obj["bar"]
        try:
            t = stamp(b["TimeStamp"], f"extract:{extract_line}")
            vals = ohlcv(
                [b[k] for k in ("Open", "High", "Low", "Close", "TotalVolume")],
                f"extract:{extract_line}",
            )
            contract = b["Contract"]
            if not isinstance(contract, str) or not contract.strip():
                raise EvidenceError("empty or invalid contract")
            if type(obj["source_line"]) is not int or obj["source_line"] < 1:
                raise EvidenceError("invalid source line")
        except (KeyError, TypeError) as exc:
            raise EvidenceError(
                f"extract:{extract_line}: missing/invalid required field {exc}"
            ) from exc
        if t in seen:
            raise EvidenceError(
                f"extract:{extract_line}: duplicate source timestamp {iso(t)}"
            )
        seen.add(t)
        records.append(
            dict(
                time=t,
                values=vals,
                contract=contract,
                source_line=obj["source_line"],
                source_ordinal=ordinal,
                extract_line=extract_line,
            )
        )
    records.sort(key=lambda x: (x["time"], x["source_ordinal"]))
    return records


def reconstruct(records):
    constituents = []
    total = Decimal(0)
    for r in records:
        constituents.append(r)
        with localcontext(ARITHMETIC):
            total += r["values"][3] * r["values"][4] * 20
            if total < 50000000:
                continue
            values = (
                constituents[0]["values"][0],
                max(x["values"][1] for x in constituents),
                min(x["values"][2] for x in constituents),
                r["values"][3],
                sum(x["values"][4] for x in constituents),
                total,
            )
        emitted = constituents
        constituents = []
        total = Decimal(0)
        yield emitted, values
    if constituents:
        raise EvidenceError(
            f'leftover source records: {len(constituents)}; first raw line {constituents[0]["source_line"]}'
        )


def read_pilots(root):
    # Evidence validation must not depend on a caller's ambient Decimal context.
    with localcontext(ARITHMETIC):
        return _read_pilots(root)


def _read_pilots(root):
    pilots = {}
    for line, r in jsonlines(root / RECON):
        try:
            if r["clock"] not in ("capture", "exchange", "exchange_diagnostic") or r[
                "convention"
            ] not in ("start", "end"):
                raise EvidenceError("unsupported reconciliation clock/convention")
            label = stamp(r["original_label"], f"reconciliation:{line}")
            native_start = stamp(r["native_minute_start"], f"reconciliation:{line}")
            expected_start = label - timedelta(minutes=r["convention"] == "end")
            if native_start != expected_start:
                raise EvidenceError("reconciliation interval disagrees with convention")
            original = ohlcv(
                [*r["original_ohlc"], r["original_volume"]], f"reconciliation:{line}"
            )
            if r["convention"] == "start" and r["native_ohlc"] is None:
                if r["coverage_qualification"] != "missing_native_T" or any(
                    r[k] is not None
                    for k in ("native_T_volume", "ohlc_delta", "volume_delta")
                ):
                    raise EvidenceError("incoherent missing native evidence")
                continue
            native = ohlcv(
                [*r["native_ohlc"], r["native_T_volume"]], f"reconciliation:{line}"
            )
            delta = [number(x, f"reconciliation:{line}") for x in r["ohlc_delta"]]
            if (
                delta != [n - o for n, o in zip(native[:4], original[:4])]
                or number(r["volume_delta"], "volume_delta") != native[4] - original[4]
            ):
                raise EvidenceError("reconciliation field deltas disagree")
            if r["convention"] != "end":
                continue
            clock = "exchange" if r["clock"] == "exchange_diagnostic" else r["clock"]
            item = pilots.setdefault(label, {})
            if clock in item:
                raise EvidenceError("duplicate reconciliation clock/label")
            item[clock] = dict(
                source_clock=r["clock"],
                line=line,
                original=original,
                native=native,
                delta=delta,
                coverage=r["coverage_qualification"],
                gap_reasons=r["gap_reasons"],
            )
        except (KeyError, TypeError) as exc:
            raise EvidenceError(
                f"reconciliation:{line}: missing/invalid field {exc}"
            ) from exc
    if not pilots or any(set(v) != {"capture", "exchange"} for v in pilots.values()):
        raise EvidenceError("missing paired end-label pilot evidence")
    return pilots


def lineage_row(index, sources, values, differences, pins):
    first, last = sources[0]["time"], sources[-1]["time"]
    contracts = sorted({x["contract"] for x in sources})
    gaps = [
        {
            "after": iso(a["time"]),
            "before": iso(b["time"]),
            "seconds": (b["time"] - a["time"]).total_seconds(),
        }
        for a, b in zip(sources, sources[1:])
        if b["time"] - a["time"] > timedelta(seconds=60)
    ]
    return dict(
        schema_version=1,
        csv_ordinal=index,
        csv_line=index + 1,
        csv_path=CSV,
        csv_sha256=pins[CSV],
        label=iso(last),
        first_source_label=iso(first),
        last_source_label=iso(last),
        constituent_count=len(sources),
        aggregation="multiple" if len(sources) > 1 else "single",
        contracts=contracts,
        contract_class="mixed" if len(contracts) > 1 else "single",
        constituents=[
            dict(
                raw_path=RAW,
                raw_sha256=pins[RAW],
                raw_line=x["source_line"],
                raw_ordinal=x["source_ordinal"],
                extract_path=EXTRACT,
                extract_sha256=pins[EXTRACT],
                extract_line=x["extract_line"],
                label=iso(x["time"]),
                contract=x["contract"],
            )
            for x in sources
        ],
        reconstructed=dict(zip(FIELDS, map(str, values))),
        differences=differences,
        timing=dict(
            label_meaning="inherited_last_source_label",
            actual_start=None,
            availability=None,
            arrival=None,
            unavailable_reason="No authenticated interval or arrival evidence",
            conditional_end=[iso(first - timedelta(seconds=60)), iso(last)],
            conditional_start=[iso(first), iso(last + timedelta(seconds=60))],
            bounds="half-open hypotheses; continuity is not proven",
            gaps=gaps,
        ),
    )


def compare(root, output, records, pilots, pins):
    counts = Counter({k: 0 for k in BASELINES})
    counts["raw_observations"] = len(records)
    total_differences = 0
    first_divergence = None
    joined = set()
    with (
        (root / CSV).open(newline="", encoding="utf-8") as f,
        (output / "lineage.jsonl").open("w") as lineage,
        (output / "pilot-differences.jsonl").open("w") as pilot_out,
    ):
        reader = csv.DictReader(f)
        if reader.fieldnames != ["timestamp", *FIELDS]:
            raise EvidenceError(f"CSV: unexpected schema {reader.fieldnames}")
        for index, (generated, original) in enumerate(
            zip_longest(reconstruct(records), reader), 1
        ):
            differences = {}
            if generated is None or original is None:
                total_differences += 1
                first_divergence = first_divergence or {
                    "csv_ordinal": index,
                    "reason": "missing/extra CSV row",
                }
                continue
            if None in original or any(v is None for v in original.values()):
                raise EvidenceError(f"CSV:{index+1}: malformed row")
            sources, values = generated
            t = stamp(original["timestamp"], f"CSV:{index+1}")
            vals = (
                *ohlcv([original[k] for k in FIELDS[:5]], f"CSV:{index+1}"),
                number(original["notional"], f"CSV:{index+1}:notional"),
            )
            if vals[-1] < 0:
                raise EvidenceError(f"CSV:{index+1}: negative notional")
            for field, actual, wanted in zip(FIELDS, vals, values):
                if actual != wanted:
                    differences[field] = {
                        "csv": str(actual),
                        "reconstructed": str(wanted),
                    }
            if t != sources[-1]["time"]:
                differences["timestamp"] = {
                    "csv": iso(t),
                    "reconstructed": iso(sources[-1]["time"]),
                }
            if differences:
                total_differences += len(differences)
                first_divergence = first_divergence or {
                    "csv_ordinal": index,
                    "differences": differences,
                }
            else:
                counts["matched_rows"] += 1
            row = lineage_row(index, sources, values, differences, pins)
            lineage.write(encode(row))
            multi, mixed = len(sources) > 1, len(row["contracts"]) > 1
            counts["multi_record"] += multi
            counts["mixed_contract"] += mixed
            if t not in pilots:
                continue
            if t in joined:
                raise EvidenceError(f"duplicate pilot CSV label {iso(t)}")
            joined.add(t)
            clocks = pilots[t]
            counts["pilot_labels"] += 1
            counts["pilot_multi_record"] += multi
            counts["pilot_mixed_contract"] += mixed
            diff = {}
            clock_output = {}
            for clock, r in sorted(clocks.items()):
                if r["original"] != vals[:5]:
                    raise EvidenceError(
                        f"pilot original differs from CSV:{index+1}, {clock}"
                    )
                diff[clock] = any(r["delta"])
                counts[clock + ("_different" if diff[clock] else "_exact")] += 1
                clock_output[clock] = dict(
                    source_clock=r["source_clock"],
                    reconciliation_path=RECON,
                    reconciliation_sha256=pins[RECON],
                    reconciliation_line=r["line"],
                    original_ohlcv=list(map(str, r["original"])),
                    native_ohlcv=list(map(str, r["native"])),
                    ohlc_delta=list(map(str, r["delta"])),
                    volume_delta=str(r["native"][4] - r["original"][4]),
                    ohlc_class="different" if diff[clock] else "exact",
                    coverage_qualification=r["coverage"],
                    gap_reasons=r["gap_reasons"],
                )
            counts[
                "capture_different_multi" if multi else "capture_different_single"
            ] += diff["capture"]
            counts["common_differences"] += diff["capture"] and diff["exchange"]
            counts["resolved_by_exchange"] += diff["capture"] and not diff["exchange"]
            counts["introduced_by_exchange"] += diff["exchange"] and not diff["capture"]
            pilot_out.write(
                encode(
                    dict(
                        schema_version=1,
                        csv_ordinal=index,
                        csv_line=index + 1,
                        label=iso(t),
                        constituent_count=len(sources),
                        contracts=row["contracts"],
                        aggregation=row["aggregation"],
                        contract_class=row["contract_class"],
                        fixed_end_interval=[iso(t - timedelta(seconds=60)), iso(t)],
                        clocks=clock_output,
                        attribution="Aggregation overlap is not causal resolution; prior native comparisons, no new DBN reconstruction.",
                        native_identifiers={
                            "symbol": "MNQM5",
                            "instrument_id": 42009475,
                        },
                    )
                )
            )
    if total_differences:
        raise EvidenceError(
            f"reconstruction: total differences={total_differences}; first divergence={encode(first_divergence).strip()}"
        )
    if joined != set(pilots):
        raise EvidenceError(f"pilot labels absent from CSV: {len(set(pilots)-joined)}")
    return dict(counts)


def repository_root():
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / RAW).is_file():
        return checkout
    gitfile = checkout / ".git"
    if gitfile.is_file():
        gitdir = Path(gitfile.read_text().strip().removeprefix("gitdir: "))
        if not gitdir.is_absolute():
            gitdir = checkout / gitdir
        common = (gitdir / (gitdir / "commondir").read_text().strip()).resolve()
        return common.parent
    return checkout


def reserve_output(output, roots, pins):
    output = output.absolute()
    resolved = output.resolve()
    for root in roots:
        # A new descendant of an evidence directory or an ancestor of an input is unsafe.
        for name in pins:
            source = (root / name).resolve()
            parent = source.parent
            # Root-level raw file protects the file, not every repository descendant.
            if (
                resolved == source
                or resolved in source.parents
                or (
                    parent != root.resolve()
                    and (resolved == parent or parent in resolved.parents)
                )
            ):
                raise EvidenceError(f"output overlaps evidence: {name}")
    if output.exists() or output.is_symlink():
        raise EvidenceError(f"output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run(output, *, root=None, pins=None, baselines=None):
    root = Path(root) if root is not None else repository_root()
    pins = (
        pins
        if pins is not None
        else json.loads(Path(__file__).with_name("pins.json").read_text())
    )
    baselines = BASELINES if baselines is None else baselines
    checkout = Path(__file__).resolve().parents[3]
    output = reserve_output(Path(output), [root, checkout], pins)
    try:
        verified = {}
        for name, expected in sorted(pins.items()):
            if not (root / name).is_file():
                raise EvidenceError(f"missing pinned input: {name}")
            observed = digest(root / name)
            if observed != expected:
                raise EvidenceError(
                    f"hash mismatch: {name}; expected {expected}; observed {observed}"
                )
            verified[name] = {"expected_sha256": expected, "observed_sha256": observed}
        records = verified_sources(root)
        pilots = read_pilots(root)
        with localcontext(ARITHMETIC):
            counts = compare(root, output, records, pilots, pins)
        for key, expected in baselines.items():
            if counts.get(key) != expected:
                raise EvidenceError(
                    f"baseline {key}: expected {expected}, observed {counts.get(key)}"
                )
        # Detect evidence modification during analysis before publishing a success.
        for name, expected in pins.items():
            if digest(root / name) != expected:
                raise EvidenceError(f"input changed during validation: {name}")
        report = dict(
            schema_version=1,
            status="PASS_PROVENANCE_CHECKS",
            data_suitability="BLOCKED",
            research_status="HOLD_VALIDATION",
            counts=counts,
            expected_baselines=baselines,
            unavailable_evidence_reasons=BLOCKERS,
            assumptions=dict(
                factor=20,
                threshold=50000000,
                emit=">=",
                whole_records=True,
                label="last constituent",
                sorting="UTC instant then original ordinal",
                scope=2025,
            ),
            verification=dict(
                hashes="PASS",
                raw_to_extract="PASS",
                all_csv_rows="PASS",
                classifications="PASS",
                input_preservation="PASS",
                deterministic_serialization="sorted keys, exact numeric strings, stable relative paths; compare fresh runs",
            ),
            qualifications=[
                "Prior hash-verified native reconciliation imported from audit commit 84f2382; no DBN reprocessing.",
                "No authenticated alias or roll policy is inferred.",
                "174-contract boundary example demonstrates numerical sufficiency only; no allocation adjustment.",
                "Frozen PASS_AUDIT_CHECKS, five cases, eight arm orders, 30 scenarios, 11 supported/19 unassessable outcomes remain unchanged.",
                "Original signal completion, 0/100/500 ms delays and 240 scheduled-opportunity pending lifetimes remain unchanged.",
                "May 28 case 4 start-label same-event ambiguity and conditional end-label entry-before-stop evidence remain intact.",
                "Pauses do not classify all 11819 locked/crossed events as benign; lock intervals and Sunday reset remain unresolved.",
                "No replacement bars, signals, fills, economic adjustments, commissions or strategy eligibility are produced.",
            ],
        )
        write_json(output / "report.json", report)
        markdown = "# YANK frozen bar provenance\n\nPASS_PROVENANCE_CHECKS\n\nData suitability: **BLOCKED**. Research status: **HOLD_VALIDATION**.\n\n"
        markdown += "| Check | Observed |\n|---|---:|\n" + "".join(
            f"| {k} | {v} |\n" for k, v in sorted(counts.items())
        )
        markdown += "\n" + "\n\n".join(BLOCKERS + report["qualifications"]) + "\n"
        (output / "report.md").write_text(markdown, encoding="utf-8")
        code_files = [
            Path(__file__),
            Path(__file__).with_name("__init__.py"),
            Path(__file__).with_name("pins.json"),
            checkout / "src/cli/check_yank_bar_provenance.py",
        ]
        manifest = dict(
            schema_version=1,
            inputs=verified,
            code_sha256={str(p.relative_to(checkout)): digest(p) for p in code_files},
            outputs_sha256={
                n: digest(output / n)
                for n in [
                    "report.md",
                    "report.json",
                    "lineage.jsonl",
                    "pilot-differences.jsonl",
                ]
            },
            manifest_hash_policy="manifest.json excluded from its own hashes; compare all five files across fresh runs",
        )
        write_json(output / "manifest.json", manifest)
        return report
    except (Exception, KeyboardInterrupt) as exc:
        for name in ("report.json", "report.md", "manifest.json"):
            (output / name).unlink(missing_ok=True)
        write_json(
            output / "report.json",
            dict(
                schema_version=1,
                status="FAIL_PROVENANCE_CHECKS",
                data_suitability="BLOCKED",
                research_status="HOLD_VALIDATION",
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        (output / "report.md").write_text(
            f"# YANK provenance failed\n\nFAIL_PROVENANCE_CHECKS\n\n{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
