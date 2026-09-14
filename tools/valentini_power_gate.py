"""Preregistered metadata-only conditional power; never admits strategy testing."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import importlib.metadata
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy import optimize, stats  # type: ignore[import-untyped]  # noqa: E402

from tools.valentini_native_audit import (  # noqa: E402
    child,
    minute_ns,
    output_path,
    publish,
    safe_input,
    strict_value,
)
from tools.valentini_reclaim import (  # noqa: E402
    Rejected,
    canonical_json,
    sha256_file,
)

PREREG_COMMIT = "42d905e2914a0d901e933dfa86651c223e0594f3"
PREREG_SHA256 = "8defef100fe89382f331858e5d02e5cc495d8280189a2fe37fabca2acac09005"
MANIFEST_SHA256 = "23f2205518d7be751fa147274441efe0af89c2a2aec52f76a8810e443a4b1667"
INVENTORY_SHA256 = "25c368725646fa9018a526a4a1f1782ce57f55e36381f0478656dfc783dc82a5"
MEASUREMENT_VERSION = "valentini-native-measurement-v1"
ARTIFACTS = frozenset(
    {
        "report.json",
        "sessions.json",
        "provenance.json",
        "histograms.jsonl",
        "observed-bars.jsonl",
        "snapshots.jsonl",
    }
)
PROVENANCE_CODE = frozenset(
    {
        "src/research/yank_native_minute/builder.py",
        "src/research/yank_native_minute/pins.json",
        "tools/valentini_native_audit.py",
        "tools/valentini_reclaim.py",
    }
)
ALPHA = 0.05
TARGET = 0.80
MAX_N = 1_000_000
EFFECTS = (0.10, 0.20, 0.30, 0.50, 1.00)


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise Rejected(f"{name} must be an integer >= {minimum}")
    return value


def effect(value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Rejected("Effect must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise Rejected("Effect must be finite and nonnegative (positive for n search)")
    return result


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise Rejected("Nonfinite conditional-model calculation")
    return result


def conditional_power(n: int, d: float) -> float | None:
    """One-sided unknown-variance t power under hypothetical iid normal sessions."""
    integer(n, "n")
    delta = effect(d)
    if n < 2:
        return None
    critical = finite(stats.t.ppf(1 - ALPHA, n - 1))
    noncentrality = finite(delta * math.sqrt(n))
    power = finite(stats.nct.sf(critical, n - 1, noncentrality))
    if not 0 <= power <= 1:
        raise Rejected("Invalid conditional probability")
    return power


def assessable_power(n: int, d: float) -> float:
    result = conditional_power(n, d)
    if result is None:
        raise Rejected("Unknown-variance model requires at least two sessions")
    return result


def detectable_effect(n: int) -> dict[str, Any]:
    integer(n, "n")
    if n < 2:
        return dict(n=n, d=None, status="UNASSESSABLE_N_LT_2")
    upper = 1.0
    for _ in range(64):
        if assessable_power(n, upper) >= TARGET:
            break
        upper *= 2
    else:
        return dict(n=n, d=None, status="UNRESOLVED_BRACKET")
    try:
        root = finite(
            optimize.brentq(
                lambda d: assessable_power(n, d) - TARGET,
                0.0,
                upper,
                xtol=1e-12,
                rtol=1e-12,
                maxiter=200,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise Rejected("Detectable-effect root failed") from exc
    achieved = assessable_power(n, root)
    if root < 0 or abs(achieved - TARGET) > 1e-9:
        raise Rejected("Detectable-effect target inversion failed")
    return dict(n=n, d=root, conditional_power=achieved, status="CONDITIONAL_MODEL")


def minimum_sessions(d: float, cap: int = MAX_N) -> dict[str, Any]:
    delta = effect(d, positive=True)
    integer(cap, "cap", 2)
    if cap > MAX_N:
        raise Rejected("Search cap exceeds preregistered maximum")
    low, high = 1, 2
    while assessable_power(high, delta) < TARGET:
        if high == cap:
            return dict(
                n=None,
                status="UNRESOLVED_ABOVE_CAP",
                cap=cap,
                power_at_cap=assessable_power(cap, delta),
            )
        low, high = high, min(cap, high * 2)
    while high - low > 1:
        middle = (low + high) // 2
        if assessable_power(middle, delta) >= TARGET:
            high = middle
        else:
            low = middle
    reached = assessable_power(high, delta)
    previous = conditional_power(high - 1, delta)
    if reached < TARGET or (previous is not None and previous >= TARGET):
        raise Rejected("Minimum-n adjacent boundary failed")
    return dict(
        n=high,
        status="CONDITIONAL_MODEL",
        power_at_n=reached,
        power_at_previous_n=previous,
        previous_n_status=("UNASSESSABLE_N_LT_2" if high == 2 else "BELOW_TARGET"),
        cap=cap,
    )


def normal_crosscheck(n: int, d: float) -> dict[str, Any]:
    """Separate known-population-variance formula, not the finite-sample t test."""
    integer(n, "n")
    delta = effect(d, positive=True)
    z_alpha = finite(stats.norm.ppf(1 - ALPHA))
    z_target = finite(stats.norm.ppf(TARGET))
    required = finite(((z_alpha + z_target) / delta) ** 2)
    return dict(
        model="known_variance_normal",
        power_at_n=(
            finite(stats.norm.sf(z_alpha - delta * math.sqrt(n))) if n else None
        ),
        required_n_continuous=required,
        required_n_integer=max(1, math.ceil(required)),
        mde_at_n=finite((z_alpha + z_target) / math.sqrt(n)) if n else None,
    )


def model_report(n: int) -> dict[str, Any]:
    integer(n, "n")
    if n > MAX_N:
        raise Rejected("Observed count exceeds model computational limit")
    return dict(
        status="CONDITIONAL_MODEL" if n >= 2 else "UNASSESSABLE_N_LT_2",
        alpha=ALPHA,
        target_power=TARGET,
        sample_search_cap=MAX_N,
        observed_session_count=n,
        observed_count_mde=detectable_effect(n),
        mde_frontier=[detectable_effect(count) for count in range(2, n + 1)],
        hypothetical_scenarios=[
            dict(
                d=d,
                hypothetical=True,
                power_at_observed_count=conditional_power(n, d),
                minimum_independent_sessions=minimum_sessions(d),
                known_variance_crosscheck=normal_crosscheck(n, d),
            )
            for d in EFFECTS
        ],
    )


def bound_bytes(path: Path, expected: str) -> bytes:
    source = safe_input(path)
    if not source.is_file():
        raise Rejected(f"Input must be a regular file: {path.name}")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise Rejected(f"Input hash mismatch: {path.name}")
    return data


def object_value(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise Rejected(f"Expected JSON object: {label}")
    return value


def bound_json(path: Path, expected: str) -> Any:
    return strict_value(bound_bytes(path, expected).decode("utf-8"))


def verify_artifacts(root: Path) -> dict[str, str]:
    manifest = object_value(
        bound_json(child(root, "manifest.json"), MANIFEST_SHA256), "manifest"
    )
    pins = object_value(manifest["artifacts"], "artifact hashes")
    if manifest.get("version") != MEASUREMENT_VERSION or set(pins) != ARTIFACTS:
        raise Rejected("Unexpected measurement manifest schema")
    for name, expected in pins.items():
        if not isinstance(expected, str) or sha256_file(child(root, name)) != expected:
            raise Rejected(f"Artifact hash mismatch: {name}")
    return {"manifest.json": MANIFEST_SHA256, **pins}


def validate_metadata(
    report: Mapping[str, Any], ledger: Any, provenance: Mapping[str, Any]
) -> int:
    conservation = object_value(report.get("conservation"), "conservation")
    if (
        report.get("version") != MEASUREMENT_VERSION
        or report.get("kind") != "native_volume_profile_measurement"
        or report.get("market_evaluation") != "NOT_ADMITTED"
        or provenance.get("version") != MEASUREMENT_VERSION
        or conservation.get("exact") is not True
    ):
        raise Rejected("Unadmitted or inconsistent measurement metadata")
    if not isinstance(ledger, list):
        raise Rejected("Session ledger must be an array")
    summaries = object_value(report["session_profile_summaries"], "session summaries")
    calendar = object_value(provenance["calendar"], "calendar")
    calendar_rows = calendar["sessions"]
    if not isinstance(calendar_rows, list):
        raise Rejected("Calendar sessions must be an array")
    evidence_by_name: dict[str, Any] = {}
    for evidence in calendar_rows:
        evidence = object_value(evidence, "calendar session")
        name = evidence["name"]
        if not isinstance(name, str) or not name or name in evidence_by_name:
            raise Rejected("Duplicate or invalid calendar session ID")
        evidence_by_name[name] = evidence
    names: set[str] = set()
    intervals: list[tuple[int, int]] = []
    eligible = comparisons = 0
    statuses: collections.Counter[str] = collections.Counter()
    for item in ledger:
        row = object_value(item, "session")
        name = row["session"]
        if not isinstance(name, str) or not name or name in names:
            raise Rejected("Duplicate or invalid session ID")
        names.add(name)
        start = integer(row["start_ns"], "start_ns")
        end = integer(row["end_ns"], "end_ns")
        if end <= start:
            raise Rejected("Invalid session interval")
        intervals.append((start, end))
        if type(row["eligible"]) is not bool:
            raise Rejected("Eligibility must be a JSON boolean")
        exclusions = object_value(row["exclusions"], "exclusions")
        if row["eligible"] != (not exclusions):
            raise Rejected("Eligibility contradicts original exclusions")
        evidence = object_value(row["evidence"], "session evidence")
        if (
            evidence != evidence_by_name.get(name)
            or minute_ns(evidence["start"]) != start
            or minute_ns(evidence["end"]) != end
        ):
            raise Rejected("Ledger differs from calendar evidence")
        expected = integer(row["expected_minutes"], "expected_minutes")
        observed = integer(row["observed_minutes"], "observed_minutes")
        if observed > expected or (
            row["eligible"]
            and (observed != expected or evidence["verification"] != "VERIFIED")
        ):
            raise Rejected("Eligible session coverage is inconsistent")
        summary = object_value(summaries[name], "per-session summary")
        count = integer(summary["comparison_count"], "comparison_count")
        counts = object_value(summary["snapshot_status_counts"], "snapshot counts")
        for key, value in counts.items():
            integer(value, "snapshot count")
        if count != counts.get("COMPARED", 0):
            raise Rejected("Comparison and status count mismatch")
        if sum(counts.values()) != (expected if row["eligible"] else 0):
            raise Rejected("Session summary differs from eligible ledger")
        comparisons += count
        statuses.update(counts)
        eligible += int(row["eligible"])
    if names != set(summaries) or names != set(evidence_by_name):
        raise Rejected("Session IDs differ between metadata artifacts")
    ordered = sorted(intervals)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise Rejected("Overlapping sessions")
    if (
        integer(report["eligible_sessions"], "eligible_sessions") != eligible
        or integer(report["excluded_sessions"], "excluded_sessions")
        != len(ledger) - eligible
    ):
        raise Rejected("Session counts differ from report")
    overall = object_value(report["profile_summary"], "profile summary")
    overall_counts = object_value(overall["snapshot_status_counts"], "overall counts")
    for value in overall_counts.values():
        integer(value, "overall snapshot count")
    if (
        integer(overall["comparison_count"], "comparison_count") != comparisons
        or dict(statuses) != overall_counts
    ):
        raise Rejected("Aggregate profile summary differs from sessions")
    code = object_value(provenance["code_sha256"], "provenance code")
    if set(code) != PROVENANCE_CODE:
        raise Rejected("Unexpected provenance code set")
    for name, digest in code.items():
        if sha256_file(child(ROOT, name)) != digest:
            raise Rejected(f"Provenance code changed: {name}")
    return eligible


def runtime_evidence() -> dict[str, Any]:
    modules = (
        "numpy",
        "scipy",
        "scipy.stats._continuous_distns",
        "scipy.optimize._zeros",
        "scipy.special._ufuncs",
    )
    hashes: dict[str, str] = {}
    for name in modules:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise Rejected(f"Missing dependency module file: {name}")
        hashes[name] = sha256_file(module.__file__)
    return dict(
        python=sys.version,
        executable_sha256=sha256_file(sys.executable),
        package_versions={
            name: importlib.metadata.version(name) for name in ("scipy", "numpy")
        },
        dependency_module_sha256=hashes,
        dependency_hash_scope="Selected loaded module files; not entire environments",
    )


def run(
    audit_dir: str | Path,
    prereg: str | Path,
    inventory: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    audit, registration, inventory_path = map(
        safe_input, (audit_dir, prereg, inventory)
    )
    output = output_path(
        output_dir,
        (audit, registration, inventory_path, ROOT / "src", ROOT / "tools"),
    )
    bound_bytes(registration, PREREG_SHA256)
    inventory_data = object_value(
        bound_json(inventory_path, INVENTORY_SHA256), "inventory"
    )
    if inventory_data.get("independent_full_session_calibration_found") is not False:
        raise Rejected("Pinned inventory must retain absent independent calibration")
    before = verify_artifacts(audit)
    metadata = {
        name: bound_json(child(audit, name), before[name])
        for name in ("report.json", "sessions.json", "provenance.json")
    }
    n = validate_metadata(
        object_value(metadata["report.json"], "report"),
        metadata["sessions.json"],
        object_value(metadata["provenance.json"], "provenance"),
    )
    code = {
        name: sha256_file(child(ROOT, name))
        for name in (*sorted(PROVENANCE_CODE), "tools/valentini_power_gate.py")
    }
    if {name: code[name] for name in PROVENANCE_CODE} != metadata["provenance.json"][
        "code_sha256"
    ]:
        raise Rejected("Code snapshot differs from pinned provenance")
    runtime = runtime_evidence()
    result = dict(
        version="valentini-conditional-power-v1",
        verdict="POWER_UNDETERMINED",
        evaluation_allowed=False,
        market_evaluation="NOT_ADMITTED",
        conditional_model=model_report(n),
        assumptions=[
            "Hypothetical iid normal session net outcomes; "
            "unknown population variance.",
            "d is hypothetical mean net session dollars / "
            "population standard deviation.",
            "No mean, variance, cost or transferable effect is estimated here.",
            "One contract; include complete zero-trade sessions in the proposed test.",
            "Observed profiles are neither independent trades "
            "nor calibrated effective n.",
            "Dependence, nonnormality and selection remain uncalibrated; "
            "no bound claimed.",
            "Scenario sample sizes are not guaranteed horizons "
            "or purchase authorization.",
        ],
        blockers=[
            "No transferable net effect for this exact construct "
            "in inspected evidence.",
            "No independent full-session calibration sample in the scoped inventory.",
            "No calibrated dependence model or validated net-cost mapping.",
        ],
        next_required_evidence=[
            "Representative full native sessions with contract/calendar/feed evidence.",
            "Separate calibration and validation roles fixed before outcomes.",
            "Defensible net effect and cost assumptions and a dependence-aware test.",
            "Subsequent preregistration before any strategy outcome test.",
        ],
        preserved_session_ledger=metadata["sessions.json"],
        inventory_evidence=inventory_data,
        input_sha256=dict(
            prereg=PREREG_SHA256,
            inventory=INVENTORY_SHA256,
            measurement=before,
        ),
        prereg_commit=PREREG_COMMIT,
        code_sha256=code,
        runtime=runtime,
        data_boundary=dict(
            parsed_measurement_artifacts=[
                "report.json",
                "sessions.json",
                "provenance.json",
            ],
            all_six_artifacts_hashed=True,
            raw_native_decode=False,
            raw_native_reconciliation=False,
            raw_source_hashes="Historical audit provenance only; not re-read here",
            strategy_outcomes_computed=False,
            historical_provenance=metadata["provenance.json"],
        ),
    )
    if verify_artifacts(audit) != before:
        raise Rejected("Measurement changed during gate")
    bound_bytes(registration, PREREG_SHA256)
    bound_bytes(inventory_path, INVENTORY_SHA256)
    if any(sha256_file(child(ROOT, name)) != digest for name, digest in code.items()):
        raise Rejected("Code changed during gate")
    if runtime_evidence() != runtime:
        raise Rejected("Runtime dependencies changed during gate")
    artifact = canonical_json(result).encode()
    publish(
        output,
        {
            "report.json": artifact,
            "manifest.json": canonical_json(
                {
                    "version": result["version"],
                    "artifacts": {"report.json": hashlib.sha256(artifact).hexdigest()},
                }
            ).encode(),
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit-dir", "prereg", "inventory", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args.audit_dir, args.prereg, args.inventory, args.output_dir)
        print(
            canonical_json(
                {key: result[key] for key in ("verdict", "evaluation_allowed")}
            ),
            end="",
        )
        return 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RuntimeError,
    ) as exc:
        print(
            canonical_json(
                {
                    "error": str(exc),
                    "evaluation_allowed": False,
                    "market_evaluation": "NOT_ADMITTED",
                }
            ),
            file=sys.stderr,
            end="",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
