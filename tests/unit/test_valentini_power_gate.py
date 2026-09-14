"""Synthetic-only design mathematics and metadata/output admission firewalls."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest
from scipy import integrate, optimize, stats  # type: ignore[import-untyped]

from tools import valentini_power_gate as gate
from tools.valentini_reclaim import Rejected, canonical_json, sha256_file


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> str:
    data = canonical_json(value).encode()
    path.write_bytes(data)
    return digest(data)


@pytest.mark.parametrize("n,d", [(2, 8.0), (7, 0.4), (12, 0.7)])
def test_power_independent_chi_square_oracle(n: int, d: float) -> None:
    df = n - 1
    critical = stats.t.ppf(0.95, df)
    # T = (Z + d sqrt(n)) / sqrt(V/df), V ~ chi-square independently of Z.
    oracle, error = integrate.quad(
        lambda v: stats.norm.sf(critical * math.sqrt(v / df) - d * math.sqrt(n))
        * stats.chi2.pdf(v, df),
        0,
        math.inf,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    assert error < 1e-9
    assert gate.conditional_power(n, d) == pytest.approx(oracle, abs=2e-10)


def test_null_alpha_and_monotonicity() -> None:
    for n in (2, 7, 12, 40):
        assert gate.conditional_power(n, 0) == pytest.approx(0.05, abs=1e-12)
    effects = [gate.assessable_power(12, d) for d in (0.0, 0.4, 0.7, 1.2)]
    counts = [gate.assessable_power(n, 0.4) for n in (2, 7, 12, 40)]
    assert all(a < b for a, b in zip(effects, effects[1:]))
    assert all(a < b for a, b in zip(counts, counts[1:]))


@pytest.mark.parametrize("d", [0.1, 0.2, 0.3, 0.5, 1.0, 20.0])
def test_minimum_n_adjacent_boundary(d: float) -> None:
    result = gate.minimum_sessions(d)
    n = result["n"]
    assert n >= 2
    assert gate.assessable_power(n, d) >= gate.TARGET
    assert n == 2 or gate.assessable_power(n - 1, d) < gate.TARGET
    assert result["status"] == "CONDITIONAL_MODEL"


@pytest.mark.parametrize("n", [2, 7, 12, 40])
def test_mde_target_inversion(n: int) -> None:
    result = gate.detectable_effect(n)
    assert gate.assessable_power(n, result["d"]) == pytest.approx(0.8, abs=1e-9)
    assert gate.assessable_power(n, result["d"] - 1e-7) < 0.8


@pytest.mark.parametrize("n", [0, 1])
def test_unassessable_counts(n: int) -> None:
    assert gate.conditional_power(n, 0.4) is None
    result = gate.model_report(n)
    assert result["status"] == "UNASSESSABLE_N_LT_2"
    assert result["mde_frontier"] == []
    assert result["observed_count_mde"]["d"] is None
    assert all(
        r["power_at_observed_count"] is None for r in result["hypothetical_scenarios"]
    )


@pytest.mark.parametrize("bad", [True, "7", None, -1, 7.1, math.nan, math.inf])
def test_invalid_counts(bad: Any) -> None:
    for operation in (gate.detectable_effect, gate.model_report):
        with pytest.raises(Rejected):
            operation(bad)
    with pytest.raises(Rejected):
        gate.conditional_power(bad, 0.4)


@pytest.mark.parametrize(
    "bad", [True, "0.4", None, -0.1, math.nan, math.inf, -math.inf]
)
def test_invalid_effects(bad: Any) -> None:
    for operation in (
        gate.minimum_sessions,
        lambda d: gate.conditional_power(7, d),
    ):
        with pytest.raises(Rejected):
            operation(bad)


def test_cap_and_numerical_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    result = gate.minimum_sessions(0.00001, cap=16)
    assert result["n"] is None
    assert result["status"] == "UNRESOLVED_ABOVE_CAP"
    with pytest.raises(Rejected):
        gate.minimum_sessions(0)
    for cap in (True, 1, 2.2, 1_000_001):
        with pytest.raises(Rejected):
            gate.minimum_sessions(0.4, cap=cap)  # type: ignore[arg-type]
    monkeypatch.setattr(stats.nct, "sf", lambda *args: math.nan)
    with pytest.raises(Rejected, match="Nonfinite"):
        gate.conditional_power(7, 0.4)


def test_failed_root_and_bracket(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> float:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(optimize, "brentq", fail)
    with pytest.raises(Rejected, match="root failed"):
        gate.detectable_effect(7)
    monkeypatch.setattr(gate, "assessable_power", lambda n, d: 0.05)
    assert gate.detectable_effect(7)["status"] == "UNRESOLVED_BRACKET"


def test_normal_crosscheck_is_separate() -> None:
    normal = gate.normal_crosscheck(12, 0.7)
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.8)
    assert normal["required_n_continuous"] == pytest.approx((z / 0.7) ** 2)
    assert normal["mde_at_n"] == pytest.approx(z / math.sqrt(12))
    assert normal["power_at_n"] > gate.assessable_power(12, 0.7)


def synthetic_metadata() -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    ledger: list[Any] = []
    summaries: dict[str, Any] = {}
    calendar = []
    for index in range(8):
        start = datetime(2030, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
        end = start + timedelta(minutes=3)
        eligible = index < 7
        evidence = dict(
            name=f"synthetic-{index}",
            start=start.isoformat(),
            end=end.isoformat(),
            verification="VERIFIED",
        )
        calendar.append(evidence)
        ledger.append(
            dict(
                session=evidence["name"],
                start_ns=int(start.timestamp()) * 1_000_000_000,
                end_ns=int(end.timestamp()) * 1_000_000_000,
                eligible=eligible,
                expected_minutes=3,
                observed_minutes=3 if eligible else 2,
                exclusions=({} if eligible else {"SYNTHETIC_MISSING_MINUTE": [0]}),
                evidence=evidence,
            )
        )
        summaries[evidence["name"]] = dict(
            comparison_count=2 if eligible else 0,
            snapshot_status_counts=(
                {"COMPARED": 2, "EMPTY_PREFIX": 1} if eligible else {}
            ),
        )
    report = dict(
        version=gate.MEASUREMENT_VERSION,
        kind="native_volume_profile_measurement",
        market_evaluation="NOT_ADMITTED",
        conservation={"exact": True},
        eligible_sessions=7,
        excluded_sessions=1,
        session_profile_summaries=summaries,
        profile_summary=dict(
            comparison_count=14,
            snapshot_status_counts={"COMPARED": 14, "EMPTY_PREFIX": 7},
        ),
    )
    provenance = dict(
        version=gate.MEASUREMENT_VERSION,
        calendar={"sessions": calendar},
        code_sha256={
            name: sha256_file(gate.ROOT / name) for name in gate.PROVENANCE_CODE
        },
    )
    return report, ledger, provenance


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    audit = tmp_path / "audit"
    audit.mkdir()
    report, ledger, provenance = synthetic_metadata()
    for name, value in (
        ("report.json", report),
        ("sessions.json", ledger),
        ("provenance.json", provenance),
    ):
        write_json(audit / name, value)
    # Deliberately invalid JSON: hashing is allowed; row parsing must never occur.
    for name in gate.ARTIFACTS - {
        "report.json",
        "sessions.json",
        "provenance.json",
    }:
        (audit / name).write_bytes(b"opaque synthetic bytes; never parse\xff")
    repin_manifest(audit, monkeypatch)
    prereg = tmp_path / "prereg.md"
    prereg.write_text("Synthetic registration, not market data\n")
    inventory = tmp_path / "inventory.json"
    write_json(inventory, {"independent_full_session_calibration_found": False})
    monkeypatch.setattr(gate, "PREREG_SHA256", sha256_file(prereg))
    monkeypatch.setattr(gate, "INVENTORY_SHA256", sha256_file(inventory))
    return dict(
        audit_dir=audit,
        prereg=prereg,
        inventory=inventory,
        output_dir=tmp_path / "result",
    )


def repin_manifest(audit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_hash = write_json(
        audit / "manifest.json",
        dict(
            version=gate.MEASUREMENT_VERSION,
            artifacts={name: sha256_file(audit / name) for name in gate.ARTIFACTS},
        ),
    )
    monkeypatch.setattr(gate, "MANIFEST_SHA256", manifest_hash)


def cli_args(inputs: dict[str, Path]) -> list[str]:
    return [
        part
        for key, path in inputs.items()
        for part in ("--" + key.replace("_", "-"), str(path))
    ]


def test_valid_terminal_gate(inputs: dict[str, Path]) -> None:
    assert gate.main(cli_args(inputs)) == 0
    report = json.loads((inputs["output_dir"] / "report.json").read_text())
    assert report["verdict"] == "POWER_UNDETERMINED"
    assert report["evaluation_allowed"] is False
    assert report["market_evaluation"] == "NOT_ADMITTED"
    model = report["conditional_model"]
    assert model["observed_session_count"] == 7
    assert sum(row["eligible"] for row in report["preserved_session_ledger"]) == 7
    frontier = model["mde_frontier"]
    assert [row["n"] for row in frontier] == list(range(2, 8))
    for row in frontier:
        assert row["status"] == "CONDITIONAL_MODEL"
        assert row["d"] > 0
        assert gate.assessable_power(row["n"], row["d"]) == pytest.approx(0.8, abs=1e-9)
    assert model["observed_count_mde"] == frontier[-1]
    assert frontier[-1] == gate.detectable_effect(7)
    scenarios = model["hypothetical_scenarios"]
    for scenario in scenarios:
        assert scenario["power_at_observed_count"] == pytest.approx(
            gate.assessable_power(7, scenario["d"]), abs=1e-12
        )
    assert [r["d"] for r in scenarios] == list(gate.EFFECTS)
    assert all(
        r["minimum_independent_sessions"]["power_at_n"] >= 0.8 for r in scenarios
    )
    assert report["preserved_session_ledger"][-1]["eligible"] is False
    assert report["prereg_commit"] == gate.PREREG_COMMIT
    assert report["runtime"]["dependency_module_sha256"]
    assert report["data_boundary"]["raw_native_decode"] is False
    manifest = json.loads((inputs["output_dir"] / "manifest.json").read_text())
    assert manifest["artifacts"]["report.json"] == sha256_file(
        inputs["output_dir"] / "report.json"
    )


@pytest.mark.parametrize(
    "target", ["prereg", "inventory", "manifest.json", *sorted(gate.ARTIFACTS)]
)
@pytest.mark.parametrize("operation", ["mutate", "delete"])
def test_bound_inputs_refuse_before_model(
    inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    operation: str,
) -> None:
    path = inputs[target] if target in inputs else inputs["audit_dir"] / target
    if operation == "mutate":
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        path.unlink()

    def forbidden(n: int) -> dict[str, Any]:
        pytest.fail("Model ran before input refusal")

    monkeypatch.setattr(gate, "model_report", forbidden)
    assert gate.main(cli_args(inputs)) == 2
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r, rows, p: rows[0].update(eligible=1),
        lambda r, rows, p: rows[0].update(eligible=False),
        lambda r, rows, p: rows[0].update(session=rows[1]["session"]),
        lambda r, rows, p: rows[1].update(start_ns=rows[0]["start_ns"]),
        lambda r, rows, p: rows[0].update(observed_minutes=2),
        lambda r, rows, p: r.update(eligible_sessions=8),
        lambda r, rows, p: r.update(eligible_sessions=True),
        lambda r, rows, p: r.update(excluded_sessions=0),
        lambda r, rows, p: r.update(market_evaluation="POWERED"),
        lambda r, rows, p: r["profile_summary"].update(comparison_count=15),
        lambda r, rows, p: r["session_profile_summaries"][rows[0]["session"]].update(
            comparison_count=3
        ),
        lambda r, rows, p: p.update(code_sha256={}),
        lambda r, rows, p: p["code_sha256"].update(
            {next(iter(gate.PROVENANCE_CODE)): "0" * 64}
        ),
    ],
)
def test_metadata_consistency(mutate: Callable[..., None]) -> None:
    report, ledger, provenance = synthetic_metadata()
    mutate(report, ledger, provenance)
    with pytest.raises(Rejected):
        gate.validate_metadata(report, ledger, provenance)


def test_overlap_even_with_matching_calendar() -> None:
    report, ledger, provenance = synthetic_metadata()
    ledger[1]["start_ns"] = ledger[0]["start_ns"]
    ledger[1]["evidence"]["start"] = ledger[0]["evidence"]["start"]
    with pytest.raises(Rejected, match="Overlapping"):
        gate.validate_metadata(report, ledger, provenance)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', "[]", "{"])
def test_strict_schema_refusal(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    (inputs["audit_dir"] / "report.json").write_text(text)
    repin_manifest(inputs["audit_dir"], monkeypatch)
    assert gate.main(cli_args(inputs)) == 2
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "target",
    [
        "histograms.jsonl",
        "sessions.json",
        "manifest.json",
        "prereg",
        "inventory",
    ],
)
def test_mutation_during_model_never_publishes(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    original = gate.model_report

    def mutate(n: int) -> dict[str, Any]:
        path = inputs[target] if target in inputs else inputs["audit_dir"] / target
        path.write_bytes(path.read_bytes() + b"changed after verification")
        return original(n)

    monkeypatch.setattr(gate, "model_report", mutate)
    assert gate.main(cli_args(inputs)) == 2
    assert not inputs["output_dir"].exists()


def test_fake_calibration_refused_even_with_synthetic_pin(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_json(
        inputs["inventory"],
        {
            "independent_full_session_calibration_found": True,
            "verdict": "POWERED",
        },
    )
    monkeypatch.setattr(gate, "INVENTORY_SHA256", sha256_file(inputs["inventory"]))
    assert gate.main(cli_args(inputs)) == 2
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize(
    "kind",
    ["existing", "audit-child", "input", "symlink", "hardlink", "protected"],
)
def test_path_firewall(inputs: dict[str, Path], kind: str) -> None:
    original_prereg = inputs["prereg"].read_bytes()
    if kind == "existing":
        inputs["output_dir"].mkdir()
    elif kind == "audit-child":
        inputs["output_dir"] = inputs["audit_dir"] / "new-output"
    elif kind == "input":
        inputs["output_dir"] = inputs["prereg"]
    elif kind == "symlink":
        link = inputs["prereg"].parent / "alias"
        link.symlink_to(inputs["prereg"])
        inputs["prereg"] = link
    elif kind == "hardlink":
        link = inputs["prereg"].parent / "hard-alias"
        link.hardlink_to(inputs["prereg"])
        inputs["prereg"] = link
    else:
        inputs["output_dir"] = inputs["output_dir"].parent / "sealed_holdout" / "output"
    assert gate.main(cli_args(inputs)) == 2
    assert inputs["prereg"].read_bytes() == original_prereg


def test_publish_collision_created_during_model(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = gate.model_report

    def collide(n: int) -> dict[str, Any]:
        inputs["output_dir"].mkdir()
        (inputs["output_dir"] / "sentinel").write_text("preserve")
        return original(n)

    monkeypatch.setattr(gate, "model_report", collide)
    assert gate.main(cli_args(inputs)) == 2
    assert sorted(p.name for p in inputs["output_dir"].iterdir()) == ["sentinel"]
    assert not list(inputs["output_dir"].parent.glob(".valentini-native-*"))


def test_no_price_trade_decoder_calls_or_promotion_option() -> None:
    source = Path(gate.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        name.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in node.names
    }
    forbidden = {
        "load_builder",
        "histogram_builder",
        "reconcile",
        "compare_profiles",
        "Bar",
        "Profile",
        "TradeLogger",
        "simulate",
        "subprocess",
        "requests",
        "sqlite3",
    }
    assert not (imported & forbidden)
    calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert not (calls & forbidden)
    assert "src.research" not in source
    with pytest.raises(SystemExit) as exc:
        gate.main(["--promote"])
    assert exc.value.code == 2


def test_exclusions_preserved() -> None:
    report, ledger, provenance = synthetic_metadata()
    original = copy.deepcopy(ledger)
    assert gate.validate_metadata(report, ledger, provenance) == 7
    assert ledger == original


def test_null_conservation_cli_refusal(
    inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = inputs["audit_dir"] / "report.json"
    report = json.loads(path.read_text())
    report["conservation"] = None
    write_json(path, report)
    repin_manifest(inputs["audit_dir"], monkeypatch)
    assert gate.main(cli_args(inputs)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"] == "Expected JSON object: conservation"
    assert error["evaluation_allowed"] is False
    assert error["market_evaluation"] == "NOT_ADMITTED"
    assert not inputs["output_dir"].exists()


def test_provenance_change_before_code_snapshot(
    inputs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_validate = gate.validate_metadata
    target = gate.ROOT / sorted(gate.PROVENANCE_CODE)[0]

    def changed_digest(path: str | Path) -> str:
        return "0" * 64 if Path(path) == target else sha256_file(path)

    def validate_then_change(
        report: Mapping[str, Any], ledger: Any, provenance: Mapping[str, Any]
    ) -> int:
        count = original_validate(report, ledger, provenance)
        monkeypatch.setattr(gate, "sha256_file", changed_digest)
        return count

    def forbidden(n: int) -> dict[str, Any]:
        pytest.fail("Model ran after provenance code changed")

    monkeypatch.setattr(gate, "validate_metadata", validate_then_change)
    monkeypatch.setattr(gate, "model_report", forbidden)
    assert gate.main(cli_args(inputs)) == 2
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("target", ["prereg", "inventory"])
@pytest.mark.parametrize("kind", ["fifo", "device", "directory"])
def test_nonregular_input_refused_without_read(
    inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    kind: str,
) -> None:
    if kind == "device":
        inputs[target] = Path("/dev/null")
    else:
        inputs[target].unlink()
        if kind == "fifo":
            os.mkfifo(inputs[target])
        else:
            inputs[target].mkdir()
    original_read = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path == inputs[target]:
            pytest.fail("Attempted read of nonregular input")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    assert gate.main(cli_args(inputs)) == 2
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("target", ["code", "runtime"])
def test_code_or_runtime_change_during_model_refused(
    inputs: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    original_model = gate.model_report
    original_runtime = gate.runtime_evidence
    code_path = Path(gate.__file__)

    def changed_digest(path: str | Path) -> str:
        return "0" * 64 if Path(path) == code_path else sha256_file(path)

    def changed_runtime() -> dict[str, Any]:
        evidence = original_runtime()
        evidence["executable_sha256"] = "0" * 64
        return evidence

    def mutate(n: int) -> dict[str, Any]:
        result = original_model(n)
        if target == "code":
            monkeypatch.setattr(gate, "sha256_file", changed_digest)
        else:
            monkeypatch.setattr(gate, "runtime_evidence", changed_runtime)
        return result

    monkeypatch.setattr(gate, "model_report", mutate)
    assert gate.main(cli_args(inputs)) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    expected = (
        "Code changed during gate"
        if target == "code"
        else "Runtime dependencies changed during gate"
    )
    assert error["error"] == expected
    assert captured.out == ""
    assert not inputs["output_dir"].exists()
