import hashlib
import json
import math
from datetime import date

import pytest

from tools import kronos_evaluation_preflight as gate


def test_weekday_ceiling():
    assert gate.weekday_ceiling(date(2025, 9, 10), date(2026, 8, 27)) == 252
    assert gate.weekday_ceiling(date(2026, 9, 19), date(2026, 9, 20)) == 0
    assert gate.weekday_ceiling(date(2026, 9, 22), date(2026, 9, 21)) == 0
    assert gate.weekday_ceiling(date(2026, 9, 21), date(2026, 9, 21)) == 1


def test_known_normal_calculation():
    result = gate.normal_scenario(252, 1)
    assert result["required_daily_observations"] == 1559
    assert result["required_252_day_years"] == pytest.approx(6.182557, rel=1e-5)
    assert result["hypothetical_power"] == pytest.approx(0.259511, rel=1e-5)
    n = result["required_daily_observations"]
    assert gate.normal_scenario(n, 1)["hypothetical_power"] >= 0.8
    assert gate.normal_scenario(n - 1, 1)["hypothetical_power"] < 0.8


def test_dependence_and_multiplicity_do_not_improve_power():
    base = gate.normal_scenario(252, 1)
    for inflation, comparisons in [(1.5, 1), (1, 3), (1.5, 3)]:
        value = gate.normal_scenario(252, 1, inflation, comparisons)
        assert (
            value["required_daily_observations"] > base["required_daily_observations"]
        )
        assert value["hypothetical_power"] < base["hypothetical_power"]
    assert gate.normal_scenario(0, 1)["hypothetical_power"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"days": -1},
        {"days": True},
        {"annual_sharpe": 0},
        {"annual_sharpe": math.nan},
        {"se_inflation": 0.5},
        {"comparisons": 0},
        {"comparisons": 1.5},
        {"alpha": 0},
        {"alpha": 1},
        {"target_power": 1},
    ],
)
def test_invalid_scenarios(kwargs):
    values = {"days": 252, "annual_sharpe": 1} | kwargs
    with pytest.raises(gate.AuditError):
        gate.normal_scenario(**values)


def test_fixed_evidence_and_gate_never_promote():
    audit, pilot = gate.read_evidence(gate.AUDIT), gate.read_evidence(gate.PILOT)
    report = gate.build_report(audit, pilot)
    assert report["post_revision_weekday_ceiling"] == 252
    assert report["admitted_untouched_sessions"] == 0
    assert report["actual_eligible_sessions"] is None
    assert report["actual_strategy_power"] is None
    assert not report["strategy_test_permitted"]
    assert report["power_verdict"] == "UNASSESSABLE"
    assert len(report["planning_scenarios"]) == 16
    # A high hypothetical Sharpe cannot upgrade the verdict.
    assert gate.normal_scenario(252, 100)["hypothetical_power"] > 0.99
    assert report["power_verdict"] == "UNASSESSABLE"


def test_input_scope_and_fingerprint(tmp_path, monkeypatch):
    with pytest.raises(gate.AuditError, match="approved"):
        gate.read_evidence("data/sealed_holdout/file.csv")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    input_path = tmp_path / gate.AUDIT
    input_path.parent.mkdir(parents=True)
    input_path.write_text("{}")
    with pytest.raises(gate.AuditError, match="fingerprint"):
        gate.read_evidence(gate.AUDIT)


def test_revision_and_dataset_mismatch():
    audit, pilot = gate.read_evidence(gate.AUDIT), gate.read_evidence(gate.PILOT)
    bad = pilot | {"model_revision": "bad"}
    with pytest.raises(gate.AuditError, match="revision"):
        gate.build_report(audit, bad)
    with pytest.raises(gate.AuditError, match="uniquely"):
        gate.build_report(audit, pilot | {"input_sha256": "bad"})


def test_output_manifest_and_cli_gate(tmp_path, monkeypatch, capsys):
    original = gate.read_evidence
    documents = {key: original(key) for key in gate.EVIDENCE}
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "read_evidence", documents.__getitem__)
    output = tmp_path / "docs/reports/kronos-evaluation-preflight/run"
    assert gate.main(["--output-dir", str(output)]) == 2
    assert json.loads(capsys.readouterr().out)["strategy_test_permitted"] is False
    for name, expected in json.loads((output / "COMPLETE.json").read_text()).items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    assert "not a measured edge" in (output / "report.md").read_text()
    with pytest.raises(gate.AuditError, match="fresh"):
        gate.run(output)
    with pytest.raises(gate.AuditError, match="fresh"):
        gate.run(tmp_path / "data" / "bad")


def test_symlink_output_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    allowed = tmp_path / "docs/reports/kronos-evaluation-preflight"
    allowed.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (allowed / "alias").symlink_to(external, target_is_directory=True)
    with pytest.raises(gate.AuditError, match="fresh"):
        gate.run(allowed / "alias" / "run")
