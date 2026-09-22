"""Synthetic-only checks for readiness accounting, isolation and false admission."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tools import trading_model_readiness as audit


def write_csv(
    path: Path, rows: list[list[str]], header: list[str] | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            header
            or ["timestamp", "open", "high", "low", "close", "volume", "contract"]
        )
        writer.writerows(rows)
    return path


def session(
    day: str = "2025-03-10", minute: int = 570, contract: str = "MNQH25"
) -> list[list[str]]:
    opening = datetime.fromisoformat(day).replace(tzinfo=audit.NY) + timedelta(
        minutes=minute
    )
    return [
        [
            (opening + timedelta(minutes=i)).isoformat(),
            "100",
            "101",
            "99",
            "100",
            "5",
            contract,
        ]
        for i in range(390)
    ]


@pytest.mark.parametrize("day", ["2025-03-07", "2025-03-10", "2025-11-03"])
@pytest.mark.parametrize(
    "minute,label,other", [(570, "start", "end"), (571, "end", "start")]
)
def test_complete_grids_keep_both_timestamp_hypotheses(
    tmp_path, day, minute, label, other
):
    result = audit.inspect_csv(write_csv(tmp_path / "input.csv", session(day, minute)))
    assert result["counts"]["valid_rows"] == 390
    assert result["grid_hypotheses"][label]["candidate_15min_contract_groups"] == 26
    assert result["grid_hypotheses"][label]["dates_with_any_full_regular_grid"] == 1
    assert result["grid_hypotheses"][other]["dates_with_any_full_regular_grid"] == 0
    assert result["data_gate"] == "HOLD_DATA"


def test_contract_days_do_not_multiply_independent_dates(tmp_path):
    result = audit.inspect_csv(
        write_csv(tmp_path / "input.csv", session() + session(contract="MNQM25"))
    )
    grid = result["grid_hypotheses"]["start"]
    assert grid["full_regular_grid_contract_days"] == 2
    assert grid["dates_with_any_full_regular_grid"] == 1
    assert result["dates_with_multiple_contracts"] == 1


def test_duplicate_and_out_of_order_minutes_cannot_look_clean(tmp_path):
    rows = session()
    result = audit.inspect_csv(write_csv(tmp_path / "input.csv", rows + [rows[10]]))
    assert result["counts"]["duplicate_weekday_contract_minutes"] == 1
    assert result["counts"]["nonincreasing_within_contract_rows"] == 1
    assert result["structural_quality"] == "INVALID"
    assert result["grid_hypotheses"]["start"]["candidate_15min_contract_groups"] == 0


@pytest.mark.parametrize(
    "column,value",
    [
        (0, "2025-03-10T09:30:00"),
        (0, "2025-03-10T09:30:01Z"),
        (1, "nan"),
        (1, "0"),
        (2, "98"),
        (3, "102"),
        (5, "-1"),
        (6, "MNQ_continuous"),
    ],
)
def test_invalid_rows_are_counted_not_silently_admitted(tmp_path, column, value):
    rows = session()
    rows[0][column] = value
    result = audit.inspect_csv(write_csv(tmp_path / "input.csv", rows))
    assert result["counts"]["invalid_rows"] == 1
    assert result["counts"]["valid_rows"] == 389
    assert result["structural_quality"] == "INVALID"
    assert result["grid_hypotheses"]["start"]["dates_with_any_full_regular_grid"] == 0


def test_missing_contract_is_not_inferred(tmp_path):
    rows = [row[:-1] for row in session()]
    result = audit.inspect_csv(
        write_csv(tmp_path / "input.csv", rows, list(audit.FIELDS))
    )
    assert result["contract_identity"] == "ABSENT"
    assert result["contracts"] == ["UNKNOWN"]
    assert result["data_gate"] == "HOLD_DATA"


def test_weekend_and_missing_minute_do_not_make_full_grid(tmp_path):
    rows = session("2025-03-08") + session()[:-1]
    result = audit.inspect_csv(write_csv(tmp_path / "input.csv", rows))
    assert result["counts"]["weekend_rows"] == 390
    assert result["observed_weekday_dates"] == 1
    assert result["grid_hypotheses"]["start"]["dates_with_any_full_regular_grid"] == 0


@pytest.mark.parametrize(
    "header",
    [["price"], ["timestamp", "TimeStamp", "open", "high", "low", "close", "volume"]],
)
def test_bad_schema_still_reports_a_hold(tmp_path, header):
    result = audit.inspect_csv(write_csv(tmp_path / "input.csv", [], header))
    assert result["status"] == "INVALID_SCHEMA"
    assert result["structural_quality"] == "INVALID"


def test_missing_and_empty_files_report_no_evidence(tmp_path):
    assert audit.inspect_csv(tmp_path / "missing.csv")["status"] == "MISSING"
    result = audit.inspect_csv(write_csv(tmp_path / "empty.csv", []))
    assert result["status"] == "EMPTY"
    assert result["first_valid_timestamp"] is None


@pytest.mark.parametrize("component", ["sealed_holdout", "Sealed_Holdout"])
def test_sealed_path_is_refused_before_read(tmp_path, component, monkeypatch):
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("opened a sealed input")
    )
    with pytest.raises(audit.AuditError, match="prohibited"):
        audit.inspect_csv(tmp_path / component / "input.csv")


def test_sealed_symlink_alias_is_refused(tmp_path):
    target = tmp_path / "sealed_holdout"
    target.mkdir()
    (tmp_path / "alias").symlink_to(target, target_is_directory=True)
    with pytest.raises(audit.AuditError, match="prohibited"):
        audit.safe_path(tmp_path / "alias" / "input.csv")


def test_input_mutation_refuses_a_report(tmp_path, monkeypatch):
    path = write_csv(tmp_path / "input.csv", session())
    hashes = iter(["before", "after"])
    monkeypatch.setattr(audit, "digest", lambda _: next(hashes))
    with pytest.raises(audit.AuditError, match="changed"):
        audit.inspect_csv(path)


def test_report_never_admits_strategy_or_fabricates_training_time(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(audit.shutil, "which", lambda _: None)
    source = write_csv(tmp_path / "input.csv", session())
    output = tmp_path / "report"
    before = source.read_bytes()
    result = audit.run([source], output, "a" * 40)
    assert source.read_bytes() == before
    assert result["strategy_testing_authorized"] is False
    assert result["power_gate"]["status"] == "UNASSESSABLE"
    assert result["compute"]["training_gpu_hours"] is None
    assert result["compute"]["training_cost"] is None
    assert result["compute"]["nvidia_probe"] == "NVIDIA_TOOL_UNAVAILABLE"
    assert json.loads((output / "report.json").read_text()) == result
    assert "HOLD" in (output / "report.md").read_text()


def test_invalid_schema_can_be_published_without_false_pass(tmp_path):
    result = audit.run(
        [write_csv(tmp_path / "input.csv", [], ["bad"])], tmp_path / "report", "a" * 40
    )
    assert result["datasets"][0]["status"] == "INVALID_SCHEMA"


def test_existing_output_and_live_destination_are_refused(tmp_path):
    source = write_csv(tmp_path / "input.csv", session())
    existing = tmp_path / "report"
    existing.mkdir()
    for destination in (
        existing,
        tmp_path / "data" / "new",
        tmp_path / "logs" / "new",
        tmp_path,
    ):
        with pytest.raises(audit.AuditError):
            audit.run([source], destination, "a" * 40)
    assert list(existing.iterdir()) == []


def test_duplicate_alias_and_unpinned_revision_are_refused(tmp_path):
    source = write_csv(tmp_path / "input.csv", session())
    alias = tmp_path / "alias.csv"
    alias.symlink_to(source)
    with pytest.raises(audit.AuditError, match="duplicate"):
        audit.run([source, alias], tmp_path / "report", "a" * 40)
    with pytest.raises(audit.AuditError, match="canonical"):
        audit.run([source], tmp_path / "report", "main")


def test_registration_tamper_is_refused_before_csv_read(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "REGISTRATION_HASH", "changed")
    monkeypatch.setattr(
        audit, "inspect_csv", lambda _: pytest.fail("read before seal check")
    )
    with pytest.raises(audit.AuditError, match="preregistration"):
        audit.run([tmp_path / "input.csv"], tmp_path / "report", "a" * 40)
    assert not (tmp_path / "report").exists()
