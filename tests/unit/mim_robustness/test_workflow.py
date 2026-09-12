"""Fixture workflow with real engine, reports and artifact lifecycle."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from research.mim_robustness import artifacts, study
from research.mim_robustness.__main__ import main
from research.mim_robustness.statistics import stationary_bootstrap
from research.mim_comparison.engine import simulate as original_simulate, Arm
from test_experiments import session

REAL_PREPARE = study.prepare


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    base = tmp_path / "research"
    base.mkdir()
    monkeypatch.setattr(artifacts, "BASE", base)
    monkeypatch.setattr(artifacts, "RUNS", base / "runs")
    monkeypatch.setattr(artifacts, "LOADED", artifacts.source_hashes())
    data = tmp_path / "input.csv"
    data.write_text("fixture market data\n")
    baseline = tmp_path / artifacts.BASELINE_NAME
    baseline.mkdir()
    prepared = []
    daily = []
    events = []
    decisions = []
    for index, day in enumerate(("2025-01-06", "2025-01-07", "2025-01-08")):
        rng = np.random.default_rng(20 + index)
        s = session(10000 + np.cumsum(rng.normal(0, 20, 390)), day=day)
        sigma = np.full(390, 0.002)
        prepared.append((s, sigma))
        for delay in (2, 1):
            d, e, c = original_simulate(s, sigma, Arm("A"), delay)
            for cost in (2.24, 3.24, 6.24):
                daily.append(
                    dict(
                        d,
                        cost=cost,
                        costs=d["turnover"] * cost / 2,
                        net=d["gross"] - d["turnover"] * cost / 2,
                    )
                )
            events.extend(dict(x, delay=delay, cost_scenario=2.24) for x in e)
            decisions.extend(dict(x, delay=delay) for x in c)
    for name, rows in [
        ("daily.csv", daily),
        ("ledger.csv", events),
        ("decisions.csv", decisions),
    ]:
        pd.DataFrame(rows).to_csv(baseline / name, index=False)
    artifacts.write_json(baseline / "manifest.json", {"source": {}})
    artifacts.write_json(
        baseline / "completion.json", {"sha256": artifacts.inventory(baseline)}
    )
    monkeypatch.setattr(artifacts, "DATA", data)
    monkeypatch.setattr(artifacts, "DATA_HASH", artifacts.digest(data))
    monkeypatch.setattr(artifacts, "BASELINE", baseline)
    monkeypatch.setattr(
        artifacts,
        "BASELINE_COMPLETION_HASH",
        artifacts.digest(baseline / "completion.json"),
    )
    # Only replace history loading/1333-session warmup. All reconciliation, engine,
    # four gates, inference, files, verification and evaluation are real.
    monkeypatch.setattr(study, "prepare", lambda d, b: (prepared, []))
    return base, data, baseline


def test_full_run_then_evaluate_preserves_all_ledgers(workspace, monkeypatch):
    base, data, baseline = workspace
    assert main(["run", "--data", str(data), "--baseline-run", str(baseline)]) == 0
    run = next((base / "runs").iterdir())
    artifacts.verify(run, complete=True)
    first = json.loads((run / "results.json").read_text())
    assert set(first["evaluation"]["candidates"]) == {"R", "E", "F", "P"}
    assert not first["deployment_authorized"]
    monkeypatch.setattr(artifacts, "DATA", Path("/absent-default-data.csv"))
    monkeypatch.setattr(artifacts, "BASELINE", Path("/absent-default-baseline"))
    assert main(["evaluate", "--run", str(run)]) == 0
    evaluation = next(p for p in (base / "runs").iterdir() if p != run)
    artifacts.verify(evaluation, complete=True)
    assert json.loads((evaluation / "results.json").read_text()) == first
    for name in (
        "daily.csv",
        "ledger.csv",
        "decisions.csv",
        "features.csv",
        "paired-daily.csv",
        "trades.csv",
    ):
        assert (run / name).read_bytes() == (evaluation / name).read_bytes()
    assert (run / "report.html").read_text().startswith("<!doctype html>")
    # No workflow output may escape the designated runs root.
    assert set(p.name for p in base.iterdir()) == {"runs"}


@pytest.mark.parametrize(
    "name,column",
    [("daily.csv", "net"), ("ledger.csv", "fill"), ("decisions.csv", "target")],
)
def test_cli_baseline_mismatch_blocks_candidates(workspace, monkeypatch, name, column):
    base, data, baseline = workspace
    frame = pd.read_csv(baseline / name)
    frame.loc[0, column] += 1
    frame.to_csv(baseline / name, index=False)
    (baseline / "completion.json").unlink()
    artifacts.write_json(
        baseline / "completion.json", {"sha256": artifacts.inventory(baseline)}
    )
    monkeypatch.setattr(
        artifacts,
        "BASELINE_COMPLETION_HASH",
        artifacts.digest(baseline / "completion.json"),
    )
    assert main(["run", "--data", str(data), "--baseline-run", str(baseline)]) == 2
    failed = next((base / "runs").iterdir())
    artifacts.verify_inventory(failed)
    assert (failed / "failure.json").exists()
    assert (
        not (failed / "features.csv").exists()
        and not (failed / "results.json").exists()
    )


def test_runtime_change_refused(workspace, monkeypatch):
    _, data, baseline = workspace
    run = artifacts.create("audit", [data, baseline / "completion.json"])
    monkeypatch.setattr(artifacts, "runtime", lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="runtime"):
        artifacts.verify(run)


@pytest.mark.parametrize(
    "defect", [None, "missing", "duplicate", "early_close", "zero_volume"]
)
def test_real_data_preparation_warmup_dst_and_exclusions(
    workspace, monkeypatch, defect
):
    _, data, baseline = workspace
    rows = [
        session(day=str(day.date()), contract="MNQM25").bars
        for day in pd.bdate_range("2025-03-03", periods=20)
    ]
    if defect == "missing":
        rows[5] = rows[5].iloc[:-1]
    if defect == "duplicate":
        rows[5] = pd.concat([rows[5], rows[5].iloc[:1]])
    if defect == "early_close":
        rows[5] = rows[5].iloc[:210]
    if defect == "zero_volume":
        rows[5]["volume"] = 0.0
    pd.concat(rows).to_csv(data, index=False)
    monkeypatch.setattr(artifacts, "DATA_HASH", artifacts.digest(data))
    monkeypatch.setitem(
        artifacts.CONFIG,
        "expected_sessions",
        5 if defect in (None, "zero_volume") else 3,
    )
    prepared, excluded = REAL_PREPARE(data, baseline)
    assert len(prepared) == artifacts.CONFIG["expected_sessions"]
    assert (
        sum(
            e["exclusion"] == "14_complete_prior_selected_sessions_required"
            for e in excluded
        )
        == 14
    )
    assert all(len(s.bars) == 390 and np.isfinite(sigma).all() for s, sigma in prepared)
    assert (
        rows[0].timestamp.iloc[0].utcoffset() != rows[-1].timestamp.iloc[0].utcoffset()
    )
    if defect in ("missing", "early_close", "duplicate"):
        reason = (
            "duplicate_minutes"
            if defect == "duplicate"
            else "missing_minutes_or_early_close"
        )
        assert any(reason in e["exclusion"] for e in excluded)


def test_real_preparation_volume_roll_uses_new_contract_close(workspace, monkeypatch):
    _, data, baseline = workspace
    rows = []
    for i, day in enumerate(pd.bdate_range("2025-01-06", periods=20)):
        h = session(day=str(day.date()))
        m = session(np.full(390, 20000.0), day=str(day.date()), contract="MNQM25")
        h.bars["volume"] = 100.0
        m.bars["volume"] = 50.0 if i < 15 else 200.0
        rows.extend([h.bars, m.bars])
    pd.concat(rows).to_csv(data, index=False)
    monkeypatch.setattr(artifacts, "DATA_HASH", artifacts.digest(data))
    monkeypatch.setitem(artifacts.CONFIG, "expected_sessions", 5)
    prepared, _ = REAL_PREPARE(data, baseline)
    assert prepared[0][0].contract == "MNQH25"
    assert (
        prepared[1][0].contract == "MNQM25" and prepared[1][0].previous_close == 20000.0
    )


def test_audit_does_not_compute_candidate_returns_and_failure_is_sealed(workspace):
    base, data, baseline = workspace
    assert main(["audit", "--data", str(data), "--baseline-run", str(baseline)]) == 0
    audit = next((base / "runs").iterdir())
    assert set(pd.read_csv(audit / "daily.csv").arm) == {"A"}
    assert not (audit / "results.json").exists()
    assert main(["evaluate", "--run", str(audit)]) == 2
    failure = next(p for p in (base / "runs").iterdir() if p != audit)
    artifacts.verify_inventory(failure)
    assert (failure / "failure.json").exists() and (failure / "report.md").exists()


def test_data_and_source_drift_are_refused(workspace):
    base, data, baseline = workspace
    artifacts.verify_baseline(data, baseline)
    data.write_text("mutated")
    with pytest.raises(ValueError, match="input hash"):
        artifacts.verify_baseline(data, baseline)
    (base / "new_module.py").write_text("# source drift")
    with pytest.raises(ValueError):
        artifacts.create("audit", [data])


def test_bootstrap_matches_independent_index_and_percentile_calculation():
    x = np.arange(1.0, 21.0)
    matrix = np.column_stack([x] + [x + i for i in range(1, 5)])
    draws = 40
    n = len(x)
    rng = np.random.default_rng(7)
    indices = np.empty((draws, n), dtype=int)
    indices[:, 0] = rng.integers(n, size=draws)
    restart = rng.random((draws, n - 1)) < 0.2
    fresh = rng.integers(n, size=(draws, n - 1))
    for j in range(1, n):
        indices[:, j] = np.where(
            restart[:, j - 1], fresh[:, j - 1], (indices[:, j - 1] + 1) % n
        )
    expected = np.sqrt(252) / np.std(x[indices], axis=1, ddof=1)
    actual = stationary_bootstrap(matrix, 5, draws=draws, batch=draws)
    np.testing.assert_allclose(
        actual["R"]["ci98_75"], np.percentile(expected, [0.625, 99.375]), rtol=1e-12
    )
    np.testing.assert_allclose(
        actual["P"]["ci95"], 4 * np.percentile(expected, [2.5, 97.5]), rtol=1e-12
    )
