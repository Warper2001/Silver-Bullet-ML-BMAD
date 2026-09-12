import json
from unittest.mock import Mock

import pytest

from research.mim_lifecycle import artifacts as a
from research.mim_lifecycle import __main__ as cli
from research.mim_lifecycle.report import write_report

from test_analysis import source_frames
from research.mim_lifecycle.analysis import analyze, summarize


def populate_success(path):
    for name in a.SUCCESS_FILES:
        target = path / name
        if not target.exists():
            target.write_text("synthetic successful output\n")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    base = tmp_path / "package"
    base.mkdir()
    monkeypatch.setattr(a, "BASE", base)
    monkeypatch.setattr(a, "RUNS", base / "runs")
    return base


def test_exclusive_immutable_runs_inventory_and_tamper(isolated):
    first, second = a.create(), a.create()
    assert first != second
    a.write_json(first / "summary.json", {"net": 12.34})
    populate_success(first)
    a.seal(first)
    expected = json.loads((first / "completion.json").read_text())["sha256"]
    assert expected == a.inventory(first)
    assert (first / "summary.json").stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        a.write_json(first / "summary.json", {"net": 0})
    with pytest.raises(FileExistsError):
        a.seal(first)
    (first / "summary.json").chmod(0o644)
    (first / "summary.json").write_text("tampered")
    assert expected != a.inventory(first)


def test_output_escape_and_symlinks_rejected(isolated, tmp_path):
    path = a.create()
    with pytest.raises(ValueError):
        a.safe_run(tmp_path / "elsewhere")
    (path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Symlink"):
        a.inventory(path)


def test_cli_failure_is_sealed_without_success_report(isolated, monkeypatch):
    def bad_bind(path):
        (path / "report.md").write_text("partial success")
        raise ValueError("synthetic integrity drift")

    monkeypatch.setattr(a, "bind", bad_bind)
    with pytest.raises(ValueError, match="synthetic integrity drift"):
        cli.run()
    path = next(a.RUNS.iterdir())
    assert not (path / "report.md").exists()
    assert json.loads((path / "failure.json").read_text())["type"] == "ValueError"
    assert json.loads((path / "completion.json").read_text())["sha256"] == a.inventory(
        path
    )


def test_spec_status_edit_preserves_frozen_contract(tmp_path):
    p = tmp_path / "spec.md"
    p.write_text(
        "status: in-progress\n<frozen-after-approval>contract</frozen-after-approval>"
    )
    before = a.frozen_spec_hash(p)
    p.write_text(
        "status: complete\n<frozen-after-approval>contract</frozen-after-approval>\nreview appended"
    )
    assert a.frozen_spec_hash(p) == before
    p.write_text("<frozen-after-approval>changed</frozen-after-approval>")
    assert a.frozen_spec_hash(p) != before


def test_report_portable_and_numerically_reproducible(isolated):
    trades, daily, bars = source_frames()
    _, life, marks = analyze(trades, daily, bars)
    summary = summarize(life, marks, daily)
    first, second = a.create(), a.create()
    write_report(first, summary, life)
    write_report(second, summary, life)
    html = (first / "report.html").read_text()
    assert html.count("<svg") == 2
    assert "<script" not in html and "<link" not in html
    assert "censored" in html.lower() and "hindsight" in html.lower()
    assert (first / "excursions.svg").read_bytes() == (
        second / "excursions.svg"
    ).read_bytes()
    assert (first / "first_positive.svg").read_bytes() == (
        second / "first_positive.svg"
    ).read_bytes()
    assert (first / "report.md").read_bytes() == (second / "report.md").read_bytes()


@pytest.fixture
def provenance(isolated, monkeypatch):
    """Approved synthetic evidence exercises lifecycle bind/verify without real history."""
    root = isolated.parent
    monkeypatch.setattr(a, "ROOT", root)
    (isolated / "engine.py").write_text("# synthetic source\n")
    for rel in (
        "research/mim_comparison/data.py",
        "research/mim_robustness/artifacts.py",
        "research/mim_robustness/__init__.py",
    ):
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic dependency\n")
    source_run = root / "source-run"
    source_run.mkdir()
    baseline = root / "baseline"
    baseline.mkdir()
    (source_run / "completion.json").write_text('{"source": "synthetic"}\n')
    (baseline / "completion.json").write_text('{"baseline": "synthetic"}\n')
    trades, daily, bars = source_frames()
    trades.drop(columns="trade_id").to_csv(source_run / "trades.csv", index=False)
    daily.to_csv(source_run / "daily.csv", index=False)
    data = root / "data.csv"
    bars.to_csv(data, index=False)
    spec, protocol = root / "spec.md", root / "protocol.md"
    spec.write_text(
        "status: in-progress\n<frozen-after-approval>frozen contract</frozen-after-approval>\n"
    )
    protocol.write_text("synthetic protocol\n")
    for key, value in (
        ("SOURCE_RUN", source_run),
        ("DATA", data),
        ("SPEC", spec),
        ("PROTOCOL", protocol),
        ("SOURCE_COMPLETION_HASH", a.digest(source_run / "completion.json")),
        ("DATA_HASH", a.digest(data)),
    ):
        monkeypatch.setattr(a, key, value)
    monkeypatch.setattr(a.original, "BASELINE", baseline)
    monkeypatch.setattr(a.original, "verify", Mock())
    monkeypatch.setattr(a.original, "verify_baseline", Mock())
    monkeypatch.setattr(a, "LOADED", a.source_hashes())
    return isolated


def test_real_binding_completed_verification_and_status_change(provenance):
    path = a.create()
    a.bind(path)
    a.original.verify.assert_called_once_with(a.SOURCE_RUN, complete=True)
    a.original.verify_baseline.assert_called_once_with(a.DATA, a.original.BASELINE)
    populate_success(path)
    a.seal(path)
    a.verify(path, complete=True)
    a.SPEC.write_text(
        a.SPEC.read_text().replace("in-progress", "complete") + "review: appended\n"
    )
    a.verify(path, complete=True)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("source", "Implementation source drift"),
        ("input", "Input drift"),
        ("snapshot", "Source snapshot drift"),
        ("frozen", "Frozen spec intent drift"),
        ("runtime", "Runtime/config drift"),
        ("protocol", "Spec/protocol drift"),
        ("inventory", "Completion inventory mismatch"),
    ],
)
def test_real_verification_detects_each_provenance_mutation(
    provenance, mutation, message
):
    path = a.create()
    a.bind(path)
    populate_success(path)
    a.seal(path)
    if mutation == "source":
        (provenance / "engine.py").write_text("# changed\n")
    elif mutation == "input":
        a.DATA.write_text(a.DATA.read_text() + "\n")
    elif mutation == "snapshot":
        target = path / "source" / next(iter(a.LOADED))
        target.chmod(0o644)
        target.write_text("# changed snapshot\n")
    elif mutation == "frozen":
        a.SPEC.write_text(
            a.SPEC.read_text().replace("frozen contract", "changed contract")
        )
    elif mutation == "runtime":
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime"]["python"] = "different-runtime"
        manifest_path.chmod(0o644)
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "protocol":
        a.PROTOCOL.write_text("changed protocol\n")
    else:
        target = path / "paths.csv"
        target.chmod(0o644)
        target.write_text("changed output\n")
    with pytest.raises(ValueError, match=message):
        a.verify(path, complete=mutation == "inventory")


@pytest.mark.parametrize("mutation", ["data", "completion", "upstream"])
def test_real_verify_inputs_rejects_pinned_or_upstream_drift(provenance, mutation):
    a.verify_inputs()
    if mutation == "data":
        a.DATA.write_text("changed input\n")
    elif mutation == "completion":
        (a.SOURCE_RUN / "completion.json").write_text("changed completion\n")
    else:
        a.original.verify.side_effect = ValueError("upstream source drift")
    with pytest.raises(ValueError):
        a.verify_inputs()


def test_failed_and_bound_incomplete_runs_cannot_verify_success(provenance):
    incomplete = a.create()
    a.bind(incomplete)
    with pytest.raises(ValueError, match="Missing successful outputs"):
        a.seal(incomplete)
    # Even an externally supplied matching completion inventory is not success.
    a.write_json(incomplete / "completion.json", {"sha256": a.inventory(incomplete)})
    with pytest.raises(ValueError, match="Missing successful outputs"):
        a.verify(incomplete, complete=True)
    failed = a.create()
    a.bind(failed)
    a.fail(failed, ValueError("synthetic failure"))
    with pytest.raises(ValueError, match="Failed invocation"):
        a.verify(failed, complete=True)
    with pytest.raises(ValueError, match="Failed invocation"):
        a.seal(failed)


def test_nested_completion_is_inventoried_and_tamper_detected(provenance):
    path = a.create()
    a.bind(path)
    populate_success(path)
    nested = path / "nested" / "completion.json"
    nested.parent.mkdir()
    nested.write_text("original nested evidence\n")
    a.seal(path)
    assert "nested/completion.json" in a.inventory(path)
    a.verify(path, complete=True)
    nested.chmod(0o644)
    nested.write_text("tampered nested evidence\n")
    with pytest.raises(ValueError, match="Completion inventory mismatch"):
        a.verify(path, complete=True)


def test_cli_real_verifiers_fail_after_analysis_and_remove_reports(
    provenance, monkeypatch
):
    from research.mim_lifecycle.analysis import select_inputs

    monkeypatch.setattr(
        cli,
        "select_inputs",
        lambda trades, daily: select_inputs(trades, daily, 1, 2, -2.24),
    )

    def report_then_drift(path, summary, life):
        write_report(path, summary, life)
        a.DATA.write_text(a.DATA.read_text() + "\n")

    monkeypatch.setattr(cli, "write_report", report_then_drift)
    with pytest.raises(ValueError, match="Input drift"):
        cli.run()
    path = next(a.RUNS.iterdir())
    assert (path / "failure.json").exists()
    assert not (path / "report.md").exists()
    assert not (path / "report.html").exists()
    assert (
        a.inventory(path)
        == json.loads((path / "completion.json").read_text())["sha256"]
    )
    with pytest.raises(ValueError, match="Failed invocation"):
        a.verify(path, complete=True)
