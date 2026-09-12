from copy import deepcopy
import sqlite3
import pytest
from research.mim_diagnostics import artifacts as a, analysis as n, feasibility as f


def valid_inventory():
    rows = []
    for category in sorted(f.REQUIRED_CATEGORIES):
        row = dict(
            category=category,
            status="unavailable",
            path=None,
            coverage_start=None,
            coverage_end=None,
            missing_evidence="Specific missing evidence",
            observation_type="synthetic source observation",
        )
        if category in f.CALENDARS:
            row.update(
                path=f.CALENDARS[category][0],
                coverage_start="2026-01-01",
                coverage_end="2026-01-02",
                rows=2,
                columns=["date", "time_et"],
                exact_duplicates=0,
                date_duplicates=0,
                timezone="unestablished",
                official_support="No causal original availability evidence",
            )
        if category == "mim_prospective":
            row.update(
                observed_at="2026-09-12T00:00Z", collection_note="unavailable journal"
            )
            f.set_eligibility_status(row)
        if category == "fomc_prospective":
            row.update(observed_at="2026-09-12T00:00Z", exists=False, rows=0)
        rows.append(row)
    return dict(
        categories=rows,
        observed_at="2026-09-12T00:00Z",
        official_checks=deepcopy(f.OFFICIAL),
        correction={
            key: "Preserved verdict/protocol"
            for key in ("policy_throttle", "impulse_following", "fomc_fade", "mim")
        },
        documents=f.DOCS,
        local_inventory_scope="bounded local scan",
        archive_scan=dict(
            observed_at="2026-09-12T00:00Z", scope="headers only", entries=[]
        ),
    )


@pytest.mark.parametrize(
    "bad",
    [
        "official_bool",
        "official_row",
        "correction_bool",
        "correction_key",
        "calendar_clock",
        "calendar_columns",
        "calendar_dates",
        "archive",
        "journal",
        "coverage",
    ],
)
def test_category_specific_schema_rejects_malformed(bad):
    value = valid_inventory()
    f.validate_inventory(value)
    calendar = next(
        r for r in value["categories"] if r["category"] == "economic_calendar"
    )
    if bad == "official_bool":
        value["official_checks"] = True
    elif bad == "official_row":
        value["official_checks"] = [{"url": 1, "checked": True, "support": []}]
    elif bad == "correction_bool":
        value["correction"] = True
    elif bad == "correction_key":
        del value["correction"]["mim"]
    elif bad == "calendar_clock":
        del calendar["timezone"]
    elif bad == "calendar_columns":
        calendar["columns"] = True
    elif bad == "calendar_dates":
        del calendar["coverage_start"]
    elif bad == "archive":
        del value["archive_scan"]
    elif bad == "journal":
        next(r for r in value["categories"] if r["category"] == "mim_prospective")[
            "eligibility_status"
        ] = "target_reached"
    else:
        calendar["coverage_start"] = []
    with pytest.raises(ValueError):
        f.validate_inventory(value)


@pytest.mark.parametrize(
    "count,status",
    [
        (0, "accumulating"),
        (119, "accumulating"),
        (120, "target_reached"),
        (121, "target_reached"),
    ],
)
def test_eligibility_status_derived_from_count(count, status):
    row = {"eligible_sessions": count}
    f.set_eligibility_status(row)
    assert row["eligibility_status"] == status
    assert f"{count} eligible" in row["missing_evidence"]
    assert "no efficacy analysis" in row["missing_evidence"]
    assert "not accumulated" not in row["missing_evidence"]


def test_all_status_queries_use_one_read_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ORIGINAL", tmp_path)
    path = tmp_path / "journal.sqlite"
    connect = sqlite3.connect
    with connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            "CREATE TABLE observations(event TEXT,collected TEXT,available INTEGER)"
        )
        db.execute("CREATE TABLE sessions(eligible INTEGER)")
        db.execute("INSERT INTO observations VALUES('2026-01-01','2026-01-02',0)")
    commands = []

    class Wrapper:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.db.rollback()
            self.db.close()

        def execute(self, query):
            commands.append(query)
            cursor = self.db.execute(query)
            if query == 'SELECT count(*) FROM "observations"':
                with connect(path) as writer:
                    writer.execute(
                        "INSERT INTO observations VALUES('2026-01-03','2026-01-04',1)"
                    )
                    writer.execute("INSERT INTO sessions VALUES(1)")
            return cursor

    monkeypatch.setattr(
        f.sqlite3, "connect", lambda *args, **kwargs: Wrapper(connect(*args, **kwargs))
    )
    result = f.observe_journal(path)
    assert commands[0] == "BEGIN"
    assert result["table_counts"]["observations"] == 1
    assert result["table_counts"]["sessions"] == 0
    assert result["eligible_sessions"] == 0 and result["available_observations"] == 0
    assert result["coverage_end"] == "2026-01-01"
    with connect(path) as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1


@pytest.mark.parametrize(
    "reader", ["digest", "csv", "json", "analysis", "journal", "inventory", "calendar"]
)
def test_actual_reader_rejects_resolved_holdout(tmp_path, monkeypatch, reader):
    monkeypatch.setattr(a, "ORIGINAL", tmp_path)
    monkeypatch.setattr(f, "ORIGINAL", tmp_path)
    forbidden = tmp_path / "data" / "sealed_holdout" / "secret"
    alias = tmp_path / "alias"
    alias.symlink_to(forbidden)
    if reader == "digest":
        call = lambda: a.digest(alias)
    elif reader == "csv":
        call = lambda: a.read_csv(alias)
    elif reader == "json":
        call = lambda: a.read_json(alias)
    elif reader == "analysis":
        call = lambda: n.analyze(alias, alias)
    elif reader == "journal":
        call = lambda: f.observe_journal(alias)
    elif reader == "inventory":
        call = lambda: a.verify_inventory(alias)
    else:
        path = tmp_path / next(iter(f.CALENDARS.values()))[0]
        path.parent.mkdir(parents=True)
        path.symlink_to(forbidden)
        call = f.build_inventory
    with pytest.raises(ValueError, match="Holdout"):
        call()


def test_source_completion_symlink_refused_before_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ORIGINAL", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "completion.json").symlink_to(tmp_path / "data/sealed_holdout/secret")
    with pytest.raises(ValueError, match="Holdout"):
        a.validate_inputs(source, tmp_path / "bars.csv")


@pytest.mark.parametrize(
    "relative", ["../outside", "/etc/passwd", "folder/../../outside"]
)
def test_snapshot_key_escape_is_rejected(tmp_path, relative):
    with pytest.raises(ValueError, match="snapshot"):
        a.child(tmp_path, relative)


def test_inventory_rejects_absolute_completion_key(tmp_path):
    a.json_write(tmp_path / "completion.json", {"sha256": {"/etc/passwd": "anything"}})
    with pytest.raises(ValueError, match="snapshot"):
        a.verify_inventory(tmp_path)


def synthetic_audit(tmp_path, monkeypatch, omitted_source=None, omitted_protocol=None):
    original = tmp_path / "original"
    original.mkdir()
    monkeypatch.setattr(a, "ORIGINAL", original)
    monkeypatch.setattr(a, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(a, "validate_inputs", lambda *args: None)
    run = a.create("audit")
    source = original / "source-run"
    data = original / "bars.csv"
    input_paths = [data, source / "completion.json"] + [
        source / n
        for n in (
            "manifest.json",
            "ledger.csv",
            "trades.csv",
            "daily.csv",
            "exclusions.csv",
        )
    ]
    input_paths += [original / r[0] for r in f.CALENDARS.values()] + [
        original / rel for rel in f.DOCS + f.PROTOCOLS
    ]
    manifest = dict(
        command="audit", source_run=str(source), data=str(data), inputs={}, source={}
    )
    for p in input_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("frozen input")
        if str(p.relative_to(original)) == omitted_protocol:
            continue
        manifest["inputs"][str(p)] = a.digest(p)
        if p != data:
            target = run / "inputs" / p.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("frozen input")
    sources = {
        "research/mim_diagnostics/" + name
        for name in (
            "__init__.py",
            "__main__.py",
            "analysis.py",
            "artifacts.py",
            "feasibility.py",
            "report.py",
            "definitions.md",
        )
    } | {
        "research/mim_comparison/data.py",
        "research/mim_lifecycle/analysis.py",
        "AGENTS.md",
    }
    for rel in sources:
        if rel == omitted_source:
            continue
        target = run / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("frozen source")
        manifest["source"][rel] = a.digest(target)
    for name in a.REQUIRED - {"completion.json", "manifest.json"}:
        (run / name).write_text("{}")
    manifest["definitions_sha256"] = a.digest(run / "definitions.md")
    a.json_write(run / "manifest.json", manifest)
    a.seal(run)
    return run


@pytest.mark.parametrize(
    "omitted",
    [
        "research/mim_comparison/data.py",
        "research/mim_lifecycle/analysis.py",
        "AGENTS.md",
    ],
)
def test_dependency_omitted_from_manifest_and_disk_still_rejected(
    tmp_path, monkeypatch, omitted
):
    run = synthetic_audit(tmp_path, monkeypatch, omitted_source=omitted)
    a.verify_inventory(run)
    with pytest.raises(ValueError, match="source inventory"):
        a.verify(run)


@pytest.mark.parametrize("omitted", f.PROTOCOLS)
def test_protocol_omitted_from_manifest_and_disk_still_rejected(
    tmp_path, monkeypatch, omitted
):
    run = synthetic_audit(tmp_path, monkeypatch, omitted_protocol=omitted)
    a.verify_inventory(run)
    with pytest.raises(ValueError, match="evidence input inventory"):
        a.verify(run)
