"""Fixture-only wrapper verification: never launch the live adapter or shadow."""

import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "research/mim_comparison/runs"
SCRIPT = ROOT / "research/mim_comparison/poll.sh"
HEADER = b"contract,timestamp,open,high,low,close,volume,received_at,evidence\n"


@pytest.fixture
def fixture():
    RUNS.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="test-poll-window-", dir=RUNS))
    source = directory / "adapter"
    source.mkdir()
    feed = source / "feed.csv"
    state = directory / "wrapper"
    yield feed, state
    shutil.rmtree(directory)


def install(feed, count, partial=b""):
    records = [
        f'MNQU26,2026-09-10T14:00:00+00:00,{index},102,99,101,10,2026-09-10T14:00:01+00:00,"evidence,{index}"\n'.encode()
        for index in range(count)
    ]
    data = HEADER + b"".join(records)
    feed.write_bytes(data + partial)
    with sqlite3.connect(feed.parent / "journal.sqlite") as db:
        db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
        db.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)",
            (
                "feed",
                json.dumps(
                    {
                        "id": count,
                        "size": len(data),
                        "hash": hashlib.sha256(data).hexdigest(),
                    }
                ),
            ),
        )
    return records


def run(feed, state, script=SCRIPT):
    return subprocess.run(
        [
            "bash",
            str(script),
            "--prepare-only",
            "--feed",
            str(feed),
            "--state",
            str(state),
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("count", [0, 1, 499, 500, 501, 2000])
def test_exact_header_and_last_500(fixture, count):
    feed, state = fixture
    rows = install(feed, count)
    before = feed.read_bytes()
    result = run(feed, state)
    assert result.returncode == 0, result.stderr
    assert (state / "window.csv").read_bytes() == HEADER + b"".join(rows[-500:])
    assert feed.read_bytes() == before
    with open(state / "window.csv") as stream:
        assert len(list(csv.DictReader(stream))) == min(count, 500)
    manifest = json.loads((state / "window-manifest.json").read_text())
    assert manifest["rows"] == min(count, 500)
    assert (
        manifest["window_sha256"]
        == hashlib.sha256((state / "window.csv").read_bytes()).hexdigest()
    )


def test_partial_tail_pending_and_atomic_replacement(fixture):
    feed, state = fixture
    install(feed, 1)
    assert run(feed, state).returncode == 0
    with open(state / "window.csv", "rb") as old_handle:
        old_bytes = old_handle.read()
        rows = install(feed, 501, partial=b"MNQU26,incomplete")
        assert run(feed, state).returncode == 0
        old_handle.seek(0)
        assert old_handle.read() == old_bytes
    assert (state / "window.csv").read_bytes() == HEADER + b"".join(rows[-500:])
    assert json.loads((state / "window-manifest.json").read_text())[
        "pending_partial_bytes"
    ] == len(b"MNQU26,incomplete")
    assert not list(state.glob(".window.csv-*"))


def test_changed_prefix_does_not_replace_window(fixture):
    feed, state = fixture
    install(feed, 2)
    assert run(feed, state).returncode == 0
    before = (state / "window.csv").read_bytes()
    feed.write_bytes(feed.read_bytes().replace(b"MNQU26", b"MNQZ26"))
    result = run(feed, state)
    assert result.returncode != 0 and "hash mismatch" in result.stderr
    assert (state / "window.csv").read_bytes() == before


def test_complete_uncommitted_tail_requires_recovery(fixture):
    feed, state = fixture
    install(feed, 1, partial=b"not,committed\n")
    result = run(feed, state)
    assert result.returncode != 0 and "recovery" in result.stderr
    assert not (state / "window.csv").exists()


def test_committed_partial_row_refused(fixture):
    feed, state = fixture
    install(feed, 1)
    data = feed.read_bytes()[:-1]
    feed.write_bytes(data)
    with sqlite3.connect(feed.parent / "journal.sqlite") as db:
        db.execute(
            "UPDATE meta SET value=? WHERE key=?",
            (
                json.dumps(
                    {
                        "id": 1,
                        "size": len(data),
                        "hash": hashlib.sha256(data).hexdigest(),
                    }
                ),
                "feed",
            ),
        )
    result = run(feed, state)
    assert result.returncode != 0 and "partial" in result.stderr


def test_wrapper_and_adapter_locks(fixture):
    feed, state = fixture
    install(feed, 1)
    assert run(feed, state).returncode == 0
    for path in (state / "lock", feed.parent / "lock"):
        with open(path, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert run(feed, state).returncode != 0


def test_frozen_source_hash_drift_refused(fixture):
    feed, state = fixture
    install(feed, 1)
    assert run(feed, state).returncode == 0
    frozen = json.loads((state / "freeze.json").read_text())
    frozen["wrapper_sha256"] = "changed"
    (state / "freeze.json").write_text(json.dumps(frozen))
    result = run(feed, state)
    assert result.returncode != 0 and "drift" in result.stderr


def test_writes_outside_runs_refused(fixture, tmp_path):
    feed, _ = fixture
    install(feed, 1)
    forbidden = tmp_path / "forbidden"
    result = run(feed, forbidden)
    assert result.returncode != 0
    assert not forbidden.exists()


def test_collector_preflight_indexes_without_changing_rows(fixture):
    feed, state = fixture
    install(feed, 1)
    journal = state.parent / "collector.sqlite"
    with sqlite3.connect(journal) as db:
        db.execute(
            "CREATE TABLE invalid_rows(identity TEXT PRIMARY KEY,event TEXT,contract TEXT,raw TEXT)"
        )
        db.execute("CREATE TABLE observations(event TEXT,decision TEXT)")
        db.execute(
            "INSERT INTO invalid_rows VALUES (?,?,?,?)",
            ("one", "2026-09-10", "MNQU26", "original"),
        )
        db.execute("INSERT INTO observations VALUES (?,?)", ("original", "unchanged"))
        before = list(db.execute("SELECT * FROM invalid_rows"))
        observations = list(db.execute("SELECT * FROM observations"))
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--prepare-only",
            "--feed",
            str(feed),
            "--state",
            str(state),
            "--collector-journal",
            str(journal),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with sqlite3.connect(journal) as db:
        plan = str(
            db.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM invalid_rows WHERE event=? AND (contract=? OR contract IS NULL OR contract=?)",
                ("event", "MNQU26", ""),
            ).fetchall()
        )
        assert "mim_feed_event_contract" in plan and "SEARCH" in plan
        assert list(db.execute("SELECT * FROM invalid_rows")) == before
        assert list(db.execute("SELECT * FROM observations")) == observations


def test_collector_preflight_missing_table_is_noop(fixture):
    feed, state = fixture
    install(feed, 1)
    journal = state.parent / "empty.sqlite"
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE observations(event TEXT)")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--prepare-only",
            "--feed",
            str(feed),
            "--state",
            str(state),
            "--collector-journal",
            str(journal),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with sqlite3.connect(journal) as db:
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()


def test_live_custom_state_refused_before_any_collector(fixture):
    _, state = fixture
    result = subprocess.run(
        ["bash", str(SCRIPT), "--state", str(state)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "custom --state" in result.stderr
    assert not state.exists()


@pytest.fixture
def harness(fixture, monkeypatch):
    """Execute the shipped embedded Python with all operational paths isolated."""
    import sys
    from types import SimpleNamespace

    feed, state = fixture
    monkeypatch.setattr(sys, "argv", ["poll-harness", str(SCRIPT)])
    source = SCRIPT.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    namespace = {}
    exec(
        compile(source.rsplit("\ntry:\n    main()", 1)[0], str(SCRIPT), "exec"),
        namespace,
    )
    namespace.update(
        ROOT=state.parent,
        RUNS=state.parent,
        ADAPTER=feed.parent,
        LIVE_STATE=state,
        PREPARE_STATE=state.parent / "prepare",
        COLLECTOR=state.parent / "collector",
        HISTORY=state.parent / "history",
        WARMUP=state.parent / "warmup.csv",
        subprocess=SimpleNamespace(
            run=lambda *a, **kw: pytest.fail("unstubbed collector")
        ),
    )
    yield namespace


def test_prepare_default_is_separate_and_live_rejected_before_writes(
    harness, monkeypatch
):
    import sys

    feed = harness["ADAPTER"] / "feed.csv"
    install(feed, 1)
    monkeypatch.setattr(sys, "argv", ["poll", "--prepare-only"])
    harness["main"]()
    assert (harness["PREPARE_STATE"] / "window.csv").exists()
    assert not harness["LIVE_STATE"].exists()
    monkeypatch.setattr(
        sys, "argv", ["poll", "--prepare-only", "--state", str(harness["LIVE_STATE"])]
    )
    with pytest.raises(SystemExit):
        harness["main"]()
    assert not harness["LIVE_STATE"].exists()


@pytest.mark.parametrize(
    "definition",
    [
        "ON invalid_rows(contract,event)",
        "ON observations(event,contract)",
        "ON invalid_rows(event,contract) WHERE event IS NOT NULL",
    ],
)
def test_preflight_rejects_incompatible_existing_index(harness, definition):
    journal = harness["RUNS"] / "bad-index.sqlite"
    with sqlite3.connect(journal) as db:
        db.execute("CREATE TABLE invalid_rows(event TEXT, contract TEXT)")
        db.execute("CREATE TABLE observations(event TEXT, contract TEXT)")
        db.execute("CREATE INDEX mim_feed_event_contract " + definition)
    with pytest.raises(ValueError, match="incompatible"):
        harness["index_collector"](journal)


@pytest.mark.parametrize(
    "failure", [None, "adapter", "index", "prepare", "validate", "shadow"]
)
def test_live_orchestration_order_arguments_failure_and_lock(harness, failure):
    feed = harness["ADAPTER"] / "feed.csv"
    install(feed, 1)
    harness["COLLECTOR"].mkdir()
    with sqlite3.connect(harness["COLLECTOR"] / "journal.sqlite") as db:
        db.execute("CREATE TABLE invalid_rows(event TEXT,contract TEXT)")
    events = []

    def stage(name):
        events.append(name)
        with open(harness["LIVE_STATE"] / "lock", "a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if name == failure:
            raise subprocess.CalledProcessError(7, name)

    def dispatch(argv, **kwargs):
        name = "adapter" if argv[2].endswith("feed_adapter") else "shadow"
        stage(name)
        assert kwargs == {"cwd": harness["ROOT"], "check": True}
        if name == "adapter":
            assert argv[1:] == [
                "-m",
                "research.mim_comparison.feed_adapter",
                "--bars",
                str(harness["ROOT"] / "data/mim_nb/bars_raw.csv"),
                "--log",
                str(harness["ROOT"] / "logs/mim_nb_live.log"),
                "--log-timezone",
                "UTC",
                "--state",
                str(harness["ADAPTER"]),
            ]
        else:
            assert argv[1:] == [
                "-m",
                "research.mim_comparison",
                "shadow",
                "--data",
                str(harness["LIVE_STATE"] / "window.csv"),
                "--labels",
                "end",
                "--warmup",
                str(harness["WARMUP"]),
                "--historical-run",
                str(harness["HISTORY"]),
                "--state",
                str(harness["COLLECTOR"]),
            ]

    harness["subprocess"].run = dispatch
    for function, name in [
        ("index_collector", "index"),
        ("prepare", "prepare"),
        ("validate_window", "validate"),
    ]:
        original = harness[function]

        def wrapped(*args, original=original, name=name):
            stage(name)
            return original(*args)

        harness[function] = wrapped
    expected = ["adapter", "index", "prepare", "validate", "shadow"]
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            harness["main"]()
        expected = expected[: expected.index(failure) + 1]
    else:
        harness["main"]()
    assert events == expected
    with open(harness["LIVE_STATE"] / "lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "fault", ["between_publications", "window_tamper", "manifest_tamper"]
)
def test_publication_fault_prevents_dispatch_and_restart_recovers(harness, fault):
    feed, state = harness["ADAPTER"] / "feed.csv", harness["LIVE_STATE"]
    install(feed, 1)
    state.mkdir()
    harness["prepare"](feed, state)
    install(feed, 2)
    harness["COLLECTOR"].mkdir()
    sqlite3.connect(harness["COLLECTOR"] / "journal.sqlite").close()
    dispatched = []
    harness["subprocess"].run = lambda argv, **kw: dispatched.append(argv[2])
    atomic = harness["atomic_bytes"]

    def broken(path, content):
        if path.name == "window-manifest.json" and fault == "between_publications":
            raise OSError("simulated crash between publications")
        atomic(path, content)
        if path.name == "window-manifest.json":
            if fault == "window_tamper":
                (state / "window.csv").write_bytes(b"tampered")
            else:
                changed = json.loads(path.read_text())
                changed["rows"] = -1
                path.write_text(json.dumps(changed))

    harness["atomic_bytes"] = broken
    with pytest.raises((OSError, ValueError)):
        harness["main"]()
    assert dispatched == ["research.mim_comparison.feed_adapter"]
    harness["atomic_bytes"] = atomic
    harness["main"]()
    assert dispatched[-1] == "research.mim_comparison"
    manifest = json.loads((state / "window-manifest.json").read_text())
    harness["validate_window"](state, manifest)


def test_two_real_collector_windows_retain_prior_observations(harness, monkeypatch):
    import pandas as pd
    from research.mim_comparison import shadow
    from research.mim_comparison.artifacts import BASE, digest
    from test_comparison import session

    feed, state = harness["ADAPTER"] / "feed.csv", harness["LIVE_STATE"]
    state.mkdir()
    collector, output = harness["COLLECTOR"], harness["RUNS"] / "output"
    collector.mkdir()
    output.mkdir()
    warm = pd.concat(
        [
            session(str(day.date())).bars
            for day in pd.bdate_range("2026-08-03", "2026-08-28")
        ]
    )
    warm.to_csv(harness["WARMUP"], index=False)
    (collector / "freeze.json").write_text(
        json.dumps(
            {
                "protocol": {"freeze": "2026-09-08T22:00:00+00:00"},
                "warmup_hash": digest(harness["WARMUP"]),
                "labels": "end",
                "source": {p.name: digest(p) for p in BASE.glob("*.py")},
            }
        )
    )
    monkeypatch.setattr(shadow, "_now", lambda: pd.Timestamp("2026-09-10T20:02:00Z"))
    monkeypatch.setattr(shadow, "install_socket_filter", lambda: None)
    stamps = list(
        pd.date_range("2026-09-09T13:31:00Z", periods=390, freq="min")
    ) + list(pd.date_range("2026-09-10T13:31:00Z", periods=111, freq="min"))
    records = [
        f"MNQU26,{stamp.isoformat()},100,102,99,101,10,{(stamp + pd.Timedelta(seconds=1)).isoformat()},fixture\n".encode()
        for stamp in stamps
    ]
    first = None
    for count in (500, 501):
        data = HEADER + b"".join(records[:count])
        install(feed, 0)
        feed.write_bytes(data)
        with sqlite3.connect(feed.parent / "journal.sqlite") as db:
            db.execute(
                "UPDATE meta SET value=? WHERE key='feed'",
                (
                    json.dumps(
                        {
                            "id": count,
                            "size": len(data),
                            "hash": hashlib.sha256(data).hexdigest(),
                        }
                    ),
                ),
            )
        manifest = harness["prepare"](feed, state)
        harness["validate_window"](state, manifest)
        if count == 501:
            assert records[0] not in (state / "window.csv").read_bytes()
            harness["index_collector"](collector / "journal.sqlite")
        shadow.collect(
            {
                "state": str(collector),
                "warmup": str(harness["WARMUP"]),
                "data": str(state / "window.csv"),
                "labels": "end",
                "result": str(output / f"status-{count}.json"),
            }
        )
        with sqlite3.connect(collector / "journal.sqlite") as db:
            rows = db.execute("SELECT * FROM observations ORDER BY event").fetchall()
            assert len(rows) == count
            assert db.execute("SELECT COUNT(*) FROM invalid_rows").fetchone()[0] == 0
        if first is None:
            first = rows
        else:
            assert rows[:500] == first
            assert all(
                row[-2:] == (0, "late_backfilled_or_future_receipt") for row in rows
            )
