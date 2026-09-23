"""Synthetic prospective-design checks; never credentials or market data."""

from datetime import date, datetime, timedelta, timezone
import base64
import json
from pathlib import Path
from statistics import NormalDist
import socket

import pytest

from research.kronos_readiness import design, evidence
from research.kronos_readiness.__main__ import main


@pytest.fixture
def pack(tmp_path):
    source = tmp_path / "document.md"
    source.write_text("Synthetic documentary calendar assumption only.")
    value = {
        "schema_version": 1,
        "sources": [
            {
                "id": "doc",
                "path": "document.md",
                "sha256": evidence.sha(source.read_bytes()),
                "date": "2026-09-22",
                "url": "synthetic:document",
                "claim": "fixture only",
            }
        ],
        "calendar": None,
        "scenarios": [
            {
                "name": "assumed",
                "roll_dates": [],
                "outage_dates": [],
                "incomplete_dates": [],
                "calibration_sessions": 0,
            }
        ],
        "economic": None,
    }
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(value))
    return value, path


def test_offline_manifest_and_park(pack, tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a: pytest.fail("network"))
    output = tmp_path / "out"
    result = design.run(pack[1], output, "2026-09-23")
    assert result["status"] == "PARK_PENDING_EVIDENCE"
    assert result["admitted_sessions"] == 0
    assert {r["months"] for r in result["horizons"]} == {3, 6, 12}
    manifest = json.loads((output / "COMPLETE.json").read_text())
    for filename, digest in manifest["sha256"].items():
        assert evidence.sha((output / filename).read_bytes()) == digest
    for path in output.glob("*.json"):
        artifact = json.loads(path.read_text())
        assert artifact["strategy_test_permitted"] is False
        assert artifact["trading_authorized"] is False
    snapshot = json.loads((output / "source-doc.json").read_text())
    assert (
        evidence.sha(base64.b64decode(snapshot["original_bytes_base64"]))
        == pack[0]["sources"][0]["sha256"]
    )
    with pytest.raises(ValueError):
        design.run(pack[1], output)


@pytest.mark.parametrize(
    "name",
    [
        "data/prices.json",
        "sealed_holdout/x.md",
        ".env.json",
        "credentials.json",
        ".access_token",
        "cache/x.json",
    ],
)
def test_forbidden_pack_before_read(name, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("read"))
    with pytest.raises(ValueError):
        design.load_pack(tmp_path / name)


def test_tamper_and_alias(pack, tmp_path):
    (tmp_path / "document.md").write_text("changed")
    with pytest.raises(ValueError, match="fingerprint"):
        design.load_pack(pack[1])
    alias = tmp_path / "alias.json"
    alias.symlink_to(pack[1])
    with pytest.raises(ValueError, match="aliases"):
        design.load_pack(alias)


def test_unknown_fields_and_duplicates(pack):
    value, path = pack
    value["token_path"] = ".access_token"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        design.load_pack(path)
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        design.load_pack(path)


def test_frozen_drift_precedes_pack_read(tmp_path, monkeypatch):
    monkeypatch.setitem(
        evidence.FROZEN, "research/kronos_replay/engine.py", "bad"
    )
    with pytest.raises(ValueError, match="frozen"):
        design.run(tmp_path / ".access_token", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_warmup_horizon_and_disjoint_gaps(pack):
    value = pack[0]
    # 4 full days yield 104 bars. Day 5 needs 24 more and cannot
    # leave a four-bar horizon; day 6 qualifies with full context.
    row = design.scenarios(value, date(2026, 9, 23))[0]
    ledger = row["session_ledger"]
    assert [r["conditional_collection_eligible"] for r in ledger[:6]] == [
        False
    ] * 5 + [True]
    assert ledger[4]["context_before"] == 104
    assert ledger[4]["first_possible_decision_bar"] == 24
    scenario = value["scenarios"][0]
    scenario.update(
        roll_dates=["2026-10-01"],
        outage_dates=["2026-10-02"],
        incomplete_dates=["2026-10-02"],
        calibration_sessions=20,
    )
    row = design.scenarios(value, date(2026, 9, 23))[0]
    by_date = {r["date"]: r for r in row["session_ledger"]}
    assert by_date["2026-10-01"]["context_before"] == 0
    assert by_date["2026-10-02"]["causes_retained"] == ["outage", "incomplete"]
    assert by_date["2026-10-02"]["context_after"] == 0
    assert (
        sum(row["deductions"].values())
        + row["conditional_collection_sessions"]
        == row["weekday_ceiling"]
    )
    assert (
        row["conditional_evaluation_sessions"]
        == row["conditional_collection_sessions"] - 20
    )


def test_calendar_coverage_holiday_early_close(pack):
    value = pack[0]
    value["calendar"] = {
        "coverage_start": "2026-09-23",
        "coverage_end": "2026-12-23",
        "source_ids": ["doc"],
        "closed_dates": ["2026-09-24"],
        "early_close_dates": ["2026-09-25"],
    }
    rows = design.scenarios(value, date(2026, 9, 23))
    assert rows[0]["calendar_coverage_complete"]
    assert not rows[1]["calendar_coverage_complete"]
    assert rows[0]["session_ledger"][1]["full_15m_bars"] == 0
    assert rows[0]["session_ledger"][2]["full_15m_bars"] == 14
    assert rows[0]["session_ledger"][2]["context_after"] == 40


def test_next_full_session_and_explicit_start(pack, tmp_path):
    after = datetime(2026, 9, 22, 23, tzinfo=timezone.utc)
    before = datetime(2026, 9, 22, 12, tzinfo=timezone.utc)
    assert design.planning_start(after, None) == date(2026, 9, 23)
    assert design.planning_start(before, None) == date(2026, 9, 22)
    assert design.planning_start(
        datetime(2026, 9, 25, 20, tzinfo=timezone.utc), None
    ) == date(2026, 9, 28)
    result = design.run(pack[1], tmp_path / "out", "2027-01-04", now=after)
    assert result["planning_start"] == "2027-01-04"


def test_received_cutoff_revision_recovery_and_clocks():
    cutoff = datetime(2026, 9, 23, 14, tzinfo=timezone.utc)
    args = dict(
        received=cutoff - timedelta(seconds=2),
        completed=cutoff - timedelta(seconds=3),
        cutoff=cutoff,
        forecast_start=cutoff + timedelta(seconds=1),
        forecast_complete=cutoff + timedelta(seconds=5),
        fill_open=cutoff + timedelta(minutes=1),
        scheduled_flatten=cutoff + timedelta(minutes=5),
    )
    assert design.causal_check(**args)["synthetic_causal_order_valid"]
    assert not design.causal_check(
        **{**args, "received": cutoff + timedelta(seconds=1)}
    )["synthetic_causal_order_valid"]
    assert not design.causal_check(
        **{**args, "fill_open": args["forecast_complete"]}
    )["synthetic_causal_order_valid"]
    assert not design.causal_check(**args, clock_error_seconds=2)[
        "synthetic_causal_order_valid"
    ]
    for field in (
        "exposed",
        "missing_flatten",
        "incomplete_session",
        "unresolved_exposure",
    ):
        result = design.causal_check(**args, **{field: True})
        assert (
            result["record_retained"]
            and not result["synthetic_causal_order_valid"]
        )


def test_paired_dollar_power_never_admits(pack, tmp_path):
    value, path = pack
    economic = {
        "k_effect": 1,
        "incremental_effect": 1,
        "k_variance": 4,
        "m_variance": 9,
        "covariance": 3,
        "se_inflation": 1,
        "independent_evidence": {
            key: "doc" for key in design.ECONOMIC_KEYS | {"dependence"}
        },
    }
    checked = design.conditional_dollars(100, economic)
    assert checked["paired_variance"] == 7
    assert checked["marginal_powers"] == pytest.approx(
        [0.9988172507018026, 0.9655961814120477]
    )
    assert checked["marginal_powers"][1] == pytest.approx(
        NormalDist().cdf(10 / (7**0.5) - NormalDist().inv_cdf(0.975))
    )
    value["economic"] = economic
    path.write_text(json.dumps(value))
    result = design.run(path, tmp_path / "out", "2026-09-23")
    assert result["status"] == "PARK_PENDING_EVIDENCE"
    assert result["strategy_test_permitted"] is False
    assert result["horizons"][-1]["dollar_power"]["marginal_powers"][0] > 0.9


def test_cli_park_exit_and_no_credentials(pack, tmp_path, capsys):
    assert (
        main(
            [
                "design",
                "--source-pack",
                str(pack[1]),
                "--planning-start",
                "2026-09-23",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert (
        json.loads(capsys.readouterr().out)["status"]
        == "PARK_PENDING_EVIDENCE"
    )
    with pytest.raises(SystemExit):
        main(
            [
                "design",
                "--source-pack",
                str(pack[1]),
                "--output-dir",
                str(tmp_path / "out2"),
                "--token-path",
                ".access_token",
            ]
        )


@pytest.mark.parametrize("target", ["pack", "source"])
@pytest.mark.parametrize("kind", ["hardlink", "fifo", "oversize"])
def test_document_file_boundaries(pack, tmp_path, monkeypatch, target, kind):
    import os

    path = pack[1] if target == "pack" else tmp_path / "document.md"
    if kind == "hardlink":
        os.link(path, tmp_path / "alias.md")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        with path.open("wb") as stream:
            stream.truncate(design.MAX_DOCUMENT_BYTES + 1)
    original = os.open

    def no_forbidden_open(candidate, *args, **kwargs):
        assert Path(candidate) != path, "invalid file opened"
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_forbidden_open)
    with pytest.raises(ValueError, match="regular document"):
        design.load_pack(pack[1])


def test_document_read_is_bounded(tmp_path, monkeypatch):
    import os

    path = tmp_path / "document.md"
    path.write_text("synthetic")
    original = os.fdopen
    reads = []

    class Stream:
        def __init__(self, descriptor, mode):
            self.stream = original(descriptor, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(os, "fdopen", Stream)
    assert design.read_document(path) == b"synthetic"
    assert reads == [design.MAX_DOCUMENT_BYTES + 1]


@pytest.mark.parametrize("folder", ["market", "market_data", "market-data"])
def test_market_paths_refused_before_stat(tmp_path, monkeypatch, folder):
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("stat"))
    with pytest.raises(ValueError, match="forbidden"):
        design.read_document(tmp_path / folder / "document.json")


@pytest.mark.parametrize(
    "key", sorted(design.ECONOMIC_KEYS | {"se_inflation"})
)
@pytest.mark.parametrize("value", [True, False, float("inf"), float("nan")])
def test_economic_types(key, value):
    economic = {
        "k_effect": 1,
        "incremental_effect": 1,
        "k_variance": 4,
        "m_variance": 9,
        "covariance": 3,
        "se_inflation": 1,
        "independent_evidence": {
            k: "doc" for k in design.ECONOMIC_KEYS | {"dependence"}
        },
    }
    economic[key] = value
    with pytest.raises(ValueError, match="finite int/float"):
        design.conditional_dollars(100, economic)


def causal_fixture():
    cutoff = datetime(2026, 9, 23, 14, tzinfo=timezone.utc)
    return dict(
        received=cutoff - timedelta(seconds=2),
        completed=cutoff - timedelta(seconds=3),
        cutoff=cutoff,
        forecast_start=cutoff + timedelta(seconds=1),
        forecast_complete=cutoff + timedelta(seconds=5),
        fill_open=cutoff + timedelta(minutes=1),
        scheduled_flatten=cutoff + timedelta(minutes=5),
    )


def test_first_minute_flatten_and_delayed_forecast():
    args = causal_fixture()
    assert design.causal_check(**args)["synthetic_causal_order_valid"]
    for change in (
        {"completed": args["received"] + timedelta(seconds=1)},
        {"fill_open": args["fill_open"] + timedelta(minutes=1)},
        {"fill_open": args["fill_open"] + timedelta(seconds=1)},
        {"forecast_complete": args["scheduled_flatten"]},
        {"scheduled_flatten": args["fill_open"]},
        {
            "forecast_complete": args["scheduled_flatten"]
            - timedelta(seconds=1),
            "fill_open": args["scheduled_flatten"],
        },
    ):
        result = design.causal_check(**{**args, **change})
        assert not result["synthetic_causal_order_valid"]
        assert result["record_retained"]
        assert result["strategy_test_permitted"] is False
    delayed = {
        **args,
        "forecast_complete": args["cutoff"] + timedelta(seconds=65),
        "fill_open": args["cutoff"] + timedelta(minutes=2),
    }
    assert design.causal_check(**delayed)["synthetic_causal_order_valid"]


def test_causal_dst_fold_uses_utc():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    # 01:59 fold=0 precedes 01:00 fold=1 in real time.
    cutoff = datetime(2026, 11, 1, 1, 0, tzinfo=zone, fold=1)
    args = dict(
        completed=datetime(2026, 11, 1, 1, 59, 57, tzinfo=zone),
        received=datetime(2026, 11, 1, 1, 59, 58, tzinfo=zone),
        cutoff=cutoff,
        forecast_start=datetime(2026, 11, 1, 1, 0, 1, tzinfo=zone, fold=1),
        forecast_complete=datetime(2026, 11, 1, 1, 0, 5, tzinfo=zone, fold=1),
        fill_open=datetime(2026, 11, 1, 1, 1, tzinfo=zone, fold=1),
        scheduled_flatten=datetime(2026, 11, 1, 1, 5, tzinfo=zone, fold=1),
    )
    assert design.causal_check(**args)["synthetic_causal_order_valid"]
    args["received"] = args["received"].replace(fold=1)
    assert not design.causal_check(**args)["synthetic_causal_order_valid"]


def test_warmup_attribution_and_population_order(pack):
    value = pack[0]
    value["scenarios"][0].update(
        roll_dates=["2026-10-15"],
        outage_dates=["2026-11-02"],
        calibration_sessions=8,
    )
    row = design.scenarios(value, date(2026, 9, 23))[0]
    breakdown = row["warmup_breakdown_subset_of_deductions"]
    assert breakdown == {"initial": 5, "contract": 5, "gap": 5}
    assert sum(breakdown.values()) == row["deductions"]["warmup_or_horizon"]
    eligible = [
        r
        for r in row["session_ledger"]
        if r["conditional_collection_eligible"]
    ]
    assert all(
        r["hypothetical_population_role"] == "hypothetical_calibration"
        for r in eligible[:8]
    )
    assert all(
        r["hypothetical_population_role"] == "hypothetical_evaluation"
        for r in eligible[8:]
    )
    assert row["last_hypothetical_calibration_date"] == eligible[7]["date"]
    assert row["first_hypothetical_evaluation_date"] == eligible[8]["date"]
    assert (
        "before the first evaluation" in row["calibration_freeze_requirement"]
    )


def test_calendar_default_skips_holiday_and_early_close(pack, tmp_path):
    value, path = pack
    value["calendar"] = {
        "coverage_start": "2026-11-25",
        "coverage_end": "2027-11-30",
        "source_ids": ["doc"],
        "closed_dates": ["2026-11-26"],
        "early_close_dates": ["2026-11-27"],
    }
    path.write_text(json.dumps(value))
    result = design.run(
        path,
        tmp_path / "out",
        now=datetime(2026, 11, 25, 22, tzinfo=timezone.utc),
    )
    assert (
        result["planning_start"]
        == result["next_full_assumed_rth"]
        == "2026-11-30"
    )
    report = (tmp_path / "out/report.md").read_text()
    assert "(comparison.md)" in report and "GENERIC readiness" in report
    assert (
        "historical replay metadata only"
        in design.protocol()["availability_precedence"]
    )
