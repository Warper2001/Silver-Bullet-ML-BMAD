from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from research.project_goals.audit import reconcile
from research.project_goals.portfolio import marked_curves, portfolio_report
from research.project_goals.accounts import model_path, sensitivity
from research.project_goals.common import input_path, digest, write_json, timestamp
from research.project_goals.power import (
    sweep,
    standalone_sweep,
    validate_registration,
    source_hashes,
)
from research.project_goals import scheduler
from research.project_goals.archive import archive_chunks, restore_chunks


def fill(**kw):
    row = dict(
        id=1,
        accountId=2,
        contractId="MNQU26",
        creationTimestamp="2026-08-13T14:00:00Z",
        price=100.0,
        profitAndLoss=None,
        fees=0.36,
        commissions=0.25,
        side=0,
        size=1,
        orderId=10,
    )
    row.update(kw)
    return row


def test_partial_duplicate_epoch_and_cost():
    report, rows = reconcile(
        [fill(), fill(), fill(id=2, size=2, fees=None), fill(id=3, accountId=3)],
        account=2,
        epoch_start="2026-08-13T00:00:00Z",
    )
    assert len(rows) == 3 and report["fills"] == 2
    assert report["missing_cost_records"] == 1
    assert report["snapshot_net_reported_realized_less_all_fill_costs"] is None
    assert rows[1]["quantity"] == 2
    assert reconcile([fill()], epoch_start="2026-08-14T00:00:00Z")[0]["fills"] == 0


def test_conflicting_duplicate_blocks_net():
    report, _ = reconcile([fill(profitAndLoss=2), fill(price=200)])
    assert "conflicting_duplicate_fill" in report["issues"]
    assert report["snapshot_net_reported_realized_less_all_fill_costs"] is None


def evidence():
    coverage = dict(
        start="2026-08-13T14:00:00Z",
        end="2026-08-13T14:02:00Z",
        initial_flat=True,
        complete_export=True,
        strategies=["MIM", "YANK"],
        baseline_units={"MIM": 1, "YANK": 2},
        grid_seconds=60,
    )
    fills = [
        dict(
            id=1,
            timestamp=coverage["start"],
            account="2",
            epoch="a",
            strategy="MIM",
            contract="MNQU26",
            signed_quantity=1,
            price=100.0,
            actual_cost=0.61,
        ),
        dict(
            id=2,
            timestamp=coverage["end"],
            account="2",
            epoch="a",
            strategy="MIM",
            contract="MNQU26",
            signed_quantity=-1,
            price=102.0,
            actual_cost=0.61,
        ),
    ]
    marks = [
        dict(
            timestamp="2026-08-13T14:0" + str(i) + ":00Z",
            contract="MNQU26",
            close=price,
            multiplier=2.0,
            corrected_contract=True,
        )
        for i, price in enumerate((100.0, 90.0, 102.0))
    ]
    return fills, marks, coverage


def test_marked_intraday_drawdown_and_actual_fees():
    fills, marks, coverage = evidence()
    status, rows = marked_curves(fills, marks, coverage)
    assert status["status"] == "DESCRIPTIVE_ONLY"
    assert rows[-2]["cumulative_net"] == pytest.approx(2.78)
    result = portfolio_report(rows)
    assert result["scenarios"]["actual"]["intraday_path"][
        "max_marked_drawdown"
    ] == pytest.approx(20.61)
    assert result["scenarios"]["actual"]["metrics"]["net"] == pytest.approx(2.78)


def test_missing_mark_never_flat():
    fills, marks, coverage = evidence()
    marks[1]["contract"] = "MNQZ26"
    status, rows = marked_curves(fills, marks, coverage)
    assert status["status"] == "INSUFFICIENT_DATA"
    assert rows[2]["cumulative_net"] is None
    assert portfolio_report(rows)["status"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize("change", ["missing_cost", "epoch", "gap", "uncertified"])
def test_unusable_evidence(change):
    fills, marks, coverage = evidence()
    if change == "missing_cost":
        fills[0]["actual_cost"] = None
    if change == "epoch":
        fills[-1]["epoch"] = "reset"
    if change == "gap":
        marks.pop(1)
    if change == "uncertified":
        coverage["initial_flat"] = False
    assert marked_curves(fills, marks, coverage)[0]["status"] == "INSUFFICIENT_DATA"


def test_shared_account_rows_are_not_strategy_curves():
    from research.project_goals.audit import shared_equity

    assert shared_equity([])["status"] == "SHARED_ACCOUNT_OBSERVATIONS_ONLY"


def test_account_floor_intraday_and_direct_capital():
    rows = [
        dict(session="2026-08-13", net_pnl=500, intraday_min_pnl=-2000, traded=True)
    ]
    assert model_path(rows)["rows"][0]["status"] == "BREACHED"
    rows = [
        dict(session="2026-08-13", net_pnl=-6000, intraday_min_pnl=-6000, traded=True)
    ]
    result = sensitivity(rows)
    assert result["direct_capital_5000.0"]["rows"][0]["status"] == "INSOLVENT"
    assert result["direct_capital_10000.0"]["end_balance"] == 4000
    assert result["direct_capital_10000.0"]["operating_cost"] == 0


def test_combine_consistency_prevents_one_day_pass():
    row = dict(session="2026-08-13", net_pnl=3500, intraday_min_pnl=0, traded=True)
    assert model_path([row])["ending_phase"] == "combine"


def test_missing_intraday_path_refuses_account_claim():
    assert (
        model_path([dict(session="2026-08-13", net_pnl=1)])["status"]
        == "INCOMPLETE_OR_STOPPED"
    )


def test_no_data_power_is_honest():
    assert sweep(None)["status"] == "INSUFFICIENT_DATA"


def test_standalone_sweep_reproducible(tmp_path):
    p = tmp_path / "trades.csv"
    p.write_text("pnl_usd\n-100\n200\n-50\n150\n20\n30\n")
    assert standalone_sweep(p) == standalone_sweep(p)
    assert standalone_sweep(p)["joint_portfolio_status"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize(
    "bad",
    [
        dict(blocks=[]),
        dict(cost_stress=[]),
        dict(cost_stress=[0]),
        dict(draws=0),
        dict(alpha=float("nan")),
    ],
)
def test_invalid_registration_refused(bad):
    reg = dict(
        blocks=[5],
        cost_stress=[0, 1],
        draws=128,
        seed=7,
        required_sessions=120,
        alpha=0.05,
        allocations={"MIM": 1, "YANK": 2, "GAP": 1},
        source_hashes=source_hashes(),
    )
    reg.update(bad)
    with pytest.raises(ValueError):
        validate_registration(reg)


def test_holdout_refused_before_open():
    with pytest.raises(ValueError):
        input_path("/tmp/data/sealed_holdout/secret.csv")


def snapshot(tmp_path, name):
    runs = tmp_path / "runs"
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "blob").write_bytes(b"abc" * 1000000)
    write_json(
        directory / "completion.json", dict(sha256={"blob": digest(directory / "blob")})
    )
    return runs, directory


def test_archive_roundtrip_and_dedup(tmp_path):
    runs, directory = snapshot(tmp_path, "20260919-shadow-one")
    archives = tmp_path / "archives"
    result = archive_chunks(directory, runs, archives, set())
    assert not directory.exists()
    restore_chunks(result["manifest"], tmp_path / "restored")
    assert (tmp_path / "restored/blob").read_bytes() == b"abc" * 1000000
    _, second = snapshot(tmp_path, "20260919-shadow-two")
    result2 = archive_chunks(second, runs, archives, set())
    assert result2["new_compressed_bytes"] == 0


def test_archive_refuses_old_or_changed_snapshot(tmp_path):
    runs, directory = snapshot(tmp_path, "20260919-shadow-one")
    with pytest.raises(ValueError):
        archive_chunks(directory, runs, tmp_path / "archive", {directory})
    (directory / "blob").write_text("changed")
    with pytest.raises(ValueError):
        archive_chunks(directory, runs, tmp_path / "archive", set())
    assert directory.exists()


def test_prefix_drift(tmp_path):
    p = tmp_path / "feed"
    p.write_bytes(b"abc")
    scheduler.prefix(p, dict(size=3, hash=digest(p)))
    with pytest.raises(ValueError):
        scheduler.prefix(p, dict(size=3, hash="bad"))


def test_poll_failure_recovers_and_latency(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "preflight", lambda root: dict(status="VERIFIED"))
    monkeypatch.setattr(scheduler, "health", lambda root: dict(eligible_sessions=0))
    ticks = iter([0, 65])
    result = scheduler.poll_once(
        tmp_path,
        tmp_path / "state",
        runner=lambda *a, **k: SimpleNamespace(returncode=1),
        clock=lambda: next(ticks),
    )
    assert result["status"] == "POLL_FAILED" and result["latency_budget_exceeded"]
    result = scheduler.poll_once(
        tmp_path,
        tmp_path / "state",
        runner=lambda *a, **k: SimpleNamespace(returncode=1),
    )
    assert result["status"] == "POLL_FAILED"


def test_source_drift_never_launches(tmp_path, monkeypatch):
    def drift(root):
        raise ValueError("frozen source drift")

    monkeypatch.setattr(scheduler, "preflight", drift)
    runner = Mock()
    result = scheduler.poll_once(tmp_path, tmp_path / "state", runner=runner)
    assert result["status"] == "PREFLIGHT_OR_HEALTH_FAILED"
    runner.assert_not_called()


def test_singleton_overlap(tmp_path):
    import fcntl

    state = tmp_path / "state"
    state.mkdir()
    with (state / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert scheduler.poll_once(tmp_path, state)["status"] == "OVERLAP_REFUSED"


def test_archive_forbidden_and_tampered_chunks(tmp_path):
    runs, directory = snapshot(tmp_path, "20260919-shadow-one")
    with pytest.raises(ValueError):
        archive_chunks(runs, runs, tmp_path / "archives", set())
    result = archive_chunks(directory, runs, tmp_path / "archives", set())
    chunk = next((tmp_path / "archives/chunks").glob("*.gz"))
    chunk.write_bytes(b"corrupt")
    with pytest.raises((ValueError, OSError, EOFError)):
        restore_chunks(result["manifest"], tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_evaluation_changed_power_never_computes(tmp_path, monkeypatch):
    from research.project_goals import power

    reg = dict(
        status="registered",
        blocks=[5],
        cost_stress=[0, 1],
        draws=128,
        seed=7,
        required_sessions=120,
        alpha=0.05,
        allocations={"MIM": 1, "YANK": 2, "GAP": 1},
        source_hashes=source_hashes(),
        power_sha256="changed",
    )
    monkeypatch.setattr(
        power,
        "committed_registration",
        lambda *a: (reg, "abc", datetime.now(timezone.utc)),
    )
    p = tmp_path / "power.json"
    write_json(p, {})
    result = power.evaluate(
        tmp_path / "missing", tmp_path / "reg", p, tmp_path / "cal", tmp_path
    )
    assert not result["confirmatory"] and "changed" in result["reason"]


def test_evaluation_insufficient_fresh_sessions(tmp_path, monkeypatch):
    from research.project_goals import power

    calibration = tmp_path / "cal"
    calibration.write_text("exposed")
    p = tmp_path / "power.json"
    write_json(
        p,
        dict(
            input_sha256=digest(calibration),
            sweep=[dict(effect_usd=10, block=1, required_sessions=20)],
            portfolio_sweep=[
                dict(effect_usd=10, cost=c, block=1, sessions=20, required_sessions=20)
                for c in (0, 1)
            ],
            design=power.calibration_design([1], [0, 1], 0.05, 128, 7),
        ),
    )
    reg = dict(
        status="registered",
        blocks=[1],
        cost_stress=[0, 1],
        draws=128,
        seed=7,
        required_sessions=20,
        alpha=0.05,
        allocations={"MIM": 1, "YANK": 2, "GAP": 1},
        source_hashes=source_hashes(),
        power_sha256=digest(p),
        code_sha256=digest(power.__file__),
        power_status="ADEQUATE_BOTH_ENDPOINTS",
        effect_usd=10,
        calibration_commit="a" * 40,
        registered_at="2026-08-01T00:00:00Z",
        freshness_after="2026-08-01T00:00:00Z",
    )
    monkeypatch.setattr(
        power,
        "committed_registration",
        lambda *a: (reg, "abc", datetime(2026, 8, 1, tzinfo=timezone.utc)),
    )
    monkeypatch.setattr(
        power,
        "git_read",
        lambda repo, args: (
            p.read_text().strip()
            if args[0] == "show" and ":" in args[-1]
            else ("2026-07-01T00:00:00Z" if args[0] == "show" else "")
        ),
    )
    data = tmp_path / "fresh.csv"
    data.write_text(
        ",".join(power.REQUIRED)
        + ",corrected_contract,complete_costs\n2026-08-13,1,2,3,10,20,30,1,1,1,true,true\n"
    )
    result = power.evaluate(data, tmp_path / "reg", p, calibration, tmp_path)
    assert result["status"] == "UNDERPOWERED" and not result["confirmatory"]
    data.write_text(data.read_text().replace("2026-08-13", "2026-07-31"))
    assert (
        "pre-registration"
        in power.evaluate(data, tmp_path / "reg", p, calibration, tmp_path)["reason"]
    )


def test_verified_prefix_attribution_survives_append(tmp_path):
    from research.project_goals.audit import attributed_snapshot

    source = tmp_path / "orders"
    source.write_text("order_id\n10\n")
    mapping = tmp_path / "map.json"
    write_json(
        mapping,
        dict(
            orders={
                "10": dict(
                    strategy="MIM",
                    source=str(source),
                    line=2,
                    sha256=digest(source),
                    source_prefix_bytes=source.stat().st_size,
                )
            }
        ),
    )
    source.write_text(source.read_text() + "11\n")
    fills = tmp_path / "fills.json"
    write_json(fills, [fill(profitAndLoss=2)])
    result, rows, _ = attributed_snapshot(fills, mapping)
    assert result["strategy_totals"]["MIM"]["net_reported"] == pytest.approx(1.39)
    source.write_text("changed")
    with pytest.raises(ValueError):
        attributed_snapshot(fills, mapping)


def test_fifo_partial_exits_fees_and_unmatched():
    from research.project_goals.reconciliation import pair_fills, realized_economics

    _, rows = reconcile(
        [
            fill(size=2, fees=0.72, commissions=0.5),
            fill(
                id=2,
                side=1,
                size=1,
                price=105,
                profitAndLoss=10,
                creationTimestamp="2026-08-13T14:01:00Z",
            ),
            fill(
                id=3,
                side=1,
                size=1,
                price=110,
                profitAndLoss=20,
                creationTimestamp="2026-08-13T14:02:00Z",
            ),
        ]
    )
    for row in rows:
        row["strategy_attribution"] = "MIM"
        row["account_epoch"] = "explicit-epoch"
    pairs, unmatched = pair_fills(rows, {"MNQU26": 2})
    assert not unmatched and len(pairs) == 2
    assert sum(p["reconstructed_net"] for p in pairs) == pytest.approx(27.56)
    assert realized_economics(rows)["realized_path"][
        "max_marked_drawdown"
    ] == pytest.approx(1.22)
    assert (
        pair_fills(rows[1:], {"MNQU26": 2})[1][0]["reason"]
        == "realized_exit_without_observed_entry"
    )


def test_reconciliation_unmatched_old_account_record():
    from research.project_goals.reconciliation import compare_records

    rows = [dict(day="2026-08-13", entry_t="10:00", dir="1", pnl_usd="7.5")]
    assert compare_records([], rows, [])[0]["status"] == "UNMATCHED"


def test_conditional_marks_cannot_be_confused_with_certified():
    fills, marks, coverage = evidence()
    coverage["initial_flat"] = False
    assert marked_curves(fills, marks, coverage)[0]["status"] == "INSUFFICIENT_DATA"
    status, rows = marked_curves(
        fills,
        marks,
        coverage,
        assumptions=["Flat start inferred, not independently certified"],
    )
    assert status["status"] == "CONDITIONAL_DESCRIPTIVE"
    assert status["assumptions"] and rows


def test_epoch_separates_pairing():
    from research.project_goals.reconciliation import pair_fills

    _, rows = reconcile(
        [
            fill(),
            fill(
                id=2, side=1, profitAndLoss=5, creationTimestamp="2026-08-14T14:00:00Z"
            ),
        ]
    )
    rows[0]["account_epoch"] = "old"
    rows[1]["account_epoch"] = "new"
    pairs, unmatched = pair_fills(rows, {"MNQU26": 2})
    assert not pairs and len(unmatched) == 2


def test_observed_zero_sessions_preserve_conditional_calendar(tmp_path):
    from research.project_goals.session_report import session_report
    from datetime import timedelta

    fills, marks, _ = evidence()
    for f in fills:
        f["contract"] = "CON.F.US.MNQ.U26"
    export = tmp_path / "marks.json"
    write_json(
        export,
        dict(
            request=dict(
                contractId="CON.F.US.MNQ.U26",
                unit=2,
                unitNumber=1,
                includePartialBar=False,
            ),
            response=dict(
                success=True,
                bars=[
                    dict(
                        t=(
                            timestamp(m["timestamp"]) - timedelta(minutes=1)
                        ).isoformat(),
                        c=m["close"],
                    )
                    for m in marks
                ],
            ),
        ),
    )
    source = tmp_path / "calendar-source"
    source.write_text("observed timestamps only")
    calendar = tmp_path / "calendar.json"
    write_json(
        calendar,
        dict(
            source=str(source),
            source_prefix_bytes=source.stat().st_size,
            sha256=digest(source),
            sessions=[
                dict(
                    session=d,
                    first_timestamp=d + "T14:00:00Z",
                    last_timestamp=d + "T14:02:00Z",
                    observed_rows=3,
                )
                for d in ("2026-08-12", "2026-08-13")
            ],
        ),
    )
    report, rows = session_report(fills, [export], calendar)
    assert report["status"] == "CONDITIONAL_DESCRIPTIVE"
    assert report["scenarios"]["actual"]["metrics"]["sessions"] == 2
    assert report["scenarios"]["actual"]["metrics"]["net"] == pytest.approx(2.78)
    assert (
        report["sessions"]["2026-08-12"]["status"]
        == "CONDITIONAL_ZERO_NO_FILLS_FLAT_CARRY"
    )
