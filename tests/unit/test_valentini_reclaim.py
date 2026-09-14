"""Synthetic-only acceptance tests. Never open market bars or a live trade ledger."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

import pytest

from tools import valentini_reclaim as vr

START = datetime(2026, 3, 8, 22, tzinfo=timezone.utc)
ZERO = vr.Costs(0, 0)


def fixture_bars():
    rows = [
        (100.5, 101, 100, 100.5, 50),
        (100.25, 100.5, 99.5, 99.75, 10),
        (99.75, 100.5, 99.25, 100.25, 20),
        (100.25, 100.75, 100, 100.5, 10),
        (100.5, 101, 100.25, 100.75, 10),
    ]
    return [vr.Bar(START + timedelta(minutes=i), *row) for i, row in enumerate(rows)]


def session(bars, name="synthetic"):
    return vr.Session(
        name, bars[0].timestamp, bars[-1].timestamp + timedelta(minutes=1)
    )


def run(bars, costs=ZERO):
    return vr.simulate_session(bars, session(bars), costs)


def test_profile_conservation_and_lower_ties():
    profile = vr.Profile()
    profile.add(fixture_bars()[0])
    assert sum(profile.volumes.values()) == 50
    assert profile.area() == (100, 100.75, 100)
    profile = vr.Profile()
    profile.add(vr.Bar(START, 100, 100.5, 100, 100, 3))
    profile.add(vr.Bar(START, 100.25, 100.25, 100.25, 100.25, 1))
    assert profile.area() == (100, 100.25, 100.25)  # Equal adjacent volumes: down.


def test_zero_volume_does_not_fabricate_profile():
    profile = vr.Profile()
    profile.add(replace(fixture_bars()[0], volume=0))
    assert profile.area() is None
    assert not run([replace(bar, volume=0) for bar in fixture_bars()]).signals


def test_hand_calculated_reclaim_and_trade_through():
    bars = fixture_bars()
    result = run(bars)
    (signal,) = result.signals
    (trade,) = result.trades
    assert (signal.val, signal.vah, signal.excursion_low, signal.stop) == (
        100,
        100.75,
        99.25,
        99,
    )
    assert signal.break_time == bars[1].timestamp
    assert signal.reclaim_time == bars[2].timestamp
    assert trade.entry_time == bars[3].timestamp
    assert (
        trade.exit_bar_time == bars[4].timestamp
    )  # Exact target touch at bar 3 does not fill.
    assert trade.exit_time is None
    assert (
        trade.entry,
        trade.exit,
        trade.reason,
        trade.net_dollars,
        trade.contracts,
    ) == (100.25, 100.75, "target", 1, 1)


def test_future_bar_changes_and_appends_do_not_change_signals():
    bars = fixture_bars()
    expected = run(bars).signals
    bars[4] = replace(bars[4], high=200, volume=99999)
    assert run(bars).signals == expected
    bars.append(replace(bars[-1], timestamp=bars[-1].timestamp + timedelta(minutes=1)))
    assert run(bars).signals == expected
    assert run(bars[:3]).signals == expected  # Confirmation at end cannot enter.
    assert not run(bars[:3]).trades


@pytest.mark.parametrize(
    "change",
    [
        {"volume": 10},  # Equal reclaim volume cancels.
        {"close": 101, "high": 101},  # First reclaim overshoots.
    ],
)
def test_first_reclaim_cancels_without_later_rescue(change):
    bars = fixture_bars()
    bars[2] = replace(bars[2], **change)
    assert not run(bars).signals
    assert not run(bars).trades


def test_equal_break_volume_and_outside_previous_close_do_not_arm():
    bars = fixture_bars()
    bars[1] = replace(bars[1], volume=50)
    assert not run(bars).signals
    bars = fixture_bars()
    bars[0] = replace(bars[0], close=101)
    assert not run(bars).signals


def test_excursion_low_includes_intermediate_and_reclaim_bars():
    bars = fixture_bars()
    extra = vr.Bar(START + timedelta(minutes=2), 99.75, 99.75, 98.5, 99.5, 5)
    bars.insert(2, extra)
    bars = [
        replace(bar, timestamp=START + timedelta(minutes=i))
        for i, bar in enumerate(bars)
    ]
    assert run(bars).signals[0].stop == 98.25


@pytest.mark.parametrize(
    "opening,slip", [(99, 0), (100.75, 0), (100.5, 0.25), (98.75, 0), (101, 0)]
)
def test_strict_entry_bounds_cancel(opening, slip):
    bars = fixture_bars()
    bars[3] = replace(
        bars[3], open=opening, low=min(opening, 100), high=max(opening, 100.75)
    )
    assert run(bars, vr.Costs(slip, 0)).trades == ()


def test_costs_target_limit_and_commissions():
    (trade,) = run(fixture_bars(), vr.Costs(0.25, 0.1)).trades
    assert trade.entry == 100.5
    assert trade.exit == 100.75
    assert trade.gross_dollars == 0.5
    assert trade.net_dollars == pytest.approx(0.3)


def test_stop_first_when_both_levels_touched():
    bars = fixture_bars()
    bars[3] = replace(bars[3], low=98.75, high=101)
    (trade,) = run(bars, vr.Costs(0.25, 0)).trades
    assert (trade.reason, trade.exit) == ("stop", 98.75)


def test_gap_stop_fills_worse_open_with_slippage():
    bars = fixture_bars()
    bars[4] = replace(bars[4], open=98.5, low=98, high=99, close=98.75)
    (trade,) = run(bars, vr.Costs(0.25, 0)).trades
    assert (trade.reason, trade.exit) == ("stop", 98.25)
    assert trade.exit_time == bars[4].timestamp
    assert trade.exit_bar_time == bars[4].timestamp


def test_last_minute_entry_and_market_session_end_costs():
    bars = fixture_bars()[:4]
    (trade,) = run(bars, vr.Costs(0.25, 0.1)).trades
    assert (trade.reason, trade.entry_time, trade.exit_time) == (
        "session_end",
        bars[3].timestamp,
        bars[3].timestamp + timedelta(minutes=1),
    )
    assert trade.exit_bar_time == bars[3].timestamp
    assert trade.exit == 100.25
    assert trade.net_dollars == pytest.approx(-0.7)


def test_no_same_bar_rearm_after_exit_or_cancel():
    bars = fixture_bars()
    bars[3] = replace(bars[3], close=99.75, low=98.75, volume=1)
    # Stop on a low-volume break bar; the following reclaim cannot reuse it.
    bars[4] = replace(bars[4], open=99.75, low=99.5, close=100.25, volume=20)
    assert len(run(bars).signals) == 1
    assert len(run(bars).trades) == 1
    bars = fixture_bars()
    bars[3] = replace(bars[3], open=99, low=98.75, close=99.75, volume=1)
    assert len(run(bars).signals) == 1
    assert not run(bars).trades


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "off_tick",
        "negative_volume",
        "nonfinite",
        "invalid_ohlc",
        "naive",
        "off_minute",
    ],
)
def test_bad_session_rejected_before_profile(monkeypatch, kind):
    bars = fixture_bars()
    schedule = session(bars)
    if kind == "missing":
        del bars[2]
    elif kind == "duplicate":
        bars[2] = bars[1]
    else:
        changes = {
            "off_tick": {"close": 100.1},
            "negative_volume": {"volume": -1},
            "nonfinite": {"volume": float("nan")},
            "invalid_ohlc": {"low": 101},
            "naive": {"timestamp": START.replace(tzinfo=None)},
            "off_minute": {"timestamp": START + timedelta(seconds=1)},
        }
        bars[0] = replace(bars[0], **changes[kind])
    monkeypatch.setattr(
        vr.Profile,
        "add",
        lambda *args: pytest.fail("Profile reached before validation"),
    )
    with pytest.raises(vr.Rejected):
        vr.simulate_session(bars, schedule, ZERO)


def test_schedule_resets_and_shortened_dst_sessions():
    bars = fixture_bars()[:3]  # Pending confirmation at the first session end.
    first = session(bars, "before_dst")
    next_start = datetime(2026, 11, 1, 23, tzinfo=timezone.utc)
    second_bars = [
        replace(bar, timestamp=next_start + timedelta(minutes=i))
        for i, bar in enumerate(fixture_bars()[:2])
    ]
    second = session(second_bars, "after_dst_shortened")
    result = vr.simulate_sessions(bars + second_bars, [first, second], ZERO)
    assert len(result.signals) == 1 and not result.trades
    with pytest.raises(vr.Rejected):
        vr.simulate_sessions(
            bars, [replace(first, end=first.end + timedelta(minutes=1))], ZERO
        )
    with pytest.raises(vr.Rejected):
        vr.simulate_sessions(bars + second_bars, [first], ZERO)


def test_null_only_uses_mismatched_pairings_and_is_deterministic():
    panel = [[float("nan"), 2, 3], [4, float("nan"), 5], [7, 8, float("nan")]]
    pairings = vr.derangements(3, 20, 44)
    assert pairings == vr.derangements(3, 20, 44)
    assert all(i != j for pairing in pairings for i, j in enumerate(pairing))
    result = vr.null_mde(
        panel,
        pairings,
        effect=None,
        alpha=0.05,
        power=0.8,
        dependence_factors=[1, 2, 4],
    )
    assert result == vr.null_mde(
        panel,
        pairings,
        effect=None,
        alpha=0.05,
        power=0.8,
        dependence_factors=[1, 2, 4],
    )
    assert (
        result["dependence_sensitivity"][2]["mde"]
        == 2 * result["dependence_sensitivity"][0]["mde"]
    )
    assert result["dependence_sensitivity"][0]["effect_exceeds_mde"] is None


@pytest.mark.parametrize("pairing", [[0, 1, 2], [0, 2, 1], [1, 1, 0]])
def test_null_rejects_identity_fixed_points_and_invalid_permutation(pairing):
    with pytest.raises(vr.Rejected):
        vr.null_mde(
            [[1] * 3] * 3,
            [pairing, pairing],
            effect=0.2,
            alpha=0.05,
            power=0.8,
            dependence_factors=[1],
        )


def csv_file(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "\n".join(
            f"{b.timestamp.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.volume}"
            for b in fixture_bars()
        )
        + "\n"
    )
    return path


def write_json(path, value):
    path.write_text(vr.canonical_json(value))
    return path


def artifacts(tmp_path):
    source = csv_file(tmp_path)
    audit = write_json(tmp_path / "audit.json", vr.make_audit([source]))
    gate = write_json(tmp_path / "gate.json", vr.make_power(audit, None))
    return source, audit, gate


def test_audit_metadata_deterministic_and_no_performance(monkeypatch, tmp_path):
    monkeypatch.setattr(vr, "simulate_session", lambda *args: pytest.fail("simulation"))
    monkeypatch.setattr(vr, "null_mde", lambda *args: pytest.fail("null"))
    source = csv_file(tmp_path)
    first = vr.make_audit([source])
    assert first == vr.make_audit([source])
    item = first["inputs"][0]
    assert item["row_count"] == 5 and item["invalid_rows"] == 0
    assert item["gap_intervals"] == item["duplicate_timestamps"] == 0
    assert all(value == "UNKNOWN" for value in item["evidence"].values())
    audit = write_json(tmp_path / "audit.json", first)
    power = vr.make_power(audit, None)
    assert power == vr.make_power(audit, None)
    assert power["verdict"] == "DATA_UNSUITABLE"
    assert power["power_status"] == "POWER_UNDETERMINED"
    assert not power["null_computed"] and not power["performance_computed"]


def test_audit_counts_invalid_duplicates_gaps_schema(tmp_path):
    source = csv_file(tmp_path)
    lines = source.read_text().splitlines()
    source.write_text(
        "\n".join([lines[0], lines[1], lines[1], lines[4], "bad,100,100,100,100,1"])
        + "\n"
    )
    item = vr.audit_csv(source)
    assert item["row_count"] == 4 and item["invalid_rows"] == 1
    assert item["duplicate_timestamps"] == 1
    assert item["gap_intervals"] == 1 and item["missing_clock_minutes"] == 2
    source.write_text("date,close\n2026-03-01,100\n")
    assert vr.audit_csv(source)["invalid_rows"] == 1
    assert "timestamp" in vr.audit_csv(source)["missing_columns"]


def test_protected_requested_and_resolved_paths(tmp_path):
    sealed = tmp_path / "sealed_holdout"
    sealed.mkdir()
    secret = sealed / "bars.csv"
    secret.write_text("do not read")
    alias = tmp_path / "alias.csv"
    alias.symlink_to(secret)
    lexical = sealed / ".." / "ordinary.csv"
    for path in (secret, alias, lexical, tmp_path / ".env", tmp_path / "trades.db"):
        with pytest.raises(vr.Rejected, match="Protected"):
            vr.protected_path(path)
    safe = csv_file(tmp_path)
    sealed_alias = sealed / "escape.csv"
    sealed_alias.symlink_to(safe)
    with pytest.raises(vr.Rejected):
        vr.make_audit([sealed_alias])


def test_input_and_code_hash_mismatch(tmp_path):
    source, audit_path, _ = artifacts(tmp_path)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(vr.Rejected, match="input hash mismatch"):
        vr.make_power(audit_path, None)
    audit = vr.read_json(audit_path)
    audit["code_sha256"] = "forged"
    write_json(audit_path, audit)
    with pytest.raises(vr.Rejected, match="code hash mismatch"):
        vr.make_power(audit_path, None)


def test_forged_powered_always_rejected_before_market_read(monkeypatch, tmp_path):
    source, audit, gate_path = artifacts(tmp_path)
    gate = vr.read_json(gate_path)
    gate["verdict"] = "POWERED"
    write_json(gate_path, gate)
    source.unlink()  # Evaluation must refuse without attempting market input access.
    monkeypatch.setattr(vr, "simulate_session", lambda *args: pytest.fail("simulation"))
    with pytest.raises(vr.Rejected, match="cannot independently"):
        vr.evaluate(audit, gate_path)


def test_gate_mismatch_absent_nonpowered_and_no_bypass(tmp_path):
    _, audit, gate_path = artifacts(tmp_path)
    with pytest.raises(vr.Rejected, match="not POWERED"):
        vr.evaluate(audit, gate_path)
    with pytest.raises(OSError):
        vr.evaluate(audit, tmp_path / "missing.json")
    gate = vr.read_json(gate_path)
    gate["audit_sha256"] = "forged"
    write_json(gate_path, gate)
    with pytest.raises(vr.Rejected, match="Gate/audit hash mismatch"):
        vr.evaluate(audit, gate_path)
    assert vr.main(["evaluate", "--audit", str(audit), "--gate", str(gate_path)]) == 2
    with pytest.raises(SystemExit):
        vr.main(
            ["evaluate", "--audit", str(audit), "--gate", str(gate_path), "--force"]
        )


def test_user_authored_evidence_does_not_promote(tmp_path):
    _, audit, _ = artifacts(tmp_path)
    evidence = write_json(
        tmp_path / "evidence.json",
        {
            "admitted": True,
            "verdict": "POWERED",
            "transferable_effect": 99,
            "independent_calibration": True,
        },
    )
    assert vr.make_power(audit, evidence)["verdict"] == "DATA_UNSUITABLE"


def test_cli_repeated_csv_strict_json_and_overwrite_protection(tmp_path):
    source = csv_file(tmp_path)
    second = tmp_path / "second.csv"
    second.write_text(source.read_text())
    audit = tmp_path / "audit.json"
    assert (
        vr.main(
            [
                "audit",
                "--csv",
                str(source),
                "--csv",
                str(second),
                "--output",
                str(audit),
            ]
        )
        == 0
    )
    assert len(json.loads(audit.read_text())["inputs"]) == 2
    before = source.read_text()
    assert vr.main(["audit", "--csv", str(source), "--output", str(source)]) == 2
    assert source.read_text() == before
    with pytest.raises(ValueError):
        vr.canonical_json({"bad": float("inf")})
    bad = tmp_path / "bad.json"
    bad.write_text('{"bad": NaN}')
    with pytest.raises(vr.Rejected):
        vr.read_json(bad)


def test_scheduled_halt_cancels_pending_keeps_profile_and_open_position():
    bars = fixture_bars()
    halt_start = bars[3].timestamp
    halt_end = halt_start + timedelta(minutes=15)
    after = [
        replace(bar, timestamp=bar.timestamp + timedelta(minutes=15))
        for bar in bars[3:]
    ]
    schedule = vr.Session(
        "halt",
        START,
        after[-1].timestamp + timedelta(minutes=1),
        ((halt_start, halt_end),),
    )
    result = vr.simulate_session(bars[:3] + after, schedule, ZERO)
    assert len(result.signals) == 1 and not result.trades
    # Missing a scheduled tradable minute still rejects the whole session.
    with pytest.raises(vr.Rejected):
        vr.simulate_session(bars[:2] + after, schedule, ZERO)
    # A position already open is held through the halt and exits on the next print.
    halt_start = bars[4].timestamp
    halt_end = halt_start + timedelta(minutes=15)
    after = replace(
        bars[4], timestamp=halt_end, open=98.5, high=99, low=98, close=98.75
    )
    schedule = vr.Session(
        "holding", START, halt_end + timedelta(minutes=1), ((halt_start, halt_end),)
    )
    result = vr.simulate_session(bars[:4] + [after], schedule, vr.Costs(0.25, 0))
    assert result.trades[0].exit == 98.25
    # Reclaim across the halt is not an adjacent one-minute volume confirmation.
    halt_start = bars[2].timestamp
    halt_end = halt_start + timedelta(minutes=15)
    shifted = [
        replace(bar, timestamp=bar.timestamp + timedelta(minutes=15))
        for bar in bars[2:]
    ]
    schedule = vr.Session(
        "no_cross_halt_reclaim",
        START,
        shifted[-1].timestamp + timedelta(minutes=1),
        ((halt_start, halt_end),),
    )
    assert not vr.simulate_session(bars[:2] + shifted, schedule, ZERO).signals


def test_fresh_break_can_arm_on_bar_after_exit():
    bars = fixture_bars()
    bars[0] = replace(bars[0], volume=5000)
    bars[3] = replace(bars[3], high=101, close=100.5, volume=100)
    bars[4] = replace(bars[4], open=100.5, high=100.5, low=99.5, close=99.75, volume=1)
    bars.extend(
        [
            vr.Bar(START + timedelta(minutes=5), 99.75, 100.5, 99.5, 100.25, 20),
            vr.Bar(START + timedelta(minutes=6), 100.25, 101, 100, 100.75, 10),
        ]
    )
    result = run(bars)
    assert len(result.signals) == len(result.trades) == 2
    assert result.signals[1].break_time == bars[4].timestamp


def test_volume_profile_crosses_untraded_ticks_contiguously():
    profile = vr.Profile()
    profile.add(vr.Bar(START, 100, 100, 100, 100, 6))
    profile.add(vr.Bar(START, 101, 101, 101, 101, 4))
    assert profile.area() == (100, 101, 100)
    assert sum(profile.volumes.values()) == 10


@pytest.mark.parametrize(
    "command",
    ["audit", "power_audit", "power_evidence", "power_csv", "power_csv_direct"],
)
def test_cli_rejects_output_aliases_preserving_all_input_bytes(tmp_path, command):
    source, audit, _ = artifacts(tmp_path)
    evidence = write_json(tmp_path / "evidence.json", {"candidate": "synthetic"})
    targets = {
        "audit": source,
        "power_audit": audit,
        "power_evidence": evidence,
        "power_csv": source,
        "power_csv_direct": source,
    }
    output = tmp_path / "alias.json"
    if command == "power_csv_direct":
        output = source
    else:
        output.hardlink_to(targets[command])
    snapshots = {path: path.read_bytes() for path in (source, audit, evidence)}
    args = (
        ["audit", "--csv", str(source)]
        if command == "audit"
        else ["power", "--audit", str(audit), "--evidence", str(evidence)]
    )
    assert vr.main(args + ["--output", str(output)]) == 2
    assert all(path.read_bytes() == data for path, data in snapshots.items())


def test_protected_intermediate_symlink_target(tmp_path):
    ordinary = csv_file(tmp_path)
    sealed = tmp_path / "sealed_holdout"
    sealed.mkdir()
    hop = sealed / "out.csv"
    hop.symlink_to(ordinary)
    alias = tmp_path / "ordinary-alias.csv"
    alias.symlink_to(hop)
    assert alias.resolve() == ordinary
    with pytest.raises(vr.Rejected, match="Protected"):
        vr.audit_csv(alias)
    # A relative intermediate link must also be checked before final resolution.
    alias.unlink()
    alias.symlink_to("sealed_holdout/out.csv")
    with pytest.raises(vr.Rejected, match="Protected"):
        vr.protected_path(alias)


def test_unterminated_csv_quote_cannot_produce_metadata(tmp_path):
    import csv

    source = csv_file(tmp_path)
    source.write_text(
        'timestamp,open,high,low,close,volume\n2026-03-08T22:00:00Z,100,100,100,100,"1'
    )
    with pytest.raises(csv.Error):
        vr.audit_csv(source)
    output = tmp_path / "bad-audit.json"
    assert vr.main(["audit", "--csv", str(source), "--output", str(output)]) == 2
    assert not output.exists()


def test_duplicate_conflicting_json_verdicts_rejected(tmp_path):
    path = tmp_path / "conflicting.json"
    path.write_text('{"verdict":"DATA_UNSUITABLE","verdict":"POWERED"}')
    with pytest.raises(vr.Rejected, match="Duplicate JSON key: verdict"):
        vr.read_json(path)


def test_finite_volume_overflow_is_rejected():
    profile = vr.Profile()
    bar = vr.Bar(START, 100, 100, 100, 100, 1e308)
    profile.add(bar)
    with pytest.raises(vr.Rejected, match="profile volume"):
        profile.add(bar)
    assert profile.volumes == {400: 1e308}
    # Independently finite bins can overflow their sum as well.
    profile.volumes = {400: 1e308, 401: 1e308}
    with pytest.raises(vr.Rejected, match="profile volume"):
        profile.area()
    with pytest.raises(vr.Rejected, match="profile volume"):
        run([replace(bar, volume=1e308) for bar in fixture_bars()])


def test_finite_commission_overflow_is_rejected():
    with pytest.raises(vr.Rejected, match="commission"):
        run(fixture_bars(), vr.Costs(0, 1e308))


def test_finite_price_scaling_overflow_is_rejected():
    with pytest.raises(vr.Rejected, match="tick index"):
        vr.validate_bar(vr.Bar(START, 1e308, 1e308, 1e308, 1e308, 1))


def test_null_mde_independent_numeric_oracle():
    from math import sqrt

    panel = [[float("nan"), 1, 4], [7, float("nan"), 2], [3, 10, float("nan")]]
    result = vr.null_mde(
        panel,
        [[1, 2, 0], [2, 0, 1]],
        effect=10,
        alpha=0.05,
        power=0.8,
        dependence_factors=[1, 4],
    )
    assert result["null_means"] == [2, 7]
    assert result["null_mean"] == 4.5
    assert result["null_sd"] == pytest.approx(sqrt(12.5))
    # Published normal quantiles; oracle does not call implementation/statistics.
    expected = (1.6448536269514722 + 0.8416212335729143) * sqrt(12.5)
    first, second = result["dependence_sensitivity"]
    assert first["mde"] == pytest.approx(expected)
    assert second["mde"] == pytest.approx(2 * expected)
    assert first["effect_exceeds_mde"] is True
    assert second["effect_exceeds_mde"] is False


@pytest.mark.parametrize(
    "panel,pairings",
    [
        ([[0, 1], [2, 0]], [[1, 0], [1, 0]]),
        ([[0, 1, 4], [7, 0, 2], [3, 10, 0]], [[1, 2, 0]] * 4),
        ([[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[1, 2, 0], [2, 0, 1]]),
    ],
)
def test_degenerate_null_calibration_rejected(panel, pairings):
    with pytest.raises(vr.Rejected, match="Degenerate calibration"):
        vr.null_mde(
            panel, pairings, effect=1, alpha=0.05, power=0.8, dependence_factors=[1]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "unrelated"),
        ("version", "future-version"),
        ("kind", None),
        ("version", None),
    ],
)
def test_evaluate_rejects_audit_schema_without_market_reads(tmp_path, field, value):
    source, audit_path, gate_path = artifacts(tmp_path)
    audit = vr.read_json(audit_path)
    audit[field] = value
    write_json(audit_path, audit)
    gate = vr.read_json(gate_path)
    gate["audit_sha256"] = vr.sha256_file(audit_path)
    gate["verdict"] = "POWERED"
    write_json(gate_path, gate)
    source.unlink()
    with pytest.raises(vr.Rejected, match="Audit schema/version mismatch"):
        vr.evaluate(audit_path, gate_path)


def test_session_end_time_is_final_tradable_close_before_trailing_break():
    bars = fixture_bars()[:4]
    close = bars[-1].timestamp + timedelta(minutes=1)
    end = close + timedelta(minutes=15)
    schedule = vr.Session("trailing_break", START, end, ((close, end),))
    (trade,) = vr.simulate_session(bars, schedule, ZERO).trades
    assert trade.reason == "session_end"
    assert trade.exit_bar_time == bars[-1].timestamp
    assert trade.exit_time == close


def test_gap_above_target_exits_at_open_before_later_stop_touch():
    bars = fixture_bars()
    bars[4] = replace(bars[4], open=101, high=101.25, low=98.5, close=99)
    (trade,) = run(bars, vr.Costs(0.25, 0)).trades
    assert (trade.reason, trade.exit) == ("target", 100.75)
    assert trade.exit_time == trade.exit_bar_time == bars[4].timestamp
