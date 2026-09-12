import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.mim_comparison.data import Session, audit_select
from research.mim_comparison.engine import (
    Arm as OriginalArm,
    simulate as original_simulate,
)
from research.mim_robustness.engine import Arm, simulate
from research.mim_robustness.features import features, gate
from research.mim_robustness.statistics import (
    metrics,
    sharpe,
    stationary_bootstrap,
    classify,
    paired_matrix,
)
from research.mim_robustness.study import (
    compare_frame,
    trade_ledger,
    reconcile_accounting,
)
from research.mim_robustness import artifacts


def session(prices=None, day="2025-01-06", contract="MNQH25"):
    c = np.full(390, 10000.0) if prices is None else np.asarray(prices, dtype=float)
    t = pd.date_range(day + " 09:31", periods=390, freq="min", tz="America/New_York")
    return Session(
        day,
        contract,
        pd.DataFrame(
            dict(
                timestamp=t,
                day=day,
                contract=contract,
                minute=np.arange(571, 961),
                open=c.copy(),
                close=c.copy(),
                high=c + 1,
                low=c - 1,
                volume=100.0,
            )
        ),
        10000.0,
    )


@pytest.mark.parametrize("direction", [1, -1])
def test_trailing_formulas_and_constant(direction):
    s = session(10000 + direction * np.arange(390))
    f = features(s, np.full(390, 0.001))
    assert np.isnan(f["slope"][:29]).all()
    assert f["slope"][29] == pytest.approx(direction)
    assert f["r2"][29] == pytest.approx(1)
    assert f["efficiency"][29] == 1
    assert gate("R", direction, 29, f) and gate("E", direction, 29, f)
    assert not gate("R", -direction, 29, f)
    constant = features(session(), np.zeros(390))
    assert not gate("R", direction, 29, constant) and not gate(
        "E", direction, 29, constant
    )


@pytest.mark.parametrize("arm,field", [("R", "r2"), ("E", "efficiency")])
def test_gate_equality_and_unavailable(arm, field):
    f = features(session(10000 + np.arange(390)), np.zeros(390))
    f[field][29] = 0.30
    assert gate(arm, 1, 29, f)
    f[field][29] = np.nextafter(0.30, 0)
    assert not gate(arm, 1, 29, f)
    assert not gate(arm, 1, 28, f)


def test_persistence_uses_previous_own_band_and_strict_equality():
    s = session()
    s.bars.loc[28:29, "close"] = 10020
    sigma = np.full(390, 0.001)
    sigma[28] = 0.003
    assert not gate("P", 1, 29, features(s, sigma))
    sigma[28] = 0.001
    assert gate("P", 1, 29, features(s, sigma))
    s.bars.loc[28, "close"] = features(s, sigma)["upper"][28]
    assert not gate("P", 1, 29, features(s, sigma))


@pytest.mark.parametrize("arm", ["A", "R", "E", "F", "P"])
@pytest.mark.parametrize("delay", [1, 2])
def test_future_mutation_preserves_prior_decisions(arm, delay):
    c = 10000 + np.sin(np.arange(390) / 16) * 60
    a = session(c)
    b = session(c.copy())
    b.bars.loc[151:, "close"] += 800
    for col in ("open", "high", "low"):
        b.bars.loc[151:, col] += 800
    sigma = np.full(390, 0.001)
    first = simulate(a, sigma, Arm(arm), delay)[2]
    second = simulate(b, sigma, Arm(arm), delay)[2]
    assert [r for r in first if r["event_timestamp"] < "2025-01-06T12:02:00-05:00"] == [
        r for r in second if r["event_timestamp"] < "2025-01-06T12:02:00-05:00"
    ]


@pytest.mark.parametrize("delay", [1, 2])
@pytest.mark.parametrize("seed", [4, 9, 18])
def test_copied_baseline_agreement_and_roundtrip_accounting(delay, seed):
    rng = np.random.default_rng(seed)
    s = session(10000 + np.cumsum(rng.normal(0, 20, 390)))
    sigma = np.full(390, 0.002)
    expected = original_simulate(s, sigma, OriginalArm("A"), delay)
    actual = simulate(s, sigma, Arm("A"), delay)
    for a, b in zip(actual, expected):
        af = pd.DataFrame([a] if isinstance(a, dict) else a)
        bf = pd.DataFrame([b] if isinstance(b, dict) else b)
        compare_frame(af, bf, "fixture")
    ledger = pd.DataFrame([dict(x, delay=delay) for x in actual[1]])
    decisions = pd.DataFrame([dict(x, delay=delay) for x in actual[2]])
    trades = trade_ledger(ledger, decisions)
    daily = pd.DataFrame([dict(actual[0], delay=delay)])
    reconcile_accounting(daily, ledger, trades)


def breakout_session():
    c = np.full(390, 10020.0)
    c[:29] = 10000
    c[59:62] = 9980
    c[62:89] = 10000
    return session(c)


def test_rejected_reversal_still_exits_and_reset_rearms_later():
    s = breakout_session()
    sigma = np.full(390, 0.001)
    f = features(s, sigma)
    f["r2"][29] = 1.0
    f["slope"][29] = 1.0
    f["r2"][59] = 0.0
    daily, ledger, decisions = simulate(s, sigma, Arm("R"), feature_values=f)
    reversal = next(
        d for d in decisions if d["event_timestamp"].endswith("10:30:00-05:00")
    )
    assert (
        reversal["position"] == 1
        and reversal["target"] == 0
        and not reversal["gate_pass"]
    )
    assert any(e["reason"] == "BAND_STOP" and e["position_after"] == 0 for e in ledger)
    _, ledger, decisions = simulate(s, sigma, Arm("F"))
    entries = [e for e in ledger if e["position_after"]]
    assert entries[0]["event_timestamp"].endswith("10:02:00-05:00")
    assert entries[1]["event_timestamp"].endswith("11:02:00-05:00")
    # Inside only on exit bar cannot rearm; subsequent closes stay outside.
    s.bars.loc[61, "close"] = 10000
    s.bars.loc[62:, "close"] = 10020
    _, ledger, _ = simulate(s, sigma, Arm("F"))
    assert len([e for e in ledger if e["position_after"]]) == 1


@pytest.mark.parametrize("arm", ["A", "R", "E", "F", "P"])
def test_entry_bar_stop_adverse_gap_and_eod(arm):
    c = np.full(390, 10020.0)
    c[:29] = 10000
    c[28] = 10020
    s = session(c)
    sigma = np.full(390, 0.001)
    f = features(s, sigma)
    f["r2"][29] = 1.0
    f["slope"][29] = 1.0
    f["efficiency"][29] = 1.0
    f["displacement"][29] = 20.0
    s.bars.loc[31, "open"] = 9700
    s.bars.loc[31, "low"] = 9699
    _, ledger, _ = simulate(s, sigma, Arm(arm), feature_values=f)
    assert ledger[0]["reason"] == "ENTRY" and ledger[1]["reason"] == "CAT_STOP"
    assert (
        ledger[1]["fill"] == 9700
        and ledger[1]["event_timestamp"] == ledger[0]["event_timestamp"]
    )
    assert ledger[-1]["position_after"] == 0


def test_guard_boundary_and_no_same_bar_reset_after_catastrophe():
    s = session()
    sigma = np.zeros(390)
    for i in (29, 59, 89):
        s.bars.loc[i : i + 4, ["close", "open", "high", "low"]] = [
            10020,
            10020,
            10021,
            10019,
        ]
    for i in (32, 62):
        s.bars.loc[i, ["open", "close", "high", "low"]] = [9770, 9770, 9771, 9769]
    _, ledger, decisions = simulate(s, sigma, Arm("A"))
    assert sum(e["reason"] == "CAT_STOP" for e in ledger) == 2
    assert ledger[-1]["reference_realized_gross"] == -1000
    assert not any(
        d["target"]
        for d in decisions
        if d["event_timestamp"] >= "2025-01-06T11:00:00-05:00"
    )
    blocked = [d for d in decisions if d["entry_disposition"] == "daily_guard"]
    assert blocked and all(
        d["risk_blocked"] and d["guard_deactivated"] for d in blocked
    )


def test_accepted_pending_reversal_preserves_baseline_after_stop():
    s = breakout_session()
    sigma = np.full(390, 0.001)
    s.bars.loc[60, "low"] = 9769
    a = simulate(s, sigma, Arm("A"))
    b = original_simulate(s, sigma, OriginalArm("A"))
    compare_frame(pd.DataFrame(a[1]), pd.DataFrame(b[1]), "pending_reversal")
    assert any(
        e["reason"] == "CAT_STOP" and e["event_timestamp"].endswith("10:31:00-05:00")
        for e in a[1]
    )
    assert any(
        e["reason"] == "REVERSAL"
        and e["position_after"] == -1
        and e["event_timestamp"].endswith("10:32:00-05:00")
        for e in a[1]
    )


@pytest.mark.parametrize(
    "which,column,value",
    [
        ("trades", "net", 999),
        ("trades", "costs", 99),
        ("ledger", "costs", 99),
        ("daily", "net", np.nan),
        ("trades", "gross", np.nan),
        ("ledger", "fill", np.inf),
    ],
)
def test_reconcile_rejects_bad_costs_and_nonfinite(which, column, value):
    s = breakout_session()
    d, e, c = simulate(s, np.full(390, 0.001), Arm("A"))
    ledger = pd.DataFrame([dict(x, delay=2) for x in e])
    decisions = pd.DataFrame([dict(x, delay=2) for x in c])
    trades = trade_ledger(ledger, decisions)
    daily = pd.DataFrame([d])
    {"trades": trades, "ledger": ledger, "daily": daily}[which].loc[0, column] = value
    with pytest.raises(ValueError):
        reconcile_accounting(daily, ledger, trades)


def test_manifest_configuration_drives_feature_threshold_and_capital(monkeypatch):
    f = features(session(10000 + np.arange(390)), np.zeros(390))
    f["r2"][29] = 0.4
    monkeypatch.setitem(artifacts.CONFIG, "threshold", 0.5)
    assert not gate("R", 1, 29, f)
    monkeypatch.setitem(artifacts.CONFIG, "window", 10)
    assert np.isfinite(features(session(), np.zeros(390))["r2"][9])


def frame(x):
    return pd.DataFrame(
        dict(
            day=pd.bdate_range("2025-01-01", periods=len(x)).strftime("%Y-%m-%d"),
            net=x,
            turnover=2,
            exposure_contract_minutes=50,
        )
    )


def test_metrics_daily_grid_drawdown_and_censoring():
    x = np.array([10.0, 0, -20, 5, 30, -2])
    m = metrics(frame(x))
    assert m["sharpe"] == pytest.approx(np.sqrt(252) * x.mean() / x.std(ddof=1))
    assert m["max_drawdown"] == 20 and m["longest_underwater_sessions"] == 2
    assert (
        m["underwater_recovery_censored"] and m["unresolved_underwater_sessions"] == 1
    )
    assert metrics(frame(np.zeros(6)))["sharpe"] is None
    assert metrics(frame(np.full(60, 1000.0)))["sharpe"] is None
    complete = pd.concat(
        [frame(x).assign(arm=a, delay=2, cost=2.24) for a in ("A", "R", "E", "F", "P")]
    )
    assert len(paired_matrix(complete)) == 6
    with pytest.raises(ValueError):
        paired_matrix(complete.iloc[1:])
    with pytest.raises(ValueError):
        paired_matrix(pd.concat([complete, complete.iloc[:1]]))


def test_bootstrap_pairing_determinism_and_undefined():
    x = np.random.default_rng(1).normal(size=60)
    matrix = np.tile(x[:, None], (1, 5))
    a = stationary_bootstrap(matrix, 5, draws=400)
    assert a == stationary_bootstrap(matrix, 5, draws=400)
    assert all(
        v["ci98_75"] == [0.0, 0.0] and v["undefined_draws"] == 0 for v in a.values()
    )
    matrix[:, 1] = 0
    assert not stationary_bootstrap(matrix, 5, draws=100)["R"]["inference_available"]
    with pytest.raises(ValueError):
        stationary_bootstrap(np.full((3, 5), np.nan), 5)


def test_classification_all_boundaries_and_dependence_conflict():
    base = dict(sharpe=1.0, max_drawdown=100.0, total_net=100.0, mean_net=10.0)
    candidate = dict(sharpe=1.2, max_drawdown=80.0, total_net=75.0, mean_net=7.5)
    primary = {
        a: dict(base if a == "A" else candidate) for a in ("A", "R", "E", "F", "P")
    }
    costly = {a: dict(x) for a, x in primary.items()}
    u = {
        str(b): {
            a: dict(inference_available=True, ci98_75=[0.01, 0.5])
            for a in ("R", "E", "F", "P")
        }
        for b in (5, 10, 20)
    }
    result = classify(primary, costly, u)
    assert result["ranked_shortlist"] == ["E", "F", "P", "R"]
    u["20"]["R"]["ci98_75"][0] = 0
    assert (
        classify(primary, costly, u)["candidates"]["R"]["classification"]
        == "promising but uncertain"
    )
    primary["E"]["total_net"] = 74.999
    assert (
        classify(primary, costly, u)["candidates"]["E"]["classification"]
        == "does not meet screen"
    )
    costly["P"]["mean_net"] = 0
    assert not classify(primary, costly, u)["candidates"]["P"]["highcost_screen"]


def test_compare_refuses_discrete_and_numeric_drift():
    expected = pd.DataFrame(dict(target=[1], sigma=[0.001], gross=[3.0]))
    compare_frame(expected.copy(), expected, "x")
    for col, value in [("target", 0), ("sigma", 0.002), ("gross", 3.1)]:
        bad = expected.copy()
        bad[col] = value
        with pytest.raises(ValueError):
            compare_frame(bad, expected, "x")


def test_artifact_exclusive_inventory_and_outside_refusal(tmp_path, monkeypatch):
    isolated = tmp_path / "research"
    isolated.mkdir()
    runs = isolated / "runs"
    runs.mkdir()
    monkeypatch.setattr(artifacts, "BASE", isolated)
    monkeypatch.setattr(artifacts, "RUNS", runs)
    run = runs / "fixture"
    run.mkdir()
    artifacts.write_json(run / "example.json", dict(value=1))
    with pytest.raises(FileExistsError):
        artifacts.write_json(run / "example.json", dict(value=2))
    artifacts.seal(run)
    artifacts.verify_inventory(run)
    assert (run / "example.json").stat().st_mode & 0o222 == 0
    (run / "example.json").chmod(0o644)
    (run / "example.json").write_text("{}")
    with pytest.raises(ValueError):
        artifacts.verify_inventory(run)
    with pytest.raises(ValueError):
        artifacts.safe_run(tmp_path / "production")
