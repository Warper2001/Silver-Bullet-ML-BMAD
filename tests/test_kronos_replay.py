"""Synthetic mechanics tests without market data or model downloads."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from research.kronos_replay.engine import (
    Costs,
    ReplayResult,
    Session,
    execute,
    run_replay,
)
from research.kronos_replay.fixtures import bundled_fixture
from research.kronos_replay.providers import StubProvider
from research.kronos_replay.__main__ import main, run
from tools.kronos_inference_pilot import digest


def test_hand_accounting_hold_reversal_flatten():
    r, c = ReplayResult(), Costs(fee_per_side=0.5, slippage_ticks=1)
    t = pd.Timestamp("2026-03-09 10:00", tz="America/New_York")
    execute(r, "kronos", 1, 100, t, "A", c, "test")
    a = r.accounts["kronos"]
    a.mark = 102
    assert a.snapshot()["equity"] == 3  # 3.50 unrealized less .50 fee
    execute(r, "kronos", 1, 102, t, "A", c, "test")
    assert len(r.fills) == 1
    execute(r, "kronos", -1, 103, t, "A", c, "test")
    assert a.realized == 5 and a.fees == 1.5 and a.turnover == 3
    a.mark = 101
    assert a.snapshot()["equity"] == 7
    execute(r, "kronos", 0, 101, t, "A", c, "test")
    assert a.realized == 8 and a.snapshot()["equity"] == 6
    assert a.slippage == 2 and a.turnover == 4
    assert a.snapshot()["drawdown"] == 1


def test_fixture_warmup_dst_short_session_delayed_fill_and_flat():
    bars, sessions = bundled_fixture()
    r = run_replay(bars, sessions, StubProvider())
    assert r.status == "COMPLETE", r.error
    assert len(r.decisions) == 6
    assert len(r.forecasts) == 6
    assert len(r.aggregated) == 136
    assert sessions[0].opening.utcoffset() != sessions[-1].opening.utcoffset()
    first = pd.Timestamp(r.decisions[0]["timestamp"])
    assert first == sessions[-1].opening + pd.Timedelta(minutes=15)
    assert pd.Timestamp(r.fills[0]["timestamp"]) == first + pd.Timedelta(
        minutes=1
    )
    assert all(a.position == 0 for a in r.accounts.values())
    assert not r.pending
    assert all(
        pd.Timestamp(f["timestamp"])
        == sessions[-1].close - pd.Timedelta(minutes=1)
        for f in r.fills
        if f["reason"] == "session_flatten"
    )
    assert r.accounts["flat"].snapshot()["equity"] == 0


@pytest.mark.parametrize("latency,delay", [(0, 1), (60, 2), (61, 2), (0.1, 1)])
def test_strict_availability(latency, delay):
    bars, sessions = bundled_fixture()
    r = run_replay(
        bars, sessions, StubProvider(), Costs(latency_seconds=latency)
    )
    assert pd.Timestamp(r.fills[0]["timestamp"]) == pd.Timestamp(
        r.decisions[0]["timestamp"]
    ) + pd.Timedelta(minutes=delay)


def test_pending_cancelled_at_flatten():
    bars, sessions = bundled_fixture()
    r = run_replay(
        bars, sessions, StubProvider(), Costs(latency_seconds=99999)
    )
    assert r.status == "COMPLETE" and not r.fills and not r.pending


def test_causality_and_determinism():
    bars, sessions = bundled_fixture()
    baseline = run_replay(bars, sessions, StubProvider())
    changed = deepcopy(bars)
    cutoff = sessions[-1].opening + pd.Timedelta(minutes=35)
    for row in changed:
        if row["timestamp"] >= cutoff:
            for key in ["open", "high", "low", "close"]:
                row[key] += 1000
    altered = run_replay(changed, sessions, StubProvider())
    assert baseline.decisions == altered.decisions
    assert [
        f for f in baseline.fills if pd.Timestamp(f["timestamp"]) < cutoff
    ] == [f for f in altered.fills if pd.Timestamp(f["timestamp"]) < cutoff]
    again = run_replay(bars, sessions, StubProvider())
    assert (
        baseline.decisions == again.decisions and baseline.fills == again.fills
    )
    for a, b in zip(baseline.forecasts, again.forecasts):
        pd.testing.assert_frame_equal(a["path"], b["path"])


@pytest.mark.parametrize(
    "kind", ["gap", "duplicate", "nan", "geometry", "roll", "last"]
)
def test_invalid_data_retains_exposure_without_fabricated_exit(kind):
    bars, sessions = bundled_fixture()
    n = 1950 + 20
    if kind == "gap":
        bars.pop(n)
    elif kind == "duplicate":
        bars.insert(n, bars[n - 1].copy())
    elif kind == "nan":
        bars[n]["close"] = float("nan")
    elif kind == "geometry":
        bars[n]["high"] = 1
    elif kind == "roll":
        bars[n]["contract"] = "NEXT"
    else:
        bars.pop()
    r = run_replay(bars, sessions, StubProvider())
    assert r.status == "INCOMPLETE" and r.error
    assert r.accounts["kronos"].position == 1
    assert not any(f["reason"] == "session_flatten" for f in r.fills)


def test_missing_data_retains_pending():
    bars, sessions = bundled_fixture()
    bars.pop(1950 + 16)
    r = run_replay(bars, sessions, StubProvider(), Costs(latency_seconds=180))
    assert r.status == "INCOMPLETE" and len(r.pending) == 3
    assert not r.fills


def test_roll_restarts_warmup():
    bars, sessions = bundled_fixture()
    for row in bars[1950:]:
        row["contract"] = "NEXT"
    r = run_replay(bars, sessions, StubProvider())
    assert r.status == "COMPLETE" and not r.decisions


class FixedProvider:
    def __init__(self, deltas=(0, 0, 0), bad=None):
        self.deltas, self.bad = deltas, bad

    def forecast(self, context, future):
        if self.bad == "exception":
            raise RuntimeError("synthetic failure")
        if self.bad == "none":
            return None
        paths = StubProvider().forecast(context, future)
        for path, delta in zip(paths, self.deltas):
            price = float(context.close.iloc[-1]) + delta
            path.loc[:, ["open", "high", "low", "close"]] = price
        if self.bad == "nan":
            paths[1].iloc[0, 0] = float("nan")
        if self.bad == "schema":
            paths[0] = paths[0].drop(columns="amount")
        if self.bad == "index":
            paths[0].index = paths[0].index + pd.Timedelta(minutes=1)
        return paths


@pytest.mark.parametrize(
    "deltas,target",
    [((0, 0, 0), 0), ((-3, 1, 2), 0), ((-3, -2, 10), 1), ((3, 2, -10), -1)],
)
def test_all_three_paths_and_exact_equality(deltas, target):
    bars, sessions = bundled_fixture()
    r = run_replay(bars, sessions, FixedProvider(deltas))
    assert r.status == "COMPLETE"
    assert all(
        d["target"] == target for d in r.decisions if d["arm"] == "kronos"
    )


@pytest.mark.parametrize(
    "bad", ["exception", "none", "nan", "schema", "index"]
)
def test_invalid_forecast_is_flat_not_replay_failure(bad):
    bars, sessions = bundled_fixture()
    r = run_replay(bars, sessions, FixedProvider(bad=bad))
    assert r.status == "COMPLETE"
    assert all(
        d["target"] == 0 and d["error"]
        for d in r.decisions
        if d["arm"] == "kronos"
    )
    if bad != "exception":
        assert r.forecasts


def test_manifests_success_failure_and_fresh_destination(
    tmp_path, monkeypatch
):
    output = tmp_path / "success"
    result = run(output)
    assert result.status == "COMPLETE"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["scope"] == "SYNTHETIC_MECHANICS_ONLY"
    assert (
        not manifest["economic_evaluation"]
        and not manifest["trading_authorized"]
    )
    assert all(
        digest(output / name) == hash_
        for name, hash_ in manifest["artifacts"].items()
    )
    assert (output / "COMPLETE.json").exists()
    assert main(["--output-dir", str(output)]) == 1

    def failed():
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(
        "research.kronos_replay.__main__.CachedProvider", failed
    )
    failed_output = tmp_path / "failed"
    assert (
        main(["--provider", "cached", "--output-dir", str(failed_output)]) == 1
    )
    assert (failed_output / "INCOMPLETE.json").exists()
    assert not (failed_output / "COMPLETE.json").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latency_seconds": -1},
        {"fee_per_side": float("nan")},
        {"slippage_ticks": 1.5},
    ],
)
def test_bad_costs(kwargs):
    with pytest.raises(ValueError):
        Costs(**kwargs)


def test_bad_sessions_and_warmup():
    with pytest.raises(ValueError):
        Session(
            pd.Timestamp("2026-03-09 09:30"), pd.Timestamp("2026-03-09 10:00")
        )
    bars, sessions = bundled_fixture()
    r = run_replay(bars[:390], sessions[:1], StubProvider())
    assert r.status == "COMPLETE" and not r.decisions
    assert (
        run_replay(bars, sessions[::-1], StubProvider()).status == "INCOMPLETE"
    )


def test_unrealized_loss_zero_origin_drawdown():
    r = ReplayResult()
    when = pd.Timestamp("2026-03-09T10:00:00-04:00")
    execute(r, "kronos", 1, 100, when, "A", Costs(), "test")
    r.accounts["kronos"].mark = 98
    state = r.accounts["kronos"].snapshot()
    assert state["unrealized"] == -4.5
    assert state["equity"] == -5 and state["drawdown"] == 5
    assert state["max_drawdown"] == 5 and state["realized"] == 0


class ContextProvider:
    def forecast(self, context, future):
        # Every context row contributes, making leaked future rows detectable.
        delta = float(context.close.mean()) / 100000
        return FixedProvider((delta, delta, delta)).forecast(context, future)


def test_future_mutation_preserves_forecasts_and_pending():
    bars, sessions = bundled_fixture()
    cutoff = 1950 + 20
    changed = deepcopy(bars)
    for row in changed[cutoff:]:
        for key in ["open", "high", "low", "close"]:
            row[key] += 500
    boundary = bars[cutoff]["timestamp"]
    before = run_replay(bars, sessions, ContextProvider())
    after = run_replay(changed, sessions, ContextProvider())
    assert before.status == after.status == "COMPLETE"
    for key, time_key in (("decisions", "timestamp"), ("fills", "timestamp")):
        assert [
            r
            for r in getattr(before, key)
            if pd.Timestamp(r[time_key]) < boundary
        ] == [
            r
            for r in getattr(after, key)
            if pd.Timestamp(r[time_key]) < boundary
        ]
    for a, b in zip(before.forecasts, after.forecasts):
        if pd.Timestamp(a["decision"]) < boundary:
            pd.testing.assert_frame_equal(a["path"], b["path"])
    assert not before.forecasts[-1]["path"].equals(after.forecasts[-1]["path"])
    # Supply different full tails; abort at the same later bad minute.
    bars[cutoff + 5]["volume"] = -1
    changed[cutoff + 5]["volume"] = -1
    costs = Costs(latency_seconds=3600)
    before = run_replay(bars, sessions, ContextProvider(), costs)
    after = run_replay(changed, sessions, ContextProvider(), costs)
    assert before.status == after.status == "INCOMPLETE"
    assert before.processed_minutes == after.processed_minutes == cutoff + 5
    assert before.pending == after.pending and len(before.pending) == 3
    assert before.decisions == after.decisions and before.fills == after.fills
    for a, b in zip(before.forecasts, after.forecasts):
        pd.testing.assert_frame_equal(a["path"], b["path"])


@pytest.mark.parametrize("field", ["volume", "price"])
def test_aggregate_overflow_is_incomplete(field):
    bars, sessions = bundled_fixture()
    for row in bars[:15]:
        if field == "volume":
            row["volume"] = 1e308
        else:
            for key in ["open", "high", "low", "close"]:
                row[key] = 1e308
    result = run_replay(bars, sessions, StubProvider())
    assert result.status == "INCOMPLETE"
    assert "aggregate" in result.error
    assert result.processed_minutes == 14 and not result.aggregated
    assert not result.decisions and not result.fills


class SequenceProvider:
    def __init__(self, targets):
        self.targets = iter(targets)

    def forecast(self, context, future):
        target = next(self.targets)
        if isinstance(target, str):
            return FixedProvider(bad=target).forecast(context, future)
        return FixedProvider((target,) * 3).forecast(context, future)


@pytest.mark.parametrize("failure", ["slippage", "fees"])
def test_failed_position_change_preserves_exposure_and_fills(failure):
    bars, sessions = bundled_fixture()
    # In the fee case the close is affordable, but the reversal's new leg
    # overflows; neither leg may be committed.
    costs = (
        Costs(slippage_ticks=100000)
        if failure == "slippage"
        else Costs(fee_per_side=6e307)
    )
    result = run_replay(bars, sessions, SequenceProvider([1, -1]), costs)
    assert result.status == "INCOMPLETE"
    account = result.accounts["kronos"]
    assert account.position == 1 and account.turnover == 1
    assert account.realized == 0 and account.fees == costs.fee_per_side
    fills = [f for f in result.fills if f["arm"] == "kronos"]
    assert len(fills) == 1 and fills[0]["action"] == "open"
    assert result.pending[0]["arm"] == "kronos"
    assert result.pending[0]["target"] == -1
    assert all(f["price"] > 0 for f in result.fills)
    # The incomplete result remains serializable with the prior valid mark.
    json.dumps(result.report(), default=str, allow_nan=False)


def test_nonpositive_initial_fill_never_opens():
    bars, sessions = bundled_fixture()
    result = run_replay(
        bars, sessions, FixedProvider((-1,) * 3), Costs(slippage_ticks=100000)
    )
    assert result.status == "INCOMPLETE" and not result.fills
    assert result.accounts["kronos"].position == 0


def test_protocol_helper_hashes_versions_and_empty_table_headers(
    tmp_path, monkeypatch
):
    from research.kronos_replay.__main__ import TABLE_SCHEMAS
    from tools import kronos_inference_pilot as pilot
    from tools import trading_model_readiness as readiness
    from pathlib import Path

    output = tmp_path / "empty_fills"
    assert run(output, costs=Costs(latency_seconds=99999)).status == "COMPLETE"
    assert pd.read_csv(output / "fills.csv").empty
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["code_hashes"][
        "tools/kronos_inference_pilot.py"
    ] == digest(Path(pilot.__file__))
    assert protocol["code_hashes"][
        "tools/trading_model_readiness.py"
    ] == digest(Path(readiness.__file__))
    assert protocol["runtime_versions"]["pandas"] == pd.__version__
    assert {
        "python",
        "numpy",
        "torch",
        "huggingface_hub",
        "safetensors",
        "einops",
    } <= protocol["runtime_versions"].keys()

    def unavailable():
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(
        "research.kronos_replay.__main__.CachedProvider", unavailable
    )
    output = tmp_path / "init_failure"
    assert run(output, "cached").status == "INCOMPLETE"
    for name, columns in TABLE_SCHEMAS.items():
        table = pd.read_csv(output / f"{name}.csv")
        assert table.empty and list(table.columns) == columns


def test_hand_checked_aggregation_and_provider_context():
    bars, sessions = bundled_fixture()
    # Anchor synthetic sessions at 09:32 to detect clock-quarter resampling.
    shift = pd.Timedelta(minutes=2)
    sessions = [Session(s.opening + shift, s.close + shift) for s in sessions]
    for n, row in enumerate(bars):
        row.update(
            timestamp=row["timestamp"] + shift,
            open=100 + n,
            high=103 + n,
            low=99 + n,
            close=102 + n,
            volume=n + 1,
        )

    class RecordingProvider(StubProvider):
        def __init__(self):
            self.contexts = []
            self.futures = []

        def forecast(self, context, future):
            self.contexts.append(context.copy())
            self.futures.append(future.copy())
            return super().forecast(context, future)

    provider = RecordingProvider()
    result = run_replay(bars, sessions, provider)
    assert result.status == "COMPLETE"
    first = result.aggregated[0]
    assert [first[k] for k in ["open", "high", "low", "close", "volume"]] == [
        100,
        117,
        99,
        116,
        120,
    ]
    assert first["amount"] == 12960  # 120 * (100 + 117 + 99 + 116) / 4
    assert first["timestamp"] == sessions[0].opening + pd.Timedelta(minutes=15)
    context = provider.contexts[0]
    assert context.shape == (128, 6)
    assert list(context.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    assert str(context.index.tz) == "America/New_York"
    assert context.index[0] == sessions[0].opening + pd.Timedelta(minutes=60)
    assert context.index[-1] == sessions[-1].opening + pd.Timedelta(minutes=15)
    assert context.iloc[0].to_dict() == {
        "open": 145,
        "high": 162,
        "low": 144,
        "close": 161,
        "volume": 795,
        "amount": 121635,
    }
    assert provider.futures[0][0] == context.index[-1] + pd.Timedelta(
        minutes=15
    )
    assert provider.futures[0][-1] == context.index[-1] + pd.Timedelta(
        minutes=60
    )


@pytest.mark.parametrize("reference,target", [(99, 1), (101, -1), (100, 0)])
def test_momentum_uses_exact_four_bar_lookback(reference, target):
    bars, sessions = bundled_fixture()
    for row in bars:
        for key in ["open", "high", "low", "close"]:
            row[key] = 100
    # The previous 1/2/3/5-bar closes point in the opposite direction.
    other = 101 if reference <= 100 else 99
    for n in [1949, 1934, 1919, 1889]:
        bars[n].update(close=other, high=max(100, other), low=min(100, other))
    bars[1904].update(
        close=reference, high=max(100, reference), low=min(100, reference)
    )
    result = run_replay(bars, sessions, FixedProvider())
    decisions = [d for d in result.decisions if d["arm"] == "momentum"]
    assert decisions[0]["target"] == target
    when = pd.Timestamp(decisions[0]["timestamp"]) + pd.Timedelta(minutes=1)
    fills = [f for f in result.fills if pd.Timestamp(f["timestamp"]) == when]
    assert len(fills) == abs(target)
    if target:
        assert fills[0]["arm"] == "momentum" and fills[0]["side"] == target
    assert not any(f["arm"] in {"kronos", "flat"} for f in result.fills)


@pytest.mark.parametrize("bad", ["nan", "exception"])
def test_invalid_forecast_closes_existing_exposure_after_latency(bad):
    bars, sessions = bundled_fixture()
    costs = Costs(latency_seconds=60)
    result = run_replay(bars, sessions, SequenceProvider([1, bad]), costs)
    control = run_replay(bars, sessions, SequenceProvider([1, 1]), costs)
    assert result.status == "COMPLETE"
    fills = [f for f in result.fills if f["arm"] == "kronos"]
    assert [f["action"] for f in fills] == ["open", "close"]
    assert [pd.Timestamp(f["timestamp"]) for f in fills] == [
        sessions[-1].opening + pd.Timedelta(minutes=17),
        sessions[-1].opening + pd.Timedelta(minutes=32),
    ]
    account = result.accounts["kronos"]
    assert (
        account.position == 0 and account.fees == 1 and account.slippage == 1
    )
    decisions = [d for d in result.decisions if d["arm"] == "kronos"]
    assert decisions[1]["target"] == 0 and decisions[1]["error"]
    assert len(result.forecasts) == (6 if bad == "nan" else 3)
    if bad == "nan":
        assert pd.isna(result.forecasts[4]["path"].iloc[0, 0])
    assert [f for f in result.fills if f["arm"] != "kronos"] == [
        f for f in control.fills if f["arm"] != "kronos"
    ]
    assert [d for d in result.decisions if d["arm"] != "kronos"] == [
        d for d in control.decisions if d["arm"] != "kronos"
    ]


def test_queued_targets_execute_in_order_and_flatten_cancels_last():
    bars, sessions = bundled_fixture()
    opening = sessions[-1].opening
    sessions[-1] = Session(opening, opening + pd.Timedelta(minutes=150))
    for n in range(90, 150):
        bars.append(
            dict(
                timestamp=opening + pd.Timedelta(minutes=n),
                contract="SYNTHETIC_MNQ",
                open=20010,
                high=20011,
                low=20009,
                close=20010,
                volume=10,
            )
        )
    result = run_replay(
        bars,
        sessions,
        SequenceProvider([1, -1, 0, 1, -1, 1]),
        Costs(latency_seconds=70 * 60),
    )
    assert result.status == "COMPLETE"
    fills = [f for f in result.fills if f["arm"] == "kronos"]
    assert [
        (
            int((pd.Timestamp(f["timestamp"]) - opening).total_seconds() / 60),
            f["action"],
            f["side"],
        )
        for f in fills
    ] == [
        (86, "open", 1),
        (101, "close", -1),
        (101, "open", -1),
        (116, "close", 1),
        (131, "open", 1),
        (146, "close", -1),
        (146, "open", -1),
        (149, "close", 1),
    ]
    assert fills[-1]["reason"] == "session_flatten"
    assert result.accounts["kronos"].turnover == 8
    assert result.accounts["kronos"].fees == 4
    assert result.accounts["kronos"].slippage == 4
    assert not result.pending and result.accounts["kronos"].position == 0
