import numpy as np
import pandas as pd
import pytest
from research.mim_diagnostics.analysis import reconstruct, half_hour, drawdown
from research.mim_lifecycle.analysis import trace_trade, validate_session


def fixtures(stop=False, reversal=False, opposite=False):
    day = "2024-03-11"
    contract = "MNQH24"

    def event(n, fill, pos, quantity, reason, basis="bar_open"):
        t = pd.Timestamp(f"{day} 10:{n:02d}", tz="America/New_York")
        return dict(
            day=day,
            contract=contract,
            event_timestamp=t.isoformat(),
            modeled_fill_timestamp=(
                None
                if basis.startswith("intrabar")
                else (
                    t - pd.Timedelta(minutes=1) if basis == "bar_open" else t
                ).isoformat()
            ),
            fill_time_basis=basis,
            fill=fill,
            quantity=quantity,
            position_after=pos,
            costs=quantity * 1.12,
            reason=reason,
        )

    rows = [event(2, 100, 1, 1, "ENTRY")]
    if reversal:
        rows.extend(
            [
                event(3, 105, -1, 2, "REVERSAL"),
                event(4, 102, 0, 1, "CAT_STOP", "intrabar_unknown_within_event_minute"),
            ]
        )
    elif opposite:
        rows.extend(
            [
                event(2, 95, 0, 1, "CAT_STOP", "intrabar_unknown_within_event_minute"),
                event(3, 101, -1, 1, "REVERSAL"),
                event(4, 103, 0, 1, "CAT_STOP", "intrabar_unknown_within_event_minute"),
            ]
        )
    else:
        rows.append(
            event(
                2 if stop else 4,
                95,
                0,
                1,
                "CAT_STOP",
                "intrabar_unknown_within_event_minute",
            )
        )
    ledger = pd.DataFrame(rows)
    ledger["source_row"] = range(len(ledger))
    expected = []
    opening = None
    for row in rows:
        if opening:
            gross = opening["position_after"] * (row["fill"] - opening["fill"]) * 2
            expected.append(
                dict(
                    day=day,
                    contract=contract,
                    direction=opening["position_after"],
                    quantity=1,
                    entry_event_timestamp=opening["event_timestamp"],
                    entry_fill_timestamp=opening["modeled_fill_timestamp"],
                    entry_fill=opening["fill"],
                    exit_event_timestamp=row["event_timestamp"],
                    exit_fill_timestamp=row["modeled_fill_timestamp"],
                    exit_fill_time_basis=row["fill_time_basis"],
                    exit_fill=row["fill"],
                    exit_reason=row["reason"],
                    gross=gross,
                    costs=2.24,
                    net=gross - 2.24,
                )
            )
        opening = row if row["position_after"] else None
    return (
        ledger,
        pd.DataFrame([dict(day=day, contract=contract)]),
        pd.DataFrame(expected),
    )


@pytest.mark.parametrize(
    "kwargs,transitions,nets",
    [
        ({"stop": True}, ["entry_from_flat", "exit_to_flat"], [-12.24]),
        (
            {"reversal": True},
            ["entry_from_flat", "true_reversal", "exit_to_flat"],
            [7.76, 3.76],
        ),
        (
            {"opposite": True},
            [
                "entry_from_flat",
                "exit_to_flat",
                "opposite_entry_after_flat",
                "exit_to_flat",
            ],
            [-12.24, -6.24],
        ),
    ],
)
def test_execution_order_and_costs(kwargs, transitions, nets):
    ledger, daily, source = fixtures(**kwargs)
    events, trades = reconstruct(ledger, daily, source)
    assert events.transition.tolist() == transitions
    assert trades.net.tolist() == pytest.approx(nets)
    assert events.costs.sum() == pytest.approx(trades.costs.sum())
    assert events.realized_net.sum() == pytest.approx(sum(nets))
    assert events.source_row.tolist() == list(range(len(ledger)))


@pytest.mark.parametrize(
    "mutation",
    [
        "join",
        "quantity",
        "costs",
        "missing",
        "timestamp",
        "position",
        "duplicate",
        "unclosed",
        "nonfinite",
        "order",
    ],
)
def test_bad_execution_fails_closed(mutation):
    l, d, t = fixtures(reversal=True)
    if mutation == "join":
        l.loc[0, "contract"] = "MNQM24"
    elif mutation == "quantity":
        l.loc[0, "quantity"] = 2
    elif mutation == "costs":
        l.loc[0, "costs"] = 0
    elif mutation == "missing":
        l = l.drop(columns="source_row")
    elif mutation == "timestamp":
        l.loc[0, "modeled_fill_timestamp"] = "2024-03-11T10:02:00-04:00"
    elif mutation == "position":
        l.loc[0, "position_after"] = 2
    elif mutation == "duplicate":
        l = pd.concat([l, l.iloc[[0]]])
    elif mutation == "unclosed":
        l = l.iloc[:-1]
    elif mutation == "nonfinite":
        l.loc[0, "fill"] = np.nan
    elif mutation == "order":
        l = l.iloc[[1, 0, 2]]
    with pytest.raises(ValueError):
        reconstruct(l, d, t)


def test_completed_minute_halfhours_and_dst():
    assert half_hour("2024-03-08T10:00:00-05:00") == "09:30-10:00"
    assert half_hour("2024-03-11T10:01:00-04:00") == "10:00-10:30"
    assert half_hour("2024-03-11T16:00:00-04:00") == "15:30-16:00"
    with pytest.raises(ValueError):
        half_hour("2024-03-11T09:30:00-04:00")
    with pytest.raises(ValueError):
        half_hour("2024-03-11T10:00:00")


def test_drawdown_initial_zero_and_unresolved_duration():
    d = pd.DataFrame(dict(net=[-2.0, 1.0, -3.0, -1.0], day=["a", "b", "c", "d"]))
    dd = drawdown(d, "net", "day")
    assert dd["depth"] == 4
    assert dd["peak_timestamp"] == "b"
    assert dd["longest_underwater_samples"] == 2
    assert dd["unrecovered_at_end"]


def test_stop_excursion_invariant_to_postexit_prices():
    l, d, t = fixtures(stop=True)
    _, trades = reconstruct(l, d, t)
    trade = trades.iloc[0].to_dict()
    b = pd.DataFrame(
        dict(
            timestamp=pd.date_range(
                "2024-03-11 09:31", periods=390, freq="min", tz="America/New_York"
            ),
            day="2024-03-11",
            contract="MNQH24",
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1,
        )
    )
    _, before, _ = trace_trade(trade, b, checkpoints=())
    b.loc[
        b.timestamp >= pd.Timestamp(trade["exit_event_timestamp"]),
        ["high", "low", "close"],
    ] = [10000.0, 1.0, 1000.0]
    _, after, _ = trace_trade(trade, b, checkpoints=())
    assert before == after
    assert after["mfe"] == 0 and after["mae"] == 10
    assert after["duration_min_lower"] == 0 and after["duration_min_upper"] == 1


@pytest.mark.parametrize("mode", ["missing", "duplicate", "contract", "nan"])
def test_exact_session_validation(mode):
    b = pd.DataFrame(
        dict(
            timestamp=pd.date_range(
                "2024-11-04 09:31", periods=390, freq="min", tz="America/New_York"
            ),
            day="2024-11-04",
            contract="MNQZ24",
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1,
        )
    )
    if mode == "missing":
        b = b.iloc[:-1]
    elif mode == "duplicate":
        b.loc[1, "timestamp"] = b.loc[0, "timestamp"]
    elif mode == "contract":
        b.loc[0, "contract"] = "MNQH25"
    else:
        b.loc[0, "close"] = np.nan
    with pytest.raises(ValueError):
        validate_session(b, "2024-11-04", "MNQZ24")


def test_identical_execution_input_is_deterministic():
    inputs = fixtures(reversal=True)
    first = reconstruct(*inputs)
    second = reconstruct(*inputs)
    for left, right in zip(first, second):
        assert left.to_csv(index=False) == right.to_csv(index=False)
