"""Independent hand examples for accounting, censoring, temporal boundaries, and integrity."""

import math

import numpy as np
import pandas as pd
import pytest

from research.mim_lifecycle.analysis import (
    analyze,
    describe,
    select_inputs,
    summarize,
    trace_trade,
    validate_session,
)


def session():
    times = pd.date_range(
        "2024-01-08 09:31", periods=390, freq="min", tz="America/New_York"
    )
    return pd.DataFrame(
        dict(
            timestamp=times,
            day="2024-01-08",
            contract="MNQH24",
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1.0,
        )
    )


def set_bar(bars, index, close, high=None, low=None, opening=100.0):
    bars.loc[index, ["open", "high", "low", "close"]] = [
        opening,
        max(opening, close) if high is None else high,
        min(opening, close) if low is None else low,
        close,
    ]


def trade(bars, direction=1, kind="open", exit_index=9, entry_index=0):
    entry_event = bars.timestamp.iloc[entry_index]
    event = bars.timestamp.iloc[exit_index]
    reason, basis = {
        "open": ("REVERSAL", "bar_open"),
        "stop": ("CAT_STOP", "intrabar_unknown_within_event_minute"),
        "eod": ("EOD_CLOSE_PROXY", "session_close_proxy"),
    }[kind]
    fill = (
        95.0
        if kind == "stop"
        else float(bars.iloc[exit_index]["open" if kind == "open" else "close"])
    )
    entry_fill = float(bars.open.iloc[entry_index])
    gross = direction * (fill - entry_fill) * 2
    return dict(
        trade_id="synthetic-1",
        day="2024-01-08",
        contract="MNQH24",
        arm="A",
        delay=2,
        direction=direction,
        entry_event_timestamp=entry_event.isoformat(),
        entry_fill_timestamp=(entry_event - pd.Timedelta(minutes=1)).isoformat(),
        exit_event_timestamp=event.isoformat(),
        exit_fill_timestamp=(
            None
            if kind == "stop"
            else (
                event - pd.Timedelta(minutes=1) if kind == "open" else event
            ).isoformat()
        ),
        exit_fill_time_basis=basis,
        entry_fill=entry_fill,
        exit_fill=fill,
        quantity=1,
        gross=gross,
        costs=2.24,
        net=gross - 2.24,
        exit_reason=reason,
    )


def test_long_hand_path_recovery_and_checkpoint():
    bars = session()
    set_bar(bars, 0, 98, high=101, low=97)
    set_bar(bars, 1, 100, high=102, low=99)
    set_bar(bars, 2, 102, high=103, low=99)
    set_bar(bars, 4, 99, high=101, low=98)
    set_bar(bars, 5, 103, high=104, low=99)
    set_bar(bars, 9, 104, high=999, low=1, opening=104)
    t = trade(bars)
    paths, life, landmarks = trace_trade(t, bars, (5, 9, 10))
    assert life["mfe"] == 8
    assert life["mae"] == 6
    assert life["net"] == pytest.approx(5.76)
    assert life["first_negative_min_lower"] == 1
    assert life["first_positive_min_lower"] == 3
    assert life["first_recovery_min_lower"] == 3
    assert life["early_negative_close_30min"]
    assert paths[0]["net_mark"] == -2.24
    assert paths[-1]["net_mark"] == pytest.approx(5.76)
    m = landmarks[0]
    assert m["current_net"] == pytest.approx(-4.24)
    assert m["mfe"] == 6
    assert m["mae"] == 6
    assert m["remaining_net_change"] == pytest.approx(10)
    assert m["positive_net_close_seen"] and m["recovered_after_red"]
    assert landmarks[1]["at_risk"]  # Previous completed close at the exit open.
    assert landmarks[2]["already_exited"]
    assert not landmarks[2]["at_risk"]


def test_short_hand_path_and_fill_net():
    bars = session()
    set_bar(bars, 0, 102, high=103, low=99)
    set_bar(bars, 1, 98, high=101, low=97)
    set_bar(bars, 9, 96, high=100, low=96, opening=96)
    paths, life, _ = trace_trade(trade(bars, direction=-1), bars)
    assert life["mfe"] == 8
    assert life["mae"] == 6
    assert life["first_positive_min_lower"] == 2
    assert life["first_negative_min_lower"] == 1
    assert life["first_recovery_min_lower"] == 2
    assert paths[-1]["net_mark"] == pytest.approx(5.76)


def test_entry_bar_included_pre_entry_and_open_exit_ohlc_excluded():
    bars = session()
    set_bar(bars, 0, 100, high=10000, low=1)
    set_bar(bars, 5, 102, high=105, low=97)
    set_bar(bars, 9, 100, high=10000, low=1)
    paths, life, _ = trace_trade(trade(bars, entry_index=5), bars)
    assert life["mfe"] == 10
    assert life["mae"] == 6
    closes = [p for p in paths if p["kind"] == "completed_close"]
    assert len(closes) == 4
    assert closes[0]["timestamp"] == bars.timestamp.iloc[5].isoformat()
    assert closes[-1]["timestamp"] == bars.timestamp.iloc[8].isoformat()


def test_stop_excludes_ambiguous_prices_no_false_recovery_interval():
    bars = session()
    set_bar(bars, 4, 100, high=900, low=1)
    t = trade(bars, kind="stop", exit_index=4)
    paths, life, landmarks = trace_trade(t, bars, (4, 5, 6))
    assert life["mfe"] == 0
    assert life["mae"] == 10  # Known actual stop fill, not stop-bar low.
    assert life["net"] == pytest.approx(-12.24)
    assert life["duration_min_lower"] == 4
    assert life["duration_min_upper"] == 5
    assert life["stop_time_uncertain"] and life["excursions_observed_lower_bounds"]
    assert life["never_positive"] and life["first_positive_min_lower"] is None
    assert not life["recovered_after_first_negative"]
    assert paths[-1]["timestamp"] is None
    assert paths[-1]["elapsed_min_lower"] == 4
    assert landmarks[0]["at_risk"]
    assert not landmarks[0]["recovered_after_red"]
    assert landmarks[1]["already_exited"] and landmarks[1]["stop_interval_unobserved"]
    assert not landmarks[2]["stop_interval_unobserved"]


def test_known_positive_stop_fill_has_interval_and_separate_close_censor():
    bars = session()
    t = trade(bars, direction=-1, kind="stop", exit_index=4)
    _, life, landmarks = trace_trade(t, bars, (3,))
    assert life["first_positive_basis"] == "terminal_fill"
    assert life["first_positive_min_lower"] == 4
    assert life["first_positive_min_upper"] == 5
    assert life["never_positive_close"]
    assert life["first_positive_close_min_lower"] is None
    assert landmarks[0]["recovered_after_red"]


def test_eod_includes_final_bar_close_proxy():
    bars = session()
    set_bar(bars, 389, 103, high=108, low=94)
    paths, life, landmarks = trace_trade(
        trade(bars, kind="eod", exit_index=389), bars, (390, 391)
    )
    assert life["mfe"] == 16 and life["mae"] == 12
    assert life["valid_completed_closes"] == 390
    assert life["duration_min_lower"] == life["duration_min_upper"] == 390
    assert paths[-1]["net_mark"] == pytest.approx(3.76)
    assert landmarks[0]["at_risk"] and landmarks[1]["already_exited"]


def test_recovery_must_follow_negative_not_precede_it():
    bars = session()
    set_bar(bars, 0, 103)
    _, life, landmarks = trace_trade(trade(bars), bars, (5,))
    assert life["first_positive_min_lower"] == 1
    assert life["first_negative_min_lower"] == 2
    assert not life["recovered_after_first_negative"]
    assert landmarks[0]["positive_net_close_seen"]
    assert not landmarks[0]["recovered_after_red"]


def test_later_prices_cannot_change_checkpoint_features():
    bars = session()
    original = trade(bars)
    before = trace_trade(original, bars, (5,))[2][0]
    changed = bars.copy()
    set_bar(changed, 5, 200, high=300, low=1)
    set_bar(changed, 9, 110, opening=110)
    after = trace_trade(trade(changed), changed, (5,))[2][0]
    for field in (
        "current_net",
        "mfe",
        "mae",
        "positive_net_close_seen",
        "at_risk",
        "cohort",
    ):
        assert before[field] == after[field]
    assert before["final_winner"] != after["final_winner"]
    assert before["recovered_after_red"] != after["recovered_after_red"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "nonfinite",
        "ohlc",
        "offminute",
        "wrong_contract",
        "wrong_day",
    ],
)
def test_bad_selected_session_fails(fault):
    bars = session()
    if fault == "missing":
        bars = bars.iloc[1:]
    elif fault == "duplicate":
        bars.loc[1, "timestamp"] = bars.timestamp.iloc[0]
    elif fault == "nonfinite":
        bars.loc[1, "high"] = np.inf
    elif fault == "ohlc":
        bars.loc[1, "high"] = 99
    elif fault == "offminute":
        bars.loc[1, "timestamp"] += pd.Timedelta(seconds=1)
    elif fault == "wrong_contract":
        bars.loc[1, "contract"] = "MNQM24"
    else:
        bars.loc[1, "day"] = "2024-01-09"
    with pytest.raises(ValueError):
        validate_session(bars, "2024-01-08", "MNQH24")


def source_frames():
    bars = session()
    t = trade(bars)
    trades = pd.DataFrame([t])
    daily = pd.DataFrame(
        [
            dict(
                day=t["day"],
                contract=t["contract"],
                arm="A",
                delay=2,
                cost=2.24,
                quantity=1,
                gross=0.0,
                costs=2.24,
                net=-2.24,
            ),
            dict(
                day="2024-01-09",
                contract="MNQH24",
                arm="A",
                delay=2,
                cost=2.24,
                quantity=1,
                gross=0.0,
                costs=0.0,
                net=0.0,
            ),
        ]
    )
    return trades, daily, bars


def test_daily_reconciles_flats_and_repeat_numeric_outputs():
    trades, daily, bars = source_frames()
    trades = trades.drop(columns="trade_id")
    selected, grid = select_inputs(trades, daily, 1, 2, -2.24)
    paths, life, marks = analyze(selected, grid, bars)
    repeated = analyze(selected, grid, bars)
    for first, second in zip((paths, life, marks), repeated):
        pd.testing.assert_frame_equal(first, second)
    summary = summarize(life, marks, grid)
    assert summary["flat_sessions"] == 1
    assert summary["daily_net"] == pytest.approx(-2.24)
    assert summary["overall"]["never_positive"] == 1
    assert summary["groups"]["hindsight_top_5pct_trade"]["selected"]["count"] == 1
    assert (
        summary["landmarks"][0]["positive_history_cohorts"]["never_positive_close_yet"][
            "count"
        ]
        == 1
    )


@pytest.mark.parametrize(
    "fault",
    [
        "daily_net",
        "daily_costs",
        "duplicate_daily",
        "duplicate_trade",
        "nan_trade",
        "missing_field",
        "bad_direction",
        "quantity",
        "contract",
        "gross",
        "total",
    ],
)
def test_source_and_daily_fail_closed(fault):
    trades, daily, _ = source_frames()
    trades = trades.drop(columns="trade_id")
    expected_trades, expected_sessions, expected_net = 1, 2, -2.24
    if fault == "daily_net":
        daily.loc[1, "net"] = 1
    elif fault == "daily_costs":
        daily.loc[1, "costs"] = 1
    elif fault == "duplicate_daily":
        daily.loc[1, "day"] = daily.day.iloc[0]
    elif fault == "duplicate_trade":
        trades = pd.concat([trades, trades], ignore_index=True)
        expected_trades = 2
    elif fault == "nan_trade":
        trades.loc[0, "entry_fill"] = np.nan
    elif fault == "missing_field":
        trades = trades.drop(columns="exit_reason")
    elif fault == "bad_direction":
        trades.loc[0, "direction"] = 0
    elif fault == "quantity":
        trades.loc[0, "quantity"] = 2
    elif fault == "contract":
        daily.loc[0, "contract"] = "MNQM24"
    elif fault == "gross":
        trades.loc[0, "gross"] = 1
    elif fault == "total":
        expected_net = 0
    with pytest.raises(ValueError):
        select_inputs(trades, daily, expected_trades, expected_sessions, expected_net)


def test_exact_contract_join_does_not_fallback():
    trades, daily, bars = source_frames()
    bars["contract"] = "MNQM24"
    with pytest.raises(ValueError, match="Missing exact"):
        analyze(trades, daily, bars)


@pytest.mark.parametrize(
    "fault", ["entry_time", "exit_time", "fill", "basis", "naive", "net"]
)
def test_trade_timestamp_and_fill_integrity(fault):
    bars = session()
    t = trade(bars)
    if fault == "entry_time":
        t["entry_event_timestamp"] = t["entry_fill_timestamp"]
    elif fault == "exit_time":
        t["exit_fill_timestamp"] = t["exit_event_timestamp"]
    elif fault == "fill":
        t["exit_fill"] = 102
    elif fault == "basis":
        t["exit_fill_time_basis"] = "unknown"
    elif fault == "naive":
        t["entry_fill_timestamp"] = "2024-01-08 09:30"
    else:
        t["net"] = 200
    with pytest.raises(ValueError):
        trace_trade(t, bars)


def test_red_close_then_profitable_open_exit_at_same_timestamp_recovers():
    bars = session()
    set_bar(bars, 4, 99)
    set_bar(bars, 5, 104, opening=104)
    paths, life, marks = trace_trade(trade(bars, exit_index=5), bars, (5,))
    assert paths[-2]["timestamp"] == paths[-1]["timestamp"]
    assert paths[-2]["net_mark"] == pytest.approx(-4.24)
    assert paths[-1]["net_mark"] == pytest.approx(5.76)
    assert marks[0]["recovered_after_red"]
    assert not marks[0]["positive_net_close_seen"]
    assert marks[0]["mfe"] == 0  # Profitable terminal execution is still future.
    assert life["recovered_after_first_negative"]


def test_positive_closes_then_losing_open_fill_has_no_negative_close_history():
    bars = session()
    for index in range(5):
        set_bar(bars, index, 103)
    set_bar(bars, 5, 95, opening=95)
    paths, life, marks = trace_trade(trade(bars, exit_index=5), bars, (5,))
    assert all(p["net_mark"] > 0 for p in paths if p["kind"] == "completed_close")
    assert life["net"] == pytest.approx(-12.24)
    assert not life["had_negative"]
    assert life["first_negative_min_lower"] is None
    assert not life["recovered_after_first_negative"]
    assert not life["early_negative_close_30min"]
    assert marks[0]["positive_net_close_seen"]


def test_hindsight_ranking_cutoff_ties_ceil_and_contribution():
    # 71 trades => ceil(3.55) = 4; 71 daily dates => 67 selected.
    # Tied values cross both cutoffs; chronological / ID tie-breaks are explicit.
    all_bars, trades, daily = [], [], []
    dates = pd.bdate_range("2024-01-08", periods=71)
    for i, date in enumerate(dates):
        bars = session()
        day = date.strftime("%Y-%m-%d")
        bars["timestamp"] = pd.date_range(
            f"{day} 09:31", periods=390, freq="min", tz="America/New_York"
        )
        bars["day"] = day
        # descending high end: 100,90,80,70,70; low cutoff: ...4,4,4,2,1
        net = {
            0: 100.0,
            1: 90.0,
            2: 80.0,
            3: 70.0,
            4: 70.0,
            65: 4.0,
            66: 4.0,
            67: 4.0,
            68: 3.0,
            69: 2.0,
            70: 1.0,
        }.get(i, 69.0 - i)
        exit_fill = 100 + (net + 2.24) / 2
        set_bar(bars, 9, exit_fill, opening=exit_fill)
        t = trade(bars)
        t.update(day=day, trade_id=f"trade-{i:03d}")
        trades.append(t)
        daily.append(dict(day=day, net=t["net"], costs=2.24))
        all_bars.append(bars)
    # Reverse input order to ensure tie behavior comes from declared sort keys.
    _, life, marks = analyze(
        pd.DataFrame(trades[::-1]),
        pd.DataFrame(daily[::-1]),
        pd.concat(all_bars, ignore_index=True),
    )
    assert set(life.loc[life.hindsight_top_5pct_trade, "trade_id"]) == {
        "trade-000",
        "trade-001",
        "trade-002",
        "trade-003",
    }
    chosen_days = set(life.loc[life.hindsight_top_67_day, "day"])
    expected_days = {date.strftime("%Y-%m-%d") for date in dates[:67]}
    assert chosen_days == expected_days
    summary = summarize(life, marks, pd.DataFrame(daily[::-1]))
    assert {row["day"] for row in summary["hindsight_top_67_days"]} == expected_days
    for field in ("hindsight_top_5pct_trade", "hindsight_top_67_day"):
        selected, remainder = (
            summary["groups"][field][name] for name in ("selected", "remainder")
        )
        assert selected["net"] + remainder["net"] == pytest.approx(life.net.sum())
        assert selected["net_contribution"] + remainder[
            "net_contribution"
        ] == pytest.approx(1)
        assert selected["count"] + remainder["count"] == 71
