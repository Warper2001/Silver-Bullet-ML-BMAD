"""Small hand-accounted crossing buckets and exported-field corruption tests."""

import pandas as pd
import pytest
from research.mim_diagnostics import analysis as n, artifacts as a
from .test_accounting import fixtures


def crossing_tables():
    ledger, daily, source = fixtures()
    for frame in (ledger, source):
        for col in frame:
            if "timestamp" in col:
                frame[col] = frame[col].map(
                    lambda v: (
                        v.replace("10:02", "10:30")
                        .replace("10:01", "10:29")
                        .replace("10:04", "10:32")
                        if isinstance(v, str)
                        else v
                    )
                )
    events, trades = n.reconstruct(ledger, daily, source)
    bars = pd.DataFrame(
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
    for clock, price in [("10:30", 102.0), ("10:31", 103.0)]:
        ix = bars.timestamp.dt.strftime("%H:%M").eq(clock)
        bars.loc[ix, ["high", "close"]] = price
    rows, exc = n.trade_rows(trades.iloc[0].to_dict(), bars)
    accrual = pd.DataFrame(rows)
    for col in ("exposure_minutes_upper", "exposure_minutes_lower"):
        trades[col] = accrual[col].sum()
    for col, value in exc.items():
        trades[col] = value
    daily["gross"] = -10.0
    daily["costs"] = 2.24
    daily["net"] = -12.24
    daily["marked_equity_net"] = -12.24
    minute = (
        bars[["day", "contract", "timestamp", "close"]]
        .merge(
            accrual[
                [
                    "timestamp",
                    "gross",
                    "costs",
                    "net",
                    "exposure_minutes_upper",
                    "exposure_minutes_lower",
                ]
            ],
            on="timestamp",
            how="left",
        )
        .fillna(0.0)
    )
    minute["accrual_half_hour"] = minute.timestamp.map(n.half_hour)
    minute["marked_equity_net"] = minute.net.cumsum()
    rg = events.copy()
    rg["timestamp"] = pd.to_datetime(rg.event_timestamp, utc=True).dt.tz_convert(
        "America/New_York"
    )
    minute = minute.merge(
        rg[["timestamp", "realized_gross"]], on="timestamp", how="left"
    ).fillna(0.0)
    for target, col in [
        ("session_realized_gross", "realized_gross"),
        ("session_costs", "costs"),
        ("session_marked_gross", "gross"),
        ("session_marked_net", "net"),
    ]:
        minute[target] = minute[col].cumsum()
    minute["session_realized_net"] = (
        minute.session_realized_gross - minute.session_costs
    )
    minute["unrealized_gross"] = (
        minute.session_marked_gross - minute.session_realized_gross
    )
    alternatives = pd.DataFrame(
        [
            dict(
                day="2024-03-11",
                arm="A",
                delay=2,
                cost=2.24,
                gross=-10.0,
                costs=2.24,
                net=-12.24,
            ),
            dict(
                day="2024-03-11",
                arm="A",
                delay=1,
                cost=3.24,
                gross=20.0,
                costs=3.24,
                net=16.76,
            ),
            dict(
                day="2024-03-11",
                arm="R",
                delay=2,
                cost=2.24,
                gross=999.0,
                costs=0.0,
                net=999.0,
            ),
        ]
    )
    tables = dict(
        events=events,
        trades=trades,
        minute=minute,
        accrual=accrual,
        daily=daily,
        partitions=n.attribution(trades, accrual),
        scenarios=n.recorded_scenarios(alternatives),
    )
    _, _, _, summary = a.expected_details(
        trades, events, minute, daily, bars, alternatives
    )
    return tables, summary, bars, alternatives


def test_crossing_halfhour_hand_accounting_and_scenarios():
    tables, summary, bars, source = crossing_tables()
    accrual = tables["accrual"]
    assert accrual.gross.tolist() == [4.0, 2.0, -16.0]
    assert accrual.costs.tolist() == [1.12, 0.0, 1.12]
    assert accrual.entry_count.tolist() == [1, 0, 0]
    part = (
        tables["partitions"]
        .query("basis == 'minute_accrual' and dimension == 'accrual_half_hour'")
        .set_index("group")
    )
    assert part.loc["10:00-10:30", "net"] == pytest.approx(2.88)
    assert part.loc["10:30-11:00", "net"] == pytest.approx(-15.12)
    assert part.loc["10:00-10:30", "count"] == 1
    assert part.loc["10:30-11:00", "count"] == 0
    assert len(part) == 13
    assert tables["scenarios"].net.tolist() == pytest.approx([16.76, -12.24])
    assert summary["drawdown_minute"]["depth"] == pytest.approx(17.12)
    assert tables["trades"].duration_min_lower.iloc[0] == 2
    assert tables["trades"].duration_min_upper.iloc[0] == 3
    a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize(
    "field",
    [
        "gross",
        "costs",
        "net",
        "entry_count",
        "trade_id",
        "entry_order",
        "year",
        "entry_half_hour",
        "accrual_half_hour",
        "timestamp",
        "exposure_minutes_lower",
        "eventual_losers_net_contribution",
    ],
)
def test_accrual_corruption_detected(field):
    tables, summary, bars, source = crossing_tables()
    frame = tables["accrual"]
    if field == "timestamp":
        frame.loc[0, field] = frame.loc[1, field]
    elif field in ("trade_id", "entry_order", "entry_half_hour", "accrual_half_hour"):
        frame.loc[0, field] = "wrong"
    else:
        frame.loc[0, field] += 1
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize(
    "table,field",
    [
        ("partitions", "net"),
        ("partitions", "losing_trade_share"),
        ("scenarios", "gross"),
        ("events", "realized_gross"),
        ("events", "realized_net"),
        ("events", "event_half_hour"),
        ("events", "transition"),
    ],
)
def test_offsetting_or_scenario_corruption_detected(table, field):
    tables, summary, bars, source = crossing_tables()
    frame = tables[table]
    if field in ("event_half_hour", "transition"):
        frame.loc[0, field] = "wrong"
    else:
        frame.loc[0, field] += 1
        frame.loc[1, field] -= 1
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize(
    "field",
    [
        "mfe",
        "mae",
        "sampled_mfe",
        "sampled_mae",
        "coverage",
        "duration_min_lower",
        "duration_min_upper",
        "valid_completed_closes",
        "entry_transition",
        "entry_original_reason",
    ],
)
def test_excursion_and_entry_metadata_corruption(field):
    tables, summary, bars, source = crossing_tables()
    tables["trades"].loc[0, field] = (
        "wrong"
        if field in ("coverage", "entry_transition", "entry_original_reason")
        else tables["trades"].loc[0, field] + 1
    )
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize(
    "field",
    [
        "realized_gross",
        "session_realized_gross",
        "session_costs",
        "session_realized_net",
        "session_marked_gross",
        "session_marked_net",
        "unrealized_gross",
        "accrual_half_hour",
    ],
)
def test_minute_derived_corruption(field):
    tables, summary, bars, source = crossing_tables()
    tables["minute"].loc[0, field] = "wrong" if field == "accrual_half_hour" else 1.0
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize(
    "field",
    [
        "flat_sessions",
        "transitions",
        "best_67_days",
        "best_67_share",
        "top_5_percent_trades_net",
        "uncertain_stop_trades",
        "primary",
        "drawdown_minute",
        "drawdown_daily",
    ],
)
def test_summary_corruption(field):
    tables, summary, bars, source = crossing_tables()
    if field.startswith("drawdown_"):
        summary[field]["longest_underwater_samples"] += 1
    elif field == "best_67_days":
        summary[field][0]["day"] = "wrong"
    elif field == "transitions":
        summary[field]["true_reversal"] = 1
    elif field == "primary":
        summary[field]["delay"] = 1
    else:
        summary[field] += 1
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize("table", ["accrual", "partitions", "scenarios"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_complete_keyed_output_grid_required(table, mutation):
    tables, summary, bars, source = crossing_tables()
    frame = tables[table]
    tables[table] = (
        frame.iloc[1:].copy()
        if mutation == "missing"
        else pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    )
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)


@pytest.mark.parametrize("clock", ["drawdown_minute", "drawdown_daily"])
@pytest.mark.parametrize(
    "field",
    [
        "depth",
        "peak_timestamp",
        "trough_timestamp",
        "recovery_timestamp",
        "peak_to_trough_samples",
        "peak_to_recovery_samples",
        "longest_underwater_samples",
        "unrecovered_at_end",
    ],
)
def test_every_drawdown_summary_field_checked(clock, field):
    tables, summary, bars, source = crossing_tables()
    old = summary[clock][field]
    summary[clock][field] = (
        (not old)
        if isinstance(old, bool)
        else old + 1 if isinstance(old, (int, float)) else "wrong"
    )
    with pytest.raises(ValueError):
        a.verify_details(tables, summary, bars, source)
