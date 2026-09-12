"""Observed paths only: no alternative rules, strategy returns, or broker imports."""

import math

import numpy as np
import pandas as pd

CHECKPOINTS = (5, 15, 30, 60, 120)
FEE = 2.24
POINT_VALUE = 2.0
ET = "America/New_York"
MINUTE = pd.Timedelta(minutes=1)
EXPECTED_TRADES = 801
EXPECTED_SESSIONS = 1323
EXPECTED_NET = 21889.76
TOP_TRADE_FRACTION = 0.05
TOP_DAILY_COUNT = 67


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None or result != result.floor("min"):
        raise ValueError("Missing, naive, or nonminute timestamp")
    return result.tz_convert(ET)


def validate_session(bars, day, contract):
    bars = bars.sort_values("timestamp").reset_index(drop=True).copy()
    if len(bars) != 390 or bars.timestamp.duplicated().any():
        raise ValueError("Selected session missing or duplicate minutes")
    expected = pd.date_range(f"{day} 09:31", periods=390, freq="min", tz=ET)
    if not pd.DatetimeIndex(bars.timestamp).equals(expected):
        raise ValueError(
            "Selected session does not contain exact 390 end-labeled minutes"
        )
    if not bars.day.eq(day).all() or not bars.contract.eq(contract).all():
        raise ValueError("Day/contract mismatch")
    values = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values[:, :4] <= 0).any()
        or (values[:, 4] < 0).any()
    ):
        raise ValueError("Nonfinite or invalid OHLCV")
    if (
        (bars.high < bars[["open", "low", "close"]].max(axis=1))
        | (bars.low > bars[["open", "high", "close"]].min(axis=1))
    ).any():
        raise ValueError("Invalid OHLC ordering")
    return bars


def select_inputs(
    trades,
    daily,
    expected_trades=EXPECTED_TRADES,
    expected_sessions=EXPECTED_SESSIONS,
    expected_net=EXPECTED_NET,
):
    required = {
        "day",
        "contract",
        "arm",
        "delay",
        "direction",
        "entry_event_timestamp",
        "entry_fill_timestamp",
        "exit_event_timestamp",
        "exit_fill_timestamp",
        "exit_fill_time_basis",
        "entry_fill",
        "exit_fill",
        "quantity",
        "gross",
        "costs",
        "net",
        "exit_reason",
    }
    if not required <= set(trades) or not {
        "day",
        "contract",
        "arm",
        "delay",
        "cost",
        "quantity",
        "gross",
        "costs",
        "net",
    } <= set(daily):
        raise ValueError("Missing source columns")
    trades = trades.loc[(trades.arm == "A") & (trades.delay == 2)].copy()
    daily = daily.loc[
        (daily.arm == "A") & (daily.delay == 2) & (daily.cost == FEE)
    ].copy()
    if len(trades) != expected_trades or len(daily) != expected_sessions:
        raise ValueError("Unexpected source trade/session counts")
    if (
        daily.day.duplicated().any()
        or trades.duplicated(["day", "contract", "entry_fill_timestamp"]).any()
    ):
        raise ValueError("Duplicate daily grid or source trade")
    if (
        trades[list(required - {"exit_fill_timestamp"})].isna().any().any()
        or daily[["day", "contract", "quantity", "gross", "costs", "net"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Missing selected source values")
    if (
        not np.isfinite(
            trades[
                [
                    "direction",
                    "entry_fill",
                    "exit_fill",
                    "quantity",
                    "gross",
                    "costs",
                    "net",
                ]
            ].to_numpy(float)
        ).all()
        or not np.isfinite(
            daily[["quantity", "gross", "costs", "net"]].to_numpy(float)
        ).all()
    ):
        raise ValueError("Nonfinite selected source accounting")
    if (
        not trades.direction.isin([-1, 1]).all()
        or not trades.quantity.eq(1).all()
        or not daily.quantity.eq(1).all()
        or not np.allclose(trades.costs, FEE, rtol=0, atol=1e-10)
    ):
        raise ValueError("Unexpected direction, quantity, or friction")
    calculated = (trades.exit_fill - trades.entry_fill) * trades.direction * POINT_VALUE
    if not np.allclose(calculated, trades.gross, rtol=0, atol=1e-8) or not np.allclose(
        trades.gross - trades.costs, trades.net, rtol=0, atol=1e-8
    ):
        raise ValueError("Source trade accounting mismatch")
    if not set(zip(trades.day, trades.contract)) <= set(zip(daily.day, daily.contract)):
        raise ValueError("Trade day/contract absent from daily grid")
    for field in ("gross", "costs", "net"):
        aligned = (
            trades.groupby("day")[field]
            .sum()
            .reindex(daily.day, fill_value=0)
            .to_numpy()
        )
        if not np.allclose(aligned, daily[field], rtol=0, atol=1e-8):
            raise ValueError(f"Daily {field} reconciliation failed")
    if (
        abs(float(trades.net.sum()) - expected_net) > 1e-8
        or abs(float(daily.net.sum()) - expected_net) > 1e-8
    ):
        raise ValueError("Pinned net total mismatch")
    trades = trades.sort_values(
        ["day", "contract", "entry_fill_timestamp"]
    ).reset_index(drop=True)
    trades.insert(
        0, "trade_id", [f"A-delay2-{i:04d}" for i in range(1, len(trades) + 1)]
    )
    return trades, daily.sort_values("day").reset_index(drop=True)


def trace_trade(trade, bars, checkpoints=CHECKPOINTS):
    """Return point/bar observations, one lifecycle row, and checkpoint rows.

    Stop fills have interval timestamps; no stop-bar OHLC enters any statistic.
    The entry fill reserves fees but is not a completed-close loss observation.
    """
    t = dict(trade)
    entry = timestamp(t["entry_fill_timestamp"])
    entry_event = timestamp(t["entry_event_timestamp"])
    event = timestamp(t["exit_event_timestamp"])
    if (
        entry_event != entry + MINUTE
        or entry.strftime("%Y-%m-%d") != t["day"]
        or event.strftime("%Y-%m-%d") != t["day"]
    ):
        raise ValueError("Entry/event timestamp or day mismatch")
    basis = t["exit_fill_time_basis"]
    if (
        basis == "intrabar_unknown_within_event_minute"
        and t["exit_reason"] == "CAT_STOP"
    ):
        if pd.notna(t["exit_fill_timestamp"]) and str(t["exit_fill_timestamp"]).strip():
            raise ValueError("Stop unexpectedly has exact fill timestamp")
        lower, upper, uncertain = event - MINUTE, event, True
        valid = bars[(bars.timestamp > entry) & (bars.timestamp < event)]
    elif basis == "bar_open" and t["exit_reason"] == "REVERSAL":
        lower = upper = timestamp(t["exit_fill_timestamp"])
        uncertain = False
        if lower != event - MINUTE:
            raise ValueError("Opening exit/event mismatch")
        valid = bars[(bars.timestamp > entry) & (bars.timestamp <= lower)]
    elif basis == "session_close_proxy" and t["exit_reason"] == "EOD_CLOSE_PROXY":
        lower = upper = timestamp(t["exit_fill_timestamp"])
        uncertain = False
        if lower != event or event.hour != 16 or event.minute != 0:
            raise ValueError("EOD exit/event mismatch")
        valid = bars[(bars.timestamp > entry) & (bars.timestamp <= event)]
    else:
        raise ValueError("Unknown exit reason/time basis")
    if (
        lower < entry
        or upper > bars.timestamp.iloc[-1]
        or entry < bars.timestamp.iloc[0] - MINUTE
    ):
        raise ValueError("Trade outside session or negative duration")
    entry_bar = bars[bars.timestamp == entry_event]
    if len(entry_bar) != 1 or not math.isclose(
        float(entry_bar.open.iloc[0]), float(t["entry_fill"]), abs_tol=1e-8, rel_tol=0
    ):
        raise ValueError("Entry fill does not match entry-bar open")
    exit_bar = bars[bars.timestamp == event]
    if len(exit_bar) != 1:
        raise ValueError("Exit event missing")
    if not uncertain:
        field = "open" if basis == "bar_open" else "close"
        if not math.isclose(
            float(exit_bar[field].iloc[0]),
            float(t["exit_fill"]),
            abs_tol=1e-8,
            rel_tol=0,
        ):
            raise ValueError("Terminal fill differs from source bar")
    direction, fill = int(t["direction"]), float(t["entry_fill"])
    if (
        direction not in (-1, 1)
        or t["quantity"] != 1
        or not math.isclose(float(t["costs"]), FEE, abs_tol=1e-10)
    ):
        raise ValueError("Invalid trade contract")

    def gross(price):
        return direction * (float(price) - fill) * POINT_VALUE

    if not math.isclose(
        gross(t["exit_fill"]) - FEE, float(t["net"]), rel_tol=0, abs_tol=1e-8
    ):
        raise ValueError("Terminal net/source mismatch")
    mfe = mae = 0.0
    observations = []

    def point(kind, lowtime, hightime, price, high, low):
        nonlocal mfe, mae
        favorable = max(gross(high), gross(low), gross(price))
        adverse = min(gross(high), gross(low), gross(price))
        mfe, mae = max(mfe, favorable), max(mae, -adverse)
        row = dict(
            trade_id=t["trade_id"],
            day=t["day"],
            contract=t["contract"],
            kind=kind,
            timestamp=lowtime.isoformat() if lowtime == hightime else None,
            timestamp_lower=lowtime.isoformat(),
            timestamp_upper=hightime.isoformat(),
            elapsed_min_lower=float((lowtime - entry) / MINUTE),
            elapsed_min_upper=float((hightime - entry) / MINUTE),
            price=float(price),
            high=float(high),
            low=float(low),
            gross_mark=gross(price),
            net_mark=gross(price) - FEE,
            mfe=mfe,
            mae=mae,
            time_uncertain=lowtime != hightime,
        )
        observations.append(row)

    point("entry_fill", entry, entry, fill, fill, fill)
    for bar in valid.itertuples():
        point(
            "completed_close",
            bar.timestamp,
            bar.timestamp,
            bar.close,
            bar.high,
            bar.low,
        )
    point("terminal_fill", lower, upper, t["exit_fill"], t["exit_fill"], t["exit_fill"])
    marks = observations[1:]
    positive = next((r for r in marks if r["net_mark"] > 0), None)
    positive_close = next(
        (r for r in marks if r["kind"] == "completed_close" and r["net_mark"] > 0), None
    )
    negative_index = next(
        (
            i
            for i, r in enumerate(marks)
            if r["kind"] == "completed_close" and r["net_mark"] < 0
        ),
        None,
    )
    negative = marks[negative_index] if negative_index is not None else None
    recovery = (
        next((r for r in marks[negative_index + 1 :] if r["net_mark"] >= 0), None)
        if negative is not None
        else None
    )
    life = {
        key: t[key]
        for key in (
            "trade_id",
            "day",
            "contract",
            "direction",
            "entry_fill",
            "exit_fill",
            "gross",
            "costs",
            "net",
            "exit_reason",
        )
    }
    life.update(
        year=int(t["day"][:4]),
        final_winner=float(t["net"]) > 0,
        mfe=mfe,
        mae=mae,
        excursions_observed_lower_bounds=uncertain,
        duration_min_lower=float((lower - entry) / MINUTE),
        duration_min_upper=float((upper - entry) / MINUTE),
        stop_time_uncertain=uncertain,
        valid_completed_closes=len(valid),
        never_positive=positive is None,
        had_negative=negative is not None,
        recovered_after_first_negative=recovery is not None,
    )
    life["never_positive_close"] = positive_close is None
    life["early_negative_close_30min"] = any(
        r["kind"] == "completed_close"
        and r["elapsed_min_upper"] <= 30
        and r["net_mark"] < 0
        for r in marks
    )
    for name, row in (
        ("first_positive_close", positive_close),
        ("first_positive", positive),
        ("first_negative", negative),
        ("first_recovery", recovery),
    ):
        life[name + "_min_lower"] = row["elapsed_min_lower"] if row else None
        life[name + "_min_upper"] = row["elapsed_min_upper"] if row else None
        life[name + "_basis"] = row["kind"] if row else None
    landmarks = []
    for checkpoint in checkpoints:
        close = next(
            (
                r
                for r in observations
                if r["kind"] == "completed_close"
                and r["elapsed_min_lower"] == checkpoint
            ),
            None,
        )
        row = dict(
            trade_id=t["trade_id"],
            checkpoint_min=checkpoint,
            at_risk=close is not None,
            already_exited=close is None,
            final_winner=life["final_winner"],
            final_net=float(t["net"]),
            stop_interval_unobserved=uncertain
            and float((lower - entry) / MINUTE)
            < checkpoint
            <= float((upper - entry) / MINUTE),
        )
        if close:
            # Observation order preserves a terminal execution following the preceding
            # close even when both carry the same minute timestamp.
            later = observations[observations.index(close) + 1 :]
            recovered = (
                any(r["net_mark"] >= 0 for r in later)
                if close["net_mark"] < 0
                else None
            )
            row.update(
                positive_net_close_seen=any(
                    r["kind"] == "completed_close"
                    and r["elapsed_min_upper"] <= checkpoint
                    and r["net_mark"] > 0
                    for r in marks
                ),
                current_net=close["net_mark"],
                mfe=close["mfe"],
                mae=close["mae"],
                cohort="red" if close["net_mark"] < 0 else "green",
                remaining_net_change=float(t["net"]) - close["net_mark"],
                recovered_after_red=recovered,
            )
        else:
            row.update(
                positive_net_close_seen=None,
                current_net=None,
                mfe=None,
                mae=None,
                cohort="exited",
                remaining_net_change=None,
                recovered_after_red=None,
            )
        landmarks.append(row)
    return observations, life, landmarks


def analyze(trades, daily, bars):
    sessions = {}
    selected_keys = set(zip(trades.day, trades.contract))
    selected = bars.loc[
        pd.MultiIndex.from_frame(bars[["day", "contract"]]).isin(selected_keys)
    ]
    for (day, contract), group in selected.groupby(["day", "contract"], sort=True):
        sessions[day, contract] = validate_session(group, day, contract)
    if set(sessions) != selected_keys:
        raise ValueError("Missing exact selected day/contract session")
    paths, lifecycle, landmarks = [], [], []
    for t in trades.to_dict("records"):
        p, l, m = trace_trade(t, sessions[t["day"], t["contract"]])
        paths.extend(p)
        lifecycle.append(l)
        landmarks.extend(m)
    life = pd.DataFrame(lifecycle)
    top_ids = set(
        life.sort_values(["net", "trade_id"], ascending=[False, True])
        .head(math.ceil(TOP_TRADE_FRACTION * len(life)))
        .trade_id
    )
    top_days = set(
        daily.sort_values(["net", "day"], ascending=[False, True])
        .head(TOP_DAILY_COUNT)
        .day
    )
    life["hindsight_top_5pct_trade"] = life.trade_id.isin(top_ids)
    life["hindsight_top_67_day"] = life.day.isin(top_days)
    return pd.DataFrame(paths), life, pd.DataFrame(landmarks)


def describe(frame, total):
    numeric = (
        "net",
        "mfe",
        "mae",
        "duration_min_lower",
        "duration_min_upper",
        "first_positive_min_lower",
        "first_positive_min_upper",
        "first_positive_close_min_lower",
        "first_negative_min_lower",
        "first_recovery_min_lower",
    )

    def quantiles(series):
        series = series.dropna()
        return {
            str(q): float(series.quantile(q)) if len(series) else None
            for q in (0, 0.25, 0.5, 0.75, 0.9, 1)
        }

    return dict(
        count=len(frame),
        net=float(frame.net.sum()),
        net_contribution=float(frame.net.sum() / total) if total else None,
        winners=int(frame.final_winner.sum()),
        never_positive=int(frame.never_positive.sum()),
        never_positive_close=int(frame.never_positive_close.sum()),
        early_negative_close_30min=int(frame.early_negative_close_30min.sum()),
        had_negative=int(frame.had_negative.sum()),
        recovered_after_first_negative=int(frame.recovered_after_first_negative.sum()),
        stop_uncertain=int(frame.stop_time_uncertain.sum()),
        quantiles={column: quantiles(frame[column]) for column in numeric},
    )


def summarize(life, landmarks, daily):
    total = float(life.net.sum())
    result = dict(
        descriptive_only=True,
        trade_count=len(life),
        sessions=len(daily),
        flat_sessions=int((daily.costs == 0).sum()),
        total_net=total,
        daily_net=float(daily.net.sum()),
        overall=describe(life, total),
        groups={},
        landmarks=[],
    )
    for field in ("final_winner", "direction", "year", "exit_reason"):
        result["groups"][field] = {
            str(key): describe(group, total)
            for key, group in life.groupby(field, sort=True)
        }
    for field in ("hindsight_top_5pct_trade", "hindsight_top_67_day"):
        chosen = life[life[field]]
        result["groups"][field] = {
            "selected": describe(chosen, total),
            "remainder": describe(life[~life[field]], total),
        }
    result["hindsight_top_67_days"] = (
        daily.sort_values(["net", "day"], ascending=[False, True])
        .head(TOP_DAILY_COUNT)[["day", "net"]]
        .to_dict("records")
    )
    for checkpoint, frame in landmarks.groupby("checkpoint_min", sort=True):
        row = dict(
            checkpoint_min=int(checkpoint),
            at_risk=int(frame.at_risk.sum()),
            already_exited=int(frame.already_exited.sum()),
            stop_interval_unobserved=int(frame.stop_interval_unobserved.sum()),
            cohorts={},
            positive_history_cohorts={},
        )
        for label in ("red", "green"):
            cohort = frame[frame.cohort == label]
            row["cohorts"][label] = dict(
                count=len(cohort),
                eventual_winner_rate=(
                    float(cohort.final_winner.mean()) if len(cohort) else None
                ),
                mean_final_net=float(cohort.final_net.mean()) if len(cohort) else None,
                mean_remaining_net_change=(
                    float(cohort.remaining_net_change.mean()) if len(cohort) else None
                ),
                recovered_after_red=(
                    int(cohort.recovered_after_red.eq(True).sum())
                    if label == "red"
                    else None
                ),
            )
        for seen, label in (
            (False, "never_positive_close_yet"),
            (True, "positive_close_seen"),
        ):
            cohort = frame[frame.at_risk & frame.positive_net_close_seen.eq(seen)]
            row["positive_history_cohorts"][label] = dict(
                count=len(cohort),
                eventual_winner_rate=(
                    float(cohort.final_winner.mean()) if len(cohort) else None
                ),
                mean_final_net=float(cohort.final_net.mean()) if len(cohort) else None,
                mean_remaining_net_change=(
                    float(cohort.remaining_net_change.mean()) if len(cohort) else None
                ),
            )
        result["landmarks"].append(row)
    return result
