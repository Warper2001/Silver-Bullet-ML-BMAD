"""Causal scheduled-mark ledger for the unchanged MIM baseline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .artifacts import input_path

ET = "America/New_York"
POINT_VALUE = 2.0
ROUND_TRIP_COST = 2.24
ENTRY_COST = 1.12


def _selected_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/"
    trades = pd.read_csv(input_path(base + "trades.csv"))
    daily = pd.read_csv(input_path(base + "daily.csv"))
    decisions = pd.read_csv(input_path(base + "decisions.csv"))
    trades = trades[(trades.arm == "A") & (trades.delay == 2)].copy()
    daily = daily[
        (daily.arm == "A")
        & (daily.delay == 2)
        & np.isclose(daily.cost, ROUND_TRIP_COST)
    ].copy()
    decisions = decisions[(decisions.arm == "A") & (decisions.delay == 2)].copy()
    if len(trades) != 801 or len(daily) != 1323 or len(decisions) != 1323 * 12:
        raise ValueError("Pinned arm-A/delay-2 source counts changed")
    if not np.isclose(trades.net.sum(), 21889.76, atol=1e-8, rtol=0):
        raise ValueError("Pinned source net changed")
    if trades.exit_reason.value_counts().to_dict() != {
        "EOD_CLOSE_PROXY": 723,
        "CAT_STOP": 71,
        "REVERSAL": 7,
    }:
        raise ValueError("Pinned source exit labels changed")
    if daily.day.duplicated().any():
        raise ValueError("Duplicate selected daily row")
    for field in ("gross", "costs", "net"):
        expected = (
            trades.groupby("day")[field]
            .sum()
            .reindex(daily.day, fill_value=0)
            .to_numpy(float)
        )
        if not np.allclose(expected, daily[field].to_numpy(float), atol=1e-8, rtol=0):
            raise ValueError(f"Selected daily {field} does not reconcile to trades")
    trades = trades.sort_values(
        ["day", "contract", "entry_event_timestamp"]
    ).reset_index(drop=True)
    trades.insert(0, "trade_id", [f"A-delay2-{i:04d}" for i in range(1, 802)])
    return trades, daily.sort_values("day").reset_index(drop=True), decisions


def _load_selected_bars(keys: set[tuple[str, str]]) -> pd.DataFrame:
    bars = pd.read_csv(
        input_path("data/mim_x/mnq_1min_by_contract.csv"),
        usecols=["contract", "timestamp", "open", "high", "low", "close", "volume"],
    )
    raw_timestamp = bars.timestamp.astype(str)
    if not raw_timestamp.str.contains(r"(?:Z|[+-]\d\d:\d\d)$").all():
        raise ValueError("Bar timestamps must include timezone")
    bars["timestamp"] = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert(ET)
    bars["day"] = bars.timestamp.dt.strftime("%Y-%m-%d")
    selected = bars.loc[
        pd.MultiIndex.from_frame(bars[["day", "contract"]]).isin(keys)
    ].copy()
    values = selected[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values[:, :4] <= 0).any()
        or (values[:, 4] < 0).any()
    ):
        raise ValueError("Invalid selected OHLCV")
    if (
        (selected.high < selected[["open", "low", "close"]].max(axis=1))
        | (selected.low > selected[["open", "high", "close"]].min(axis=1))
    ).any():
        raise ValueError("Invalid selected OHLC ordering")
    return selected.sort_values(["day", "contract", "timestamp"]).reset_index(drop=True)


def _trade_intervals(trades: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    for column in (
        "entry_fill_timestamp",
        "entry_event_timestamp",
        "exit_event_timestamp",
        "exit_fill_timestamp",
    ):
        result[column + "_parsed"] = pd.to_datetime(
            result[column], utc=True, errors="coerce"
        )
    if (
        result[
            [
                "entry_fill_timestamp_parsed",
                "entry_event_timestamp_parsed",
                "exit_event_timestamp_parsed",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Missing source trade clock")
    return result


def _summary(trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, object]:
    winning = float(trades.loc[trades.net > 0, "net"].sum())
    losing = float(-trades.loc[trades.net < 0, "net"].sum())
    entries = pd.to_datetime(trades.entry_fill_timestamp, utc=True)
    events = pd.to_datetime(trades.exit_event_timestamp, utc=True)
    fills = pd.to_datetime(trades.exit_fill_timestamp, utc=True, errors="coerce")
    upper = np.where(
        trades.exit_reason.eq("CAT_STOP"),
        (events - entries).dt.total_seconds() / 60,
        (fills - entries).dt.total_seconds() / 60,
    )
    lower = np.where(trades.exit_reason.eq("CAT_STOP"), upper - 1, upper)
    equity = daily.net.cumsum().to_numpy(float)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :]
    drawdown = equity - peaks
    top_count = math.ceil(0.05 * len(trades))
    top_trade_net = float(trades.nlargest(top_count, "net").net.sum())
    top_day_net = float(daily.nlargest(67, "net").net.sum())
    return {
        "closed_trade_net_pf": winning / losing,
        "net_profit": float(trades.net.sum()),
        "trades": len(trades),
        "sessions": len(daily),
        "winners": int((trades.net > 0).sum()),
        "losers": int((trades.net < 0).sum()),
        "winning_dollars": winning,
        "losing_dollars": losing,
        "exposure_contract_minutes_lower": float(np.sum(lower)),
        "exposure_contract_minutes_upper": float(np.sum(upper)),
        "daily_max_drawdown": float(-drawdown.min()),
        "top_5pct_trade_count": top_count,
        "top_5pct_trade_net": top_trade_net,
        "top_5pct_trade_fraction_of_net": top_trade_net / float(trades.net.sum()),
        "top_67_day_net": top_day_net,
        "top_67_day_fraction_of_net": top_day_net / float(trades.net.sum()),
        "exit_labels": {
            str(k): int(v)
            for k, v in trades.exit_reason.value_counts().sort_index().items()
        },
        "accounting_tolerance": 1e-8,
    }


def _lineage() -> pd.DataFrame:
    sources = {
        "MIM V1": (
            "_bmad-output/preregistration_mim_noise_bands.md",
            "V1 (primary, tight stop)",
        ),
        "MIM V2": (
            "_bmad-output/preregistration_mim_noise_bands.md",
            "V2 (wide stop)",
        ),
        "published neutral exit": (
            "research/mim_comparison/RESULTS.md",
            "neutral-sample exit accounts",
        ),
        "MIM lifecycle": (
            "research/mim_lifecycle/RESULTS.md",
            "catastrophe-stop minute has unknown price ordering",
        ),
    }
    for relative, marker in sources.values():
        if marker not in input_path(relative).read_text():
            raise ValueError(f"Pinned prior-work marker missing: {relative}")
    return pd.DataFrame(
        [
            {
                "reference": "MIM V1",
                "mechanism": "tight VWAP/band stop",
                "overlap": "uses adverse path state",
                "distinctness": "not the scheduled-mark giveback description",
                "candidate_returns_calculated": False,
                "source_path": sources["MIM V1"][0],
                "source_marker": sources["MIM V1"][1],
            },
            {
                "reference": "MIM V2",
                "mechanism": "opposite-band stop",
                "overlap": "uses adverse path state",
                "distinctness": "not the scheduled-mark giveback description",
                "candidate_returns_calculated": False,
                "source_path": sources["MIM V2"][0],
                "source_marker": sources["MIM V2"][1],
            },
            {
                "reference": "published neutral exit",
                "mechanism": "flat on neutral sample",
                "overlap": "acts at sampled decisions",
                "distinctness": "already exposed; excluded as a new hypothesis",
                "candidate_returns_calculated": False,
                "source_path": sources["published neutral exit"][0],
                "source_marker": sources["published neutral exit"][1],
            },
            {
                "reference": "MIM lifecycle",
                "mechanism": "held-interval excursions and stop censoring",
                "overlap": "same causal measurement boundary",
                "distinctness": "measurement method only; no exit rule",
                "candidate_returns_calculated": False,
                "source_path": sources["MIM lifecycle"][0],
                "source_marker": sources["MIM lifecycle"][1],
            },
        ]
    )


def _distributions(marks: pd.DataFrame) -> pd.DataFrame:
    """Fixed, exhaustive descriptive summaries; no bin is a candidate threshold."""
    metrics = (
        "position_age_minutes",
        "current_net",
        "mfe",
        "mae",
        "giveback",
        "prior_mark_net_change",
    )
    groups = [("all", "ALL", marks)]
    for dimension in (
        "final_outcome",
        "direction",
        "final_exit_reason",
        "top_5pct_trade",
    ):
        for value, frame in marks.groupby(dimension, dropna=False, sort=True):
            groups.append((dimension, str(value), frame))
    rows = []
    quantiles = [i / 10 for i in range(11)]
    for dimension, value, frame in groups:
        for metric in metrics:
            series = frame[metric].dropna().astype(float)
            row = {
                "group_dimension": dimension,
                "group_value": value,
                "metric": metric,
                "total_marks": len(frame),
                "observed_values": len(series),
                "mean": float(series.mean()) if len(series) else None,
                "sample_std": float(series.std(ddof=1)) if len(series) > 1 else None,
                "candidate_threshold": None,
                "candidate_return_calculated": False,
            }
            for q, observed in zip(quantiles, series.quantile(quantiles)):
                row[f"q{int(q * 100):03d}"] = float(observed)
            rows.append(row)
    return pd.DataFrame(rows)


def _verdict(marks: pd.DataFrame, lineage: pd.DataFrame) -> dict[str, object]:
    complete = len(marks) == 6673 and not marks.mark_id.duplicated().any()
    prior_rule_duplicate = bool(
        lineage.distinctness.str.contains(
            "duplicates scheduled-mark giveback", case=False
        ).any()
    )
    if not complete:
        verdict = "INSUFFICIENT_PATH_COVERAGE"
    elif prior_rule_duplicate:
        verdict = "DUPLICATES_PRIOR_WORK"
    else:
        verdict = "DISTINCT_HYPOTHESIS_REMAINS"
    return {
        "verdict": verdict,
        "scheduled_marks": len(marks),
        "coverage_complete": complete,
        "prior_rule_duplicate": prior_rule_duplicate,
        "descriptive_only": True,
        "predictive_value_established": False,
        "candidate_returns_calculated": False,
        "threshold_selected": False,
        "reasons": [
            "Profit giveback at the existing scheduled decisions is distinct from V1/V2 band stops and the exposed neutral-sample exit.",
            "The lifecycle study supplied the measurement boundary and explicitly left this giveback mechanism untested.",
            "Winner/loss labels and top-trade membership are retrospective descriptions and cannot validate a decision rule.",
        ],
        "required_next_gates": [
            "carry-specific work remains separate",
            "derive any rule from a declared sweep rather than hand-setting a threshold",
            "run an appropriate power gate before a new strategy test",
            "commit a preregistration before independent unseen-data evaluation",
        ],
    }


HYPOTHESIS_SPECIFICATION = """# Future MIM profit-giveback hypothesis specification

**Status: specification only; no strategy test or deployment is authorized.**

At an existing completed 10:00--15:30 ET decision mark while a baseline position remains open, a causal measure of retreat from the trade's prior favorable path may contain information about remaining payoff and adverse risk beyond V1/V2 band state and the already exposed neutral-sample exit.

The future predictor family may use only favorable excursion accumulated through the mark, current marked P&L, retreat from that running maximum, position age, and recent completed-mark change. Original outcome, exit reason, top-winner membership and future bars are labels only. The falsifier is failure to improve net PF after costs while preserving net profit and the large winners that carry the baseline, or inability to obtain adequate independent power.

Before any return calculation: define closed-trade accounting and re-entry behavior; derive any cutoff from a declared development sweep; run a suitable power gate; commit a preregistration with winner-retention, concentration and drawdown checks; and evaluate once on data unseen by the choice. Ambiguous evidence is FAIL.
"""


def held_mark_features(
    session: pd.DataFrame,
    entry_fill_time: pd.Timestamp,
    mark_time: pd.Timestamp,
    direction: int,
    entry_fill: float,
    signal_price: float,
) -> dict[str, float | int]:
    """Measure exactly the unique completed-minute interval after entry through mark."""
    entry_fill_time = pd.Timestamp(entry_fill_time)
    mark_time = pd.Timestamp(mark_time)
    if (
        entry_fill_time.tzinfo is None
        or mark_time.tzinfo is None
        or mark_time <= entry_fill_time
    ):
        raise ValueError("Held interval requires ordered timezone-aware clocks")
    known = session[
        (session.timestamp > entry_fill_time) & (session.timestamp <= mark_time)
    ].copy()
    expected = pd.date_range(
        entry_fill_time + pd.Timedelta(minutes=1), mark_time, freq="min"
    )
    actual = pd.DatetimeIndex(known.timestamp)
    if known.timestamp.duplicated().any() or not actual.equals(expected):
        raise ValueError("Held interval is not a unique continuous one-minute grid")
    if not np.isclose(float(known.close.iloc[-1]), signal_price, atol=1e-8, rtol=0):
        raise ValueError("Decision mark differs from bound completed close")
    favorable = np.maximum(
        direction * (known.high.to_numpy(float) - entry_fill) * POINT_VALUE,
        direction * (known.low.to_numpy(float) - entry_fill) * POINT_VALUE,
    )
    adverse = np.minimum(
        direction * (known.high.to_numpy(float) - entry_fill) * POINT_VALUE,
        direction * (known.low.to_numpy(float) - entry_fill) * POINT_VALUE,
    )
    current_gross = direction * (signal_price - entry_fill) * POINT_VALUE
    mfe = max(0.0, float(favorable.max()))
    mae = max(0.0, float(-adverse.min()))
    return {
        "current_gross": current_gross,
        "current_net": current_gross - ENTRY_COST,
        "mfe": mfe,
        "mae": mae,
        "giveback": mfe - current_gross,
        "completed_bar_coverage": len(known),
    }


def build_marks() -> dict[str, object]:
    trades, daily, decisions = _selected_sources()
    intervals = _trade_intervals(trades)
    active = decisions[decisions.position != 0].copy()
    if len(active) != 6673:
        raise ValueError("Expected exactly 6,673 active scheduled decisions")
    active["mark_parsed"] = pd.to_datetime(active.event_timestamp, utc=True)
    keys = set(zip(trades.day, trades.contract))
    bars = _load_selected_bars(keys)
    sessions = {
        (day, contract): group.reset_index(drop=True)
        for (day, contract), group in bars.groupby(["day", "contract"], sort=False)
    }

    mark_rows: list[dict[str, object]] = []
    for decision in active.sort_values(["day", "event_timestamp"]).itertuples(
        index=False
    ):
        candidates = intervals[
            (intervals.day == decision.day)
            & (intervals.contract == decision.contract)
            & (intervals.entry_event_timestamp_parsed <= decision.mark_parsed)
            & (intervals.exit_event_timestamp_parsed > decision.mark_parsed)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Active decision does not map to exactly one exiting-leg trade"
            )
        trade = candidates.iloc[0]
        if int(decision.position) != int(trade.direction):
            raise ValueError("Decision position and mapped trade side disagree")
        session = sessions[decision.day, decision.contract]
        entry_fill_time = trade.entry_fill_timestamp_parsed.tz_convert(ET)
        mark_time = decision.mark_parsed.tz_convert(ET)
        direction = int(trade.direction)
        entry = float(trade.entry_fill)
        features = held_mark_features(
            session,
            entry_fill_time,
            mark_time,
            direction,
            entry,
            float(decision.signal_price),
        )
        mark_rows.append(
            {
                "mark_id": f"{trade.trade_id}-{pd.Timestamp(decision.mark_parsed).strftime('%H%M')}",
                "trade_id": trade.trade_id,
                "day": decision.day,
                "contract": decision.contract,
                "mark_timestamp": pd.Timestamp(decision.event_timestamp).isoformat(),
                "entry_event_timestamp": trade.entry_event_timestamp,
                "entry_fill_timestamp": trade.entry_fill_timestamp,
                "entry_fill": entry,
                "position_age_minutes": (mark_time - entry_fill_time).total_seconds()
                / 60,
                "direction": direction,
                "signal_price": float(decision.signal_price),
                "current_gross": features["current_gross"],
                "current_net": features["current_net"],
                "mfe": features["mfe"],
                "mae": features["mae"],
                "giveback": features["giveback"],
                "prior_mark_net_change": None,
                "completed_bar_coverage": features["completed_bar_coverage"],
                "decision_reason": decision.reason,
                "decision_target": int(decision.target),
                "decision_vwap": float(decision.vwap),
                "decision_upper": float(decision.upper),
                "decision_lower": float(decision.lower),
                "decision_sigma": float(decision.sigma),
                "reversal_mark_exiting_leg": bool(decision.reason == "REVERSAL"),
                "final_outcome": "WIN" if float(trade.net) > 0 else "LOSS",
                "final_net": float(trade.net),
                "final_exit_reason": trade.exit_reason,
                "candidate_return": None,
            }
        )
    marks = (
        pd.DataFrame(mark_rows)
        .sort_values(["trade_id", "mark_timestamp"])
        .reset_index(drop=True)
    )
    marks["prior_mark_net_change"] = marks.groupby(
        "trade_id", sort=False
    ).current_net.diff()
    top_count = math.ceil(0.05 * len(trades))
    top_ids = set(trades.nlargest(top_count, "net").trade_id)
    marks["top_5pct_trade"] = marks.trade_id.isin(top_ids)
    if (
        marks.mark_id.duplicated().any()
        or int(marks.reversal_mark_exiting_leg.sum()) != 7
    ):
        raise ValueError("Scheduled mark identity or reversal ownership failure")
    exported = trades[
        [
            "trade_id",
            "day",
            "contract",
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
        ]
    ].copy()
    lineage = _lineage()
    return {
        "trades": exported,
        "marks": marks,
        "lineage": lineage,
        "distributions": _distributions(marks),
        "verdict": _verdict(marks, lineage),
        "hypothesis": HYPOTHESIS_SPECIFICATION,
        "summary": _summary(trades, daily),
    }
