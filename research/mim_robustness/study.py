"""Historical-only study with a mandatory ledger-level baseline gate."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.mim_comparison.data import audit_select, load
from . import artifacts
from .engine import ARMS, simulate
from .features import features


def prepare(data, baseline):
    artifacts.verify_baseline(data, baseline)
    consumed = Path(data).read_bytes()
    if hashlib.sha256(consumed).hexdigest() != artifacts.DATA_HASH:
        raise ValueError("Consumed historical bytes changed")
    sessions, exclusions = audit_select(load(io.BytesIO(consumed), "end"))
    moves = [
        np.abs(s.bars.close.to_numpy() / float(s.bars.open.iloc[0]) - 1)
        for s in sessions
    ]
    prepared = []
    for i, session in enumerate(sessions):
        if i < 14:
            exclusions.append(
                dict(
                    day=session.day,
                    contract=session.contract,
                    eligible=False,
                    exclusion="14_complete_prior_selected_sessions_required",
                )
            )
        else:
            prepared.append((session, np.mean(moves[i - 14 : i], axis=0)))
    if len(prepared) != artifacts.CONFIG["expected_sessions"]:
        raise ValueError(
            f"Baseline grid changed: {len(prepared)} != {artifacts.CONFIG['expected_sessions']}"
        )
    return prepared, exclusions


def simulate_arm(prepared, arm, feature_writer=None):
    daily, events, decisions = [], [], []
    for session, sigma in prepared:
        values = features(session, sigma)
        if feature_writer:
            for i, row in enumerate(session.bars.itertuples(index=False)):
                feature_writer.writerow(
                    dict(
                        day=session.day,
                        contract=session.contract,
                        event_timestamp=row.timestamp.isoformat(),
                        receipt_timestamp=None,
                        close=row.close,
                        sigma=float(sigma[i]),
                        **{name: value[i] for name, value in values.items()},
                    )
                )
        for delay in artifacts.CONFIG["delays"]:
            result, fills, signals = simulate(
                session,
                sigma,
                arm,
                delay=delay,
                cost=artifacts.CONFIG["costs"][0],
                quantity=artifacts.CONFIG["quantity"],
                feature_values=values,
            )
            for cost in artifacts.CONFIG["costs"]:
                daily.append(
                    dict(
                        result,
                        cost=cost,
                        costs=result["turnover"] * cost / 2,
                        net=result["gross"] - result["turnover"] * cost / 2,
                    )
                )
            events.extend(
                dict(e, delay=delay, cost_scenario=artifacts.CONFIG["costs"][0])
                for e in fills
            )
            decisions.extend(dict(d, delay=delay) for d in signals)
    return pd.DataFrame(daily), pd.DataFrame(events), pd.DataFrame(decisions)


def compare_frame(actual, expected, name):
    """Compare all original fields, never silently align away missing events."""
    if len(actual) != len(expected):
        raise ValueError(
            f"{name} baseline row count mismatch: {len(actual)} != {len(expected)}"
        )
    missing = set(expected.columns) - set(actual.columns)
    if missing:
        raise ValueError(f"{name} missing baseline fields: {sorted(missing)}")
    a, b = actual.reset_index(drop=True), expected.reset_index(drop=True)
    for col in b:
        if pd.api.types.is_numeric_dtype(b[col]) and b[col].notna().any():
            tolerance = 1e-9 if col in ("sigma", "upper", "lower", "vwap") else 1e-8
            if col in (
                "quantity",
                "turnover",
                "position",
                "position_after",
                "target",
                "delay",
                "eligible",
            ):
                tolerance = 0
            x, y = pd.to_numeric(a[col], errors="raise"), pd.to_numeric(
                b[col], errors="raise"
            )
            if not np.allclose(x, y, atol=tolerance, rtol=0, equal_nan=True):
                raise ValueError(f"{name}.{col} baseline numerical mismatch")
        else:
            normalize = (
                lambda s: s.fillna("").astype(str).replace({"None": "", "nan": ""})
            )
            if not normalize(a[col]).equals(normalize(b[col])):
                raise ValueError(f"{name}.{col} baseline discrete mismatch")
    return dict(rows=len(a), original_columns=len(b.columns), agreement=True)


def reconcile(frames, baseline):
    result = {}
    for actual, filename in zip(frames, ("daily.csv", "ledger.csv", "decisions.csv")):
        expected = pd.read_csv(Path(baseline) / filename, float_precision="round_trip")
        expected = expected[expected.arm == "A"]
        result[filename] = compare_frame(actual, expected, filename)
    return result


def write_frame(path, frame):
    with open(path, "x", newline="") as stream:
        frame.to_csv(stream, index=False)


def trade_ledger(ledger, decisions):
    """Round trips reconciled from actual fills, splitting reversal turnover."""
    results = []
    signals = {
        (r.day, r.arm, r.delay, r.event_timestamp): r._asdict()
        for r in decisions.itertuples(index=False)
    }
    for (day, arm, delay), fills in ledger.groupby(["day", "arm", "delay"], sort=False):
        position, entry = 0, None
        for row in fills.to_dict("records"):
            target = int(row["position_after"])
            if position:
                gross = position * (row["fill"] - entry["fill"]) * 2
                results.append(
                    dict(
                        day=day,
                        contract=row["contract"],
                        arm=arm,
                        arm_hash=row["arm_hash"],
                        delay=delay,
                        direction=position,
                        entry_event_timestamp=entry["event_timestamp"],
                        entry_signal_timestamp=entry["signal_timestamp"],
                        entry_fill_timestamp=entry["modeled_fill_timestamp"],
                        exit_event_timestamp=row["event_timestamp"],
                        exit_fill_timestamp=row["modeled_fill_timestamp"],
                        exit_fill_time_basis=row["fill_time_basis"],
                        entry_signal_price=entry["signal_price"],
                        exit_signal_price=row["signal_price"],
                        entry_fill=entry["fill"],
                        exit_fill=row["fill"],
                        quantity=1,
                        gross=gross,
                        costs=artifacts.CONFIG["costs"][0],
                        net=gross - artifacts.CONFIG["costs"][0],
                        exit_reason=row["reason"],
                        entry_slope=entry["features"].get("slope"),
                        entry_r2=entry["features"].get("r2"),
                        entry_efficiency=entry["features"].get("efficiency"),
                        entry_displacement=entry["features"].get("displacement"),
                        eligible=True,
                        exclusion=None,
                    )
                )
            if target:
                signal_time = (
                    pd.Timestamp(row["event_timestamp"]) - pd.Timedelta(minutes=delay)
                ).isoformat()
                feature = signals.get((day, arm, delay, signal_time))
                if feature is None or feature["target"] != target:
                    raise ValueError("Entry fill lacks matching earlier decision")
                entry = dict(row, signal_timestamp=signal_time, features=feature)
            else:
                entry = None
            position = target
        if position:
            raise ValueError("Unclosed daily trade")
    columns = [
        "day",
        "contract",
        "arm",
        "arm_hash",
        "delay",
        "direction",
        "entry_event_timestamp",
        "entry_signal_timestamp",
        "entry_fill_timestamp",
        "exit_event_timestamp",
        "exit_fill_timestamp",
        "exit_fill_time_basis",
        "entry_signal_price",
        "exit_signal_price",
        "entry_fill",
        "exit_fill",
        "quantity",
        "gross",
        "costs",
        "net",
        "exit_reason",
        "entry_slope",
        "entry_r2",
        "entry_efficiency",
        "entry_displacement",
        "eligible",
        "exclusion",
    ]
    return pd.DataFrame(results, columns=columns)


def reconcile_accounting(daily, ledger, trades):
    keys = ["day", "arm", "delay"]
    for frame, columns in (
        (daily, ["gross", "costs", "net", "turnover", "cost", "quantity"]),
        (ledger, ["quantity", "costs", "fill", "signal_price", "position_after"]),
        (
            trades,
            [
                "gross",
                "costs",
                "net",
                "entry_fill",
                "exit_fill",
                "quantity",
                "direction",
            ],
        ),
    ):
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite accounting input")
    primary_cost = artifacts.CONFIG["costs"][0]
    if (
        not np.allclose(trades.costs, primary_cost, rtol=0, atol=1e-8)
        or not np.allclose(trades.net, trades.gross - trades.costs, rtol=0, atol=1e-8)
        or not np.allclose(
            ledger.costs, ledger.quantity * primary_cost / 2, rtol=0, atol=1e-8
        )
    ):
        raise ValueError("Trade/ledger net costs mismatch")
    if (
        not np.allclose(
            trades.gross,
            trades.direction * (trades.exit_fill - trades.entry_fill) * 2,
            rtol=0,
            atol=1e-8,
        )
        or not trades.quantity.eq(1).all()
        or not trades.direction.isin([-1, 1]).all()
    ):
        raise ValueError("Trade fill PnL mismatch")
    daily_keys = set(daily[keys].itertuples(index=False, name=None))
    if any(
        not set(frame[keys].itertuples(index=False, name=None)) <= daily_keys
        for frame in (ledger, trades)
    ):
        raise ValueError("Orphan trade or fill outside daily grid")
    fill_totals = ledger.groupby(keys).quantity.sum().to_dict()
    trade_totals = trades.groupby(keys).gross.agg(["sum", "size"]).to_dict("index")
    for row in daily.itertuples(index=False):
        key = (row.day, row.arm, row.delay)
        t = trade_totals.get(key, {"sum": 0.0, "size": 0})
        if (
            int(fill_totals.get(key, 0)) != row.turnover
            or t["size"] * 2 != row.turnover
        ):
            raise ValueError("Turnover / round-trip reconciliation failed")
        if (
            abs(t["sum"] - row.gross) > 1e-8
            or abs(row.costs - row.turnover * row.cost / 2) > 1e-8
        ):
            raise ValueError("Gross / cost reconciliation failed")
        if abs(row.gross - row.costs - row.net) > 1e-8:
            raise ValueError("Daily PnL reconciliation failed")


def execute(path, data, baseline, audit_only=False):
    artifacts.verify(path)
    prepared, exclusions = prepare(data, baseline)
    base = simulate_arm(prepared, ARMS[0])
    agreement = reconcile(base, baseline)
    artifacts.write_json(path / "baseline-reconciliation.json", agreement)
    write_frame(path / "exclusions.csv", pd.DataFrame(exclusions))
    artifacts.write_json(
        path / "audit.json",
        dict(
            eligible_sessions=len(prepared),
            first=prepared[0][0].day,
            last=prepared[-1][0].day,
            baseline_agreement=True,
            history_exposed=True,
            candidate_returns_computed=not audit_only,
        ),
    )
    if audit_only:
        for frame, name in zip(base, ("daily.csv", "ledger.csv", "decisions.csv")):
            write_frame(path / name, frame)
        with open(path / "report.md", "x") as out:
            out.write(
                "# Baseline audit\n\nAll 1,323 sessions, both timings and three costs agree with the frozen baseline. No candidate returns calculated.\n"
            )
    else:
        frames = [base]
        names = [
            "upper",
            "lower",
            "slope",
            "r2",
            "efficiency",
            "displacement",
            "persistence_long",
            "persistence_short",
        ]
        with open(path / "features.csv", "x", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "day",
                    "contract",
                    "event_timestamp",
                    "receipt_timestamp",
                    "close",
                    "sigma",
                ]
                + names,
            )
            writer.writeheader()
            for index, arm in enumerate(ARMS[1:]):
                print(f"Simulating {arm.name}", flush=True)
                frames.append(
                    simulate_arm(prepared, arm, writer if index == 0 else None)
                )
        daily, ledger, decisions = [
            pd.concat([f[i] for f in frames], ignore_index=True) for i in range(3)
        ]
        trades = trade_ledger(ledger, decisions)
        reconcile_accounting(daily, ledger, trades)
        config_hash = json.loads((path / "manifest.json").read_text())["config_hash"]
        for frame, name in [
            (daily, "daily.csv"),
            (ledger, "ledger.csv"),
            (decisions, "decisions.csv"),
            (trades, "trades.csv"),
        ]:
            frame["configuration_hash"] = config_hash
            write_frame(path / name, frame)
        paired = daily.pivot(
            index=["day", "contract", "delay", "cost"], columns="arm", values="net"
        ).reset_index()
        for arm in ("R", "E", "F", "P"):
            paired[arm + "-A"] = paired[arm] - paired.A
        write_frame(path / "paired-daily.csv", paired)
        from .report import generate

        generate(path, daily, trades)
    artifacts.verify_baseline(data, baseline)
    artifacts.verify(path)
