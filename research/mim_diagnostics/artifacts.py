"""Exclusive, hash-bound diagnostics artifacts; source runs remain read-only."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import shutil
import sys
import uuid
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ORIGINAL = Path("/root/Silver-Bullet-ML-BMAD")
RUNS = BASE / "runs"
SOURCE = ORIGINAL / "research/mim_robustness/runs/20260912T151842-run-f8608e71fb"
DATA = ORIGINAL / "data/mim_x/mnq_1min_by_contract.csv"
DATA_HASH = "ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea"
SOURCE_HASH = "b747ceff679c3874a6236d71eb960c4a68890aa5267aaa0575070b34b097a9a7"
REQUIRED = {
    "manifest.json",
    "audit.json",
    "definitions.md",
    "feasibility.json",
    "feasibility.md",
    "completion.json",
}
RUN_REQUIRED = {
    "events.csv",
    "trades.csv",
    "minute.csv",
    "accrual.csv",
    "daily.csv",
    "partitions.csv",
    "scenarios.csv",
    "summary.json",
    "exclusions.csv",
    "report.md",
    "report.html",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def readable(path):
    path = Path(path).resolve()
    if "sealed_holdout" in path.parts:
        raise ValueError("Holdout input prohibited")
    return path


def child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe snapshot-relative path")
    root = readable(root)
    result = readable(root / relative)
    if not result.is_relative_to(root):
        raise ValueError("Snapshot path escaped containment")
    return result


def read_json(path):
    return json.loads(readable(path).read_text())


def read_csv(path):
    return pd.read_csv(readable(path))


def digest(path):
    path = readable(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    with open(path, "x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def permitted(path):
    path = readable(path)
    if not path.is_relative_to(ORIGINAL) or path.is_relative_to(RUNS):
        raise ValueError("Input must be existing original-checkout evidence")
    return path


def inventory(path):
    entries = {}
    path = readable(path)
    for p in sorted(path.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlink in evidence inventory")
        if p.is_file() and p != Path(path) / "completion.json":
            entries[str(p.relative_to(path))] = digest(p)
    return entries


def verify_inventory(path):
    recorded = read_json(child(path, "completion.json"))["sha256"]
    for rel in recorded:
        child(path, rel)
    if inventory(path) != recorded:
        raise ValueError("Incomplete or changed inventory")


def validate_inputs(source, data):
    source = permitted(source)
    data = permitted(data)
    if digest(child(source, "completion.json")) != SOURCE_HASH:
        raise ValueError("Unapproved source completion")
    verify_inventory(source)
    if digest(data) != DATA_HASH:
        raise ValueError("Data hash mismatch")
    manifest = read_json(child(source, "manifest.json"))
    if manifest["inputs"].get(str(DATA)) != DATA_HASH:
        raise ValueError("Source data binding mismatch")
    comparison = permitted(manifest["bindings"]["baseline_run"])
    if (
        digest(child(comparison, "completion.json"))
        != manifest["inputs"][str(comparison / "completion.json")]
    ):
        raise ValueError("Comparison completion binding mismatch")
    verify_inventory(comparison)
    return source, data


def safe_run(path):
    raw = Path(path)
    if raw.is_symlink() or RUNS.is_symlink() or raw.resolve().parent != RUNS.resolve():
        raise ValueError("Output must be a fresh direct runs child")
    return raw.resolve()


def create(command, output=None):
    if RUNS.is_symlink():
        raise ValueError("Symlink runs directory")
    RUNS.mkdir(exist_ok=True)
    path = safe_run(
        output
        or RUNS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + "-"
            + command
            + "-"
            + uuid.uuid4().hex[:10]
        )
    )
    path.mkdir()  # refuses overwrite, including previous failures
    return path


def freeze(path, command, source, data, evidence):
    source, data = permitted(source), permitted(data)
    evidence = [permitted(p) for p in evidence]
    files = list(BASE.glob("*.py")) + [
        BASE / "definitions.md",
        ROOT / "research/mim_comparison/data.py",
        ROOT / "research/mim_lifecycle/analysis.py",
        ROOT / "AGENTS.md",
    ]
    sources = {
        str(p.relative_to(ROOT)): digest(child(ROOT, p.relative_to(ROOT)))
        for p in sorted(files)
    }
    inputs = (
        [data, source / "completion.json"]
        + [
            source / n
            for n in (
                "manifest.json",
                "ledger.csv",
                "trades.csv",
                "daily.csv",
                "exclusions.csv",
            )
        ]
        + evidence
    )
    inputs = [permitted(p) for p in inputs]
    bindings = {str(p): digest(p) for p in sorted(set(inputs))}
    manifest = dict(
        command=command,
        created_at=now(),
        runtime=dict(
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            machine=platform.machine(),
        ),
        source=sources,
        inputs=bindings,
        source_run=str(source),
        data=str(data),
        historical_only=True,
        definitions_sha256=digest(BASE / "definitions.md"),
        deployment_authorized=False,
    )
    for p in files:
        target = path / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(child(ROOT, p.relative_to(ROOT)), target)
    for p in inputs:
        if p == data:
            continue  # 141MB data bound by SHA; all compact evidence snapshotted
        target = path / "inputs" / p.relative_to(ORIGINAL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(readable(p), target)
    shutil.copyfile(readable(BASE / "definitions.md"), path / "definitions.md")
    json_write(path / "manifest.json", manifest)


def seal(path):
    json_write(path / "completion.json", dict(sha256=inventory(path)))
    for p in path.rglob("*"):
        if p.is_file():
            p.chmod(0o444)


def verify(path, sealed=True):
    """Independently validate inventory, joins and recorded accounting, not engine code."""
    path = safe_run(path)
    if sealed:
        verify_inventory(path)
    if (path / "failure.json").exists():
        raise ValueError("Failed invocation is preserved, not a successful run")
    files = set(inventory(path)) | {"completion.json"}
    if not REQUIRED <= files:
        raise ValueError("Missing mandatory inventory")
    m = read_json(child(path, "manifest.json"))
    required_inputs = {m["data"], str(Path(m["source_run"]) / "completion.json")} | {
        str(Path(m["source_run"]) / n)
        for n in (
            "manifest.json",
            "ledger.csv",
            "trades.csv",
            "daily.csv",
            "exclusions.csv",
        )
    }
    if not required_inputs <= set(m["inputs"]):
        raise ValueError("Missing input binding inventory")
    for p, h in m["inputs"].items():
        input_path = permitted(p)
        if digest(input_path) != h:
            raise ValueError("Frozen input drift")
        if (
            input_path != Path(m["data"])
            and digest(path / "inputs" / input_path.relative_to(ORIGINAL)) != h
        ):
            raise ValueError("Input snapshot drift")
    for rel, h in m["source"].items():
        if digest(child(path / "source", rel)) != h:
            raise ValueError("Source snapshot drift")
    expected_sources = {
        "research/mim_diagnostics/" + n
        for n in (
            "__init__.py",
            "__main__.py",
            "analysis.py",
            "artifacts.py",
            "feasibility.py",
            "report.py",
            "definitions.md",
        )
    }
    expected_sources |= {
        "research/mim_comparison/data.py",
        "research/mim_lifecycle/analysis.py",
        "AGENTS.md",
    }
    if not expected_sources <= set(m["source"]):
        raise ValueError("Missing implementation source inventory")
    if digest(path / "definitions.md") != m["definitions_sha256"]:
        raise ValueError("Definitions drift")
    validate_inputs(m["source_run"], m["data"])
    from .feasibility import validate_inventory, CALENDARS, DOCS, PROTOCOLS

    expected_evidence = {str(ORIGINAL / row[0]) for row in CALENDARS.values()} | {
        str(ORIGINAL / rel) for rel in DOCS + PROTOCOLS
    }
    if not expected_evidence <= set(m["inputs"]):
        raise ValueError("Missing evidence input inventory")
    validate_inventory(read_json(child(path, "feasibility.json")))
    if m["command"] == "audit":
        return dict(verified=True, command="audit")
    if m["command"] != "run" or not RUN_REQUIRED <= files:
        raise ValueError("Missing mandatory analytical inventory")
    tables = {
        n: read_csv(child(path, n + ".csv"))
        for n in (
            "events",
            "trades",
            "minute",
            "accrual",
            "daily",
            "partitions",
            "scenarios",
        )
    }
    summary = read_json(child(path, "summary.json"))
    t, d, minute, e, a, p = (
        tables[n]
        for n in ("trades", "daily", "minute", "events", "accrual", "partitions")
    )

    def agree(x, y, label):
        if not np.allclose(x, y, atol=1e-8, rtol=0):
            raise ValueError("Independent verification: " + label)

    if (len(d), len(t), len(e), len(minute)) != (1323, 801, 1595, 1323 * 390):
        raise ValueError("Pinned counts")
    if (
        d.day.duplicated().any()
        or minute.duplicated(["day", "contract", "timestamp"]).any()
        or t.trade_id.duplicated().any()
    ):
        raise ValueError("Duplicate output key")
    agree((t.exit_fill - t.entry_fill) * t.direction * 2, t.gross, "fill payoff")
    agree(t.gross - t.costs, t.net, "trade net")
    agree(t.costs, 2.24, "trade fees")
    agree(e.costs, e.quantity * 1.12, "event fees")
    for f in ("gross", "costs", "net"):
        agree(d[f].sum(), summary[f], "summary " + f)
        for table in (
            t,
            minute,
            a,
            e.rename(columns={"realized_gross": "gross", "realized_net": "net"}),
        ):
            if not set(zip(table.day, table.contract)) <= set(zip(d.day, d.contract)):
                raise ValueError("Missing accounting join")
            agree(
                table.groupby("day")[f].sum().reindex(d.day, fill_value=0),
                d[f],
                "daily " + f,
            )
        for _, g in p.groupby(["basis", "dimension"]):
            agree(g[f].sum(), d[f].sum(), "partition " + f)
    expected_parts = {
        ("entry_cohort", x)
        for x in ("direction", "exit_reason", "year", "entry_order", "entry_half_hour")
    } | {
        ("minute_accrual", x)
        for x in (
            "direction",
            "exit_reason",
            "year",
            "entry_order",
            "entry_half_hour",
            "accrual_half_hour",
        )
    }
    if set(zip(p.basis, p.dimension)) != expected_parts:
        raise ValueError("Missing attribution partition")
    agree(d.net.sum(), 21889.76, "pinned net")
    agree(minute.net.cumsum(), minute.marked_equity_net, "marked equity")
    agree(
        minute.session_marked_gross - minute.session_realized_gross,
        minute.unrealized_gross,
        "unrealized",
    )
    agree(
        minute.groupby("day").exposure_minutes_upper.sum().reindex(d.day),
        d.exposure_contract_minutes,
        "exposure",
    )
    if not minute.groupby("day").size().eq(390).all():
        raise ValueError("Missing minute grid")
    expected_dates = pd.to_datetime(minute.timestamp, utc=True).dt.tz_convert(
        "America/New_York"
    )
    if not (expected_dates.dt.strftime("%Y-%m-%d") == minute.day).all():
        raise ValueError("Minute day/timezone")
    for _, g in minute.assign(
        clock=expected_dates.dt.hour * 60 + expected_dates.dt.minute
    ).groupby("day"):
        if g.clock.tolist() != list(range(571, 961)):
            raise ValueError("Nonexact minute grid")
    src = Path(m["source_run"])
    if digest(path / "exclusions.csv") != digest(src / "exclusions.csv"):
        raise ValueError("Exclusions changed")
    source_daily = read_csv(permitted(src / "daily.csv"))
    source_daily = source_daily[
        (source_daily.arm == "A")
        & (source_daily.delay == 2)
        & (source_daily.cost == 2.24)
    ]
    if list(zip(d.day, d.contract)) != list(
        zip(source_daily.day, source_daily.contract)
    ):
        raise ValueError("Selected session grid changed")
    for f in ("gross", "costs", "net"):
        agree(d[f], source_daily[f], "source " + f)
    source_events = read_csv(permitted(src / "ledger.csv"))
    source_events["source_row"] = np.arange(len(source_events))
    source_events = source_events[
        (source_events.arm == "A")
        & (source_events.delay == 2)
        & (source_events.cost_scenario == 2.24)
    ]
    if (
        not e[source_events.columns]
        .fillna("")
        .reset_index(drop=True)
        .equals(source_events.fillna("").reset_index(drop=True))
    ):
        raise ValueError("Exported execution ledger changed")
    source_trades = read_csv(permitted(src / "trades.csv"))
    source_trades = source_trades[
        (source_trades.arm == "A") & (source_trades.delay == 2)
    ]
    joint = t.merge(
        source_trades,
        on=["day", "contract", "entry_event_timestamp"],
        suffixes=("", "_source"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not joint["_merge"].eq("both").all():
        raise ValueError("Source trade join")
    for col in (
        "entry_fill_timestamp",
        "exit_event_timestamp",
        "exit_fill_timestamp",
        "exit_fill_time_basis",
        "exit_reason",
    ):
        if not joint[col].fillna("").eq(joint[col + "_source"].fillna("")).all():
            raise ValueError("Source trade clock/label mismatch")
    for col in ("entry_fill", "exit_fill", "direction", "gross", "costs", "net"):
        agree(joint[col], joint[col + "_source"], "source trade " + col)
    for _, g in p.groupby(["basis", "dimension"]):
        agree(g["count"].sum(), 801, "partition count")
        agree(g.exposure_minutes_upper.sum(), 222148, "partition exposure upper")
        agree(g.exposure_minutes_lower.sum(), 222077, "partition exposure lower")
        agree(g.losing_trade_dollars.sum(), 76626.36, "partition losing dollars")
    agree(minute.exposure_minutes_lower.sum(), 222077, "exposure lower")
    agree(
        minute.marked_equity_net.to_numpy()
        - np.maximum.accumulate(np.r_[0.0, minute.marked_equity_net.to_numpy()])[1:],
        minute.marked_equity_net.to_numpy()
        - np.maximum.accumulate(np.r_[0.0, minute.net.cumsum().to_numpy()])[1:],
        "drawdown",
    )
    dd = -(
        minute.marked_equity_net.to_numpy()
        - np.maximum.accumulate(np.r_[0.0, minute.marked_equity_net.to_numpy()])[1:]
    ).min()
    agree(dd, summary["drawdown_minute"]["depth"], "drawdown summary")
    for label, count, net in [
        ("CAT_STOP", 71, -35331.04),
        ("EOD_CLOSE_PROXY", 723, 59824.48),
        ("REVERSAL", 7, -2603.68),
    ]:
        g = t[t.exit_reason == label]
        if len(g) != count:
            raise ValueError("Pinned exit count")
        agree(g.net.sum(), net, "pinned exit net")
    agree(d.nlargest(67, "net").net.sum(), 39552.92, "best 67")
    # Rebuild sampled gross changes from exported fills and each exact joined close.
    from research.mim_comparison.data import load

    bars = load(permitted(m["data"]), "end")
    joined = minute[["day", "contract", "timestamp", "close"]].copy()
    joined["timestamp"] = pd.to_datetime(joined.timestamp, utc=True)
    bars["timestamp"] = pd.to_datetime(bars.timestamp, utc=True)
    joined = joined.merge(
        bars[["day", "contract", "timestamp", "close"]],
        on=["day", "contract", "timestamp"],
        suffixes=("", "_bar"),
        how="left",
        validate="one_to_one",
    )
    agree(joined.close, joined.close_bar, "source close marks")
    index = {
        (r.day, r.contract): g.index.to_numpy()
        for (day, contract), g in minute.groupby(["day", "contract"])
        for r in [g.iloc[0]]
    }
    expected_gross = np.zeros(len(minute))
    expected_cost = np.zeros(len(minute))
    expected_position = np.zeros(len(minute))
    clock = pd.to_datetime(minute.timestamp, utc=True)
    for row in t.itertuples():
        ix = index[row.day, row.contract]
        ix = ix[
            (clock.iloc[ix] >= pd.Timestamp(row.entry_event_timestamp))
            & (clock.iloc[ix] <= pd.Timestamp(row.exit_event_timestamp))
        ]
        if not len(ix):
            raise ValueError("Empty minute trade join")
        marks = (minute.close.iloc[ix].to_numpy() - row.entry_fill) * row.direction * 2
        marks[-1] = row.gross
        expected_gross[ix] += np.diff(np.r_[0.0, marks])
        expected_cost[ix[0]] += 1.12
        expected_cost[ix[-1]] += 1.12
        expected_position[ix[:-1]] = row.direction
    agree(expected_gross, minute.gross, "independent minute payoff")
    agree(expected_cost, minute.costs, "independent minute costs")
    agree(expected_position, minute.position_after, "independent position")
    verify_details(tables, summary, bars, read_csv(permitted(src / "daily.csv")))
    return dict(
        verified=True,
        command="run",
        sessions=len(d),
        trades=len(t),
        net=float(d.net.sum()),
    )


def compare_values(actual, expected, label):
    """Recursive equality with accounting tolerance and exact structure/labels."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label + " keys differ")
        for key in expected:
            compare_values(actual[key], expected[key], label + "." + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label + " rows differ")
        for left, right in zip(actual, expected):
            compare_values(left, right, label)
    elif isinstance(expected, (float, int, np.number)) and not isinstance(
        expected, bool
    ):
        if (
            not isinstance(actual, (float, int, np.number))
            or not np.isfinite(actual)
            or abs(actual - expected) > 1e-8
        ):
            raise ValueError(label + " numeric mismatch")
    elif actual != expected:
        raise ValueError(label + " value mismatch")


def compare_frame(actual, expected, keys, label):
    if (
        set(actual) != set(expected)
        or actual.duplicated(keys).any()
        or expected.duplicated(keys).any()
    ):
        raise ValueError(label + " columns/duplicate keys")
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if len(actual) != len(expected):
        raise ValueError(label + " row count")
    for col in expected:
        if pd.api.types.is_numeric_dtype(expected[col]):
            if not np.allclose(actual[col], expected[col], rtol=0, atol=1e-8):
                raise ValueError(label + " " + col)
        elif (
            not actual[col]
            .fillna("")
            .astype(str)
            .eq(expected[col].fillna("").astype(str))
            .all()
        ):
            raise ValueError(label + " " + col)


def independent_drawdown(frame, clock):
    equity = frame.net.cumsum().to_numpy()
    high = 0.0
    depths = []
    longest = current = 0
    for value in equity:
        high = max(high, value)
        depth = value - high
        depths.append(depth)
        current = current + 1 if depth < 0 else 0
        longest = max(longest, current)
    trough = int(np.argmin(depths))
    peak = next((i for i in reversed(range(trough)) if depths[i] == 0), -1)
    recovery = next((i for i in range(trough + 1, len(depths)) if depths[i] == 0), None)
    return dict(
        depth=float(-depths[trough]),
        peak_timestamp=str(frame.iloc[peak][clock]) if peak >= 0 else "initial_zero",
        trough_timestamp=str(frame.iloc[trough][clock]),
        recovery_timestamp=(
            str(frame.iloc[recovery][clock]) if recovery is not None else None
        ),
        peak_to_trough_samples=trough - peak,
        peak_to_recovery_samples=recovery - peak if recovery is not None else None,
        longest_underwater_samples=longest,
        unrecovered_at_end=bool(depths[-1] < 0),
    )


def expected_details(trades, events, minute, daily, bars, source_daily):
    """Independent audit from source-bound fills/OHLC; no analysis helper imports."""
    dimensions = ("direction", "exit_reason", "year", "entry_order", "entry_half_hour")

    def bucket(clock):
        clock = pd.Timestamp(clock).tz_convert("America/New_York")
        start = 570 + ((clock.hour * 60 + clock.minute - 571) // 30) * 30
        return (
            f"{start//60:02d}:{start%60:02d}-{(start+30)//60:02d}:{(start+30)%60:02d}"
        )

    bar_groups = {
        key: g.sort_values("timestamp") for key, g in bars.groupby(["day", "contract"])
    }
    expected_trades = []
    contributions = []
    day_order = {}
    for t in trades.to_dict("records"):
        day_order[t["day"]] = day_order.get(t["day"], 0) + 1
        seq = day_order[t["day"]]
        cohorts = dict(
            direction=t["direction"],
            exit_reason=t["exit_reason"],
            year=int(t["day"][:4]),
            entry_order="first" if seq == 1 else "subsequent",
            entry_half_hour=bucket(t["entry_event_timestamp"]),
        )
        for key, value in cohorts.items():
            compare_values(t[key], value, "trade cohort " + key)
        execution = events[
            (events.day == t["day"])
            & (events.contract == t["contract"])
            & (events.event_timestamp == t["entry_event_timestamp"])
            & (events.position_after == t["direction"])
        ]
        if len(execution) != 1:
            raise ValueError("Trade entry execution join")
        compare_values(
            t["entry_original_reason"],
            execution.iloc[0].reason,
            "entry original reason",
        )
        compare_values(
            t["entry_transition"], execution.iloc[0].transition, "entry transition"
        )
        compare_values(t["trade_id"], f"{t['day']}-{seq:02d}", "trade id")
        b = bar_groups[t["day"], t["contract"]]
        clock = pd.to_datetime(b.timestamp, utc=True)
        start = pd.Timestamp(t["entry_event_timestamp"])
        end = pd.Timestamp(t["exit_event_timestamp"])
        entry = pd.Timestamp(t["entry_fill_timestamp"])
        selected = b[(clock >= start) & (clock <= end)]
        if selected.empty:
            raise ValueError("Missing audit trade bars")
        marks = (selected.close.to_numpy() - t["entry_fill"]) * t["direction"] * 2
        marks[-1] = t["gross"]
        changes = np.diff(np.r_[0.0, marks])
        fees = np.zeros(len(selected))
        fees[0] += 1.12
        fees[-1] += 1.12
        upper = np.ones(len(selected))
        lower = np.ones(len(selected))
        uncertain = t["exit_fill_time_basis"] == "intrabar_unknown_within_event_minute"
        if t["exit_fill_time_basis"] == "bar_open":
            upper[-1] = lower[-1] = 0
        elif uncertain:
            lower[-1] = 0
        for i, row in enumerate(selected.itertuples()):
            contributions.append(
                dict(
                    trade_id=t["trade_id"],
                    day=t["day"],
                    contract=t["contract"],
                    timestamp=pd.Timestamp(row.timestamp)
                    .tz_convert("America/New_York")
                    .isoformat(),
                    **cohorts,
                    accrual_half_hour=bucket(row.timestamp),
                    gross=changes[i],
                    costs=fees[i],
                    net=changes[i] - fees[i],
                    entry_count=int(i == 0),
                    exposure_minutes_upper=upper[i],
                    exposure_minutes_lower=lower[i],
                    eventual_losers_net_contribution=(
                        changes[i] - fees[i] if t["net"] < 0 else 0.0
                    ),
                )
            )
        terminal_lower = (
            end - pd.Timedelta(minutes=1)
            if uncertain
            else pd.Timestamp(t["exit_fill_timestamp"])
        )
        terminal_upper = end if uncertain else terminal_lower
        held = b[
            (clock > entry) & (clock < end if uncertain else clock <= terminal_lower)
        ]
        samples = np.r_[t["entry_fill"], held.close.to_numpy(), t["exit_fill"]]
        ranges = np.r_[samples, held.high.to_numpy(), held.low.to_numpy()]
        sample_payoff = (samples - t["entry_fill"]) * t["direction"] * 2
        range_payoff = (ranges - t["entry_fill"]) * t["direction"] * 2
        details = dict(
            mfe=float(range_payoff.max()),
            mae=float(-range_payoff.min()),
            sampled_mfe=float(sample_payoff.max()),
            sampled_mae=float(-sample_payoff.min()),
            coverage=(
                "incomplete_stop_interval"
                if uncertain
                else "complete_recorded_bar_path"
            ),
            duration_min_lower=float(
                (terminal_lower - entry) / pd.Timedelta(minutes=1)
            ),
            duration_min_upper=float(
                (terminal_upper - entry) / pd.Timedelta(minutes=1)
            ),
            valid_completed_closes=len(held),
            exposure_minutes_upper=float(upper.sum()),
            exposure_minutes_lower=float(lower.sum()),
        )
        for key, value in details.items():
            compare_values(t[key], value, "excursion " + key)
        expected_trades.append(dict(t, **details))
    a = pd.DataFrame(contributions)
    t = pd.DataFrame(expected_trades)
    parts = []
    loss_total = float(-t.net.clip(upper=0).sum())
    for basis, frame, dims in [
        ("entry_cohort", t, dimensions),
        ("minute_accrual", a, dimensions + ("accrual_half_hour",)),
    ]:
        for dim in dims:
            for key, g in frame.groupby(dim, sort=True):
                loss = (
                    float(-g.net.clip(upper=0).sum())
                    if basis == "entry_cohort"
                    else float(-g.eventual_losers_net_contribution.sum())
                )
                parts.append(
                    dict(
                        basis=basis,
                        dimension=dim,
                        group=str(key),
                        count=(
                            len(g)
                            if basis == "entry_cohort"
                            else int(g.entry_count.sum())
                        ),
                        gross=float(g.gross.sum()),
                        costs=float(g.costs.sum()),
                        net=float(g.net.sum()),
                        losing_trade_dollars=loss,
                        losing_trade_share=loss / loss_total if loss_total else 0.0,
                        exposure_minutes_upper=float(g.exposure_minutes_upper.sum()),
                        exposure_minutes_lower=float(g.exposure_minutes_lower.sum()),
                    )
                )
    p = pd.DataFrame(parts)
    for basis, dim in [
        ("entry_cohort", "entry_half_hour"),
        ("minute_accrual", "entry_half_hour"),
        ("minute_accrual", "accrual_half_hour"),
    ]:
        present = set(p.loc[(p.basis == basis) & (p.dimension == dim), "group"])
        for m in range(570, 960, 30):
            label = f"{m//60:02d}:{m%60:02d}-{(m+30)//60:02d}:{(m+30)%60:02d}"
            if label not in present:
                row = {col: 0.0 for col in p}
                row.update(basis=basis, dimension=dim, group=label)
                p.loc[len(p)] = row
    scenarios = (
        source_daily[source_daily.arm.eq("A")]
        .groupby(["delay", "cost"])
        .agg(
            sessions=("day", "size"),
            gross=("gross", "sum"),
            costs=("costs", "sum"),
            net=("net", "sum"),
        )
        .reset_index()
    )
    transitions = {}
    last_day = None
    position = 0
    previous_direction = None
    entry_basis = None
    for e in events.to_dict("records"):
        if e["day"] != last_day:
            position = 0
            previous_direction = None
            last_day = e["day"]
        after = e["position_after"]
        gross = position * (e["fill"] - entry_basis) * 2 if position else 0.0
        compare_values(e["realized_gross"], gross, "event realized gross")
        compare_values(e["realized_net"], gross - e["costs"], "event realized net")
        if after:
            entry_basis = e["fill"]
        label = (
            "true_reversal"
            if position and after
            else (
                "opposite_entry_after_flat"
                if after
                and previous_direction is not None
                and after != previous_direction
                else "entry_from_flat" if after else "exit_to_flat"
            )
        )
        compare_values(e["transition"], label, "event transition")
        compare_values(e["position_before"], position, "position before")
        compare_values(
            e["event_half_hour"], bucket(e["event_timestamp"]), "event clock"
        )
        transitions[label] = transitions.get(label, 0) + 1
        if after:
            previous_direction = after
        position = after
    top = daily.sort_values(["net", "day"], ascending=[False, True]).head(67)
    summary = dict(
        descriptive_only=True,
        primary=dict(arm="A", delay=2, roundtrip_cost=2.24, quantity=1, point_value=2),
        sessions=len(daily),
        trades=len(t),
        events=len(events),
        flat_sessions=int(daily.costs.eq(0).sum()),
        gross=float(daily.gross.sum()),
        costs=float(daily.costs.sum()),
        net=float(daily.net.sum()),
        best_67_net=float(top.net.sum()),
        remainder_net=float(daily.net.sum() - top.net.sum()),
        best_67_share=float(top.net.sum() / daily.net.sum()),
        best_67_days=top[["day", "net"]].to_dict("records"),
        top_5_percent_trades_net=float(
            t.nlargest(int(np.ceil(len(t) * 0.05)), "net").net.sum()
        ),
        losing_trade_dollars=loss_total,
        exposure_minutes_upper=float(a.exposure_minutes_upper.sum()),
        exposure_minutes_lower=float(a.exposure_minutes_lower.sum()),
        uncertain_stop_trades=int(t.coverage.eq("incomplete_stop_interval").sum()),
        transitions=transitions,
        drawdown_minute=independent_drawdown(minute, "timestamp"),
        drawdown_daily=independent_drawdown(daily, "day"),
    )
    return a, p, scenarios, summary


def verify_details(tables, summary, bars, source_daily):
    expected_accrual, expected_partitions, expected_scenarios, expected_summary = (
        expected_details(
            tables["trades"],
            tables["events"],
            tables["minute"],
            tables["daily"],
            bars,
            source_daily,
        )
    )
    actual = tables["accrual"].copy()
    actual["timestamp"] = (
        pd.to_datetime(actual.timestamp, utc=True)
        .dt.tz_convert("America/New_York")
        .map(lambda x: x.isoformat())
    )
    compare_frame(actual, expected_accrual, ["trade_id", "timestamp"], "accrual")
    aggregate = (
        expected_accrual.groupby(["day", "contract", "timestamp"])[
            [
                "gross",
                "costs",
                "net",
                "exposure_minutes_upper",
                "exposure_minutes_lower",
            ]
        ]
        .sum()
        .reset_index()
    )
    minute = tables["minute"].copy()
    minute["timestamp"] = (
        pd.to_datetime(minute.timestamp, utc=True)
        .dt.tz_convert("America/New_York")
        .map(lambda x: x.isoformat())
    )
    joined = (
        minute[["day", "contract", "timestamp"]]
        .merge(
            aggregate,
            on=["day", "contract", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        .fillna(0)
    )
    compare_frame(
        minute[joined.columns],
        joined,
        ["day", "contract", "timestamp"],
        "minute accrual",
    )
    labels = pd.to_datetime(minute.timestamp, utc=True).dt.tz_convert(
        "America/New_York"
    )
    starts = 570 + ((labels.dt.hour * 60 + labels.dt.minute - 571) // 30) * 30
    buckets = starts.map(
        lambda m: f"{m//60:02d}:{m%60:02d}-{(m+30)//60:02d}:{(m+30)%60:02d}"
    )
    if not minute.accrual_half_hour.eq(buckets).all():
        raise ValueError("Minute bucket clock")
    events = tables["events"].copy()
    events["timestamp"] = (
        pd.to_datetime(events.event_timestamp, utc=True)
        .dt.tz_convert("America/New_York")
        .map(lambda x: x.isoformat())
    )
    realized = (
        events.groupby(["day", "contract", "timestamp"])
        .realized_gross.sum()
        .reset_index()
    )
    computed = minute[["day", "contract", "timestamp", "gross", "costs", "net"]].merge(
        realized, on=["day", "contract", "timestamp"], how="left", validate="one_to_one"
    )
    computed["realized_gross"] = computed.realized_gross.fillna(0.0)
    for target, col in [
        ("session_realized_gross", "realized_gross"),
        ("session_costs", "costs"),
        ("session_marked_gross", "gross"),
        ("session_marked_net", "net"),
    ]:
        computed[target] = computed.groupby("day")[col].cumsum()
    computed["session_realized_net"] = (
        computed.session_realized_gross - computed.session_costs
    )
    computed["unrealized_gross"] = (
        computed.session_marked_gross - computed.session_realized_gross
    )
    compare_frame(
        minute[computed.columns],
        computed,
        ["day", "contract", "timestamp"],
        "minute accounting",
    )
    compare_frame(
        tables["partitions"],
        expected_partitions,
        ["basis", "dimension", "group"],
        "partition",
    )
    compare_frame(
        tables["scenarios"], expected_scenarios, ["delay", "cost"], "scenario"
    )
    compare_values(summary, expected_summary, "summary")
