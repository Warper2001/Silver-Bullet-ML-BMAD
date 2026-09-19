"""Attributable synchronized marks; shared-account balance logs are not inputs."""

from collections import defaultdict
from math import sqrt
from statistics import mean, stdev
from .common import number, timestamp, known_epoch
from datetime import timezone
from zoneinfo import ZoneInfo


def metrics(values, capital=10000.0):
    """Session cash P&L, including zero sessions; equity begins at stated capital."""
    if not values or any(v is None for v in values):
        return dict(status="INSUFFICIENT_DATA")
    equity, peak, max_dd, duration, longest = capital, capital, 0.0, 0, 0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        duration = duration + 1 if equity < peak else 0
        longest = max(longest, duration)
    sd = stdev(values) if len(values) > 1 else 0.0
    return dict(
        status="DESCRIPTIVE_ONLY",
        sessions=len(values),
        net=sum(values),
        mean_session=mean(values),
        session_sd=sd,
        net_sharpe=sqrt(252) * mean(values) / sd if sd else None,
        max_drawdown=max_dd,
        max_underwater_sessions=longest,
        unrecovered_at_end=equity < peak,
        end_equity=equity,
        return_on_initial_capital=sum(values) / capital if capital > 0 else None,
    )


def marked_curves(fills, marks, coverage, *, assumptions=None):
    """Return complete grid curves, or unknown. No forward-filled open marks.

    coverage: {start,end,initial_flat:true,strategies:[...],baseline_units:{...}}
    fills: id, timestamp, account, epoch, strategy, contract, signed_quantity,
           price, actual_cost (cash total). marks: timestamp, contract, close,
           multiplier, corrected_contract:true. All exports cover the whole interval.
    """
    required = {
        "id",
        "timestamp",
        "account",
        "epoch",
        "strategy",
        "contract",
        "signed_quantity",
        "price",
        "actual_cost",
    }
    if not assumptions and (
        coverage.get("initial_flat") is not True
        or coverage.get("complete_export") is not True
    ):
        return (
            dict(
                status="INSUFFICIENT_DATA",
                reason="certified flat start and complete export required",
            ),
            [],
        )
    start, end = timestamp(coverage["start"]), timestamp(coverage["end"])
    if end <= start:
        raise ValueError("coverage end must exceed start")
    strategies = coverage["strategies"]
    units = coverage["baseline_units"]
    if set(strategies) != set(units) or any(float(u) <= 0 for u in units.values()):
        raise ValueError("positive baseline units required for each strategy")
    seen, clean = {}, []
    unknown_epoch = any(not known_epoch(fill.get("epoch")) for fill in fills)
    single_session = (
        start.astimezone(ZoneInfo("America/New_York")).date()
        == end.astimezone(ZoneInfo("America/New_York")).date()
    )
    if unknown_epoch and (not assumptions or not single_session):
        return (
            dict(
                status="INSUFFICIENT_DATA",
                reason="explicit account epoch required outside per-session conditional accounting",
            ),
            [],
        )
    if unknown_epoch:
        required = required - {"epoch"}
    for fill in fills:
        if not required <= fill.keys() or any(fill[k] in ("", None) for k in required):
            return (
                dict(
                    status="INSUFFICIENT_DATA",
                    reason="fill attribution or cost missing",
                ),
                [],
            )
        key = (str(fill["account"]), str(fill.get("epoch")), str(fill["id"]))
        if key in seen:
            if seen[key] != fill:
                raise ValueError("conflicting duplicate fill")
            continue
        seen[key] = fill
        if fill["strategy"] not in strategies:
            raise ValueError("unregistered strategy")
        time = timestamp(fill["timestamp"])
        if not start <= time <= end:
            raise ValueError("fill outside certified coverage")
        if not number(fill["signed_quantity"]):
            raise ValueError("zero fill quantity")
        clean.append(fill)
    clean.sort(key=lambda f: (timestamp(f["timestamp"]), str(f["id"])))
    epochs = defaultdict(set)
    for fill in clean:
        epochs[fill["account"]].add(fill.get("epoch"))
    if any(len(v) > 1 for v in epochs.values()):
        return (
            dict(
                status="INSUFFICIENT_DATA",
                reason="split account reset epochs into separate covered runs",
            ),
            [],
        )
    grid = defaultdict(dict)
    multipliers = {}
    for mark in marks:
        time = timestamp(mark["timestamp"])
        if not start <= time <= end:
            continue
        if str(mark.get("corrected_contract")).lower() != "true":
            raise ValueError("marks require corrected contract attestation")
        contract = mark["contract"]
        multiplier = number(mark["multiplier"])
        if multiplier is None or multiplier <= 0:
            raise ValueError("positive multiplier required")
        if contract in multipliers and multipliers[contract] != multiplier:
            raise ValueError("contract multiplier changed")
        multipliers[contract] = multiplier
        if contract in grid[time] and grid[time][contract] != mark:
            raise ValueError("conflicting duplicate mark")
        grid[time][contract] = mark
    if not grid or max(grid) != end or min(grid) != start:
        return (
            dict(
                status="INSUFFICIENT_DATA",
                reason="marks must cover certified endpoints",
            ),
            [],
        )
    # Explicit grid cadence detects entirely absent minute rows, not only absent contracts.
    times = sorted(grid)
    cadence = float(coverage["grid_seconds"])
    if cadence <= 0 or any(
        (b - a).total_seconds() != cadence for a, b in zip(times, times[1:])
    ):
        return (
            dict(
                status="INSUFFICIENT_DATA",
                reason="missing synchronized grid timestamps",
            ),
            [],
        )
    positions, cash = defaultdict(float), defaultdict(float)
    rows, index, missing = [], 0, []
    for time in times:
        while index < len(clean) and timestamp(clean[index]["timestamp"]) <= time:
            fill = clean[index]
            contract, strategy = fill["contract"], fill["strategy"]
            if contract not in multipliers:
                return (
                    dict(
                        status="INSUFFICIENT_DATA", reason="missing fill contract marks"
                    ),
                    [],
                )
            qty = number(fill["signed_quantity"])
            key = (strategy, fill["account"], fill.get("epoch"), contract)
            positions[key] += qty
            cash[strategy] -= qty * number(fill["price"]) * multipliers[
                contract
            ] + number(fill["actual_cost"])
            index += 1
        for strategy in strategies:
            value, exposure, valid = cash[strategy], 0.0, True
            for (s, account, epoch, contract), qty in positions.items():
                if s != strategy or abs(qty) < 1e-12:
                    continue
                if (
                    contract not in grid[time]
                    or number(grid[time][contract].get("close")) is None
                ):
                    valid = False
                    missing.append(
                        dict(
                            timestamp=time.isoformat(),
                            strategy=strategy,
                            contract=contract,
                        )
                    )
                    continue
                price = number(grid[time][contract]["close"])
                value += qty * price * multipliers[contract]
                exposure += abs(qty * price * multipliers[contract])
            rows.append(
                dict(
                    timestamp=time.isoformat(),
                    strategy=strategy,
                    cumulative_net=value if valid else None,
                    gross_notional=exposure if valid else None,
                    baseline_units=float(units[strategy]),
                )
            )
    return (
        dict(
            status=(
                "INSUFFICIENT_DATA"
                if missing
                else ("CONDITIONAL_DESCRIPTIVE" if assumptions else "DESCRIPTIVE_ONLY")
            ),
            assumptions=assumptions or [],
            epoch_scope="PER_SESSION_ONLY" if unknown_epoch else "EXPLICIT_EPOCH",
            missing_marks=missing,
            intraminute_uncertainty="minute closes omit excursions and within-minute fill order",
            final_open_positions=[
                dict(strategy=k[0], account=k[1], epoch=k[2], contract=k[3], quantity=v)
                for k, v in positions.items()
                if abs(v) > 1e-12
            ],
        ),
        rows,
    )


def portfolio_report(
    rows, capital=10000.0, operating_cost_monthly=0.0, cost_stress=(0.0, 1.0, 2.0)
):
    if not rows or any(r["cumulative_net"] is None for r in rows):
        return dict(
            status="INSUFFICIENT_DATA",
            reason="complete attributable fills, costs and matching marks required",
        )
    curves = defaultdict(dict)
    for row in rows:
        curves[timestamp(row["timestamp"]).astimezone(timezone.utc).isoformat()][
            row["strategy"]
        ] = row
    strategies = set(next(iter(curves.values())))
    if not {"MIM", "YANK"} <= strategies or any(
        set(v) != strategies for v in curves.values()
    ):
        return dict(
            status="INSUFFICIENT_DATA", reason="synchronized MIM and YANK required"
        )
    scenarios = {
        "actual": None,
        "MIM1_YANK2": {"MIM": 1.0, "YANK": 2.0},
        "MIM1_YANK1": {"MIM": 1.0, "YANK": 1.0},
    }
    report = {}
    actual_notional = sum(
        sum(r["gross_notional"] for r in v.values()) for v in curves.values()
    )
    for name, weights in scenarios.items():
        series, notionals = [], []
        for time, values in sorted(curves.items(), key=lambda item: timestamp(item[0])):
            factors = {
                s: (
                    weights.get(s, 0.0) / values[s]["baseline_units"]
                    if weights
                    else 1.0
                )
                for s in strategies
            }
            series.append(
                (
                    time,
                    sum(values[s]["cumulative_net"] * factors[s] for s in strategies),
                )
            )
            notionals.append(
                sum(values[s]["gross_notional"] * abs(factors[s]) for s in strategies)
            )
        for normalized in (False, True):
            denom = sum(notionals)
            key = name + ("_same_gross_exposure" if normalized else "")
            if normalized and denom == 0 and actual_notional > 0:
                report[key] = dict(
                    status="INSUFFICIENT_DATA",
                    reason="zero candidate exposure cannot match positive actual exposure",
                    exposure_scale=None,
                )
                continue
            factor = actual_notional / denom if normalized and denom else 1.0
            daily = {}
            for time, value in series:
                # ET session date, not UTC calendar date.
                daily[
                    timestamp(time)
                    .astimezone(ZoneInfo("America/New_York"))
                    .date()
                    .isoformat()
                ] = (value * factor)
            prev, pnls = 0.0, []
            for day, value in sorted(daily.items()):
                pnls.append((day, value - prev))
                prev = value
            months = defaultdict(float)
            for day, pnl in pnls:
                months[day[:7]] += pnl
            net_months = {m: v - operating_cost_monthly for m, v in months.items()}
            key = name + ("_same_gross_exposure" if normalized else "")
            report[key] = dict(
                metrics=metrics([v for _, v in pnls], capital),
                intraday_path=path_risk([(t, v * factor) for t, v in series]),
                monthly_before_operating_cost=dict(months),
                monthly_after_operating_cost=net_months,
                monthly_variability_sd=(
                    stdev(net_months.values()) if len(net_months) > 1 else None
                ),
                exposure_scale=factor,
                gross_notional_time_sum=sum(notionals) * factor,
                capital_sensitivity={
                    str(c): metrics([v for _, v in pnls], c)
                    for c in (5000.0, 10000.0, 20000.0)
                },
            )
    daily_strategy = defaultdict(dict)
    for time, values in sorted(curves.items(), key=lambda item: timestamp(item[0])):
        daily_strategy[
            timestamp(time).astimezone(ZoneInfo("America/New_York")).date().isoformat()
        ] = {s: v["cumulative_net"] for s, v in values.items()}
    prev = {s: 0.0 for s in strategies}
    concurrent = 0
    for values in daily_strategy.values():
        concurrent += all(values[s] - prev[s] < 0 for s in ("MIM", "YANK"))
        prev = values
    return dict(
        status="DESCRIPTIVE_ONLY",
        scenarios=report,
        concurrent_loss_sessions=concurrent,
        cost_stress_note="Use fill_cost_sweep to apply additional per-filled-contract modeled slippage.",
        intraminute_uncertainty="not bounded by close-only marks",
        scaling_warning="No linear inference to $20,000 monthly income; capacity and evidence unproven",
    )


def fill_cost_sweep(
    fills, marks, coverage, costs, capital=10000.0, operating_cost_monthly=0.0
):
    result = {}
    for extra in costs:
        extra = float(extra)
        if extra < 0:
            raise ValueError("cost stress must be nonnegative")
        stressed = [
            dict(
                f,
                actual_cost=number(f["actual_cost"])
                + extra * abs(number(f["signed_quantity"])),
            )
            for f in fills
        ]
        status, rows = marked_curves(stressed, marks, coverage)
        result[str(extra)] = dict(
            modeled_additional_cost_per_filled_contract=extra,
            marks_status=status,
            portfolio=portfolio_report(rows, capital, operating_cost_monthly),
        )
    return result


def path_risk(series):
    peak = 0.0
    peak_time = None
    deepest = 0.0
    underway = None
    longest = 0.0
    recovery = None
    previous = 0.0
    losses = 0
    for time, value in sorted(series, key=lambda item: timestamp(item[0])):
        now = timestamp(time)
        losses += value < previous
        previous = value
        if value >= peak:
            if underway is not None:
                elapsed = (now - underway).total_seconds()
                if elapsed >= longest:
                    longest = elapsed
                    recovery = time
            peak = value
            peak_time = now
            underway = None
        else:
            if underway is None:
                underway = peak_time or now
            deepest = max(deepest, peak - value)
    open_seconds = (
        (max(timestamp(t) for t, _ in series) - underway).total_seconds()
        if underway
        else 0.0
    )
    return dict(
        max_marked_drawdown=deepest,
        longest_recovered_seconds=longest,
        longest_recovery_at=recovery,
        current_underwater_seconds=open_seconds,
        losing_mark_intervals=losses,
        intraminute_excursions="unknown",
    )
