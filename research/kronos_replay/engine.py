"""Minute-open replay with explicit schedules and causal event ordering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import statistics
from typing import Any, Iterable

import pandas as pd

from tools import kronos_inference_pilot as pilot
from .providers import ForecastProvider

SCOPE = "SYNTHETIC_MECHANICS_ONLY"
ARMS = ("kronos", "momentum", "flat")
MINUTE = pd.Timedelta(minutes=1)


def stamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if (
        pd.isna(result)
        or result.tzinfo is None
        or result != result.floor("min")
    ):
        raise ValueError("timestamps must be aware whole minutes")
    return result.tz_convert("America/New_York")


@dataclass(frozen=True)
class Session:
    opening: pd.Timestamp
    close: pd.Timestamp

    def __post_init__(self) -> None:
        object.__setattr__(self, "opening", stamp(self.opening))
        object.__setattr__(self, "close", stamp(self.close))
        if self.close <= self.opening or (
            self.close - self.opening
        ) % pd.Timedelta(minutes=15):
            raise ValueError(
                "sessions must contain positive complete 15-minute intervals"
            )
        if self.opening.date() != self.close.date():
            raise ValueError("session must be contained in one New York date")


@dataclass(frozen=True)
class Costs:
    latency_seconds: float = 0.0
    fee_per_side: float = 0.5
    slippage_ticks: int = 1

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) and v >= 0
            for v in (
                self.latency_seconds,
                self.fee_per_side,
                self.slippage_ticks,
            )
        ) or self.slippage_ticks != int(self.slippage_ticks):
            raise ValueError(
                "costs/latency must be finite and nonnegative; ticks integral"
            )


@dataclass
class Account:
    position: int = 0
    entry: float = 0.0
    realized: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    turnover: int = 0
    mark: float = 0.0
    peak: float = 0.0
    max_drawdown: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        unrealized = self.position * (self.mark - self.entry) * 2
        equity = self.realized + unrealized - self.fees
        peak = max(self.peak, equity)
        drawdown = peak - equity
        if not all(
            math.isfinite(v)
            for v in (
                *asdict(self).values(),
                unrealized,
                equity,
                peak,
                drawdown,
            )
        ):
            raise ValueError("non-finite projected accounting")
        self.peak = peak
        self.max_drawdown = max(self.max_drawdown, drawdown)
        return dict(
            asdict(self),
            unrealized=unrealized,
            equity=equity,
            drawdown=self.peak - equity,
        )


@dataclass
class ReplayResult:
    status: str = "INCOMPLETE"
    error: str | None = None
    accounts: dict[str, Account] = field(
        default_factory=lambda: {a: Account() for a in ARMS}
    )
    pending: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    forecasts: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    equity: list[dict[str, Any]] = field(default_factory=list)
    aggregated: list[dict[str, Any]] = field(default_factory=list)
    processed_minutes: int = 0

    def report(self) -> dict[str, Any]:
        return dict(
            scope=SCOPE,
            economic_evaluation=False,
            trading_authorized=False,
            strategy_test_permitted=False,
            status=self.status,
            error=self.error,
            processed_minutes=self.processed_minutes,
            pending=self.pending,
            accounts={a: s.snapshot() for a, s in self.accounts.items()},
        )


def execute(
    result: ReplayResult,
    arm: str,
    target: int,
    price: float,
    when: pd.Timestamp,
    contract: str,
    costs: Costs,
    reason: str,
) -> None:
    """Reversal fills two sides; disclose fees and slippage separately."""
    account = result.accounts[arm]
    if target == account.position:
        return
    proposed = replace(account, mark=price)
    fills = []
    sides = []
    if account.position:
        sides.append((-account.position, "close"))
    if target:
        sides.append((target, "open"))
    for side, action in sides:
        fill = price + side * costs.slippage_ticks * 0.25
        if not math.isfinite(fill) or fill <= 0:
            raise ValueError("nonpositive or non-finite proposed fill")
        if action == "close":
            proposed.realized += (
                proposed.position * (fill - proposed.entry) * 2
            )
            proposed.position, proposed.entry = 0, 0.0
        else:
            proposed.position, proposed.entry = side, fill
        proposed.fees += costs.fee_per_side
        proposed.slippage += costs.slippage_ticks * 0.25 * 2
        proposed.turnover += 1
        proposed.snapshot()
        fills.append(
            dict(
                timestamp=when.isoformat(),
                arm=arm,
                contract=contract,
                side=side,
                action=action,
                price=fill,
                reference_open=price,
                fee=costs.fee_per_side,
                reason=reason,
            )
        )
    account.__dict__.update(asdict(proposed))
    result.fills.extend(fills)


def run_replay(
    bars: Iterable[dict[str, Any]],
    sessions: list[Session],
    provider: ForecastProvider,
    costs: Costs = Costs(),
) -> ReplayResult:
    result = ReplayResult()
    context: list[dict[str, Any]] = []
    bucket: list[dict[str, Any]] = []
    contract: str | None = None
    iterator = iter(bars)
    try:
        if not sessions or any(
            b.opening <= a.close for a, b in zip(sessions, sessions[1:])
        ):
            raise ValueError(
                "explicit nonoverlapping ordered sessions required"
            )
        for session in sessions:
            for expected in pd.date_range(
                session.opening, session.close - MINUTE, freq="min"
            ):
                try:
                    row = next(iterator)
                except StopIteration as exc:
                    raise ValueError(
                        f"missing minute {expected.isoformat()}"
                    ) from exc
                when = stamp(row["timestamp"])
                if when != expected:
                    raise ValueError(
                        f"invalid sequence: expected {expected}, got {when}"
                    )
                identity = row["contract"]
                if not isinstance(identity, str) or not identity.strip():
                    raise ValueError("contract identity required")
                values = [
                    float(row[k]) for k in pilot.PRICE_COLUMNS + ["volume"]
                ]
                opening, high, low, close, volume = values
                if (
                    not all(math.isfinite(v) for v in values)
                    or min(values[:4]) <= 0
                    or volume < 0
                    or high < max(opening, close, low)
                    or low > min(opening, close, high)
                ):
                    raise ValueError("invalid minute OHLCV")
                # Validate the supplied minute before using its opening.
                if identity != contract:
                    if any(a.position for a in result.accounts.values()):
                        raise ValueError(
                            "contract change with outstanding exposure"
                        )
                    if result.pending or bucket:
                        raise ValueError(
                            "roll with pending orders or partial bucket"
                        )
                    context = []
                    contract = identity
                if when == session.close - MINUTE:
                    for arm in ARMS:
                        execute(
                            result,
                            arm,
                            0,
                            opening,
                            when,
                            contract,
                            costs,
                            "session_flatten",
                        )
                    result.pending.clear()
                else:
                    ready = [
                        order
                        for order in result.pending
                        if when > order["available"]
                    ]
                    for order in ready:
                        execute(
                            result,
                            order["arm"],
                            order["target"],
                            opening,
                            when,
                            contract,
                            costs,
                            "forecast_target",
                        )
                        result.pending.remove(order)
                bucket.append(
                    dict(
                        open=opening,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                    )
                )
                end = when + MINUTE
                if len(bucket) == 15:
                    bar = dict(
                        timestamp=end,
                        contract=contract,
                        open=bucket[0]["open"],
                        high=max(r["high"] for r in bucket),
                        low=min(r["low"] for r in bucket),
                        close=close,
                        volume=sum(r["volume"] for r in bucket),
                    )
                    bar["amount"] = (
                        bar["volume"]
                        * sum(bar[k] for k in pilot.PRICE_COLUMNS)
                        / 4
                    )
                    if not all(
                        math.isfinite(bar[k]) for k in pilot.VALUE_COLUMNS
                    ):
                        raise ValueError("non-finite aggregate OHLCV/amount")
                    context.append(bar)
                    result.aggregated.append(bar.copy())
                    bucket = []
                    if (
                        len(context) >= pilot.CONTEXT
                        and end + pd.Timedelta(minutes=60) <= session.close
                    ):
                        frame = pd.DataFrame(context[-128:]).set_index(
                            "timestamp"
                        )[pilot.VALUE_COLUMNS]
                        future = pd.date_range(
                            end + pd.Timedelta(minutes=15),
                            periods=4,
                            freq="15min",
                        )
                        paths: list[pd.DataFrame] = []
                        error = None
                        try:
                            raw = provider.forecast(
                                frame.copy(deep=True), future
                            )
                            paths = raw if isinstance(raw, list) else [raw]
                            if not isinstance(raw, list) or len(paths) != 3:
                                raise ValueError(
                                    "exactly three forecast paths required"
                                )
                            for path in paths:
                                check = pilot.validate_forecast(path, future)
                                if check["invalid_candles"]:
                                    raise ValueError("invalid forecast candle")
                            terminal = statistics.mean(
                                float(p.close.iloc[-1]) for p in paths
                            )
                            target = int(terminal > close) - int(
                                terminal < close
                            )
                        except Exception as exc:
                            target = 0
                            paths = getattr(exc, "paths", paths)
                            error = f"{type(exc).__name__}: {exc}"
                        for seed, path in enumerate(paths):
                            # Retain malformed paths as evidence too.
                            result.forecasts.append(
                                dict(
                                    decision=end.isoformat(),
                                    seed=seed,
                                    path=(
                                        path.copy(deep=True)
                                        if isinstance(path, pd.DataFrame)
                                        else repr(path)
                                    ),
                                )
                            )
                        momentum = int(close > context[-5]["close"]) - int(
                            close < context[-5]["close"]
                        )
                        available = end + pd.Timedelta(
                            seconds=costs.latency_seconds
                        )
                        for arm, request in zip(ARMS, (target, momentum, 0)):
                            order = dict(
                                arm=arm, target=request, available=available
                            )
                            result.pending.append(order)
                            result.decisions.append(
                                dict(
                                    timestamp=end.isoformat(),
                                    contract=contract,
                                    arm=arm,
                                    target=request,
                                    available=available.isoformat(),
                                    observed_close=close,
                                    context_start=frame.index[0].isoformat(),
                                    context_end=end.isoformat(),
                                    error=error if arm == "kronos" else None,
                                )
                            )
                for arm, account in result.accounts.items():
                    marked = replace(account, mark=close)
                    snapshot = marked.snapshot()
                    account.__dict__.update(asdict(marked))
                    result.equity.append(
                        dict(
                            timestamp=end.isoformat(),
                            arm=arm,
                            **snapshot,
                        )
                    )
                result.processed_minutes += 1
        sentinel = object()
        if next(iterator, sentinel) is not sentinel:
            raise ValueError("extra unscheduled or duplicate minute")
        result.status = "COMPLETE"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result
