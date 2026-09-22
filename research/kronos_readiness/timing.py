"""Offline pinned CPU timing on bundled synthetic context; never price
inputs."""

from __future__ import annotations
from pathlib import Path
import time
import statistics
import importlib.metadata
from typing import Callable, Any

import pandas as pd
from research.kronos_replay.fixtures import bundled_fixture
from research.kronos_replay.providers import (
    CachedProvider,
    ForecastProvider,
    ForecastFailure,
)
from tools import kronos_inference_pilot as pilot
from . import FLAGS
from .evidence import finish, new_output, write_json, sha
from .protocol import candidate


def run(
    output: Path,
    cache: Path,
    decisions: int = 3,
    *,
    factory: Callable[[Path], ForecastProvider] = CachedProvider,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if type(decisions) is not int or not 1 <= decisions <= 10:
        raise ValueError("request between one and ten synthetic decisions")
    output = new_output(output)
    bars, sessions = bundled_fixture()
    # Aggregate only fixture minutes. No replay accounting or real prices are
    # used.
    aggregates = []
    for offset in range(0, len(bars), 15):
        end_offset = offset + 15
        chunk = bars[offset:end_offset]
        row = {
            "timestamp": chunk[-1]["timestamp"] + pd.Timedelta(minutes=1),
            "open": chunk[0]["open"],
            "high": max(r["high"] for r in chunk),
            "low": min(r["low"] for r in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(r["volume"] for r in chunk),
        }
        row["amount"] = (
            row["volume"] * sum(row[k] for k in pilot.PRICE_COLUMNS) / 4
        )
        aggregates.append(row)
        session = next(
            s for s in sessions if s.opening <= chunk[0]["timestamp"] < s.close
        )
        if (
            len(aggregates) >= 128
            and row["timestamp"] + pd.Timedelta(minutes=60) <= session.close
        ):
            break
    context = pd.DataFrame(aggregates[-128:]).set_index("timestamp")[
        pilot.VALUE_COLUMNS
    ]
    future = pd.date_range(
        context.index[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min"
    )
    write_json(output / "protocol.json", candidate())
    context.to_csv(output / "synthetic-context.csv", index_label="timestamp")
    (output / "synthetic-context.csv").chmod(0o444)
    started = clock()
    timings, checks, targets = [], [], []
    startup = None
    error: str | None = None
    try:
        provider = factory(cache)
        startup = clock() - started
        for index in range(decisions):
            tick = clock()
            paths: list[Any] = []
            try:
                paths = provider.forecast(context.copy(deep=True), future)
                if len(paths) != 3:
                    raise ValueError("three paths required")
                decision_checks = [
                    pilot.validate_forecast(path, future) for path in paths
                ]
                valid = all(c["invalid_candles"] == 0 for c in decision_checks)
                target = 0
                if valid:
                    terminal = statistics.mean(
                        float(p.close.iloc[-1]) for p in paths
                    )
                    target = int(terminal > context.close.iloc[-1]) - int(
                        terminal < context.close.iloc[-1]
                    )
                timings.append(clock() - tick)
                targets.append(target)
                checks.append(decision_checks)
            except ForecastFailure as exc:
                paths = exc.paths
                raise
            finally:
                for seed, path in enumerate(paths):
                    destination = output / f"decision-{index}-seed-{seed}.csv"
                    if isinstance(path, pd.DataFrame):
                        path.to_csv(destination, index_label="timestamp")
                        destination.chmod(0o444)
                    else:
                        write_json(
                            destination.with_suffix(".json"),
                            {
                                **FLAGS,
                                "invalid_path_type": type(path).__name__,
                            },
                        )
        status = (
            "TIMING_COMPLETED"
            if all(c["invalid_candles"] == 0 for cs in checks for c in cs)
            else "TIMING_INVALID_OUTPUT"
        )
    except Exception as exc:
        error = (
            "inference"
            if isinstance(exc, ForecastFailure)
            else "startup" if startup is None else "invalid-output"
        )
        status = "TIMING_FAILED"
    if status == "TIMING_INVALID_OUTPUT":
        error = "invalid-output"
    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "numpy",
        "pandas",
        "huggingface_hub",
        "safetensors",
        "einops",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return finish(
        output,
        {
            **FLAGS,
            "status": status,
            "scope": "SYNTHETIC_TIMING_ONLY",
            "error_category": error,
            "startup_seconds": startup,
            "versions": versions,
            "targets": targets,
            "context_start": context.index[0].isoformat(),
            "context_end": context.index[-1].isoformat(),
            "future": [t.isoformat() for t in future],
            "context_sha256": sha(
                (output / "synthetic-context.csv").read_bytes()
            ),
            "future_sha256": sha(
                "\n".join(t.isoformat() for t in future).encode()
            ),
            "three_seed_decision_seconds": timings,
            "decision_checks": checks,
            "latency_adopted_seconds": None,
            "cache": str(cache.absolute()),
            "device": "cpu",
            "seeds": list(pilot.SEEDS),
            "interpretation": (
                "Repeated fixed synthetic context, three sequential seeds per"
                " decision; no economic outcomes."
            ),
        },
    )
