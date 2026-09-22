"""Synthetic-only replay CLI; accepts no market-data path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import importlib.metadata
import platform
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from tools import kronos_inference_pilot as pilot
from tools import trading_model_readiness as readiness
from .engine import Account, Costs, ReplayResult, SCOPE, run_replay
from .fixtures import bundled_fixture
from .providers import CachedProvider, StubProvider

REGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "_bmad-output/preregistration_kronos_synthetic_mechanics_20260922.md"
)


TABLE_SCHEMAS = {
    "decisions": [
        "timestamp",
        "contract",
        "arm",
        "target",
        "available",
        "observed_close",
        "context_start",
        "context_end",
        "error",
    ],
    "fills": [
        "timestamp",
        "arm",
        "contract",
        "side",
        "action",
        "price",
        "reference_open",
        "fee",
        "reason",
    ],
    "equity": ["timestamp", "arm", *Account().snapshot().keys()],
    "aggregated": ["timestamp", "contract", *pilot.VALUE_COLUMNS],
}


def runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in (
        "numpy",
        "pandas",
        "torch",
        "huggingface_hub",
        "safetensors",
        "einops",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def encode(value: Any) -> str:
    return (
        json.dumps(
            value, default=str, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n"
    )


def run(
    output: Path, provider_name: str = "stub", costs: Costs = Costs()
) -> ReplayResult:
    destination = pilot.check_destination(output)
    destination.mkdir(parents=True, exist_ok=False)
    scope = dict(
        scope=SCOPE,
        economic_evaluation=False,
        trading_authorized=False,
        strategy_test_permitted=False,
    )
    result = ReplayResult()
    try:
        if provider_name not in {"stub", "cached"}:
            raise ValueError("provider must be stub or cached")
        rows, sessions = bundled_fixture()
        protocol = dict(
            scope,
            minute_labels="interval_open",
            context_bars=128,
            horizon_bars=4,
            provider=provider_name,
            source_revision=pilot.SOURCE_REVISION,
            source_hashes=pilot.SOURCE_FILES,
            model_revision=pilot.MODEL_REVISION,
            tokenizer_revision=pilot.TOKENIZER_REVISION,
            checkpoint_hashes=pilot.CHECKPOINT_HASHES,
            seeds=list(pilot.SEEDS),
            sampling=dict(temperature=1, top_p=0.9, top_k=0, sample_count=1),
            costs=asdict(costs),
            point_value=2,
            tick_size=0.25,
            preregistration_sha256=pilot.digest(REGISTRATION),
            code_hashes={
                str(p.relative_to(REGISTRATION.parent.parent)): pilot.digest(p)
                for p in [
                    *Path(__file__).parent.glob("*.py"),
                    Path(pilot.__file__),
                    Path(readiness.__file__),
                ]
            },
            runtime_versions=runtime_versions(),
            sessions=[asdict(s) for s in sessions],
        )
        (destination / "protocol.json").write_text(encode(protocol))
        pd.DataFrame(rows).to_csv(
            destination / "synthetic_fixture.csv", index=False
        )
        provider = (
            CachedProvider() if provider_name == "cached" else StubProvider()
        )
        result = run_replay(rows, sessions, provider, costs)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    # Retain evidence for incomplete runs and initialization failures.
    for name, columns in TABLE_SCHEMAS.items():
        pd.DataFrame(getattr(result, name), columns=columns).to_csv(
            destination / f"{name}.csv", index=False
        )
    evidence = []
    for number, forecast in enumerate(result.forecasts):
        path = forecast["path"]
        filename = (
            f"forecast-{number:04d}.csv"
            if isinstance(path, pd.DataFrame)
            else f"forecast-{number:04d}.txt"
        )
        if isinstance(path, pd.DataFrame):
            path.to_csv(destination / filename, index_label="timestamp")
        else:
            (destination / filename).write_text(str(path))
        evidence.append(
            dict(
                decision=forecast["decision"],
                seed=forecast["seed"],
                artifact=filename,
            )
        )
    (destination / "forecasts.json").write_text(
        encode(dict(scope, paths=evidence))
    )
    (destination / "report.json").write_text(encode(result.report()))
    manifest = dict(
        scope,
        status=result.status,
        artifacts={
            p.name: pilot.digest(p)
            for p in destination.iterdir()
            if p.is_file()
        },
    )
    (destination / "manifest.json").write_text(encode(manifest))
    marker = (
        "COMPLETE.json" if result.status == "COMPLETE" else "INCOMPLETE.json"
    )
    (destination / marker).write_text(
        encode(
            dict(
                scope,
                status=result.status,
                manifest_sha256=pilot.digest(destination / "manifest.json"),
            )
        )
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--provider", choices=("stub", "cached"), default="stub"
    )
    parser.add_argument("--latency-seconds", type=float, default=0)
    parser.add_argument("--fee-per-side", type=float, default=0.5)
    parser.add_argument("--slippage-ticks", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        result = run(
            args.output_dir,
            args.provider,
            Costs(
                args.latency_seconds, args.fee_per_side, args.slippage_ticks
            ),
        )
    except (ValueError, OSError) as exc:
        print(f"Replay refused: {exc}")
        return 1
    print(encode(result.report()))
    return 0 if result.status == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
