"""Pinned pretrained Kronos CPU inference; no training, scoring or trading."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import resource
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.trading_model_readiness import (  # noqa: E402
    AuditError,
    DATA_ROOT,
    digest,
    safe_path,
)

SOURCE_REVISION = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
SOURCE_FILES = {
    "model/__init__.py": "f8f856ca3fedadcaac97e196be23d1aeda1c3c9ffe8903d66d43ea3bcac6240c",
    "model/kronos.py": "0a5f90282e2039c2de0771473419715c845def154896dbd0f5747837e6241032",
    "model/module.py": "a07edbadc0e96804c8158c021bbc6063bb7cc43b34d7fc470d5c8ff2005a409f",
    "LICENSE": "acb2d194d378204e5f2be4dcd24d39ecac437903620c790c3315a96dab388fdc",
}
INPUT = DATA_ROOT / "data/mim_x/mnq_1min_by_contract.csv"
INPUT_HASH = "ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea"
CACHE = DATA_ROOT / ".venv-research/kronos"
CUTOFF = "2025-02-03T19:30:00Z"
CONTRACT = "MNQH25"
CONTEXT = 128
HORIZON = 4
SEEDS = (0, 1, 2)
PRICE_COLUMNS = ["open", "high", "low", "close"]
VALUE_COLUMNS = PRICE_COLUMNS + ["volume", "amount"]
CHECKPOINT_HASHES = {
    "Kronos-small": {
        "config.json": "5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
        "model.safetensors": "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
    },
    "Kronos-Tokenizer-base": {
        "config.json": "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
        "model.safetensors": "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
    },
}


def build_context(frame: pd.DataFrame, cutoff: str, lookback: int) -> pd.DataFrame:
    """Aggregate only pre-cutoff, end-labelled RTH minutes without filling gaps."""
    if lookback <= 0 or lookback > 512:
        raise AuditError("context must be between 1 and 512 bars")
    text = frame["timestamp"].astype(str)
    if not text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$").all():
        raise AuditError("input timestamps must declare a UTC offset")
    stamps = pd.to_datetime(text, utc=True, format="ISO8601")
    end = pd.Timestamp(cutoff)
    if end.tzinfo is None:
        raise AuditError("cutoff must be timezone-aware")
    selected = frame.loc[(frame.contract == CONTRACT) & (stamps <= end)].copy()
    selected.index = pd.DatetimeIndex(stamps.loc[selected.index]).tz_convert(
        "America/New_York"
    )
    selected = selected.sort_index()
    index = pd.DatetimeIndex(selected.index)
    minutes = index.hour * 60 + index.minute
    selected = selected.loc[
        (index.weekday < 5) & (minutes >= 571) & (minutes <= 960)
    ].copy()
    if selected.empty or selected.index.has_duplicates:
        raise AuditError("empty or duplicate context minutes")
    if not (selected.index == pd.DatetimeIndex(selected.index).floor("min")).all():
        raise AuditError("context timestamps must be whole minutes")
    columns = PRICE_COLUMNS + ["volume"]
    selected[columns] = selected[columns].apply(pd.to_numeric, errors="raise")
    values = selected[columns].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values[:, :4] <= 0).any()
        or (values[:, 4] < 0).any()
    ):
        raise AuditError("invalid context OHLCV")
    if (
        (selected.high < selected[PRICE_COLUMNS].max(axis=1))
        | (selected.low > selected[PRICE_COLUMNS].min(axis=1))
    ).any():
        raise AuditError("inconsistent context OHLC")
    resampler = selected.resample("15min", label="right", closed="right")
    bars = resampler.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    bars["minute_count"] = resampler.size()
    bars = bars.loc[bars.minute_count > 0].tail(lookback)
    if len(bars) != lookback or not bars.minute_count.eq(15).all():
        raise AuditError("insufficient context or incomplete 15-minute bucket")
    for _, day_bars in bars.groupby(pd.DatetimeIndex(bars.index).date):
        first, last = day_bars.index[0], day_bars.index[-1]
        if first != bars.index[0] and (first.hour, first.minute) != (9, 45):
            raise AuditError("missing session-opening context bucket")
        if last != bars.index[-1] and (last.hour, last.minute) != (16, 0):
            raise AuditError("missing session-closing context bucket")
        if (
            not day_bars.index.to_series()
            .diff()
            .dropna()
            .eq(pd.Timedelta(minutes=15))
            .all()
        ):
            raise AuditError("missing whole intraday context bucket")
    if bars.index[-1] != end:
        raise AuditError("context must end exactly at the fixed cutoff")
    bars = bars[columns].copy()
    bars["amount"] = bars.volume * bars[PRICE_COLUMNS].mean(axis=1)
    return bars


def load_context() -> pd.DataFrame:
    path = safe_path(INPUT)
    if digest(path) != INPUT_HASH:
        raise AuditError("audited input fingerprint changed")
    chunks = []
    for chunk in pd.read_csv(path, chunksize=100_000):
        # Keep only the named contract before aggregation; no future labels used.
        chosen = chunk.loc[chunk.contract == CONTRACT]
        if not chosen.empty:
            chunks.append(chosen)
    if digest(path) != INPUT_HASH:
        raise AuditError("input changed while loading")
    if not chunks:
        raise AuditError("fixed contract is absent")
    return build_context(pd.concat(chunks, ignore_index=True), CUTOFF, CONTEXT)


def prepare_source(cache: Path, offline: bool) -> Path:
    vendor = safe_path(cache / "source" / SOURCE_REVISION)
    for name, expected in SOURCE_FILES.items():
        target = safe_path(vendor / name)
        if not target.exists():
            if offline:
                raise AuditError("pinned source is not cached for offline inference")
            url = f"https://raw.githubusercontent.com/shiyu-coder/Kronos/{SOURCE_REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise AuditError("downloaded source hash mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
        if digest(target) != expected:
            raise AuditError("cached source hash mismatch")
    return vendor


def validate_forecast(frame: pd.DataFrame, future: pd.DatetimeIndex) -> dict[str, Any]:
    if list(frame.columns) != VALUE_COLUMNS or not frame.index.equals(future):
        raise AuditError("unexpected forecast schema or timestamps")
    values = frame.to_numpy(dtype=float)
    finite = bool(np.isfinite(values).all())
    invalid = (
        (frame.high < frame[PRICE_COLUMNS].max(axis=1))
        | (frame.low > frame[PRICE_COLUMNS].min(axis=1))
        | (frame[PRICE_COLUMNS] <= 0).any(axis=1)
        | (frame[["volume", "amount"]] < 0).any(axis=1)
        | ~np.isfinite(values).all(axis=1)
    )
    return {"all_finite": finite, "invalid_candles": int(invalid.sum())}


def check_destination(output: Path) -> Path:
    output = safe_path(output)
    if output.exists() or any(
        part.lower() in {"data", "logs", "models", ".git", ".venv", ".venv-research"}
        for part in output.parts
    ):
        raise AuditError("use a fresh report directory outside live/data/cache paths")
    return output


def verify_checkpoints(checkpoints: dict[str, Path]) -> None:
    for name, path in checkpoints.items():
        for filename, expected in CHECKPOINT_HASHES[name].items():
            if digest(path / filename) != expected:
                raise AuditError("checkpoint fingerprint changed")


def run(output: Path, offline: bool = False) -> dict[str, Any]:
    output = check_destination(output)
    started = time.perf_counter()
    context = load_context()
    future = pd.date_range(
        context.index[-1] + pd.Timedelta(minutes=15), periods=HORIZON, freq="15min"
    )
    if any(
        t.date() != context.index[-1].date() or t.hour * 60 + t.minute > 960
        for t in future
    ):
        raise AuditError("forecast crosses the fixed RTH session")
    cache = safe_path(CACHE)
    vendor = prepare_source(cache, offline)
    hub = importlib.import_module("huggingface_hub")
    checkpoints = {}
    prepare_started = time.perf_counter()
    for name, revision in (
        ("Kronos-small", MODEL_REVISION),
        ("Kronos-Tokenizer-base", TOKENIZER_REVISION),
    ):
        checkpoints[name] = Path(
            hub.snapshot_download(
                repo_id=f"NeoQuasar/{name}",
                revision=revision,
                token=False,
                cache_dir=str(cache / "huggingface"),
                allow_patterns=["config.json", "model.safetensors"],
                local_files_only=offline,
            )
        )
    prepare_seconds = time.perf_counter() - prepare_started
    verify_checkpoints(checkpoints)
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if "model" in sys.modules:
        raise AuditError(
            "upstream model namespace is already loaded; use a fresh process"
        )
    sys.path.insert(0, str(vendor))
    upstream = importlib.import_module("model")
    load_started = time.perf_counter()
    tokenizer = upstream.KronosTokenizer.from_pretrained(
        str(checkpoints["Kronos-Tokenizer-base"]), strict=True
    )
    model = upstream.Kronos.from_pretrained(
        str(checkpoints["Kronos-small"]), strict=True
    )
    for component in (model, tokenizer):
        component.eval()
        component.requires_grad_(False)
    predictor = upstream.KronosPredictor(
        model, tokenizer, device="cpu", max_context=512
    )
    load_seconds = time.perf_counter() - load_started

    def predict(seed: int) -> pd.DataFrame:
        torch.manual_seed(seed)
        np.random.seed(seed)
        with torch.inference_mode():
            result = predictor.predict(
                context,
                pd.Series(context.index),
                pd.Series(future),
                pred_len=HORIZON,
                T=1.0,
                top_k=0,
                top_p=0.9,
                sample_count=1,
                verbose=False,
            )
            if not isinstance(result, pd.DataFrame):
                raise AuditError("upstream returned a non-DataFrame forecast")
            return result

    warmup_started = time.perf_counter()
    warmup = predict(SEEDS[0])
    warmup_seconds = time.perf_counter() - warmup_started
    timings, frames, checks = [], [], []
    for seed in SEEDS:
        tick = time.perf_counter()
        forecast = predict(seed)
        timings.append(time.perf_counter() - tick)
        checks.append(validate_forecast(forecast, future))
        saved = forecast.copy()
        saved.insert(0, "seed", seed)
        frames.append(saved)
    predictions = pd.concat(frames)
    finite = all(check["all_finite"] for check in checks)
    quality = all(check["invalid_candles"] == 0 for check in checks)
    verify_checkpoints(checkpoints)
    report = {
        "status": "INFERENCE_COMPLETED" if finite else "INFERENCE_INVALID_OUTPUT",
        "economic_status": "UNTESTED_RESEARCH_CANDIDATE",
        "trading_authorized": False,
        "training_performed": False,
        "script_sha256": digest(Path(__file__)),
        "source_revision": SOURCE_REVISION,
        "source_hashes": SOURCE_FILES,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "input_path": str(INPUT),
        "input_sha256": INPUT_HASH,
        "contract": CONTRACT,
        "cutoff_utc": CUTOFF,
        "context_bars": CONTEXT,
        "context_start": context.index[0].isoformat(),
        "context_end": context.index[-1].isoformat(),
        "forecast_bars": HORIZON,
        "frequency": "15min",
        "seeds": list(SEEDS),
        "sampling": {"temperature": 1.0, "top_p": 0.9, "top_k": 0, "sample_count": 1},
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_parameters": sum(p.numel() for p in tokenizer.parameters()),
        "checkpoint_prepare_seconds": prepare_seconds,
        "model_load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": timings,
        "median_inference_seconds": statistics.median(timings),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "elapsed_seconds": time.perf_counter() - started,
        "output_checks": checks,
        "all_candles_valid": quality,
        "same_seed_repeat_equal": bool(
            np.array_equal(warmup.to_numpy(), frames[0][VALUE_COLUMNS].to_numpy())
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "numpy",
                "pandas",
                "huggingface_hub",
                "safetensors",
                "einops",
            )
        },
        "checkpoint_hashes": {
            name: {
                filename: digest(path / filename)
                for filename in ("config.json", "model.safetensors")
            }
            for name, path in checkpoints.items()
        },
        "qualifications": [
            "Assumed end-labelled development bars and NY calendar features.",
            "Synthetic amount proxy is not authenticated turnover.",
            "No actual future prices, returns, forecast scores or strategy decisions.",
            "Seed variation is not calibrated uncertainty.",
            "Model pretraining exposure is not audited; this is not OOS evidence.",
        ],
    }
    # Non-finite forecasts are retained as CSV evidence; JSON never emits NaN.
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    context.to_csv(output / "context.csv", index_label="timestamp")
    predictions.to_csv(output / "forecasts.csv", index_label="timestamp")
    (output / "report.json").write_text(encoded)
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {
                name: digest(output / name)
                for name in ("context.csv", "forecasts.csv", "report.json")
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "median_inference_seconds",
                    "process_peak_rss_mib",
                    "all_candles_valid",
                    "economic_status",
                )
            }
        ),
        flush=True,
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--offline", action="store_true", help="Use only already cached pinned assets"
    )
    args = parser.parse_args(argv)
    report = run(args.output_dir, args.offline)
    return (
        0
        if report["status"] == "INFERENCE_COMPLETED" and report["all_candles_valid"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
