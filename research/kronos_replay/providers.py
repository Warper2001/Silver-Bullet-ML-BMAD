"""Forecast adapters using the unchanged pilot's pinned offline assets."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from tools import kronos_inference_pilot as pilot


class ForecastProvider(Protocol):
    def forecast(
        self, context: pd.DataFrame, future: pd.DatetimeIndex
    ) -> list[pd.DataFrame]: ...


class StubProvider:
    """Three deterministic paths; not a model or an economic hypothesis."""

    def forecast(
        self, context: pd.DataFrame, future: pd.DatetimeIndex
    ) -> list[pd.DataFrame]:
        result = []
        for seed in pilot.SEEDS:
            close = float(context.close.iloc[-1])
            rows = []
            for step in range(len(future)):
                opening = close
                close = opening + (seed + 1) * 0.25
                rows.append(
                    [
                        opening,
                        close + 0.25,
                        opening - 0.25,
                        close,
                        10,
                        close * 10,
                    ]
                )
            result.append(
                pd.DataFrame(rows, columns=pilot.VALUE_COLUMNS, index=future)
            )
        return result


class CachedProvider:
    """Load pinned CPU assets once, with no network or credentials."""

    def __init__(self, cache: Path = pilot.CACHE) -> None:
        cache = cache.absolute()
        if pilot.safe_path(cache) != cache:
            raise pilot.AuditError("source cache aliases are prohibited")
        vendor = pilot.prepare_source(cache, offline=True)
        hub = importlib.import_module("huggingface_hub")
        self.checkpoints: dict[str, Path] = {}
        for name, revision in (
            ("Kronos-small", pilot.MODEL_REVISION),
            ("Kronos-Tokenizer-base", pilot.TOKENIZER_REVISION),
        ):
            self.checkpoints[name] = Path(
                hub.snapshot_download(
                    repo_id=f"NeoQuasar/{name}",
                    revision=revision,
                    token=False,
                    cache_dir=str(cache / "huggingface"),
                    allow_patterns=["config.json", "model.safetensors"],
                    local_files_only=True,
                )
            )
        pilot.verify_checkpoints(self.checkpoints)
        self.torch: Any = importlib.import_module("torch")
        self.torch.set_num_threads(1)
        self.torch.set_num_interop_threads(1)
        if "model" in sys.modules:
            raise pilot.AuditError(
                "upstream model namespace already loaded; use a fresh process"
            )
        sys.path.insert(0, str(vendor))
        upstream = importlib.import_module("model")
        tokenizer = upstream.KronosTokenizer.from_pretrained(
            str(self.checkpoints["Kronos-Tokenizer-base"]), strict=True
        )
        model = upstream.Kronos.from_pretrained(
            str(self.checkpoints["Kronos-small"]), strict=True
        )
        for component in (model, tokenizer):
            component.eval()
            component.requires_grad_(False)
        self.predictor = upstream.KronosPredictor(
            model, tokenizer, device="cpu", max_context=512
        )

    def forecast(
        self, context: pd.DataFrame, future: pd.DatetimeIndex
    ) -> list[pd.DataFrame]:
        pilot.verify_checkpoints(self.checkpoints)
        paths = []
        try:
            for seed in pilot.SEEDS:
                self.torch.manual_seed(seed)
                np.random.seed(seed)
                with self.torch.inference_mode():
                    frame = self.predictor.predict(
                        context.copy(deep=True),
                        pd.Series(context.index),
                        pd.Series(future),
                        pred_len=pilot.HORIZON,
                        T=1.0,
                        top_k=0,
                        top_p=0.9,
                        sample_count=1,
                        verbose=False,
                    )
                paths.append(frame)
            pilot.verify_checkpoints(self.checkpoints)
        except Exception as exc:
            raise ForecastFailure(str(exc), paths) from exc
        return paths


class ForecastFailure(RuntimeError):
    """Inference failed after zero or more paths; retain partial evidence."""

    def __init__(self, message: str, paths: list[pd.DataFrame]) -> None:
        super().__init__(message)
        self.paths = paths
