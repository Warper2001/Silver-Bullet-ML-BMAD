"""Mock offline adapter integration; no network or real model is needed."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pandas as pd
import pytest

from research.kronos_replay.providers import (
    CachedProvider,
    ForecastFailure,
    StubProvider,
)
from tools import kronos_inference_pilot as pilot


def test_cached_load_once_settings_seeds_partial_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, "model", raising=False)
    prepare = Mock(return_value=tmp_path / "vendor")
    verify = Mock()
    monkeypatch.setattr(pilot, "prepare_source", prepare)
    monkeypatch.setattr(pilot, "verify_checkpoints", verify)
    snapshot = Mock(return_value=str(tmp_path / "checkpoint"))
    torch = SimpleNamespace(
        set_num_threads=Mock(),
        set_num_interop_threads=Mock(),
        manual_seed=Mock(),
        inference_mode=nullcontext,
    )
    component = Mock()

    def forecast(context, x, y, **kwargs):
        return StubProvider().forecast(context, pd.DatetimeIndex(y))[0]

    predictor = Mock()
    predictor.predict.side_effect = forecast
    upstream = SimpleNamespace(
        Kronos=SimpleNamespace(from_pretrained=Mock(return_value=component)),
        KronosTokenizer=SimpleNamespace(
            from_pretrained=Mock(return_value=component)
        ),
        KronosPredictor=Mock(return_value=predictor),
    )
    modules = {
        "torch": torch,
        "model": upstream,
        "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
    }
    monkeypatch.setattr(
        "research.kronos_replay.providers.importlib.import_module",
        lambda name: modules[name],
    )
    numpy_seed = Mock(wraps=np.random.seed)
    monkeypatch.setattr(np.random, "seed", numpy_seed)
    provider = CachedProvider(tmp_path)
    context = pd.DataFrame(
        [[100, 101, 99, 100, 10, 1000]] * 128,
        columns=pilot.VALUE_COLUMNS,
        index=pd.date_range(
            end="2026-03-09T10:00:00-04:00", periods=128, freq="15min"
        ),
    )
    future = pd.date_range(
        "2026-03-09T10:15:00-04:00", periods=4, freq="15min"
    )
    first, second = provider.forecast(context, future), provider.forecast(
        context, future
    )
    assert len(first) == 3
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)
    prepare.assert_called_once_with(tmp_path, offline=True)
    upstream.Kronos.from_pretrained.assert_called_once()
    upstream.KronosTokenizer.from_pretrained.assert_called_once()
    assert verify.call_count == 5
    assert [c.args[0] for c in torch.manual_seed.call_args_list] == [
        0,
        1,
        2,
    ] * 2
    assert [c.args[0] for c in numpy_seed.call_args_list] == [0, 1, 2] * 2
    for call, revision in zip(
        snapshot.call_args_list,
        (pilot.MODEL_REVISION, pilot.TOKENIZER_REVISION),
    ):
        assert call.kwargs["revision"] == revision
        assert (
            call.kwargs["local_files_only"] is True
            and call.kwargs["token"] is False
        )
    for call in predictor.predict.call_args_list:
        assert call.kwargs == dict(
            pred_len=4,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )
    predictor.predict.side_effect = [
        first[0],
        RuntimeError("failure on second seed"),
    ]
    with pytest.raises(ForecastFailure) as caught:
        provider.forecast(context, future)
    assert len(caught.value.paths) == 1
