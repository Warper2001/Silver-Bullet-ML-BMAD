"""Synthetic runtime checks: no torch, network, credentials or market data needed."""

import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tools import kronos_inference_pilot as pilot


@pytest.fixture
def source(monkeypatch):
    payload = b"# pinned synthetic source\n"
    monkeypatch.setattr(
        pilot,
        "SOURCE_FILES",
        {"model/__init__.py": hashlib.sha256(payload).hexdigest()},
    )
    monkeypatch.setattr(
        pilot.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload)
    )
    return payload


def test_concurrent_source_publication(tmp_path, monkeypatch, source):
    barrier = Barrier(2)

    def download(*args, **kwargs):
        barrier.wait(timeout=5)
        return io.BytesIO(source)

    monkeypatch.setattr(pilot.urllib.request, "urlopen", download)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(pilot.prepare_source, tmp_path, False) for _ in range(2)]
        paths = [future.result(timeout=10) for future in futures]
    assert paths[0] == paths[1]
    assert (paths[0] / "model/__init__.py").read_bytes() == source
    assert not list(tmp_path.rglob("*.part"))
    assert pilot.prepare_source(tmp_path, True) == paths[0]


def test_interrupted_publication_is_retryable(tmp_path, monkeypatch, source):
    original_link = pilot.os.link
    monkeypatch.setattr(pilot.os, "link", Mock(side_effect=OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        pilot.prepare_source(tmp_path, False)
    assert not list(tmp_path.rglob("__init__.py"))
    assert not list(tmp_path.rglob("*.part"))
    monkeypatch.setattr(pilot.os, "link", original_link)
    vendor = pilot.prepare_source(tmp_path, False)
    assert (vendor / "model/__init__.py").read_bytes() == source


def test_bad_download_never_published(tmp_path, monkeypatch, source):
    monkeypatch.setattr(
        pilot.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"bad")
    )
    with pytest.raises(pilot.AuditError, match="downloaded source hash"):
        pilot.prepare_source(tmp_path, False)
    assert not list(tmp_path.rglob("__init__.py"))


def test_failed_flush_never_published(tmp_path, monkeypatch, source):
    monkeypatch.setattr(pilot.os, "fsync", Mock(side_effect=OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        pilot.prepare_source(tmp_path, False)
    assert not list(tmp_path.rglob("__init__.py"))
    assert not list(tmp_path.rglob("*.part"))


def test_interruption_after_link_preserves_complete_source(
    tmp_path, monkeypatch, source
):
    original = pilot.os.link

    def interrupted(staging, target):
        original(staging, target)
        raise KeyboardInterrupt()

    monkeypatch.setattr(pilot.os, "link", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pilot.prepare_source(tmp_path, False)
    assert not list(tmp_path.rglob("*.part"))
    vendor = pilot.prepare_source(tmp_path, True)
    assert (vendor / "model/__init__.py").read_bytes() == source


def test_invalid_concurrent_winner_is_preserved(tmp_path, monkeypatch, source):
    def winner(staging, target):
        target.write_bytes(b"invalid concurrent publication")
        raise FileExistsError()

    monkeypatch.setattr(pilot.os, "link", winner)
    with pytest.raises(pilot.AuditError, match="cached source hash"):
        pilot.prepare_source(tmp_path, False)
    target = tmp_path / "source" / pilot.SOURCE_REVISION / "model/__init__.py"
    assert target.read_bytes() == b"invalid concurrent publication"
    assert not list(tmp_path.rglob("*.part"))


def test_source_directory_alias_refused(tmp_path, source):
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    (tmp_path / "source").symlink_to(redirected, target_is_directory=True)
    with pytest.raises(pilot.AuditError, match="aliases"):
        pilot.prepare_source(tmp_path, False)
    assert not list(redirected.iterdir())


def test_corrupt_cache_preserved_and_alias_refused(tmp_path, monkeypatch, source):
    vendor = pilot.prepare_source(tmp_path, False)
    target = vendor / "model/__init__.py"
    target.write_bytes(b"corrupt")
    download = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(pilot.urllib.request, "urlopen", download)
    with pytest.raises(pilot.AuditError, match="cached source hash"):
        pilot.prepare_source(tmp_path, False)
    assert target.read_bytes() == b"corrupt"
    target.unlink()
    target.symlink_to(tmp_path / "unreadable-target")
    with pytest.raises(pilot.AuditError, match="aliases"):
        pilot.prepare_source(tmp_path, False)
    download.assert_not_called()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # Isolate process-global state touched by the production CLI.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delitem(sys.modules, "model", raising=False)
    state = np.random.get_state()
    numpy_seed = Mock(wraps=np.random.seed)
    monkeypatch.setattr(np.random, "seed", numpy_seed)
    context = pd.DataFrame(
        [[100.0, 102.0, 99.0, 101.0, 10.0, 1005.0]] * pilot.CONTEXT,
        columns=pilot.VALUE_COLUMNS,
        index=pd.date_range(
            end="2025-02-03T14:30:00-05:00", periods=pilot.CONTEXT, freq="15min"
        ),
    )
    monkeypatch.setattr(pilot, "load_context", lambda: context)
    monkeypatch.setattr(pilot, "CACHE", tmp_path / "cache")
    prepare = Mock(return_value=tmp_path / "vendor")
    monkeypatch.setattr(pilot, "prepare_source", prepare)
    paths, hashes = {}, {}
    for name in ("Kronos-small", "Kronos-Tokenizer-base"):
        path = tmp_path / "checkpoints" / name
        path.mkdir(parents=True)
        hashes[name] = {}
        for filename in ("config.json", "model.safetensors"):
            content = f"synthetic {name} {filename}".encode()
            (path / filename).write_bytes(content)
            hashes[name][filename] = hashlib.sha256(content).hexdigest()
        paths[name] = path
    monkeypatch.setattr(pilot, "CHECKPOINT_HASHES", hashes)
    snapshot = Mock(side_effect=lambda **kw: str(paths[kw["repo_id"].split("/")[-1]]))
    torch = SimpleNamespace(
        set_num_threads=Mock(),
        set_num_interop_threads=Mock(),
        manual_seed=Mock(),
        inference_mode=Mock(side_effect=nullcontext),
        get_num_threads=lambda: 1,
    )
    components = [Mock(), Mock()]
    for component in components:
        component.parameters.return_value = [SimpleNamespace(numel=lambda: 7)]

    def forecast(frame, x, y, **kwargs):
        assert frame is context
        pd.testing.assert_series_equal(x, pd.Series(context.index))
        offset = np.random.uniform(0, 0.5)
        return pd.DataFrame(
            [[100.0 + offset, 102.0, 99.0, 101.0, 10.0, 1005.0]] * len(y),
            columns=pilot.VALUE_COLUMNS,
            index=pd.DatetimeIndex(y),
        )

    predictor = SimpleNamespace(predict=Mock(side_effect=forecast))
    upstream = SimpleNamespace(
        Kronos=SimpleNamespace(from_pretrained=Mock(return_value=components[0])),
        KronosTokenizer=SimpleNamespace(
            from_pretrained=Mock(return_value=components[1])
        ),
        KronosPredictor=Mock(return_value=predictor),
    )
    modules = {
        "torch": torch,
        "huggingface_hub": SimpleNamespace(snapshot_download=snapshot),
        "model": upstream,
    }
    real_import = pilot.importlib.import_module
    monkeypatch.setattr(
        pilot.importlib,
        "import_module",
        lambda name, *a, **kw: (
            modules[name] if name in modules else real_import(name, *a, **kw)
        ),
    )
    monkeypatch.setattr(pilot.importlib.metadata, "version", lambda _: "synthetic-test")
    yield SimpleNamespace(
        torch=torch,
        upstream=upstream,
        components=components,
        predictor=predictor,
        forecast=forecast,
        snapshot=snapshot,
        prepare=prepare,
        paths=paths,
        numpy_seed=numpy_seed,
    )
    np.random.set_state(state)


@pytest.mark.parametrize("offline", [False, True])
def test_run_orchestration(tmp_path, runtime, offline):
    output = tmp_path / "report"
    report = pilot.run(output, offline)
    assert report["status"] == "INFERENCE_COMPLETED"
    assert report["same_seed_repeat_equal"]
    assert not report["training_performed"] and not report["trading_authorized"]
    runtime.prepare.assert_called_once_with(tmp_path / "cache", offline)
    for call, revision in zip(
        runtime.snapshot.call_args_list,
        (pilot.MODEL_REVISION, pilot.TOKENIZER_REVISION),
    ):
        assert call.kwargs["revision"] == revision
        assert call.kwargs["local_files_only"] is offline
        assert call.kwargs["token"] is False
        assert call.kwargs["allow_patterns"] == ["config.json", "model.safetensors"]
    runtime.torch.set_num_threads.assert_called_once_with(1)
    runtime.torch.set_num_interop_threads.assert_called_once_with(1)
    assert [c.args[0] for c in runtime.torch.manual_seed.call_args_list] == [0, 0, 1, 2]
    assert [c.args[0] for c in runtime.numpy_seed.call_args_list] == [0, 0, 1, 2]
    assert runtime.torch.inference_mode.call_count == 4
    for component in runtime.components:
        component.eval.assert_called_once_with()
        component.requires_grad_.assert_called_once_with(False)
    runtime.upstream.KronosPredictor.assert_called_once_with(
        *runtime.components, device="cpu", max_context=512
    )
    runtime.upstream.Kronos.from_pretrained.assert_called_once_with(
        str(runtime.paths["Kronos-small"]), strict=True
    )
    runtime.upstream.KronosTokenizer.from_pretrained.assert_called_once_with(
        str(runtime.paths["Kronos-Tokenizer-base"]), strict=True
    )
    for call in runtime.predictor.predict.call_args_list:
        assert call.kwargs == dict(
            pred_len=4, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False
        )
    for name, expected in json.loads((output / "COMPLETE.json").read_text()).items():
        assert pilot.digest(output / name) == expected
    assert len(pd.read_csv(output / "forecasts.csv")) == 12


@pytest.mark.parametrize("bad", ["nan", "geometry"])
def test_invalid_forecast_retained_but_cli_fails(tmp_path, runtime, bad):
    def forecast(*args, **kwargs):
        frame = runtime.forecast(*args, **kwargs)
        frame.iloc[0, 1] = np.nan if bad == "nan" else 1
        return frame

    runtime.predictor.predict.side_effect = forecast
    output = tmp_path / "invalid-report"
    assert pilot.main(["--offline", "--output-dir", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert not report["all_candles_valid"]
    assert report["status"] == "INFERENCE_INVALID_OUTPUT"
    assert (output / "forecasts.csv").exists()
    assert (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("when", ["before", "during"])
def test_checkpoint_tampering_prevents_publication(tmp_path, runtime, when):
    path = runtime.paths["Kronos-small"] / "model.safetensors"
    if when == "before":
        path.write_bytes(b"bad")
    else:

        def forecast(*args, **kwargs):
            path.write_bytes(b"changed during prediction")
            return runtime.forecast(*args, **kwargs)

        runtime.predictor.predict.side_effect = forecast
    output = tmp_path / "refused-report"
    with pytest.raises(pilot.AuditError, match="checkpoint"):
        pilot.run(output, True)
    assert not output.exists()
    if when == "before":
        runtime.upstream.Kronos.from_pretrained.assert_not_called()


def test_run_rejects_cache_root_alias(tmp_path, monkeypatch, runtime):
    target = tmp_path / "redirected-cache"
    target.mkdir()
    alias = tmp_path / "alias-cache"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(pilot, "CACHE", alias)
    with pytest.raises(pilot.AuditError, match="aliases"):
        pilot.run(tmp_path / "report", True)
    runtime.prepare.assert_not_called()
    runtime.snapshot.assert_not_called()


def test_differing_seed_zero_repeat_is_reported(tmp_path, runtime):
    def forecast(*args, **kwargs):
        frame = runtime.forecast(*args, **kwargs)
        if runtime.predictor.predict.call_count == 2:
            frame.iloc[0, 3] += 0.25
        return frame

    runtime.predictor.predict.side_effect = forecast
    assert not pilot.run(tmp_path / "repeat-report", True)["same_seed_repeat_equal"]


def test_invalid_warmup_retained_and_rejected(tmp_path, runtime):
    def forecast(*args, **kwargs):
        frame = runtime.forecast(*args, **kwargs)
        if runtime.predictor.predict.call_count == 1:
            frame.iloc[0, 0] = np.nan
        return frame

    runtime.predictor.predict.side_effect = forecast
    output = tmp_path / "warmup-report"
    assert pilot.main(["--offline", "--output-dir", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["warmup_check"]["invalid_candles"] == 1
    assert all(check["invalid_candles"] == 0 for check in report["output_checks"])
    assert (output / "warmup.csv").exists()
    assert "warmup.csv" in json.loads((output / "COMPLETE.json").read_text())
