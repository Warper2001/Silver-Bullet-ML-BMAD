"""Full guarded startup evidence on a shared host; no production certification."""
import asyncio
import base64
import builtins
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from test_evidence import prepared_runtime, startup, e, account
from test_observation_benchmark import startup_bars, ROOT, SNAPSHOT


def code_pins():
    paths = [Path(__file__), Path(__file__).with_name('test_evidence.py'),
             Path(__file__).with_name('test_observation_benchmark.py')]
    paths += [ROOT / 'src/research/yank_deployed_validation' / name
              for name in ('startup.py', 'evidence.py', 'account.py', 'capture.py', 'adapter.py')]
    paths.append(ROOT / 'src/cli/check_yank_deployed_replay.py')
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def with_shadows(setup, bars):
    runtime, trader, *_ = setup
    async def fetch(*args, **kwargs):
        return bars[-15:]
    runtime.module.fetch_px_ts_shaped = fetch
    trader._data_shadow = True
    trader._px_auth = object()
    trader._px_data_contract_id = 'OFFLINE_SYNTHETIC'
    return setup


def test_full_verified_startup_2880_bars(tmp_path, monkeypatch):
    pins = code_pins()
    bars = startup_bars(2880, datetime(2025, 5, 19, 13, 59, tzinfo=timezone.utc))
    baseline = with_shadows(prepared_runtime(tmp_path / 'baseline', bars=bars), bars)
    observed = with_shadows(prepared_runtime(tmp_path / 'observed', bars=bars), bars)
    # The release expectation comes from the separate, uninstrumented reference
    # instance, before its poll; this remains a synthetic release, not attestation.
    expected = dict(baseline[3], runtime=startup.runtime_identity(baseline[1]))
    assert startup.runtime_identity(observed[1]) == expected['runtime']
    assert e.sha(observed[2].checkpoint()) == expected['checkpoint_sha256']
    limits = dict(capacity=16, max_bytes=64_000_000, max_nodes=4_000_000)
    metrics = {}; states = {}; decisions = {}; shadow_counts = {}
    io_events = []; phase = 'prepare'; main = threading.get_ident()
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_os_open = os.open

    def record(operation, path):
        event = dict(phase=phase, operation=operation, path=str(path),
                     thread=threading.current_thread().name)
        io_events.append(event)
        if phase == 'poll' and threading.get_ident() == main:
            raise AssertionError('filesystem access on guarded poll thread: ' + str(event))

    class RecordingFile:
        def __init__(self, inner): self.inner = inner
        def __enter__(self): self.inner.__enter__(); return self
        def __exit__(self, *args): return self.inner.__exit__(*args)
        def __getattr__(self, name): return getattr(self.inner, name)
        def write(self, value):
            record('capture_write', 'capture.jsonl')
            return self.inner.write(value)

    def path_open(path, mode='r', *args, **kwargs):
        record('Path.open', path)
        inner = original_path_open(path, mode, *args, **kwargs)
        return RecordingFile(inner) if path.name == 'capture.jsonl' and mode == 'x' else inner

    def builtin_open(path, *args, **kwargs):
        record('builtins.open', path)
        return original_builtin_open(path, *args, **kwargs)

    def os_open(path, *args, **kwargs):
        record('os.open', path)
        return original_os_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', path_open)
    monkeypatch.setattr(builtins, 'open', builtin_open)
    monkeypatch.setattr(os, 'open', os_open)
    session = None
    for mode, setup in (('baseline', baseline), ('guarded', observed)):
        runtime, trader, view, _, key = setup
        original_poll = trader._poll_and_process
        start = time.perf_counter()
        if mode == 'guarded':
            session = startup.prepare_observation(trader, enabled=True, expected_release=expected,
                output_dir=tmp_path / 'capture', signer=key, key_id='benchmark',
                pseudonym_salt=b'test', capture_limits=limits)
            assert isinstance(session, startup.ObservationSession), session
        metrics[mode] = dict(prepare_seconds=time.perf_counter() - start)
        counts = {name: 0 for name in ('_update_shadow_bullish_m15_choch',
                  '_advance_shadow_trade', '_detect_shadow_bullish_entry')}
        decision_rows = []
        # Identical passive profiler on both independent instances records actual
        # strategy filter decisions without replacing any runtime bindings.
        codes = {getattr(trader, name).__func__.__code__: name for name in counts}
        decision_code = type(trader)._log_filter_decision.__code__
        def profile(frame, event, arg):
            if event == 'call':
                if frame.f_code in codes: counts[codes[frame.f_code]] += 1
                if frame.f_code is decision_code:
                    decision_rows.append({k: v.isoformat() if isinstance(v, datetime) else v
                                          for k, v in frame.f_locals.items() if k != 'self'})
        phase = 'poll'; start = time.perf_counter()
        previous_profile = sys.getprofile()
        try:
            sys.setprofile(profile)
            asyncio.run(trader._poll_and_process())
        finally:
            sys.setprofile(previous_profile)
            metrics[mode]['poll_seconds'] = time.perf_counter() - start
            phase = 'close'
            close_start = time.perf_counter()
            if session is not None:
                coverage = session.close()
            metrics[mode]['close_seconds'] = time.perf_counter() - close_start
        assert trader._poll_and_process == original_poll
        states[mode] = view.state(); decisions[mode] = decision_rows
        shadow_counts[mode] = counts
        assert counts == {name: 2880 for name in counts}
        assert len(trader._shadow_logger.rows) == 1
        assert trader._shadow_logger is not trader._shadow_trade_logger
        assert not trader._shadow_trade_logger.rows
        assert not runtime.logger.errors
        metrics[mode]['process_high_water_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        phase = 'prepare'
    assert states['baseline'] == states['guarded']
    assert states['guarded']['bar_count'] == 2880
    assert decisions['baseline'] == decisions['guarded']
    assert decisions['baseline'], 'Equality must include actual filter decisions'
    assert coverage['valid_coverage'], coverage
    assert coverage['accepted'] == coverage['written'] == 1
    phase = 'verification'; start = time.perf_counter()
    root = tmp_path / 'capture'
    manifest = json.loads((root / 'manifest.json').read_text())
    bundle = json.loads((root / 'evidence.json').read_text())
    rows = [json.loads(line) for line in (root / 'capture.jsonl').read_text().splitlines()]
    assert manifest['publication'] == 'COMPLETE'
    for name in ('capture', 'coverage', 'evidence'):
        suffix = 'jsonl' if name == 'capture' else 'json'
        assert manifest[name + '_sha256'] == hashlib.sha256((root / (name + '.' + suffix)).read_bytes()).hexdigest()
    keys = {'benchmark': base64.b64encode(observed[4].public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}
    verdict = e.verify_bundle(bundle, keys, dict(expected, process=session.process), rows,
                              coverage, manifest['initial_state'], SNAPSHOT)
    assert {k: verdict[k] for k in ('signature', 'identity', 'checkpoint', 'coverage')} == {
        k: 'PASS' for k in ('signature', 'identity', 'checkpoint', 'coverage')}
    assert verdict['decision'] == 'HOLD_VALIDATION'
    trace = rows[0]['trace']
    assert trace['input']['bars'] == bars
    assert len([d for d in trace['decisions'] if d.get('kind') == 'decision_transition']) == 2880
    assert trace['after'] == states['baseline']
    capture_bytes = (root / 'capture.jsonl').stat().st_size
    assert capture_bytes <= limits['max_bytes']
    assert session.capture.capacity == limits['capacity']
    assert not session.capture.queue and not session.capture.worker.is_alive()
    write_threads = sorted({r['thread'] for r in io_events if r['operation'] == 'capture_write'})
    assert write_threads == ['yank-decision-capture']
    # Signed evidence cannot turn a storage/drop failure into valid coverage.
    failed_coverage = dict(coverage, valid_coverage=False, invalid_reasons=['writer_failure'])
    failed_payload = dict(bundle['payload'], coverage_sha256=e.sha(failed_coverage))
    failed = e.verify_bundle(e.sign_bundle(failed_payload, 'benchmark', observed[4]), keys,
        dict(expected, process=session.process), rows, failed_coverage, manifest['initial_state'], SNAPSHOT)
    assert failed['signature'] == 'PASS' and failed['coverage'] == 'FAIL' and not failed['eligible']
    verify_seconds = time.perf_counter() - start
    assert pins == code_pins()
    report = dict(scope='PRIVATE_PINNED_OFFLINE_FULL_VERIFIED_STARTUP', bar_count=2880,
        code_sha256=pins, measurements=metrics, verification_seconds=verify_seconds,
        poll_overhead_seconds=metrics['guarded']['poll_seconds'] - metrics['baseline']['poll_seconds'],
        poll_ratio=metrics['guarded']['poll_seconds'] / metrics['baseline']['poll_seconds'],
        state_equal=True, decision_equal=True, filter_decision_count=len(decisions['baseline']),
        shadow_method_calls=shadow_counts, coverage=coverage, admission=verdict,
        failure_control=failed, limits=limits, capture_bytes=capture_bytes,
        queue_retained_byte_upper_bound=limits['capacity'] * limits['max_bytes'],
        storage_write_threads=write_threads, filesystem_events=io_events,
        decision='HOLD_VALIDATION', limitations=[
            'One run per mode on a shared host with identical passive profiling; descriptive timings only.',
            'RSS is cumulative process high-water, not incremental collector memory.',
            'Configured queue envelope bound is not continuous peak-memory instrumentation.',
            'Path.open, builtins.open and os.open are monitored; this is not a kernel syscall audit.',
            'Constant synthetic provider-shaped backfill exercises both shadows but opens no trade.',
            'Feed, account identity, release reference, signer and shadow storage are private synthetic fixtures.',
            'No authenticated requests, production certification or independently observed account evidence.',
            'Large explicit offline bounds do not qualify default collector settings.'])
    (tmp_path / 'startup-benchmark.json').write_text(json.dumps(report, sort_keys=True))
    print('FULL_STARTUP_BENCHMARK=' + json.dumps(report, sort_keys=True), flush=True)
