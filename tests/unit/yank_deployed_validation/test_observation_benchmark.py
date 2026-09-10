"""Offline poll-path evidence, deliberately not deployment latency certification.

Both existing shadow method paths run. Their storage and second-feed transport
are private in-memory/synthetic substitutes; no live identity is established.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
import hashlib
from pathlib import Path
import resource
import threading
import time

from src.research.yank_deployed_validation.adapter import Adapter, PrivateSnapshot, canonical
from src.research.yank_deployed_validation.capture import DecisionCapture, identities, install_poll_observer

ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT = ROOT / 'docs/yank-validation/snapshot/v1'
STATE = dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT', daily_pnl=0., daily_halted=False,
             last_trading_date=None, on_combine=True, is_backfill=True)


def benchmark_code_hashes():
    paths = [Path(__file__), ROOT/'src/research/yank_deployed_validation/capture.py',
             ROOT/'src/research/yank_deployed_validation/adapter.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def startup_bars(count, end):
    """Synthetic full documented bar shape; never observed provider data."""
    rows = []
    for i in range(count):
        label = end-timedelta(minutes=count-1-i)
        rows.append(dict(TimeStamp=label.isoformat(), Epoch=int(label.timestamp()*1000),
            Open='100', High='101', Low='99', Close='100', TotalVolume='100',
            BarStatus='Closed', IsEndOfHistory=i == count-1, IsRealtime=False,
            OpenInterest='0', DownTicks=5, DownVolume=50, TotalTicks=10,
            UnchangedTicks=0, UnchangedVolume=0, UpTicks=5, UpVolume=50))
    return rows


def runtime_for(bars):
    runtime = PrivateSnapshot(SNAPSHOT)
    runtime.clock = datetime(2025, 5, 19, 14, tzinfo=timezone.utc)
    trader = runtime.construct(STATE)
    reader = object.__new__(Adapter)
    reader.trader = trader
    reader._buffer_cache_key = None
    reader._buffer_hash = None
    reader._warmup = {}
    counts = dict(parity_fetch=0, bullish_update=0, bullish_advance=0, bullish_detect=0)
    async def fetch(*args, **kwargs):
        counts['parity_fetch'] += 1
        return bars[-15:]
    runtime.module.fetch_px_ts_shaped = fetch
    trader._data_shadow = True
    trader._px_auth = object()
    trader._px_data_contract_id = 'OFFLINE_SYNTHETIC'
    for name, key in (('_update_shadow_bullish_m15_choch', 'bullish_update'),
                      ('_advance_shadow_trade', 'bullish_advance'),
                      ('_detect_shadow_bullish_entry', 'bullish_detect')):
        original = getattr(trader, name)
        def call(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)
        setattr(trader, name, call)
    class Auth:
        async def authenticate(self):
            return 'OFFLINE'
    class Response:
        status_code = 200
        content = canonical({'Bars': bars}).encode()
        def json(self):
            return {'Bars': bars}
    class Client:
        async def get(self, *args, **kwargs):
            return Response()
    trader.auth = Auth()
    trader.client = Client()
    return runtime, trader, reader, counts


def test_startup_7500_bar_original_poller_both_shadow_paths(tmp_path, monkeypatch):
    code_hashes = benchmark_code_hashes()
    end = datetime(2025, 5, 19, 13, 59, tzinfo=timezone.utc)
    bars = startup_bars(7500, end)
    measurements = {}
    states = {}
    decisions = {}
    counts_by_mode = {}
    coverage = None
    queue_peak = 0
    queue_byte_peak = 0
    # Observe actual capture.jsonl opens/writes, including injected storage failure
    # in the separate test below. Existing shadow sinks remain private MemoryLog.
    original_open = Path.open
    writes = []
    class RecordingFile:
        def __init__(self, inner): self.inner = inner
        def __enter__(self): self.inner.__enter__(); return self
        def __exit__(self, *args): return self.inner.__exit__(*args)
        def __getattr__(self, name): return getattr(self.inner, name)
        def write(self, value):
            writes.append(threading.current_thread().name)
            return self.inner.write(value)
    def opened(path, mode='r', *args, **kwargs):
        inner = original_open(path, mode, *args, **kwargs)
        return RecordingFile(inner) if path.name == 'capture.jsonl' and mode == 'x' else inner
    monkeypatch.setattr(Path, 'open', opened)
    for mode in ('baseline', 'disabled', 'enabled'):
        print('STARTUP_BENCHMARK_MODE=' + mode, flush=True)
        runtime, trader, reader, counts = runtime_for(bars)
        capture = DecisionCapture(enabled=mode == 'enabled', output_dir=tmp_path / mode,
                                  capacity=2)
        rollback = install_poll_observer(trader, capture, identity=identities(SNAPSHOT),
                    readiness={'equivalent_feed_and_state': False}, state_reader=reader.state)
        start = time.perf_counter()
        asyncio.run(trader._poll_and_process())
        measurements[mode] = {'poll_seconds': time.perf_counter()-start,
                             'process_high_water_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        rollback()
        with capture.lock:
            queue_peak = max(queue_peak, len(capture.queue))
            queue_byte_peak = max(queue_byte_peak, sum(len(row.encode()) for row in capture.queue))
        if mode == 'enabled':
            # Establish writer-thread evidence even when the startup record cannot
            # fit default bounds; this sentinel is explicitly not a poll record.
            assert capture.emit({'kind': 'benchmark_writer_probe'})
            coverage = capture.close()
        else:
            assert capture.worker is None
            assert not (tmp_path / mode).exists()
        states[mode] = reader.state()
        decisions[mode] = runtime.decisions
        counts_by_mode[mode] = counts
        assert not runtime.logger.errors
        assert counts == dict(parity_fetch=1, bullish_update=7500,
                              bullish_advance=7500, bullish_detect=7500)
        assert len(trader._shadow_logger.rows) == 1
        assert trader._shadow_logger is not trader._shadow_trade_logger
        assert trader._ts_client.intentions == []
    assert states['baseline'] == states['disabled'] == states['enabled']
    assert decisions['baseline'] == decisions['disabled'] == decisions['enabled']
    assert states['enabled']['bar_count'] == 7500
    assert queue_peak <= 2 and queue_byte_peak <= 4_000_000
    assert writes and set(writes) == {'yank-decision-capture'}
    # Current defaults reject the startup envelope: recording that limitation is
    # required evidence, not a reason to claim successful complete coverage.
    assert not coverage['valid_coverage']
    assert coverage['invalid_reasons']
    assert code_hashes == benchmark_code_hashes()
    report = dict(scope='PRIVATE_PINNED_OFFLINE_RUNTIME', bar_count=7500, code_sha256=code_hashes,
                  measurements=measurements, state_equal=True, decision_equal=True,
                  shadow_method_calls=counts_by_mode, coverage=coverage,
                  queue_capacity=2, max_record_bytes=2_000_000,
                  sampled_queue_records=queue_peak, sampled_queue_bytes=queue_byte_peak,
                  storage_write_threads=sorted(set(writes)), decision='HOLD_VALIDATION',
                  limitations=['Direct poll observer benchmark; verified-startup guards have separate integration tests.',
                      'Default capture bounds do not preserve complete startup poll evidence.',
                      'RSS is process high-water including baseline and libraries; not observer-only memory.',
                      'Queue snapshots are samples, not continuous peak instrumentation.',
                      'Synthetic constant-price backfill invokes bullish watcher but opens no shadow trade.',
                      'ProjectX transport and both shadow storage sinks are synthetic/in-memory.',
                      'One run per mode on a shared host; timings are descriptive, not isolated production latency certification.'])
    (tmp_path / 'observation-benchmark.json').write_text(canonical(report))
    print('OBSERVATION_BENCHMARK=' + json.dumps(report, sort_keys=True))


def test_actual_capture_writer_failure_invalidates_coverage(tmp_path, monkeypatch):
    original_open = Path.open
    attempted = threading.Event()
    threads = []
    def broken_open(path, mode='r', *args, **kwargs):
        if path.name == 'capture.jsonl' and mode == 'x':
            threads.append(threading.current_thread().name)
            attempted.set()
            raise OSError('synthetic storage failure')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', broken_open)
    capture = DecisionCapture(enabled=True, output_dir=tmp_path / 'failed')
    assert attempted.wait(2)
    assert capture.emit({'kind': 'synthetic_poll'})
    summary = capture.close()
    assert threads == ['yank-decision-capture']
    assert not summary['valid_coverage']
    assert {'writer_failure', 'unwritten_records'} <= set(summary['invalid_reasons'])


def test_queue_memory_stays_bounded_under_slow_storage(tmp_path, monkeypatch):
    import tracemalloc
    release = threading.Event()
    entered = threading.Event()
    original = DecisionCapture._drain
    def delayed(self):
        entered.set()
        if release.wait(5):
            original(self)
    monkeypatch.setattr(DecisionCapture, '_drain', delayed)
    capture = DecisionCapture(enabled=True, output_dir=tmp_path / 'slow', capacity=2,
                              max_bytes=250_000, max_nodes=100)
    assert entered.wait(2)
    payload = {'kind': 'synthetic_bounded_record', 'body': 'x' * 200_000}
    tracemalloc.start()
    try:
        results = [capture.emit(payload) for _ in range(12)]
        _, peak = tracemalloc.get_traced_memory()
        with capture.lock:
            assert len(capture.queue) == 2
            assert sum(len(row.encode()) for row in capture.queue) <= 500_000
        assert results == [True, True] + [False] * 10
        # Covers both retained queue and temporary serialization for rejected
        # producers. This is deliberately separate from full-runtime RSS.
        assert peak < 3_000_000
    finally:
        tracemalloc.stop()
        release.set()
        summary = capture.close()
    assert summary['written'] == 2
    assert not summary['valid_coverage']
    assert 'queue_overflow' in summary['invalid_reasons']
    print('BOUNDED_QUEUE_TRACEMALLOC_PEAK_BYTES=' + str(peak))


def test_48_hour_startup_captures_with_explicit_offline_bounds(tmp_path):
    code_hashes = benchmark_code_hashes()
    end = datetime(2025, 5, 19, 13, 59, tzinfo=timezone.utc)
    bars = startup_bars(2880, end)
    runtime, trader, reader, counts = runtime_for(bars)
    before = reader.state()
    capture = DecisionCapture(enabled=True, output_dir=tmp_path / '48h', capacity=1,
                              max_bytes=64_000_000, max_nodes=4_000_000)
    rollback = install_poll_observer(trader, capture, identity=identities(SNAPSHOT),
                 readiness={'equivalent_feed_and_state': False}, state_reader=reader.state)
    start = time.perf_counter()
    try:
        asyncio.run(trader._poll_and_process())
    finally:
        rollback()
        summary = capture.close()
    elapsed = time.perf_counter() - start
    assert summary['valid_coverage'], summary
    assert summary['accepted'] == summary['written'] == 1
    path = tmp_path / '48h' / 'capture.jsonl'
    assert path.stat().st_size <= capture.max_bytes
    row = json.loads(path.read_text())
    assert row['trace']['before'] == before
    assert row['trace']['after'] == reader.state()
    assert row['trace']['input']['bars'] == bars
    assert len(row['trace']['input']['label_evidence']) == 2880
    transitions = [item for item in row['trace']['decisions'] if item.get('kind') == 'decision_transition']
    assert len(transitions) == 2880
    assert len(row['trace']['input']['decision_times']) == len(transitions)
    assert row['trace']['input']['decision_times'] == sorted(row['trace']['input']['decision_times'])
    assert not runtime.logger.errors
    assert counts == dict(parity_fetch=1, bullish_update=2880,
                          bullish_advance=2880, bullish_detect=2880)
    assert code_hashes == benchmark_code_hashes()
    report = dict(scope='PRIVATE_PINNED_OFFLINE_RUNTIME', bar_count=2880, code_sha256=code_hashes,
                  poll_and_close_seconds=elapsed, capture_bytes=path.stat().st_size,
                  max_bytes=capture.max_bytes, max_nodes=capture.max_nodes, capacity=1,
                  decision_transitions=len(transitions), coverage=summary,
                  shadow_method_calls=counts, decision='HOLD_VALIDATION',
                  limitations=['Direct poll observer benchmark; verified-startup guards have separate integration tests.',
                    'Explicit larger offline capture bounds; default settings not qualified.',
                    'Backfill invokes both shadow paths with synthetic feed and memory storage.',
                    'No independent observed account, real broker, or deployment identity evidence.'])
    (tmp_path / 'observation-48h-benchmark.json').write_text(canonical(report))
    print('OBSERVATION_48H_BENCHMARK=' + json.dumps(report, sort_keys=True))
