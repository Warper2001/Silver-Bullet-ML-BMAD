"""Focused contracts for the standalone unprofiled measurement harness."""
from observer_overhead import LIMITS, MODES, schedule
import pytest



def quiescent_drain(capture):
    """Test-only pacing outside all observed polls, never a collector wait."""
    import time
    deadline = time.monotonic()+2
    while capture.written != capture.accepted:
        assert capture.worker.is_alive() and time.monotonic() < deadline
        time.sleep(.001)


def test_counterbalanced_fresh_cell_plan():
    jobs = schedule()
    assert len(jobs) == len(set(jobs)) == 18
    for workload in ('startup', 'steady'):
        orders = [[mode for work, rep, mode in jobs if work == workload and rep == repeat]
                  for repeat in range(3)]
        assert all(set(order) == set(MODES) for order in orders)
        assert all({order[position] for order in orders} == set(MODES) for position in range(3))
    assert LIMITS == dict(capacity=1, max_bytes=64_000_000, max_nodes=4_000_000)


def test_both_shadows_accepted_and_rejected_decisions(tmp_path):
    import asyncio
    import importlib.util
    import json
    from pathlib import Path
    from src.research.yank_deployed_validation.capture import DecisionCapture, identities, install_poll_observer
    path = Path(__file__).resolve().parents[2]/'research/test_yank_deployed_capture.py'
    spec = importlib.util.spec_from_file_location('overhead_synthetic_entries', path)
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    for accepted in (True, False):
        observed, control = fixtures.seeded_entry_adapter(), fixtures.seeded_entry_adapter()
        counts = []
        for adapter in (observed, control):
            adapter.trader.ml_filter.threshold = 0. if accepted else 1.
            counter = dict(parity=0, update=0, advance=0, detect=0)
            counts.append(counter)
            async def fetch(*args, _counter=counter, **kwargs):
                _counter['parity'] += 1
                return []
            adapter.runtime.module.fetch_px_ts_shaped = fetch
            adapter.trader._data_shadow = True
            adapter.trader._px_auth = object()
            adapter.trader._px_data_contract_id = 'SYNTHETIC'
            for name, label in (('_update_shadow_bullish_m15_choch','update'),
                                ('_advance_shadow_trade','advance'), ('_detect_shadow_bullish_entry','detect')):
                original = getattr(adapter.trader, name)
                def call(*args, _original=original, _counter=counter, _label=label, **kwargs):
                    _counter[_label] += 1
                    return _original(*args, **kwargs)
                setattr(adapter.trader, name, call)
        capture = DecisionCapture(enabled=True, output_dir=tmp_path/str(accepted), **LIMITS)
        rollback = install_poll_observer(observed.trader, capture, identity=identities(fixtures.SNAPSHOT),
            readiness={'equivalent_feed_and_state':True}, state_reader=observed.state)
        bars = [dict(TimeStamp='2025-05-19T14:00:00Z', Open=102, High=103, Low=101, Close=102, TotalVolume=100),
                dict(TimeStamp='2025-05-19T14:01:00Z', Open=104.5, High=105, Low=104, Close=104.5, TotalVolume=100),
                dict(TimeStamp='2025-05-19T14:02:00Z', Open=104, High=112, Low=103, Close=111, TotalVolume=100)]
        try:
            for index, bar in enumerate(bars):
                event = dict(bars=[bar], receipt_time=f'2025-05-19T14:0{index+1}:00Z', request_id=str(index), status_code=200)
                asyncio.run(observed.poll(event)); asyncio.run(control.poll(event))
                assert observed.state() == control.state()
                assert observed.runtime.decisions == control.runtime.decisions
                assert observed.trader._ts_client.intentions == control.trader._ts_client.intentions
                quiescent_drain(capture)
        finally:
            rollback(); summary = capture.close()
        assert summary['valid_coverage'], summary
        assert counts[0] == counts[1] == dict(parity=3, update=3, advance=3, detect=3)
        rows = [json.loads(line) for line in (tmp_path/str(accepted)/'capture.jsonl').read_text().splitlines()]
        intentions = [i for row in rows for i in row['trace']['intentions']]
        assert bool(intentions) is accepted
        predictions = [d for row in rows for d in row['trace']['decisions'] if d.get('kind') == 'ml_prediction']
        assert predictions
        assert bool(observed.trader.completed_trades) is accepted


@pytest.mark.parametrize('synthetic_multi_poll', [False, True], ids=['single-poll', 'consistent-multi-poll'])
def test_fully_guarded_synthetic_execution(tmp_path, monkeypatch, synthetic_multi_poll):
    import asyncio
    import base64
    import datetime as datetime_module
    from datetime import datetime as real_datetime, timezone
    import importlib.util
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    from test_evidence import startup, e, account, cli
    from test_startup_benchmark import with_shadows
    from src.research.yank_deployed_validation.capture import identities
    path = Path(__file__).resolve().parents[2]/'research/test_yank_deployed_capture.py'
    spec = importlib.util.spec_from_file_location('guarded_synthetic_entries', path)
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    adapter_module = cli.load_tool('adapter')
    fixtures.Adapter = adapter_module.Adapter
    normalize = adapter_module.normalize
    for accepted in (True, False):
        observed, control = fixtures.seeded_entry_adapter(), fixtures.seeded_entry_adapter()
        for adapter in (observed, control):
            del adapter.trader._log_filter_decision
            del adapter.trader._detect_and_enter
            del adapter.trader.ml_filter.predict_proba
            adapter.trader.ml_filter.threshold = 0. if accepted else 1.
            with_shadows((adapter.runtime, adapter.trader), [])
            adapter.trader._ts_client._http = SimpleNamespace()
            adapter.trader._ts_client._account_id = 123
            adapter.trader._ts_client._contract_id = 'MNQ'
        expected = dict(runtime=startup.runtime_identity(control.trader), collector_sha256=startup.collector_identity(),
            checkpoint_sha256=e.sha(control.checkpoint()), snapshot_identity=identities(fixtures.SNAPSHOT),
            account_pseudonym=account.pseudonym(123, b'test'), contract='MNQ', account_max_age_seconds=30)
        key = Ed25519PrivateKey.generate()
        clock = [real_datetime(2025, 5, 19, 14, tzinfo=timezone.utc)]
        class CollectorClock(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return clock[0].astimezone(tz) if tz is not None else clock[0].replace(tzinfo=None)
        # Bind a synthetic clock only into the install-time collector closures.
        # Restore the datetime module before any strategy polling takes place.
        with monkeypatch.context() as clock_patch:
            if synthetic_multi_poll:
                clock_patch.setattr(datetime_module, 'datetime', CollectorClock)
            session = startup.prepare_observation(observed.trader, enabled=True, expected_release=expected,
                output_dir=tmp_path/str(accepted), signer=key, key_id='synthetic',
                pseudonym_salt=b'test', capture_limits=LIMITS)
        assert isinstance(session, startup.ObservationSession), session
        bars = [dict(TimeStamp='2025-05-19T14:00:00Z', Open=102, High=103, Low=101, Close=102, TotalVolume=100),
                dict(TimeStamp='2025-05-19T14:01:00Z', Open=104.5, High=105, Low=104, Close=104.5, TotalVolume=100),
                dict(TimeStamp='2025-05-19T14:02:00Z', Open=104, High=112, Low=103, Close=111, TotalVolume=100)]
        decisions = [[], []]
        shadows = [dict(update=0, advance=0, detect=0), dict(update=0, advance=0, detect=0)]
        def run_profiled(adapter, event, side):
            decision_code = type(adapter.trader)._log_filter_decision.__code__
            shadow_codes = {getattr(type(adapter.trader), name).__code__: label for name, label in (
                ('_update_shadow_bullish_m15_choch', 'update'), ('_advance_shadow_trade', 'advance'),
                ('_detect_shadow_bullish_entry', 'detect'))}
            def profile(frame, event, arg):
                if event == 'call':
                    if frame.f_code is decision_code:
                        decisions[side].append(normalize(
                            {k:v for k,v in frame.f_locals.items() if k != 'self'}))
                    if frame.f_code in shadow_codes:
                        shadows[side][shadow_codes[frame.f_code]] += 1
            previous = sys.getprofile()
            try:
                sys.setprofile(profile)
                asyncio.run(adapter.poll(event))
            finally:
                sys.setprofile(previous)
        try:
            # One response contains all three ordered decisions. The strategy's
            # private historical clock and collector's real receipt clock must
            # not be presented as three chronologically coherent live polls.
            events = [dict(bars=bars, receipt_time='2025-05-19T14:03:00Z', request_id='synthetic', status_code=200)]
            if synthetic_multi_poll:
                events = [dict(bars=[bar], receipt_time=f'2025-05-19T14:0{index+1}:00Z', request_id=str(index), status_code=200)
                          for index, bar in enumerate(bars)]
            for event in events:
                clock[0] = real_datetime.fromisoformat(event['receipt_time'].replace('Z', '+00:00'))
                run_profiled(observed, event, 0); run_profiled(control, event, 1)
                assert observed.state() == control.state()
                assert observed.trader._ts_client.intentions == control.trader._ts_client.intentions
                quiescent_drain(session.capture)
        finally:
            coverage = session.close()
        assert decisions[0] and decisions[0] == decisions[1]
        assert shadows[0] == shadows[1] == dict(update=3, advance=3, detect=3)
        assert coverage['valid_coverage'], coverage
        rows = [json.loads(line) for line in (tmp_path/str(accepted)/'capture.jsonl').read_text().splitlines()]
        assert len(rows) == (3 if synthetic_multi_poll else 1)
        intentions = [item for row in rows for item in row['trace']['intentions']]
        assert bool(intentions) is accepted
        assert bool(observed.trader.completed_trades) is accepted
        assert all(row['trace']['after']['risk'] == control.state()['risk'] for row in rows[-1:])
        manifest = json.loads((tmp_path/str(accepted)/'manifest.json').read_text())
        assert manifest['publication'] == 'COMPLETE'
        bundle = json.loads((tmp_path/str(accepted)/'evidence.json').read_text())
        keys = {'synthetic':base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}
        verdict = e.verify_bundle(bundle, keys, dict(expected, process=session.process), rows, coverage,
            manifest['initial_state'], fixtures.SNAPSHOT)
        assert {name:verdict[name] for name in ('signature','identity','checkpoint','coverage')} == {
            name:'PASS' for name in ('signature','identity','checkpoint','coverage')}, verdict
        assert verdict['account'] == 'UNKNOWN' and verdict['decision'] == 'HOLD_VALIDATION'
        assert verdict['eligible'] is False
        (tmp_path/str(accepted)/'verification.json').write_text(json.dumps(verdict, indent=2))
        (tmp_path/str(accepted)/'public-keys.json').write_text(json.dumps(keys, indent=2))
        (tmp_path/str(accepted)/'expected-release.json').write_text(json.dumps(dict(expected, process=session.process), indent=2))
        assert session.close() == coverage


def test_timestamp_cache_preserves_representations_and_is_bounded():
    from datetime import datetime, timedelta, timezone
    from src.research.yank_deployed_validation.adapter import _timestamp_parts, _utc_timestamp_parts
    _utc_timestamp_parts.cache_clear()
    utc = datetime(2025, 5, 19, 14, 13, tzinfo=timezone.utc)
    local = utc.astimezone(timezone(timedelta(hours=2)))
    class CustomTimestamp(datetime):
        def isoformat(self): return 'CUSTOM'
    values = [utc, local, utc.replace(tzinfo=None), CustomTimestamp(2025, 5, 19, tzinfo=timezone.utc)]
    for value in values:
        assert _timestamp_parts(value) == (value.isoformat(), value.replace(minute=0, second=0, microsecond=0))
    assert _utc_timestamp_parts.cache_info().currsize == 1
    assert _timestamp_parts(local)[0] != _timestamp_parts(utc)[0]
    for i in range(9000):
        _timestamp_parts(utc + timedelta(minutes=i))
    assert _utc_timestamp_parts.cache_info().currsize == 8192
    assert _timestamp_parts(utc)[0] == utc.isoformat()


def test_cached_timestamp_does_not_cache_mutable_bar_values(tmp_path):
    from datetime import datetime, timezone
    import hashlib
    from test_evidence import prepared_runtime, cli
    from test_observation_benchmark import startup_bars
    import asyncio
    import sys
    bars = startup_bars(3, datetime(2025, 5, 19, 13, 59, tzinfo=timezone.utc))
    runtime, trader, view, *_ = prepared_runtime(tmp_path, bars=bars)
    a = sys.modules[type(view).__module__]
    asyncio.run(trader._poll_and_process())
    original = view.state()
    trader.dollar_bars[0].timestamp = datetime.fromisoformat(trader.dollar_bars[0].timestamp.isoformat())
    timestamp = trader.dollar_bars[0].timestamp
    assert type(timestamp) is datetime and timestamp.tzinfo is timezone.utc
    # Keep a timestamp hot while mutable fields change; force a new extraction
    # just as an advancing decision boundary does in the existing state reader.
    a._timestamp_parts(timestamp)
    hits = a._utc_timestamp_parts.cache_info().hits
    a._timestamp_parts(timestamp)
    assert a._utc_timestamp_parts.cache_info().hits == hits+1
    trader.dollar_bars[0].high += 5
    view._buffer_cache_key = None
    after = view.state()
    assert after['accepted_buffer_sha256'] != original['accepted_buffer_sha256']
    rows = [[b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume,
             b.notional_value, b.is_forward_filled] for b in trader.dollar_bars]
    assert after['accepted_buffer_sha256'] == hashlib.sha256(a.canonical(rows).encode()).hexdigest()
    # Clearing the immutable cache cannot change a decision-boundary snapshot.
    a._utc_timestamp_parts.cache_clear(); view._buffer_cache_key = None
    assert view.state() == after


def test_capacity_one_retains_failure_and_bounded_queue(tmp_path, monkeypatch):
    import threading
    import tracemalloc
    from src.research.yank_deployed_validation.capture import DecisionCapture
    entered, release = threading.Event(), threading.Event()
    original = DecisionCapture._drain
    def delayed(self):
        entered.set()
        if release.wait(10): original(self)
    monkeypatch.setattr(DecisionCapture, '_drain', delayed)
    capture = DecisionCapture(enabled=True, output_dir=tmp_path/'capture', **LIMITS)
    assert entered.wait(2)
    payload = {'kind':'synthetic_bounded_record', 'body':'x'*200_000}
    tracemalloc.start()
    try:
        outcomes = [capture.emit(payload) for _ in range(12)]
        _, peak = tracemalloc.get_traced_memory()
        assert outcomes == [True] + [False]*11
        assert len(capture.queue) == 1
        retained = sum(len(row.encode()) for row in capture.queue)
        assert retained < 201_000
        assert peak < 2_000_000
    finally:
        tracemalloc.stop(); release.set(); coverage = capture.close()
    assert coverage['accepted'] == coverage['written'] == 1
    assert coverage['dropped'] == 11
    assert coverage['valid_coverage'] is False
    assert 'queue_overflow' in coverage['invalid_reasons']
    print(f'CAPACITY_ONE_OBSERVED_QUEUE_BYTES={retained} TRACEMALLOC_PEAK_BYTES={peak}')


def test_diagnostic_profile_aggregates_reloaded_code_objects(tmp_path):
    from profile_observer_overhead import AggregateProfile
    namespaces = [{}, {}]
    for namespace in namespaces:
        exec(compile('def repeated():\n    return 1\n', 'synthetic-reloaded.py', 'exec'), namespace)
    profile = AggregateProfile()
    profile.enable()
    try:
        for _ in range(3): namespaces[0]['repeated']()
        for _ in range(5): namespaces[1]['repeated']()
    finally:
        profile.disable()
    entries = [item for item in profile.getstats() if getattr(item.code, 'co_name', '') == 'repeated']
    assert len(entries) == 2
    profile.create_stats()
    row = profile.stats[('synthetic-reloaded.py', 1, 'repeated')]
    assert row[:2] == (8, 8)
    assert row[2] == sum(item.inlinetime for item in entries)
    assert row[3] == sum(item.totaltime for item in entries)
    import pstats
    profile.dump_stats(tmp_path/'diagnostic.pstats')
    assert pstats.Stats(str(tmp_path/'diagnostic.pstats')).stats[('synthetic-reloaded.py', 1, 'repeated')][:2] == (8, 8)
    assert pstats.Stats(str(tmp_path/'diagnostic.pstats.standard-lossy')).stats[('synthetic-reloaded.py', 1, 'repeated')][1] in (3, 5)


def test_diagnostic_profile_ignores_orphan_caller_edges():
    from types import SimpleNamespace
    from profile_observer_overhead import AggregateProfile
    profile = AggregateProfile()
    entry = SimpleNamespace(code='retained', callcount=3, reccallcount=1,
        inlinetime=.2, totaltime=.4, calls=[SimpleNamespace(code='absent builtin')])
    profile.getstats = lambda:[entry]
    profile.snapshot_stats()
    assert profile.stats == {('~', 0, 'retained'):(2, 3, .2, .4, {})}
