"""Trust boundaries: synthetic test keys never authorize production collection."""
import base64
import copy
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('evidence_cli', ROOT/'src/cli/check_yank_deployed_replay.py')
cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
e = cli.load_tool('evidence'); account = cli.load_tool('account'); startup = cli.load_tool('startup')


def fixture():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    key = Ed25519PrivateKey.generate()
    observations = [dict(kind='account_boundary', schema_version=1, request_id=str(i), endpoint=endpoint,
        account_pseudonym='pseudo', contract='MNQ', request_time='2025-05-19T14:00:00+00:00',
        receipt_time='2025-05-19T14:00:01+00:00', status_code=200, success=True, complete=True,
        snapshot_id='synthetic-atomic-version', records=[{'balance': 50000}] if field == 'accounts' else [])
        for i, (endpoint, field) in enumerate(account.ENDPOINTS.items())]
    requests = [{k: v for k, v in dict(o, kind='account_request').items() if k in ('kind', 'schema_version', 'request_id', 'endpoint', 'account_pseudonym', 'contract', 'request_time')} for o in observations]
    rows = requests + observations + [dict(kind='poll', trace=dict(sequence=1, poll_time='2025-05-19T14:00:02+00:00', receipt_time='2025-05-19T14:00:02+00:00', decisions=[], input=dict(poll_time='2025-05-19T14:00:02+00:00', receipt_time='2025-05-19T14:00:02+00:00', clock_reads=['2025-05-19T14:00:02+00:00'], decision_times=[])))]
    coverage = dict(valid_coverage=True, capture_closed=True, invalid_reasons=[], accepted=len(rows), written=len(rows), dropped=0)
    adapter_module = cli.load_tool('adapter')
    state = dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT', daily_pnl=0., daily_halted=False, last_trading_date=None, on_combine=True, is_backfill=True)
    initial = dict(state, classification='OBSERVED_STATE', checkpoint=adapter_module.Adapter(ROOT/'docs/yank-validation/snapshot/v1', state).checkpoint(), account_evidence=dict(status='OBSERVED', source_sha256='a'*64, receipt_time='2025-05-19T14:00:01+00:00'))
    rows[-1]['trace']['before'] = initial['checkpoint']['state']
    expected = dict(runtime={'model': 'pin'}, collector_sha256={'collector': 'pin'}, process=dict(boot_id='boot', pid=1, start_ticks=2),
                    checkpoint_sha256=e.sha(initial['checkpoint']), account_pseudonym='pseudo', contract='MNQ', account_max_age_seconds=30)
    payload = {k: copy.deepcopy(expected[k]) for k in ('runtime', 'collector_sha256', 'process', 'checkpoint_sha256')}
    payload.update(initial_state_sha256=e.sha(initial), capture_sha256=e.capture_digest(rows), coverage_sha256=e.sha(coverage))
    keys = {'test': base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}
    return key, keys, expected, payload, rows, coverage, initial


def verify(parts, bundle=None):
    key, keys, expected, payload, rows, coverage, initial = parts
    return e.verify_bundle(bundle or e.sign_bundle(payload, 'test', key), keys, expected, rows, coverage, initial, ROOT/'docs/yank-validation/snapshot/v1')


def test_trusted_complete_fixture_and_hold():
    v = verify(fixture())
    assert v['eligible'] and v['decision'] == 'HOLD_VALIDATION'


@pytest.mark.parametrize('target', ['signature', 'key', 'model', 'config', 'restart', 'checkpoint', 'sequence', 'coverage', 'missing_record'])
def test_tampering_never_admitted(target):
    f = fixture(); key, keys, expected, payload, rows, coverage, initial = f
    bundle = e.sign_bundle(payload, 'test', key)
    if target == 'signature': bundle['payload']['runtime'] = {'altered': True}
    elif target == 'key': keys.clear()
    elif target == 'model': expected['runtime'] = {'model': 'other'}
    elif target == 'config': expected['runtime'] = {'config': 'other'}
    elif target == 'restart': expected['process']['start_ticks'] += 1
    elif target == 'checkpoint': initial['checkpoint'] = {'wrong': True}
    elif target == 'sequence': rows[-1]['trace']['sequence'] = 2
    elif target == 'coverage': coverage['valid_coverage'] = False
    elif target == 'missing_record': rows.pop()
    result = verify(f, bundle)
    assert not result['eligible']
    if target == 'restart': assert result['signature'] == 'PASS' and result['identity'] == 'FAIL'


def test_legacy_declarations_cannot_upgrade():
    f = fixture()
    assert not e.verify_bundle(None, f[1], f[2], f[4], f[5], f[6])['eligible']


@pytest.mark.parametrize('failure', ['partial', 'stale', 'failed', 'missing', 'pending', 'non_atomic', 'contradictory'])
def test_account_unknown(failure):
    observations = fixture()[4][:-1]
    reply = next(o for o in observations if o['kind'] == 'account_boundary')
    if failure == 'partial': reply['complete'] = False
    elif failure == 'stale':
        observations[0]['request_time'] = reply['request_time'] = '2025-05-19T13:00:00+00:00'
        reply['receipt_time'] = '2025-05-19T13:00:01+00:00'
    elif failure == 'failed': reply['status_code'] = 500
    elif failure == 'missing': observations.pop()
    elif failure == 'pending': observations.append(dict(kind='account_request', request_id='pending', account_pseudonym='pseudo', request_time='2025-05-19T14:00:00+00:00'))
    elif failure == 'non_atomic': reply.pop('snapshot_id')
    else: observations[-1]['records'] = [dict(type=1, size=1, contractId='MNQ'), dict(type=2, size=1, contractId='MNQ')]
    result = account.account_verdict(observations, '2025-05-19T14:00:02+00:00', 'pseudo', 'MNQ')
    assert result['verdict'] == 'UNKNOWN' and result['state'] == 'UNKNOWN'


def test_disabled_startup_never_reads_trader():
    class Unavailable:
        def __getattribute__(self, key): raise AssertionError(key)
    assert startup.prepare_observation(Unavailable())['enabled'] is False


def test_startup_mismatch_no_wrappers(monkeypatch):
    from types import SimpleNamespace
    trader = SimpleNamespace(original='untouched')
    monkeypatch.setattr(startup, 'runtime_identity', lambda t: {'model': 'substituted'})
    result = startup.prepare_observation(trader, enabled=True, expected_release={'runtime': {'model': 'pinned'}}, signer=object(), key_id='test', pseudonym_salt=b'test')
    assert not result['enabled'] and vars(trader) == {'original': 'untouched'}


def test_transport_redaction_and_return_preserved(tmp_path):
    import asyncio
    from types import SimpleNamespace
    capture_module = cli.load_tool('capture')
    capture = capture_module.DecisionCapture(enabled=True, output_dir=tmp_path/'capture')
    class Response:
        status_code = 200
        def json(self): return dict(success=True, accounts=[dict(id=123, name='private', balance=10)])
    response = Response()
    class HTTP:
        async def post(self, *a, **kw): return response
    original = HTTP(); client = SimpleNamespace(_http=original)
    undo = account.install_account_observer(client, capture, 123, 'MNQ', b'salt')
    assert asyncio.run(client._http.post('https://example/Account/search', headers={'Authorization': 'secret'})) is response
    undo(); coverage = capture.close()
    assert client._http is original and coverage['valid_coverage']
    text = (tmp_path/'capture/capture.jsonl').read_text()
    assert 'private' not in text and 'secret' not in text and '123' not in text
    assert 'account_request' in text and 'account_boundary' in text


def test_direct_admission_dictionary_rejected():
    compare = cli.load_tool('compare')
    with pytest.raises(TypeError):
        compare.replay_capture([], {}, None, admission={'eligible': True})


def test_complete_signed_capture_replays_without_strategy_validation():
    import asyncio
    capture = cli.load_tool('capture'); compare = cli.load_tool('compare'); adapter_module = cli.load_tool('adapter')
    f = fixture(); key, keys, expected, payload, rows, coverage, initial = f
    adapter = adapter_module.Adapter(ROOT/'docs/yank-validation/snapshot/v1', initial)
    trace = asyncio.run(adapter.poll(dict(receipt_time='2025-05-19T14:00:02+00:00', request_id='test', status_code=200, bars=[], poll_time='2025-05-19T14:00:02+00:00', poll_observation='already_admitted', clock_reads=['2025-05-19T14:00:02+00:00'], execution_replies=[]), already_admitted=True))
    rows[-1] = dict(schema_version=1, kind='poll', trace=trace, identities=capture.identities(ROOT/'docs/yank-validation/snapshot/v1'), readiness=dict(equivalent_feed_and_state=True, clock_semantics='observed_per_call', identity_verification_scope='UNVERIFIED_LIVE_DECLARATION'))
    payload['capture_sha256'] = e.capture_digest(rows)
    result = asyncio.run(compare.replay_capture(rows, {'initial_state': initial}, ROOT/'docs/yank-validation/snapshot/v1', bundle=e.sign_bundle(payload, 'test', key), trusted_keys=keys, expected_release=expected, coverage=coverage))
    assert result['results'][0]['status'] == 'MATCH'
    assert result['results'][0]['verification_scope'] == 'TRUSTED_COLLECTOR_LIVE_COMPARISON'


def test_loaded_release_gate_detects_model_threshold_and_method_changes(tmp_path):
    from types import SimpleNamespace
    a = cli.load_tool('adapter'); c = cli.load_tool('capture')
    runtime = a.PrivateSnapshot(ROOT/'docs/yank-validation/snapshot/v1')
    state = dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT', daily_pnl=0., daily_halted=False, last_trading_date=None, on_combine=True, is_backfill=True)
    trader = runtime.construct(state)
    # Remove the private runtime's diagnostic logging closure so release pinning
    # sees the original installed class implementation; no live import or I/O.
    del trader._log_filter_decision
    trader._ts_client = SimpleNamespace(_http=object(), _account_id=123, _contract_id='MNQ')
    view = a.Adapter.__new__(a.Adapter); view.trader = trader
    view._buffer_cache_key = None; view._buffer_hash = None; view._warmup = {}
    expected = dict(runtime=startup.runtime_identity(trader), collector_sha256=startup.collector_identity(),
                    checkpoint_sha256=e.sha(view.checkpoint()), snapshot_identity=c.identities(runtime.root),
                    account_pseudonym=account.pseudonym(123, b'test'), contract='MNQ')
    original_poll = trader._poll_and_process
    trader.ml_filter.threshold += .01
    failed = startup.prepare_observation(trader, enabled=True, expected_release=expected, output_dir=tmp_path/'bad', signer=object(), key_id='test', pseudonym_salt=b'test')
    assert not failed['enabled'] and trader._poll_and_process == original_poll and not (tmp_path/'bad').exists()
    trader.ml_filter.threshold -= .01
    trader.ml_filter.model.synthetic_modified_attribute = True
    assert startup.runtime_identity(trader) != expected['runtime']
    del trader.ml_filter.model.synthetic_modified_attribute
    old_parse = trader._parse_bar
    trader._parse_bar = lambda row: None
    assert startup.runtime_identity(trader) != expected['runtime']
    trader._parse_bar = old_parse
    # Signed empty session remains ineligible, but install/rollback are real.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    session = startup.prepare_observation(trader, enabled=True, expected_release=expected, output_dir=tmp_path/'ok', signer=Ed25519PrivateKey.generate(), key_id='test', pseudonym_salt=b'test')
    assert isinstance(session, startup.ObservationSession)
    assert trader._poll_and_process != original_poll
    session.close()
    assert trader._poll_and_process == original_poll and not hasattr(trader, '_decision_capture')


def test_account_change_during_slow_poll_unknown():
    observations = fixture()[4][:-1]
    changed = copy.deepcopy(observations[-1])
    changed.update(request_id='changed', request_time='2025-05-19T14:00:03+00:00', receipt_time='2025-05-19T14:00:04+00:00')
    observations.append({k: v for k, v in dict(changed, kind='account_request').items() if k in ('kind', 'schema_version', 'request_id', 'endpoint', 'account_pseudonym', 'contract', 'request_time')})
    observations.append(changed)
    result = account.account_verdict(observations, '2025-05-19T14:00:05+00:00', 'pseudo', 'MNQ', interval_start='2025-05-19T14:00:02+00:00')
    assert result['state'] == 'UNKNOWN'


def test_account_position_disagrees_with_checkpoint():
    f = fixture()
    f[4][-2]['records'] = [dict(contractId='MNQ', type=1, size=2)]
    f[3]['capture_sha256'] = e.capture_digest(f[4])
    assert verify(f)['account'] == 'UNKNOWN'


def test_incomplete_checkpoint_cannot_pass_even_if_signed_and_pinned():
    f = fixture()
    f[6]['checkpoint'] = {'complete': 'caller-asserted'}
    f[2]['checkpoint_sha256'] = f[3]['checkpoint_sha256'] = e.sha(f[6]['checkpoint'])
    f[3]['initial_state_sha256'] = e.sha(f[6])
    assert verify(f)['checkpoint'] == 'FAIL'


@pytest.mark.parametrize('age, verdict', [(29.999, 'PASS'), (30.001, 'UNKNOWN')])
def test_freshness_ceiling_with_valid_request_chronology(age, verdict):
    from datetime import datetime, timedelta
    observations = fixture()[4][:-1]
    now = datetime.fromisoformat('2025-05-19T14:00:32+00:00')
    receipt = now - timedelta(seconds=age)
    start = receipt - timedelta(seconds=1)
    for o in observations:
        o['request_time'] = start.isoformat()
        if o['kind'] == 'account_boundary': o['receipt_time'] = receipt.isoformat()
    result = account.account_verdict(observations, now.isoformat(), 'pseudo', 'MNQ', max_age_seconds=30)
    assert result['verdict'] == verdict


@pytest.mark.parametrize('alteration', ['orphan', 'duplicate_request', 'duplicate_reply', 'endpoint', 'account', 'contract', 'start'])
def test_account_reply_requires_matching_unique_boundary(alteration):
    observations = fixture()[4][:-1]
    if alteration == 'orphan': observations.pop(0)
    elif alteration == 'duplicate_request': observations.insert(0, copy.deepcopy(observations[0]))
    elif alteration == 'duplicate_reply': observations.append(copy.deepcopy(observations[-1]))
    else:
        row = observations[3]
        field = {'endpoint': 'endpoint', 'account': 'account_pseudonym', 'contract': 'contract', 'start': 'request_time'}[alteration]
        row[field] = '2025-05-19T13:59:59+00:00' if alteration == 'start' else 'other'
    assert account.account_verdict(observations, '2025-05-19T14:00:02+00:00', 'pseudo', 'MNQ')['verdict'] == 'UNKNOWN'


def resign(f):
    f[3]['capture_sha256'] = e.capture_digest(f[4])
    f[5]['accepted'] = f[5]['written'] = len(f[4])
    f[3]['coverage_sha256'] = e.sha(f[5])
    return verify(f)


@pytest.mark.parametrize('field, value', [('size', 3), ('side', 0), ('limitPrice', 100.25), ('type', 4)])
def test_pending_exposure_contradictions(field, value):
    f = fixture()
    before = f[4][-1]['trace']['before']
    before['active_trade'] = dict(pending_entry=True, direction='bearish')
    before['_active_entry_decision'] = dict(direction='bearish', contracts=2, entry_price=100.)
    orders = next(r for r in f[4] if r.get('kind') == 'account_boundary' and r['endpoint'] == '/Order/searchOpen')
    orders['records'] = [dict(contractId='MNQ', type=1, side=1, size=2, limitPrice=100.)]
    # Account dimensions are independent of checkpoint reconstruction in this
    # focused contradiction control; coverage/signature remain authentic.
    assert resign(f)['account'] == 'PASS'
    orders['records'][0][field] = value
    result = resign(f)
    assert result['signature'] == 'PASS' and result['account'] == 'UNKNOWN'


@pytest.mark.parametrize('change', ['backwards_poll', 'receipt_before_start', 'input_disagreement', 'backwards_decisions', 'missing_decision_time', 'backwards_account_receipt'])
def test_signed_chronology_negative_controls(change):
    f = fixture(); poll = f[4][-1]
    if change == 'backwards_poll':
        another = copy.deepcopy(poll); another['trace']['sequence'] = 2
        another['trace']['poll_time'] = another['trace']['receipt_time'] = '2025-05-19T14:00:01+00:00'
        another['trace']['input'].update(poll_time=another['trace']['poll_time'], receipt_time=another['trace']['receipt_time'], clock_reads=[another['trace']['poll_time']])
        f[4].append(another)
    elif change == 'receipt_before_start': poll['trace']['receipt_time'] = '2025-05-19T14:00:01+00:00'
    elif change == 'input_disagreement': poll['trace']['input']['receipt_time'] = '2025-05-19T14:00:03+00:00'
    elif change in ('backwards_decisions', 'missing_decision_time'):
        poll['trace']['decisions'] = [{'kind': 'decision_transition'}, {'kind': 'decision_transition'}]
        poll['trace']['input']['decision_times'] = ['2025-05-19T14:00:04+00:00', '2025-05-19T14:00:03+00:00'] if change == 'backwards_decisions' else ['2025-05-19T14:00:03+00:00']
    else: f[4][-2]['receipt_time'] = '2025-05-19T13:59:59+00:00'
    result = resign(f)
    assert result['signature'] == 'PASS' and result['coverage'] == 'FAIL'
    assert any('chronology' in reason for reason in result['reasons'])


def prepared_runtime(tmp_path, *, signer=None, bars=None):
    from types import SimpleNamespace
    from datetime import datetime, timezone
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    a = cli.load_tool('adapter'); c = cli.load_tool('capture')
    runtime = a.PrivateSnapshot(ROOT/'docs/yank-validation/snapshot/v1')
    runtime.clock = datetime(2025, 5, 19, 14, tzinfo=timezone.utc)
    state = dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT', daily_pnl=0., daily_halted=False, last_trading_date=None, on_combine=True, is_backfill=True)
    trader = runtime.construct(state); del trader._log_filter_decision
    class HTTP:
        async def post(self, url, **kwargs):
            field = account.ENDPOINTS[next(p for p in account.ENDPOINTS if url.endswith(p))]
            payload = dict(success=True, **{field: [dict(id=123, balance=50000)] if field == 'accounts' else []})
            return SimpleNamespace(status_code=200, json=lambda: payload)
        async def get(self, *args, **kwargs):
            payload = {'Bars': bars or []}
            return SimpleNamespace(status_code=200, json=lambda: payload, content=a.canonical(payload).encode())
    class Auth:
        async def authenticate(self): return 'OFFLINE'
    trader.client = HTTP(); trader.auth = Auth()
    trader._ts_client = SimpleNamespace(_http=trader.client, _account_id=123, _contract_id='MNQ')
    view = a.Adapter.__new__(a.Adapter); view.trader = trader
    view._buffer_cache_key = None; view._buffer_hash = None; view._warmup = {}
    expected = dict(runtime=startup.runtime_identity(trader), collector_sha256=startup.collector_identity(),
                    checkpoint_sha256=e.sha(view.checkpoint()), snapshot_identity=c.identities(runtime.root),
                    account_pseudonym=account.pseudonym(123, b'test'), contract='MNQ', account_max_age_seconds=30)
    key = signer or Ed25519PrivateKey.generate()
    return runtime, trader, view, expected, key


def install_prepared(tmp_path, setup):
    _, trader, _, expected, key = setup
    session = startup.prepare_observation(trader, enabled=True, expected_release=expected, output_dir=tmp_path,
                                          signer=key, key_id='test', pseudonym_salt=b'test')
    assert isinstance(session, startup.ObservationSession), session
    return session


def test_runtime_pins_defining_helper_global_property_and_inference_fields(tmp_path):
    from zoneinfo import ZoneInfo
    setup = prepared_runtime(tmp_path); runtime, trader, _, expected, _ = setup
    original_multiplier = runtime.module.strategy_core.MNQ_NOTIONAL_MULTIPLIER
    runtime.module.strategy_core.MNQ_NOTIONAL_MULTIPLIER = original_multiplier + 1
    assert startup.runtime_identity(trader) != expected['runtime']
    runtime.module.strategy_core.MNQ_NOTIONAL_MULTIPLIER = original_multiplier
    helper = runtime.module.check_exit
    original_zone = helper.__globals__['_NY_TZ']
    helper.__globals__['_NY_TZ'] = ZoneInfo('UTC')
    assert startup.runtime_identity(trader) != expected['runtime']
    helper.__globals__['_NY_TZ'] = original_zone
    # A helper imported by a helper is verified in its defining module.
    globals_dict = runtime.module.resample_to_h1.__globals__
    original_validate = globals_dict['_validate_bars']
    globals_dict['_validate_bars'] = lambda *args, **kwargs: None
    assert startup.runtime_identity(trader) != expected['runtime']
    globals_dict['_validate_bars'] = original_validate
    risk_cls = type(trader._risk_manager); original_property = risk_cls.daily_pnl
    risk_cls.daily_pnl = property(lambda self: 123.)
    assert startup.runtime_identity(trader) != expected['runtime']
    risk_cls.daily_pnl = original_property
    ml_cls = type(trader.ml_filter); original_columns = ml_cls.FEATURE_COLS
    ml_cls.FEATURE_COLS = list(reversed(original_columns))
    assert startup.runtime_identity(trader) != expected['runtime']
    ml_cls.FEATURE_COLS = original_columns
    trader.ml_filter.FEATURE_COLS = original_columns[:-1]
    assert startup.runtime_identity(trader) != expected['runtime']
    del trader.ml_filter.FEATURE_COLS
    assert startup.runtime_identity(trader) == expected['runtime']


def test_changed_wrapped_binding_detected_before_owned_rollback(tmp_path):
    setup = prepared_runtime(tmp_path); trader = setup[1]
    session = install_prepared(tmp_path/'capture', setup)
    changed = lambda *args: None
    trader.ml_filter.predict_proba = changed
    summary = session.close()
    assert 'installed_binding_changed' in summary['invalid_reasons']
    assert trader.ml_filter.predict_proba is changed
    assert not summary['valid_coverage']


def test_close_is_idempotent_and_cannot_erase_new_session(tmp_path):
    setup = prepared_runtime(tmp_path); trader = setup[1]
    first = install_prepared(tmp_path/'first', setup)
    summary = first.close()
    second = install_prepared(tmp_path/'second', setup)
    second_poll = trader._poll_and_process; second_http = trader._ts_client._http
    assert first.close() == summary
    assert trader._poll_and_process is second_poll and trader._ts_client._http is second_http
    assert trader._decision_capture is second.capture
    second.close()


@pytest.mark.parametrize('failure', ['signer', 'manifest_write', 'evidence_write'])
def test_close_publication_failure_persists_invalid_coverage(tmp_path, monkeypatch, failure):
    import json
    class FailedSigner:
        def sign(self, message): raise RuntimeError('injected signing failure')
    setup = prepared_runtime(tmp_path, signer=FailedSigner() if failure == 'signer' else None)
    session = install_prepared(tmp_path/'capture', setup)
    original = Path.write_text
    def failing(path, *args, **kwargs):
        if path.name == {'manifest_write': 'manifest.json.pending', 'evidence_write': 'evidence.json.pending'}.get(failure):
            raise OSError('injected publication failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', failing)
    original_read = Path.read_text
    def bounded_read(path, *args, **kwargs):
        assert path.name != 'capture.jsonl', 'close must stream capture digest'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', bounded_read)
    summary = session.close()
    persisted = json.loads((tmp_path/'capture/coverage.json').read_text())
    assert not summary['valid_coverage'] and not persisted['valid_coverage']
    assert 'package_publication_failed' in persisted['invalid_reasons']
    assert not (tmp_path/'capture/manifest.json').exists()
    assert not (tmp_path/'capture/evidence.json').exists()


def test_nonempty_verified_startup_package_and_real_schema_account_unknown(tmp_path):
    import asyncio
    import json
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    bars = [dict(TimeStamp=f'2025-05-19T13:3{i}:00Z', Open=100, High=101, Low=99, Close=100, TotalVolume=1) for i in range(2)]
    setup = prepared_runtime(tmp_path, bars=bars); _, trader, _, expected, key = setup
    original_poll = trader._poll_and_process
    session = install_prepared(tmp_path/'capture', setup)
    async def exercise():
        for endpoint in account.ENDPOINTS:
            await trader._ts_client._http.post('https://offline' + endpoint, json={'accountId': 123})
        return await trader._poll_and_process()
    result = asyncio.run(exercise())
    assert result is None
    summary = session.close()
    assert summary['valid_coverage'], summary
    assert trader._poll_and_process == original_poll
    rows = [json.loads(line) for line in (tmp_path/'capture/capture.jsonl').read_text().splitlines()]
    trace = rows[-1]['trace']; times = trace['input']['decision_times']
    assert len(times) == len([d for d in trace['decisions'] if d.get('kind') == 'decision_transition']) == 2
    assert times == sorted(times)
    expected['process'] = session.process
    keys = {'test': base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode()}
    (tmp_path/'keys.json').write_text(json.dumps(keys)); (tmp_path/'release.json').write_text(json.dumps(expected))
    compare = cli.load_tool('compare')
    report = compare.run(tmp_path/'capture/manifest.json', tmp_path/'capture', ROOT/'docs/yank-validation/snapshot/v1', tmp_path/'comparison', evidence_path=tmp_path/'capture/evidence.json', trusted_keys_path=tmp_path/'keys.json', expected_release_path=tmp_path/'release.json')
    assert {k: report['admission'][k] for k in ('signature', 'identity', 'checkpoint', 'coverage')} == dict(signature='PASS', identity='PASS', checkpoint='PASS', coverage='PASS')
    assert report['admission']['account'] == 'UNKNOWN' and report['decision'] == 'HOLD_VALIDATION'
    # The observed times drive admission: a refresh after original poll start
    # but before the last decision cannot be ignored using poll-start state.
    boundaries = fixture()[4][:-1]
    clock = trace['input']['decision_times'][-1]
    from datetime import datetime, timedelta
    at = datetime.fromisoformat(clock); start = at-timedelta(seconds=3)
    for o in boundaries:
        o['request_time'] = (start-timedelta(seconds=2)).isoformat()
        if o['kind'] == 'account_boundary': o['receipt_time'] = (start-timedelta(seconds=1)).isoformat()
    changed = copy.deepcopy(boundaries[-1]); changed['request_id'] = 'during'
    changed['request_time'] = (at-timedelta(seconds=1)).isoformat(); changed['receipt_time'] = clock
    boundaries += [{k: v for k, v in dict(changed, kind='account_request').items() if k in ('kind','schema_version','request_id','endpoint','account_pseudonym','contract','request_time')}, changed]
    assert account.account_verdict(boundaries, clock, 'pseudo', 'MNQ', interval_start=start.isoformat())['verdict'] == 'UNKNOWN'


@pytest.mark.parametrize('change', ['threshold', 'model'])
def test_mid_session_change_invalidates_without_changing_poll_state(tmp_path, change):
    import asyncio
    bars = [dict(TimeStamp='2025-05-19T13:30:00Z', Open=100, High=101, Low=99, Close=100, TotalVolume=1)]
    baseline = prepared_runtime(tmp_path/'baseline', bars=bars)
    expected_result = asyncio.run(baseline[1]._poll_and_process()); expected_state = baseline[2].state()
    setup = prepared_runtime(tmp_path/'observed', bars=bars); trader = setup[1]
    session = install_prepared(tmp_path/'capture', setup)
    if change == 'threshold': trader.ml_filter.threshold += .01
    else: trader.ml_filter.model.synthetic_modified_attribute = True
    result = asyncio.run(trader._poll_and_process())
    assert result == expected_result and setup[2].state() == expected_state
    # Restoring settings before close must not erase the per-poll failure.
    if change == 'threshold': trader.ml_filter.threshold -= .01
    else: del trader.ml_filter.model.synthetic_modified_attribute
    summary = session.close()
    assert not summary['valid_coverage'] and 'runtime_changed_during_capture' in summary['invalid_reasons']


def test_original_projectx_reconcile_failed_http_keeps_flat_with_unknown_evidence(tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace
    from src.research.projectx_client import ProjectXClient
    capture_module = cli.load_tool('capture')
    class FailedHTTP:
        async def post(self, *args, **kwargs):
            return SimpleNamespace(status_code=500, json=lambda: {'success': False, 'orders': [], 'positions': []})
    client = ProjectXClient.__new__(ProjectXClient)
    client._http = FailedHTTP(); client._account_id = 123; client._contract_id = 'MNQ'
    async def headers(): return {}
    client._headers = headers
    baseline = asyncio.run(client.reconcile_state('123'))
    capture = capture_module.DecisionCapture(enabled=True, output_dir=tmp_path/'account')
    undo = account.install_account_observer(client, capture, 123, 'MNQ', b'test')
    observed = asyncio.run(client.reconcile_state('123'))
    undo(); assert capture.close()['valid_coverage']
    assert baseline.status == observed.status == 'FLAT'
    rows = [json.loads(line) for line in (tmp_path/'account/capture.jsonl').read_text().splitlines()]
    assert len([r for r in rows if r['kind'] == 'account_request']) == 2
    replies = [r for r in rows if r['kind'] == 'account_boundary']
    assert len(replies) == 2 and all(r['status_code'] == 500 for r in replies)
    result = account.account_verdict(rows, replies[-1]['receipt_time'], account.pseudonym(123, b'test'), 'MNQ')
    assert result['state'] == 'UNKNOWN' and result['verdict'] == 'UNKNOWN'


def test_signed_account_admission_consumes_final_decision_boundary():
    f = fixture(); trace = f[4][-1]['trace']
    trace['decisions'] = [{'kind': 'decision_transition'}, {'kind': 'decision_transition'}]
    trace['input']['decision_times'] = ['2025-05-19T14:00:03+00:00', '2025-05-19T14:00:04+00:00']
    assert resign(f)['account'] == 'PASS'
    changed = copy.deepcopy(f[4][-2]); changed['request_id'] = 'refresh-during-poll'
    changed['request_time'] = '2025-05-19T14:00:03.100000+00:00'; changed['receipt_time'] = '2025-05-19T14:00:03.200000+00:00'
    request = {k: v for k, v in dict(changed, kind='account_request').items() if k in ('kind','schema_version','request_id','endpoint','account_pseudonym','contract','request_time')}
    f[4][-1:-1] = [request, changed]
    result = resign(f)
    assert result['signature'] == result['coverage'] == 'PASS'
    assert result['account'] == 'UNKNOWN'
    assert 'within decision interval' in result['account_details'][0]['reasons'][0]


def test_per_poll_method_change_is_not_hidden_by_restoring_before_close(tmp_path):
    import asyncio
    setup = prepared_runtime(tmp_path); trader = setup[1]
    session = install_prepared(tmp_path/'capture', setup)
    original = trader._parse_bar
    trader._parse_bar = lambda row: None
    assert asyncio.run(trader._poll_and_process()) is None
    trader._parse_bar = original
    result = session.close()
    assert not result['valid_coverage'] and 'runtime_changed_during_capture' in result['invalid_reasons']


def test_guarded_poll_does_not_read_collector_or_process_files(tmp_path, monkeypatch):
    import asyncio
    setup = prepared_runtime(tmp_path)
    session = install_prepared(tmp_path/'capture', setup)
    def forbidden(*args, **kwargs): raise AssertionError('observer filesystem access during poll')
    with monkeypatch.context() as scoped:
        scoped.setattr(startup, 'collector_identity', forbidden)
        scoped.setattr(startup, 'process_identity', forbidden)
        scoped.setattr(Path, 'read_text', forbidden)
        scoped.setattr(Path, 'read_bytes', forbidden)
        assert asyncio.run(setup[1]._poll_and_process()) is None
        assert not session.capture.invalid_reasons
    assert session.close()['valid_coverage']


def test_in_place_observer_code_change_invalidates_before_rollback(tmp_path):
    setup = prepared_runtime(tmp_path); trader = setup[1]
    session = install_prepared(tmp_path/'capture', setup)
    observer = trader._detect_and_enter
    observer.__code__ = observer.__code__.replace(co_consts=tuple('discarded_times' if value == 'decision_times' else value for value in observer.__code__.co_consts))
    result = session.close()
    assert not result['valid_coverage']
    assert 'installed_observer_code_changed' in result['invalid_reasons']
