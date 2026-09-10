"""Offline acquisition boundaries; no provider credentials or network needed."""
import base64
import hashlib
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import httpx
import pytest

from src.research.yank_deployed_validation.acquire import (
    ENDPOINT, HOST, REQUIRED_EVIDENCE, GateError, acquire, cache_token_provider,
    gate_blockers, http_get,
)
from src.research.yank_deployed_validation.tradestation import request_ledger, validate_archive


class FakeClock:
    def __init__(self):
        self.elapsed = 0
        self.waits = []

    def now(self):
        return datetime(2026, 9, 10, tzinfo=timezone.utc) + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.elapsed += seconds


def gate():
    return dict(credential_sha256=hashlib.sha256(b'test-only-token').hexdigest(), symbol='MNQM25', endpoint=HOST + ENDPOINT, incremental_cost_usd=0,
                entitlement_verified=True, acquisition_authorized=True, contract_verified=True,
                evidence_references={key: dict(reference='independently-reviewed-record', reviewed_by='operator',
                    credential_sha256=hashlib.sha256(b'test-only-token').hexdigest(), reviewed_at='2026-09-10T00:00:00Z', symbol='MNQM25', endpoint=HOST + ENDPOINT)
                    for key in REQUIRED_EVIDENCE})


def reply(request, status=200, headers=None):
    raw = json.dumps(dict(Bars=[dict(TimeStamp=request['first_inclusive'], Open='1', High='2', Low='1',
                                   Close='2', TotalVolume='7', BarStatus='Closed', IsEndOfHistory=True)]),
                     indent=3).encode() + b'\n'
    return SimpleNamespace(status_code=status, headers=headers or {}, content=raw)


def run(tmp_path, transport, **kwargs):
    return acquire(gate(), tmp_path / 'out', transport=transport,
                   clock=kwargs.pop('clock', FakeClock()), token_provider=lambda: 'test-only-token', **kwargs)


@pytest.mark.parametrize('missing', ['symbol', 'endpoint', 'incremental_cost_usd', 'entitlement_verified',
                                     'acquisition_authorized', 'contract_verified', 'evidence_references'])
def test_gate_before_token_or_network(tmp_path, missing):
    declaration = gate()
    del declaration[missing]
    def forbidden(*args):
        pytest.fail('gate must run before token/network')
    result = acquire(declaration, tmp_path / 'out', token_provider=forbidden, transport=forbidden)
    assert result['status'] == 'BLOCKED' and result['blockers']
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('value', [False, '0', None, 0.01, float('nan')])
def test_nonzero_or_ambiguous_cost(value):
    declaration = gate()
    declaration['incremental_cost_usd'] = value
    assert gate_blockers(declaration)


def test_evidence_pinned_to_exact_contract_and_endpoint():
    declaration = gate()
    declaration['evidence_references']['zero_cost']['symbol'] = '@MNQ'
    assert 'evidence not pinned' in gate_blockers(declaration)[0]


def test_nine_fixed_raw_requests_spacing_and_archive_validation(tmp_path):
    clock = FakeClock()
    calls, originals = [], []
    def transport(request, token):
        calls.append((dict(request), clock.elapsed))
        response = reply(request)
        originals.append(response.content)
        return response
    assert run(tmp_path, transport, clock=clock)['status'] == 'COMPLETE'
    archive = json.loads((tmp_path / 'out/archive.json').read_bytes())
    assert len(archive) == len(calls) == 9
    assert [at for _, at in calls] == list(range(0, 540, 60))
    for (actual, _), expected, envelope, raw in zip(calls, request_ledger()['requests'], archive, originals):
        assert {k: v for k, v in actual.items() if k != 'started_at'} == expected
        assert base64.b64decode(envelope['raw_response_base64']) == raw
    validation = validate_archive(archive)
    assert not validation['counts']
    assert all(entry['verified'] for entry in validation['archive_byte_integrity'])
    assert not validation['historical_arrival_evidence']
    assert validation['calendar_coverage_admission'] == 'UNKNOWN'


@pytest.mark.parametrize('status', [401, 403, 402, 302, 400])
def test_auth_charge_redirect_rejections_stop(tmp_path, status):
    calls = []
    def transport(request, token):
        calls.append(request)
        return reply(request, status)
    assert run(tmp_path, transport)['status'] == 'INCOMPLETE'
    assert len(calls) == 1
    assert not (tmp_path / 'out/archive.json').exists()
    assert len(list((tmp_path / 'out').glob('*-result.json'))) == 1


@pytest.mark.parametrize('retry_after,expected', [('125', 125), ('Thu, 10 Sep 2026 00:03:00 GMT', 180),
                                               ('garbage', 60), ('-10', 60)])
def test_retry_after_and_retry_budget(tmp_path, retry_after, expected):
    clock = FakeClock()
    calls = []
    def transport(request, token):
        calls.append(clock.elapsed)
        return reply(request, 429, {'Retry-After': retry_after})
    assert run(tmp_path, transport, clock=clock)['status'] == 'INCOMPLETE'
    assert len(calls) == 3
    assert calls[1] == expected
    assert calls[2] - calls[1] >= 60
    assert all(wait <= 60 for wait in clock.waits)


def test_timeout_retry_then_success(tmp_path):
    calls = []
    def transport(request, token):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout('do not persist test-only-token')
        return reply(request)
    assert run(tmp_path, transport)['status'] == 'COMPLETE'
    assert len(calls) == 10
    assert 'test-only-token' not in ''.join(p.read_text() for p in (tmp_path / 'out').iterdir())


@pytest.mark.parametrize('payload,headers', [({'Bars': [], 'NextToken': 'opaque'}, {}),
    ({'Bars': []}, {'Link': '<https://evil.invalid>; rel="next"'}), ({'Bars': []}, {}),
    ({'Error': 'entitlement unavailable'}, {}), ('not-json', {}), ({'Bars': [None]}, {})])
def test_partial_malformed_pagination_rejected(tmp_path, payload, headers):
    def transport(request, token):
        return SimpleNamespace(status_code=200, content=json.dumps(payload).encode(), headers=headers)
    assert run(tmp_path, transport)['status'] == 'INCOMPLETE'
    assert not (tmp_path / 'out/archive.json').exists()
    assert list((tmp_path / 'out').glob('*-result.json'))


@pytest.mark.parametrize('payload', [{'access_token': 'other-secret'}, {'nested': {'refreshToken': 'other-secret'}},
                                  {'error': 'Bearer other-secret'}, {'error': 'test-only-token'}])
def test_response_credentials_redact_and_invalidate(tmp_path, payload):
    def transport(request, token):
        return SimpleNamespace(status_code=200, content=json.dumps(payload).encode(), headers={})
    result = run(tmp_path, transport)
    assert result['blockers'] == ['credential_redaction_invalidates_raw_evidence']
    saved = ''.join(p.read_text() for p in (tmp_path / 'out').iterdir())
    assert 'other-secret' not in saved and 'test-only-token' not in saved
    envelope = json.loads(next((tmp_path / 'out').glob('*-result.json')).read_text())['envelope']
    assert envelope['redacted'] and not envelope['raw_integrity_valid']
    assert 'raw_response_base64' not in envelope
    assert 'raw_sha256' not in envelope['response_metadata']


def test_interruption_keeps_completed_attempts_and_never_overwrites(tmp_path):
    calls = []
    def transport(request, token):
        calls.append(request)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return reply(request)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, transport)
    assert len(list((tmp_path / 'out').glob('*-started.json'))) == 2
    assert len(list((tmp_path / 'out').glob('*-result.json'))) == 1
    assert not (tmp_path / 'out/archive.json').exists()
    assert json.loads((tmp_path / 'out/status.json').read_text())['status'] == 'INCOMPLETE'
    original = {p.name: p.read_bytes() for p in (tmp_path / 'out').iterdir()}
    with pytest.raises(GateError, match='fresh output'):
        run(tmp_path, transport)
    assert original == {p.name: p.read_bytes() for p in (tmp_path / 'out').iterdir()}


def test_read_only_cache_refuses_missing_expiry_and_malformed(tmp_path):
    cache = tmp_path / 'cache.json'
    provider = cache_token_provider(cache, FakeClock())
    with pytest.raises(GateError, match='missing or malformed'):
        provider()
    for contents in ['not json', json.dumps({'access_token': 'secret'}),
                     json.dumps({'access_token': 'secret', 'expires_at': '2026-09-09T00:00:00Z'})]:
        cache.write_text(contents)
        with pytest.raises(GateError):
            provider()
        assert cache.read_text() == contents
    cache.write_text(json.dumps({'access_token': 'secret', 'expires_at': '2026-09-10T00:05:00Z'}))
    before = cache.read_bytes()
    assert provider() == 'secret'
    assert cache.read_bytes() == before


def test_expired_provider_no_network(tmp_path):
    def transport(*args):
        pytest.fail('expired cache must block network')
    result = acquire(gate(), tmp_path / 'out', transport=transport,
                     token_provider=cache_token_provider(tmp_path / 'missing', FakeClock()))
    assert result['status'] == 'BLOCKED'
    assert not list((tmp_path / 'out').glob('*-started.json'))


def test_http_transport_fixed_host_no_redirect_or_environment(monkeypatch):
    calls = []
    class Client:
        def __init__(self, **options):
            assert options == dict(follow_redirects=False, timeout=60, trust_env=False)
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, **kwargs):
            calls.append((url, kwargs))
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    request = request_ledger()['requests'][0]
    http_get(request, 'test-token')
    assert calls == [(HOST + ENDPOINT, dict(params=request['query'], headers={'Authorization': 'Bearer test-token'}))]


def test_raw_token_file_expiry_hints_never_write(tmp_path):
    from src.research.yank_deployed_validation.acquire import file_token_provider
    path = tmp_path / '.access_token'
    clock = FakeClock()
    def jwt(exp):
        claim = base64.urlsafe_b64encode(json.dumps({'exp': exp}).encode()).decode().rstrip('=')
        return 'eyJhbGciOiJub25lIn0.' + claim + '.unverified'
    good = jwt((clock.now() + timedelta(hours=1)).timestamp())
    path.write_text(good)
    assert file_token_provider(path, clock=clock)() == good
    assert path.read_text() == good
    for bad in [jwt((clock.now() - timedelta(seconds=1)).timestamp()), jwt(True), 'opaque', 'a.invalid.c']:
        path.write_text(bad)
        with pytest.raises(GateError):
            file_token_provider(path, clock=clock)()
        assert path.read_text() == bad
    path.write_text('opaque')
    assert file_token_provider(path, expires_at='2026-09-10T02:00:00Z', clock=clock)() == 'opaque'
    path.write_text(jwt((clock.now() - timedelta(seconds=1)).timestamp()))
    with pytest.raises(GateError):
        file_token_provider(path, expires_at='2027-01-01T00:00:00Z', clock=clock)()


def test_full_retry_budget_twenty_seven_attempts(tmp_path):
    attempts = {}
    def transport(request, token):
        identifier = request['request_id']
        attempts[identifier] = attempts.get(identifier, 0) + 1
        return reply(request, 503 if attempts[identifier] < 3 else 200)
    assert run(tmp_path, transport)['status'] == 'COMPLETE'
    assert len(attempts) == 9 and set(attempts.values()) == {3}
    assert len(list((tmp_path / 'out').glob('*-result.json'))) == 27


def test_bad_ohlcv_preserved_without_coverage_upgrade(tmp_path):
    def transport(request, token):
        response = reply(request)
        payload = json.loads(response.content)
        payload['Bars'][0]['High'] = 'NaN'
        response.content = json.dumps(payload).encode()
        return response
    assert run(tmp_path, transport)['status'] == 'COMPLETE'
    archive = json.loads((tmp_path / 'out/archive.json').read_bytes())
    assert validate_archive(archive)['counts']['invalid_ohlcv'] == 9
    assert not any(row['usable_for_calendar_coverage'] for row in validate_archive(archive)['rows'])


def test_revisions_and_unknown_flags_retained_for_validator(tmp_path):
    def transport(request, token):
        response = reply(request)
        payload = json.loads(response.content)
        original = payload['Bars'][0]
        original.pop('IsEndOfHistory')
        original.pop('BarStatus')
        payload['Bars'].append(dict(original, Close='1.5'))
        response.content = json.dumps(payload).encode()
        return response
    assert run(tmp_path, transport)['status'] == 'COMPLETE'
    archive = json.loads((tmp_path / 'out/archive.json').read_bytes())
    report = validate_archive(archive)
    assert report['counts']['revision'] == 9
    assert report['counts']['unknown_end_of_history_flag'] == 18
    assert report['counts']['partial_or_unknown_bar_status'] == 18
    assert not any(row['usable_for_calendar_coverage'] for row in report['rows'])
    assert len(report['rows']) == 18
    assert all(entry['verified'] for entry in report['archive_byte_integrity'])


def test_pagination_token_is_preserved_as_control_evidence(tmp_path):
    def transport(request, token):
        response = reply(request)
        payload = json.loads(response.content)
        payload['NextToken'] = 'opaque-page-cursor'
        response.content = json.dumps(payload).encode()
        return response
    result = run(tmp_path, transport)
    assert result['blockers'] == ['unknown_response_fields_or_pagination']
    evidence = json.loads(next((tmp_path / 'out').glob('*-result.json')).read_bytes())['envelope']
    assert evidence['response']['NextToken'] == 'opaque-page-cursor'
    assert evidence['raw_response_base64']


@pytest.mark.parametrize('raw', [b'{"Bars": [{"Open": 1e400}]}', b'{"Bars": [NaN]}', b'{"Bars": [Infinity]}'])
def test_nonfinite_json_preserves_raw_with_null_parse(tmp_path, raw):
    response = SimpleNamespace(status_code=200, content=raw, headers={})
    assert run(tmp_path, lambda *args: response)['status'] == 'INCOMPLETE'
    record = json.loads(next((tmp_path / 'out').glob('*-result.json')).read_bytes())['envelope']
    assert base64.b64decode(record['raw_response_base64']) == raw
    assert record['response'] is None
    assert record['response_metadata']['parse_diagnostic'] == 'nonfinite_json_representation'
    assert not list((tmp_path / 'out').glob('*-error.json'))


def test_escaped_surrogate_preserved_and_scanner_safe(tmp_path):
    raw = b'{"Bars": [], "text": "\\ud800"}'
    assert run(tmp_path, lambda *args: SimpleNamespace(status_code=200, content=raw, headers={}))['status'] == 'INCOMPLETE'
    record = json.loads(next((tmp_path / 'out').glob('*-result.json')).read_bytes())['envelope']
    assert base64.b64decode(record['raw_response_base64']) == raw
    assert record['response']['text'] == '\ud800'


def test_directory_entries_synced_before_dispatch(tmp_path, monkeypatch):
    import os
    import stat
    from pathlib import Path
    from src.research.yank_deployed_validation import acquire as module
    events = []
    synced_directories = []
    fsync, link = os.fsync, os.link
    def synced(fd):
        events.append('directory-sync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file-sync')
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced_directories.append(Path(os.readlink(f'/proc/self/fd/{fd}')))
        return fsync(fd)
    def linked(source, target):
        events.append(('link', str(target)))
        return link(source, target)
    monkeypatch.setattr(module.os, 'fsync', synced)
    monkeypatch.setattr(module.os, 'link', linked)
    def transport(request, token):
        assert tmp_path / 'out' in synced_directories
        assert tmp_path in synced_directories
        assert events[-1] == 'directory-sync'
        latest_link = next(event for event in reversed(events) if isinstance(event, tuple))
        assert latest_link[1].endswith('-started.json')
        assert events.index('directory-sync') < events.index(latest_link)
        return reply(request, 401)
    run(tmp_path, transport)


@pytest.mark.parametrize('changed_at', [0, 1])
def test_credential_pin_checked_each_attempt(tmp_path, changed_at):
    calls = []
    supplied = []
    def provider():
        token = 'test-only-token' if len(supplied) < changed_at else 'different-token'
        supplied.append(token)
        return token
    def transport(request, token):
        calls.append(request)
        return reply(request)
    result = acquire(gate(), tmp_path / 'out', token_provider=provider, transport=transport, clock=FakeClock())
    assert result['status'] == 'BLOCKED' and result['blockers'] == ['credential_changed_or_not_reviewed']
    assert len(calls) == changed_at
    assert json.loads((tmp_path / 'out/status.json').read_text())['status'] == 'BLOCKED'
    assert 'different-token' not in ''.join(p.read_text() for p in (tmp_path / 'out').iterdir())


def test_missing_or_unbound_credential_pin_blocks():
    declaration = gate()
    del declaration['credential_sha256']
    assert gate_blockers(declaration, FakeClock().now())
    declaration = gate()
    declaration['evidence_references']['zero_cost']['credential_sha256'] = 'a' * 64
    assert 'evidence not pinned to reviewed credential: zero_cost' in gate_blockers(declaration, FakeClock().now())


def test_future_review_injected_clock_skew(tmp_path):
    declaration = gate()
    declaration['evidence_references']['zero_cost']['reviewed_at'] = '2026-09-10T00:01:01Z'
    def forbidden(*args):
        pytest.fail('future review must block token/network')
    result = acquire(declaration, tmp_path / 'out', clock=FakeClock(), token_provider=forbidden, transport=forbidden)
    assert result['blockers'] == ['future evidence review time: zero_cost']
    declaration['evidence_references']['zero_cost']['reviewed_at'] = '2026-09-10T00:01:00Z'
    assert not gate_blockers(declaration, FakeClock().now())


def test_total_http_deadline_cancels_body_and_closes_client(monkeypatch):
    import asyncio
    from src.research.yank_deployed_validation import acquire as module
    events = []
    class Client:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            events.append('closed')
        async def get(self, *args, **kwargs):
            try:
                events.append('body-download')
                await asyncio.Event().wait()
            finally:
                events.append('cancelled')
    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    monkeypatch.setattr(module, 'REQUEST_TIMEOUT_SECONDS', 0.01)
    with pytest.raises(TimeoutError):
        http_get(request_ledger()['requests'][0], 'test-only-token')
    assert events == ['body-download', 'cancelled', 'closed']


@pytest.mark.parametrize('delay', ['4000', '1e400', 'Thu, 10 Sep 2026 03:00:00 GMT'])
def test_huge_retry_delay_stops_without_early_retry(tmp_path, delay):
    clock = FakeClock()
    calls = []
    def transport(request, token):
        calls.append(request)
        return reply(request, 429, {'Retry-After': delay, 'Set-Cookie': 'never-persist'})
    result = run(tmp_path, transport, clock=clock)
    assert result['status'] == 'INCOMPLETE' and 'run_budget_exhausted' in result['blockers'][0]
    assert len(calls) == 1 and clock.elapsed == 0
    record = json.loads(next((tmp_path / 'out').glob('*-result.json')).read_bytes())['envelope']
    controls = record['response_metadata']['control_flags']
    assert controls['delay_exceeds_run_budget'] and controls['retry_after_present']
    assert controls['run_deadline_at'] == '2026-09-10T01:00:00Z'
    assert 'never-persist' not in json.dumps(record)


@pytest.mark.parametrize('rate_limited,expiry_seconds,expected_elapsed', [(False, 100, 60), (True, 150, 120)])
def test_real_cache_expires_during_spacing_retains_evidence(tmp_path, rate_limited, expiry_seconds, expected_elapsed):
    clock = FakeClock()
    cache = tmp_path / 'shared.json'
    cache.write_text(json.dumps(dict(access_token='test-only-token',
        expires_at=(clock.now() + timedelta(seconds=expiry_seconds)).isoformat())))
    original = cache.read_bytes()
    calls = []
    def transport(request, token):
        calls.append(request)
        return reply(request, 429 if rate_limited else 200, {'Retry-After': '120'} if rate_limited else {})
    result = acquire(gate(), tmp_path / 'out', clock=clock,
                     token_provider=cache_token_provider(cache, clock), transport=transport)
    assert result['status'] == 'BLOCKED'
    persisted = json.loads((tmp_path / 'out/status.json').read_bytes())
    assert persisted['status'] == 'BLOCKED'
    assert persisted['completed_requests'] == (0 if rate_limited else 1)
    assert len(calls) == 1 and clock.elapsed == expected_elapsed
    assert len(list((tmp_path / 'out').glob('*-result.json'))) == 1
    assert cache.read_bytes() == original
