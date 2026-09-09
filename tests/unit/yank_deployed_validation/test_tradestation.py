import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.research.yank_deployed_validation.tradestation import (
    END, EVALUATION, MINUTE, START, request_ledger, timestamp, validate_archive,
)


def envelope(bars):
    return dict(request=dict(request_id='one', symbol='MNQM25', interval=1, unit='Minute',
                             first_inclusive='2025-05-19T00:00:00Z', last_exclusive='2025-05-26T00:00:00Z',
                             started_at='2026-09-09T00:00:00Z'),
                response_metadata=dict(received_at='2026-09-09T00:00:01Z', http_status=200, raw_sha256='unverified'),
                response=dict(Bars=bars))


def bar(label='2025-05-19T00:00:00Z', **changes):
    return dict(TimeStamp=label, Open='1', High='2', Low='1', Close='2', TotalVolume='7',
                BarStatus='Closed', IsEndOfHistory=False, **changes)


def test_ledger_exact_windows_overlap_and_closed_gates():
    ledger = request_ledger()
    assert ledger == request_ledger()
    requests = ledger['requests']
    assert timestamp(requests[0]['first_inclusive']) == START
    assert timestamp(requests[-1]['last_exclusive']) == END
    for i, request in enumerate(requests):
        assert (timestamp(request['last_exclusive']) - timestamp(request['first_inclusive'])).total_seconds() <= 7*86400
        assert request['earliest_offset_seconds'] == i*60
        if i:
            assert timestamp(request['first_inclusive']) == timestamp(requests[i-1]['last_exclusive'])-MINUTE
    assert timestamp(ledger['evaluation']['first_inclusive']) == EVALUATION
    assert not any(ledger['gates'].values())
    with pytest.raises(ValueError):
        request_ledger('@MNQ')


def test_raw_revisions_duplicates_flags_and_no_arrival_upgrade():
    original = bar()
    revised = dict(original, Close='1.5', BarStatus='Open', IsEndOfHistory=True, IsClosed=True)
    archive = [envelope([original, original, revised, bar('2025-05-19T00:02:00Z')])]
    saved = copy.deepcopy(archive)
    report = validate_archive(archive)
    assert archive == saved == report['raw_envelopes']
    assert report['counts']['revision'] == report['counts']['duplicate'] == 1
    assert report['counts']['conflicting_completion_flags'] == 1
    assert not report['historical_arrival_evidence']
    assert report['coverage']['warmup']['status'] == 'insufficient'
    assert report['coverage']['evaluation']['status'] == 'unverified'
    assert report['coverage']['evaluation']['calendar_gap_intervals'][0]['minutes'] == 1
    assert report['rows'][0]['interval_candidates']['end_label'] == ['2025-05-18T23:59:00Z', '2025-05-19T00:00:00Z']


def test_dst_offsets_naive_future_and_malformed():
    archive = [envelope([bar('2025-11-02T01:30:00'), bar('2025-11-02T01:30:00-04:00'),
                         bar('2025-11-02T01:30:00-05:00'), bar('2027-01-01T00:00:00Z'), None])]
    report = validate_archive(archive)
    assert report['counts']['invalid_or_dst_ambiguous_timestamp'] == 1
    assert report['counts']['future_label_at_receipt'] == 1
    assert report['counts']['malformed_bar'] == 1
    assert report['rows'][1]['interval_candidates'] != report['rows'][2]['interval_candidates']
    assert 'duplicate' not in report['counts']


def test_empty_and_missing_metadata_fail_closed():
    report = validate_archive([dict(response=dict(Bars=[])), None])
    assert report['counts']['missing_request_response_metadata'] == 1
    assert report['counts']['malformed_envelope'] == 1
    assert all(value['status'] == 'insufficient' for value in report['coverage'].values())


def test_cli_is_offline_and_deterministic(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = root/'src/research/yank_deployed_validation/tradestation.py'
    command = [sys.executable, str(script), 'ledger']
    assert subprocess.check_output(command) == subprocess.check_output(command)
    archive = tmp_path/'archive.json'
    archive.write_text(json.dumps([envelope([bar()])]))
    report = json.loads(subprocess.check_output([sys.executable, str(script), 'validate', str(archive)]))
    assert len(report['input_archive_sha256']) == 64
    assert report['decision'] == 'HOLD_VALIDATION'


def signed_raw_envelope(bars, first='2025-05-19T00:00:00Z', last='2025-05-26T00:00:00Z'):
    import base64
    import hashlib
    item = envelope(bars)
    item['request'].update(first_inclusive=first, last_exclusive=last)
    raw = json.dumps(item['response']).encode()
    item['raw_response_base64'] = base64.b64encode(raw).decode()
    item['response_metadata']['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    return item


def test_independent_calendar_and_raw_integrity_admit_only_usable_rows():
    warm = '2025-04-01T00:00:00Z'
    evaluation = '2025-05-19T00:00:00Z'
    calendar = dict(symbol='MNQM25', minute_starts=[warm, evaluation], label_semantics='start')
    archive = [signed_raw_envelope([bar(warm)], warm, '2025-04-08T00:00:00Z'),
               signed_raw_envelope([bar(evaluation)])]
    report = validate_archive(archive, calendar)
    assert report['calendar_coverage_admission'] == 'PASS'
    assert all(x['verified'] for x in report['archive_byte_integrity'])
    assert report['historical_arrival_evidence'] is False
    archive[1]['response']['Bars'][0]['Close'] = '999'
    failed = validate_archive(archive, calendar)
    assert failed['calendar_coverage_admission'] == 'FAIL'
    assert failed['coverage']['evaluation']['interval_candidates']['start']['covered_minutes'] == 0
    assert failed['counts']['raw_response_integrity_unverified'] == 1


def test_missing_hash_partial_revised_wrong_contract_do_not_cover():
    label = '2025-05-19T00:00:00Z'
    calendar = dict(symbol='MNQM25', minute_starts=[label], label_semantics='start')
    partial = signed_raw_envelope([dict(bar(label), BarStatus='Open')])
    revised = signed_raw_envelope([bar(label), dict(bar(label), Close='1.1')])
    wrong = signed_raw_envelope([bar(label)])
    wrong['request']['symbol'] = 'MNQU25'
    bad_hash = signed_raw_envelope([bar(label)])
    bad_hash['response_metadata']['raw_sha256'] = 'wrong'
    for archive in ([envelope([bar(label)])], [partial], [revised], [wrong], [bad_hash]):
        report = validate_archive(archive, calendar)
        assert report['coverage']['evaluation']['interval_candidates']['start']['covered_minutes'] == 0


def test_unresolved_label_semantics_reports_both_calendar_candidates():
    calendar = dict(symbol='MNQM25', minute_starts=['2025-05-19T00:00:00Z'])
    report = validate_archive([signed_raw_envelope([bar('2025-05-19T00:01:00Z')])], calendar)
    assert report['calendar_coverage_admission'] == 'UNKNOWN'
    candidates = report['coverage']['evaluation']['interval_candidates']
    assert candidates['start']['status'] == 'INSUFFICIENT'
    assert candidates['end']['status'] == 'PASS'
    request = request_ledger()['requests'][0]
    assert request['endpoint'] == '/v3/marketdata/barcharts/MNQM25'
    assert set(request['query']) == {'interval', 'unit', 'firstdate', 'lastdate'}
    assert request['provider_boundary_inclusion'] == 'unverified'


def test_bridge_existing_replay_admission_and_partitioned_synthetic_events(tmp_path):
    from src.research.yank_deployed_validation.tradestation import export_replay
    from src.research.yank_deployed_validation.replay import admit, events
    archive = [signed_raw_envelope([bar('2025-04-01T00:01:00Z')], '2025-04-01T00:00:00Z', '2025-04-08T00:00:00Z'),
               signed_raw_envelope([bar('2025-05-19T00:01:00Z')])]
    out = tmp_path/'bridge'
    result = export_replay(archive, out)
    assert result['historical_arrival_evidence'] is False
    for interpretation in ('start', 'end'):
        folder = out/interpretation
        manifest = admit(folder/'manifest.json', folder)
        stream = list(events(manifest, folder))
        assert len(stream) == 2
        assert [e['phase'] for e in stream] == ['warmup', 'evaluation']
        assert all(e['receipt_classification'] == 'SYNTHETIC_INTERVAL_END' for e in stream)
        assert len((folder/'warmup.jsonl').read_text().splitlines()) == 1
        assert manifest['execution_state']['classification'] == 'SYNTHETIC_UNKNOWN_ACCOUNT'
    with pytest.raises(ValueError):
        export_replay(archive, out)


def test_archive_and_evaluation_boundaries_are_candidate_specific(tmp_path):
    from src.research.yank_deployed_validation.tradestation import export_replay
    end = signed_raw_envelope([bar('2025-05-31T00:00:00Z')], '2025-05-24T00:00:00Z', '2025-05-31T00:00:00Z')
    start = signed_raw_envelope([bar('2025-04-01T00:00:00Z')], '2025-04-01T00:00:00Z', '2025-04-08T00:00:00Z')
    boundary = signed_raw_envelope([bar('2025-05-19T00:00:00Z')])
    report = validate_archive([end, start, boundary])
    first, second, third = report['rows']
    assert first['candidate_scope'] == {'start': {'within_archive': False, 'phase': None}, 'end': {'within_archive': True, 'phase': 'evaluation'}}
    assert first['usable_by_interpretation'] == {'start': False, 'end': True}
    assert second['usable_by_interpretation'] == {'start': True, 'end': False}
    assert third['candidate_scope']['start']['phase'] == 'evaluation'
    assert third['candidate_scope']['end']['phase'] == 'warmup'
    assert 'outside_archive_window' not in report['counts']
    out = tmp_path/'edge-bridge'
    export_replay([end, start, boundary], out)
    end_stream = [json.loads(line) for line in (out/'end'/'evaluation.jsonl').read_text().splitlines()]
    assert end_stream[-1]['bars'][0]['TimeStamp'] == '2025-05-30T23:59:00Z'


def test_interval_complete_at_receipt_is_candidate_specific():
    item = signed_raw_envelope([bar('2025-05-19T00:00:00Z')])
    item['request']['started_at'] = '2025-05-18T23:59:59Z'
    item['response_metadata']['received_at'] = '2025-05-19T00:00:00Z'
    row = validate_archive([item])['rows'][0]
    assert row['usable_by_interpretation'] == {'start': False, 'end': True}


def test_timestamp_overflow_and_invalid_optional_completion_are_reported():
    report = validate_archive([signed_raw_envelope([bar('9999-12-31T23:59:59Z'), bar('0001-01-01T00:00:00Z'), bar(IsClosed='false')])])
    assert report['counts']['unrepresentable_interval_candidates'] == 2
    assert report['counts']['malformed_completion_indicator'] == 1
    assert not any(r.get('usable_for_calendar_coverage') for r in report['rows'])
    offset = validate_archive([signed_raw_envelope([bar('0001-01-01T00:00:00+14:00')])])
    assert offset['counts']['invalid_or_dst_ambiguous_timestamp'] == 1


def test_other_contract_revision_does_not_suppress_selected_contract():
    label = '2025-05-19T00:00:00Z'
    chosen = signed_raw_envelope([bar(label)])
    other = signed_raw_envelope([bar(label), dict(bar(label), Close='1.5')])
    other['request']['symbol'] = 'MNQU25'
    report = validate_archive([chosen, other], dict(symbol='MNQM25', minute_starts=[label], label_semantics='start'))
    assert report['coverage']['evaluation']['interval_candidates']['start']['covered_minutes'] == 1
    assert report['rows'][0]['usable_by_interpretation']['start']
