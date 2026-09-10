"""Offline TradeStation pilot planning and lossless archive diagnostics.

Input: JSON list of envelopes with ``request``, ``response_metadata`` and raw
``response`` (a provider JSON object containing ``Bars``). No client/auth imports.
A pinned exchange calendar is deliberately required before coverage can pass.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote

UTC = timezone.utc
MINUTE = timedelta(minutes=1)
START = datetime(2025, 4, 1, tzinfo=UTC)
EVALUATION = datetime(2025, 5, 19, tzinfo=UTC)
END = datetime(2025, 5, 31, tzinfo=UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('timezone required; local DST interpretation is ambiguous')
    return result.astimezone(UTC)


def request_ledger(symbol='MNQM25', minimum_spacing_seconds=60):
    """Proposed half-open UTC windows; provider spelling remains an approval gate."""
    if not symbol or symbol.startswith('@'):
        raise ValueError('explicit contract required; continuous symbols prohibited')
    if minimum_spacing_seconds < 1:
        raise ValueError('positive request spacing required')
    windows = []
    first = START
    while first < END:
        last = min(first + timedelta(days=7), END)
        windows.append(dict(request_id=f'may2025-{len(windows)+1:03d}', symbol=symbol,
                            interval=1, unit='Minute', method='GET',
                            endpoint='/v3/marketdata/barcharts/' + quote(symbol, safe=''),
                            query=dict(interval=1, unit='Minute', firstdate=iso(first), lastdate=iso(last)),
                            provider_boundary_inclusion='unverified', first_inclusive=iso(first),
                            last_exclusive=iso(last), earliest_offset_seconds=len(windows)*minimum_spacing_seconds))
        if last == END:
            break
        first = last - MINUTE
    return dict(schema_version=1, decision='HOLD_VALIDATION', mode='local-only-proposal',
                archive=dict(first_inclusive=iso(START), last_exclusive=iso(END)),
                evaluation=dict(first_inclusive=iso(EVALUATION), last_exclusive=iso(END)),
                overlap_minutes=1, maximum_window_days=7,
                rate_schedule=dict(minimum_spacing_seconds=minimum_spacing_seconds,
                                   documented_requests_per_5_minutes=500, documented_max_bars=57600,
                                   documentation_reference='docs/yank-tradestation-semantics.md',
                                   provider_limit_verified=True, retries='not scheduled; gate remains closed'),
                gates=dict(symbol_verified=False, expiration_verified=False, session_calendar_verified=False,
                           label_semantics_verified=False, entitlement_verified=False, cost_verified=False,
                           acquisition_authorized=False), authorized_spend_usd=0, requests=windows)


def validate_archive(envelopes, expected_calendar=None):
    """Retain every raw envelope; report ambiguity without normalizing it away."""
    if not isinstance(envelopes, list):
        raise ValueError('archive must be a list of request/response envelopes')
    expected = None
    if expected_calendar is not None:
        if not isinstance(expected_calendar, dict) or not expected_calendar.get('symbol'):
            raise ValueError('independent calendar requires symbol and minute_starts')
        minutes = expected_calendar.get('minute_starts')
        if not isinstance(minutes, list) or not minutes:
            raise ValueError('independent calendar minute_starts must be nonempty')
        expected = {timestamp(value) for value in minutes}
        if len(expected) != len(minutes) or any(t.second or t.microsecond or not START <= t < END for t in expected):
            raise ValueError('calendar minutes must be unique aligned UTC instants within archive window')
        if expected_calendar.get('label_semantics') not in (None, 'start', 'end'):
            raise ValueError('calendar label_semantics must be start, end or absent')
    findings, rows = [], []
    integrity = []
    usable = {'start': set(), 'end': set()}
    revised = set()
    seen = {}
    coverage = {'warmup': set(), 'evaluation': set()}
    def flag(kind, chunk, row=None):
        findings.append(dict(kind=kind, chunk=chunk, row=row))
    for chunk, envelope in enumerate(envelopes):
        if not isinstance(envelope, dict):
            flag('malformed_envelope', chunk)
            continue
        chunk_findings_start = len(findings)
        request = envelope.get('request', {})
        metadata = envelope.get('response_metadata', {})
        if not isinstance(request, dict) or not isinstance(metadata, dict):
            flag('malformed_metadata', chunk)
            request, metadata = {}, {}
        required_request = ('request_id', 'symbol', 'interval', 'unit', 'first_inclusive', 'last_exclusive', 'started_at')
        required_response = ('received_at', 'http_status', 'raw_sha256')
        if any(request.get(k) is None for k in required_request) or any(metadata.get(k) is None for k in required_response):
            flag('missing_request_response_metadata', chunk)
        if request.get('interval') != 1 or request.get('unit') != 'Minute':
            flag('wrong_interval', chunk)
        if not request.get('symbol') or str(request.get('symbol')).startswith('@'):
            flag('explicit_contract_unverified', chunk)
        if expected_calendar is not None and request.get('symbol') != expected_calendar['symbol']:
            flag('wrong_calendar_contract', chunk)
        if metadata.get('http_status') != 200:
            flag('http_response_not_successful', chunk)
        try:
            receipt = timestamp(metadata['received_at'])
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            receipt = None
            flag('invalid_receipt_timestamp', chunk)
        try:
            request_first = timestamp(request['first_inclusive'])
            request_last = timestamp(request['last_exclusive'])
            started = timestamp(request['started_at'])
            if request_last <= request_first or request_last-request_first > timedelta(days=7):
                flag('invalid_request_window', chunk)
            if receipt is not None and receipt < started:
                flag('receipt_precedes_request', chunk)
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            request_first = request_last = None
            flag('invalid_request_timestamps', chunk)
        response = envelope.get('response', {})
        raw_verified = False
        try:
            raw_bytes = base64.b64decode(envelope['raw_response_base64'], validate=True)
            if hashlib.sha256(raw_bytes).hexdigest() != metadata.get('raw_sha256'):
                raise ValueError('raw hash mismatch')
            if json.loads(raw_bytes) != response:
                raise ValueError('parsed response mismatch')
            raw_verified = True
        except (KeyError, ValueError, TypeError, binascii.Error, UnicodeError):
            flag('raw_response_integrity_unverified', chunk)
        integrity.append(dict(chunk=chunk, verified=raw_verified))
        chunk_valid = raw_verified and len(findings) == chunk_findings_start
        bars = response.get('Bars') if isinstance(response, dict) else None
        if not isinstance(bars, list):
            flag('missing_bars_array', chunk)
            continue
        previous = None
        for index, bar in enumerate(bars):
            row_findings_start = len(findings)
            record = dict(chunk=chunk, row=index, raw=bar)
            rows.append(record)
            if not isinstance(bar, dict):
                flag('malformed_bar', chunk, index)
                continue
            try:
                label = timestamp(bar['TimeStamp'])
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                flag('invalid_or_dst_ambiguous_timestamp', chunk, index)
                continue
            try:
                record['interval_candidates'] = dict(start_label=[iso(label), iso(label+MINUTE)],
                                                     end_label=[iso(label-MINUTE), iso(label)])
            except OverflowError:
                flag('unrepresentable_interval_candidates', chunk, index)
                continue
            record['label_semantics'] = 'unresolved'
            record['candidate_scope'] = {}
            for interpretation, interval_start in [('start', label), ('end', label-MINUTE)]:
                within = START <= interval_start < END
                record['candidate_scope'][interpretation] = dict(
                    within_archive=within,
                    phase=('warmup' if interval_start < EVALUATION else 'evaluation') if within else None)
            if request_first is not None and not request_first <= label <= request_last:
                flag('outside_request_window_both_label_candidates', chunk, index)
            if label.second or label.microsecond:
                flag('off_minute_timestamp', chunk, index)
            if previous is not None and label < previous:
                flag('out_of_order', chunk, index)
            previous = label
            key = (request.get('symbol'), iso(label))
            values = {k: bar.get(k) for k in ('Open', 'High', 'Low', 'Close', 'TotalVolume', 'BarStatus', 'IsEndOfHistory', 'IsClosed')}
            if key in seen:
                flag('duplicate' if seen[key] == values else 'revision', chunk, index)
                if seen[key] != values:
                    revised.add(key)
            seen[key] = values
            try:
                prices = {name: float(bar[name]) for name in ('Open', 'High', 'Low', 'Close', 'TotalVolume')}
                if not all(math.isfinite(v) for v in prices.values()) or prices['TotalVolume'] < 0:
                    raise ValueError('nonfinite price or invalid volume')
                if not prices['Low'] <= min(prices['Open'], prices['Close']) <= max(prices['Open'], prices['Close']) <= prices['High']:
                    raise ValueError('inconsistent OHLC')
            except (KeyError, ValueError, TypeError, OverflowError):
                flag('invalid_ohlcv', chunk, index)
            status, end = bar.get('BarStatus'), bar.get('IsEndOfHistory')
            if status != 'Closed':
                flag('partial_or_unknown_bar_status', chunk, index)
            if not isinstance(end, bool):
                flag('unknown_end_of_history_flag', chunk, index)
            if status != 'Closed' and end is True:
                flag('historical_end_with_partial_or_unknown_bar', chunk, index)
            if 'IsClosed' in bar and type(bar['IsClosed']) is not bool:
                flag('malformed_completion_indicator', chunk, index)
            if isinstance(bar.get('IsClosed'), bool) and status in ('Open', 'Closed') and bar['IsClosed'] != (status == 'Closed'):
                flag('conflicting_completion_flags', chunk, index)
            if receipt is not None and label > receipt:
                flag('future_label_at_receipt', chunk, index)
            elif receipt is not None and label + MINUTE > receipt:
                flag('start_label_interval_incomplete_at_receipt', chunk, index)
            if not any(scope['within_archive'] for scope in record['candidate_scope'].values()):
                flag('outside_archive_window_both_candidates', chunk, index)
            if START <= label < END and not label.second and not label.microsecond:
                coverage['warmup' if label < EVALUATION else 'evaluation'].add(label)
            row_valid = all(f['kind'] in ('duplicate', 'start_label_interval_incomplete_at_receipt') for f in findings[row_findings_start:])
            candidate_valid = {}
            for interpretation, interval_start in [('start', label), ('end', label-MINUTE)]:
                eligible = bool(chunk_valid and row_valid and START <= interval_start < END
                                and receipt is not None and interval_start+MINUTE <= receipt)
                candidate_valid[interpretation] = eligible
                if eligible:
                    usable[interpretation].add(interval_start)
            record['usable_by_interpretation'] = candidate_valid
            record['usable_for_calendar_coverage'] = any(candidate_valid.values())
    usable = {'start': set(), 'end': set()}
    for record in rows:
        if 'interval_candidates' in record and (envelopes[record['chunk']]['request'].get('symbol'), record['interval_candidates']['start_label'][0]) in revised:
            record['usable_for_calendar_coverage'] = False
            record['usable_by_interpretation'] = {'start': False, 'end': False}
        for interpretation, admitted in record.get('usable_by_interpretation', {}).items():
            if admitted: usable[interpretation].add(timestamp(record['interval_candidates'][interpretation + '_label'][0]))
    summaries = {}
    candidate_minutes = usable
    for name, first, last in [('warmup', START, EVALUATION), ('evaluation', EVALUATION, END)]:
        labels = sorted(coverage[name])
        gaps = []
        cursor = first
        for label in labels:
            if label > cursor:
                gaps.append(dict(first_inclusive=iso(cursor), last_exclusive=iso(label), minutes=int((label-cursor)/MINUTE)))
            cursor = label + MINUTE
        if cursor < last:
            gaps.append(dict(first_inclusive=iso(cursor), last_exclusive=iso(last), minutes=int((last-cursor)/MINUTE)))
        summaries[name] = dict(status='insufficient' if not labels else 'unverified',
                               unique_labels=len(labels), calendar_minutes=int((last-first)/MINUTE),
                               calendar_gap_intervals=gaps,
                               reason='Exchange session/calendar and interval labels unresolved; calendar gaps include scheduled closures')
        if expected is not None:
            needed = {t for t in expected if first <= t < last}
            candidates = {}
            for interpretation, available in candidate_minutes.items():
                missing = needed - available
                candidates[interpretation] = dict(status='PASS' if needed and not missing else 'INSUFFICIENT',
                                                  expected_minutes=len(needed), covered_minutes=len(needed & available),
                                                  missing_minutes=[iso(t) for t in sorted(missing)])
            summaries[name]['interval_candidates'] = candidates
            chosen = expected_calendar.get('label_semantics')
            summaries[name]['status'] = candidates[chosen]['status'] if chosen else 'unverified_label_semantics'
            summaries[name]['reason'] = 'Coverage against independently supplied expected exchange minute starts; full feature readiness remains separate'
    admission = 'UNKNOWN'
    if expected_calendar is not None and expected_calendar.get('label_semantics'):
        admission = 'PASS' if all(s['status'] == 'PASS' for s in summaries.values()) else 'FAIL'
    return dict(schema_version=1, decision='HOLD_VALIDATION', raw_envelopes=envelopes,
                calendar_coverage_admission=admission,
                expected_calendar_sha256=hashlib.sha256(json.dumps(expected_calendar, sort_keys=True, separators=(',', ':')).encode()).hexdigest() if expected_calendar is not None else None,
                independently_supplied_label_semantics=expected_calendar.get('label_semantics') if expected_calendar else None,
                rows=rows, findings=findings, counts=dict(sorted(Counter(x['kind'] for x in findings).items())),
                coverage=summaries, warmup_readiness='not_established: H1 ATR, LR and daily-range checks required',
                archive_byte_integrity=integrity,
                historical_arrival_evidence=False, label_semantics='unresolved_start_and_end_candidates_retained')



def export_replay(envelopes, output_directory, expected_calendar=None):
    """Write conditional polls-v1 diagnostics; synthetic receipts never gain trust."""
    report = validate_archive(envelopes, expected_calendar)
    records = [r for r in report['rows'] if r.get('usable_for_calendar_coverage')]
    symbols = {envelopes[r['chunk']]['request']['symbol'] for r in records}
    if len(symbols) != 1 or not records:
        raise ValueError('replay bridge requires usable raw-verified bars for exactly one contract')
    out = Path(output_directory)
    if out.exists() or out.is_symlink():
        raise ValueError('fresh replay output directory required')
    out.mkdir(parents=True)
    def write_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    write_json(out/'archive-validation.json', report)
    manifests = {}
    for interpretation in ('start', 'end'):
        folder = out/interpretation
        folder.mkdir()
        events = {}
        for record in records:
            if not record['usable_by_interpretation'][interpretation]:
                continue
            raw = record['raw']
            label = timestamp(raw['TimeStamp']) - (MINUTE if interpretation == 'end' else timedelta())
            if not START <= label < END:
                continue
            normalized = dict(raw, TimeStamp=iso(label))
            # Preserve raw flags in archive-validation.json. The unchanged poll
            # adapter receives a synthetic finalized stream, not actual requests.
            event = dict(receipt_time=iso(label+MINUTE), request_id=f"synthetic-{interpretation}-{iso(label)}",
                         status_code=200, bars=[normalized],
                         historical_arrival_evidence=False, receipt_classification='SYNTHETIC_INTERVAL_END',
                         interval_interpretation=interpretation, interval_semantics_verified=False,
                         phase='warmup' if label < EVALUATION else 'evaluation',
                         raw_reference=dict(chunk=record['chunk'], row=record['row']))
            events.setdefault(label, event)
        ordered = [events[t] for t in sorted(events)]
        archive_bytes = (out/'archive-validation.json').read_bytes()
        (folder/'archive-validation.json').write_bytes(archive_bytes)
        files = [dict(file='archive-validation.json', sha256=hashlib.sha256(archive_bytes).hexdigest())]
        for filename, selected in [('polls.jsonl', ordered),
                                   ('warmup.jsonl', [e for e in ordered if e['phase'] == 'warmup']),
                                   ('evaluation.jsonl', [e for e in ordered if e['phase'] == 'evaluation'])]:
            data = ''.join(json.dumps(e, sort_keys=True, allow_nan=False)+'\n' for e in selected).encode()
            (folder/filename).write_bytes(data)
            files.append(dict(file=filename, sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(schema_version=1, classification='DEVELOPMENT_2025', format='polls-v1',
                        source=dict(provider='TradeStation', classification='CONDITIONAL_HISTORICAL_DIAGNOSTIC',
                                    archive_report_sha256=hashlib.sha256((out/'archive-validation.json').read_bytes()).hexdigest()),
                        request=dict(classification='SYNTHETIC_SINGLE_BAR_POLLS_NOT_HISTORICAL_REQUESTS'),
                        files=files, contracts=dict(source_symbol=next(iter(symbols)), deployed_specs_symbol='MNQU26'),
                        adjustments=dict(status='UNVERIFIED'),
                        interval=dict(start=iso(START), end_exclusive=iso(END+MINUTE),
                                      archive_end_exclusive=iso(END), evaluation_start=iso(EVALUATION)),
                        availability=dict(classification='SYNTHETIC_INTERVAL_END', historical_arrival_evidence=False,
                                          conditional_interval_interpretation=interpretation, interval_semantics_verified=False),
                        session=dict(status='UNVERIFIED' if expected_calendar is None else 'INDEPENDENT_CALENDAR_SUPPLIED'),
                        coverage=report['coverage'],
                        warmup=dict(status='FEATURE_READINESS_UNVERIFIED', first_inclusive=iso(START),
                                    last_exclusive=iso(EVALUATION), stream='warmup.jsonl', evaluation_stream='evaluation.jsonl'),
                        execution_state=dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT', daily_pnl=0.,
                                             daily_halted=False, last_trading_date=None, on_combine=True, is_backfill=True),
                        decision='HOLD_VALIDATION')
        write_json(folder/'manifest.json', manifest)
        manifests[interpretation] = str(folder/'manifest.json')
    return dict(decision='HOLD_VALIDATION', manifests=manifests, historical_arrival_evidence=False,
                warning='Conditional synthetic replay only; no account, live identity, arrival or feature readiness upgrade')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ledger = commands.add_parser('ledger')
    ledger.add_argument('--symbol', default='MNQM25')
    ledger.add_argument('--minimum-spacing-seconds', type=int, default=60)
    validate = commands.add_parser('validate')
    validate.add_argument('archive', type=Path)
    validate.add_argument('--expected-calendar', type=Path, help='Independent JSON with symbol, minute_starts and optional label_semantics start/end')
    bridge = commands.add_parser('bridge')
    bridge.add_argument('archive', type=Path)
    bridge.add_argument('--output-directory', type=Path, required=True)
    bridge.add_argument('--expected-calendar', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'ledger':
        result = request_ledger(args.symbol, args.minimum_spacing_seconds)
    else:
        raw = args.archive.read_bytes()
        calendar = json.loads(args.expected_calendar.read_bytes()) if args.expected_calendar else None
        if args.command == 'bridge':
            result = export_replay(json.loads(raw), args.output_directory, calendar)
        else:
            result = validate_archive(json.loads(raw), calendar)
        result['input_archive_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
