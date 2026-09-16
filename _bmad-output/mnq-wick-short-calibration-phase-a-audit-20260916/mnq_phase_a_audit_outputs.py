"""Check persisted Phase A ledgers independently of its implementation."""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from mnq_phase_a_independent_audit import assert_number, reference_summary


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit_metric(metric, values, days):
    desc = metric['descriptive']
    assert desc['observation_count'] == len(values)
    assert_number(desc['total'], sum(values), 'total')
    results = []
    for key, groups in [
        ('session_clustered', days),
        ('iso_week_clustered', [tuple(date.fromisoformat(d).isocalendar()[:2]) for d in days]),
    ]:
        expected = reference_summary(values, groups)
        actual = metric[key]
        assert actual['N'] == expected['n'] and actual['G'] == expected['g']
        assert_number(desc['mean'], expected['mean'], 'mean')
        assert_number(desc['sample_sd'], expected['sd'], 'sd')
        assert_number(actual['variance'], expected['variance'], 'variance')
        assert sorted(row['count'] for row in actual['cluster_sizes']) == sorted(Counter(groups).values())
        if expected['ci'] is None:
            assert actual['status'] == 'UNASSESSABLE' and actual['interval'] is None
        else:
            assert actual['status'] == 'ASSESSABLE'
            assert actual['degrees_of_freedom'] == expected['g'] - 1
            assert_number(actual['se'], expected['se'], 'SE')
            for a, b in zip(actual['interval'], expected['ci']):
                assert_number(a, b, 'interval endpoint')
        results.append(expected['ci'])
    if all(result is not None for result in results):
        envelope = [min(r[0] for r in results), max(r[1] for r in results)]
        for a, b in zip(metric['envelope']['interval'], envelope):
            assert_number(a, b, 'envelope endpoint')
    else:
        assert metric['envelope']['interval'] is None
    frequencies = Counter(values)
    ecdf = desc['distribution']['ecdf']
    assert [(r['value'],r['count']) for r in ecdf] == sorted(frequencies.items())
    ordered = sorted(values)
    for quantile, actual in desc['distribution']['quantiles'].items():
        index = float(quantile) * (len(ordered) - 1)
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        expected = ordered[low] + (index-low) * (ordered[high] - ordered[low])
        assert_number(actual, expected, 'quantile')


def audit(output):
    complete = json.loads((output/'COMPLETE.json').read_text())
    assert complete['evaluation_allowed'] is False
    assert complete['manifest_sha256'] == hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()
    manifest = json.loads((output/'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == digest, name
    report = json.loads((output/'report.json').read_text())
    assert report['original_gate_verdict']=='POWER_UNDETERMINED'
    assert report['evaluation_allowed'] is False and report['confirmation_authorized'] is False
    eligibility = read_rows(output/'eligibility.jsonl')
    outcomes = read_rows(output/'outcomes.jsonl')
    sessions = read_rows(output/'sessions.jsonl')
    assert (len(eligibility),len(outcomes),len(sessions)) == (576,585,515)
    assert len({row['signal_id'] for row in outcomes}) == len(outcomes)
    assert len({row['session_id'] for row in sessions}) == len(sessions)
    eligible = {row['session_id'] for row in eligibility if row['eligible']}
    assert eligible == {row['session_id'] for row in sessions}
    assert Counter(row['primary_exclusion_reason'] for row in eligibility if not row['eligible']) == {'INCOMPLETE_RTH_MINUTES':23,'MIXED_CONTRACT':38}
    by_day=defaultdict(list)
    for row in outcomes:
        assert row['sample_role']=='calibration-development-only'
        slot=row['signal_slot']
        assert 0 <= slot <= 75 and row['following_slot']==slot+1
        label=datetime.fromisoformat(row['signal_bar_label_local'])
        assert label.hour*60+label.minute==9*60+35+5*slot
        assert datetime.fromisoformat(row['reference_interval_start_local']) == label
        assert datetime.fromisoformat(row['reference_interval_end_local']) == label+timedelta(minutes=5)
        assert label.date().isoformat() == row['session_id']
        for key, offset in [('signal_component_minutes',-4),('reference_component_minutes',1)]:
            assert len(row[key])==5
            for index,minute in enumerate(row[key]):
                stamp=datetime.fromisoformat(minute['minute_label_local'])
                assert stamp==label+timedelta(minutes=offset+index)
                assert datetime.fromisoformat(minute['minute_label_utc'])==stamp
                assert minute['contract']==row['contract']
                assert stamp.astimezone(timezone.utc)<datetime(2026,3,1,tzinfo=timezone.utc)
        parts=row['signal_component_minutes']
        opening, close = parts[0]['open'], parts[-1]['close']
        high, low=max(x['high'] for x in parts),min(x['low'] for x in parts)
        assert row['body']==abs(close-opening)>0
        assert row['upper_wick']==high-max(opening,close)>=2*row['body']
        assert row['lower_wick']==min(opening,close)-low<=.1*row['upper_wick']
        assert row['next_open']==row['reference_component_minutes'][0]['open']
        assert row['next_close']==row['reference_component_minutes'][-1]['close']
        assert_number(row['gross_dollars'],2*(row['next_open']-row['next_close']),'short dollars')
        for cost, value in row['net_dollars'].items():
            assert_number(value,row['gross_dollars']-float(cost),'net dollars')
        by_day[row['session_id']].append(row)
    assert sum(not row['signal_count'] for row in sessions)==164
    for row in sessions:
        signals=by_day[row['session_id']]
        assert row['signal_count']==len(signals)
        gross=sum(r['gross_dollars'] for r in signals)
        assert_number(row['gross_total_dollars'],gross,'session gross')
        for cost, value in row['net_total_dollars'].items():
            assert_number(value,gross-float(cost)*len(signals),'session net')
    signal_days=[row['session_id'] for row in outcomes]
    session_days=[row['session_id'] for row in sessions]
    stats=report['statistics']
    audit_metric(stats['per_signal']['gross_dollars'],[r['gross_dollars'] for r in outcomes],signal_days)
    audit_metric(stats['per_eligible_session']['signal_count'],[r['signal_count'] for r in sessions],session_days)
    audit_metric(stats['per_eligible_session']['gross_total_dollars'],[r['gross_total_dollars'] for r in sessions],session_days)
    for cost in ('1.22','2.22','3.22'):
        audit_metric(stats['per_signal']['net_dollars'][cost],[r['net_dollars'][cost] for r in outcomes],signal_days)
        audit_metric(stats['per_eligible_session']['net_total_dollars'][cost],[r['net_total_dollars'][cost] for r in sessions],session_days)
    print('INDEPENDENT AUDIT PASSED: output hashes, 576 eligibility rows, 585 intervals/outcomes, 515 session totals, 164 zero-signal sessions, and all nine estimands with both intervals/envelope/distributions.')
    print(json.dumps({name:metric['descriptive']['mean'] for name,metric in stats['per_signal']['net_dollars'].items()}))


if __name__=='__main__':
    audit(Path(sys.argv[1]))
