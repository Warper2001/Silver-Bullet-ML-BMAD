"""Validate complete measurement batches and extract separate profile attribution."""
import argparse
import json
from pathlib import Path
import pstats
import statistics

from observer_overhead import LIMITS, MODES, schedule


def validate(rows):
    assert len(rows) == 18, 'incomplete measurement batch'
    assert [(r['workload'], r['mode']) for r in rows] == [(w, m) for w, _, m in schedule()]
    assert len({r['pid'] for r in rows}) == 18, 'cells must use distinct fresh processes'
    assert all(not r['diagnostic'] and r['limits'] == LIMITS for r in rows)
    assert all(r['code_sha256'] == rows[0]['code_sha256'] for r in rows)
    for workload in ('startup', 'steady'):
        selected = [r for r in rows if r['workload'] == workload]
        assert all(r['fixture_sha256'] == selected[0]['fixture_sha256'] for r in selected)
        assert len({r['final_state_sha256'] for r in selected}) == 1, 'state mismatch'
        assert all(r['final_bar_count'] == (2880 if workload == 'startup' else 7500) for r in selected)
    result = {}
    for workload in ('startup', 'steady'):
        result[workload] = {}
        for mode in MODES:
            cells = [r for r in rows if (r['workload'], r['mode']) == (workload, mode)]
            result[workload][mode] = dict(
                wall_seconds=[r['poll']['wall_seconds'] for r in cells],
                median_wall_seconds=statistics.median(r['poll']['wall_seconds'] for r in cells),
                cpu_seconds=[r['poll']['cpu_seconds'] for r in cells],
                rss_high_water_kib=[r['process_high_water_rss_kib'] for r in cells],
                capture_bytes=[r['capture_bytes'] for r in cells],
                coverage=[r['coverage'] for r in cells])
        base = result[workload]['baseline']['median_wall_seconds']
        guard = result[workload]['guarded']['median_wall_seconds']
        result[workload]['guarded_minus_baseline_median_seconds'] = guard-base
        result[workload]['guarded_to_baseline_median_ratio'] = guard/base
    return result


def attribution(path):
    stats = pstats.Stats(str(path))
    run = json.loads(path.with_name('result.json').read_text())
    groups = {'extraction': [('adapter.py','state')],
              'bounded_copy': [('capture.py','copy')],
              'state_normalization': [('adapter.py','normalize')],
              'timestamp_derivation': [('~', "<method 'isoformat' of 'datetime.datetime' objects>"),
                                       ('~', "<method 'replace' of 'datetime.datetime' objects>")],
              'serialization': [('adapter.py','canonical'), ('__init__.py','loads')],
              'runtime_checks': [('startup.py','runtime_identity')],
              'storage_handoff': [('capture.py','emit')],
              'decision_record': [('capture.py','record')]}
    result = {}
    for group, wanted in groups.items():
        result[group] = [dict(file=filename, line=line, function=function, primitive_calls=cc,
            total_calls=nc, self_seconds=tt, cumulative_seconds=ct)
            for (filename, line, function), (cc,nc,tt,ct,_) in stats.stats.items()
            if (Path(filename).name,function) in wanted]
    return dict(scope='SEPARATE_DIAGNOSTIC_MAIN_THREAD_PROFILE_NOT_LATENCY',
        identity={key:run[key] for key in ('workload', 'git_head', 'code_sha256', 'fixture_sha256', 'profile_driver_sha256', 'profile_aggregation')},
        limitations=['Cumulative categories overlap and must not be summed.',
                     'Distinct compiled code objects sharing labels are aggregated; caller graph omitted.',
                     'Separate standard-lossy pstats is supplemental and can overwrite same-label code objects.',
                     'emit includes bounded producer work; writer-thread disk time is not profiled.'],
        attribution=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.before.read_text()); after = json.loads(args.after.read_text())
    result = dict(before=validate(before), after=validate(after))
    assert all(a['fixture_sha256'] == b['fixture_sha256'] and a['final_state_sha256'] == b['final_state_sha256']
               for a,b in zip(before,after)), 'before/after behavior or fixture mismatch'
    result['decision'] = 'HOLD_VALIDATION'
    result['limitations'] = ['Shared host descriptive timings, no numerical production latency target.',
        'RSS is absolute process high-water including libraries, fixtures and warmup, not observer-only peak.',
        'Queue bounds are configured limits, not continuously observed peaks.',
        'Fixed private synthetic shadow tail, synthetic account, and self-derived release expectations are not live evidence.']
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__': main()
