"""Verify published measurement hashes and independent sparse-volume oracle."""
import hashlib
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
evidence = Path(__file__).resolve().parent
manifest = json.loads((run / 'manifest.json').read_text())
for name, expected in manifest['artifacts'].items():
    assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected, name
provenance = json.loads((run / 'provenance.json').read_text())
repo = evidence.parents[1]
for name, expected in provenance['code_sha256'].items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == expected, name
assert hashlib.sha256((repo / 'docs/valentini-native/calendar-2025-05.json').read_bytes()).hexdigest() == provenance['calendar_sha256']
oracle = json.loads((evidence / 'oracle.json').read_text())['minutes']
histograms = {str(r['start_ns']): r['tick_volumes'] for r in map(json.loads, (run / 'histograms.jsonl').open())}
saved_bars = {str(r['start_ns']): r for r in map(json.loads, Path('/root/Silver-Bullet-ML-BMAD-yank-minute/data/yank/native-minute-reviewed-a/bars.jsonl').open())}
for minute, expected in oracle.items():
    assert histograms[minute] == expected['tick_volumes'], minute
    bar = saved_bars[minute]
    assert expected['trade_count'] == bar['trade_count']
    assert expected['trade_sha256'] == bar['trade_sha256']
    assert sum(v for _, v in expected['tick_volumes']) == bar['ohlcv'][4]
report = json.loads((run / 'report.json').read_text())
sessions = json.loads((run / 'sessions.json').read_text())
rows = list(map(json.loads, (run / 'snapshots.jsonl').open()))
eligible = {r['session'] for r in sessions if r['eligible']}
assert all(row['session'] in eligible for row in rows)
compared = [row for row in rows if row['status'] == 'COMPARED']
assert len(compared) == report['profile_summary']['comparison_count']
assert all(row['available_ns'] <= row['boundary_ns'] and row['prior_bar_count'] > 0 for row in compared)
for index, level in enumerate(('VAL','VAH','POC')):
    differences = [r['proxy_ticks'][index] - r['native_ticks'][index] for r in compared]
    assert differences == [r['signed_proxy_minus_native_ticks'][index] for r in compared]
    stats = report['profile_summary']['levels'][level]
    assert stats['denominator'] == len(differences)
    assert stats['exact_agreement_count'] == differences.count(0)
    assert stats['absolute_ticks']['mean'] == sum(map(abs,differences)) / len(differences)
assert report['market_evaluation'] == 'NOT_ADMITTED'
for session in sessions:
    selected = [r for r in compared if r['session'] == session['session']]
    summary = report['session_profile_summaries'][session['session']]
    assert summary['comparison_count'] == len(selected)
    for i, level in enumerate(('VAL', 'VAH', 'POC')):
        expected = sum(r['absolute_ticks'][i] for r in selected) / len(selected) if selected else None
        actual = summary['levels'][level]['absolute_ticks']
        assert (actual['mean'] if actual else None) == expected
print(json.dumps({'artifact_hashes':'PASS','independent_histogram_oracle_minutes':len(oracle),'independent_oracle_trade_count':sum(x['trade_count'] for x in oracle.values()),'eligible_sessions':len(eligible),'comparison_count':len(compared),'snapshot_accounting':'PASS','availability':'PASS'},sort_keys=True))
