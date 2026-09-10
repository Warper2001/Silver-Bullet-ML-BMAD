"""Verify two completed local audits against frozen evidence; never runs acquisition."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = '84f2382ed238ff66f42c470bf8021c0fd7e4e4e2'
PILOT = 'd213110437374f8b6173aedc0b47dcc257adb5a7'
REPLAY = ROOT.parent / 'Silver-Bullet-ML-BMAD-yank-replay'
MAIN = ROOT.parent / 'Silver-Bullet-ML-BMAD'
NAMES = ('artifacts.json', 'event-extracts.jsonl', 'reconciliation.jsonl', 'report.json', 'report.md')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd)


def verify(first, second):
    original = ROOT / 'docs/reports/yank-execution-pilot/final-run1'
    saved_repeat = MAIN / 'docs/reports/yank-execution-pilot/final-run2'
    historical_hashes = read(original.parent / 'verification.json')['canonical_artifact_sha256']
    canonical_hashes = {}
    for name in NAMES:
        pinned = hashlib.sha256(git('show', ORIGINAL + ':' + str((original / name).relative_to(ROOT)))).hexdigest()
        require(sha(original / name) == sha(saved_repeat / name) == historical_hashes[name] == pinned,
                'historical artifact changed: ' + name)
        canonical_hashes[name] = sha(first / name)
        require(canonical_hashes[name] == sha(second / name), 'repeat mismatch: ' + name)
    report = read(first / 'report.json')
    old_report = read(original / 'report.json')
    require(report['audit_status'] == 'PASS_AUDIT_CHECKS' and report['research_status'] == 'HOLD_VALIDATION', 'audit status')
    for name, expected in read(first / 'artifacts.json').items():
        require(sha(first / name) == expected, 'artifact manifest mismatch: ' + name)
    changed_fields = sorted(k for k in set(report) | set(old_report) if report.get(k) != old_report.get(k))
    require(changed_fields == ['code_hashes'], 'unexpected findings differences: ' + str(changed_fields))
    code_hashes = report['code_hashes']
    for name, expected in code_hashes.items():
        require(sha(ROOT / name) == expected, 'audit code changed: ' + name)
    changed_code = sorted(name for name in code_hashes if code_hashes[name] != old_report['code_hashes'][name])
    require(changed_code == ['src/cli/check_yank_execution_pilot.py', 'src/research/yank_execution_pilot/audit.py'], 'unexpected code changes')
    unchanged_artifacts = [name for name in NAMES if sha(first / name) == sha(original / name)]
    require(unchanged_artifacts == ['event-extracts.jsonl', 'reconciliation.jsonl', 'report.md'], 'unexpected historical artifact differences')
    conditions = [s for case in report['cases'] for s in case['conditions']]
    require(len(report['cases']) == 5 and len(conditions) == 30, 'case/scenario coverage')
    source_orders = [(s['arm'], s['order_id']) for c in report['cases'] for s in c['sources']]
    require(len(source_orders) == len(set(source_orders)) == 8, 'arm order coverage')
    require(Counter(arm for arm, _ in source_orders) == {'no-ml': 4, 'ml050': 4}, 'arm coverage')
    for case in report['cases']:
        require({(s['convention'], s['delay_ms']) for s in case['conditions']} ==
                {(c, d) for c in ('start', 'end') for d in (0, 100, 500)}, 'conditional interpretations')
    require({w['convention'] for w in report['may28_timelines']} == {'start', 'end'}, 'May 28 interpretations')
    # Full report equality above preserves every schedule endpoint, gap and archived order history.
    archive = REPLAY / 'docs/reports/yank-signals/development-run1'
    manifest = read(archive / 'manifest.json')
    require(sha(archive / 'manifest.json') == report['input_hashes']['archive/manifest.json'], 'archive manifest changed')
    archive_hashes = {}
    for name, expected in manifest['artifacts'].items():
        archive_hashes[name] = sha(archive / name)
        require(archive_hashes[name] == expected, 'frozen archive changed: ' + name)
    for name, expected in report['input_hashes'].items():
        if name.startswith('acquisition/'):
            path = MAIN / 'data/yank/databento-pilot-20260907' / name.removeprefix('acquisition/')
        elif name == 'original_bars':
            path = MAIN / manifest['declared_manifest']['development_data']['path']
        elif name == 'archive/engine.py':
            path = REPLAY / 'src/research/yank_signals/engine.py'
        else:
            path = archive / name.removeprefix('archive/')
        require(sha(path) == expected, 'input changed: ' + name)
    parent = read(ROOT / 'docs/reports/yank-native-minute/parent-verification.json')
    for folder in parent['canonical_pair']:
        for name, expected in parent['canonical_sha256'].items():
            require(sha(ROOT / folder / name) == expected, 'minute artifact changed: ' + folder + '/' + name)
    pilot_revision = git('rev-parse', PILOT).decode().strip()
    paths = git('ls-tree', '-r', '--name-only', pilot_revision, '--', 'src/research/yank_native_minute',
                'src/cli/check_yank_native_minute.py', 'docs/reports/yank-native-minute').decode().splitlines()
    for path in paths:
        require(sha(ROOT / path) == hashlib.sha256(git('show', pilot_revision + ':' + path)).hexdigest(),
                'minute implementation or follow-up changed: ' + path)
    ledger = json.loads(subprocess.check_output([sys.executable, '-O',
        str(ROOT / 'docs/reports/yank-native-minute/verify-published-replay.py'),
        str(ROOT / parent['canonical_pair'][0] / 'replay.json')], cwd=ROOT))
    tests = {}
    for name, expected in [('audit-minute-tests.xml', 132), ('frozen-tests.xml', 133)]:
        suites = ET.parse(HERE / name).getroot().findall('testsuite')
        require(sum(int(s.attrib['tests']) for s in suites) == expected, 'test count: ' + name)
        require(all(int(s.attrib[k]) == 0 for s in suites for k in ('failures', 'errors', 'skipped')), 'tests did not pass')
        tests[name] = {'passed': expected, 'sha256': sha(HERE / name)}
    return dict(status='PASS_INTEGRATION_CHECKS', research_status='HOLD_VALIDATION',
                original_revision=ORIGINAL, pilot_base_revision=pilot_revision,
                frozen_replay_revision=git('rev-parse', 'HEAD', cwd=REPLAY).decode().strip(),
                canonical_pair=[str(first), str(second)], canonical_sha256=canonical_hashes,
                byte_identical=True, historical_sha256=historical_hashes,
                historical_saved_repeat_matches=True, changed_report_fields=changed_fields,
                changed_audit_code=changed_code, code_hashes=code_hashes,
                unchanged_historical_artifacts=unchanged_artifacts,
                input_hashes=report['input_hashes'], archive_artifact_sha256=archive_hashes,
                all_inputs_and_archives_unchanged=True, case_count=5, arm_order_count=8,
                conditional_findings=len(conditions), outcomes=dict(Counter(s['outcome'] for s in conditions)),
                full_findings_and_schedules_equal_original=True, may28_both_interpretations=True,
                native_minute_canonical_sha256=parent['canonical_sha256'],
                native_minute_sources_and_followups_unchanged=True, published_ledger=ledger,
                tests=tests, verification_script_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.first.resolve(), args.second.resolve()), sort_keys=True, indent=2))
