"""Incremental attribution over unchanged frozen scenarios; offline native bytes only."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
from collections import Counter
from datetime import datetime, timezone


def load_audit():
    path = Path(__file__).resolve().parents[1] / 'yank_execution_pilot'
    name = '_yank_gap_native_audit'
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + '.audit', path / 'audit.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ns(value):
    # Frozen reports have canonical UTC labels with nine fractional digits.
    base, _, fraction = value.removesuffix('Z').partition('.')
    return int(datetime.fromisoformat(base).replace(tzinfo=timezone.utc).timestamp()) * 10**9 + int((fraction + '000000000')[:9])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


class EventTrace:
    """Bounded run-length evidence, preserving full native event boundaries."""
    def __init__(self, source, arrivals):
        self.source = source
        self.arrivals = sorted(set(arrivals))
        self.event_id = 0
        self.first = self.last = self.first_live = None
        self.pending_errors = set()
        self.completed_index = -1
        self.previous = None
        self.invalid_runs = []
        self.open_run = None
        self.straddlers = []
        self.initial_boundary = None
        self.records = self.completed = 0

    def ref(self, record):
        return record.ref(self.source, f'{Path(self.source).name}:{self.event_id}')

    def apply(self, record):
        self.records += 1
        if self.first is None:
            self.first = record
        self.last = record
        if not record.flags & 32 and self.first_live is None:
            self.first_live = record
        if self.initial_boundary is None:
            self.initial_boundary = self.ref(record)

    def error(self, reason, index):
        # The original stream adds Book.complete's reason after our callback.
        if index != self.completed_index:
            self.pending_errors.add(reason)

    def complete(self, book_error):
        if book_error:
            self.pending_errors.add(book_error)
        self.completed_index = self.last.index
        if self.first_live is not None:
            from bisect import bisect_right
            left = bisect_right(self.arrivals, self.first_live.ts)
            right = bisect_right(self.arrivals, self.last.ts)
            crossings = self.arrivals[left:right]
            raw_event = (self.first_live, self.last, sorted(self.pending_errors), self.event_id)
            def materialize(raw):
                first, last, errors, event_id = raw
                name = f'{Path(self.source).name}:{event_id}'
                return dict(first=first.ref(self.source,name), last=last.ref(self.source,name),
                            reasons=errors, valid=not errors)
            if crossings or self.pending_errors or self.open_run is not None:
                event = materialize(raw_event)
                for arrival in crossings:
                    self.straddlers.append(dict(arrival_ns=arrival, event=event,
                        strictly_previous_event=materialize(self.previous) if self.previous else None,
                        recovery_boundary=event['last'] if event['valid'] else None))
                if event['reasons']:
                    if self.open_run is None:
                        self.open_run = dict(first_event=event, last_event=event, completed_events=0,
                                             reasons=[], recovery_event=None)
                        self.invalid_runs.append(self.open_run)
                    self.open_run['last_event'] = event
                    self.open_run['completed_events'] += 1
                    self.open_run['reasons'] = sorted(set(self.open_run['reasons']) | set(event['reasons']))
                elif self.open_run is not None:
                    self.open_run['recovery_event'] = event
                    self.open_run = None
            self.previous = raw_event
        self.completed += 1
        self.event_id += 1
        self.first = self.last = self.first_live = None
        self.pending_errors.clear()

    def result(self):
        return dict(source=self.source, reconstruction_boundary=self.initial_boundary,
                    records=self.records, completed_events=self.completed,
                    invalid_runs=self.invalid_runs, arrival_straddlers=self.straddlers)


def scan(audit, path, root, conditions, chunk_size):
    source = str(path.relative_to(root))
    trace = EventTrace(source, [c['arrival_ns'] for c in conditions])
    OriginalBook = audit.Book

    class TracedBook(OriginalBook):
        def apply(self, record):
            trace.apply(record)
            return super().apply(record)

        def complete(self):
            error = super().complete()
            trace.complete(error)
            return error

    class TracedGaps(audit.Gaps):
        def add(self, reason, source, index, t, exchange=None):
            trace.error(reason, index)
            super().add(reason, source, index, t, exchange)

    # No archive loading: windows force the original full native reconstruction,
    # while its empty timeline only requests evidence, never computes new outcomes.
    start = min(c['arrival_ns'] for c in conditions)
    end = max(ns(c['expiry_exclusive']) for c in conditions)
    window = dict(start=start, end=end, entry=10**30, stop=10**30, target=-1,
                  trades=0, first_entry=None, first_strict_entry=None, first_stop=None, first_target=None)
    gaps = TracedGaps([v for c in conditions for v in (c['arrival_ns'], ns(c['expiry_exclusive']))])
    audit.Book = TracedBook
    try:
        audit.stream_file(path, [], {}, {}, gaps, [window], io.StringIO(), chunk_size)
    finally:
        audit.Book = OriginalBook
    result = trace.result()
    result['gap_counts'] = dict(gaps.counts)
    return result


def inventory(report, scans, hashes):
    rows = []
    status_rows = report['status_records']
    for case in report['cases']:
        for condition in case['conditions']:
            start, end = condition['arrival_ns'], ns(condition['expiry_exclusive'])
            blockers = []
            for scan_result in scans:
                for event in scan_result['arrival_straddlers']:
                    if event['arrival_ns'] == start and 'arrival_during_incomplete_event' in condition['evidence_gaps']:
                        blockers.append(dict(kind='arrival_incomplete_event', native=event,
                            source_sha256=hashes[scan_result['source']], disposition='resolved by cited evidence',
                            remaining='Exact event and completion explain unavailable arrival book. Retain unassessable outcome: completion after arrival cannot retroactively supply a prior completed book. Hypothetical queue remains unobservable.'))
                for run in scan_result['invalid_runs']:
                    lo = run['first_event']['first']['ts_recv_ns']
                    hi = run['last_event']['last']['ts_recv_ns'] + 1
                    if lo < end and hi > start:
                        blockers.append(dict(kind='invalid_native_completed_events', native=run,
                            source_sha256=hashes[scan_result['source']], full_window_intersection_ns=[max(start,lo), min(end,hi)],
                            overlapping_frozen_reasons=['invalid_completed_book_or_event','native_evidence_gaps_in_pending_interval'],
                            disposition='requires specified additional data',
                            remaining='Obtain provider sequence/channel recovery or independently captured event-complete book evidence for this exact interval, and documented semantics explaining locked/crossed completed books. Later uncrossing establishes recovery only, not validity of the preceding events. No acquisition absence is inferred.'))
            for interval in condition['status_intervals']:
                state = interval['state']
                recovery = next((s for s in status_rows if s['source']==state['source'] and s['ts_recv_ns']==ns(interval['end'])), None)
                blockers.append(dict(kind='observed_nontrading_status', interval=interval,
                    source_sha256=hashes[state['source']], next_status_boundary=recovery,
                    full_window_intersection_ns=[max(start,ns(interval['start'])),min(end,ns(interval['end']))],
                    disposition='resolved by cited evidence',
                    remaining='Observed exchange status explains this period; it is not missing acquisition coverage. The conservative frozen full-window policy remains unassessable. Resume status does not establish hypothetical order acceptance, persistence or queue position.'))
            if condition['missing_scheduled_minutes']:
                blockers.append(dict(kind='missing_acquisition_coverage', minutes=condition['missing_scheduled_minutes'],
                    disposition='requires specified additional data', remaining='Acquire complete native events covering each listed minute plus valid reconstruction boundary.'))
            if condition['outcome']=='unassessable' and not blockers:
                raise ValueError('unexplained unassessable scenario')
            rows.append(dict(case_id=case['case_id'], convention=condition['convention'], delay_ms=condition['delay_ms'],
                frozen_outcome=condition['outcome'], full_window_ns=[start,end], frozen_schedule=condition['schedule'],
                frozen_evidence_gaps=condition['evidence_gaps'], blockers=blockers,
                queue=dict(disposition='remains unobservable', remaining='A historical hypothetical order had no actual queue position or fills. More market data alone cannot observe them; retain no-impact trade-through as conditional evidence only.')))
    return rows


def run(manifest_path, input_dir, output_dir, chunk_size=100000):
    manifest_path, root, output = Path(manifest_path).resolve(), Path(input_dir).resolve(), Path(output_dir)
    if output.is_symlink() or output.exists():
        raise ValueError('fresh output directory required')
    destination = output.resolve()
    code_root = Path(__file__).resolve().parents[3]
    forbidden = [root, manifest_path.parent, code_root/'src', code_root/'tests', code_root/'docs/yank-validation/snapshot']
    if destination == code_root or any(destination==p or destination.is_relative_to(p) for p in forbidden):
        raise ValueError('output overlaps protected input/source tree')
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported manifest')
    paths = {}
    def verify_inputs():
        if digest(manifest_path) != manifest_hash:
            raise ValueError('manifest changed')
        for item in [manifest['report'], *manifest['native_files']]:
            is_report = item is manifest['report']
            path = (manifest_path.parent/item['file']).resolve() if is_report else (root/item['file']).resolve()
            if (not is_report and not path.is_relative_to(root)) or digest(path) != item['sha256']:
                raise ValueError('input pin/path mismatch: '+item['file'])
            paths[item['file']] = path
    verify_inputs()
    report_path = paths[manifest['report']['file']]
    if destination == report_path.parent or destination.is_relative_to(report_path.parent):
        raise ValueError('output overlaps report input tree')
    report = json.loads(report_path.read_text())
    conditions = [c for case in report['cases'] for c in case['conditions']]
    if len(report['cases'])!=5 or len(conditions)!=30 or Counter(c['outcome'] for c in conditions)!={'supported':11,'unassessable':19}:
        raise ValueError('unexpected frozen scenario contract')
    for case in report['cases']:
        if {(c['convention'],c['delay_ms']) for c in case['conditions']} != {(v,d) for v in ('start','end') for d in (0,100,500)}:
            raise ValueError('missing timing convention')
    if any(c['schedule']['scheduled_count']!=240 or c['schedule']['clipped'] for c in conditions):
        raise ValueError('full 240 opportunities required')
    audit = load_audit()
    audit.ACQUISITION = root
    code_paths = [Path(__file__), Path(audit.__file__), Path(audit.__file__).with_name('core.py'), code_root/'src/cli/check_yank_execution_gaps.py']
    code_hashes = {str(p.relative_to(code_root)):digest(p) for p in code_paths}
    hashes = {i['file']:i['sha256'] for i in manifest['native_files']}
    for item in manifest['native_files']:
        if report['input_hashes'].get('acquisition/'+item['file'])!=item['sha256']:
            raise ValueError('native hash not bound to frozen report')
    scans = []
    for item in manifest['native_files']:
        if '.mbo.' in item['file']:
            print('Scanning '+Path(item['file']).name, flush=True)
            scans.append(scan(audit, paths[item['file']], root, conditions, chunk_size))
    rows = inventory(report, scans, hashes)
    # Ensure requested case-1 native evidence really survived reconstruction.
    arrival_case = next(r for r in rows if r['case_id']=='case-1' and r['convention']=='end' and r['delay_ms']==0)
    if not any(b['kind']=='arrival_incomplete_event' for b in arrival_case['blockers']):
        raise ValueError('case-1 straddling event not recovered')
    for row in rows:
        if row['case_id'] in ('case-3','case-4','case-5') and not any(b['kind']=='invalid_native_completed_events' for b in row['blockers']):
            raise ValueError('May28 invalid event evidence missing')
    result = dict(schema_version=1, engineering='PASS_GAP_ASSESSMENT', research_status='HOLD_VALIDATION',
        frozen_outcomes=dict(Counter(r['frozen_outcome'] for r in rows)), scenario_count=30,
        manifest_sha256=manifest_hash, report_sha256=manifest['report']['sha256'], code_hashes=code_hashes,
        native_sha256=hashes, scans=scans, scenarios=rows,
        audit_defect='None confirmed; unchanged audit algorithm and outputs retained. Observer subclass records boundaries only.',
        qualifications=['Native ordering and F_LAST semantics preserved from daily reset/snapshot.',
                        'Runs group consecutive invalid completed events; intersections do not shorten frozen schedules.',
                        'Later recovery or crossing evidence never upgrades an outcome or revises frozen P&L.'])
    verify_inputs()
    if any(digest(code_root/p)!=h for p,h in code_hashes.items()):
        raise ValueError('implementation changed during scan')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.yank-gaps-', dir=destination.parent))
    try:
        (stage/'report.json').write_text(canonical(result))
        lines=['# Incremental frozen execution gaps','','HOLD_VALIDATION. 30 scenarios; 11 supported, 19 unassessable; no outcome changes.','',
               '| Case | Label | Delay ms | Frozen outcome | Attributed blockers |','|---|---|---:|---|---:|']
        for row in rows:
            lines.append(f"| {row['case_id']} | {row['convention']} | {row['delay_ms']} | {row['frozen_outcome']} | {len(row['blockers'])} |")
        lines += ['', 'Blocker counts overlap; they are not independent failed scenarios. The JSON retains each native record/event reference, file hash, recovery boundary and full-window intersection.', '',
                  'Case 1 is an event spanning arrival; its later terminator cannot supply a strictly prior completed book. Cases 3–5 combine invalid completed books with observed market-event and scheduled nontrading statuses. Nontrading is observed evidence, not presumed missing acquisition. Invalid book recovery does not validate preceding events. Hypothetical queue position remains inherently unobservable.', '',
                  'Disposition details and exact additional evidence requirements are recorded separately for every blocker in report.json. Neither this report nor any local engineering pass establishes profitability.']
        (stage/'report.md').write_text('\n'.join(lines)+'\n')
        (stage/'artifacts.json').write_text(canonical({p.name:digest(p) for p in sorted(stage.iterdir())}))
        # Atomic exclusive publication; no replacing a racing empty directory.
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(destination), 1):
            raise OSError(ctypes.get_errno(), 'exclusive output publication failed')
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    try:
        run(args.manifest,args.input_dir,args.output_dir)
    except Exception as exc:
        print(f'FAIL_GAP_ASSESSMENT: {exc}',file=sys.stderr)
        return 1
    print('PASS_GAP_ASSESSMENT; HOLD_VALIDATION')
    return 0
