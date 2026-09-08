"""Pinned, streaming, deterministic MNQM5 evidence audit."""
import csv
import gzip
import hashlib
import importlib.metadata
import json
import warnings
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path

import databento as db
from databento.common.error import BentoWarning
import numpy as np

from .core import Book, Record, DAY, MINUTE, NS, SCALE, UNDEF, iso, ns, opportunities, ordering, outcome

ROOT = Path(__file__).resolve().parents[3]
INPUT_ROOT = ROOT.parent / 'Silver-Bullet-ML-BMAD'
ACQUISITION = INPUT_ROOT / 'data/yank/databento-pilot-20260907'
ARCHIVE = ROOT.parent / 'Silver-Bullet-ML-BMAD-yank-replay/docs/reports/yank-signals/development-run1'
START = ns('2025-05-19T00:00:00Z')
END = ns('2025-05-31T00:00:00Z')
LOCK = {'acquisition': '3ddb0268a5e224be817eb95a68bad4544ad927ef8129317ed6fc3d0445159156',
        'archive': 'af16cffe7556dd5dfb8e308e3cf226988668e0d10bc91beef7ef289e534f0cb1',
        'bars': '3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'input hash mismatch: {path}')
    return actual


def canonical(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'


def load_inputs():
    hashes = {}
    for label, base in [('acquisition', ACQUISITION), ('archive', ARCHIVE)]:
        hashes[label + '/manifest.json'] = verify(base / 'manifest.json', LOCK[label])
    acq = json.loads((ACQUISITION / 'manifest.json').read_text())
    archive = json.loads((ARCHIVE / 'manifest.json').read_text())
    files = []
    for item in acq['files']:
        name = item['file']
        if name.endswith('.dbn.zst') or name.endswith('/metadata.json') or name.endswith('/condition.json'):
            path = ACQUISITION / name
            if not path.resolve().is_relative_to(ACQUISITION.resolve()):
                raise ValueError('input path escapes acquisition')
            hashes['acquisition/' + name] = verify(path, item['sha256'])
            if name.endswith('.dbn.zst'):
                files.append(path)
    bars_desc = archive['declared_manifest']['development_data']
    if bars_desc['sha256'] != LOCK['bars']:
        raise ValueError('unexpected declared bars')
    bars_path = INPUT_ROOT / bars_desc['path']
    hashes['original_bars'] = verify(bars_path, LOCK['bars'])
    engine = ROOT.parent / 'Silver-Bullet-ML-BMAD-yank-replay/src/research/yank_signals/engine.py'
    hashes['archive/engine.py'] = verify(engine, archive['source_hashes']['src/research/yank_signals/engine.py'])
    cases = {}
    for arm in ('no-ml', 'ml050'):
        name = arm + '-events.jsonl.gz'
        hashes['archive/' + name] = verify(ARCHIVE / name, archive['artifacts'][name])
        events = []
        with gzip.open(ARCHIVE / name, 'rt') as f:
            for line_number, line in enumerate(f, 1):
                event = json.loads(line)
                if event['kind'] in ('ORDER', 'FILL', 'EXIT', 'EXPIRE'):
                    events.append((line_number, event))
        orders = [(line, e) for line, e in events if e['kind'] == 'ORDER' and START <= ns(e['signal_time']) < END]
        if len(orders) != 4:
            raise ValueError('expected four frozen orders per arm')
        for line, e in orders:
            key = tuple(e[k] for k in ('signal_time', 'entry_price', 'sl_price', 'tp_price', 'quantity'))
            if e['quantity'] != -5:
                raise ValueError('unexpected frozen quantity')
            c = cases.setdefault(key, dict(signal_time=e['signal_time'], entry=e['entry_price'], stop=e['sl_price'], target=e['tp_price'], quantity=e['quantity'], sources=[]))
            history = [dict(line_number=n, event=r) for n, r in events if r.get('order_id') == e['order_id']]
            c['sources'].append(dict(arm=arm, file=name, order_id=e['order_id'], order_line=line, history=history))
    if len(cases) != 5:
        raise ValueError('expected five deduplicated cases')
    bars = {}
    labels = []
    with bars_path.open(newline='') as f:
        for row in csv.DictReader(f):
            t = ns(row['timestamp'])
            if labels and t <= labels[-1]:
                raise ValueError('original bar labels not strictly increasing')
            labels.append(t)
            if START-MINUTE <= t <= END+DAY:
                bars[t] = [int(float(row[k])*SCALE) for k in ('open', 'high', 'low', 'close')] + [int(float(row['volume']))]
    return files, list(cases.values()), bars, labels, hashes


class Gaps:
    """Bounded diagnostics partitioned at every assessment boundary and minute.

    Within a partition only first/last affected nanoseconds and count are retained.
    Assessment endpoints are partitions, so no prearrival/postexpiry point leaks in.
    Without registered boundaries, exact singleton points are retained for tests.
    """
    def __init__(self, boundaries=()):
        self.counts = Counter()
        self.samples = []
        self._sample_keys = []
        self.boundaries = sorted(set(boundaries))
        self.intervals = {}
        self.minute_reasons = {'capture': {}, 'exchange_diagnostic': {}}

    def affect(self, reason, t, exchange=None):
        if START <= t < END:
            bucket = (t // MINUTE, bisect_right(self.boundaries, t)) if self.boundaries else t
            key = (reason, bucket)
            if key in self.intervals:
                span = self.intervals[key]
                span[0], span[1] = min(span[0], t), max(span[1], t)
            else:
                if len(self.intervals) >= 1_000_000:
                    raise ValueError('gap partition bound exceeded')
                self.intervals[key] = [t, t]
            self.minute_reasons['capture'].setdefault(t//MINUTE, set()).add(reason)
        if exchange is not None and START <= exchange < END:
            self.minute_reasons['exchange_diagnostic'].setdefault(exchange//MINUTE, set()).add(reason)

    def add(self, reason, source, index, t, exchange=None):
        self.counts[reason] += 1
        key=(source,index,reason,t)
        if len(self.samples)<100 or key<self._sample_keys[-1]:
            position=bisect_left(self._sample_keys,key)
            self._sample_keys.insert(position,key)
            self.samples.insert(position,dict(reason=reason,source=source,record_index=index,ts_recv_ns=t))
            if len(self.samples)>100:
                self.samples.pop()
                self._sample_keys.pop()
        self.affect(reason, t, exchange)

    def overlaps(self, start, end, include_start=False):
        return sorted({reason for (reason, _), (first, last) in self.intervals.items()
                       if first < end and (last >= start if include_start else last > start)})


class Scenario:
    def __init__(self, case, labels, convention, delay):
        self.case = case
        self.convention = convention
        self.delay = delay
        self.schedule = opportunities(labels, ns(case['signal_time']), convention, END)
        self.arrival = self.schedule['signal_completion'] + delay * 1_000_000
        self.expiry = self.schedule['expiry']
        self.limit = int(case['entry'] * SCALE)
        self.arrival_book = None
        self.arrival_done = False
        self.touch = None
        self.through = None
        self.at = 0
        self.above = 0
        self.gaps = set()
        self.observed_minutes = set()
        self.first_observation = None
        self.last_observation = None
        self.equal_boundary_records = 0

    def trade(self, r, ref, valid):
        if r.ts == self.arrival:
            self.equal_boundary_records += 1
        if not self.arrival < r.ts < self.expiry:
            return
        if not valid:
            self.gaps.add('invalid_trade_event')
            return
        if r.price == self.limit:
            self.at += r.size
            if self.touch is None:
                self.touch = ref
        if r.price > self.limit:
            self.above += r.size
            if self.through is None:
                self.through = ref
            if self.touch is None:
                self.touch = ref

    def result(self, status, global_gaps, bars):
        expected = []
        offset = 0 if self.convention == 'start' else -MINUTE
        for t in bars:
            start = t + offset
            if start < self.expiry and start + MINUTE > self.arrival:
                expected.append(start // MINUTE)
        missing = sorted(set(expected) - self.observed_minutes)
        if missing:
            self.gaps.add('scheduled_minutes_without_live_MBO')
        if global_gaps.overlaps(self.arrival, self.expiry):
            self.gaps.add('native_evidence_gaps_in_pending_interval')
        status_gaps = status.overlaps(self.arrival, self.expiry)
        if status_gaps:
            self.gaps.add('non_trading_or_unknown_status_in_pending_interval')
        if self.schedule['clipped']:
            self.gaps.add('pending_lifetime_clipped')
        if self.arrival_book is None:
            self.gaps.add('arrival_book_unavailable')
        covered = not missing and bool(self.observed_minutes)
        return dict(convention=self.convention, delay_ms=self.delay, arrival=iso(self.arrival), arrival_ns=self.arrival,
                    expiry_exclusive=iso(self.expiry), schedule={k: iso(v) if k in ('signal_completion', 'first_opportunity', 'last_opportunity', 'expiry') else v for k,v in self.schedule.items()},
                    arrival_book=self.arrival_book, first_touch_or_better=self.touch, first_strictly_higher_trade=self.through,
                    volume_at_limit=self.at, volume_strictly_above_limit=self.above, volume_at_or_through=self.at+self.above,
                    outcome=outcome(self.through, self.touch, self.gaps, covered), evidence_gaps=sorted(self.gaps),
                    missing_scheduled_minutes=[iso(t*MINUTE) for t in missing], status_intervals=status_gaps, native_gap_reasons=global_gaps.overlaps(self.arrival,self.expiry),
                    equal_arrival_time_records_excluded=self.equal_boundary_records,
                    first_live_event=iso(self.first_observation) if self.first_observation else None,
                    last_live_event=iso(self.last_observation) if self.last_observation else None,
                    interpretation='Strict trade-through supports execution only under a no-impact assumption; no fill, queue position, or partial fill is inferred.')


class Status:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: (r['ts_recv_ns'], r['source'], r['record_index']))
        self.times = [r['ts_recv_ns'] for r in self.rows]

    def overlaps(self, start, end):
        i = bisect_right(self.times, start)-1
        spans = []
        cursor = start
        while cursor < end:
            row = self.rows[i] if i >= 0 else None
            nxt = self.times[i+1] if i+1 < len(self.times) else END
            stop = min(end, nxt)
            if row is None or row['is_trading'] != 'Y':
                spans.append(dict(start=iso(cursor), end=iso(stop), state=row))
            if stop == end:
                break
            cursor = stop
            i += 1
        return spans


def status_flag(value):
    """Normalize DBN's Optional[bool] API to the source's Y/N/~ notation."""
    if value is True or value == 'Y':
        return 'Y'
    if value is False or value == 'N':
        return 'N'
    if value is None or value == '~':
        return '~'
    raise ValueError('unsupported decoded status flag')


def read_auxiliary(files):
    statuses, definitions = [], []
    for path in sorted(files):
        if '.mbo.' in path.name:
            continue
        store = db.DBNStore.from_file(path)
        validate_metadata(store, path)
        for i, r in enumerate(store):
            if r.instrument_id != 42009475:
                raise ValueError('wrong auxiliary instrument')
            common = dict(source=str(path.relative_to(ACQUISITION)), record_index=i, ts_recv_ns=r.ts_recv, ts_event_ns=r.ts_event,
                          ts_recv=iso(r.ts_recv), ts_event=iso(r.ts_event), carried_initial_state=r.ts_recv == store.metadata.start and r.ts_event < store.metadata.start)
            if '.status.' in path.name:
                statuses.append(dict(**common, action=str(r.action), reason=str(r.reason), is_trading=status_flag(r.is_trading), is_quoting=status_flag(r.is_quoting)))
            else:
                if r.raw_symbol != 'MNQM5' or r.min_price_increment != 250_000_000:
                    raise ValueError('definition identity or tick mismatch')
                definitions.append(dict(**common, raw_symbol=r.raw_symbol, min_price_increment=r.min_price_increment, currency=r.currency, expiration=iso(r.expiration)))
    return Status(statuses), definitions


def validate_metadata(store, path):
    m = store.metadata
    if m.dataset != 'GLBX.MDP3' or m.symbols != ['MNQM5'] or m.ts_out or m.version != 3:
        raise ValueError(f'unsupported DBN metadata: {path}')
    if str(m.schema) not in ('mbo', 'definition', 'status'):
        raise ValueError('unexpected native schema')


def aggregate(trades, target, field):
    """Chunk-independent native-order OHLC. Timestamp bins never reorder trades."""
    if len(trades) == 0:
        return
    times = trades[field] // MINUTE
    boundaries = np.r_[0, np.flatnonzero(times[1:] != times[:-1])+1, len(trades)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        t = int(times[a])*MINUTE
        px = trades['price'][a:b]
        vol = int(trades['size'][a:b].sum(dtype=np.uint64))
        if t not in target:
            target[t] = [int(px[0]), int(px.max()), int(px.min()), int(px[-1]), vol, int(b-a)]
        else:
            old = target[t]
            old[1] = max(old[1], int(px.max()))
            old[2] = min(old[2], int(px.min()))
            old[3] = int(px[-1])
            old[4] += vol
            old[5] += int(b-a)


def timeline_windows(cases):
    windows = []
    for case in cases:
        if not case['signal_time'].startswith('2025-05-28'):
            continue
        for source in case['sources'][:1]:
            history = [r['event'] for r in source['history']]
            fill = next((r for r in history if r['kind']=='FILL'), None)
            exit_ = next((r for r in history if r['kind']=='EXIT'), None)
            if not fill or not exit_:
                continue
            for convention in ('start','end'):
                shift = 0 if convention=='start' else -MINUTE
                windows.append(dict(case_id=case['case_id'], convention=convention,
                                    start=ns(fill['timestamp'])+shift, end=ns(exit_['timestamp'])+shift+MINUTE,
                                    entry=int(case['entry']*SCALE), stop=int(case['stop']*SCALE), target=int(case['target']*SCALE),
                                    first_entry=None, first_strict_entry=None, first_stop=None, first_target=None,
                                    first_stop_after_entry=None, first_target_after_entry=None, trades=0, observed_minutes=set()))
    return windows


def stream_file(path, scenarios, recv_bars, exchange_bars, gaps, windows, extracts, chunk_size):
    store = db.DBNStore.from_file(path)
    validate_metadata(store, path)
    source = str(path.relative_to(ACQUISITION))
    day_start, day_end = store.metadata.start, store.metadata.end
    active = [s for s in scenarios if s.arrival < day_end and s.expiry > day_start]
    local_windows = [w for w in windows if w['start'] < day_end and w['end'] > day_start]
    replay = bool(active or local_windows)
    first_interest = min([s.arrival for s in active]+[w['start'] for w in local_windows],default=day_end)
    last_interest = max([s.expiry for s in active]+[w['end'] for w in local_windows],default=day_start)
    book = Book()
    next_arrival = min((s.arrival for s in active), default=END+DAY)
    record_index = event_id = 0
    pending_trades, pending_live = [], []
    event_reasons = set()
    event_open = False
    previous_complete = None
    last_live = None
    counts = Counter()
    live_minutes = set()
    max_book_orders = 0
    tail = None
    prior_capture = None
    for chunk in store.to_ndarray(count=chunk_size):
        a = np.concatenate((tail, chunk)) if tail is not None else chunk
        ends = np.flatnonzero((a['flags'] & 128) != 0)
        if not len(ends):
            tail = a
            if len(tail) > 100000:
                raise ValueError('unbounded event or incomplete snapshot')
            continue
        cut = int(ends[-1])+1
        tail = a[cut:].copy() if cut < len(a) else None
        a = a[:cut]
        if not np.all((a['rtype']==160) & (a['length']==14) & (a['instrument_id']==42009475)):
            raise ValueError('unsupported record layout or instrument in MBO stream')
        snapshot = (a['flags'] & 32) != 0
        valid_time = (a['ts_recv'] >= day_start) & (a['ts_recv'] < day_end) & (a['ts_event'] > 0) & (a['ts_event'] < END+DAY)
        problems = {}
        def mark(mask, reason):
            for j in np.flatnonzero(mask):
                problems.setdefault(int(j), set()).add(reason)
        live_indices = np.flatnonzero((~snapshot) & valid_time)
        if len(live_indices):
            times = a['ts_recv'][live_indices]
            previous = np.r_[np.uint64(prior_capture if prior_capture is not None else int(times[0])), times[:-1]]
            for j in live_indices[times < previous]:
                problems.setdefault(int(j), set()).add('capture_time_regression')
            prior_capture = int(times[-1])
        mark((~snapshot)&(~valid_time), 'invalid_timestamp')
        mark((a['flags'] & 87)!=0, 'unsupported_flags')
        mark((~snapshot)&((a['flags']&8)!=0), 'bad_live_capture_timestamp')
        mark(~np.isin(a['action'], [b'R',b'A',b'M',b'C',b'T',b'F',b'N']), 'unsupported_action')
        mark(snapshot & (~np.isin(a['action'],[b'R',b'A'])), 'unsupported_snapshot_action')
        malformed_px_size=(a['price']<=0)|(a['price']>=UNDEF)|(a['size']==0)
        mark((a['action']==b'T')&(~snapshot)&malformed_px_size, 'invalid_trade_fields')
        mark(np.isin(a['action'],[b'A',b'M',b'C']) & (malformed_px_size | (~np.isin(a['side'],[b'A',b'B']))), 'invalid_order_fields')
        if record_index == 0 and (a['action'][0] != b'R' or not (int(a['flags'][0]) & 32)):
            problems.setdefault(0,set()).add('missing_initial_snapshot')
        # Retain raw, finite T observations. Diagnostic qualifications below prevent
        # observations in bad events from being presented as validated clean OHLC.
        trades = a[(a['action']==b'T') & (~snapshot) & valid_time & (~malformed_px_size)]
        aggregate(trades, recv_bars, 'ts_recv')
        aggregate(trades, exchange_bars, 'ts_event')
        for action, count in zip(*np.unique(a['action'], return_counts=True)):
            counts[action.decode()] += int(count)
        live_minutes.update(int(t) for t in np.unique(a['ts_recv'][(~snapshot)&valid_time]//MINUTE))
        if not replay:
            # Diagnostics emitted in native record order, irrespective of chunks.
            for j in sorted(problems):
                for reason in sorted(problems[j]):
                    gaps.add(reason,source,record_index+j,int(a['ts_recv'][j]),int(a['ts_event'][j]))
            event_ends = np.flatnonzero((a['flags']&128)!=0)
            affected_events = {}
            for j,reasons in problems.items():
                e=int(np.searchsorted(event_ends,j))
                affected_events.setdefault(e,set()).update(reasons)
            for e,reasons in affected_events.items():
                event_start=int(event_ends[e-1])+1 if e else 0
                event_end=int(event_ends[e])+1
                for t,ex in zip(a['ts_recv'][event_start:event_end],a['ts_event'][event_start:event_end]):
                    for reason in reasons:
                        gaps.affect(reason,int(t),int(ex))
            record_index += len(a)
            continue
        for j, row in enumerate(a.tolist()):
            _, _, _, _, ex, oid, price, size, flags, _, action, side, ts, _, sequence = row
            r = Record(record_index, ts, ex, action.decode(), side.decode(), oid, price, size, flags, sequence)
            record_index += 1
            snap = bool(flags & 32)
            if not snap:
                for scenario in (active if ts >= next_arrival else ()):
                    if not scenario.arrival_done and ts >= scenario.arrival:
                        scenario.arrival_done = True
                        if previous_complete and previous_complete[0] < scenario.arrival and previous_complete[2]:
                            scenario.arrival_book = dict(**book.quote(scenario.limit, scenario.case['quantity']),
                                as_of=iso(previous_complete[0]), source=source, event_id=previous_complete[1],
                                capture_age_ns=scenario.arrival-previous_complete[0]) if not event_open else None
                        if event_open:
                            scenario.gaps.add('arrival_during_incomplete_event')
                if ts >= next_arrival:
                    next_arrival = min((s.arrival for s in active if not s.arrival_done), default=END+DAY)
                pending_live.append((ts,ex))
            event_open = True
            reasons = problems.get(j,set()).copy()
            error = book.apply(r)
            if error:
                reasons.add(error)
            for reason in sorted(reasons):
                gaps.add(reason,source,r.index,ts,ex)
            event_reasons.update(reasons)
            if not snap and r.action == 'T' and 0 < price < UNDEF and size > 0 and day_start <= ts < day_end and 0 < ex < END+DAY:
                pending_trades.append(r)
            if len(pending_live) > 100000:
                raise ValueError('unbounded event without F_LAST')
            if flags & 128:
                book_error = book.complete()
                if not snap:
                    if book_error:
                        gaps.add(book_error,source,r.index,ts,ex)
                        event_reasons.add(book_error)
                    valid = not event_reasons
                    event_name = f'{path.name}:{event_id}'
                    # Credit each record's minute only after its whole event validates.
                    for t, exchange in pending_live:
                        for reason in event_reasons:
                            gaps.affect(reason,t,exchange)
                        for scenario in (active if first_interest <= t < last_interest else ()):
                            if scenario.arrival < t < scenario.expiry:
                                if valid:
                                    scenario.observed_minutes.add(t//MINUTE)
                                    scenario.first_observation = min(scenario.first_observation or t,t)
                                    scenario.last_observation = max(scenario.last_observation or t,t)
                                else:
                                    scenario.gaps.add('invalid_completed_book_or_event')
                        for w in (local_windows if first_interest <= t < last_interest else ()):
                            if valid and w['start'] <= t < w['end']:
                                w.setdefault('observed_minutes',set()).add(t//MINUTE)
                    for trade in pending_trades:
                        relevant_scenarios = [s for s in active if s.arrival <= trade.ts < s.expiry]
                        relevant_windows = [w for w in local_windows if w['start'] <= trade.ts < w['end']]
                        if not relevant_scenarios and not relevant_windows:
                            continue
                        ref = trade.ref(source,event_name)
                        for scenario in relevant_scenarios:
                            scenario.trade(trade,ref,valid)
                        for w in relevant_windows:
                            w['trades'] += 1
                            roles=[]
                            for key,crossed in [('first_entry',trade.price>=w['entry']),('first_strict_entry',trade.price>w['entry']),('first_stop',trade.price>=w['stop']),('first_target',trade.price<=w['target'])]:
                                if crossed and w[key] is None:
                                    w[key]=dict(ref,completed_event_valid=valid)
                                    roles.append(key)
                            if w['first_entry']:
                                for key,crossed in [('first_stop_after_entry',trade.price>=w['stop']),('first_target_after_entry',trade.price<=w['target'])]:
                                    if crossed and w[key] is None:
                                        w[key]=dict(ref,completed_event_valid=valid)
                                        roles.append(key)
                            if roles:
                                extracts.write(json.dumps(dict(case_id=w['case_id'],convention=w['convention'],roles=roles,record=ref),sort_keys=True)+'\n')
                    previous_complete=(ts,event_name,valid)
                    last_live=ts
                pending_trades.clear()
                pending_live.clear()
                event_reasons.clear()
                event_open=False
                event_id += 1
                max_book_orders=max(max_book_orders,len(book.orders))
    if tail is not None and len(tail):
        gaps.add('incomplete_final_event_or_snapshot',source,record_index,int(tail['ts_recv'][0]),int(tail['ts_event'][0]))
        for t,ex in zip(tail['ts_recv'],tail['ts_event']):
            gaps.affect('incomplete_final_event_or_snapshot',int(t),int(ex))
        record_index += len(tail)
    if hasattr(store,'reader'):
        store.reader.close()
    return dict(source=source,start=iso(day_start),end_exclusive=iso(day_end),records=record_index,
                action_counts=dict(counts),max_book_orders=max_book_orders,book_replayed=replay,
                live_minutes=len(live_minutes),last_completed_live_event=iso(last_live) if last_live else None)


def reconciliation(bars, recv, exchange, output, gaps=None, book_days=()):
    summaries = {}
    with (output/'reconciliation.jsonl').open('w') as f:
        for clock, native in [('capture',recv),('exchange_diagnostic',exchange)]:
            for convention in ('start','end'):
                counts = Counter()
                for t, original in bars.items():
                    start = t if convention=='start' else t-MINUTE
                    if not START <= start < END:
                        continue
                    observed = native.get(start)
                    counts['original_minutes'] += 1
                    if observed is None:
                        counts['missing_native_T_minutes'] += 1
                        delta = None
                    else:
                        counts['covered_minutes'] += 1
                        delta = [(observed[i]-original[i])/SCALE for i in range(4)]
                        counts['exact_OHLC_minutes' if not any(delta) else 'different_OHLC_minutes'] += 1
                    reasons = sorted(gaps.minute_reasons[clock].get(start//MINUTE,set())) if gaps else []
                    qualification = 'missing_native_T' if observed is None else ('contaminated_raw_T' if reasons else 'raw_T_no_detected_basic_gap')
                    if reasons:
                        counts['contaminated_minutes'] += 1
                    row = dict(clock=clock, convention=convention, original_label=iso(t), native_minute_start=iso(start),
                               original_ohlc=[v/SCALE for v in original[:4]], native_ohlc=[v/SCALE for v in observed[:4]] if observed else None,
                               ohlc_delta=delta, original_volume=original[4], native_T_volume=observed[4] if observed else None,
                               volume_delta=observed[4]-original[4] if observed else None, gap_reasons=reasons,
                               coverage_qualification=qualification, book_replayed_for_capture_day=start//DAY in book_days)
                    f.write(json.dumps(row,sort_keys=True)+'\n')
                native_minutes = set(native)
                mapped = {t if convention=='start' else t-MINUTE for t in bars if START <= (t if convention=='start' else t-MINUTE)<END}
                counts['native_T_minutes_without_original_bar'] = len(native_minutes-mapped)
                summaries[clock+'/'+convention] = dict(counts)
    return summaries


def finalize_timeline(w,status,gaps):
    start,end=w['start'],w['end']
    w['status_gaps']=status.overlaps(start,end)
    w['native_gap_reasons']=gaps.overlaps(start,end,include_start=True)
    expected=set(range(start//MINUTE,(end-1)//MINUTE+1))
    missing=sorted(expected-w.pop('observed_minutes',set()))
    w['missing_live_minutes']=[iso(t*MINUTE) for t in missing]
    qualifications=[]
    if w['status_gaps']: qualifications.append('non_trading_or_unknown_status')
    if w['native_gap_reasons']: qualifications.append('native_evidence_gaps')
    if missing: qualifications.append('missing_validated_live_minutes')
    w['assessability_gaps']=qualifications
    for label,entry,barrier in [('entry_stop_ordering','first_entry','first_stop'),('strict_entry_stop_ordering','first_strict_entry','first_stop'),('entry_target_ordering','first_entry','first_target')]:
        w[label+'_raw_endpoint_comparison']=ordering(w[entry],w[barrier])
        w[label]='unassessable_interval_gaps' if qualifications else w[label+'_raw_endpoint_comparison']
    w['start'],w['end']=iso(start),iso(end)
    w['fill_conclusion']='AMBIGUOUS_ACTUAL_FILL_AND_QUEUE_UNOBSERVED'


def output_path(path):
    path = Path(path).resolve()
    forbidden = [INPUT_ROOT, ROOT/'src', ROOT/'tests', ROOT/'data', ROOT/'.agents', ROOT/'_bmad', ROOT.parent/'Silver-Bullet-ML-BMAD-yank-replay']
    if path == ROOT or any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError('output directory is in a source/input tree')
    if path.exists():
        raise ValueError('output directory must be fresh')
    return path


def run(output_dir, chunk_size=100000):
    output = output_path(output_dir)
    files, cases, bars, labels, input_hashes = load_inputs()
    code_paths = sorted((ROOT/'src/research/yank_execution_pilot').glob('*.py')) + [ROOT/'src/cli/check_yank_execution_pilot.py']
    code_hashes = {str(p.relative_to(ROOT)):digest(p) for p in code_paths}
    status, definitions = read_auxiliary(files)
    scenarios = []
    for i, case in enumerate(cases,1):
        case['case_id'] = f'case-{i}'
        for convention in ('start','end'):
            for delay in (0,100,500):
                scenarios.append(Scenario(case,labels,convention,delay))
    windows = timeline_windows(cases)
    gap_boundaries = [v for scenario in scenarios for v in (scenario.arrival,scenario.arrival+1,scenario.expiry)]
    gap_boundaries += [v for w in windows for v in (w['start'],w['end'])]
    gaps = Gaps(gap_boundaries)
    recv, exchange, native = {}, {}, []
    output.mkdir(parents=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', BentoWarning)
            with (output/'event-extracts.jsonl').open('w') as extracts:
                for path in sorted(p for p in files if '.mbo.' in p.name):
                    print(f'Processing {path.name}', flush=True)
                    native.append(stream_file(path,scenarios,recv,exchange,gaps,windows,extracts,chunk_size))
        for w in windows:
            finalize_timeline(w,status,gaps)
        for case in cases:
            case['conditions'] = [s.result(status,gaps,bars) for s in scenarios if s.case is case]
        recon = reconciliation(bars,recv,exchange,output,gaps,{ns(f['start'])//DAY for f in native if f['book_replayed']})
        report = dict(audit_status='PASS_AUDIT_CHECKS', research_status='HOLD_VALIDATION',
                      instrument='MNQM5', instrument_id=42009475, purchased_interval=[iso(START),iso(END)],
                      decoder=dict(databento=importlib.metadata.version('databento'),databento_dbn=importlib.metadata.version('databento-dbn'),numpy=importlib.metadata.version('numpy')),
                      policy=dict(bar_label_provenance='UNRESOLVED_BOTH_CONDITIONS_RETAINED', timing='ts_recv is capture proxy; ts_event diagnostic only; synthetic snapshot times excluded',
                                  boundary='Arrival equality excluded; pending expiry is exclusive end of 240th scheduled bar; every last-bar trade before its end included',
                                  limitations=['No queue reconstruction for hypothetical orders','No impact assumption for trade-through evidence','No partial fills invented','Historical P&L reference only','Historical $4 cost is not a verified broker charge','No revised returns or downstream fill propagation']),
                      input_hashes=input_hashes,code_hashes=code_hashes,cases=cases,may28_timelines=windows,
                      reconciliation_summary=recon,native_files=native,status_records=status.rows,definitions=definitions,
                      evidence_gaps=dict(counts=dict(gaps.counts),samples=gaps.samples),
                      audit_checks=dict(pinned_input_hashes='PASS',four_orders_per_arm='PASS',five_unique_cases='PASS',conditional_findings=len(scenarios)))
        lines = ['# YANK frozen-order execution pilot','', 'PASS_AUDIT_CHECKS — HOLD_VALIDATION','',
                 'Five frozen cases; both bar-label interpretations and 0/100/500 ms delays are retained. Capture time is an observable proxy, not broker arrival time. Strict trade-through supports a no-impact hypothesis; it does not prove a fill or queue position.', '',
                 '| Case | Signal UTC | Interpretation | Delay ms | Outcome | Arrival spread | At / through volume |',
                 '|---|---|---|---:|---|---:|---:|']
        for case in cases:
            for s in case['conditions']:
                quote=s['arrival_book'] or {}
                lines.append(f"| {case['case_id']} | {case['signal_time']} | {s['convention']} | {s['delay_ms']} | {s['outcome']} | {quote.get('spread','unavailable')} | {s['volume_at_limit']} / {s['volume_strictly_above_limit']} |")
        lines += ['', 'Reconciliation counts (full diagnostics retain both conventions):', '', '```json', canonical(recon).strip(), '```', '']
        for case in cases:
            lines.append(f"{case['case_id']}: " + ', '.join(f"{r['arm']} order {r['order_id']}" for r in case['sources']))
            for condition in case['conditions']:
                lines.append(f"- {condition['convention']} +{condition['delay_ms']}ms gaps: " + (', '.join(condition['evidence_gaps']) or 'none detected'))
        lines += ['', 'May 28 crossing evidence:', '']
        for w in windows:
            lines.append(f"- {w['case_id']} {w['convention']} interval {w['start']} to {w['end']}: {w['entry_stop_ordering']}.")
            lines.append('  Qualifications: ' + (', '.join(w['assessability_gaps']) or 'none detected') + '.')
            for key in ('first_entry','first_strict_entry','first_stop','first_target'):
                ref=w[key]
                if ref:
                    lines.append(f"  {key}: {ref['ts_recv']}, price {ref['price']}, {ref['source']} record {ref['record_index']}, event {ref['event_id']}.")
                else:
                    lines.append(f"  {key}: unavailable in this interval.")
        lines += ['', 'May 28 native crossing timelines are in report.json and event-extracts.jsonl. Equal capture times or a shared event remain ambiguous. Actual fills remain unobserved. Status interruptions and coverage gaps affect assessability.', '',
                  'Minute-by-minute unchanged-price comparisons appear in reconciliation.jsonl. Empirical agreement cannot establish independent bar-label provenance.', '',
                  'Archived order, fill and exit economics are retained as historical references. The historical $4 cost is not a verified broker charge. No revised strategy return is computed.', '',
                  'Input and implementation hashes and decoder versions appear in report.json. Artifact hashes appear in artifacts.json. A passing audit reports successful integrity/implementation checks, not strategy validation.', '']
        # Re-read pinned evidence after processing to catch modification during the run.
        _, _, _, _, after = load_inputs()
        if after != input_hashes or any(digest(ROOT/p)!=h for p,h in code_hashes.items()):
            raise ValueError('inputs or implementation changed during audit')
        (output/'report.json').write_text(canonical(report))
        (output/'report.md').write_text('\n'.join(lines))
        (output/'artifacts.json').write_text(canonical({p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
        return report
    except BaseException:
        # A failed/incomplete run must never leave a PASS report behind.
        for name in ('report.json','report.md','artifacts.json'):
            (output/name).unlink(missing_ok=True)
        (output/'FAILED').write_text('Audit failed; no PASS_AUDIT_CHECKS. See command error.\n')
        raise
