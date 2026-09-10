"""Explicit-manifest causal replay of the installed snapshot, development only."""
import argparse
import asyncio
from datetime import datetime, timezone, timedelta
import hashlib
import json
import platform
from pathlib import Path
from .adapter import Adapter, canonical, digest

REQUIRED = {'schema_version','classification','source','request','files','contracts','adjustments','interval','availability','session','coverage','warmup','execution_state','format'}


def admit(manifest_path,input_dir):
    m=json.loads(Path(manifest_path).read_text());root=Path(input_dir).resolve()
    if not REQUIRED<=m.keys():raise ValueError('missing manifest fields: '+str(sorted(REQUIRED-m.keys())))
    for field in ('source','request','contracts','adjustments','interval','availability','session','coverage','warmup','execution_state'):
        if type(m[field]) is not dict:raise ValueError('manifest object required: '+field)
    if type(m['files']) is not list:raise ValueError('manifest files array required')
    if m['schema_version']!=1 or m['classification']!='DEVELOPMENT_2025':raise ValueError('only explicitly admitted development 2025 is supported')
    if m['format'] not in ('native-minute-v1','polls-v1'):raise ValueError('unknown input format')
    for field in REQUIRED-{'execution_state','schema_version'}:
        if m[field] is None or m[field]=='' or m[field]=={} or m[field]==[]:raise ValueError('empty admission field: '+field)
    if not {'start','end_exclusive'}<=m['interval'].keys():raise ValueError('interval boundaries required')
    start=datetime.fromisoformat(m['interval']['start'].replace('Z','+00:00'));end=datetime.fromisoformat(m['interval']['end_exclusive'].replace('Z','+00:00'))
    if start.tzinfo is None or end.tzinfo is None or not start<end or start.astimezone(timezone.utc).year!=2025 or end.astimezone(timezone.utc)>datetime(2026,1,1,tzinfo=timezone.utc):raise ValueError('development interval only')
    for item in m['files']:
        if type(item) is not dict:raise ValueError('file pin object required')
        if set(item)!={'file','sha256'}:raise ValueError('file pin schema')
        p=(root/item['file']).resolve()
        if not p.is_relative_to(root) or digest(p)!=item['sha256']:raise ValueError('input pin mismatch')
    expected='bars.jsonl' if m['format']=='native-minute-v1' else 'polls.jsonl'
    if expected not in {x['file'] for x in m['files']}:raise ValueError('missing stream pin')
    if m['format']=='native-minute-v1':
        if 'coverage.jsonl' not in {x['file'] for x in m['files']}:raise ValueError('coverage pin required')
        if not {'bars','rows','status'}<=m['coverage'].keys() or any(type(m['coverage'][k]) is not int for k in ('bars','rows')):raise ValueError('numeric native coverage counts required')
        if start.second or start.microsecond or end.second or end.microsecond:raise ValueError('native minute-aligned interval required')
        minute=60_000_000_000;cursor=int(start.timestamp())*1_000_000_000;stop=int(end.timestamp())*1_000_000_000
        coverage={}
        with (root/'coverage.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                if type(row['start_ns']) is not int or type(row['end_ns']) is not int or row['start_ns']!=cursor or row['end_ns']!=cursor+minute:raise ValueError('coverage interval discontinuity')
                if row['coverage'] not in ('TRADED','NO_TRADE_MIXED_STATUS','NO_TRADE_OBSERVED_NONTRADING'):raise ValueError('unsupported coverage status')
                coverage[cursor]=row;cursor+=minute
        if cursor!=stop or len(coverage)!=m['coverage']['rows']:raise ValueError('coverage count/range mismatch')
        seen=set()
        with (root/'bars.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                if not {'start_ns','end_ns','availability_ns','incomplete_event','ohlcv','first','last'}<=row.keys():raise ValueError('native bar schema')
                if any(type(row[k]) is not int for k in ('start_ns','end_ns','availability_ns')):raise ValueError('native nanoseconds must be integers')
                label=row['start_ns'];ohlcv=row['ohlcv']
                if label%minute or row['end_ns']!=label+minute or row['availability_ns']<row['end_ns'] or row['availability_ns']>stop or row['incomplete_event'] is not False:raise ValueError('native bar timing/completion invalid')
                if type(ohlcv) is not list or len(ohlcv)!=5 or any(type(v) is not int for v in ohlcv) or min(ohlcv[:4])<=0 or ohlcv[4]<0 or ohlcv[2]>min(ohlcv[0],ohlcv[3]) or ohlcv[1]<max(ohlcv[0],ohlcv[3]) or ohlcv[2]>ohlcv[1]:raise ValueError('native OHLCV invalid')
                for ref in ('first','last'):
                    if type(row[ref]) is not dict or not {'file','record_index','sequence','ts_event_ns','ts_recv_ns'}<=row[ref].keys():raise ValueError('native lineage incomplete')
                if label in seen or label not in coverage or coverage[label]['coverage']!='TRADED' or coverage[label]['ohlcv']!=row['ohlcv']:raise ValueError('bar/coverage mismatch')
                seen.add(label)
        if len(seen)!=m['coverage']['bars'] or len(seen)!=sum(r['coverage']=='TRADED' for r in coverage.values()):raise ValueError('native bar count mismatch')
    return m


def events(m,root):
    start=datetime.fromisoformat(m['interval']['start'].replace('Z','+00:00'));end=datetime.fromisoformat(m['interval']['end_exclusive'].replace('Z','+00:00'))
    if m['format']=='polls-v1':
        with (root/'polls.jsonl').open() as f:
            for line in f:
                event=json.loads(line);receipt=datetime.fromisoformat(event['receipt_time'].replace('Z','+00:00'))
                if not start<=receipt<end:raise ValueError('event outside admitted interval')
                yield event
    else:
        # Stable receipt order from completed-event availability, original file/row
        # order breaks equal-time ties. Never read arbitrary annual/holdout data.
        rows=[]
        with (root/'bars.jsonl').open() as f:
            for i,line in enumerate(f):
                row=json.loads(line)
                if row['incomplete_event'] or row['availability_ns']<row['end_ns']:raise ValueError('incomplete/noncausal native bar')
                rows.append((row['availability_ns'],i,row))
        for available,i,row in sorted(rows,key=lambda x:(x[0],x[1])):
            receipt=datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=(available+999)//1000)
            label=datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(microseconds=row['start_ns']//1000)
            if not start<=label<end or not start<=receipt<=end:raise ValueError('native row outside admitted interval')
            o,h,l,c,v=row['ohlcv']
            yield dict(receipt_time=receipt.isoformat(),request_id=f'native-completed-{i}',status_code=200,bars=[dict(TimeStamp=label.isoformat(),Open=o/1e9,High=h/1e9,Low=l/1e9,Close=c/1e9,TotalVolume=v,IsEndOfHistory=True)],native_line=i+1,native=row,receipt_ns=available)


async def execute(m,root,snapshot,stream):
    adapter=Adapter(snapshot,m['execution_state']);count=accepted=intentions=poll_errors=transitions=0
    for event in events(m,root):
        trace=await adapter.poll(event);stream.write(canonical(trace))
        poll_errors+=len(trace['errors']);transitions+=sum(d.get('kind')=='decision_transition' for d in trace['decisions'])
        count+=1;accepted=trace['after']['bar_count'];intentions+=len(trace['intentions'])
    adapter.runtime.verify()
    return dict(polls=count,poll_errors=poll_errors,decision_transitions=transitions,final_consumed_history=adapter.state()['consumed_history'],final_buffer_rows=accepted,intentions=intentions,completed_simulated_trades=len(adapter.trader.completed_trades))


def run(manifest_path,input_dir,snapshot,output_dir):
    root=Path(input_dir).resolve();out=Path(output_dir)
    if out.exists() or out.is_symlink():raise ValueError('fresh output required')
    if out.resolve().is_relative_to(root) or root.is_relative_to(out.resolve()):raise ValueError('output/input overlap')
    code_paths=[Path(__file__),Path(__file__).with_name('adapter.py'),Path(__file__).resolve().parents[2]/'cli/check_yank_deployed_replay.py']
    code_hashes={str(p.name):digest(p) for p in code_paths}
    manifest_hash=digest(manifest_path);m=admit(manifest_path,root)
    out.mkdir(parents=True)
    with (out/'trace.jsonl').open('x') as stream:counts=asyncio.run(execute(m,root,snapshot,stream))
    admit(manifest_path,root)
    if digest(manifest_path)!=manifest_hash:raise ValueError('manifest changed during replay')
    if code_hashes!={str(p.name):digest(p) for p in code_paths}:raise ValueError('replay code changed during run')
    report=dict(code_sha256=code_hashes,python_version=platform.python_version(),decision='HOLD_VALIDATION',classification='DEVELOPMENT_2025',counts=counts,manifest_sha256=manifest_hash,trace_sha256=digest(out/'trace.jsonl'),
                readiness=dict(poll_processing='BLOCKED_POLL_ERRORS' if counts['poll_errors'] else 'NO_POLL_ERRORS',source='NATIVE_DIAGNOSTIC_NOT_TRADESTATION' if m['format']=='native-minute-v1' else 'PROVIDER_SEMANTICS_UNVERIFIED',coverage=m['coverage'],warmup=m['warmup'],fidelity='EXACT_SNAPSHOT_METHODS; see independent equal-state tests',account_risk=m['execution_state']['classification'],execution='NO_OBSERVED_ACKNOWLEDGEMENTS_OR_FILLS',economics='NOT_ASSESSED'),
                limitations=['Finalized native minutes do not reproduce TradeStation request/receipt/revision path','No initial account balance or broker state inferred','Source symbol MNQU26 is used for installed specs; native data contract remains MNQM5 diagnostic','Scheduler uses installed fixed UTC closure; skipped minutes remain in input evidence','Synthetic initialization: first single native bar is backfill, unlike installed 48-hour initial request; no startup fidelity claim','Nanosecond native availability retained; datetime replay clock rounds upward to next microsecond, never earlier','No terminal liquidation or rewritten archived P&L'])
    (out/'report.json').write_text(canonical(report));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--input-dir',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();run(a.manifest,a.input_dir,a.snapshot,a.output_dir)

if __name__=='__main__':main()
