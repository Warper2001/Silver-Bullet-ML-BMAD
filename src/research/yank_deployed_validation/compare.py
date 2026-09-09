"""Offline decision capture replay; equivalent state is required for fidelity claims."""
import argparse
import asyncio
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from .adapter import Adapter,canonical,digest
from .capture import identities


def differences(expected,actual,path=''):
    exact=[];floats=[]
    if isinstance(expected,dict) and isinstance(actual,dict):
        for key in sorted(set(expected)|set(actual)):
            p=path+'.'+key
            if key not in expected or key not in actual:exact.append(dict(path=p,expected=expected.get(key),actual=actual.get(key)))
            else:
                e,f=differences(expected[key],actual[key],p);exact+=e;floats+=f
    elif isinstance(expected,list) and isinstance(actual,list):
        if len(expected)!=len(actual):exact.append(dict(path=path+'.length',expected=len(expected),actual=len(actual)))
        for i,(a,b) in enumerate(zip(expected,actual)):
            e,f=differences(a,b,path+f'[{i}]');exact+=e;floats+=f
    elif expected!=actual:
        item=dict(path=path,expected=expected,actual=actual)
        if path.rsplit('.',1)[-1] in ('entry_price','sl_price','tp_price','exit_price'):
            try:
                e=Decimal(str(expected));a=Decimal(str(actual))
                if not e.is_finite() or not a.is_finite():raise ValueError('nonfinite price')
                item['expected_ticks']=str(e/Decimal('.25'));item['actual_ticks']=str(a/Decimal('.25'))
            except (InvalidOperation,TypeError,ValueError):item['invalid_price_value']=True
            exact.append(item)
        elif path.rsplit('.',1)[-1] in ('quantity','contracts','bar_count','bars_held','sequence'):exact.append(item)
        elif ('.features.' in path or path.rsplit('.',1)[-1]=='probability') and isinstance(expected,float) and isinstance(actual,(float,int)):floats.append(item)
        else:exact.append(item)
    return exact,floats


async def replay_capture(rows,manifest,snapshot):
    expected_identity=identities(snapshot);results=[];observations=[];expected_sequence=1
    try:adapter=Adapter(snapshot,manifest['initial_state'])
    except ValueError as e:return dict(results=[dict(status='UNASSESSABLE',reason=str(e))],broker_observations=[])
    for row in rows:
        if row.get('schema_version')!=1:raise ValueError('capture schema')
        if row['kind']=='broker_observation':observations.append(row);continue
        if row['kind']!='poll':raise ValueError('capture kind')
        trace=row['trace'];reasons=[]
        if trace.get('sequence')!=expected_sequence:reasons.append('capture_sequence_gap_or_duplicate')
        expected_sequence+=1
        scope=row['readiness'].get('identity_verification_scope')
        if scope!='PRIVATE_PINNED_OFFLINE_RUNTIME':reasons.append('live_identity_unverified')
        if adapter.account_verification=='UNVERIFIED_EXTERNAL_DECLARATION':reasons.append('account_evidence_unverified')
        if row['identities']!=expected_identity:reasons.append('identity_mismatch')
        if not row['readiness'].get('equivalent_feed_and_state',False):reasons.append('feed_or_state_unavailable')
        clock_reads=trace['input'].get('clock_reads')
        original_now=adapter.runtime.Clock.now
        remaining=[]
        if clock_reads:
            from datetime import datetime
            if trace['input'].get('poll_observation')!='already_admitted':reasons.append('scheduler_boundary_unavailable')
            remaining=[datetime.fromisoformat(v) for v in clock_reads]
            def observed_now(cls,tz=None):
                if not remaining:raise RuntimeError('clock evidence exhausted')
                value=remaining.pop(0)
                return value.astimezone(tz) if tz else value.replace(tzinfo=None)
            adapter.runtime.Clock.now=classmethod(observed_now)
        elif row['readiness'].get('clock_semantics')!='synthetic_constant':reasons.append('clock_observations_unavailable')
        if trace['input'].get('poll_observation')=='already_admitted':adapter.trader._ts_client.require_replies=True
        try:actual=await adapter.poll(trace['input'],already_admitted=trace['input'].get('poll_observation')=='already_admitted')
        except (ValueError,RuntimeError) as e:
            results.append(dict(sequence=trace.get('sequence'),status='UNASSESSABLE',reasons=[str(e)]));continue
        finally:adapter.runtime.Clock.now=original_now
        if remaining:reasons.append('unused_clock_observations')
        if trace['before']!=actual['before']:reasons.append('predecision_state_differs')
        if trace.get('errors') or actual.get('errors'):reasons.append('poll_error')
        exact=[];floats=[]
        for field in ('decisions','intentions','after','scheduled'):
            e,f=differences(trace[field],actual[field],field);exact+=e;floats+=f
        results.append(dict(sequence=trace['sequence'],status='UNASSESSABLE' if reasons else ('MISMATCH' if exact else ('MATCH_DISCRETE_WITH_FLOAT_DIFFERENCES' if floats else 'MATCH')),reasons=reasons,verification_scope='PRIVATE_OFFLINE_DIAGNOSTIC_ONLY',exact_differences=exact,float_differences=floats))
    adapter.runtime.verify()
    return dict(results=results,broker_observations=observations)


def run(manifest_path,input_dir,snapshot,output_dir):
    manifest_hash=digest(manifest_path)
    code_paths=[Path(__file__),Path(__file__).with_name('capture.py'),Path(__file__).with_name('adapter.py')]
    code_hashes={p.name:digest(p) for p in code_paths}
    m=json.loads(Path(manifest_path).read_text());root=Path(input_dir).resolve();out=Path(output_dir)
    if not {'capture_sha256','coverage_sha256','initial_state'}<=m.keys():raise ValueError('capture manifest schema')
    if out.exists() or out.is_symlink() or out.resolve().is_relative_to(root) or root.is_relative_to(out.resolve()):raise ValueError('fresh separate output required')
    def verify():
        for name,key in [('capture.jsonl','capture_sha256'),('coverage.json','coverage_sha256')]:
            p=root/name
            if not p.resolve().is_relative_to(root) or digest(p)!=m[key]:raise ValueError('capture pin mismatch')
    verify();coverage=json.loads((root/'coverage.json').read_text())
    rows=[json.loads(line) for line in (root/'capture.jsonl').read_text().splitlines()]
    if not coverage.get('valid_coverage') or coverage.get('capture_sha256')!=m['capture_sha256'] or coverage.get('written')!=len(rows):
        result=dict(results=[dict(status='UNASSESSABLE',reason='invalid_capture_coverage')],broker_observations=[])
    else:result=asyncio.run(replay_capture(rows,m,snapshot))
    verify()
    if digest(manifest_path)!=manifest_hash or code_hashes!={p.name:digest(p) for p in code_paths}:raise ValueError('comparison inputs changed')
    result.update(manifest_sha256=manifest_hash,code_sha256=code_hashes,decision='HOLD_VALIDATION',capture_sha256=m['capture_sha256'],economics='NOT_ASSESSED',execution='Observed venue records reported separately; no fills inferred from intentions')
    out.mkdir(parents=True);(out/'comparison.json').write_text(canonical(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--input-dir',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();run(a.manifest,a.input_dir,a.snapshot,a.output_dir)

if __name__=='__main__':main()
