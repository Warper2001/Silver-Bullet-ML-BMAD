import copy
import json
from pathlib import Path
import pytest
from src.research.yank_deployed_validation.replay import admit,events,run
from src.research.yank_deployed_validation.adapter import digest

ROOT=Path(__file__).resolve().parents[2]


def fixture(tmp_path):
    data=tmp_path/'input';data.mkdir()
    row=dict(receipt_time='2025-05-19T00:01:00Z',request_id='p',bars=[],status_code=200)
    (data/'polls.jsonl').write_text(json.dumps(row)+'\n')
    m=json.loads((ROOT/'docs/yank-validation/native-admission.json').read_text());m['format']='polls-v1';m['files']=[dict(file='polls.jsonl',sha256=digest(data/'polls.jsonl'))]
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m));return data,p,m


def test_manifest_required_and_pins(tmp_path):
    data,p,m=fixture(tmp_path);assert admit(p,data)['format']=='polls-v1'
    del m['availability'];p.write_text(json.dumps(m))
    with pytest.raises(ValueError):admit(p,data)


def test_replay_deterministic_fresh_and_pin_failure(tmp_path):
    data,p,m=fixture(tmp_path);snap=ROOT/'docs/yank-validation/snapshot/v1'
    a=run(p,data,snap,tmp_path/'a');b=run(p,data,snap,tmp_path/'b')
    assert a==b and a['decision']=='HOLD_VALIDATION'
    with pytest.raises(ValueError):run(p,data,snap,tmp_path/'a')
    (data/'polls.jsonl').write_text('changed')
    with pytest.raises(ValueError):admit(p,data)


def test_observed_state_cannot_default_to_flat(tmp_path):
    from src.research.yank_deployed_validation.adapter import Adapter
    _,_,m=fixture(tmp_path);m['execution_state']['classification']='OBSERVED_STATE'
    with pytest.raises(ValueError):Adapter(ROOT/'docs/yank-validation/snapshot/v1',m['execution_state'])


def test_private_cli_executes_tiny_stream_with_import_and_network_traps(tmp_path):
    import subprocess,sys
    data,p,m=fixture(tmp_path)
    script="""
import builtins,importlib.util,socket,sys
original=builtins.__import__
def guarded(name,*a,**kw):
    if name.startswith(('src.research','src.data.auth','httpx')):raise RuntimeError('live infrastructure import: '+name)
    return original(name,*a,**kw)
builtins.__import__=guarded
socket.create_connection=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('network attempted'))
spec=importlib.util.spec_from_file_location('isolated_cli',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.load_tool('replay').run(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
"""
    result=subprocess.run([sys.executable,'-c',script,str(ROOT/'src/cli/check_yank_deployed_replay.py'),str(p),str(data),str(ROOT/'docs/yank-validation/snapshot/v1'),str(tmp_path/'isolated')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'isolated/report.json').exists()


def test_valid_native_replay_receipts_prices_ties_and_late_delivery(tmp_path):
    """Complete native events never become available before their integer ns time."""
    from datetime import datetime, timezone
    data,p,m=fixture(tmp_path)
    start=int(datetime(2025,5,19,tzinfo=timezone.utc).timestamp())*1_000_000_000
    minute=60_000_000_000
    rows=[]
    for i in range(3):
        ref=dict(file='pinned-native',record_index=i,sequence=i,
                 ts_event_ns=start+i*minute,ts_recv_ns=start+i*minute+1)
        rows.append(dict(start_ns=start+i*minute,end_ns=start+(i+1)*minute,
                         availability_ns=start+3*minute+(1 if i==0 else 0),
                         incomplete_event=False,ohlcv=[100_250_000_000,101_500_000_000,99_750_000_000,100_500_000_000,7+i],
                         first=ref,last=ref))
    # Equal-time order must follow native stream order, even when labels go back.
    ordered=[rows[0],rows[2],rows[1]]
    (data/'bars.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ordered))
    coverage=[dict(start_ns=start+i*minute,end_ns=start+(i+1)*minute,
                   coverage='TRADED' if i<3 else 'NO_TRADE_OBSERVED_NONTRADING',
                   ohlcv=rows[i]['ohlcv'] if i<3 else None) for i in range(4)]
    (data/'coverage.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in coverage))
    m.update(format='native-minute-v1',interval=dict(start='2025-05-19T00:00:00Z',end_exclusive='2025-05-19T00:04:00Z'),
             coverage=dict(bars=3,rows=4,status='synthetic timing test'),
             files=[dict(file=name,sha256=digest(data/name)) for name in ('bars.jsonl','coverage.jsonl')])
    p.write_text(json.dumps(m))
    result=run(p,data,ROOT/'docs/yank-validation/snapshot/v1',tmp_path/'native')
    trace=[json.loads(line) for line in (tmp_path/'native/trace.jsonl').read_text().splitlines()]
    assert [t['input']['native_line'] for t in trace]==[2,3,1]
    assert [t['input']['receipt_time'] for t in trace]==['2025-05-19T00:03:00+00:00','2025-05-19T00:03:00+00:00','2025-05-19T00:03:00.000001+00:00']
    assert [t['input']['receipt_ns'] for t in trace]==[start+3*minute,start+3*minute,start+3*minute+1]
    for t in trace:
        b=t['input']['bars'][0]
        assert [b[k] for k in ('Open','High','Low','Close')]==[100.25,101.5,99.75,100.5]
    assert [t['after']['bar_count'] for t in trace]==[1,1,1]
    assert result['counts']['poll_errors']==0


def test_development_exclusive_year_boundary(tmp_path):
    data,p,m=fixture(tmp_path)
    m['interval']['end_exclusive']='2026-01-01T00:00:00Z';p.write_text(json.dumps(m))
    assert admit(p,data)['classification']=='DEVELOPMENT_2025'
    m['interval']['end_exclusive']='2026-01-01T00:00:00.000001Z';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='development interval'):
        admit(p,data)
