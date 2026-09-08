import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('gap_checks', ROOT/'src/research/yank_deployed_validation/gaps.py')
gaps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gaps)
DTYPE=[('length','u1'),('rtype','u1'),('publisher_id','u2'),('instrument_id','u4'),('ts_event','u8'),('order_id','u8'),('price','i8'),('size','u4'),('flags','u1'),('channel_id','u1'),('action','S1'),('side','S1'),('ts_recv','u8'),('ts_in_delta','i4'),('sequence','u4')]


def test_native_arrival_invalid_recovery_chunk_boundaries(monkeypatch, tmp_path):
    audit=gaps.load_audit()
    t=audit.START+10*audit.MINUTE
    def row(action,ts,oid=0,side='N',price=0,size=0,flags=128):
        return (14,160,1,42009475,ts-1,oid,int(price*audit.SCALE),size,flags,8,action.encode(),side.encode(),ts,0,321)
    rows=[row('R',audit.START,flags=40),row('A',audit.START,1,'B',99,10,40),row('A',audit.START,2,'A',100,10,40),row('N',t-10),
          row('T',t-1,price=99,size=1,flags=0),row('N',t+2),
          row('M',t+3,1,'B',101,10),row('N',t+4),row('M',t+5,1,'B',99,10)]
    arr=np.array(rows,dtype=DTYPE)
    class Store:
        metadata=SimpleNamespace(dataset='GLBX.MDP3',symbols=['MNQM5'],ts_out=False,version=3,schema='mbo',start=audit.START,end=audit.START+audit.DAY)
        def to_ndarray(self,count):
            return (arr[i:i+count] for i in range(0,len(arr),count))
    monkeypatch.setattr(audit.db.DBNStore,'from_file',lambda _:Store())
    audit.ACQUISITION=tmp_path
    conditions=[dict(arrival_ns=t,expiry_exclusive=audit.iso(t+100))]
    results=[gaps.scan(audit,tmp_path/'fake.mbo.dbn.zst',tmp_path,conditions,n) for n in (1,2,100)]
    assert results[0]==results[1]==results[2]
    result=results[0]
    crossing=result['arrival_straddlers'][0]
    assert crossing['event']['first']['record_index']==4
    assert crossing['event']['last']['record_index']==5
    assert crossing['recovery_boundary']['ts_recv_ns']==t+2
    assert crossing['strictly_previous_event']['last']['record_index']==3
    invalid=result['invalid_runs'][0]
    assert invalid['completed_events']==2
    assert invalid['first_event']['first']['record_index']==6
    assert invalid['last_event']['last']['record_index']==7
    assert invalid['recovery_event']['last']['record_index']==8
    assert invalid['recovery_event']['valid']
    assert result['gap_counts']=={'locked_or_crossed_book':2}
    assert result['reconstruction_boundary']['action']=='R'


def test_nanosecond_status_boundary_preserved():
    assert gaps.ns('2025-05-28T20:20:05.063483407Z')-gaps.ns('2025-05-28T20:20:05.063483000Z')==407


def test_inventory_full_window_overlaps_and_dispositions():
    c=dict(arrival_ns=10,expiry_exclusive='1970-01-01T00:00:00.000000100Z',convention='end',delay_ms=0,outcome='unassessable',
           evidence_gaps=['invalid_completed_book_or_event','native_evidence_gaps_in_pending_interval'],schedule={'scheduled_count':240},missing_scheduled_minutes=[],status_intervals=[])
    run=dict(first_event={'first':{'ts_recv_ns':5}},last_event={'last':{'ts_recv_ns':20}},recovery_event={'last':{'ts_recv_ns':200}})
    report=dict(cases=[dict(case_id='case-3',conditions=[c])],status_records=[])
    scan=dict(source='native',arrival_straddlers=[],invalid_runs=[run])
    row=gaps.inventory(report,[scan],{'native':'hash'})[0]
    assert row['full_window_ns']==[10,100]
    assert row['blockers'][0]['full_window_intersection_ns']==[10,21]
    assert len(row['blockers'])==1  # two overlapping labels, one native blocker
    assert row['frozen_outcome']=='unassessable'
    assert row['queue']['disposition']=='remains unobservable'


@pytest.mark.parametrize('symlink',[False,True])
def test_existing_and_symlink_outputs_rejected(tmp_path,symlink):
    out=tmp_path/'output'
    if symlink: out.symlink_to(tmp_path/'absent',target_is_directory=True)
    else: out.mkdir()
    with pytest.raises(ValueError,match='fresh'):
        gaps.run(tmp_path/'manifest',tmp_path/'input',out)


def test_cli_private_loading_no_credentials_or_network():
    code='''
import builtins, runpy, socket, sys
orig=builtins.__import__
def guarded(name,*a,**kw):
    if name.startswith(('src.research','src.data.auth','httpx')):
        raise AssertionError(name)
    return orig(name,*a,**kw)
builtins.__import__=guarded
socket.socket=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network'))
sys.argv=['check_yank_execution_gaps','--help']
runpy.run_path('src/cli/check_yank_execution_gaps.py',run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
