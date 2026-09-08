from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.research.yank_execution_pilot import audit
from src.research.yank_execution_pilot.core import MINUTE, SCALE

DTYPE=[('length','u1'),('rtype','u1'),('publisher_id','u2'),('instrument_id','u4'),('ts_event','u8'),('order_id','u8'),('price','i8'),('size','u4'),('flags','u1'),('channel_id','u1'),('action','S1'),('side','S1'),('ts_recv','u8'),('ts_in_delta','i4'),('sequence','u4')]
SIGNAL=audit.START+10*MINUTE


def row(action, ts, oid=0, side='N', price=0, size=0, flags=128):
    return (14,160,1,42009475,ts-1,oid,int(price*SCALE),size,flags,8,action.encode(),side.encode(),ts,0,1)


def initial():
    return [row('R',audit.START,flags=40),row('A',audit.START,1,'B',99,10,40),row('A',audit.START,2,'A',100,10,40),row('N',SIGNAL-1)]


def execute(monkeypatch, rows, count, scenarios=True, timelines=True, expiry=None):
    arr=np.array(rows,dtype=DTYPE)
    class Store:
        metadata=SimpleNamespace(dataset='GLBX.MDP3', symbols=['MNQM5'],ts_out=False,version=3,schema='mbo',start=audit.START,end=audit.START+audit.DAY)
        def to_ndarray(self,count):
            return (arr[i:i+count] for i in range(0,len(arr),count))
    monkeypatch.setattr(audit.db.DBNStore,'from_file',lambda _:Store())
    case=dict(case_id='case-1',signal_time=audit.iso(SIGNAL),entry=100,stop=101,target=90,quantity=-5)
    labels=[SIGNAL+i*MINUTE for i in range(242)]
    conditions=[audit.Scenario(case,labels,'end',delay) for delay in (0,100,500)] if scenarios else []
    windows=[dict(case_id='case-1',convention='end',start=SIGNAL,end=SIGNAL+MINUTE,entry=100*SCALE,stop=101*SCALE,target=90*SCALE,first_entry=None,first_strict_entry=None,first_stop=None,first_target=None,first_stop_after_entry=None,first_target_after_entry=None,trades=0)]
    if not timelines: windows=[]
    if expiry is not None:
        for scenario in conditions: scenario.expiry=expiry
    recv={};exchange={};gaps=audit.Gaps();extract=StringIO()
    result=audit.stream_file(audit.ACQUISITION/'test.mbo.dbn.zst',conditions,recv,exchange,gaps,windows,extract,count)
    return conditions,recv,gaps,result,windows,extract.getvalue()


def test_split_F_LAST_chunk_determinism_trade_fill_and_arrival(monkeypatch):
    rows=initial()+[row('T',SIGNAL,price=100,size=1),row('T',SIGNAL+100_000_000,price=100,size=2,flags=0),row('F',SIGNAL+100_000_000,1,'B',99,2,0),row('C',SIGNAL+100_000_000,1,'B',99,2),row('T',SIGNAL+600_000_000,price=101,size=3)]
    a=execute(monkeypatch,rows,2)
    b=execute(monkeypatch,rows,100)
    assert a[1]==b[1]
    assert a[2].counts==b[2].counts
    assert a[5]==b[5]
    assert [s.at+s.above for s in a[0]]==[5,3,3]
    assert a[1][SIGNAL][4]==6  # F is not trade volume
    assert a[0][0].arrival_book['bid_size']==10
    assert a[0][1].arrival_book['bid_size']==10  # equality trade excluded; strictly prior complete quote retained
    assert a[0][2].arrival_book['bid_size']==8


def test_incomplete_snapshot_is_gap_not_trade_observation(monkeypatch):
    conditions,recv,gaps,_,_,_=execute(monkeypatch,initial()[:3],1)
    assert not recv
    assert gaps.counts['incomplete_final_event_or_snapshot']
    assert all(s.arrival_book is None for s in conditions)


def test_incomplete_event_does_not_enter_reconciliation(monkeypatch):
    conditions,recv,gaps,_,_,_=execute(monkeypatch,initial()+[row('T',SIGNAL+1,price=102,size=4,flags=0)],3)
    assert not recv
    assert gaps.counts['incomplete_final_event_or_snapshot']
    assert conditions[0].through is None


def test_invalid_native_fields_on_non_case_days_are_gaps(monkeypatch):
    rows=initial()+[row('T',SIGNAL,price=0,size=0),row('T',SIGNAL-10,price=101,size=1),row('N',SIGNAL+1,flags=129)]
    _,_,gaps,_,_,_=execute(monkeypatch,rows,100,False,False)
    assert gaps.counts['invalid_trade_fields']
    assert gaps.counts['capture_time_regression']
    assert gaps.counts['unsupported_flags']


def test_same_event_crossings_remain_ambiguous_and_invalid_event_unassessable(monkeypatch):
    rows=initial()+[row('T',SIGNAL+1,price=101,size=1,flags=0),row('M',SIGNAL+2,999,'B',99,2)]
    conditions,_,gaps,_,windows,_=execute(monkeypatch,rows,2)
    w=windows[0]
    assert gaps.counts['unknown_order_update']
    assert audit.ordering(w['first_entry'],w['first_stop'])=='unassessable_invalid_completed_event'
    assert conditions[0].through is None


def test_240th_scheduled_opportunity_not_historical_fill_is_included(monkeypatch):
    last=SIGNAL+240*MINUTE
    rows=initial()+[row('T',last-1,price=101,size=2),row('T',last,price=102,size=3)]
    conditions,_,_,_,_,_=execute(monkeypatch,rows,3)
    assert conditions[0].above==2
    assert conditions[0].through['ts_recv_ns']==last-1


def test_event_straddling_expiry_keeps_prior_trade(monkeypatch):
    last=SIGNAL+240*MINUTE
    rows=initial()+[row('T',last-1,price=101,size=2,flags=0),row('N',last+1)]
    conditions,_,_,_,_,_=execute(monkeypatch,rows,2)
    assert conditions[0].above==2


def test_event_straddling_arrival_reports_unavailable_quote(monkeypatch):
    rows=initial()+[row('T',SIGNAL+50_000_000,price=100,size=2,flags=0),row('N',SIGNAL+200_000_000)]
    conditions,_,_,_,_,_=execute(monkeypatch,rows,2)
    assert conditions[1].arrival_book is None
    assert 'arrival_during_incomplete_event' in conditions[1].gaps


@pytest.mark.parametrize('active',[True,False])
def test_malformed_gap_diagnostics_are_chunk_independent(monkeypatch,active):
    rows=initial()+[row('T',SIGNAL+20,price=100,size=1),row('T',SIGNAL+10,price=100,size=1),row('T',SIGNAL+5,price=100,size=1),row('A',SIGNAL+30,3,'N',0,0),row('N',SIGNAL+40,flags=129)]
    a=execute(monkeypatch,rows,1,active,active)[2]
    b=execute(monkeypatch,rows,100,active,active)[2]
    assert a.counts==b.counts
    assert a.samples==b.samples
    assert a.intervals==b.intervals
    regressions=[r for r in a.samples if r['reason']=='capture_time_regression']
    assert [(r['record_index'],r['ts_recv_ns']) for r in regressions]==[(5,SIGNAL+10),(6,SIGNAL+5)]
    assert a.counts['capture_time_regression']==2
    assert a.counts['invalid_order_fields']==1


def test_timeline_without_pending_scenario_is_still_replayed(monkeypatch):
    result=execute(monkeypatch,initial()+[row('T',SIGNAL+1,price=100,size=1),row('T',SIGNAL+2,price=101,size=1)],2,False)
    assert result[3]['book_replayed']
    assert result[4][0]['first_entry']['ts_recv_ns']==SIGNAL+1
    assert result[4][0]['first_stop']['ts_recv_ns']==SIGNAL+2


def test_straddling_expiry_credits_trade_minute_in_result(monkeypatch):
    last=SIGNAL+240*MINUTE
    result=execute(monkeypatch,initial()+[row('T',last-1,price=101,size=2,flags=0),row('N',last+1)],2)
    scenario=result[0][0]
    scenario.observed_minutes.update(range(SIGNAL//MINUTE,(last-1)//MINUTE))
    status=audit.Status([dict(ts_recv_ns=SIGNAL,source='status',record_index=0,is_trading='Y')])
    bars={SIGNAL+i*MINUTE:[0]*5 for i in range(1,241)}
    final=scenario.result(status,result[2],bars)
    assert final['missing_scheduled_minutes']==[]
    assert final['outcome']=='supported'


def test_bad_event_contaminates_raw_reconciliation_minute(monkeypatch,tmp_path):
    result=execute(monkeypatch,initial()+[row('T',SIGNAL+1,price=101,size=1,flags=0),row('M',SIGNAL+2,999,'B',99,1)],2)
    audit.reconciliation({SIGNAL:[100*SCALE]*4+[3]},result[1],{},tmp_path,result[2])
    import json
    row_=next(json.loads(line) for line in (tmp_path/'reconciliation.jsonl').read_text().splitlines() if json.loads(line)['clock']=='capture' and json.loads(line)['convention']=='start')
    assert row_['native_ohlc']==[101]*4
    assert row_['coverage_qualification']=='contaminated_raw_T'
    assert 'unknown_order_update' in row_['gap_reasons']


def test_timeline_keeps_crossings_after_pending_expiry(monkeypatch):
    result=execute(monkeypatch,initial()+[row('T',SIGNAL+2,price=101,size=1)],2,expiry=SIGNAL+1)
    assert result[0][0].through is None
    assert result[4][0]['first_stop']['ts_recv_ns']==SIGNAL+2
