from pathlib import Path
import json
import numpy as np
import pytest

from src.research.yank_execution_pilot.core import Book, Record, MINUTE, SCALE, opportunities, outcome, ordering
from src.research.yank_execution_pilot.audit import aggregate, canonical, verify, output_path, Scenario, Status, Gaps


def record(action, oid=0, side='N', price=0, size=0, flags=128, ts=100):
    return Record(0, ts, ts-1, action, side, oid, price*SCALE, size, flags, 1)


def book():
    b=Book()
    b.apply(record('R',flags=40))
    b.apply(record('A',1,'B',100,10,40))
    b.apply(record('A',2,'A',101,8,40))
    assert b.complete() is None
    return b


def test_snapshot_and_actions_preserve_trade_fill_separation():
    b=book()
    for action in ('T','F','N'):
        assert b.apply(record(action,1,'B',100,3)) is None
    assert b.orders[1][2] == 10
    b.apply(record('C',1,'B',100,3))
    assert b.orders[1][2] == 7
    b.apply(record('M',1,'B',99,6))
    assert b.orders[1] == ('B',99*SCALE,6)
    assert b.complete() is None
    b.apply(record('C',1,'B',99,6))
    assert 1 not in b.orders
    assert b.complete() == 'one_sided_or_empty_book'


@pytest.mark.parametrize('r,reason',[(record('A',3,'B',99,1,flags=64),'unsupported_flags'),(record('T',flags=8),'bad_live_capture_timestamp'),(record('F',flags=40),'unsupported_snapshot_action'),(record('X'),'unsupported_action'),(record('M',9,'A',101,1),'unknown_order_update'),(record('C',1,'B',100,11),'malformed_update'),(record('A',1,'B',100,1),'duplicate_add')])
def test_invalid_book_is_not_silently_repaired(r,reason):
    b=book()
    assert b.apply(r)==reason
    assert not b.valid


def test_missing_snapshot_crossed_book_and_size():
    b=Book()
    assert b.apply(record('A',1,'B',100,1))=='missing_initial_snapshot'
    b=book()
    q=b.quote(100*SCALE,-11)
    assert q['marketable'] and q['insufficient_displayed_size']
    assert q['displayed_bid_size_at_or_above_limit']==10
    b.apply(record('A',3,'B',102,1))
    assert b.complete()=='locked_or_crossed_book'


def test_240_actual_bars_include_last_and_skip_schedule_hole():
    labels=[0]+[i*MINUTE for i in range(1,120)]+[(i+60)*MINUTE for i in range(120,250)]
    s=opportunities(labels,0,'start',1000*MINUTE)
    assert s['scheduled_count']==240
    assert s['last_opportunity']==300*MINUTE
    assert s['expiry']==301*MINUTE
    assert s['signal_completion']==MINUTE
    e=opportunities(labels,0,'end',200*MINUTE)
    assert e['expiry']==200*MINUTE and e['clipped']


def test_touch_through_gaps_and_insufficient_coverage():
    assert outcome(None,None,[],True)=='unsupported'
    assert outcome(None,{},[],True)=='unsupported'  # missing reference
    assert outcome(None,{'record':1},[],True)=='touch-only'
    assert outcome({'record':2},{'record':1},[],True)=='supported'
    assert outcome({'record':2},None,['halt'],True)=='unassessable'
    assert outcome({'record':2},None,[],False)=='unassessable'


@pytest.mark.parametrize('time,event,result',[(11,'b','entry_evidence_before_barrier_not_fill_proof'),(9,'b','barrier_before_possible_entry'),(10,'b','ambiguous_same_event_or_equal_capture_time'),(11,'a','ambiguous_same_event_or_equal_capture_time')])
def test_equal_time_or_event_ordering(time,event,result):
    assert ordering({'ts_recv_ns':10,'event_id':'a'},{'ts_recv_ns':time,'event_id':event})==result


def test_native_order_ohlc_is_independent_of_chunk_boundaries():
    a=np.array([(1,2,100,2),(2,3,103,1),(3,4,99,4),(MINUTE+1,MINUTE+2,101,3)],dtype=[('ts_recv','u8'),('ts_event','u8'),('price','i8'),('size','u4')])
    whole={}; pieces={}
    aggregate(a,whole,'ts_recv')
    for row in np.array_split(a,3): aggregate(row,pieces,'ts_recv')
    assert canonical(whole)==canonical(pieces)
    assert whole[0]==[100,103,99,99,7,3]


def test_integrity_and_fresh_output(tmp_path):
    p=tmp_path/'input';p.write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'): verify(p,'0'*64)
    with pytest.raises(ValueError,match='fresh'): output_path(tmp_path)
    with pytest.raises(ValueError,match='source/input'): output_path(Path(__file__).resolve().parents[3]/'src'/'bad-report')
    assert output_path(tmp_path/'fresh')==tmp_path/'fresh'


def test_status_unknown_and_halt_boundaries():
    status=Status([dict(ts_recv_ns=100,source='s',record_index=0,is_trading='Y'),dict(ts_recv_ns=200,source='s',record_index=1,is_trading='N'),dict(ts_recv_ns=300,source='s',record_index=2,is_trading='Y')])
    assert status.overlaps(100,200)==[]
    assert status.overlaps(200,301)[0]['state']['is_trading']=='N'
    assert status.overlaps(90,101)[0]['state'] is None
