import json
from pathlib import Path

import pytest

from src.research.yank_execution_pilot import audit
from src.research.yank_execution_pilot.core import MINUTE,SCALE

SIGNAL=audit.START+10*MINUTE


def prepared():
    case=dict(signal_time=audit.iso(SIGNAL),entry=100,quantity=-5)
    labels=[SIGNAL+i*MINUTE for i in range(241)]
    scenario=audit.Scenario(case,labels,'end',0)
    scenario.arrival_book={'bid':99}
    scenario.through={'record_index':1}
    scenario.observed_minutes=set(range(SIGNAL//MINUTE,scenario.expiry//MINUTE))
    bars={t:[100*SCALE]*4+[10] for t in labels[1:]}
    status=audit.Status([dict(ts_recv_ns=SIGNAL,source='status',record_index=0,is_trading='Y')])
    return scenario,bars,status,audit.Gaps([SIGNAL,SIGNAL+1,scenario.expiry])


@pytest.mark.parametrize('fault',[None,'halt','missing','native','arrival'])
def test_scenario_assessability_control_and_each_gap(fault):
    s,bars,status,gaps=prepared()
    if fault=='halt': status.rows[0]['is_trading']='N'
    if fault=='missing': s.observed_minutes.remove(SIGNAL//MINUTE+1)
    if fault=='native': gaps.add('bad', 'source',0,SIGNAL+1)
    if fault=='arrival': s.arrival_book=None
    result=s.result(status,gaps,bars)
    assert result['outcome']==('supported' if fault is None else 'unassessable')


def test_exact_gap_boundaries_exclude_prearrival_and_expiry():
    s,bars,status,gaps=prepared()
    for i,t in enumerate([SIGNAL-1,SIGNAL,s.expiry,s.expiry+1]): gaps.add('outside','source',i,t)
    assert s.result(status,gaps,bars)['outcome']=='supported'
    gaps.add('inside','source',10,SIGNAL+1)
    assert gaps.overlaps(SIGNAL,s.expiry)==['inside']


def test_reconciliation_both_mappings_deltas_and_missing(tmp_path):
    bars={SIGNAL:[100*SCALE]*4+[10],SIGNAL+MINUTE:[101*SCALE]*4+[12],SIGNAL+2*MINUTE:[102*SCALE]*4+[20]}
    native={SIGNAL:[101*SCALE,103*SCALE,99*SCALE,102*SCALE,15,4]}
    summary=audit.reconciliation(bars,native,native,tmp_path)
    rows=[json.loads(line) for line in (tmp_path/'reconciliation.jsonl').read_text().splitlines()]
    start=next(r for r in rows if r['clock']=='capture' and r['convention']=='start' and r['original_label']==audit.iso(SIGNAL))
    end=next(r for r in rows if r['clock']=='capture' and r['convention']=='end' and r['original_label']==audit.iso(SIGNAL+MINUTE))
    assert start['ohlc_delta']==[1,3,-1,2] and start['volume_delta']==5
    assert end['ohlc_delta']==[0,2,-2,1] and end['volume_delta']==3
    assert start['native_minute_start']==end['native_minute_start']==audit.iso(SIGNAL)
    assert summary['capture/start']['missing_native_T_minutes']==2
    assert summary['capture/end']['missing_native_T_minutes']==2


@pytest.mark.parametrize('fault',['halt','native','coverage'])
def test_timeline_intervening_gap_qualifies_endpoint_ordering(fault):
    status=audit.Status([dict(ts_recv_ns=SIGNAL,source='s',record_index=0,is_trading='Y')])
    gaps=audit.Gaps([SIGNAL,SIGNAL+2*MINUTE])
    entry={'event_id':'a','ts_recv_ns':SIGNAL+1,'completed_event_valid':True}
    stop={'event_id':'b','ts_recv_ns':SIGNAL+MINUTE+1,'completed_event_valid':True}
    w=dict(start=SIGNAL,end=SIGNAL+2*MINUTE,first_entry=entry,first_strict_entry=entry,first_stop=stop,first_target=None,observed_minutes={SIGNAL//MINUTE,SIGNAL//MINUTE+1})
    if fault=='halt': status.rows[0]['is_trading']='N'
    if fault=='native': gaps.add('bad','s',1,SIGNAL+10)
    if fault=='coverage': w['observed_minutes'].pop()
    audit.finalize_timeline(w,status,gaps)
    assert w['entry_stop_ordering']=='unassessable_interval_gaps'
    assert w['entry_stop_ordering_raw_endpoint_comparison']=='entry_evidence_before_barrier_not_fill_proof'
    assert w['assessability_gaps']


@pytest.mark.parametrize('failure',[ValueError,KeyboardInterrupt])
def test_final_integrity_failure_cannot_publish_PASS(monkeypatch,tmp_path,failure):
    calls=0
    def inputs():
        nonlocal calls
        calls+=1
        if calls==2: raise failure('final verification failed')
        return [],[],{},[],{}
    monkeypatch.setattr(audit,'load_inputs',inputs)
    monkeypatch.setattr(audit,'read_auxiliary',lambda files:(audit.Status([]),[]))
    output=tmp_path/'result'
    with pytest.raises(failure): audit.run(output)
    assert (output/'FAILED').exists()
    assert not any((output/name).exists() for name in ('report.json','report.md','artifacts.json'))


def test_interrupt_during_publication_cleans_partial_PASS(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,'load_inputs',lambda:([],[],{},[],{}))
    monkeypatch.setattr(audit,'read_auxiliary',lambda files:(audit.Status([]),[]))
    original=Path.write_text
    def interrupted(self,*args,**kwargs):
        if self.name=='report.md': raise KeyboardInterrupt()
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',interrupted)
    output=tmp_path/'result'
    with pytest.raises(KeyboardInterrupt): audit.run(output)
    assert not (output/'report.json').exists()
    assert not (output/'artifacts.json').exists()


@pytest.mark.parametrize('decoded,expected', [(True,'Y'), (False,'N'), (None,'~')])
def test_real_DBN_status_boolean_api_normalized_before_assessment(monkeypatch,decoded,expected):
    import databento_dbn as dbn
    from types import SimpleNamespace
    record=dbn.StatusMsg(publisher_id=1,instrument_id=42009475,
        ts_event=SIGNAL-1,ts_recv=SIGNAL,is_trading=dbn.TriState(expected),is_quoting=dbn.TriState(expected))
    class Store:
        metadata=SimpleNamespace(dataset='GLBX.MDP3',symbols=['MNQM5'],
            ts_out=False,version=3,schema='status',start=SIGNAL)
        def __iter__(self):return iter([record])
    monkeypatch.setattr(audit.db.DBNStore,'from_file',lambda path:Store())
    status,_=audit.read_auxiliary([audit.ACQUISITION/'fixture.status.dbn.zst'])
    assert status.rows[0]['is_trading']==expected
    assert status.rows[0]['is_quoting']==expected
    assert bool(status.overlaps(SIGNAL,SIGNAL+MINUTE)) == (decoded is not True)
