import asyncio
import copy
import json
from pathlib import Path
import pytest
from src.research.yank_deployed_validation.adapter import Adapter,digest,canonical
from src.research.yank_deployed_validation.capture import DecisionCapture,DecisionHooks,identities
from src.research.yank_deployed_validation.compare import replay_capture,differences,run

ROOT=Path(__file__).resolve().parents[2];SNAPSHOT=ROOT/'docs/yank-validation/snapshot/v1'
STATE=dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT',daily_pnl=0.,daily_halted=False,last_trading_date=None,on_combine=True,is_backfill=True)


def capture_row():
    a=Adapter(SNAPSHOT,STATE)
    trace=asyncio.run(a.poll(dict(receipt_time='2026-09-09T13:31:00Z',request_id='safe-request-hash',status_code=200,bars=[dict(TimeStamp='2026-09-09T13:30:00Z',Open=100,High=101,Low=99,Close=100,TotalVolume=1)])))
    return dict(schema_version=1,kind='poll',trace=trace,identities=identities(SNAPSHOT),readiness={'equivalent_feed_and_state':True,'clock_semantics':'synthetic_constant','identity_verification_scope':'PRIVATE_PINNED_OFFLINE_RUNTIME'})


def test_disabled_capture_no_io_or_thread(tmp_path):
    c=DecisionCapture(output_dir=tmp_path/'absent');assert c.worker is None
    assert not DecisionHooks(c).poll_completed(trace={},identity={},readiness={})
    assert not (tmp_path/'absent').exists()


def test_capture_comparator_replays_future_poll(tmp_path):
    row=capture_row();c=DecisionCapture(enabled=True,output_dir=tmp_path/'capture')
    assert c.emit(row);summary=c.close();assert summary['valid_coverage']
    m=dict(capture_sha256=digest(tmp_path/'capture/capture.jsonl'),coverage_sha256=digest(tmp_path/'capture/coverage.json'),initial_state=STATE)
    p=tmp_path/'manifest.json';p.write_text(canonical(m))
    result=run(p,tmp_path/'capture',SNAPSHOT,tmp_path/'out')
    assert result['results'][0]['status']=='MATCH'
    assert result['decision']=='HOLD_VALIDATION'


def test_mismatch_vs_unavailable_vs_float():
    row=capture_row();changed=copy.deepcopy(row);changed['trace']['after']['bar_count']=2
    result=asyncio.run(replay_capture([changed],{'initial_state':STATE},SNAPSHOT))
    assert result['results'][0]['status']=='MISMATCH'
    changed['readiness']['equivalent_feed_and_state']=False
    assert asyncio.run(replay_capture([changed],{'initial_state':STATE},SNAPSHOT))['results'][0]['status']=='UNASSESSABLE'
    exact,floats=differences({'entry_price':100.,'probability':.6,'quantity':2},{'entry_price':100.25,'probability':.60000001,'quantity':3})
    assert len(exact)==2 and len(floats)==1
    assert float(exact[0]['expected_ticks'])==400


def test_capture_contention_oversize_credentials_and_writer_failure(tmp_path,monkeypatch):
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'capture',max_nodes=20,max_bytes=200)
    c.lock.acquire()
    try:assert not c.emit({'kind':'poll'})
    finally:c.lock.release()
    assert not c.emit({'Authorization':'secret'})
    assert not c.emit({'long':'a'*300})
    summary=c.close();assert not summary['valid_coverage'] and summary['dropped']==3
    c2=DecisionCapture(enabled=True,output_dir=tmp_path/'failed')
    c2.invalid_reasons.add('writer_failure')
    assert not c2.close()['valid_coverage']


def test_overflow_is_permanent_and_venues_separate(tmp_path):
    # Deterministic queue overflow without a racing worker by stubbing drain.
    original=DecisionCapture._drain
    DecisionCapture._drain=lambda self:self.wake.wait(.05)
    try:
        c=DecisionCapture(enabled=True,output_dir=tmp_path/'bounded',capacity=1)
        assert c.emit({'kind':'first'})
        assert not c.emit({'kind':'second'})
        assert 'queue_overflow' in c.close()['invalid_reasons']
    finally:DecisionCapture._drain=original
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'venues');hooks=DecisionHooks(c)
    assert hooks.broker_observed(venue='ProjectX',kind='fill',receipt_time='2026-09-09T13:31:00Z',observation={'quantity':1,'price':100.25})
    assert hooks.broker_observed(venue='TradeStation_SIM',kind='acknowledgement',receipt_time='2026-09-09T13:31:01Z',observation={'order_id':'observed-sim'})
    assert c.close()['valid_coverage']
    rows=[json.loads(x) for x in (tmp_path/'venues/capture.jsonl').read_text().splitlines()]
    assert [x['venue'] for x in rows]==['ProjectX','TradeStation_SIM']


def test_disabled_installer_and_enabled_observer_preserve_poller(tmp_path):
    from src.research.yank_deployed_validation.capture import install_poll_observer
    from src.research.yank_deployed_validation.adapter import PrivateSnapshot
    from datetime import datetime,timezone
    runtime=PrivateSnapshot(SNAPSHOT);runtime.clock=datetime(2025,5,19,13,5,tzinfo=timezone.utc)
    trader=runtime.construct(STATE)
    reader=object.__new__(Adapter);reader.trader=trader;reader._buffer_cache_key=None;reader._buffer_hash=None;reader._warmup={}
    original=trader._poll_and_process
    install_poll_observer(trader,DecisionCapture(),identity={},readiness={},state_reader=reader.state)()
    assert trader._poll_and_process==original
    class Auth:
        async def authenticate(self):return 'offline'
    class Response:
        status_code=200
        def json(self):return {'Bars':[dict(TimeStamp='2025-05-19T13:00:00Z',Open=100,High=101,Low=99,Close=100,TotalVolume=1)]}
    class Client:
        async def get(self,*args,**kwargs):return Response()
    trader.auth=Auth();trader.client=Client()
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'hooked')
    rollback=install_poll_observer(trader,c,identity=identities(SNAPSHOT),readiness={'equivalent_feed_and_state':True},state_reader=reader.state)
    asyncio.run(trader._poll_and_process());rollback();summary=c.close()
    assert summary['valid_coverage']
    assert trader._poll_and_process==original
    assert trader._shadow_logger is not trader._shadow_trade_logger
    row=json.loads((tmp_path/'hooked/capture.jsonl').read_text())
    assert len(row['trace']['input']['clock_reads'])==1
    label=row['trace']['input']['label_evidence'][0]
    assert label['semantics']=='UNVERIFIED'
    assert label['start_labeled_interval']==['2025-05-19T13:00:00+00:00','2025-05-19T13:01:00+00:00']
    assert label['end_labeled_interval']==['2025-05-19T12:59:00+00:00','2025-05-19T13:00:00+00:00']
    result=asyncio.run(replay_capture([row],{'initial_state':STATE},SNAPSHOT))
    assert result['results'][0]['status']=='MATCH'


def test_sequence_and_missing_clock_are_unassessable():
    row=capture_row();row['trace']['sequence']=999
    assert asyncio.run(replay_capture([row],{'initial_state':STATE},SNAPSHOT))['results'][0]['status']=='UNASSESSABLE'
    row=capture_row();del row['readiness']['clock_semantics']
    assert asyncio.run(replay_capture([row],{'initial_state':STATE},SNAPSHOT))['results'][0]['status']=='UNASSESSABLE'
    exact,floats=differences({'quantity':2.0},{'quantity':2.1})
    assert exact and not floats


@pytest.mark.parametrize('value',[None,'not-a-price',{},float('inf')])
def test_malformed_price_is_mismatch_not_decimal_exception(value):
    exact,floats=differences({'entry_price':value},{'entry_price':100.25})
    assert exact and not floats and exact[0]['invalid_price_value']


def test_risk_and_config_floats_exact_feature_floats_explicit():
    exact,floats=differences({'risk':{'daily_pnl':1.},'config':{'sl_multiplier':2.},'features':{'atr':1.}},
                            {'risk':{'daily_pnl':1.1},'config':{'sl_multiplier':2.1},'features':{'atr':1.1}})
    assert len(exact)==2 and len(floats)==1
    row=capture_row();row['trace']['decisions'].append({'kind':'ml_prediction','probability':.5})
    # Unit-level path classification above covers model features; an inserted
    # prediction is a discrete event-count mismatch, not merely a float diff.
    assert asyncio.run(replay_capture([row],{'initial_state':STATE},SNAPSHOT))['results'][0]['status']=='MISMATCH'


def test_nominal_reference_and_account_declarations_not_live_proof():
    row=capture_row();row['readiness']['identity_verification_scope']='UNVERIFIED_LIVE_DECLARATION'
    result=asyncio.run(replay_capture([row],{'initial_state':STATE},SNAPSHOT))
    assert result['results'][0]['status']=='UNASSESSABLE'
    assert 'live_identity_unverified' in result['results'][0]['reasons']


def test_bounds_precede_copy_and_custom_calls(tmp_path):
    from src.research.yank_deployed_validation.capture import bounded_payload
    class Hostile:
        def __str__(self):raise AssertionError('str called')
        def model_dump(self):raise AssertionError('model_dump called')
        def __iter__(self):raise AssertionError('iter called')
    for value in ([None]*10000,{str(i):None for i in range(10000)},Hostile()):
        with pytest.raises(ValueError):bounded_payload(value,20,200)
    deep={};cursor=deep
    for _ in range(100):cursor['x']={};cursor=cursor['x']
    with pytest.raises(ValueError):bounded_payload(deep,1000,10000)
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'hostile')
    assert not c.emit(Hostile());assert not c.close()['valid_coverage']


def test_known_live_decision_shape_is_bounded_without_model_dump():
    from dataclasses import make_dataclass
    from enum import Enum
    from src.research.yank_deployed_validation.capture import bounded_payload
    direction=Enum('Direction',{'BEARISH':'bearish'},module='src.research.strategy_core')
    decision=make_dataclass('EntryDecision',['direction','entry_price','sl_price','tp_price','contracts'],module='src.research.strategy_core')
    encoded=bounded_payload(decision(direction.BEARISH,100.,101.,90.,2),50,1000)
    assert json.loads(encoded)['direction']=='bearish'


def seeded_entry_adapter():
    """Synthetic H1/sweep fixture forces a nonzero branch using the pinned model."""
    from datetime import datetime,timedelta,timezone
    adapter=Adapter(SNAPSHOT,{**STATE,'is_backfill':False});t=adapter.trader;m=adapter.runtime.module
    for i in range(20):
        raw=dict(TimeStamp=(datetime(2025,5,19,13,40,tzinfo=timezone.utc)+timedelta(minutes=i)).isoformat(),Open=100,High=101,Low=99,Close=100,TotalVolume=100)
        if i==18:raw.update(Open=107,High=108,Low=106,Close=107)
        if i==19:raw.update(Open=107,High=108,Low=102,Close=103)
        t.dollar_bars.append(t._parse_bar(raw))
    t._last_processed_timestamp=t.dollar_bars[-1].timestamp
    t.h1_bearish_sweep_active=True;t._m15_choch_active=True;t._h1_atr=10.;t._h1_slope=-10.
    t._cached_sweep=m.SweepSignal(m.Direction.BEARISH,1,112.)
    return adapter


def test_installed_observer_entry_exit_and_reply_evidence(tmp_path,monkeypatch):
    from src.research.yank_deployed_validation.capture import install_poll_observer
    a=seeded_entry_adapter();control=seeded_entry_adapter();checkpoint=a.checkpoint()
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'trading')
    rollback=install_poll_observer(a.trader,c,identity=identities(SNAPSHOT),readiness={'equivalent_feed_and_state':True},state_reader=a.state)
    bars=[dict(TimeStamp='2025-05-19T14:00:00Z',Open=102,High=103,Low=101,Close=102,TotalVolume=100),
          dict(TimeStamp='2025-05-19T14:01:00Z',Open=104.5,High=105,Low=104,Close=104.5,TotalVolume=100),
          dict(TimeStamp='2025-05-19T14:02:00Z',Open=104,High=112,Low=103,Close=111,TotalVolume=100)]
    for index,bar in enumerate(bars):
        event=dict(bars=[bar],receipt_time=f'2025-05-19T14:0{index+1}:00Z',request_id=str(index),status_code=200)
        asyncio.run(a.poll(event));asyncio.run(control.poll(event))
        assert a.state()==control.state()
    rollback();assert c.close()['valid_coverage']
    rows=[json.loads(line) for line in (tmp_path/'trading/capture.jsonl').read_text().splitlines()]
    assert [x['kind'] for row in rows for x in row['trace']['intentions']]==['submit_bracket','place_exits','cancel']
    assert [x['operation'] for row in rows for x in row['trace']['input']['execution_replies']]==['submit_bracket_order','place_exit_orders','cancel_order']
    assert a.trader.completed_trades[-1].exit_type=='sl'
    initial={**STATE,'classification':'OBSERVED_STATE','checkpoint':checkpoint,'account_evidence':{'status':'OBSERVED','source_sha256':'0'*64,'receipt_time':'2025-05-19T14:00:00Z'}}
    result=asyncio.run(replay_capture(rows,{'initial_state':initial},SNAPSHOT))
    assert all(not row['exact_differences'] and not row['float_differences'] for row in result['results'])
    assert all(row['reasons']==['account_evidence_unverified'] for row in result['results'])
    # Reconcile the same fixture as explicitly synthetic private state; this is
    # deliberately distinct from externally claimed account evidence above.
    import src.research.yank_deployed_validation.compare as comparator
    class SeededAdapter(Adapter):
        def __init__(self,snapshot,state):
            super().__init__(snapshot,state)
            self.restore_checkpoint(checkpoint,initial['account_evidence'])
    monkeypatch.setattr(comparator,'Adapter',SeededAdapter)
    private=asyncio.run(replay_capture(rows,{'initial_state':STATE},SNAPSHOT))
    assert all(row['status']=='MATCH' for row in private['results'])
    changed=copy.deepcopy(rows)
    prediction=next(d for d in changed[0]['trace']['decisions'] if d.get('kind')=='ml_prediction')
    prediction['probability']-=.000001
    floated=asyncio.run(replay_capture(changed,{'initial_state':STATE},SNAPSHOT))
    assert floated['results'][0]['status']=='MATCH_DISCRETE_WITH_FLOAT_DIFFERENCES'
    missing=copy.deepcopy(rows);missing[0]['trace']['input']['execution_replies']=[]
    result=asyncio.run(replay_capture(missing,{'initial_state':STATE},SNAPSHOT))
    assert result['results'][0]['status']=='UNASSESSABLE'
    assert 'missing execution reply' in result['results'][0]['reasons'][0]


@pytest.mark.parametrize('failure',[503,'timeout'])
def test_installed_failed_poll_keeps_envelope_and_replays(tmp_path,failure):
    from src.research.yank_deployed_validation.capture import install_poll_observer
    from datetime import datetime,timezone
    a=Adapter(SNAPSHOT,STATE);a.runtime.clock=datetime(2025,5,19,22,tzinfo=timezone.utc)
    class Auth:
        async def authenticate(self):return 'offline'
    class Response:
        status_code=503
        content=b'bounded offline failure response'
        def json(self):raise AssertionError('non-200 JSON must not be read')
    class Client:
        async def get(self,*args,**kwargs):
            if failure=='timeout':raise asyncio.TimeoutError('offline timeout')
            return Response()
    a.trader.auth=Auth();a.trader.client=Client()
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'failure')
    rollback=install_poll_observer(a.trader,c,identity=identities(SNAPSHOT),readiness={'equivalent_feed_and_state':True},state_reader=a.state)
    asyncio.run(a.trader._poll_and_process());rollback();assert c.close()['valid_coverage']
    row=json.loads((tmp_path/'failure/capture.jsonl').read_text());event=row['trace']['input']
    assert event['bars'] is None
    import hashlib
    assert event['raw_http_bytes_sha256']==(None if failure=='timeout' else hashlib.sha256(Response.content).hexdigest())
    assert event['bars_hash_scope']=='canonical_parsed_Bars_not_HTTP_bytes'
    assert event['status_code']==(None if failure=='timeout' else 503)
    assert event['request_failure']==('timeout' if failure=='timeout' else None)
    result=asyncio.run(replay_capture([row],{'initial_state':STATE},SNAPSHOT))
    assert result['results'][0]['status']=='MATCH'
    # At22UTC the scheduler is closed; matching proves no fabricated recheck.
    assert row['trace']['scheduled'] is True


def test_close_while_serializing_prohibits_late_enqueue(tmp_path,monkeypatch):
    import threading
    import src.research.yank_deployed_validation.capture as module
    entered=threading.Event();release=threading.Event();original=module.bounded_payload
    def delayed(*args,**kwargs):entered.set();release.wait(2);return original(*args,**kwargs)
    monkeypatch.setattr(module,'bounded_payload',delayed)
    c=DecisionCapture(enabled=True,output_dir=tmp_path/'race');result=[]
    producer=threading.Thread(target=lambda:result.append(c.emit({'kind':'poll'})));producer.start();assert entered.wait(1)
    summary=c.close();release.set();producer.join(2)
    assert result==[False] and c.accepted==0 and not c.queue
    assert not summary['valid_coverage'] and 'producer_active_at_close' in summary['invalid_reasons']
    assert not json.loads((tmp_path/'race/coverage.json').read_text())['valid_coverage']
