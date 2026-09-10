import asyncio
import copy
from datetime import datetime,timezone,timedelta
from pathlib import Path
import shutil
import pytest
from src.research.yank_deployed_validation.adapter import Adapter,PrivateSnapshot,canonical

SNAPSHOT=Path(__file__).resolve().parents[2]/'docs/yank-validation/snapshot/v1'
STATE=dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT',daily_pnl=0.,daily_halted=False,last_trading_date=None,on_combine=True,is_backfill=True)

def bar(minute,**kw):
    return dict(TimeStamp=f'2025-05-19T13:{minute:02}:00Z',Open='100',High='101',Low='99',Close='100',TotalVolume=1,**kw)

def event(rows,clock='2025-05-19T13:05:00Z'):
    return dict(receipt_time=clock,request_id='test',bars=rows,status_code=200)

def test_poll_matches_independently_invoked_snapshot():
    adapter=Adapter(SNAPSHOT,STATE);reference=Adapter(SNAPSHOT,STATE)
    raw=event([bar(0),bar(2,IsRealtime=True),bar(1),bar(2,IsEndOfHistory=False),bar(6)])
    trace=asyncio.run(adapter.poll(raw))
    r=reference.runtime;t=reference.trader;r.clock=datetime.fromisoformat(raw['receipt_time'].replace('Z','+00:00'))
    class Auth:
        async def authenticate(self):return 'offline'
    class Response:
        status_code=200
        def json(self):return {'Bars':raw['bars']}
    class Client:
        async def get(self,*a,**kw):return Response()
    t.auth=Auth();t.client=Client()
    # Independently invoke original snapshot descriptor, bypass Adapter.poll.
    asyncio.run(r.module.Tier2StreamingTrader._poll_and_process(t))
    assert trace['after']==reference.state()
    assert len(t.dollar_bars)==2
    assert [b.timestamp.minute for b in t.dollar_bars]==[0,2]
    assert t._is_backfill is False


def test_revisions_future_and_delayed_complete():
    a=Adapter(SNAPSHOT,STATE)
    asyncio.run(a.poll(event([bar(0,IsEndOfHistory=False)])))
    revised=bar(0);revised['Close']='101'
    trace=asyncio.run(a.poll(event([revised,bar(1)],'2025-05-19T14:10:00Z')))
    assert a.trader.dollar_bars[0].close==100
    assert trace['after']['_data_stale'] is True
    assert trace['after']['bar_count']==2


def test_causal_receipts_and_scheduler():
    a=Adapter(SNAPSHOT,STATE)
    trace=asyncio.run(a.poll(event([],clock='2025-05-19T22:05:00Z')))
    assert not trace['scheduled']
    with pytest.raises(ValueError):asyncio.run(a.poll(event([])))


def test_isolation_and_pin_failure(tmp_path,monkeypatch):
    import socket
    monkeypatch.setattr(socket,'create_connection',lambda *a,**kw:pytest.fail('socket attempted'))
    a=Adapter(SNAPSHOT,STATE)
    with pytest.raises(RuntimeError):asyncio.run(a.trader.initialize())
    with pytest.raises(RuntimeError):a.runtime.importer('src.data.auth_v3',fromlist=['TradeStationAuthV3']).TradeStationAuthV3()
    with pytest.raises(RuntimeError):a.runtime.SafePath(tmp_path/'escape').write_text('no')
    with pytest.raises(RuntimeError):a.runtime.importer('subprocess')
    assert a.trader._shadow_logger is not a.trader._shadow_trade_logger
    copied=tmp_path/'snapshot';shutil.copytree(SNAPSHOT,copied)
    (copied/'strategy_config.yaml').write_text('changed')
    with pytest.raises(ValueError):PrivateSnapshot(copied)


def test_pending_counter_full_opportunities_and_unknown_broker():
    a=Adapter(SNAPSHOT,STATE);m=a.runtime.module;t=a.trader
    t._is_backfill=False
    t.active_trade=m.ActiveTrade(0,datetime(2025,5,19,13,tzinfo=timezone.utc),'SHORT',110,120,90)
    t._active_entry_decision=m.EntryDecision(direction=m.Direction.BEARISH,entry_price=110,sl_price=120,tp_price=90,contracts=2)
    b=t._parse_bar(bar(1))
    for _ in range(239):asyncio.run(t._advance_active_trade(b))
    assert t.active_trade is not None and t.active_trade.bars_held==239
    asyncio.run(t._advance_active_trade(b))
    assert t.active_trade is None
    assert asyncio.run(t._ts_client.is_order_open('unknown')) is None


def test_active_same_bar_exit_and_risk_carry():
    a=Adapter(SNAPSHOT,{**STATE,'daily_pnl':-301.,'daily_halted':True});m=a.runtime.module;t=a.trader
    assert t._risk_manager.check_and_update(datetime(2025,5,19,tzinfo=timezone.utc),-300,is_backfill=True) is False
    assert t._risk_manager.daily_pnl==-301.
    t._is_backfill=False
    t.active_trade=m.ActiveTrade(0,datetime(2025,5,19,13,tzinfo=timezone.utc),'SHORT',100,99,101)
    t._active_entry_decision=m.EntryDecision(direction=m.Direction.BEARISH,entry_price=100,sl_price=101,tp_price=99,contracts=2)
    asyncio.run(t._advance_active_trade(t._parse_bar(bar(1))))
    assert t.active_trade is None
    assert t.completed_trades[-1].exit_type=='sl'


def test_nonzero_entry_fill_exit_fidelity():
    a=Adapter(SNAPSHOT,{**STATE,'is_backfill':False});b=Adapter(SNAPSHOT,{**STATE,'is_backfill':False})
    for adapter in (a,b):
        m=adapter.runtime.module;t=adapter.trader
        t._cached_sweep=m.SweepSignal(m.Direction.BEARISH,1,102.)
        fvg=m.FVGSignal(m.Direction.BEARISH,1.,100.13,101.,100.)
        candle=t._parse_bar(bar(1))
        # Instance dispatch versus direct original class descriptor.
        call=t._enter_trade if adapter is a else lambda *args,**kw:m.Tier2StreamingTrader._enter_trade(t,*args,**kw)
        asyncio.run(call(fvg,candle,1,False,ml_proba=.7))
    assert a.state()==b.state()
    assert a.trader._ts_client.intentions==b.trader._ts_client.intentions
    assert a.trader._ts_client.intentions[0]['decision']['contracts']==2
    assert a.trader.active_trade.entry_price==100.25
    for adapter in (a,b):
        candle=adapter.trader._parse_bar(bar(2))
        asyncio.run(adapter.runtime.module.Tier2StreamingTrader._advance_active_trade(adapter.trader,candle))
    assert a.state()==b.state()
    assert [i['kind'] for i in a.trader._ts_client.intentions]==['submit_bracket','place_exits']
    assert a.trader.active_trade.pending_entry is False
    for adapter in (a,b):
        raw=bar(3);raw['High']='105'
        asyncio.run(adapter.trader._advance_active_trade(adapter.trader._parse_bar(raw)))
    assert a.state()==b.state()
    assert a.trader.completed_trades[-1].exit_type=='sl'
    assert a.trader._ts_client.intentions==b.trader._ts_client.intentions


def test_complete_checkpoint_restores_equal_state_and_rejects_missing():
    a=Adapter(SNAPSHOT,{**STATE,'is_backfill':False})
    asyncio.run(a.poll(event([bar(0),bar(1)])))
    checkpoint=a.checkpoint()
    state={**STATE,'classification':'OBSERVED_STATE','checkpoint':checkpoint,'account_evidence':{'status':'OBSERVED','source_sha256':'0'*64,'receipt_time':'2025-05-19T13:05:00Z'}}
    b=Adapter(SNAPSHOT,state)
    assert a.state()==b.state()
    assert asyncio.run(a.poll(event([bar(2)])))['after']==asyncio.run(b.poll(event([bar(2)])))['after']
    del checkpoint['state']['_m15_last_bar_ts']
    with pytest.raises(ValueError):Adapter(SNAPSHOT,state)


def observed_state(adapter):
    return {**STATE,'classification':'OBSERVED_STATE','checkpoint':adapter.checkpoint(),
            'account_evidence':{'status':'OBSERVED','source_sha256':'0'*64,'receipt_time':'2025-05-19T13:05:00Z'}}


def test_checkpoint_shadow_entry_decision_continues_exit():
    a=Adapter(SNAPSHOT,{**STATE,'is_backfill':False});m=a.runtime.module
    a.trader._shadow_trade=dict(entry_time=datetime(2025,5,19,13,tzinfo=timezone.utc),entry_price=100.,tp_price=101.,sl_price=99.,gap_size=1.,h1_sweep_bars_ago=1,pending=False,bars_held=0,
        entry_decision=m.EntryDecision(m.Direction.BULLISH,100.,99.,101.,2))
    b=Adapter(SNAPSHOT,observed_state(a))
    assert isinstance(b.trader._shadow_trade['entry_decision'],b.runtime.module.EntryDecision)
    for adapter in (a,b):adapter.trader._advance_shadow_trade(adapter.trader._parse_bar(bar(1)))
    assert a.state()==b.state() and a.trader._shadow_trade is None
    assert a.trader._shadow_trade_logger.rows==b.trader._shadow_trade_logger.rows


def test_checkpoint_execution_routing_is_restored_and_compared():
    a=Adapter(SNAPSHOT,{**STATE,'on_combine':False})
    restored=Adapter(SNAPSHOT,observed_state(a))
    assert restored.trader._on_combine is False
    assert restored.state()==a.state()
    restored.trader._on_combine=True
    assert restored.state()!=a.state()


def test_observed_missing_or_exhausted_replies_cannot_succeed():
    a=Adapter(SNAPSHOT,{**STATE,'is_backfill':False});m=a.runtime.module
    a.trader.active_trade=m.ActiveTrade(0,datetime(2025,5,19,13,tzinfo=timezone.utc),'SHORT',100,99,101)
    a.trader._active_entry_decision=m.EntryDecision(m.Direction.BEARISH,100.,101.,99.,2)
    b=Adapter(SNAPSHOT,observed_state(a))
    assert b.account_verification=='UNVERIFIED_EXTERNAL_DECLARATION'
    with pytest.raises(ValueError,match='missing execution reply'):
        asyncio.run(b.poll(event([bar(1)])))
    b=Adapter(SNAPSHOT,observed_state(a))
    supplied=event([bar(1)]);supplied['execution_replies']=[{'operation':'place_exit_orders','value':['real-tp','real-sl']}]
    # Same bar exits at SL and needs a subsequent cancel response as well.
    with pytest.raises(ValueError,match='missing execution reply: cancel_order'):
        asyncio.run(b.poll(supplied))


def test_already_admitted_poll_does_not_recheck_scheduler():
    a=Adapter(SNAPSHOT,STATE)
    trace=asyncio.run(a.poll(event([bar(0)],'2025-05-19T22:00:00Z'),already_admitted=True))
    assert trace['scheduled'] and trace['after']['bar_count']==1
