"""Boundary tests of the pinned poller, including deliberately preserved defects."""
import asyncio
import copy
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('matrix_cli',ROOT/'src/cli/check_yank_deployed_replay.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
replay=cli.load_tool('replay')
Adapter=replay.Adapter
SNAPSHOT=ROOT/'docs/yank-validation/snapshot/v1'
STATE=dict(classification='SYNTHETIC_UNKNOWN_ACCOUNT',daily_pnl=0.,daily_halted=False,last_trading_date=None,on_combine=True,is_backfill=True)


def bar(label,**extra):
    return dict(TimeStamp=label,Open='100',High='101',Low='99',Close='100',TotalVolume=1,**extra)


def poll(adapter,bars,receipt='2025-05-19T13:11:00Z'):
    return asyncio.run(adapter.poll(dict(receipt_time=receipt,request_id='boundary-test',status_code=200,bars=bars)))


@pytest.mark.parametrize('labels,receipt,offsets',[
    (['2025-03-09T06:30:00Z','2025-03-09T07:30:00Z'],'2025-03-10T12:00:00Z',[-5,-4]),
    (['2025-11-02T05:30:00Z','2025-11-02T06:30:00Z'],'2025-11-03T12:00:00Z',[-4,-5]),
])
def test_dst_missing_minutes_and_repeated_local_hour(labels,receipt,offsets):
    a=Adapter(SNAPSHOT,STATE)
    trace=poll(a,[bar(x) for x in labels],receipt)
    assert trace['after']['bar_count']==2
    assert [b.timestamp.astimezone(a.runtime.module.ET_TZ).utcoffset().total_seconds()/3600 for b in a.trader.dollar_bars]==offsets
    assert len(a.trader._daily_ranges)==0
    assert trace['after']['consumed_history']['lr_ready'] is False
    assert trace['after']['consumed_history']['adr_full_ready'] is False


def test_contract_labels_preserved_but_baseline_has_no_autoroll():
    a=Adapter(SNAPSHOT,STATE)
    rows=[bar('2025-05-19T13:00:00Z',Contract='MNQH25'),bar('2025-05-19T13:01:00Z',Contract='MNQM25')]
    trace=poll(a,rows)
    assert [r['Contract'] for r in trace['input']['bars']]==['MNQH25','MNQM25']
    assert a.trader._symbol=='MNQU26'
    assert len(a.trader.dollar_bars)==2
    assert trace['decision']=='HOLD_VALIDATION'  # no implied mapping/roll validation


def test_missing_minutes_advance_pending_only_on_accepted_bars():
    a=Adapter(SNAPSHOT,{**STATE,'is_backfill':False});m=a.runtime.module
    a.trader.active_trade=m.ActiveTrade(0,datetime(2025,5,19,13,tzinfo=timezone.utc),'SHORT',110,100,120)
    a.trader._active_entry_decision=m.EntryDecision(m.Direction.BEARISH,110,120,100,2)
    poll(a,[bar('2025-05-19T13:00:00Z'),bar('2025-05-19T13:10:00Z')])
    assert a.trader.active_trade.bars_held==2
    assert len(a.trader.dollar_bars)==2


def test_empty_poll_does_not_silently_change_preserved_stale_behavior():
    a=Adapter(SNAPSHOT,STATE)
    poll(a,[bar('2025-05-19T13:00:00Z')],'2025-05-19T13:01:00Z')
    trace=poll(a,[],'2025-05-19T15:00:00Z')
    assert trace['after']['_data_stale'] is False
    assert trace['after']['bar_count']==1


def test_recovery_unknown_broker_keeps_explicit_unknown_and_backfill_guard():
    a=Adapter(SNAPSHOT,STATE);m=a.runtime.module
    # The only dependency of the original recovery branch is a TradeState type;
    # inject the equivalent local snapshotted type, never import another trader.
    a.runtime.stubs['src.research.tier2_streaming_working']=SimpleNamespace(TradeState=m.TradeState)
    a.runtime.clock=datetime(2025,5,19,13,tzinfo=timezone.utc)
    state=dict(direction='SHORT',entry_price=100.,tp_price=90.,sl_price=110.,entry_time='2025-05-19T12:00:00+00:00',
               sim_entry_order_id='E1',sim_tp_order_id='TP1',sim_sl_order_id='SL1',daily_pnl=-200.,daily_halted=False,last_trading_date='2025-05-19')
    a.runtime.Persistence.save_state(state)
    asyncio.run(a.trader._recover_from_state())
    assert a.trader.active_trade is not None
    assert a.trader.active_trade.pending_entry is False  # preserved protective assumption, not fill evidence
    assert a.trader._risk_manager.daily_pnl==-200.
    assert asyncio.run(a.trader._ts_client.is_order_open('E1')) is None
    asyncio.run(a.trader._advance_active_trade(a.trader._parse_bar(bar('2025-05-19T13:00:00Z'))))
    assert a.trader.active_trade.bars_held==0
    assert a.trader._ts_client.intentions==[]


def admission(tmp_path):
    m=json.loads((ROOT/'docs/yank-validation/native-admission.json').read_text())
    m['format']='polls-v1';p=tmp_path/'polls.jsonl';p.write_text('')
    m['files']=[dict(file=p.name,sha256=replay.digest(p))]
    manifest=tmp_path/'admission.json';manifest.write_text(json.dumps(m))
    return m,manifest


@pytest.mark.parametrize('fault',['missing_coverage','empty_contract','future','naive','hash','path_escape'])
def test_admission_refuses_incomplete_or_unpinned_inputs(tmp_path,fault):
    m,path=admission(tmp_path)
    if fault=='missing_coverage':del m['coverage']
    elif fault=='empty_contract':m['contracts']={}
    elif fault=='future':m['interval']['end_exclusive']='2026-01-01T00:01:00Z'
    elif fault=='naive':m['interval']['start']='2025-05-19T00:00:00'
    elif fault=='hash':m['files'][0]['sha256']='0'*64
    else:m['files'][0]['file']='../outside.jsonl'
    path.write_text(json.dumps(m))
    with pytest.raises((ValueError,FileNotFoundError)):
        replay.admit(path,tmp_path)


def test_complete_native_event_availability_is_required(tmp_path):
    m,path=admission(tmp_path);m['format']='native-minute-v1'
    (tmp_path/'bars.jsonl').write_text(json.dumps(dict(incomplete_event=True,availability_ns=100,end_ns=200))+'\n')
    with pytest.raises(ValueError,match='incomplete'):
        list(replay.events(m,tmp_path))


def test_both_existing_shadow_features_run_without_attribute_collision():
    a=Adapter(SNAPSHOT,{**STATE,'is_backfill':False});t=a.trader;m=a.runtime.module
    class BullishLog:
        def __init__(self):self.rows=[]
        def append_trade(self,trade):self.rows.append(trade)
    class ParityLog:
        def __init__(self):self.polls=[]
        def log_poll(self,*args):self.polls.append(args)
    bullish= BullishLog();parity=ParityLog()
    t._shadow_trade_logger=bullish;t._shadow_logger=parity;t._data_shadow=True
    t._shadow_trade=dict(entry_time=datetime(2025,5,19,13,tzinfo=timezone.utc),entry_price=100.,tp_price=108.,sl_price=96.,
                        gap_size=4.,h1_sweep_bars_ago=0,pending=False,bars_held=0,
                        entry_decision=m.EntryDecision(m.Direction.BULLISH,100.,96.,108.,2))
    candle=t._parse_bar(dict(TimeStamp='2025-05-19T13:02:00Z',Open=100,High=109,Low=100,Close=108,TotalVolume=10))
    t.dollar_bars=[candle]
    t._advance_shadow_trade(candle)
    asyncio.run(t._run_shadow_parity(datetime(2025,5,19,13,3,tzinfo=timezone.utc)))
    assert len(bullish.rows)==1 and len(parity.polls)==1
    assert t._shadow_trade is None
    assert t._ts_client.intentions==[]
    assert t._shadow_trade_logger is bullish and t._shadow_logger is parity
