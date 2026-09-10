"""Private exact-method offline runtime. No live trader imports or initialization."""
import ast
import asyncio
import builtins
import dataclasses
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import sys
import types

SNAPSHOT_MANIFEST_SHA256 = '4eb9dbb8be66f2cb4b9824caf318e2e5163d2f14537460429db7db95cf915f92'


@lru_cache(maxsize=8192)
def _utc_timestamp_parts(timestamp):
    """Cache only immutable UTC datetime derivations, never bar/runtime values."""
    return timestamp.isoformat(), timestamp.replace(minute=0, second=0, microsecond=0)


def _timestamp_parts(timestamp):
    # Datetime equality merges equal instants with different representations and
    # can ignore fold. Restrict cache keys to exact UTC datetimes; custom types
    # and all other timezones retain their original uncached behavior.
    if type(timestamp) is datetime and timestamp.tzinfo is timezone.utc:
        return _utc_timestamp_parts(timestamp)
    return timestamp.isoformat(), timestamp.replace(minute=0, second=0, microsecond=0)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'


def normalize(value):
    if dataclasses.is_dataclass(value):return normalize(dataclasses.asdict(value))
    if hasattr(value,'model_dump'):return normalize(value.model_dump())
    if isinstance(value,dict):return {str(k):normalize(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [normalize(v) for v in value]
    if isinstance(value,(set,frozenset)):return sorted(normalize(v) for v in value)
    if isinstance(value,Enum):return normalize(value.value)
    if hasattr(value,'isoformat'):return value.isoformat()
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    return value


def denied(*args,**kwargs):raise RuntimeError('offline capability denied')
async def denied_async(*args,**kwargs):denied()


class MemoryLog:
    def __init__(self,*a,**kw):self.rows=[]
    def append_trade(self,row):self.rows.append(normalize(row))
    def log_poll(self,*a,**kw):self.rows.append(normalize(a))


class MemoryLogger:
    def __init__(self):self.errors=[]
    def isEnabledFor(self,*a):return False
    def info(self,*a,**kw):pass
    warning=info;debug=info;critical=info
    def error(self,msg,*a,**kw):self.errors.append(msg % a if a else str(msg))


class Broker:
    """Records intentions with synthetic IDs; these are never acknowledgements."""
    def __init__(self):self.intentions=[];self.replies=[];self.require_replies=False;self.evidence_errors=[]
    def reply(self,kind,default):
        if not self.replies:
            if self.require_replies:
                self.evidence_errors.append('missing execution reply: '+kind)
                raise ValueError(self.evidence_errors[-1])
            return default
        item=self.replies.pop(0)
        if item['operation']!=kind:raise ValueError('execution reply order mismatch')
        return item['value']
    async def submit_bracket_order(self,decision,account):
        self.intentions.append(dict(kind='submit_bracket',decision=normalize(decision),venue='ProjectX',evidence='INTENTION_ONLY'))
        return self.reply('submit_bracket_order',(f'offline-intention-{len(self.intentions)}',None,None))
    async def place_exit_orders(self,decision,account):
        self.intentions.append(dict(kind='place_exits',decision=normalize(decision),venue='ProjectX',evidence='INTENTION_ONLY'))
        return self.reply('place_exit_orders',('offline-tp','offline-sl'))
    async def cancel_order(self,oid):
        self.intentions.append(dict(kind='cancel',order_id=oid,venue='ProjectX',evidence='INTENTION_ONLY'));return self.reply('cancel_order',None)
    async def close_position_at_market(self,direction,account,contracts=None):
        self.intentions.append(dict(kind='close',direction=direction,quantity=contracts,venue='ProjectX',evidence='INTENTION_ONLY'));return self.reply('close_position_at_market',None)
    async def is_order_open(self,*a):return None
    reconcile_state=denied_async;cancel_all_pending_orders=denied_async


class PrivateSnapshot:
    def __init__(self,root):
        self.root=Path(root).resolve();self.modules={};self.logger=MemoryLogger()
        self.verify()
        self.environment={'YANK_CONTRACTS':'2','YANK_MAX_GAP_ATR_RATIO':'0.426','YANK_DATA_SHADOW':'1','YANK_MIRROR_TS_SIM':'1','SIM_INVVOL':'1','SYMBOL':'MNQU26'}
        self.clock=datetime(2025,5,19,tzinfo=timezone.utc)
        owner=self
        class Clock(datetime):
            @classmethod
            def now(cls,tz=None):return owner.clock.astimezone(tz) if tz else owner.clock.replace(tzinfo=None)
        self.Clock=Clock
        class SafePath(type(Path())):
            def open(self,mode='r',*a,**kw):
                if any(c in mode for c in 'wax+') or not self.resolve().is_relative_to(owner.root):denied()
                return super().open(mode,*a,**kw)
            def mkdir(self,*a,**kw):denied()
            def unlink(self,*a,**kw):denied()
            def rename(self,*a,**kw):denied()
            def replace(self,*a,**kw):denied()
        self.SafePath=SafePath
        class Persistence:
            value=None
            @classmethod
            def save_state(cls,state):cls.value=normalize(state)
            @classmethod
            def load_state(cls):return cls.value
            @classmethod
            def clear_state(cls):cls.value=None
        self.Persistence=Persistence
        self.stubs={
            'os':types.SimpleNamespace(environ=self.environment),
            'sys':types.SimpleNamespace(path=types.SimpleNamespace(insert=lambda *a:None),stdout=types.SimpleNamespace(isatty=lambda:False)),
            'pathlib':types.SimpleNamespace(Path=SafePath),
            'httpx':types.SimpleNamespace(TimeoutException=TimeoutError,AsyncClient=denied),
            'src.data.auth_v3':types.SimpleNamespace(TradeStationAuthV3=denied),
            'src.research.projectx_bars':types.SimpleNamespace(fetch_px_ts_shaped=denied_async,ProjectXBarFetchError=RuntimeError,_to_contract_id=denied),
            'src.monitoring.trade_db':types.SimpleNamespace(TradeDatabase=lambda:types.SimpleNamespace(log_trade=lambda **kw:None)),
        }
        self.module=self.load('src.research.yank_streaming_working')
        m=self.module;m.datetime=Clock;m.logger=self.logger;m.StatePersistence=Persistence;m.TradeLogger=MemoryLog
        self.stubs['src.research.tier2_streaming_working']=types.SimpleNamespace(TradeState=m.TradeState)
        m.MetaLabelingFilter._log_decision=lambda *a,**kw:None
        cls=m.Tier2StreamingTrader
        for name in ('initialize','start_streaming','stop'):setattr(cls,name,denied_async)
        for name in ('_log_ml_canary','_write_equity_curve','_log_trade_metrics'):
            setattr(cls,name,lambda *a,**kw:None)

    def verify(self):
        if digest(self.root/'manifest.json')!=SNAPSHOT_MANIFEST_SHA256:raise ValueError('snapshot manifest pin mismatch')
        manifest=json.loads((self.root/'manifest.json').read_text())
        for name,pin in manifest['files'].items():
            path=(self.root/name).resolve()
            if not path.is_relative_to(self.root) or digest(path)!=pin['sha256']:raise ValueError('snapshot pin mismatch: '+name)
        for name,version in manifest['versions'].items():
            if importlib.metadata.version(name)!=version:raise ValueError('runtime version mismatch: '+name)

    def importer(self,name,globals=None,locals=None,fromlist=(),level=0):
        if level:raise RuntimeError('relative private import forbidden')
        if name in self.stubs:module=self.stubs[name]
        elif name.startswith('src.'):
            if name not in {'src.data.models','src.research.strategy_core','src.research.config_loader','src.research.lr_channel','src.ml.regime_detection.lr_channel_detector','src.research.shadow_parity'}:denied()
            module=self.load(name)
        else:
            if name.split('.')[0] not in {'__future__','asyncio','csv','itertools','json','logging','time','dataclasses','datetime','typing','joblib','numpy','pandas','pytz','zoneinfo','enum','math','yaml','hashlib','pydantic'}:denied()
            return builtins.__import__(name,globals,locals,fromlist,level)
        if fromlist:return module
        if name.startswith('src.'):
            parts=name.split('.');node=module
            for part in reversed(parts[1:]):node=types.SimpleNamespace(**{part:node})
            return node
        return module

    def load(self,name):
        if name in self.modules:return self.modules[name]
        path=self.root/(name.replace('.','/')+'.py')
        private_name='_yank_private_'+str(id(self))+'_'+name.replace('.','_')
        module=types.ModuleType(private_name);module.__file__=str(path)
        self.modules[name]=module;sys.modules[private_name]=module
        env=dict(vars(builtins));env['__import__']=self.importer;env['open']=denied
        module.__dict__['__builtins__']=env
        tree=ast.parse(path.read_text(),filename=str(path))
        if name=='src.research.yank_streaming_working':
            # Remove module infrastructure only; function/class AST nodes are unchanged.
            keep=[]
            for node in tree.body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom)):keep.append(node)
                elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    names={t.id for t in targets if isinstance(t,ast.Name)}
                    if not names.intersection({'log_dir','_handlers','_log_level','logger'}):keep.append(node)
            tree.body=keep
            module.logger=self.logger
        exec(compile(tree,str(path),'exec'),module.__dict__)
        return module

    def construct(self,state):
        required={'classification','daily_pnl','daily_halted','last_trading_date','on_combine','is_backfill'}
        if not required<=state.keys():raise ValueError('explicit execution/risk state required')
        if state['classification'] not in ('SYNTHETIC_UNKNOWN_ACCOUNT','OBSERVED_STATE'):raise ValueError('unknown state classification')
        if state['classification']=='OBSERVED_STATE' and not {'checkpoint','account_evidence'}<=state.keys():raise ValueError('complete observed checkpoint and account evidence required')
        trader=self.module.Tier2StreamingTrader('MNQU26')
        if trader.ml_filter.model is None:raise ValueError('pinned model failed to load')
        trader._risk_manager._daily_pnl=state['daily_pnl'];trader._risk_manager._daily_halted=state['daily_halted']
        trader._risk_manager._last_trading_date=datetime.fromisoformat(state['last_trading_date']).date() if state['last_trading_date'] else None
        trader._is_backfill=state['is_backfill'];trader._on_combine=state['on_combine'];trader._exec_account='offline-unknown'
        trader._ts_client=Broker();trader._shadow_logger=MemoryLog()
        trader._data_shadow=False # No second feed supplied; state/readiness says unavailable.
        self.decisions=[]
        trader._log_filter_decision=lambda *a,**kw:self.decisions.append(normalize({'args':a,'kwargs':kw}))
        trader._private_snapshot_runtime=self
        trader._ts_client.require_replies=state['classification']=='OBSERVED_STATE'
        return trader


class Adapter:
    def __init__(self,snapshot,state):
        self.runtime=PrivateSnapshot(snapshot);self.trader=self.runtime.construct(state)
        self.sequence=0;self.previous_receipt=None;self.state_classification=state['classification']
        self.account_verification='UNVERIFIED_EXTERNAL_DECLARATION' if state['classification']=='OBSERVED_STATE' else 'SYNTHETIC_INPUT_ONLY'
        self._buffer_cache_key=None;self._buffer_hash=None;self._warmup={}
        if state['classification']=='OBSERVED_STATE':self.restore_checkpoint(state['checkpoint'],state['account_evidence'])
        original_detect=self.trader._detect_and_enter
        async def observe_detect(bar,is_backfill):
            before=self.state();offset=len(self.trader._ts_client.intentions)
            result=await original_detect(bar,is_backfill)
            self.runtime.decisions.append(dict(kind='decision_transition',bar_time=bar.timestamp.isoformat(),before=before,after=self.state(),intentions=self.trader._ts_client.intentions[offset:]))
            return result
        self.trader._detect_and_enter=observe_detect
        original_predict=self.trader.ml_filter.predict_proba
        def observe_predict(features):
            result=original_predict(features)
            self.runtime.decisions.append(dict(kind='ml_prediction',features=normalize(features),probability=result,threshold=self.trader.ml_filter.threshold))
            return result
        self.trader.ml_filter.predict_proba=observe_predict

    def checkpoint(self):
        """Complete redacted state, including accepted bars; no auth/account IDs."""
        state=self.state()
        return dict(schema_version=1,snapshot_sha256=SNAPSHOT_MANIFEST_SHA256,state=state,bars=normalize(self.trader.dollar_bars))

    def restore_checkpoint(self,checkpoint,evidence):
        if not {'status','source_sha256','receipt_time'}<=evidence.keys() or evidence['status']!='OBSERVED' or len(evidence['source_sha256'])!=64:raise ValueError('observed account evidence required')
        if checkpoint.get('schema_version')!=1 or checkpoint.get('snapshot_sha256')!=SNAPSHOT_MANIFEST_SHA256:raise ValueError('checkpoint identity mismatch')
        expected=self.state();values=checkpoint.get('state',{})
        if set(values)!=set(expected) or 'bars' not in checkpoint:raise ValueError('complete checkpoint fields required')
        t=self.trader;m=self.runtime.module
        def dt(value):return datetime.fromisoformat(value) if value is not None else None
        def number(value):return float(value) if value in ('nan','inf','-inf') else value
        dates={'_last_processed_timestamp','_bullish_sweep_expires','_bearish_sweep_expires','_last_bullish_sweep_h1_ts','_last_bearish_sweep_h1_ts','_m15_last_bar_ts','_shadow_m15_last_bar_ts','session_start_time'}
        derived={'risk','bar_count','accepted_buffer_sha256','consumed_history'}
        for name,value in values.items():
            if name in derived:continue
            if name in dates:value=dt(value)
            elif name=='_current_day':value=datetime.fromisoformat(value).date() if value else None
            elif name=='active_trade' and value is not None:
                value=dict(value);value['entry_time']=dt(value['entry_time']);value={k:number(v) for k,v in value.items()};value=m.ActiveTrade(**value)
            elif name=='completed_trades':value=[m.CompletedTrade(**{**v,'entry_time':dt(v['entry_time']),'exit_time':dt(v['exit_time'])}) for v in value]
            elif name=='_cached_sweep' and value is not None:value=m.SweepSignal(**{**value,'direction':m.Direction(value['direction'])})
            elif name=='_active_entry_decision' and value is not None:value=m.EntryDecision(**{**value,'direction':m.Direction(value['direction'])})
            elif name=='_shadow_trade' and value is not None:
                value=dict(value)
                if value.get('entry_decision') is not None:
                    decision=value['entry_decision'];value['entry_decision']=m.EntryDecision(**{**decision,'direction':m.Direction(decision['direction'])})
                for k in ('entry_time','timestamp'):
                    if k in value:value[k]=dt(value[k])
            elif isinstance(value,str):value=number(value)
            setattr(t,name,value)
        t.dollar_bars=[m.DollarBar(**{**v,'timestamp':dt(v['timestamp'])}) for v in checkpoint['bars']]
        if len(t.dollar_bars)>7500:raise ValueError('checkpoint buffer overflow')
        if any(a.timestamp>=b.timestamp for a,b in zip(t.dollar_bars,t.dollar_bars[1:])):raise ValueError('checkpoint bars not ordered')
        risk=values['risk']
        if set(risk)!={'daily_pnl','daily_halted','last_trading_date'}:raise ValueError('risk checkpoint incomplete')
        t._risk_manager._daily_pnl=risk['daily_pnl'];t._risk_manager._daily_halted=risk['daily_halted'];t._risk_manager._last_trading_date=datetime.fromisoformat(risk['last_trading_date']).date() if risk['last_trading_date'] else None
        self._buffer_cache_key=None
        if self.state()!=values:raise ValueError('checkpoint derived state mismatch')

    def state(self):
        t=self.trader
        names=['_on_combine','_last_processed_timestamp','active_trade','completed_trades','_is_backfill','h1_bullish_sweep_active','h1_bearish_sweep_active','_m15_choch_active','_m15_last_bar_ts','_shadow_bullish_m15_choch_active','_shadow_m15_last_bar_ts','_shadow_trade','_cached_sweep','_active_entry_decision','_h1_atr','_h1_slope','_vol_regime_high','_last_vol_regime_pct','_current_day','_session_open_price','_session_high','_session_low','_daily_ranges','_data_stale','_last_entry_bar','_bullish_sweep_bar','_bearish_sweep_bar','_bullish_sweep_expires','_bearish_sweep_expires','_last_bullish_sweep_h1_ts','_last_bearish_sweep_h1_ts','_h1_atr_history','session_start_time']
        key=(len(t.dollar_bars),t._last_processed_timestamp)
        if key != self._buffer_cache_key:
            rows=[];buckets={}
            for b in t.dollar_bars:
                label,hour=_timestamp_parts(b.timestamp)
                rows.append([label,b.open,b.high,b.low,b.close,b.volume,b.notional_value,b.is_forward_filled])
                if hour not in buckets:buckets[hour]=[b.high,b.low,b.close,0,b.timestamp]
                item=buckets[hour];item[0]=max(item[0],b.high);item[1]=min(item[1],b.low);item[2]=b.close;item[3]+=1
            self._buffer_hash=hashlib.sha256(canonical(rows).encode()).hexdigest();self._buffer_cache_key=key
            completed=list(buckets.values())[:-1];tr=[]
            for i,item in enumerate(completed):
                tr.append(item[0]-item[1] if i==0 else max(item[0]-item[1],abs(item[0]-completed[i-1][2]),abs(item[1]-completed[i-1][2])))
            positive=[sum(tr[max(0,i-19):i+1])/len(tr[max(0,i-19):i+1]) for i in range(4,len(tr))]
            positive=[v for v in positive if v>0][-t._strategy_config.vol_regime_lookback:]
            self._warmup=dict(accepted_m1=len(t.dollar_bars),lr_required=t.lr_filter.slow_len,lr_ready=len(t.dollar_bars)>=t.lr_filter.slow_len,completed_h1=len(completed),partial_completed_h1=sum(item[3]!=60 for item in completed),positive_atr_observations=len(positive),volatility_minimum_ready=len(positive)>=20,volatility_full_ready=len(positive)>=120,prior_daily_ranges=len(t._daily_ranges),adr_full_ready=len(t._daily_ranges)>=20,initial_bucket_complete=bool(completed and completed[0][3]==60 and completed[0][4].minute==0))
        return normalize({**{k:getattr(t,k) for k in names},'risk':t._risk_manager.to_state_dict(),'bar_count':len(t.dollar_bars),'accepted_buffer_sha256':self._buffer_hash,'consumed_history':self._warmup})

    async def poll(self,event,*,already_admitted=False):
        if not {'receipt_time','request_id','bars','status_code'}<=event.keys():raise ValueError('incomplete poll event')
        receipt=datetime.fromisoformat(event['receipt_time'].replace('Z','+00:00'))
        if receipt.tzinfo is None or (self.previous_receipt and receipt<self.previous_receipt):raise ValueError('receipt order must be causal')
        self.previous_receipt=receipt
        poll_clock=datetime.fromisoformat(event.get('poll_time',event['receipt_time']).replace('Z','+00:00'))
        if poll_clock.tzinfo is None or poll_clock>receipt:raise ValueError('invalid poll clock')
        self.runtime.clock=poll_clock
        self.trader._ts_client.replies=list(event.get('execution_replies',[]))
        self.trader._ts_client.evidence_errors=[]
        if event.get('request_failure') not in (None,'timeout','request_error'):raise ValueError('unknown request failure')
        if event.get('request_failure') and (event['status_code'] is not None or event['bars'] is not None):raise ValueError('failed request cannot contain response')
        if event['status_code']==200 and not isinstance(event['bars'],list):raise ValueError('successful response needs bars')
        before=self.state();start=len(self.trader._ts_client.intentions);decisions=len(self.runtime.decisions);errors=len(self.runtime.logger.errors)
        class Auth:
            async def authenticate(self):return 'OFFLINE_PLACEHOLDER'
        class Response:
            status_code=event['status_code']
            def json(self):return {'Bars':event['bars']}
        class Client:
            async def get(self,*a,**kw):
                if event.get('request_failure')=='timeout':raise asyncio.TimeoutError('captured request timeout')
                if event.get('request_failure')=='request_error':raise RuntimeError('captured request error')
                return Response()
        self.trader.auth=Auth();self.trader.client=Client()
        if not already_admitted and not self.trader._is_market_open():scheduled=False
        else:
            scheduled=True
            await self.trader._poll_and_process()
        if self.trader._ts_client.evidence_errors:raise ValueError('; '.join(self.trader._ts_client.evidence_errors))
        if self.trader._ts_client.replies:raise ValueError('unused execution reply evidence')
        self.sequence+=1
        return dict(sequence=self.sequence,input=event,receipt_time=receipt.isoformat(),poll_time=poll_clock.isoformat(),request_id=event['request_id'],raw_sha256=hashlib.sha256(canonical(event['bars']).encode()).hexdigest(),scheduled=scheduled,before=before,after=self.state(),decisions=self.runtime.decisions[decisions:],intentions=self.trader._ts_client.intentions[start:],account_evidence=self.state_classification,account_verification=self.account_verification,errors=self.runtime.logger.errors[errors:],broker_observations=[],decision='HOLD_VALIDATION')
