"""Disabled prospective hooks. Importing this module starts no worker or collection."""
from collections import deque
import hashlib
import json
from pathlib import Path
import threading
from .adapter import canonical, digest, normalize, SNAPSHOT_MANIFEST_SHA256


def identities(snapshot):
    root=Path(snapshot)
    if digest(root/'manifest.json')!=SNAPSHOT_MANIFEST_SHA256:raise ValueError('snapshot identity mismatch')
    return dict(reference_verification_scope='PINNED_REFERENCE_FILES_NOT_PROCESS_IDENTITY',**{name:digest(root/path) for name,path in {
        'source':'src/research/yank_streaming_working.py','model':'models/xgboost/tier2_meta_labeling_model.pkl',
        'yaml':'strategy_config.yaml','threshold':'models/xgboost/tier2_threshold.json',
        'lr_config':'models/xgboost/lr_regime_config.json','service':'trader-yank.service','snapshot':'manifest.json'}.items()})


def bounded_payload(value,max_nodes,max_bytes):
    """Bound traversal before copying; only known data types can be normalized."""
    import dataclasses
    from datetime import date,datetime,time
    from enum import Enum
    import math
    import numpy as np
    nodes=0;size=0
    forbidden={'authorization','token','password','cookie','account_id','secret','api_key','access_token','refresh_token'}
    def copy(item,depth=0):
        nonlocal nodes,size
        nodes+=1
        if nodes>max_nodes or depth>64:raise ValueError('capture traversal bound')
        kind=type(item)
        if kind is dict:
            if len(item)>(max_nodes-nodes)//2:raise ValueError('capture node bound')
            result={}
            for key,value in item.items():
                if type(key) is not str:raise ValueError('capture key type')
                safe_key=copy(key,depth+1)
                if safe_key.lower() in forbidden:raise ValueError('credential-bearing capture field')
                result[safe_key]=copy(value,depth+1)
            return result
        if kind in (list,tuple):
            if len(item)>max_nodes-nodes:raise ValueError('capture node bound')
            return [copy(v,depth+1) for v in item]
        if kind is str:
            if len(item)>max_bytes-size:raise ValueError('capture byte bound')
            size+=len(item.encode('utf-8'))
            if size>max_bytes:raise ValueError('capture byte bound')
            return item
        if item is None or kind is bool:return item
        if kind is int:
            if item.bit_length()>max_bytes*3:raise ValueError('capture integer bound')
            return item
        if kind is float:return item if math.isfinite(item) else str(item)
        if isinstance(item,datetime):return copy(datetime.isoformat(item),depth+1)
        if kind is date:return copy(date.isoformat(item),depth+1)
        if kind is time:return copy(time.isoformat(item),depth+1)
        if kind in (np.float64,np.float32):return copy(float(item),depth+1)
        if kind in (np.int64,np.int32):return copy(int(item),depth+1)
        # Only the installed strategy's known decisions are accepted outside the
        # private module. No model_dump/iteration/custom serializer is invoked.
        known_live=kind.__module__ in ('src.research.strategy_core','__main__') and kind.__name__ in ('EntryDecision','Direction')
        if kind.__module__.startswith('_yank_private_') or known_live:
            if isinstance(item,Enum):return copy(object.__getattribute__(item,'_value_'),depth+1)
            if dataclasses.is_dataclass(kind):
                fields=dataclasses.fields(kind)
                if known_live and {f.name for f in fields}!={'direction','entry_price','sl_price','tp_price','contracts'}:raise ValueError('unknown live decision shape')
                if len(fields)>(max_nodes-nodes)//2:raise ValueError('capture node bound')
                return copy({field.name:object.__getattribute__(item,field.name) for field in fields},depth+1)
        raise ValueError('capture value type')
    normalized=copy(value)
    encoded=canonical(normalized)
    if len(encoded.encode())>max_bytes:raise ValueError('capture byte bound')
    return encoded


class DecisionCapture:
    """Bounded memory queue; contention/failure/drop permanently invalidates coverage.

    enabled=False is the default and performs no filesystem or worker activity.
    The worker alone writes. Hook callers never wait for a lock or storage.
    """
    def __init__(self,*,enabled=False,output_dir=None,capacity=64,max_bytes=2_000_000,max_nodes=100_000):
        if capacity<1 or max_bytes<1 or max_nodes<1:raise ValueError('positive capture bounds required')
        self.enabled=enabled;self.capacity=capacity;self.max_bytes=max_bytes;self.max_nodes=max_nodes
        self.invalid_reasons=set();self.accepted=0;self.written=0;self.dropped=0
        self.queue=deque();self.lock=threading.Lock();self.wake=threading.Event();self.stopping=False;self.worker=None;self.producers=0
        if not enabled:return
        if output_dir is None:raise ValueError('explicit fresh output required')
        self.output=Path(output_dir)
        if self.output.exists() or self.output.is_symlink():raise ValueError('fresh capture output required')
        self.output.mkdir(parents=True)
        self.worker=threading.Thread(target=self._drain,name='yank-decision-capture',daemon=True);self.worker.start()

    def invalidate(self,reason):self.invalid_reasons.add(reason);self.dropped+=1

    def emit(self,payload):
        if not self.enabled:return False
        if not self.lock.acquire(False):self.invalidate('queue_contention');return False
        try:
            if self.stopping:self.invalidate('capture_after_close');return False
            self.producers+=1
        finally:self.lock.release()
        try:
            try:encoded=bounded_payload(payload,self.max_nodes,self.max_bytes)
            except Exception:self.invalidate('invalid_or_oversize_payload');return False
            if not self.lock.acquire(False):self.invalidate('queue_contention');return False
            try:
                if self.stopping:self.invalidate('capture_closed_during_observation');return False
                if len(self.queue)>=self.capacity:self.invalidate('queue_overflow');return False
                self.queue.append(encoded);self.accepted+=1
            finally:self.lock.release()
            self.wake.set();return True
        finally:self.producers-=1

    def _drain(self):
        try:
            with (self.output/'capture.jsonl').open('x') as f:
                while True:
                    self.wake.wait(.1);self.wake.clear()
                    while True:
                        with self.lock:row=self.queue.popleft() if self.queue else None
                        if row is None:break
                        f.write(row);self.written+=1
                    f.flush()
                    if self.stopping:break
        except Exception:self.invalid_reasons.add('writer_failure')

    def close(self):
        if not self.enabled:return {'enabled':False,'decision':'HOLD_VALIDATION'}
        with self.lock:
            self.stopping=True
            if self.producers:self.invalid_reasons.add('producer_active_at_close')
        self.wake.set();self.worker.join(timeout=2)
        if self.worker.is_alive():self.invalid_reasons.add('writer_not_finished')
        if self.accepted!=self.written:self.invalid_reasons.add('unwritten_records')
        summary=dict(enabled=True,coverage_scope='accepted_prefix_before_close',capture_closed=True,valid_coverage=not self.invalid_reasons,invalid_reasons=sorted(self.invalid_reasons),accepted=self.accepted,written=self.written,dropped=self.dropped,decision='HOLD_VALIDATION')
        if not self.worker.is_alive():
            try:
                path=self.output/'capture.jsonl'
                summary['capture_sha256']=digest(path) if path.exists() else None
                (self.output/'coverage.json').write_text(canonical(summary))
            except Exception:self.invalid_reasons.add('coverage_write_failure');summary['valid_coverage']=False;summary['invalid_reasons']=sorted(self.invalid_reasons)
        return summary


class DecisionHooks:
    """Separate hook surface for a future explicitly approved installation.

    Call at poll completion with exact raw envelope and before/after decision state.
    Capture clock readings at original call sites; never fabricate them from labels.
    The API is deliberately separate from settled parity and bullish trade logging.
    """
    def __init__(self,capture):self.capture=capture
    def poll_completed(self,*,trace,identity,readiness):
        if not self.capture.enabled:return False
        return self.capture.emit(dict(schema_version=1,kind='poll',trace=trace,identities=identity,readiness=readiness))
    def broker_observed(self,*,venue,kind,receipt_time,observation):
        if not self.capture.enabled:return False
        if venue not in ('ProjectX','TradeStation_SIM') or kind not in ('acknowledgement','fill','rejection','cancel_acknowledgement'):
            self.capture.invalidate('invalid_broker_observation');return False
        # Caller must supply a credential-free normalized observation. It is never
        # substituted for local simulated fills or for another execution venue.
        return self.capture.emit(dict(schema_version=1,kind='broker_observation',venue=venue,observation_kind=kind,receipt_time=receipt_time,observation=observation))


def install_poll_observer(trader,capture,*,identity,readiness,state_reader):
    """Explicit opt-in helper; disabled capture returns without touching trader.

    Records response ordering and every datetime.now observation during the exact
    original poll. It never calls initialize. Installation remains a proposal.
    state_reader must provide the complete normalized Adapter.state contract.
    Decision and order-intention methods are observed without changing returned values.
    """
    if not capture.enabled:return lambda:None
    import contextvars
    from datetime import datetime,timezone,timedelta
    import sys
    module=sys.modules[trader.__class__.__module__]
    if hasattr(trader,'_decision_capture'):raise ValueError('capture already installed')
    original_poll=trader._poll_and_process;original_clock=module.datetime;original_logger=module.logger
    original_detect=trader._detect_and_enter;original_filter_log=trader._log_filter_decision;original_predict=trader.ml_filter.predict_proba
    context=contextvars.ContextVar('yank_decision_poll',default=None)
    class ObservedClock:
        def __getattr__(self,name):return getattr(original_clock,name)
        def now(self,*args,**kwargs):
            value=original_clock.now(*args,**kwargs);current=context.get()
            if current is not None:
                try:record('clock_reads',value.isoformat())
                except Exception:capture.invalidate('clock_capture_failed')
            return value
    module.datetime=ObservedClock()
    trader._decision_capture=capture
    identity_scope='UNVERIFIED_LIVE_DECLARATION'
    runtime=getattr(trader,'_private_snapshot_runtime',None)
    if runtime is not None and runtime.module is module and runtime.module.Tier2StreamingTrader is trader.__class__:
        runtime.verify()
        if identity==identities(runtime.root):identity_scope='PRIVATE_PINNED_OFFLINE_RUNTIME'
    hooks=DecisionHooks(capture);sequence=0
    def record(kind,value):
        current=context.get()
        if current is not None and not current['observation_dropped']:
            try:
                encoded=bounded_payload(value,capture.max_nodes,capture.max_bytes)
                size=len(encoded.encode())
                if current['observation_bytes']+size>capture.max_bytes or current['observation_records']>=capture.max_nodes:
                    current['observation_dropped']=True;capture.invalidate('poll_observation_overflow');return
                current[kind].append(json.loads(encoded));current['observation_bytes']+=size;current['observation_records']+=1
            except Exception:
                current['observation_dropped']=True;capture.invalidate('decision_capture_failed')
    def filter_log(*args,**kwargs):
        result=original_filter_log(*args,**kwargs)
        record('decisions',dict(args=args,kwargs=kwargs));return result
    def predict(features):
        result=original_predict(features)
        record('decisions',dict(kind='ml_prediction',features=features,probability=result,threshold=trader.ml_filter.threshold));return result
    async def detect(bar,is_backfill):
        current=context.get()
        if current is None or current['observation_dropped']:return await original_detect(bar,is_backfill)
        try:before=state_reader()
        except Exception:before=None;capture.invalidate('decision_state_capture_failed')
        current=context.get();offset=len(current['intentions']) if current is not None else 0
        try:record('decision_times',datetime.now(timezone.utc).isoformat())
        except Exception:capture.invalidate('decision_time_capture_failed')
        result=await original_detect(bar,is_backfill)
        try:record('decisions',dict(kind='decision_transition',bar_time=bar.timestamp.isoformat(),before=before,after=state_reader(),intentions=current['intentions'][offset:] if current is not None else []))
        except Exception:capture.invalidate('decision_state_capture_failed')
        return result
    class ObservedLogger:
        def __getattr__(self,name):return getattr(original_logger,name)
        def error(self,message,*args,**kwargs):
            try:record('errors',message % args if args else str(message))
            except Exception:capture.invalidate('error_capture_failed')
            return original_logger.error(message,*args,**kwargs)
    module.logger=ObservedLogger()
    trader._log_filter_decision=filter_log;trader.ml_filter.predict_proba=predict;trader._detect_and_enter=detect
    async def observed_poll():
        nonlocal sequence
        observation={'bars':None,'status_code':None,'request_failure':None,'poll_observation':'already_admitted','label_evidence':[],'clock_reads':[],'decision_times':[],'decisions':[],'intentions':[],'execution_replies':[],'errors':[],'observation_bytes':0,'observation_records':0,'observation_dropped':False};token=context.set(observation)
        original_client=trader.client;original_execution=trader._ts_client
        class Execution:
            def __getattr__(self,name):return getattr(original_execution,name)
            async def submit_bracket_order(self,decision,account):
                record('intentions',dict(kind='submit_bracket',decision=decision,venue='ProjectX',evidence='INTENTION_ONLY'))
                value=await original_execution.submit_bracket_order(decision,account)
                record('execution_replies',dict(operation='submit_bracket_order',value=value))
                return value
            async def place_exit_orders(self,decision,account):
                record('intentions',dict(kind='place_exits',decision=decision,venue='ProjectX',evidence='INTENTION_ONLY'))
                value=await original_execution.place_exit_orders(decision,account)
                record('execution_replies',dict(operation='place_exit_orders',value=value))
                return value
            async def cancel_order(self,order_id):
                record('intentions',dict(kind='cancel',order_id=order_id,venue='ProjectX',evidence='INTENTION_ONLY'))
                value=await original_execution.cancel_order(order_id)
                record('execution_replies',dict(operation='cancel_order',value=value))
                return value
            async def close_position_at_market(self,direction,account,contracts=None):
                record('intentions',dict(kind='close',direction=direction,quantity=contracts,venue='ProjectX',evidence='INTENTION_ONLY'))
                value=await original_execution.close_position_at_market(direction,account,contracts)
                record('execution_replies',dict(operation='close_position_at_market',value=value))
                return value
        trader._ts_client=Execution()
        before=None
        try:before=state_reader()
        except Exception:capture.invalidate('state_capture_failed')
        class Response:
            def __init__(self,value):self.value=value
            def __getattr__(self,name):return getattr(self.value,name)
            def json(self):
                value=self.value.json()
                try:
                    bars=value.get('Bars',[])
                    # Serialization bound protects observation work. Return the
                    # original response object untouched even on logging failure.
                    encoded=bounded_payload(bars,capture.max_nodes,capture.max_bytes)
                    observation['bars']=json.loads(encoded)
                    for bar in observation['bars']:
                        original=bar.get('TimeStamp') if type(bar) is dict else None
                        evidence=dict(original_label=original,semantics='UNVERIFIED',normalized_utc_label=None,start_labeled_interval=None,end_labeled_interval=None)
                        try:
                            label=datetime.fromisoformat(original.replace('Z','+00:00'))
                            if label.tzinfo is None:raise ValueError('naive label')
                            label=label.astimezone(timezone.utc)
                            evidence.update(normalized_utc_label=label.isoformat(),start_labeled_interval=[label.isoformat(),(label+timedelta(minutes=1)).isoformat()],end_labeled_interval=[(label-timedelta(minutes=1)).isoformat(),label.isoformat()])
                        except (ValueError,AttributeError,TypeError):pass
                        record('label_evidence',evidence)
                except Exception:capture.invalidate('response_capture_failed')
                return value
        class Client:
            def __getattr__(self,name):return getattr(original_client,name)
            async def get(self,url,*args,**kwargs):
                observed='/marketdata/barcharts/' in str(url)
                if observed:observation['request_id']=hashlib.sha256(str(url).encode()).hexdigest()
                try:response=await original_client.get(url,*args,**kwargs)
                except Exception as exc:
                    if observed:
                        observation['receipt_time']=datetime.now(timezone.utc).isoformat()
                        observation['request_failure']='timeout' if isinstance(exc,(TimeoutError,module.httpx.TimeoutException)) else 'request_error'
                    raise
                if observed:
                    observation['receipt_time']=datetime.now(timezone.utc).isoformat()
                    observation['status_code']=response.status_code
                    try:
                        content=getattr(response,'content',None)
                        if type(content) is bytes:
                            if len(content)>capture.max_bytes:capture.invalidate('response_body_exceeds_capture_bound')
                            else:observation['raw_http_bytes_sha256']=hashlib.sha256(content).hexdigest()
                    except Exception:pass  # Unavailable/streaming body identity stays unknown.
                    return Response(response)
                return response
        trader.client=Client()
        try:return await original_poll()
        finally:
            trader.client=original_client;trader._ts_client=original_execution;context.reset(token);sequence+=1
            try:
                if not observation['clock_reads'] or 'receipt_time' not in observation or before is None or (observation['status_code']==200 and observation['bars'] is None):raise ValueError('missing poll evidence')
                poll_time=observation['clock_reads'][0]
                event={k:observation[k] for k in ('bars','receipt_time','request_id','status_code','clock_reads','decision_times','execution_replies','request_failure','poll_observation','label_evidence')}
                event['poll_time']=poll_time
                event['bars_hash_scope']='canonical_parsed_Bars_not_HTTP_bytes'
                event['raw_http_bytes_sha256']=observation.get('raw_http_bytes_sha256')
                trace=dict(sequence=sequence,input=event,poll_time=poll_time,receipt_time=event['receipt_time'],request_id=event['request_id'],raw_sha256=hashlib.sha256(canonical(event['bars']).encode()).hexdigest(),scheduled=True,before=before,after=state_reader(),decisions=observation['decisions'],intentions=observation['intentions'],errors=observation['errors'],broker_observations=[],decision='HOLD_VALIDATION')
                qualified=dict(readiness,clock_semantics='observed_per_call',identity_verification_scope=identity_scope)
                hooks.poll_completed(trace=trace,identity=identity,readiness=qualified)
            except Exception:capture.invalidate('poll_capture_failed')
    trader._poll_and_process=observed_poll
    def rollback():
        trader._poll_and_process=original_poll;trader._detect_and_enter=original_detect
        trader._log_filter_decision=original_filter_log;trader.ml_filter.predict_proba=original_predict
        if isinstance(module.datetime,ObservedClock):module.datetime=original_clock
        if isinstance(module.logger,ObservedLogger):module.logger=original_logger
        if getattr(trader,'_decision_capture',None) is capture:del trader._decision_capture
    return rollback
