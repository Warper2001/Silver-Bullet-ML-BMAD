"""Disabled maintenance-window integration for a separately approved collector.

Release expectations are built and distributed independently. Never derive the
expected release from the process being admitted. Signers remain external.
"""
import inspect
import os
from pathlib import Path
import sys
import types
from .adapter import Adapter, canonical, digest, normalize
from .capture import DecisionCapture, install_poll_observer
from .evidence import sha, sign_bundle


def process_identity():
    # Linux boot + PID + kernel process start ticks disambiguate PID reuse.
    stat = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()
    return dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), pid=os.getpid(), start_ticks=int(stat[19]))


def collector_identity():
    return {name: digest(Path(__file__).with_name(name)) for name in ('startup.py', 'evidence.py', 'account.py', 'capture.py', 'adapter.py')}


def code_identity(function, _seen=None):
    function = getattr(function, '__func__', function)
    _seen = set() if _seen is None else set(_seen)
    if id(function) in _seen: return 'recursive:' + function.__qualname__
    if len(_seen) >= 256: raise ValueError('runtime dependency traversal bound')
    _seen.add(id(function))
    if not inspect.isfunction(function): raise ValueError('unverifiable runtime callable')
    def code(c):
        return dict(bytecode=c.co_code.hex(), constants=[code(x) if isinstance(x, types.CodeType) else normalize(x) if isinstance(x, (str, int, float, bool, tuple, frozenset, type(None))) else repr(x) for x in c.co_consts],
                    names=c.co_names, varnames=c.co_varnames, freevars=c.co_freevars, cellvars=c.co_cellvars,
                    argcount=c.co_argcount, posonly=c.co_posonlyargcount, kwonly=c.co_kwonlyargcount,
                    flags=c.co_flags, exceptiontable=getattr(c, 'co_exceptiontable', b'').hex())
    def referenced(c):
        names = set(c.co_names)
        for value in c.co_consts:
            if isinstance(value, types.CodeType): names.update(referenced(value))
        return names
    def project_module(value):
        return isinstance(value, types.ModuleType) and (value.__name__.startswith('src.') or value.__name__.startswith('_yank_private_'))
    def binding(value):
        if project_module(value):
            if id(value) in _seen: return {'recursive_module': value.__name__.split('_src_')[-1]}
            if len(_seen) >= 256: raise ValueError('runtime dependency traversal bound')
            _seen.add(id(value))
            try:
                selected = {}
                for attr in sorted(referenced(function.__code__)):
                    if attr not in vars(value) or attr in ('datetime', 'logger'): continue
                    member = vars(value)[attr]
                    if project_module(member) or inspect.isfunction(member) or inspect.isclass(member) or type(member) in (str, int, float, bool, tuple, list, dict, set, frozenset, type(None)) or type(member).__module__ in ('zoneinfo', 'datetime'):
                        selected[attr] = binding(member)
                return {'project_module_attributes': selected}
            finally: _seen.remove(id(value))
        if inspect.isfunction(value): return {'function': code_identity(value, _seen)}
        if inspect.isclass(value) or inspect.isbuiltin(value): return {'callable': value.__module__ + '.' + value.__qualname__}
        if type(value) in (str, int, float, bool, tuple, list, dict, set, frozenset, type(None)): return normalize(value)
        if type(value).__module__ == 'zoneinfo': return {'timezone': value.key}
        if type(value).__module__.startswith('pytz') and hasattr(value, 'zone'): return {'timezone': value.zone}
        if type(value).__module__ == 'datetime': return {'datetime_value': str(value)}
        if type(value).__module__ == 'dataclasses': return {'dataclass_sentinel': type(value).__name__}
        raise ValueError('unverifiable runtime closure/default: ' + type(value).__name__)
    closure = [binding(cell.cell_contents) for cell in function.__closure__ or ()]
    # Pin helpers in their defining modules, including referenced module attributes.
    globals_pin = {}
    for name in sorted(referenced(function.__code__)):
        if name in ('datetime', 'logger') or name not in function.__globals__: continue
        value = function.__globals__[name]
        if project_module(value):
            globals_pin[name] = binding(value)
        elif inspect.isfunction(value) and (value.__module__.startswith('src.') or value.__module__.startswith('_yank_private_')):
            globals_pin[name] = binding(value)
        elif type(value) in (str, int, float, bool, tuple, list, dict, set, frozenset, type(None)) or type(value).__module__ in ('zoneinfo', 'datetime') or type(value).__module__.startswith('pytz'):
            globals_pin[name] = binding(value)
    return sha(dict(code=code(function.__code__), closure=closure, globals=globals_pin,
                    defaults=[binding(x) for x in function.__defaults__ or ()],
                    kwdefaults={k: binding(v) for k, v in (function.__kwdefaults__ or {}).items()}))



def runtime_identity(trader, *, include_methods=True, original_bindings=None):
    """Fingerprint loaded method objects, model and effective strategy settings.

    This is a release gate within a trusted collector, not host attestation.
    Instance overrides and imported strategy functions participate in the pin.
    """
    import joblib
    module = sys.modules[type(trader).__module__]
    methods = {}
    original_bindings = original_bindings or {}
    def effective(obj, name): return original_bindings.get((id(obj), name), getattr(obj, name))
    for name, value in (vars(type(trader)).items() if include_methods else ()):
        if inspect.isfunction(value): methods['trader.' + name] = code_identity(effective(trader, name))
    for label, obj in ((('ml', trader.ml_filter), ('lr', trader.lr_filter), ('risk', trader._risk_manager)) if include_methods else ()):
        for name, value in vars(type(obj)).items():
            if inspect.isfunction(value): methods[label + '.' + name] = code_identity(effective(obj, name))
    for name, value in (vars(module).items() if include_methods else ()):
        if inspect.isfunction(value): methods['module.' + name] = code_identity(value)
        elif inspect.isclass(value) and (value.__module__.startswith('src.') or value.__module__ == module.__name__):
            for parent in value.__mro__:
                if not (parent.__module__.startswith('src.') or parent.__module__ == module.__name__): continue
                for attr, member in vars(parent).items():
                    member = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
                    if inspect.isfunction(member): methods['class.' + name + '.' + parent.__name__ + '.' + attr] = code_identity(member)
                    elif isinstance(member, property):
                        for accessor in ('fget', 'fset', 'fdel'):
                            function = getattr(member, accessor)
                            if function is not None: methods['property.' + name + '.' + parent.__name__ + '.' + attr + '.' + accessor] = code_identity(function)
    globals_pin = {}
    for name, value in vars(module).items():
        if name.isupper() and type(value) in (str, int, float, bool, dict, list, tuple, set, frozenset, type(None)):
            globals_pin[name] = normalize(value)
    settings = {k: normalize(getattr(trader, k)) for k in ('_symbol', '_point_value', '_tick_size', '_contracts', '_on_combine', '_data_source', '_data_shadow', '_data_px_live')}
    inference = {}
    for label, obj in (('ml', trader.ml_filter), ('lr', trader.lr_filter), ('risk', trader._risk_manager)):
        values = {}
        for parent in reversed(type(obj).__mro__):
            for name, value in vars(parent).items():
                if not name.startswith('__') and type(value) in (str, int, float, bool, tuple, list, dict, set, frozenset, type(None)):
                    values[name] = normalize(value)
        excluded = {'model'} if label == 'ml' else {'_daily_pnl', '_daily_halted', '_last_trading_date'} if label == 'risk' else set()
        for name, value in vars(obj).items():
            if name not in excluded and not callable(value): values[name] = normalize(value)
        inference[label] = values
    settings.update(inference=inference, strategy=normalize(trader._strategy_config), threshold=trader.ml_filter.threshold,
                    lr={k: normalize(v) for k, v in vars(trader.lr_filter).items() if not k.startswith('_')}, globals=globals_pin)
    if trader.ml_filter.model is None: raise ValueError('missing loaded model')
    return dict(methods=methods, effective_configuration_sha256=sha(settings), model_sha256=sha(joblib.hash(trader.ml_filter.model, hash_name='sha1')))


class ObservationSession:
    def __init__(self, capture, initial_state, runtime, process, key_id, signer, trader, bindings, collector):
        self.capture = capture; self.initial_state = initial_state
        self.collector = collector; self.trader = trader; self.bindings = bindings
        self.runtime = runtime; self.process = process; self.key_id = key_id; self.signer = signer
        self.closed = False; self.summary = None
        self.installed_code = {}

    @staticmethod
    def binding_code(value):
        function = getattr(value, '__func__', value)
        code = getattr(function, '__code__', None)
        if code is not None: return (function, code)
        return tuple((name, member, member.__code__) for name, member in vars(type(value)).items() if inspect.isfunction(member))

    def check_runtime(self, *, quiescent=False):
        try:
            if any(getattr(obj, name, None) is not installed for obj, name, original, installed in self.bindings):
                self.capture.invalidate('installed_binding_changed')
            if any(self.binding_code(installed) != self.installed_code[(id(obj), name)] for obj, name, original, installed in self.bindings):
                self.capture.invalidate('installed_observer_code_changed')
            originals = {(id(obj), name): original for obj, name, original, installed in self.bindings}
            if runtime_identity(self.trader, original_bindings=originals) != self.runtime:
                self.capture.invalidate('runtime_changed_during_capture')
            if os.getpid() != self.process['pid']:
                self.capture.invalidate('collector_or_process_changed')
            if quiescent and (collector_identity() != self.collector or process_identity() != self.process):
                self.capture.invalidate('collector_or_process_changed')
        except Exception: self.capture.invalidate('runtime_verification_failed')

    def rollback(self):
        # Restore only bindings still owned by this session. Another owner must
        # never have its wrappers erased by a stale close/rollback call.
        for obj, name, original, installed in reversed(self.bindings):
            if getattr(obj, name, None) is installed:
                if name == '_decision_capture': delattr(obj, name)
                else: setattr(obj, name, original)

    def close(self):
        """Quiescent, idempotent close. Final manifest is the completion marker."""
        if self.closed: return self.summary
        self.closed = True
        self.check_runtime(quiescent=True)  # Verify before rollback can hide changed wrappers.
        self.rollback()
        coverage = self.capture.close()
        root = self.capture.output
        def publish(name, value):
            temporary = root / (name + '.pending')
            temporary.write_text(canonical(value))
            temporary.replace(root / name)
        try:
            if self.capture.worker.is_alive(): raise ValueError('storage worker incomplete')
            payload = dict(runtime=self.runtime, process=self.process, collector_sha256=self.collector,
                           checkpoint_sha256=sha(self.initial_state['checkpoint']), initial_state_sha256=sha(self.initial_state),
                           capture_sha256=digest(root/'capture.jsonl'), coverage_sha256=sha(coverage))
            bundle = sign_bundle(payload, self.key_id, self.signer)
            publish('evidence.json', bundle)
            manifest = dict(schema_version=1, publication='COMPLETE', initial_state=self.initial_state,
                            capture_sha256=digest(root/'capture.jsonl'), coverage_sha256=digest(root/'coverage.json'),
                            evidence_sha256=digest(root/'evidence.json'))
            publish('manifest.json', manifest)  # Published last, atomic rename.
        except Exception:
            self.capture.invalid_reasons.add('package_publication_failed')
            coverage = dict(coverage, valid_coverage=False, signing_error=True,
                            invalid_reasons=sorted(self.capture.invalid_reasons))
            # Remove all admission artifacts before attempting failure metadata.
            for name in ('manifest.json', 'evidence.json', 'manifest.json.pending', 'evidence.json.pending'):
                try: (root/name).unlink(missing_ok=True)
                except OSError: pass
            try: publish('coverage.json', coverage)
            except OSError: pass  # Absent completion manifest remains fail-closed.
        self.summary = coverage
        return coverage


def prepare_observation(trader, *, enabled=False, expected_release=None, output_dir=None,
                        signer=None, key_id=None, pseudonym_salt=None, capture_limits=None):
    """No wrappers on disabled/mismatched startup; no initialize or network calls.

    expected_release must come from independent release engineering. The caller
    cannot pass identity, readiness, state_reader or account trust declarations.
    Invoke only while polling is quiescent; close in the same quiescent state.
    """
    if not enabled: return dict(enabled=False, decision='HOLD_VALIDATION')
    try:
        if signer is None or not key_id or not pseudonym_salt: raise ValueError('external signer and pseudonym salt required')
        runtime = runtime_identity(trader)
        if runtime != expected_release['runtime'] or collector_identity() != expected_release['collector_sha256']:
            raise ValueError('loaded runtime or collector release mismatch')
        if hasattr(trader, '_decision_capture'): raise ValueError('capture already installed')
        client = trader._ts_client
        if not hasattr(client, '_http') or not hasattr(client, '_contract_id'): raise ValueError('direct ProjectX boundary unavailable; mirror integration requires approved adapter')
        from .account import install_account_observer, pseudonym
        if pseudonym(client._account_id, pseudonym_salt) != expected_release['account_pseudonym'] or client._contract_id != expected_release['contract']:
            raise ValueError('account or contract mismatch')
        view = Adapter.__new__(Adapter); view.trader = trader
        view._buffer_cache_key = None; view._buffer_hash = None; view._warmup = {}
        checkpoint = view.checkpoint()
        if sha(checkpoint) != expected_release['checkpoint_sha256']: raise ValueError('checkpoint mismatch')
        risk = checkpoint['state']['risk']
        initial = dict(classification='OBSERVED_STATE', **risk, on_combine=trader._on_combine, is_backfill=trader._is_backfill,
                       checkpoint=checkpoint, account_evidence=dict(status='OBSERVED', source_sha256=sha(checkpoint), receipt_time=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    except Exception as exc:
        return dict(enabled=False, decision='HOLD_VALIDATION', error=str(exc))
    limits = capture_limits or {}
    if set(limits) - {'capacity', 'max_bytes', 'max_nodes'}: raise ValueError('unknown capture bound')
    capture = DecisionCapture(enabled=True, output_dir=output_dir, **limits)
    undo_account = lambda: None; undo_poll = lambda: None
    module = sys.modules[type(trader).__module__]
    saved_module = {k: getattr(module, k) for k in ('datetime', 'logger')}
    saved_trader = {k: getattr(trader, k) for k in ('_poll_and_process', '_detect_and_enter', '_log_filter_decision')}
    saved_predict = trader.ml_filter.predict_proba
    original_http = client._http
    collector = collector_identity()
    try:
        undo_account = install_account_observer(client, capture, client._account_id, client._contract_id, pseudonym_salt)
        # Readiness is collector-owned; feed/state equality is still tested offline.
        undo_poll = install_poll_observer(trader, capture, identity=expected_release['snapshot_identity'],
                                         readiness=dict(equivalent_feed_and_state=True), state_reader=view.state)
    except Exception:
        undo_poll(); undo_account()
        for k, value in saved_module.items(): setattr(module, k, value)
        for k, value in saved_trader.items(): setattr(trader, k, value)
        trader.ml_filter.predict_proba = saved_predict
        if getattr(trader, '_decision_capture', None) is capture: del trader._decision_capture
        capture.invalidate('installation_failed'); capture.close()
        raise
    observed_poll = trader._poll_and_process
    bindings = [(module, name, value, getattr(module, name)) for name, value in saved_module.items()]
    bindings += [(trader, name, value, getattr(trader, name)) for name, value in saved_trader.items()]
    bindings += [(trader.ml_filter, 'predict_proba', saved_predict, trader.ml_filter.predict_proba),
                 (client, '_http', original_http, client._http),
                 (trader, '_decision_capture', None, capture)]
    session = ObservationSession(capture, initial, runtime, process_identity(), key_id, signer, trader, bindings, collector)
    async def guarded_poll():
        session.check_runtime()
        try: return await observed_poll()
        finally: session.check_runtime()
    trader._poll_and_process = guarded_poll
    session.bindings = [(obj, name, original, guarded_poll if obj is trader and name == '_poll_and_process' else installed)
                        for obj, name, original, installed in bindings]
    session.installed_code = {(id(obj), name): session.binding_code(installed) for obj, name, original, installed in session.bindings}
    return session
