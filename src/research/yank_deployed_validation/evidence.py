"""Collector provenance admission. Public trust inputs must arrive independently.

A signature authenticates the collector's statement, never the integrity of its host.
No key generation, secret loading, authentication or collection occurs on import.
"""
import base64
import hashlib
from .adapter import canonical

VERSION = 1


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def capture_digest(rows):
    h = hashlib.sha256()
    for row in rows: h.update(canonical(row).encode())
    return h.hexdigest()


def sign_bundle(payload, key_id, signer):
    """Signer is an externally provisioned Ed25519 private-key/HSM interface."""
    message = dict(schema_version=VERSION, key_id=key_id, payload=payload)
    return {**message, 'signature': base64.b64encode(signer.sign(canonical(message).encode())).decode()}


def chronology_reasons(rows):
    """Validate recorded order; never repair it by sorting observations."""
    from datetime import datetime, timezone
    def instant(value):
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None: raise ValueError('naive chronology')
        return parsed.astimezone(timezone.utc)
    reasons = []; previous_receipt = None; previous_decision = None; previous_account = None; previous_broker = None
    requests = {}; replied = set()
    try:
        for row in rows:
            kind = row.get('kind')
            if kind == 'account_request':
                rid = row['request_id']; when = instant(row['request_time'])
                if rid in requests or (previous_account is not None and when < previous_account): raise ValueError('account_request_order')
                requests[rid] = row; previous_account = when
            elif kind == 'account_boundary':
                rid = row['request_id']; when = instant(row['receipt_time'])
                if rid not in requests or rid in replied: raise ValueError('account_reply_without_unique_request')
                if any(row.get(k) != requests[rid].get(k) for k in ('request_time', 'endpoint', 'account_pseudonym', 'contract')): raise ValueError('account_request_reply_disagreement')
                if instant(row['request_time']) > when or (previous_account is not None and when < previous_account): raise ValueError('account_receipt_order')
                replied.add(rid); previous_account = when
            elif kind == 'broker_observation':
                when = instant(row['receipt_time'])
                if previous_broker is not None and when < previous_broker: raise ValueError('broker_receipt_order')
                previous_broker = when
            elif kind == 'poll':
                trace = row['trace']; event = trace['input']
                start = instant(trace['poll_time']); receipt = instant(trace['receipt_time'])
                if start > receipt or (previous_receipt is not None and receipt < previous_receipt): raise ValueError('poll_receipt_order')
                if previous_decision is not None and start < previous_decision: raise ValueError('overlapping_or_backwards_polls')
                if instant(event['receipt_time']) != receipt or instant(event.get('poll_time', trace['poll_time'])) != start: raise ValueError('poll_boundary_disagreement')
                clocks = [instant(v) for v in event.get('clock_reads', [])]
                if not clocks or clocks[0] != start or any(a > b for a, b in zip(clocks, clocks[1:])): raise ValueError('clock_read_order')
                times = [instant(v) for v in event.get('decision_times', [])]
                transitions = [d for d in trace.get('decisions', []) if d.get('kind') == 'decision_transition']
                if len(times) != len(transitions): raise ValueError('decision_boundary_count')
                if any(t < receipt for t in times) or any(a > b for a, b in zip(times, times[1:])): raise ValueError('decision_boundary_order')
                previous_receipt = receipt; previous_decision = times[-1] if times else receipt
        if set(requests) != replied: raise ValueError('missing_account_reply')
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
        reasons.append('invalid_capture_chronology:' + str(exc))
    return reasons


def verify_bundle(bundle, trusted_keys, expected, rows, coverage, initial_state, snapshot=None):
    """Never consume trust declarations, public keys or release pins from a capture."""
    verdict = dict(signature='UNKNOWN', identity='UNKNOWN', checkpoint='UNKNOWN',
                   account='UNKNOWN', coverage='UNKNOWN', eligible=False,
                   decision='HOLD_VALIDATION', reasons=[])
    if bundle is None:
        verdict['reasons'].append('signed_collector_evidence_unavailable')
        return verdict
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if set(bundle) != {'schema_version', 'key_id', 'payload', 'signature'} or bundle['schema_version'] != VERSION:
            raise ValueError('bundle schema')
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(trusted_keys[bundle['key_id']], validate=True))
        message = {k: bundle[k] for k in ('schema_version', 'key_id', 'payload')}
        key.verify(base64.b64decode(bundle['signature'], validate=True), canonical(message).encode())
        verdict['signature'] = 'PASS'
    except Exception:
        verdict['signature'] = 'FAIL'
        verdict['reasons'].append('signature_invalid_or_key_untrusted')
        return verdict
    p = bundle['payload']
    try:
        required = {'runtime', 'process', 'checkpoint_sha256', 'initial_state_sha256', 'capture_sha256',
                    'coverage_sha256', 'collector_sha256'}
        if not required <= p.keys(): raise ValueError('incomplete payload')
        process = p['process']
        verdict['identity'] = 'PASS' if (
            p['runtime'] == expected['runtime'] and p['collector_sha256'] == expected['collector_sha256']
            and process == expected['process'] and set(process) == {'boot_id', 'pid', 'start_ticks'}
            and all(process.values())) else 'FAIL'
        checkpoint = initial_state.get('checkpoint')
        checkpoint_valid = False
        if snapshot is not None:
            try:
                from .adapter import Adapter
                Adapter(snapshot, initial_state)
                checkpoint_valid = True
            except (ValueError, KeyError, TypeError, AttributeError): pass
        verdict['checkpoint'] = 'PASS' if (checkpoint_valid and checkpoint is not None
            and sha(checkpoint) == p['checkpoint_sha256'] == expected['checkpoint_sha256']
            and sha(initial_state) == p['initial_state_sha256']
            and initial_state.get('classification') == 'OBSERVED_STATE') else 'FAIL'
        polls = [r for r in rows if r.get('kind') == 'poll']
        verdict['coverage'] = 'PASS' if (bool(polls) and coverage.get('valid_coverage') is True
            and coverage.get('capture_closed') is True and not coverage.get('invalid_reasons')
            and coverage.get('accepted') == coverage.get('written') == len(rows)
            and coverage.get('dropped') == 0 and capture_digest(rows) == p['capture_sha256']
            and sha(coverage) == p['coverage_sha256']
            and [r['trace']['sequence'] for r in polls] == list(range(1, len(polls)+1))) else 'FAIL'
        chronology = chronology_reasons(rows)
        if chronology:
            verdict['coverage'] = 'FAIL'
            verdict['reasons'].extend(chronology)
        from .account import account_verdict
        account_results = []
        for row in polls:
            trace = row['trace']; event = trace.get('input', {})
            decision_times = event.get('decision_times', [])
            at = decision_times[-1] if decision_times else trace.get('receipt_time', trace['poll_time'])
            result = account_verdict([r for r in rows if r.get('kind') in ('account_boundary', 'account_request')], at,
                       expected['account_pseudonym'], expected['contract'], expected['account_max_age_seconds'], interval_start=trace['poll_time'])
            if any(d.get('kind') == 'decision_transition' for d in trace.get('decisions', [])) and not decision_times:
                result.update(verdict='UNKNOWN', state='UNKNOWN', reasons=['decision_boundary_time_unavailable'])
            account_results.append(result)
        for row, result in zip(polls, account_results):
            before = row['trace'].get('before', {})
            if not isinstance(before, dict) or 'active_trade' not in before:
                result.update(verdict='UNKNOWN', state='UNKNOWN', reasons=['checkpoint_state_unavailable'])
            active = before.get('active_trade')
            decision = before.get('_active_entry_decision')
            expected_state = 'FLAT' if active is None else ('PENDING' if active.get('pending_entry') else 'ACTIVE')
            if result['verdict'] == 'PASS' and result['state'] != expected_state:
                result.update(verdict='UNKNOWN', state='UNKNOWN', reasons=['account_checkpoint_state_contradiction'])
            if result['verdict'] == 'PASS' and expected_state == 'ACTIVE':
                quantity = decision.get('contracts') if isinstance(decision, dict) else None
                sign = 1 if active.get('direction') == 'bullish' else -1 if active.get('direction') == 'bearish' else 0
                if type(quantity) is not int or not sign or result['quantity'] != sign * quantity:
                    result.update(verdict='UNKNOWN', state='UNKNOWN', reasons=['account_checkpoint_quantity_contradiction'])
            if result['verdict'] == 'PASS' and expected_state == 'PENDING':
                from decimal import Decimal, InvalidOperation
                orders = result['orders']
                try:
                    if not isinstance(decision, dict) or type(decision.get('contracts')) is not int or decision['contracts'] <= 0 or len(orders) != 1: raise ValueError('ambiguous pending exposure')
                    order = orders[0]
                    side = 0 if decision.get('direction') == 'bullish' else 1 if decision.get('direction') == 'bearish' else None
                    if side is None or order.get('type') != 1 or order.get('side') != side or order.get('size') != decision.get('contracts'):
                        raise ValueError('pending quantity/side/type mismatch')
                    price = Decimal(str(order.get('limitPrice'))); expected_price = Decimal(str(decision['entry_price']))
                    if not price.is_finite() or not expected_price.is_finite() or price != expected_price: raise ValueError('pending entry price mismatch')
                except (KeyError, TypeError, ValueError, InvalidOperation):
                    result.update(verdict='UNKNOWN', state='UNKNOWN', reasons=['account_checkpoint_pending_contradiction'])
        verdict['account'] = 'PASS' if account_results and all(x['verdict'] == 'PASS' for x in account_results) else 'UNKNOWN'
        verdict['account_details'] = account_results
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        verdict['reasons'].append('missing_or_malformed_independent_binding')
    verdict['eligible'] = all(verdict[k] == 'PASS' for k in ('signature', 'identity', 'checkpoint', 'account', 'coverage'))
    for k in ('identity', 'checkpoint', 'account', 'coverage'):
        if verdict[k] != 'PASS': verdict['reasons'].append(k + '_not_admitted')
    return verdict
