"""Passive, redacted ProjectX HTTP boundaries; never changes reconciliation."""
from datetime import datetime, timezone
import hashlib
import hmac
import math

ENDPOINTS = {'/Account/search': 'accounts', '/Order/searchOpen': 'orders', '/Position/searchOpen': 'positions'}
FIELDS = {'accounts': {'balance', 'canTrade', 'isVisible', 'simulated'},
          'orders': {'contractId', 'status', 'type', 'side', 'size', 'limitPrice', 'stopPrice'},
          'positions': {'contractId', 'type', 'size', 'averagePrice', 'profitAndLoss'}}


def pseudonym(account, salt):
    return hmac.new(salt, str(account).encode(), hashlib.sha256).hexdigest()


def account_verdict(observations, at, account, contract, max_age_seconds=30, interval_start=None):
    """Latest completed reply per endpoint, bounded by decision receipt time."""
    result = dict(verdict='UNKNOWN', state='UNKNOWN', reasons=[])
    try:
        now = datetime.fromisoformat(at.replace('Z', '+00:00'))
        if now.tzinfo is None or not math.isfinite(max_age_seconds) or max_age_seconds <= 0: raise ValueError()
        if interval_start is not None:
            start_boundary = datetime.fromisoformat(interval_start.replace('Z', '+00:00'))
            if start_boundary.tzinfo is None or start_boundary > now: raise ValueError('invalid decision interval')
            for o in observations:
                if o.get('account_pseudonym') != account: continue
                for name in ('request_time', 'receipt_time'):
                    if name in o and start_boundary <= datetime.fromisoformat(o[name]) <= now:
                        raise ValueError('account changed or refreshed within decision interval')
        latest = {}
        requests = {}; replies = {}
        for o in observations:
            if o.get('account_pseudonym') != account: continue
            rid = o.get('request_id')
            if not isinstance(rid, str) or not rid: raise ValueError('missing request identity')
            if o.get('kind') == 'account_request':
                if rid in requests: raise ValueError('duplicate request start')
                requests[rid] = o
            elif o.get('kind') == 'account_boundary':
                if rid in replies or rid not in requests: raise ValueError('orphan or duplicate account reply')
                request = requests[rid]
                if any(o.get(k) != request.get(k) for k in ('endpoint', 'account_pseudonym', 'contract', 'request_time')):
                    raise ValueError('account request/reply mismatch')
                replies[rid] = o
        if any(datetime.fromisoformat(o['request_time']) <= now and rid not in replies for rid, o in requests.items()): raise ValueError('missing account reply')
        for o in observations:
            if o['account_pseudonym'] != account or o.get('kind') == 'account_request': continue
            start = datetime.fromisoformat(o['request_time']); end = datetime.fromisoformat(o['receipt_time'])
            if start.tzinfo is None or end.tzinfo is None or start > end: raise ValueError()
            # A request in flight at the decision prevents use of an older reply.
            if start <= now < end: raise ValueError('request pending')
            if end <= now and (o['endpoint'] not in latest or end >= latest[o['endpoint']][0]): latest[o['endpoint']] = (end, o)
        if set(latest) != set(ENDPOINTS): raise ValueError('missing account replies')
        for end, o in latest.values():
            if (now-end).total_seconds() > max_age_seconds or o.get('status_code') != 200 or o.get('success') is not True or o.get('complete') is not True: raise ValueError('failed, partial or stale reply')
            if o.get('contract') != contract: raise ValueError('contract mismatch')
        versions = {o.get('snapshot_id') for _, o in latest.values()}
        if len(versions) != 1 or any(type(v) not in (str, int) or not v or (isinstance(v, str) and not v.strip()) for v in versions): raise ValueError('non-atomic account snapshot')
        accounts = latest['/Account/search'][1]['records']
        if len(accounts) != 1 or type(accounts[0].get('balance')) not in (int, float) or not math.isfinite(accounts[0]['balance']): raise ValueError('balance unavailable')
        positions = latest['/Position/searchOpen'][1]['records']; orders = latest['/Order/searchOpen'][1]['records']
        for records in (positions, orders):
            for r in records:
                if r.get('contractId') != contract or type(r.get('size')) is not int or r['size'] <= 0: raise ValueError('ambiguous contract or quantity')
        if len({r['type'] for r in positions}) > 1 or any(r.get('type') not in (1, 2) for r in positions): raise ValueError('contradictory positions')
        result.update(verdict='PASS', state='ACTIVE' if positions else ('PENDING' if orders else 'FLAT'), orders=orders, balance=accounts[0]['balance'], quantity=sum(r['size'] * (1 if r['type'] == 1 else -1) for r in positions))
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result['reasons'].append(str(exc) or 'malformed account evidence')
    return result


def install_account_observer(client, capture, account_id, contract, salt):
    """Observe only requests the existing client makes. Keep the original response."""
    if not capture.enabled: return lambda: None
    original = client._http
    import uuid
    account = pseudonym(account_id, salt)
    class HTTP:
        def __getattr__(self, name): return getattr(original, name)
        async def post(self, url, *args, **kwargs):
            endpoint = next((p for p in ENDPOINTS if str(url).endswith(p)), None)
            if endpoint is None: return await original.post(url, *args, **kwargs)
            row = dict(request_id=uuid.uuid4().hex, kind='account_boundary', schema_version=1, endpoint=endpoint,
                       account_pseudonym=account, contract=contract, request_time=datetime.now(timezone.utc).isoformat(),
                       status_code=None, success=None, complete=False, records=[], redacted_fields=['credentials', 'account identifiers', 'names', 'order identifiers'])
            try:
                capture.emit({k: v for k, v in dict(row, kind='account_request').items() if k in ('kind', 'schema_version', 'request_id', 'endpoint', 'account_pseudonym', 'contract', 'request_time')})
            except Exception: capture.invalidate('account_request_capture_failure')
            try:
                response = await original.post(url, *args, **kwargs)
                row['status_code'] = response.status_code
                try:
                    data = response.json(); field = ENDPOINTS[endpoint]; records = data.get(field)
                    row['success'] = data.get('success')
                    row['snapshot_id'] = data.get('snapshotId')
                    if type(records) is list:
                        if len(records) > capture.max_nodes // 20: raise ValueError('account response bound')
                        requested = kwargs.get('json', {}).get('accountId')
                        row['complete'] = endpoint == '/Account/search' or str(requested) == str(account_id)
                        if field == 'accounts': records = [r for r in records if str(r.get('id')) == str(account_id)]
                        row['records'] = [{k: r[k] for k in FIELDS[field] if k in r} for r in records]
                        # Unknown pagination/truncation cannot establish complete state.
                        if any(data.get(k) for k in ('nextToken', 'nextPage', 'hasMore', 'truncated')): row['complete'] = False
                except Exception: row['complete'] = False
                return response
            finally:
                row['receipt_time'] = datetime.now(timezone.utc).isoformat()
                try: capture.emit(row)
                except Exception: capture.invalidate('account_capture_failure')
    client._http = HTTP()
    def rollback(): client._http = original
    return rollback
