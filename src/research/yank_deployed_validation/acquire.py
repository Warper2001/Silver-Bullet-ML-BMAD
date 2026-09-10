"""Explicit, gated raw acquisition for the fixed YANK pilot (never refresh auth)."""
from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .tradestation import iso, request_ledger, timestamp, validate_archive

HOST = 'https://api.tradestation.com'
ENDPOINT = '/v3/marketdata/barcharts/MNQM25'
REQUIRED_EVIDENCE = ('contract', 'endpoint', 'zero_cost', 'entitlement', 'approval')
REQUEST_TIMEOUT_SECONDS = 60
RUN_BUDGET_SECONDS = 3600


class GateError(ValueError):
    """A safe, non-secret operational blocker."""


class Clock:
    def now(self):
        return datetime.now(timezone.utc)

    def monotonic(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(seconds)


def gate_blockers(gate, now=None):
    """Declarations reference operator-reviewed evidence; they are not proof."""
    if not isinstance(gate, dict):
        return ['gate must be an operator-supplied JSON object']
    now = now or Clock().now()
    expected = dict(symbol='MNQM25', endpoint=HOST + ENDPOINT,
                    incremental_cost_usd=0, entitlement_verified=True,
                    acquisition_authorized=True, contract_verified=True)
    blockers = []
    pin = gate.get('credential_sha256')
    if not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin):
        blockers.append('missing or invalid operator-reviewed credential_sha256')
    for key, value in expected.items():
        actual = gate.get(key)
        if actual != value or (isinstance(value, bool) and actual is not value):
            blockers.append('missing or unconfirmed ' + key)
    if type(gate.get('incremental_cost_usd')) not in (int, float):
        blockers.append('incremental_cost_usd must be numeric zero')
    references = gate.get('evidence_references', {})
    for name in REQUIRED_EVIDENCE:
        record = references.get(name) if isinstance(references, dict) else None
        if not isinstance(record, dict) or not all(
                isinstance(record.get(key), str) and record[key].strip()
                for key in ('reference', 'reviewed_by', 'reviewed_at')):
            blockers.append('missing evidence reference: ' + name)
        else:
            try:
                if timestamp(record['reviewed_at']) > now + timedelta(seconds=60):
                    blockers.append('future evidence review time: ' + name)
            except (ValueError, TypeError, OverflowError):
                blockers.append('invalid evidence review time: ' + name)
            if record.get('symbol') != 'MNQM25' or record.get('endpoint') != HOST + ENDPOINT:
                blockers.append('evidence not pinned to contract and endpoint: ' + name)
            if record.get('credential_sha256') != pin:
                blockers.append('evidence not pinned to reviewed credential: ' + name)
    return blockers


def cache_token_provider(path=None, clock=None):
    """Return a read-only provider. Never import the auto-refreshing auth client."""
    cache = Path(path) if path else Path.home() / '.tradestation/token_cache.json'
    clock = clock or Clock()

    def read():
        try:
            data = json.loads(cache.read_bytes())
            expiry = timestamp(data['expires_at'])
            token = data['access_token']
            if not isinstance(token, str) or not token or '\n' in token or '\r' in token:
                raise ValueError()
            if (expiry - clock.now()).total_seconds() <= 60:
                raise GateError('cached token expired or within 60 seconds of expiry; refresh prohibited')
            return token
        except GateError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            raise GateError('cached token missing or malformed; refresh prohibited') from None
    return read


def http_get(request, token):
    """Only the official host, fixed path and ledger query can reach the network."""
    import httpx
    async def fetch():
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(follow_redirects=False, timeout=60, trust_env=False) as client:
                # get() consumes the body before returning; cancellation and
                # async context cleanup finish before asyncio.run returns.
                return await client.get(HOST + ENDPOINT, params=request['query'],
                                        headers={'Authorization': 'Bearer ' + token})
    return asyncio.run(fetch())


def file_token_provider(path, *, expires_at=None, clock=None):
    """Read a raw token without writes; JWT exp is a refusal hint, not proof.

    Opaque tokens require an explicit independently established expiry. A JWT's
    exp is never overridden by a later operator timestamp or signature-verified.
    """
    clock = clock or Clock()

    def read():
        try:
            token = Path(path).read_text().strip()
            if not token or any(character.isspace() for character in token):
                raise ValueError()
            parts = token.split('.')
            if len(parts) == 3:
                claims = json.loads(base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4)))
                expiry = claims['exp']
                if type(expiry) not in (int, float) or not math.isfinite(expiry):
                    raise ValueError()
                expiry = datetime.fromtimestamp(expiry, timezone.utc)
                if expires_at:
                    expiry = min(expiry, timestamp(expires_at))
            else:
                expiry = timestamp(expires_at)
            if (expiry - clock.now()).total_seconds() <= 60:
                raise ValueError()
            return token
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            raise GateError('raw token missing, malformed, expired or expiry unknown; refresh prohibited') from None
    return read


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(directory, name, value):
    """No existing evidence is replaced, including after an interrupted write."""
    final = directory / name
    temporary = directory / (name + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, final)
    temporary.unlink()
    _sync_directory(directory)


def _retry_after(value, now):
    if not value:
        return 60.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value) - now).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 60.0
    return max(60.0, seconds) if not math.isnan(seconds) else 60.0


_SECRET_KEYS = re.compile(r'(^token$|id.?token|session.?token|access.?token|refresh.?token|authorization|password|secret|cookie|api.?key)', re.I)
_SECRET_TEXT = re.compile(rb'(?i)(eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|bearer\s+\S+|\b(?:token|access[_-]?token|refresh[_-]?token|authorization|password|client[_-]?secret|api[_-]?key)\s*["\x27]?\s*[:=])')


def _sensitive(raw, parsed, token):
    if token.encode() in raw or _SECRET_TEXT.search(raw):
        return True
    if isinstance(parsed, dict):
        return any(_SECRET_KEYS.search(str(key)) or _sensitive(b'', value, token)
                   for key, value in parsed.items())
    if isinstance(parsed, list):
        return any(_sensitive(b'', value, token) for value in parsed)
    return isinstance(parsed, str) and (token in parsed or bool(_SECRET_TEXT.search(parsed.encode(errors='surrogatepass'))))


def _response_issue(parsed, headers):
    # Never follow provider URLs or silently discard pages. Only a Bars object
    # with no extra provider control fields qualifies for this narrow collector.
    if any(key.lower() in ('link', 'content-range') for key in headers):
        return 'pagination_or_partial_response'
    if not isinstance(parsed, dict) or not isinstance(parsed.get('Bars'), list):
        return 'malformed_response'
    if set(parsed) != {'Bars'}:
        return 'unknown_response_fields_or_pagination'
    if not parsed['Bars']:
        return 'empty_response_completeness_unverified'
    for bar in parsed['Bars']:
        if not isinstance(bar, dict) or not all(k in bar for k in (
                'TimeStamp', 'Open', 'High', 'Low', 'Close', 'TotalVolume')):
            return 'malformed_bar'
    return None


def acquire(gate, output_directory, *, transport=http_get, clock=None, token_provider=None):
    """Collect at most 27 attempts, with durable diagnostics and no auth refresh.

    Transport returns an object with status_code, content (bytes), headers.
    Clock exposes now(), monotonic(), sleep(seconds). All are injectable offline.
    """
    clock = clock or Clock()
    blockers = gate_blockers(gate, clock.now())
    if blockers:
        return dict(status='BLOCKED', decision='HOLD_VALIDATION', blockers=blockers)
    token_provider = token_provider or cache_token_provider(clock=clock)
    out = Path(output_directory)
    if out.exists() or out.is_symlink():
        raise GateError('fresh output directory required')
    out.mkdir(parents=True, exist_ok=False)
    # Sync every newly reachable directory entry, including nested parents.
    for directory in (out, *out.parents):
        _sync_directory(directory)
    # Do not archive free-form declarations, which may inadvertently contain
    # credentials. The caller retains the original gate; hash pins that input.
    _atomic(out, 'plan.json', dict(ledger=request_ledger(),
            operator_supplied_evidence=True, provider_proof=False,
            gate_sha256=hashlib.sha256(json.dumps(gate, sort_keys=True).encode()).hexdigest()))
    envelopes = []
    next_allowed = clock.monotonic()
    deadline = next_allowed + RUN_BUDGET_SECONDS
    deadline_at = iso(clock.now() + timedelta(seconds=RUN_BUDGET_SECONDS))
    status, reason = 'INCOMPLETE', 'interrupted'
    try:
        for request in request_ledger()['requests']:
            for attempt in range(3):
                if max(next_allowed, clock.monotonic()) + REQUEST_TIMEOUT_SECONDS > deadline:
                    reason = 'run_budget_exhausted; required delay or request exceeds remaining budget'
                    return dict(status='INCOMPLETE', decision='HOLD_VALIDATION', blockers=[reason])
                while clock.monotonic() < next_allowed:
                    clock.sleep(min(60, next_allowed - clock.monotonic()))
                try:
                    token = token_provider()
                except Exception:
                    status = 'BLOCKED'
                    reason = 'token_provider_unavailable_or_expired; no refresh attempted'
                    return dict(status='BLOCKED', decision='HOLD_VALIDATION', blockers=[reason])
                if not isinstance(token, str) or not token or '\n' in token or '\r' in token:
                    status = 'BLOCKED'
                    reason = 'invalid_token_provider_result'
                    return dict(status='BLOCKED', decision='HOLD_VALIDATION', blockers=[reason])
                if hashlib.sha256(token.encode()).hexdigest() != gate['credential_sha256']:
                    status, reason = 'BLOCKED', 'credential_changed_or_not_reviewed'
                    return dict(status=status, decision='HOLD_VALIDATION', blockers=[reason])
                prefix = f"{request['request_id']}-attempt-{attempt + 1}"
                request = dict(request, started_at=iso(clock.now()))
                _atomic(out, prefix + '-started.json', dict(request=request, attempt=attempt + 1))
                next_allowed = clock.monotonic() + 60
                retry = False
                try:
                    response = transport(request, token)
                    raw = response.content
                    received = clock.now()
                    try:
                        parsed = json.loads(raw)
                        # Python accepts nonfinite JSON numbers; archive JSON
                        # does not. Preserve bytes even when its parsed form
                        # cannot be represented, retaining a safe diagnostic.
                        try:
                            json.dumps(parsed, allow_nan=False)
                            parse_issue = None
                        except ValueError:
                            parse_issue = 'nonfinite_json_representation'
                    except (ValueError, UnicodeError):
                        parsed = None
                        parse_issue = 'invalid_json'
                    metadata = dict(received_at=iso(received), http_status=response.status_code)
                    envelope = dict(request=request, response_metadata=metadata)
                    if _sensitive(raw, parsed, token):
                        reason = 'credential_redaction_invalidates_raw_evidence'
                        envelope.update(redacted=True, raw_integrity_valid=False)
                    else:
                        if parse_issue:
                            parsed = None
                        metadata['raw_sha256'] = hashlib.sha256(raw).hexdigest()
                        metadata['parse_diagnostic'] = parse_issue
                        metadata['control_flags'] = dict(
                            pagination_header_present=any(k.lower() in ('link', 'content-range') for k in response.headers),
                            extra_response_fields=isinstance(parsed, dict) and set(parsed) != {'Bars'},
                            retry_after_present='retry-after' in {k.lower() for k in response.headers},
                            run_deadline_at=deadline_at)
                        envelope.update(raw_response_base64=base64.b64encode(raw).decode(), response=parsed)
                        if response.status_code == 200:
                            reason = _response_issue(parsed, response.headers)
                            if reason is None:
                                envelope['validation_diagnostic_counts'] = validate_archive([envelope])['counts']
                        elif response.status_code in (408, 429, 500, 502, 503, 504):
                            reason, retry = 'transient_http_failure', True
                            next_allowed = max(next_allowed, clock.monotonic() +
                                               _retry_after(response.headers.get('Retry-After'), received))
                            delay = next_allowed - clock.monotonic()
                            metadata['control_flags'].update(
                                required_delay_seconds=delay if math.isfinite(delay) else None,
                                delay_exceeds_run_budget=next_allowed + REQUEST_TIMEOUT_SECONDS > deadline)
                        else:
                            reason = 'http_rejection_or_unknown_charge; no refresh or subscription action'
                    _atomic(out, prefix + '-result.json', dict(envelope=envelope, issue=reason))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as error:
                    # Exception strings and response headers can echo credentials.
                    # Retry only known transport failures, never arbitrary bugs.
                    import httpx
                    retry = isinstance(error, (httpx.TimeoutException, httpx.NetworkError, TimeoutError))
                    reason = 'transport_timeout_or_network_failure' if retry else 'transport_failure'
                    _atomic(out, prefix + '-error.json', dict(issue=reason, received_at=iso(clock.now())))
                if reason is None:
                    envelopes.append(envelope)
                    break
                if not retry or attempt == 2:
                    return dict(status='INCOMPLETE', decision='HOLD_VALIDATION', blockers=[reason])
            else:
                raise AssertionError('retry loop did not terminate')
        _atomic(out, 'archive.json', envelopes)
        status, reason = 'COMPLETE', None
        return dict(status=status, decision='HOLD_VALIDATION', requests=len(envelopes),
                    completeness='nine raw requests only; calendar and labels unverified',
                    historical_arrival_evidence=False)
    except (KeyboardInterrupt, SystemExit):
        reason = 'interrupted'
        raise
    finally:
        _atomic(out, 'status.json', dict(status=status, reason=reason,
                completed_requests=len(envelopes), decision='HOLD_VALIDATION',
                run_deadline_at=deadline_at, total_run_budget_seconds=RUN_BUDGET_SECONDS,
                historical_arrival_evidence=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', required=True, type=Path)
    parser.add_argument('--output-directory', required=True, type=Path)
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument('--token-cache', type=Path)
    auth.add_argument('--token-file', type=Path)
    parser.add_argument('--token-expires-at', help='Required expiry for opaque raw tokens; UTC ISO timestamp')
    args = parser.parse_args(argv)
    try:
        provider = (file_token_provider(args.token_file, expires_at=args.token_expires_at)
                    if args.token_file else cache_token_provider(args.token_cache))
        result = acquire(json.loads(args.gate.read_bytes()), args.output_directory, token_provider=provider)
    except (GateError, OSError, ValueError):
        result = dict(status='BLOCKED', decision='HOLD_VALIDATION',
                      blockers=['invalid gate, output directory, or local I/O; no refresh attempted'])
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
