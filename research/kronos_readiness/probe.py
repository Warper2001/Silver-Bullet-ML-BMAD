"""One bounded current SIM observation; never refreshes credentials or retries."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import signal
import time
from typing import Any, Callable
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from . import FLAGS
from .evidence import finish, load_json, new_output, sha, sources, write_json

HOST = "https://sim-api.tradestation.com"
SYMBOL = "MNQZ26"
METADATA = HOST + "/v3/marketdata/symbols/" + SYMBOL
BARS = (
    HOST + "/v3/marketdata/barcharts/" + SYMBOL + "?interval=1&unit=Minute&barsback=3"
)
MAX_SECONDS = 900
MAX_REQUESTS = 180
MAX_BYTES = 1_048_576


def aware(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("explicit timezone required")
    return result.astimezone(timezone.utc)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect refused")


@contextmanager
def hard_timeout(seconds: float):
    """Wall deadline also bounds DNS and trickling response bodies on Linux."""
    if seconds <= 0:
        raise TimeoutError("deadline")
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    if old_timer[0] or old_timer[1]:
        raise ValueError("existing alarm prevents safe deadline enforcement")

    def expired(signum, frame):
        raise TimeoutError("deadline")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def guarded_get(
    url: str, token: str, timeout: float, method: str = "GET"
) -> tuple[int, bytes]:
    if method != "GET" or url not in (METADATA, BARS):
        raise ValueError("endpoint or method refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token}, method="GET"
    )
    with hard_timeout(timeout):
        try:
            response = opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise ValueError("response size cap")
            return response.code, body


def sanitize(body: bytes, token: str) -> bytes:
    # Replace the actual credential before encoding or parsing even malformed bodies.
    body = body.replace(token.encode(), b"[REDACTED]")
    try:
        document = json.loads(body)
    except (ValueError, UnicodeError):
        return body

    def redact(value):
        if isinstance(value, dict):
            return {
                k: (
                    "[REDACTED]"
                    if k.lower().replace("-", "_")
                    in {
                        "authorization",
                        "access_token",
                        "refresh_token",
                        "token",
                        "cookie",
                        "set_cookie",
                    }
                    else redact(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    cleaned = redact(document)
    return body if cleaned == document else json.dumps(cleaned).encode()


def preconditions(
    plan: dict[str, Any], now: datetime, register: list[dict[str, Any]]
) -> tuple[datetime, datetime]:
    valid = {r["sha256"] for r in register if r["verification"] == "HASH_VERIFIED_ONLY"}
    session, contract = plan["session"], plan["contract"]
    if session.get("verified") is not True or session["source_sha256"] not in valid:
        raise ValueError("verified dated session evidence required")
    opening, close = aware(session["open"]), aware(session["close"])
    ny = ZoneInfo("America/New_York")
    local_open, local_close = opening.astimezone(ny), close.astimezone(ny)
    if (
        local_open.date().isoformat() != session["date"]
        or local_close.date() != local_open.date()
        or local_open.hour != 9
        or local_open.minute != 30
        or local_open.second
        or (local_close.hour, local_close.minute, local_close.second) > (16, 0, 0)
        or local_open.microsecond
        or local_close.microsecond
        or local_open.weekday() >= 5
        or close <= opening
        or now.astimezone(ny).date() != local_open.date()
    ):
        raise ValueError("invalid dated RTH boundaries")
    if (
        contract["symbol"] != SYMBOL
        or contract["source_sha256"] not in valid
        or contract.get("verified") is not True
    ):
        raise ValueError(
            "explicit verified recent successful request evidence required"
        )
    observed = aware(contract["observed_at"])
    if observed > now or observed.astimezone(ny).date() != now.astimezone(ny).date():
        raise ValueError("request evidence must be from current NY session date")
    if now < opening or now + timedelta(seconds=MAX_SECONDS) > close:
        raise ValueError("full bounded capture window unavailable")
    return opening, close


def validate_metadata(document: dict[str, Any], now: datetime) -> None:
    rows = document.get("Symbols")
    if document.get("Errors") or not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("invalid metadata")
    row = rows[0]
    if (
        any(
            row.get(k) != v
            for k, v in {
                "Symbol": SYMBOL,
                "Root": "MNQ",
                "AssetType": "FUTURE",
                "Exchange": "CME",
                "Currency": "USD",
            }.items()
        )
        or aware(row["ExpirationDate"]) <= now
    ):
        raise ValueError("metadata mismatch or expired")


def describe(observations: list[dict[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    revisions, ambiguities, gaps, delays, incomplete = [], [], [], [], []
    for observation in observations:
        if observation["url"] != BARS or "response" not in observation:
            continue
        response = observation["response"]
        if not isinstance(response, dict) or not isinstance(response.get("Bars"), list):
            ambiguities.append(
                {
                    "sequence": observation["sequence"],
                    "reason": "malformed bar response",
                }
            )
            continue
        for bar in response["Bars"]:
            if not isinstance(bar, dict):
                ambiguities.append(
                    {"sequence": observation["sequence"], "reason": "malformed bar"}
                )
                continue
            stamp = bar.get("TimeStamp")
            if not isinstance(stamp, str):
                stamp = None
            bar_status = bar.get("BarStatus")
            if bar_status in ("Open", "Closed"):
                if bar_status == "Open":
                    incomplete.append(
                        {"sequence": observation["sequence"], "timestamp": stamp}
                    )
            else:
                ambiguities.append(
                    {
                        "sequence": observation["sequence"],
                        "timestamp": stamp,
                        "reason": "unknown or missing completion status",
                    }
                )
            if not stamp or "BarStatus" not in bar:
                ambiguities.append(
                    {
                        "sequence": observation["sequence"],
                        "timestamp": stamp,
                        "reason": "missing timestamp or completion status",
                    }
                )
            try:
                when = aware(stamp)
                delays.append(
                    (aware(observation["receipt_utc"]) - when).total_seconds()
                )
            except (ValueError, TypeError, AttributeError):
                ambiguities.append(
                    {
                        "sequence": observation["sequence"],
                        "timestamp": stamp,
                        "reason": "timestamp timezone missing or invalid",
                    }
                )
            if stamp in seen and seen[stamp] != bar:
                revisions.append(
                    {"sequence": observation["sequence"], "timestamp": stamp}
                )
            if isinstance(stamp, str):
                seen[stamp] = bar
    parsed = []
    for stamp in seen:
        try:
            parsed.append(aware(stamp))
        except ValueError:
            pass
    parsed = sorted(set(parsed))
    for before, after in zip(parsed, parsed[1:]):
        if (after - before).total_seconds() > 60:
            gaps.append({"before": before.isoformat(), "after": after.isoformat()})
    return {
        "revisions": revisions,
        "ambiguities": ambiguities,
        "explicitly_open_bars": incomplete,
        "observed_timestamp_gaps": gaps,
        "receipt_minus_provider_timestamp_seconds": delays,
        "delay_interpretation": "Descriptive label-to-receipt differences, not authenticated availability latency.",
        "completion_inferred_from_age": False,
        "historical_authentication": False,
    }


def run(
    plan_path: Path,
    token_path: Path,
    output: Path,
    *,
    get: Callable = guarded_get,
    utc: Callable = lambda: datetime.now(timezone.utc),
    monotonic: Callable = time.monotonic,
    sleep: Callable = time.sleep,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    register = sources(plan, plan_path.parent)
    output = new_output(output)
    write_json(output / "plan.json", {**plan, **FLAGS})
    write_json(output / "source-verification.json", {**FLAGS, "sources": register})
    observations: list[dict[str, Any]] = []
    status, reason, bars = "PENDING", "unverified prerequisites", 0
    try:
        _, close = preconditions(plan, utc(), register)
    except (ValueError, KeyError, TypeError):
        return finish(
            output,
            {
                "status": status,
                "reason": "unverified session/contract or full RTH window unavailable",
                "token_read": False,
                "bar_requests": 0,
            },
        )
    try:
        # Existing plain token only; no live auth imports or shared writes.
        token = token_path.read_text().strip()
        if not token or any(c.isspace() for c in token):
            raise ValueError("invalid existing credential")
    except (OSError, ValueError):
        return finish(
            output,
            {
                "status": "BLOCKED",
                "reason": "existing credential unavailable or invalid",
                "token_read": True,
                "bar_requests": 0,
            },
        )
    start = monotonic()
    deadline = start + MAX_SECONDS
    last_bar_start: float | None = None

    def request(url: str) -> dict[str, Any]:
        request_mono = monotonic()
        remaining = min(deadline - request_mono, (close - utc()).total_seconds())
        if remaining <= 0:
            raise TimeoutError("deadline")
        record = {
            **FLAGS,
            "sequence": len(observations),
            "method": "GET",
            "url": url,
            "request_utc": utc().isoformat(),
            "request_monotonic": request_mono,
            "request_elapsed_seconds": request_mono - start,
        }
        try:
            code, raw = get(url, token, min(15.0, remaining))
            body = sanitize(raw, token)
            record.update(
                http_status=code,
                raw_response_base64=base64.b64encode(body).decode(),
                sanitized_body_sha256=sha(body),
                raw_redacted=(body != raw),
            )
            try:
                record["response"] = json.loads(body)
            except (ValueError, UnicodeError):
                record["parse_error"] = True
        except Exception:
            record["error"] = "request failed; no refresh or retry"
        record.update(
            receipt_utc=utc().isoformat(),
            receipt_monotonic=monotonic(),
            receipt_elapsed_seconds=monotonic() - start,
        )
        observations.append(record)
        write_json(output / f"observation-{record['sequence']:04d}.json", record)
        if (
            "error" in record
            or record.get("http_status") != 200
            or record.get("parse_error")
        ):
            raise ValueError("request failed")
        if not isinstance(record["response"], dict):
            raise ValueError("invalid response object")
        return record["response"]

    try:
        metadata = request(METADATA)
        validate_metadata(metadata, utc())
        while bars < MAX_REQUESTS:
            if last_bar_start is not None:
                wait = max(0.0, last_bar_start + 5 - monotonic())
                if (
                    monotonic() + wait >= deadline
                    or utc() + timedelta(seconds=wait) >= close
                ):
                    break
                if wait:
                    sleep(wait)
            if monotonic() >= deadline or utc() >= close:
                break
            last_bar_start = monotonic()
            bars += 1
            document = request(BARS)
            if (
                document.get("Errors")
                or not isinstance(document.get("Bars"), list)
                or any(
                    not isinstance(b, dict)
                    or ("Symbol" in b and b["Symbol"] != SYMBOL)
                    or ("Root" in b and b["Root"] != "MNQ")
                    for b in document["Bars"]
                )
                or len(document["Bars"]) > 3
            ):
                raise ValueError("invalid bar response")
        status, reason = (
            "OBSERVED_CURRENT_ONLY",
            "request/duration/session bound reached",
        )
    except Exception:
        status, reason = (
            "STOPPED",
            "request, metadata or response validation failed; no retry",
        )
    return finish(
        output,
        {
            "status": status,
            "reason": reason,
            "token_read": True,
            "bar_requests": bars,
            "elapsed_seconds": monotonic() - start,
            "description": describe(observations),
        },
    )
