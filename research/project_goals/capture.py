"""Read-only ProjectX evidence collector; no execution-client dependency."""

import asyncio
import fcntl
import hashlib
import json
from typing import Any
import os
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from .common import input_path
from .scheduler import atomic, process_identity

ENDPOINTS = {
    "Account/search": "accounts",
    "Order/search": "orders",
    "Order/searchOpen": "orders",
    "Trade/search": "trades",
    "Position/searchOpen": "positions",
}
STATE_PATHS = {"MIM": "data/mim_nb/state.json", "YANK": "logs/active_trade_state.json"}
UNITS = {"MIM": "trader-mim-nb", "YANK": "trader-yank"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical(data: Any) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(data: Any) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def immutable(directory: Path, data: Any) -> str:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = sha(data)
    target = directory / (key + ".json")
    if not target.exists():
        # Publish only a complete fsynced file; hard link is atomic and exclusive.
        import tempfile

        fd, name = tempfile.mkstemp(dir=directory, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(canonical(data) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(name, target)
            except FileExistsError:
                pass
            dfd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        finally:
            os.unlink(name)
    return key


class ReadOnlyBroker:
    def __init__(self, auth: Any, client: httpx.AsyncClient) -> None:
        self.auth, self.client = auth, client

    async def request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if endpoint not in ENDPOINTS:
            raise ValueError("endpoint not allowlisted")
        started = utcnow().isoformat()
        result = dict(endpoint=endpoint, request=payload, started_at=started)
        for attempt in range(3):
            try:
                token = await self.auth.authenticate()
                response = await self.client.post(
                    "https://api.topstepx.com/api/" + endpoint,
                    json=payload,
                    headers={"Authorization": "Bearer " + token},
                    timeout=15,
                )
                if response.status_code == 401:
                    # A cached but rejected JWT must not be retried unchanged.
                    self.auth._token = None
                    self.auth._token_expires_at = None
                response.raise_for_status()
                body = response.json()
                result.update(
                    response=body,
                    observed_at=utcnow().isoformat(),
                    attempts=attempt + 1,
                )
                if body.get("success") is not True or not isinstance(
                    body.get(ENDPOINTS[endpoint]), list
                ):
                    raise ValueError("unsuccessful_or_malformed_response")
                result["ok"] = True
                return result
            except Exception as exc:
                # Do not persist exception messages that could include authentication material.
                result.update(
                    ok=False,
                    error=type(exc).__name__,
                    observed_at=utcnow().isoformat(),
                    attempts=attempt + 1,
                )
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        return result


def local_evidence(root: Path) -> dict[str, Any]:
    result = {}
    recent_after = utcnow() - timedelta(days=9)
    for strategy, relative in STATE_PATHS.items():
        path = input_path(Path(root) / relative)
        try:
            stat = path.stat()
            raw = path.read_bytes()
            value = json.loads(raw)
            omitted = []
            if isinstance(value, dict):
                omitted = [
                    field for field in ("sigma_hist", "sigma_days") if field in value
                ]
                value = {key: item for key, item in value.items() if key not in omitted}
            result[strategy] = dict(
                path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                mtime_ns=stat.st_mtime_ns,
                state=value,
                omitted_state_fields=omitted,
                stable=(stat.st_mtime_ns == path.stat().st_mtime_ns),
                producer=producer_identity(root, strategy),
            )
        except (OSError, ValueError):
            result[strategy] = dict(path=str(path), error="missing_or_malformed")
    # Raw immutable local evidence, never account attribution from an accountless CSV.
    for label, relative in {
        "mim_orders": "data/mim_nb/orders.csv",
        "yank_log": "logs/yank_streaming_working.log",
    }.items():
        try:
            path = input_path(Path(root) / relative)
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 16384))
                raw = stream.read()
            result[label] = dict(
                path=str(path),
                sha256=hashlib.sha256(raw).hexdigest(),
                tail=raw.decode("utf-8", "replace"),
                bounded_tail=True,
            )
        except OSError:
            result[label] = {"error": "missing"}
    # MIM submissions include explicit account and contract evidence.
    result["mim_submissions"] = []
    import csv

    try:
        with input_path(Path(root) / "data/mim_nb/orders.csv").open() as stream:
            for row in csv.DictReader(stream):
                if row.get("event") == "PLACE" and row.get("outcome") == "OK":
                    try:
                        observed = datetime.fromisoformat(
                            row["ts_utc"].replace("Z", "+00:00")
                        )
                        if observed.tzinfo is None or observed < recent_after:
                            continue
                        identity = json.loads(row.get("detail", ""))
                        if (
                            isinstance(identity, dict)
                            and identity.get("accountId")
                            and identity.get("contractId")
                            and identity.get("orderId")
                        ):
                            result["mim_submissions"].append(identity)
                    except (ValueError, KeyError, TypeError):
                        pass
    except OSError:
        pass
    result["mim_fills"] = []
    for path in sorted(
        (Path(root) / "data/mim_nb/broker_fill_evidence").glob("*.json")
    ):
        try:
            if path.stat().st_mtime < recent_after.timestamp():
                continue
            result["mim_fills"].append(json.loads(path.read_text()))
        except (OSError, ValueError):
            result["mim_fills"].append({"error": "malformed"})
    return result


def producer_identity(root: Path, strategy: str) -> dict[str, Any]:
    try:
        output = subprocess.run(
            ["systemctl", "show", UNITS[strategy], "-p", "MainPID", "--value"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        pid = int(output.stdout.strip())
        identity = process_identity(pid)
        env = dict(
            item.split("=", 1)
            for item in Path(f"/proc/{pid}/environ").read_text().split("\0")
            if "=" in item
        )
        cmd = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ")
        expected = (
            "mim_nb_live.py" if strategy == "MIM" else "yank_streaming_working.py"
        )
        if identity is None:
            raise ValueError("missing process identity")
        boot_time = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        started_epoch = boot_time + int(identity["start_ticks"]) / os.sysconf(
            "SC_CLK_TCK"
        )
        return dict(
            started_at=datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
            identity=identity,
            account=env.get("PROJECTX_ACCOUNT_ID"),
            matches_source=(
                expected in cmd
                and Path(f"/proc/{pid}/cwd").resolve() == Path(root).resolve()
            ),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"error": "producer_identity_unknown"}


async def capture_once(
    broker: ReadOnlyBroker, root: Path, output: Path, account: str
) -> tuple[dict[str, Any], str]:
    now = utcnow()
    before = local_evidence(root)
    requests = []
    deadline = time.monotonic() + 50

    async def bounded(endpoint, payload):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return dict(
                endpoint=endpoint,
                request=payload,
                ok=False,
                error="cycle_budget_exhausted",
            )
        try:
            async with asyncio.timeout(remaining):
                return await broker.request(endpoint, payload)
        except TimeoutError:
            return dict(
                endpoint=endpoint, request=payload, ok=False, error="cycle_timeout"
            )

    requests.append(await bounded("Account/search", {"onlyActiveAccounts": False}))
    for endpoint in ("Order/searchOpen", "Position/searchOpen"):
        requests.append(await bounded(endpoint, {"accountId": int(account)}))
    # Query cutoff follows the balance observation, so bridges can cover it.
    history_end = utcnow()
    # Eight bounded daily windows cover weekends and prior-session corrections.
    for offset in range(8):
        end = history_end - timedelta(days=offset)
        start = end - timedelta(days=1)
        for endpoint in ("Trade/search", "Order/search"):
            requests.append(
                await bounded(
                    endpoint,
                    dict(
                        accountId=int(account),
                        startTimestamp=start.isoformat(),
                        endTimestamp=end.isoformat(),
                    ),
                )
            )
    after = local_evidence(root)
    from .order_evidence import prospective_yank_orders

    yank_orders, pending_cursor = prospective_yank_orders(
        root, Path(output), after.get("YANK", {}).get("producer", {})
    )
    after["yank_orders"] = yank_orders
    snapshot = dict(
        schema=1,
        account=str(account),
        started_at=now.isoformat(),
        observed_at=utcnow().isoformat(),
        requests=requests,
        local_before=before,
        local_after=after,
        coverage_note="Bounded eight-day query; pagination/retention limits remain explicit uncertainty.",
    )
    snapshot["local_before_hash"] = immutable(
        Path(output) / "sources", snapshot.pop("local_before")
    )
    snapshot["local_after_hash"] = immutable(
        Path(output) / "sources", snapshot.pop("local_after")
    )
    key = immutable(Path(output) / "snapshots", snapshot)
    if pending_cursor is not None:
        atomic(Path(output) / "yank-log-cursor.json", pending_cursor)
    from .daily import observation, response_rows, balance

    observation_path = Path(output) / "observation.json"
    try:
        observed = json.loads(observation_path.read_text())
    except (OSError, ValueError):
        observed = {"status": "PENDING_FLAT_EVIDENCE"}
    hydrated = dict(snapshot, local_before=before, local_after=after)
    if observed.get("account") not in (None, str(account)):
        observed = dict(
            observed,
            status="INTERRUPTED",
            reason="account_identity_transition",
            evaluated_through=snapshot["observed_at"],
        )
    elif observed.get("observed_at") and any(
        after.get(name, {}).get("producer", {}).get("account")
        not in (None, str(account))
        for name in ("MIM", "YANK")
    ):
        observed = dict(
            observed,
            status="INTERRUPTED",
            reason="producer_account_identity_transition",
            evaluated_through=snapshot["observed_at"],
        )
    elif (
        observed.get("observed_at")
        and response_rows(hydrated, "Account/search") is not None
        and balance(hydrated, account) is None
    ):
        observed = dict(
            observed,
            status="INTERRUPTED",
            reason="account_disappeared_or_invalid",
            evaluated_through=snapshot["observed_at"],
        )
    elif observed.get("status") not in ("STARTED", "INTERRUPTED"):
        previous = []
        try:
            beat = json.loads((Path(output) / "heartbeat.json").read_text())
            old_key = beat["snapshot"]
            old = json.loads(
                (Path(output) / "snapshots" / (old_key + ".json")).read_text()
            )
            for field in ("local_before", "local_after"):
                old[field] = json.loads(
                    (
                        Path(output) / "sources" / (old[field + "_hash"] + ".json")
                    ).read_text()
                )
            previous.append((old_key, old))
        except (OSError, ValueError, KeyError):
            pass
        observed = observation(previous + [(key, hydrated)], str(account))
    atomic(observation_path, observed)
    immutable(Path(output) / "observations", observed)
    atomic(
        Path(output) / "heartbeat.json",
        dict(
            status="CAPTURED" if all(r["ok"] for r in requests) else "INCOMPLETE",
            observed_at=snapshot["observed_at"],
            snapshot=key,
            account=str(account),
            identity=process_identity(os.getpid()),
        ),
    )
    return snapshot, key


def audit_health(output: Path) -> dict[str, Any]:
    try:
        beat = json.loads((Path(output) / "heartbeat.json").read_text())
        age = (utcnow() - datetime.fromisoformat(beat["observed_at"])).total_seconds()
        result = dict(
            status=beat["status"] if 0 <= age < 300 else "STALE",
            detail=f'capture age {age:.0f}s; account {beat["account"]}',
        )
        latest = Path(output) / "latest_report.json"
        observation_path = Path(output) / "observation.json"
        result["observation"] = (
            json.loads(observation_path.read_text())
            if observation_path.exists()
            else {"status": "MISSING"}
        )
        result["report"] = (
            json.loads(latest.read_text()) if latest.exists() else {"status": "MISSING"}
        )
        return result
    except (OSError, ValueError, KeyError):
        return dict(status="MISSING", detail="No readable broker audit heartbeat")


async def serve(
    root: Path, output: Path, account: str, credentials: Path, once: bool = False
) -> None:
    from src.research.projectx_auth import ProjectXAuth
    from .daily import reconcile_day, scheduled_events, refresh_corrections

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        auth = ProjectXAuth.from_file(str(credentials))
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                broker = ReadOnlyBroker(auth, client)
                while True:
                    started = time.monotonic()
                    try:
                        snapshot, _ = await capture_once(broker, root, output, account)
                        for event, day in scheduled_events(utcnow()):
                            receipt = (
                                output / "schedule" / str(account) / (event + ".json")
                            )
                            if not receipt.exists():
                                report = reconcile_day(output, account, day)
                                receipt.parent.mkdir(parents=True, exist_ok=True)
                                atomic(
                                    receipt,
                                    dict(
                                        event=event,
                                        day=day,
                                        report_status=report["status"],
                                    ),
                                )
                        refresh_corrections(output, account, snapshot)
                    except Exception as exc:
                        atomic(
                            output / "heartbeat.json",
                            dict(
                                status="FAILED",
                                observed_at=utcnow().isoformat(),
                                account=str(account),
                                error=type(exc).__name__,
                            ),
                        )
                        if once:
                            raise
                    if once:
                        return
                    await asyncio.sleep(max(1, 60 - (time.monotonic() - started)))
        finally:
            await auth.cleanup()
