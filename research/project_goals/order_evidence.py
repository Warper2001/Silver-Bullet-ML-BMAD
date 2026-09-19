"""Prospective YANK order identities from bytes appended under one live producer."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .scheduler import process_identity

ORDER = re.compile(r"ProjectX (entry limit|TP limit|SL stop|market close) #(\d+)(?!\d)")
MAX_BYTES = 8 * 1024 * 1024


def prospective_yank_orders(
    root: Path, output: Path, producer: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Prepare evidence and cursor. Caller publishes snapshot BEFORE committing cursor.

    On startup/rotation/producer change, begin at EOF: old accountless log bytes
    cannot inherit a current account. Poll gaps exceeding the bound remain explicit.
    """
    from .capture import immutable

    cursor_path = output / "yank-log-cursor.json"
    try:
        cursor = json.loads(cursor_path.read_text())
    except (OSError, ValueError):
        cursor = {}
    identity = producer.get("identity")
    if (
        not isinstance(identity, dict)
        or not identity.get("pid")
        or not producer.get("matches_source")
        or not producer.get("account")
        or process_identity(identity["pid"]) != identity
    ):
        return dict(status="UNKNOWN_PRODUCER", orders=[]), None
    try:
        path = root / "logs/yank_streaming_working.log"
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            file_id = [stat.st_dev, stat.st_ino]
            next_cursor = dict(
                file_identity=file_id, offset=stat.st_size, producer=producer
            )
            if (
                cursor.get("file_identity") != file_id
                or cursor.get("producer") != producer
                or not isinstance(cursor.get("offset"), int)
                or not 0 <= cursor["offset"] <= stat.st_size
            ):
                return dict(status="BASELINE_ONLY", orders=[]), next_cursor
            if stat.st_size - cursor["offset"] > MAX_BYTES:
                return dict(status="LOG_CAPTURE_GAP", orders=[]), next_cursor
            stream.seek(cursor["offset"])
            raw = stream.read(stat.st_size - cursor["offset"])
            # Do not lose a final partially written line.
            end = raw.rfind(b"\n") + 1
            raw = raw[:end]
            next_cursor["offset"] = cursor["offset"] + len(raw)
        if process_identity(identity["pid"]) != identity:
            return dict(status="PRODUCER_CHANGED", orders=[]), None
        source = dict(
            path=str(path),
            file_identity=file_id,
            start_offset=cursor["offset"],
            end_offset=next_cursor["offset"],
            producer=producer,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            text=raw.decode("utf-8", "replace"),
        )
        # Retain only matching source chunks; empty/debug-only spans are not audit evidence.
        matches = [
            (n, match)
            for n, line in enumerate(source["text"].splitlines(), 1)
            for match in [ORDER.search(line)]
            if match
        ]
        source_hash = immutable(output / "order-sources", source) if matches else None
        orders = [
            dict(
                accountId=str(producer["account"]),
                orderId=match.group(2),
                kind=match.group(1),
                source_hash=source_hash,
                line=n,
                producer=identity,
            )
            for n, match in matches
        ]
        return (
            dict(status="CAPTURED", orders=orders, source_hash=source_hash),
            next_cursor,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return dict(status="LOG_UNAVAILABLE", orders=[]), None
