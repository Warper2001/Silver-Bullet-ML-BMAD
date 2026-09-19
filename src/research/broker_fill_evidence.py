"""Pure exact-match validation and durable broker fill evidence. No trading calls."""

import hashlib
import json
from typing import Any
import math
import os
from pathlib import Path


def matching_fills(
    trades: list[dict[str, Any]], identity: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected, issues, seen = [], [], {}
    for trade in trades:
        if any(
            str(trade.get(k)) != str(identity[k])
            for k in ("accountId", "contractId", "orderId")
        ):
            continue
        key = str(trade.get("id"))
        if key in seen:
            if seen[key] != trade:
                issues.append({"reason": "conflicting_duplicate", "fill_id": key})
            continue
        seen[key] = trade
        try:
            valid = (
                str(trade.get("id", "")).isdigit()
                and int(trade["id"]) > 0
                and trade.get("voided") is False
                and type(trade.get("side")) is int
                and trade["side"] == identity["side"]
                and not isinstance(trade.get("size"), bool)
                and not isinstance(trade.get("price"), bool)
                and float(trade["size"]).is_integer()
                and 0 < float(trade["size"]) <= identity["size"]
                and math.isfinite(float(trade["price"]))
                and float(trade["price"]) > 0
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            issues.append({"reason": "invalid_or_void_fill", "fill_id": key})
        else:
            selected.append(trade)
    if sum(float(t["size"]) for t in selected) > identity["size"]:
        issues.append({"reason": "quantity_exceeds_submission"})
    # Contradictions invalidate this batch; raw response remains diagnostic evidence.
    return ([] if issues else selected), issues


def persist_fill(path: Path, identity: dict[str, Any], trade: dict[str, Any]) -> str:
    """Exclusive, fsynced evidence file is the persistent dedup authority."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{trade['accountId']}:{trade['id']}".encode()).hexdigest()
    target = path / (key + ".json")
    data = {"submission": identity, "fill": trade}
    encoded = json.dumps(data, sort_keys=True, allow_nan=False) + "\n"
    import tempfile

    fd, name = tempfile.mkstemp(dir=path, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, target)
        except FileExistsError:
            return "duplicate" if json.loads(target.read_text()) == data else "conflict"
        dfd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        return "new"
    finally:
        os.unlink(name)
