"""Stable artifacts and read-only input inventories."""

import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime


def input_path(path):
    path = Path(path).resolve()
    if "sealed_holdout" in path.parts:
        raise ValueError("sealed holdout access is prohibited")
    return path


def digest(path):
    h = hashlib.sha256()
    with input_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(paths):
    result = []
    for value in sorted(set(map(str, paths))):
        path = input_path(value)
        if path.exists():
            before = path.stat()
            sha = digest(path)
            after = path.stat()
            stable = (before.st_size, before.st_mtime_ns) == (
                after.st_size,
                after.st_mtime_ns,
            )
            result.append(
                dict(
                    path=str(path),
                    bytes=after.st_size,
                    sha256=sha,
                    stable_during_hash=stable,
                )
            )
        else:
            result.append(dict(path=str(path), missing=True))
    return result


def read_csv(path):
    with input_path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamps must include UTC offset")
    return dt


def number(value):
    if value is None or value == "":
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite input")
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({k for row in rows for k in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def known_epoch(value):
    """Unknown account-reset provenance must not become a synthetic epoch identity."""
    if value is None or not str(value).strip():
        return False
    label = str(value).strip().lower()
    return label not in {"none", "null"} and not any(
        token in label
        for token in (
            "unknown",
            "requires_statement",
            "unverified",
            "unconfirmed",
            "placeholder",
            "assumed",
        )
    )
