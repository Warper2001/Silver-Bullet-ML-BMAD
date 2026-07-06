"""Evaluation heartbeat writer for the YANK live trader (ops instrumentation only).

Design (spec: _bmad-output/spec_yank_evaluation_heartbeat.md): the trader overwrites a
single latest-state JSON file once per main-loop iteration so the combine ops
healthcheck can verify the strategy loop is *evaluating bars*, not merely that the
process is alive — log-file mtime stays fresh even during a 401-loop, which is exactly
the silent-failure class that burned us on 2026-06-07.

Isolation-by-construction (same pattern as ts_sim_mirror): write() never raises. A
broken heartbeat must be structurally unable to take down a real-money trader; the
healthcheck treats a missing/stale file as the alarm condition, so swallowing writer
errors fails safe (alarm fires) rather than dangerous (trader dies).
"""
from __future__ import annotations

import json
import os
from pathlib import Path


class HeartbeatWriter:
    """Atomically replace *path* with the given payload each cycle (tmp + os.replace)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        # tmp lives next to the target so os.replace stays a same-filesystem atomic rename
        self._tmp = self._path.with_name(self._path.name + ".tmp")

    def write(self, payload: dict) -> bool:
        """Write the heartbeat. Returns True on success, False on any failure (never raises)."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._tmp.write_text(json.dumps(payload, indent=1, default=str))
            os.replace(self._tmp, self._path)
            return True
        except Exception:
            return False
