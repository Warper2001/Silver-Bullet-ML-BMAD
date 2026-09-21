"""Rebuild the 2025 dollar-bar CSV front-month-only, with identical construction.

Purpose: measure GAP-1's sensitivity to the roll-week splices without changing anything
else. Earlier attempts dropped whole sessions, which also shifts the next session's
prior-close reference, and rebuilding from raw minute bars would swap the bar construction
too. Here the SAME dollar-bar writer that produced the frozen CSV is re-run on a
front-month-filtered version of the same pinned raw extract, so only contract identity
differs.

Inputs, both hash-pinned:
  - the raw 2025 extract used by the provenance work (provenance-raw-2025.jsonl)
  - the historical writer source retained at docs/reports/yank-provenance-closure/

Gate: re-running the writer on the UNFILTERED extract must reproduce the frozen CSV
byte-for-byte (sha 3f20ec70…). If it does not, nothing below is interpreted.

Front-month rule: within each Globex session (18:00 ET opens the next day's session), keep
only the contract holding the most minutes; drop the other contract's minutes. Every
session is kept, so the prior-close chain stays intact.

Writes only inside a scratch directory and this folder. Never writes to the live CSV.

Run: .venv/bin/python _bmad-output/diagnostics_gap_fade_splice_20260916/rebuild_2025_frontmonth.py
"""
from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import os
import tempfile
import types
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
MAIN = Path("/root/Silver-Bullet-ML-BMAD")
EXTRACT = (MAIN / "_bmad-output/planning-artifacts/research"
           / "technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/imports/provenance-raw-2025.jsonl")
WRITER = MAIN / "docs/reports/yank-provenance-closure/historical-writer.py.txt"
EXTRACT_SHA = "baeb1a060250c6c6071fe658459123604c85595df0688d0af278ecb3b6fa1b40"
WRITER_SHA = "8f603a91acce9dfcec199b73e5f1ecb3f2d19904bbafeb0eb0bb018523c8f27e"
FROZEN_CSV_SHA = "3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822"
OUT_REL = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
ET = ZoneInfo("America/New_York")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def session_key(ts: str):
    t = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(ET)
    return (t.replace(tzinfo=None) + timedelta(hours=6)).date()


def run_writer(bars: list[dict], out_dir: Path) -> bytes:
    """Execute the retained writer over `bars`, with its json.load and open() intercepted."""
    source = WRITER.read_bytes()
    ns = {"__name__": "splice_candidate"}
    exec(compile(source, str(WRITER), "exec"), ns)
    ns["json"] = types.SimpleNamespace(load=lambda stream: bars)

    def isolated_open(path, *args, **kwargs):
        if str(path) == "/root/mnq_historical.json":
            return io.StringIO("extract injected; raw file never decoded")
        return builtins.open(path, *args, **kwargs)

    ns["open"] = isolated_open
    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            code = ns["main"]()
        if code != 0:
            raise SystemExit(f"writer returned {code}")
        return (out_dir / OUT_REL).read_bytes()
    finally:
        os.chdir(cwd)


def main() -> int:
    if sha_file(EXTRACT) != EXTRACT_SHA:
        raise SystemExit("extract hash mismatch")
    if sha_file(WRITER) != WRITER_SHA:
        raise SystemExit("writer hash mismatch")
    bars = [json.loads(line)["bar"] for line in EXTRACT.open()]
    if not all(b["TimeStamp"].startswith("2025-") for b in bars):
        raise SystemExit("extract is not 2025-only")

    # front-month per Globex session
    per_session: dict = defaultdict(Counter)
    for b in bars:
        per_session[session_key(b["TimeStamp"])][b["Contract"]] += 1
    front = {d: c.most_common(1)[0][0] for d, c in per_session.items()}
    mixed = {str(d): dict(c) for d, c in per_session.items() if len(c) > 1}
    kept = [b for b in bars if b["Contract"] == front[session_key(b["TimeStamp"])]]

    res = {"extract_records": len(bars), "kept_records": len(kept),
           "dropped_records": len(bars) - len(kept),
           "mixed_sessions": len(mixed), "mixed_session_detail": mixed,
           "contract_switches_between_sessions": sorted(
               str(d) for i, d in enumerate(sorted(front))
               if i and front[d] != front[sorted(front)[i - 1]])}

    with tempfile.TemporaryDirectory(prefix="gapfade-splice-") as tmp:
        base = Path(tmp)
        (base / "orig").mkdir()
        (base / "corr").mkdir()
        orig_bytes = run_writer(bars, base / "orig")
        res["gate_reproduces_frozen_csv"] = hashlib.sha256(orig_bytes).hexdigest() == FROZEN_CSV_SHA
        res["orig_sha256"] = hashlib.sha256(orig_bytes).hexdigest()
        if not res["gate_reproduces_frozen_csv"]:
            (HERE / "rebuild_meta.json").write_text(json.dumps(res, indent=2, default=str))
            raise SystemExit("GATE FAILED: writer did not reproduce the frozen CSV; nothing interpreted")
        corr_bytes = run_writer(kept, base / "corr")
        res["corrected_sha256"] = hashlib.sha256(corr_bytes).hexdigest()
        (HERE / "mnq_1min_2025_frontmonth.csv").write_bytes(corr_bytes)

    res["orig_rows"] = orig_bytes.count(b"\n") - 1
    res["corrected_rows"] = corr_bytes.count(b"\n") - 1
    (HERE / "rebuild_meta.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "mixed_session_detail"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
