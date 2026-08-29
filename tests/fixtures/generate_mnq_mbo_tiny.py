"""Regenerate ``tests/fixtures/mnq_mbo_tiny.dbn.zst``.

A committed, ~7 KB slice of the first ``N`` front-month MNQ MBO records from the
full GLBX MDP3 test capture (``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst``,
not committed). It lets the *non-integration* unit suite exercise the real
``databento.DBNStore.from_file`` + iteration + ``events.DbnMboSource`` path.

Run from the repo root::

    .venv/bin/python tests/fixtures/generate_mnq_mbo_tiny.py

The output is a valid ``.dbn.zst``: the source file's DBN metadata header
followed by the raw bytes of the selected records, zstd-compressed.
"""

from __future__ import annotations

from pathlib import Path

import zstandard
from databento import DBNStore, MBOMsg

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "data" / "tick" / "_test" / "glbx-mdp3-20260622.mbo.dbn.zst"
OUT = REPO_ROOT / "tests" / "fixtures" / "mnq_mbo_tiny.dbn.zst"
FRONT_MONTH_INSTRUMENT_ID = 42004800
N_RECORDS = 500


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"source capture not present: {SOURCE}")

    store = DBNStore.from_file(str(SOURCE))
    payload = bytearray(store.metadata.encode())

    kept = 0
    for record in store:
        if not isinstance(record, MBOMsg):
            continue
        if record.instrument_id != FRONT_MONTH_INSTRUMENT_ID:
            continue
        payload += bytes(record)
        kept += 1
        if kept >= N_RECORDS:
            break

    if kept < N_RECORDS:
        raise SystemExit(f"only {kept} front-month records available")

    OUT.write_bytes(zstandard.ZstdCompressor().compress(bytes(payload)))

    # Validate the round trip.
    reread = list(DBNStore.from_file(str(OUT)))
    assert len(reread) == N_RECORDS, len(reread)
    assert all(isinstance(r, MBOMsg) for r in reread)
    assert all(r.instrument_id == FRONT_MONTH_INSTRUMENT_ID for r in reread)
    print(
        f"wrote {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size} bytes, {kept} records)"
    )


if __name__ == "__main__":
    main()
