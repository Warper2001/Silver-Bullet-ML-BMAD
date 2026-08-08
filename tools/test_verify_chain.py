#!/usr/bin/env python3
"""Tests for tools/verify_chain.py.

The point of these tests is the uncomfortable one from ruling V9: a chain that
VERIFIES is not a chain that is COMPLETE. `test_truncated_file_still_verifies`
pins that property deliberately — if it ever starts failing, someone has confused
tamper evidence for completeness, which is the mistake that produced a withdrawn
PF of 1.595.

Run:   .venv/bin/python -m pytest tools/test_verify_chain.py -v
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "tools"))

vc = importlib.import_module("verify_chain")

FIELDS = ["date_et", "pnl_usd"]


def write_chained(path: Path, rows, corrupt_at=None):
    """Write a correctly chained file; optionally corrupt one row's chain value."""
    head = vc.GENESIS
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["chain"])
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            payload = "|".join(str(r.get(k, "")) for k in FIELDS)
            head = hashlib.sha256((head + "|" + payload).encode()).hexdigest()[:16]
            out = dict(r)
            out["chain"] = "0" * 16 if i == corrupt_at else head
            w.writerow(out)


ROWS = [{"date_et": f"2026-08-{d:02d}", "pnl_usd": v}
        for d, v in [(3, 100), (4, -50), (5, 200), (6, 646)]]


def test_intact_chain_verifies(tmp_path):
    p = tmp_path / "t.csv"
    write_chained(p, ROWS)
    n, bad, key, err = vc.verify(p)
    assert (n, bad, err) == (4, None, None)


def test_edited_row_is_caught(tmp_path):
    p = tmp_path / "t.csv"
    write_chained(p, ROWS, corrupt_at=3)
    n, bad, key, err = vc.verify(p)
    assert bad == 3 and key == "2026-08-05"


def test_only_the_first_break_is_reported(tmp_path):
    """Resync after a mismatch — one break must not report as three."""
    p = tmp_path / "t.csv"
    write_chained(p, ROWS, corrupt_at=2)
    n, bad, key, err = vc.verify(p)
    assert bad == 2, "the walk must resync and report the first break only"


def test_truncated_file_still_verifies(tmp_path):
    """THE V9 PROPERTY. A prefix of an append-only chain is a valid chain.

    This is why trades.csv verified while missing 2026-08-06. Completeness is NOT
    a chain property and must never be inferred from a PASS.
    """
    p = tmp_path / "t.csv"
    write_chained(p, ROWS[:2])          # the 2026-08-06 row is simply gone
    n, bad, key, err = vc.verify(p)
    assert n == 2 and bad is None, "truncation is invisible to the chain — by design"


def test_reordered_rows_are_caught(tmp_path):
    p = tmp_path / "t.csv"
    write_chained(p, [ROWS[1], ROWS[0], ROWS[2], ROWS[3]])
    # re-chained in the new order, so instead assert the ORIGINAL order now fails
    write_chained(p, ROWS)
    good = p.read_text().splitlines()
    swapped = [good[0], good[2], good[1]] + good[3:]
    p.write_text("\n".join(swapped) + "\n")
    n, bad, key, err = vc.verify(p)
    assert bad is not None, "reordering must break the chain"


def test_header_only_file_reports_no_rows(tmp_path):
    p = tmp_path / "t.csv"
    write_chained(p, [])
    n, bad, key, err = vc.verify(p)
    assert (n, bad, err) == (0, None, None)


def test_empty_file_is_reported_not_crashed(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("")
    n, bad, key, err = vc.verify(p)
    assert err is not None and "empty" in err


def test_missing_file_is_unreadable_not_a_pass(tmp_path):
    n, bad, key, err = vc.verify(tmp_path / "nope.csv")
    assert err is not None


def test_exit_codes_are_ordered():
    """A broken chain must never be quieter than a clean one."""
    assert vc.OK < vc.BROKEN < vc.UNREADABLE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
