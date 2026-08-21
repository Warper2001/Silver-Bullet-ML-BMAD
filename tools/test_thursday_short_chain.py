#!/usr/bin/env python3
"""Tests for thursday_short's hardened ChainedCsv.

Same fix as gap-fade got on 2026-08-12 (see tools/test_gap_fade_ledger_repair.py and
_bmad-output/ledger_incident_20260806_gap_fade.md), but the dedupe key is the delicate
part here and gets most of the attention: this bot writes TWO legs per Thursday (MBT and
MET), so a first-column guard on `thursday` would silently refuse every second leg — a
safety feature that deletes half the evidence. `test_composite_key_allows_the_second_leg`
is the regression test for exactly that mistake.

Run:   .venv/bin/python -m pytest tools/test_thursday_short_chain.py -v
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

ts = importlib.import_module("thursday_short")
from tools.verify_chain import verify  # noqa: E402

FIELDS = ["thursday", "symbol", "pnl_usd"]


def leg(day, sym, pnl="1.0"):
    return {"thursday": day, "symbol": sym, "pnl_usd": pnl}


def rows_of(path: Path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# ── The dedupe key ───────────────────────────────────────────────────────────

def test_composite_key_allows_the_second_leg(tmp_path):
    """MBT and MET on the same Thursday are two different rows, not a duplicate."""
    led = ts.ChainedCsv(tmp_path / "trades.csv", FIELDS,
                        key_fields=("thursday", "symbol"))
    assert led.append(leg("2026-07-02", "MBTN26")) is True
    assert led.append(leg("2026-07-02", "METN26")) is True
    assert len(rows_of(led.path)) == 2
    assert verify(led.path)[1] is None


def test_composite_key_refuses_a_true_duplicate(tmp_path):
    led = ts.ChainedCsv(tmp_path / "trades.csv", FIELDS,
                        key_fields=("thursday", "symbol"))
    assert led.append(leg("2026-07-02", "MBTN26", "1.0")) is True
    assert led.append(leg("2026-07-02", "MBTN26", "999.0")) is False
    assert len(rows_of(led.path)) == 1
    assert rows_of(led.path)[0]["pnl_usd"] == "1.0", "the original must survive"


def test_none_key_allows_repeat_rows(tmp_path):
    """decisions.csv opts out: NO_MARKS/REJECTED can legitimately repeat in a day."""
    led = ts.ChainedCsv(tmp_path / "decisions.csv", ["ts_utc", "thursday", "action"],
                        key_fields=None)
    for i in range(3):
        assert led.append({"ts_utc": f"t{i}", "thursday": "2026-07-02",
                           "action": "NO_MARKS"}) is True
    assert len(rows_of(led.path)) == 3
    assert verify(led.path)[1] is None


def test_key_fields_is_required():
    """No default: every construction site must state its key, or None on purpose."""
    with pytest.raises(TypeError):
        ts.ChainedCsv(Path("/tmp/x.csv"), FIELDS)


# ── The structural fix ───────────────────────────────────────────────────────

def test_external_revert_does_not_corrupt_the_chain(tmp_path):
    """The 2026-08-06 scenario: an outside writer truncates a file mid-run."""
    led = ts.ChainedCsv(tmp_path / "trades.csv", FIELDS,
                        key_fields=("thursday", "symbol"))
    for d in ("2026-07-02", "2026-07-09", "2026-07-16"):
        led.append(leg(d, "MBTN26"))

    kept = rows_of(led.path)[:1]
    with led.path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["chain"])
        w.writeheader()
        w.writerows(kept)

    led.append(leg("2026-07-23", "MBTN26"))          # bot keeps running

    n, bad, _key, err = verify(led.path)
    assert err is None
    assert n == 2, "two rows were lost, as expected"
    assert bad is None, "chain must still verify after an external revert"


def test_two_writers_produce_one_valid_chain(tmp_path):
    a = ts.ChainedCsv(tmp_path / "trades.csv", FIELDS, key_fields=("thursday", "symbol"))
    b = ts.ChainedCsv(tmp_path / "trades.csv", FIELDS, key_fields=("thursday", "symbol"))
    a.append(leg("2026-07-02", "MBTN26"))
    b.append(leg("2026-07-02", "METN26"))
    a.append(leg("2026-07-09", "MBTN26"))
    n, bad, _key, err = verify(a.path)
    assert (n, bad, err) == (3, None, None)


# ── The live ledgers ─────────────────────────────────────────────────────────

def test_live_ledger_keys_are_actually_unique():
    """The keys chosen at the construction sites must hold on the real files.

    If this fails, the guard would refuse a legitimate append the next time the bot
    runs — so it is checked against real data, not just synthetic rows.
    """
    for name, keys in (("trades.csv", ("thursday", "symbol")),
                       ("counterfactuals.csv", ("thursday", "symbol"))):
        path = BASE / "data/thursday_ts" / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        seen = [tuple(r[k] for k in keys) for r in rows_of(path)]
        assert len(seen) == len(set(seen)), f"{name}: {keys} is not unique on live data"


def test_live_ledgers_still_verify():
    for name in ("trades.csv", "decisions.csv", "counterfactuals.csv"):
        path = BASE / "data/thursday_ts" / name
        if not path.exists():
            pytest.skip(f"{name} not present")
        n, bad, _key, err = verify(path)
        assert err is None and bad is None, f"{name}: chain broken at row {bad}"
