#!/usr/bin/env python3
"""Tests for the 2026-08-06 ledger-loss fix: hardened ChainedCsv + the repair tool.

The incident: `data/gap_fade/{trades,decisions,fills}.csv` were git-tracked, a
branch checkout on 2026-08-06 reverted them to their last commit, and two sessions
vanished. Worse than the loss, the running bot kept appending against an in-memory
chain head the file no longer contained, so the chain itself was corrupted —
decisions.csv still fails verification at row 30 and always will.

The first four tests pin the structural fix: a chain head read from disk at append
time cannot be invalidated by another writer, so this class of accident degrades to
a completeness question instead of destroying the evidence too.

Run:   .venv/bin/python -m pytest tools/test_gap_fade_ledger_repair.py -v
"""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

gf = importlib.import_module("src.research.gap_fade_live")
repair = importlib.import_module("tools.gap_fade_ledger_repair")
from tools.verify_chain import verify  # noqa: E402

FIELDS = ["date_et", "value"]


def make(tmp_path: Path, name="led.csv"):
    return gf.ChainedCsv(tmp_path / name, list(FIELDS))


def rows_of(path: Path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# ── The structural fix ───────────────────────────────────────────────────────

def test_normal_appends_verify(tmp_path):
    led = make(tmp_path)
    for d in ("2026-08-03", "2026-08-04", "2026-08-05"):
        assert led.append({"date_et": d, "value": "1"}) is True
    n, bad, _key, err = verify(led.path)
    assert (n, bad, err) == (3, None, None)


def test_external_revert_does_not_corrupt_the_chain(tmp_path):
    """The 2026-08-06 scenario, replayed.

    Rows are appended, an outside actor (git checkout) truncates the file back to
    an earlier state, and the SAME long-lived object appends again. Before the fix
    the next row chained onto a head that was no longer in the file and the chain
    broke permanently. Now the head is re-read from disk, so the chain still
    verifies — the damage is confined to the rows that were lost.
    """
    led = make(tmp_path)
    for d in ("2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"):
        led.append({"date_et": d, "value": "1"})

    kept = rows_of(led.path)[:2]                       # revert to the 08-04 state
    with led.path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["chain"])
        w.writeheader()
        w.writerows(kept)

    led.append({"date_et": "2026-08-07", "value": "1"})  # bot keeps running

    n, bad, _key, err = verify(led.path)
    assert err is None
    assert n == 3, "two rows were lost, as expected"
    assert bad is None, "chain must still verify after an external revert"


def test_duplicate_key_is_refused_and_writes_nothing(tmp_path):
    """The 2026-06-25 defect class: the same session appended twice."""
    led = make(tmp_path)
    assert led.append({"date_et": "2026-06-25", "value": "a"}) is True
    assert led.append({"date_et": "2026-06-25", "value": "b"}) is False
    assert len(rows_of(led.path)) == 1
    n, bad, _key, err = verify(led.path)
    assert (n, bad, err) == (1, None, None)


def test_deleted_file_is_recreated_and_chain_restarts_cleanly(tmp_path):
    led = make(tmp_path)
    led.append({"date_et": "2026-08-03", "value": "1"})
    led.path.unlink()
    with pytest.raises(Exception):
        rows_of(led.path)
    led2 = gf.ChainedCsv(led.path, list(FIELDS))
    led2.append({"date_et": "2026-08-04", "value": "1"})
    n, bad, _key, err = verify(led.path)
    assert (n, bad, err) == (1, None, None)


def test_head_is_not_trusted_across_instances(tmp_path):
    """Two writers on one file must still produce one valid chain."""
    a = make(tmp_path)
    b = gf.ChainedCsv(a.path, list(FIELDS))
    a.append({"date_et": "2026-08-03", "value": "1"})
    b.append({"date_et": "2026-08-04", "value": "1"})
    a.append({"date_et": "2026-08-05", "value": "1"})
    n, bad, _key, err = verify(a.path)
    assert (n, bad, err) == (3, None, None)


# ── The repair tool ──────────────────────────────────────────────────────────

def test_recovered_rows_match_the_live_schemas():
    """A recovered row with a stray or missing field would silently chain wrong."""
    for name, row, _source, _note in repair.RECOVERED:
        assert set(row) == set(repair.SCHEMA[name]), name


def test_recovered_schemas_match_the_bot():
    """SCHEMA here must stay identical to the ChainedCsv field lists in the bot."""
    src = (BASE / "src/research/gap_fade_live.py").read_text()
    for name, fields in repair.SCHEMA.items():
        for f in fields:
            assert f'"{f}"' in src, f"{name}: field {f} not found in gap_fade_live.py"


def test_trade_row_arithmetic_is_self_consistent():
    t = next(r for f, r, _s, _n in repair.RECOVERED if f == "trades.csv")
    assert t["pnl_pts"] == round(t["exit_px"] - t["entry"], 2)      # long
    assert t["pnl_usd"] == t["pnl_pts"] * gf.MNQ_PV * 1
    assert t["gap_abs_pts"] == round(abs(t["entry"] - t["target"]), 2)


def test_fill_row_arithmetic_is_self_consistent():
    f = next(r for fn, r, _s, _n in repair.RECOVERED if fn == "fills.csv")
    realized = round((f["exit_exec"] - f["entry_exec"]) * gf.MNQ_PV * f["qty"], 2)
    assert realized == f["realized_pnl_usd"]
    assert round(f["realized_pnl_usd"] - f["modeled_pnl_usd"], 2) == f["delta_usd"]


def test_cross_check_detects_a_corrupted_recovery(monkeypatch):
    """If the recovered trade ever stops matching trades.db, the tool must refuse."""
    monkeypatch.setattr(repair, "db_trade_2026_08_06",
                        lambda: [("L", 29344.5, 29667.5, 999.0, "fill", "{}")])
    bad = repair.cross_check_trade(
        next(r for f, r, _s, _n in repair.RECOVERED if f == "trades.csv"))
    assert any("pnl_usd" in m for m in bad)


def test_cross_check_passes_against_the_real_db():
    assert repair.cross_check_trade(
        next(r for f, r, _s, _n in repair.RECOVERED if f == "trades.csv")) == []


def test_repair_is_idempotent(tmp_path, monkeypatch):
    """Second run must append nothing — the tool may be re-run safely."""
    for name, fields in repair.SCHEMA.items():
        with (tmp_path / name).open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields + ["chain"]).writeheader()
    monkeypatch.setattr(repair, "DATA", tmp_path)

    for name, row, _s, _n in repair.RECOVERED:
        assert row["date_et"] not in repair.existing_keys(name)
        repair.append_row(name, row)

    for name, row, _s, _n in repair.RECOVERED:
        assert row["date_et"] in repair.existing_keys(name), "guard must now skip it"

    for name in repair.SCHEMA:
        n, bad, _key, err = verify(tmp_path / name)
        assert err is None and bad is None, f"{name} chain broke during repair"
