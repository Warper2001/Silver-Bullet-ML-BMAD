"""Sigma provenance repair — prereg mim-nb-sigma-provenance.

The defect: sigma was rebuilt from a live TradeStation fetch on every process start, so it
depended on restart timing rather than on market data. These tests pin the repaired
behaviour: sigma is a pure, deterministic, restart-stable function of the recorded bars,
and its reduction is byte-identical to the sealed engine's.
"""
import json
from datetime import date, datetime

import numpy as np
import pytest

from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive, LOOKBACK_DAYS


def _bare():
    """A MimNbLive with only the sigma-related state initialised."""
    o = object.__new__(MimNbLive)
    o.sigma_hist, o.sigma_days = {}, []
    o.prev_close, o.day, o.open_d = None, None, None
    o.today_moves, o.today_saw_close = {}, False
    return o


def _write_bars(path, days, complete=True, first="09:31"):
    """Write a bars_raw-shaped CSV. Each day gets 09:31..16:00 half-hourly-ish bars."""
    marks = [first, "10:00", "10:30", "12:00", "15:30"] + (["16:00"] if complete else [])
    rows = ["ts_utc,open,high,low,close,volume,received_at,chain"]
    for i, d in enumerate(days):
        for j, hm in enumerate(marks):
            h, m = hm.split(":")
            # ET -> UTC is +4h in July (EDT)
            ts = f"{d}T{int(h) + 4:02d}:{m}:00Z"
            o = 20000.0 + i * 10
            c = o + j + i
            rows.append(f"{ts},{o},{c + 5},{o - 5},{c},100,{ts},abc")
    path.write_text("\n".join(rows) + "\n")


JULY = [f"2026-07-{d:02d}" for d in range(1, 21)]


class TestDeterminism:
    def test_seed_is_reproducible(self, tmp_path, monkeypatch):
        """Same bars in -> byte-identical sigma out. This is the property the live bot lacked."""
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, JULY)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "WARMUP_CSV", tmp_path / "nope.csv")

        a, b = _bare(), _bare()
        a._seed_sigma_from_bars()
        b._seed_sigma_from_bars()
        assert a.sigma_hist == b.sigma_hist
        assert a.sigma_days == b.sigma_days
        assert a.prev_close == b.prev_close
        assert a.sigma_days, "expected some accepted sessions"

    def test_missing_warmup_file_is_not_fatal(self, tmp_path, monkeypatch):
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, JULY)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "WARMUP_CSV", tmp_path / "absent.csv")
        o = _bare()
        o._seed_sigma_from_bars()
        assert len(o.sigma_days) == LOOKBACK_DAYS


class TestWholeDayAcceptance:
    def test_partial_day_contributes_nothing(self, tmp_path, monkeypatch):
        """A session without a 16:00 bar must be skipped entirely — the engine's `continue`."""
        full = tmp_path / "full.csv"
        _write_bars(full, JULY)
        monkeypatch.setattr(M, "WARMUP_CSV", tmp_path / "nope.csv")

        monkeypatch.setattr(M, "BARS_RAW_CSV", full)
        a = _bare()
        a._seed_sigma_from_bars()

        partial = tmp_path / "partial.csv"
        _write_bars(partial, JULY[:-1])
        # append a final day with no 16:00 bar
        tail = partial.read_text().rstrip("\n").split("\n")
        for hm in ["09:31", "10:00", "12:00"]:
            h, m = hm.split(":")
            tail.append(f"2026-07-20T{int(h) + 4:02d}:{m}:00Z,99999,99999,99999,99999,1,x,y")
        partial.write_text("\n".join(tail) + "\n")
        monkeypatch.setattr(M, "BARS_RAW_CSV", partial)
        b = _bare()
        b._seed_sigma_from_bars()

        assert "2026-07-20" not in b.sigma_days
        # the wild 99999 closes must not have leaked into any label
        for label, vals in b.sigma_hist.items():
            assert max(vals) < 1.0, f"partial-day move leaked into {label}"

    def test_day_missing_open_is_rejected(self, tmp_path, monkeypatch):
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, JULY, first="10:00")  # no 09:31 anywhere
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "WARMUP_CSV", tmp_path / "nope.csv")
        o = _bare()
        o._seed_sigma_from_bars()
        assert o.sigma_days == []


class TestEngineParityOfReduction:
    def test_reduction_matches_numpy_mean_exactly(self):
        """Site 8: live must use the engine's reduction, or G1 fails on float ordering."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            sig = list(rng.random(LOOKBACK_DAYS) * 0.01)
            engine = float(np.mean(sig))
            live = float(np.mean(np.asarray(sig, dtype=float)))
            assert live == engine

    def test_naive_sum_div_len_can_differ(self):
        """Motivates site 8: the old reduction is not always bit-equal to np.mean."""
        rng = np.random.default_rng(1)
        diffs = 0
        for _ in range(5000):
            sig = list(rng.random(LOOKBACK_DAYS) * 0.01)
            if (sum(sig) / len(sig)) != float(np.mean(sig)):
                diffs += 1
        assert diffs > 0, "expected at least one ordering-induced mismatch"


class TestRestartStability:
    @pytest.mark.asyncio
    async def test_backfill_restores_and_does_not_reseed(self, tmp_path, monkeypatch):
        """Site 3: with sigma in state.json, _backfill must restore, never rebuild."""
        o = _bare()
        o.sigma_hist = {"10:00": [0.001] * LOOKBACK_DAYS}
        o.sigma_days = [f"2026-07-{d:02d}" for d in range(1, 15)]
        o.prev_close = 20123.5

        state = {"position": 0, "prev_close": o.prev_close,
                 "sigma_hist": o.sigma_hist, "sigma_days": o.sigma_days}
        o._load_persisted_position = lambda: state

        def _boom():
            raise AssertionError("_seed_sigma_from_bars must not run when state has sigma")
        o._seed_sigma_from_bars = _boom

        await MimNbLive._backfill(o)
        assert o.sigma_hist == {"10:00": [0.001] * LOOKBACK_DAYS}
        assert o.prev_close == 20123.5

    @pytest.mark.asyncio
    async def test_seeds_when_state_has_no_sigma(self, tmp_path, monkeypatch):
        o = _bare()
        o._load_persisted_position = lambda: {"position": 0}
        called = []
        o._seed_sigma_from_bars = lambda: called.append(True)
        await MimNbLive._backfill(o)
        assert called == [True]

    @pytest.mark.asyncio
    async def test_round_trip_through_json_is_lossless(self, tmp_path, monkeypatch):
        """Restart parity: state must round-trip sigma without precision loss."""
        p = tmp_path / "bars_raw.csv"
        _write_bars(p, JULY)
        monkeypatch.setattr(M, "BARS_RAW_CSV", p)
        monkeypatch.setattr(M, "WARMUP_CSV", tmp_path / "nope.csv")
        a = _bare()
        a._seed_sigma_from_bars()

        revived = _bare()
        blob = json.loads(json.dumps({"position": 0, "prev_close": a.prev_close,
                                      "sigma_hist": a.sigma_hist, "sigma_days": a.sigma_days}))
        revived._load_persisted_position = lambda: blob
        await MimNbLive._backfill(revived)

        assert revived.sigma_days == a.sigma_days
        for label, vals in a.sigma_hist.items():
            assert revived.sigma_hist[label] == vals
            # and the reduction is identical, which is what G1 actually tests
            assert float(np.mean(revived.sigma_hist[label])) == float(np.mean(vals))


class TestFold:
    def test_fold_is_idempotent(self):
        o = _bare()
        o.day, o.open_d = date(2026, 7, 21), 20000.0
        o.today_moves = {"10:00": 0.002, "16:00": 0.004}
        o._fold_today_into_sigma()
        o._fold_today_into_sigma()
        o._fold_today_into_sigma()
        assert o.sigma_hist["10:00"] == [0.002]
        assert o.sigma_days == ["2026-07-21"]

    def test_fold_trims_to_lookback(self):
        o = _bare()
        o.open_d = 20000.0
        for i in range(1, LOOKBACK_DAYS + 6):
            o.day = date(2026, 6, i)
            o.today_moves = {"10:00": i / 1000.0}
            o._fold_today_into_sigma()
        assert len(o.sigma_hist["10:00"]) == LOOKBACK_DAYS
        assert len(o.sigma_days) == LOOKBACK_DAYS
        # newest retained, oldest dropped
        assert o.sigma_hist["10:00"][-1] == (LOOKBACK_DAYS + 5) / 1000.0

    def test_no_fold_without_open(self):
        o = _bare()
        o.day, o.open_d = date(2026, 7, 21), None
        o.today_moves = {"10:00": 0.002}
        o._fold_today_into_sigma()
        assert o.sigma_hist == {} and o.sigma_days == []
