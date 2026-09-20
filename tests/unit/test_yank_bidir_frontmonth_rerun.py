"""Synthetic tests for tools/yank_bidir_frontmonth_rerun.py. No market data, no holdout, no engine run."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from tools import yank_bidir_frontmonth_rerun as h


def tr(direction: str, pnl: float, month: str = "2025-03", day: int = 3):
    y, m = month.split("-")
    return SimpleNamespace(
        direction=direction,
        pnl_usd=pnl,
        timestamp_entry=datetime(int(y), int(m), day, 15, tzinfo=timezone.utc),
    )


def bull(
    n: int,
    pf_win: float = 200.0,
    pf_loss: float = -100.0,
    months=("2025-03", "2025-04", "2025-05"),
):
    """n bullish trades, spread over months, 2 wins : 1 loss."""
    return [
        tr("BULLISH", pf_win if i % 3 else pf_loss, months[i % len(months)], 1 + i % 20)
        for i in range(n)
    ]


def bear(pnls):
    return [tr("BEARISH", p, "2025-06", 1 + i % 20) for i, p in enumerate(pnls)]


BASE = [tr("BEARISH", 100.0), tr("BEARISH", 100.0), tr("BEARISH", -100.0)]  # PF 2.0


def test_all_gates_pass():
    g = h.evaluate_gates(BASE, bear([100.0, 100.0, -100.0]) + bull(18))
    assert g["G1"] and g["G2"] and g["G3"] and g["G4"] and g["all_pass"]
    assert h.verdict(g) == "ALL FOUR GATES PASS on this input"


def test_g1_boundary_is_at_least_15():
    assert h.evaluate_gates(BASE, bear([100.0, 100.0, -100.0]) + bull(15))["G1"]
    assert not h.evaluate_gates(BASE, bear([100.0, 100.0, -100.0]) + bull(14))["G1"]


def test_g2_pf_must_strictly_exceed_1_3():
    # 13 wins of 100 vs 10 losses of 100 -> PF exactly 1.3: must FAIL (strict >)
    exact = [
        tr("BULLISH", 100.0, "2025-0%d" % (3 + i % 3), 1 + i) for i in range(13)
    ] + [tr("BULLISH", -100.0, "2025-0%d" % (3 + i % 3), 1 + i) for i in range(10)]
    g = h.evaluate_gates(BASE, bear([100.0, 100.0, -100.0]) + exact)
    assert g["bullish"]["pf"] == pytest.approx(1.3) and not g["G2"]
    assert h.verdict(g).startswith("H0")


def test_g3_bearish_pf_must_stay_within_ten_percent_of_baseline():
    # baseline PF 2.0. Bearish PF 2.15 (+7.5%) is inside the band; 2.3 (+15%) is outside.
    g_ok = h.evaluate_gates(BASE, bear([107.5, 107.5, -100.0]) + bull(18))
    g_bad = h.evaluate_gates(BASE, bear([115.0, 115.0, -100.0]) + bull(18))
    assert g_ok["G3"] and g_ok["G3_diff_pct"] == pytest.approx(7.5)
    assert not g_bad["G3"] and h.verdict(g_bad).startswith("NOT JUDGED")


def test_g4_month_concentration_boundary_forty_percent():
    def spread(n_mar: int, n_apr: int, n_may: int):
        return (
            [tr("BULLISH", 200.0, "2025-03", 1 + i) for i in range(n_mar)]
            + [tr("BULLISH", 200.0, "2025-04", 1 + i) for i in range(n_apr)]
            + [tr("BULLISH", 200.0, "2025-05", 1 + i) for i in range(n_may)]
        )

    at = h.evaluate_gates(
        BASE, bear([100.0, 100.0, -100.0]) + spread(6, 5, 4)
    )  # 6/15 = 40.0%: pass
    over = h.evaluate_gates(
        BASE, bear([100.0, 100.0, -100.0]) + spread(7, 4, 4)
    )  # 7/15 = 46.7%: fail
    assert at["G4"] and at["G4_worst_pct"] == pytest.approx(40.0)
    assert not over["G4"] and h.verdict(over).startswith("H1 NOT CONFIRMED")


def test_holdout_assertion_rejects_march_and_empty():
    with pytest.raises(SystemExit):
        h.assert_before_holdout(pd.Series([pd.Timestamp("2026-03-01", tz="UTC")]))
    with pytest.raises(SystemExit):
        h.assert_before_holdout(pd.Series([], dtype="datetime64[ns, UTC]"))
    h.assert_before_holdout(pd.Series([pd.Timestamp("2026-02-27 21:00", tz="UTC")]))


def test_build_corrected_refuses_foreign_contract_labels(tmp_path, monkeypatch):
    import tools.yank_frontmonth_revalidation as rv

    f = tmp_path / "f.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-06-02 14:00:00+00:00"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1],
            "notional": [1.0],
        }
    ).to_csv(f, index=False)
    ts = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(
        rv,
        "read_raw",
        lambda *a, **k: [(ts, 1, 1, 1, 1, 1, "MNQH26"), (ts, 1, 1, 1, 1, 1, "MNQM26")],
    )
    with pytest.raises(SystemExit, match="labels"):
        h.build_corrected(f, tmp_path / "raw.json", tmp_path / "o.csv")


def test_build_corrected_writes_only_pre_holdout_rows(tmp_path, monkeypatch):
    import tools.yank_frontmonth_revalidation as rv

    f = tmp_path / "f.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-06-02 14:00:00+00:00", "2025-12-31 23:00:00+00:00"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
            "notional": [1.0, 1.0],
        }
    ).to_csv(f, index=False)
    a = datetime(2026, 1, 5, 15, tzinfo=timezone.utc)
    b = datetime(2026, 2, 27, 20, tzinfo=timezone.utc)
    monkeypatch.setattr(
        rv,
        "read_raw",
        lambda *a_, **k: [(a, 2, 2, 2, 2, 3, "MNQH26"), (b, 2, 2, 2, 2, 3, "MNQH26")],
    )
    meta = h.build_corrected(f, tmp_path / "raw.json", tmp_path / "o.csv")
    out = pd.read_csv(tmp_path / "o.csv", parse_dates=["timestamp"])
    assert meta["rows_total"] == 4 and out["timestamp"].max() < pd.Timestamp(
        "2026-03-01", tz="UTC"
    )
    assert (
        out.loc[out["timestamp"] == pd.Timestamp(a), "notional"].iloc[0] == 2 * 3 * 20.0
    )


def test_one_shot_refuses_when_results_exist(tmp_path):
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(SystemExit, match="one-shot"):
        h.main(
            [
                "--preregistration",
                "abcdef1",
                "--front2025",
                str(tmp_path / "x.csv"),
                "--out-dir",
                str(tmp_path),
            ]
        )


def test_sealing_check_rejects_non_sha_and_missing_amendment(tmp_path):
    with pytest.raises(SystemExit, match="not a commit SHA"):
        h.check_sealing_commit(tmp_path, "HEAD; rm -rf", b"x")
    with pytest.raises(SystemExit):
        h.check_sealing_commit(
            tmp_path, "abcdef1", b"x"
        )  # not a git repo -> doc missing


def test_g0_check_detects_any_mismatch():
    good = {
        "baseline": {"n": 46, "pf": 1.0534},
        "bullish": {"n": 23, "pf": 1.4301, "net": 2566.75},
        "G3_diff_pct": 3.32,
        "G4_worst_month": "2026-01",
        "G4_worst_pct": 17.39,
    }
    halves = {
        "H1_2025-01..07": {"n": 10, "net": -261.25},
        "H2_2025-08..2026-02": {"n": 13, "net": 2828.0},
    }
    assert h.g0_check(good, halves)["pass"]
    bad = dict(good, bullish={"n": 22, "pf": 1.4301, "net": 2566.75})
    assert not h.g0_check(bad, halves)["pass"]


def test_dispersion_reports_mde_but_makes_no_claim():
    d = h.dispersion([tr("BULLISH", p) for p in (300.0, -150.0, 250.0, -100.0, 400.0)])
    assert d["n"] == 5 and d["mde_80pct_power_one_sided_05"] == pytest.approx(
        2.487 * d["se"]
    )
