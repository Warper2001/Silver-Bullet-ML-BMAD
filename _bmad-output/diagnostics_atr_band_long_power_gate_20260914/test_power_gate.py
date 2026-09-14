"""Synthetic checks for power_gate.py. No market data.

Run: .venv/bin/python -m pytest _bmad-output/diagnostics_atr_band_long_power_gate_20260914/test_power_gate.py -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import power_gate as pg  # noqa: E402

RTH = pg.VARIANTS["RTH"]


def synth_raw(days: int = 6, contract_of=lambda d: "MNQH25", drop=None, seed: int = 0) -> pd.DataFrame:
    """1-min RTH records, close-stamped (09:31..16:00), gentle random walk.

    drop=(day, minute_index, points) forces that minute's low down by `points`.
    """
    rng = np.random.default_rng(seed)
    rows, px = [], 20000.0
    start = pd.Timestamp("2025-01-06", tz=pg.NY)          # a Monday
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        for i in range(390):
            ts = day + pd.Timedelta(hours=9, minutes=31 + i)
            o = px
            px = round((px + rng.normal(0, 1.0)) * 4) / 4
            hi, lo = max(o, px) + 0.5, min(o, px) - 0.5
            if drop and drop[0] == d and drop[1] == i:
                lo = o - drop[2]
            rows.append((ts, o, hi, lo, px, contract_of(d)))
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "contract"])
    return df.set_index("ts")


def test_planted_drop_is_detected_at_its_slot():
    # day 4, minute index 62 = the 1-min bar closing 10:33 -> 5-min bar closing 10:35 = slot 12
    b, info = pg.build(synth_raw(drop=(4, 62, 200.0)), RTH)
    ev = pg.detect(b, RTH)
    assert info["sessions_eligible"] == 6
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["session"] == 4 and e["slot"] == 12 and e["through"] and e["touch"]
    assert e["r_pts"] == pytest.approx(round(1.5 * e["atr"] / 0.25) * 0.25)
    assert e["tp_pts"] == pytest.approx(round(2.0 * e["atr"] / 0.25) * 0.25)


def test_no_event_without_drop_and_warmup_is_respected():
    b, _ = pg.build(synth_raw(drop=(1, 62, 200.0)), RTH)     # drop inside warm-up sessions
    assert len(pg.detect(b, RTH)) == 0


def test_touch_but_not_through():
    b, _ = pg.build(synth_raw(), RTH)
    ev0 = pg.detect(b, RTH)
    assert len(ev0) == 0
    # force bar t+1's low to exactly the limit of bar t
    b2 = b.copy()
    t = int(np.flatnonzero((b2["sidx"] == 4) & (b2["slot"] == 11))[0])
    lim = np.floor((b2["close"].iat[t] - 3.1 * b2["atr"].iat[t]) / 0.25) * 0.25
    b2.iloc[t + 1, b2.columns.get_loc("low")] = lim
    ev = pg.detect(b2, RTH)
    assert len(ev) == 1 and ev.iloc[0]["touch"] and not ev.iloc[0]["through"]


def test_contract_switch_resets_true_range():
    idx = pd.date_range("2025-03-10 09:35", periods=3, freq="5min", tz=pg.NY)
    b = pd.DataFrame({"open": [100, 101, 341], "high": [102, 102, 343], "low": [99, 100, 340],
                      "close": [101, 101, 342], "contract": ["MNQH25", "MNQH25", "MNQM25"]}, index=idx)
    tr = pg.add_atr(b)["tr"].to_numpy()
    assert tr[2] == 3.0          # high - low, not |high - prev close| = 242


def test_mixed_contract_session_is_ineligible():
    raw = synth_raw(contract_of=lambda d: "MNQH25")
    day2 = raw.index.date == raw.index.date[390 * 2]
    raw.loc[raw.index[day2][::7], "contract"] = "MNQM25"
    ok = pg.eligible(pg.session_minutes(raw, RTH), RTH)
    assert ok.sum() == 5 and not ok.iloc[2]


def test_identity_pairing_refused():
    b, _ = pg.build(synth_raw(drop=(4, 62, 200.0)), RTH)
    ev = pg.detect(b, RTH)
    H, L, C = pg.grids(b, RTH)
    nd = H.shape[0]
    for k in (0, nd, 2 * nd):
        with pytest.raises(AssertionError, match="FIREWALL"):
            pg.placebo(ev, H, L, C, k, RTH["flatten_slot"])
    p = pg.placebo(ev, H, L, C, 1, RTH["flatten_slot"])
    assert len(p["x"]) == 1


def test_parse_raw_skips_post_cutoff(tmp_path):
    recs = [{"High": "1", "Low": "1", "Open": "1", "Close": "1", "TimeStamp": ts, "TotalVolume": "1",
             "Contract": "MNQH26"} for ts in ("2026-02-27T21:00:00Z", "2026-03-01T23:01:00Z")]
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(recs, indent=2))
    df = pg.parse_raw(p)
    assert len(df) == 1 and df.attrs["skipped_post_cutoff"] == 1
