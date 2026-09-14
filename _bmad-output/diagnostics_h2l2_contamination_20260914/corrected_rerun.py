"""C1a / C1b corrected re-runs of the H2/L2 and wedge/MM power gates (correction_plan.md).

Only the bar loader is replaced; both gates' committed main() runs unchanged. Outputs go to
C1a/ and C1b/ in this folder; the original gate folders are not written.

Guard: before anything runs, the C1a loader with both corrections switched off must
reproduce the original gate's bars exactly, or the script aborts.

Run:  .venv/bin/python _bmad-output/diagnostics_h2l2_contamination_20260914/corrected_rerun.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contamination_check import MAIN, atr, h2, wm  # noqa: E402  (loads the gates, verifies CSV hashes)

HERE = Path(__file__).resolve().parent
PLAN = HERE / "correction_plan.md"
PLAN_SHA = "18a16b33742a69b6314d06793dfac178f3774757646284d658071d5faa526302"
NY = atr.NY
ORIGINAL_LOADER = h2.load_5min
h2.REPO = MAIN          # relative-path printing of INPUTS only
wm.REPO = MAIN
CSV_2026_SEGMENT = "MNQM26-csv"
DRY = "--dry" in sys.argv          # guard + loader stats only; no gate, no power


def rth_window(m: pd.DataFrame) -> pd.DataFrame:
    """The gate's own window and RTH filter (h2.load_5min)."""
    m = m[(m.index >= pd.Timestamp(h2.START, tz=NY)) & (m.index < h2.CUTOFF)]
    t = m.index.hour * 60 + m.index.minute
    return m[(t > 9 * 60 + 30) & (t <= 16 * 60)]


RAW = atr.parse_raw()
RAW_W = rth_window(RAW)
_ncon = RAW_W.groupby(RAW_W.index.date)["contract"].nunique()
INTERLEAVED = set(_ncon[_ncon > 1].index)
SESSION_CONTRACT = RAW_W.groupby(RAW_W.index.date)["contract"].first()


def to_5min(m: pd.DataFrame, seg_of: dict | None) -> pd.DataFrame:
    """h2.load_5min's resampling, EMA and slots; EMA restarted per contract segment if seg_of is given."""
    b = m.resample("5min", closed="right", label="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    tb = b.index.hour * 60 + b.index.minute
    b = b[(tb >= 9 * 60 + 35) & (tb <= 16 * 60)].copy()
    assert b.index.max() < h2.CUTOFF, "post-cutoff bar survived"
    if seg_of is None:
        b["ema"] = b["close"].ewm(span=h2.EMA_SPAN, adjust=False).mean()
    else:
        seg = pd.Series(b.index.date, index=b.index).map(seg_of)
        assert seg.notna().all(), "session without a contract segment"
        b["ema"] = b.groupby(seg.to_numpy())["close"].transform(
            lambda s: s.ewm(span=h2.EMA_SPAN, adjust=False).mean())
    b["session"] = b.index.date
    b["slot"] = ((b.index.hour * 60 + b.index.minute) - (9 * 60 + 35)) // 5
    return b


def csv_minutes() -> pd.DataFrame:
    frames = []
    for p in h2.INPUTS:
        d = pd.read_csv(p)
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True, format="ISO8601").dt.tz_convert(NY)
        frames.append(d)
    return rth_window(pd.concat(frames).drop_duplicates("timestamp").set_index("timestamp").sort_index())


def load_c1a(exclude: bool = True, restart: bool = True) -> pd.DataFrame:
    m = csv_minutes()
    if exclude:
        m = m[~pd.Index(m.index.date).isin(list(INTERLEAVED))]
    seg_of = None
    if restart:
        seg_of = {d: (SESSION_CONTRACT[d] if d.year == 2025 else CSV_2026_SEGMENT) for d in set(m.index.date)}
    return to_5min(m, seg_of)


def load_c1b() -> pd.DataFrame:
    m = RAW_W[~pd.Index(RAW_W.index.date).isin(list(INTERLEAVED))][["open", "high", "low", "close"]]
    return to_5min(m, SESSION_CONTRACT.to_dict())


def main() -> int:
    if atr.sha(PLAN) != PLAN_SHA:
        sys.exit(f"PLAN HASH MISMATCH: {atr.sha(PLAN)} != pinned {PLAN_SHA}")
    orig = ORIGINAL_LOADER()
    replica = load_c1a(exclude=False, restart=False)
    pd.testing.assert_frame_equal(orig[replica.columns], replica, check_freq=False)
    print("guard: loader replica reproduces the original gate's bars exactly", flush=True)

    meta = {"plan_sha256": PLAN_SHA, "script_sha256": atr.sha(Path(__file__)),
            "raw_sha256": atr.sha(atr.RAW), "interleaved_sessions": sorted(str(d) for d in INTERLEAVED),
            "loaders": {}}
    for name, loader in (("C1a", load_c1a), ("C1b", load_c1b)):
        b = loader()
        meta["loaders"][name] = {"sessions": int(b["session"].nunique()), "bars": int(len(b)),
                                 "first": str(b.index.min()), "last": str(b.index.max())}
        if name == "C1b":
            both = orig.join(b[["close"]], rsuffix="_c1b", how="inner")
            d = (both["close"] - both["close_c1b"]).abs()
            m25 = both.index.year == 2025
            meta["loaders"][name]["vs_original_close_abs_diff_pts"] = {
                "2025_median": float(d[m25].median()), "2025_share_exact": float((d[m25] == 0).mean()),
                "2026_median": float(d[~m25].median())}
        print(name, meta["loaders"][name], flush=True)
        if DRY:
            continue
        h2.load_5min = loader
        for mod, sub in ((h2, "h2l2"), (wm, "wedge_mm")):
            mod.HERE = HERE / name / sub
            mod.HERE.mkdir(parents=True, exist_ok=True)
            print(f"\n===== {name} {sub}", flush=True)
            mod.main()
    h2.load_5min = ORIGINAL_LOADER
    if DRY:
        print("dry run: loaders checked, no gate executed")
        return 0
    (HERE / "corrected_rerun_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    summary = {}
    for name in ("C1a", "C1b"):
        summary[name] = {
            "H2L2": json.loads((HERE / name / "h2l2/results.json").read_text())["verdict"],
            "wedge_mm": json.loads((HERE / name / "wedge_mm/results.json").read_text())["VERDICT"]}
    print("\nSUMMARY", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
