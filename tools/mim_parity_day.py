#!/usr/bin/env python3
"""Per-session MIM-NB parity check — G1/G2 scoped to one day.

Why this exists alongside `mim_parity_replay.py`:

  * that tool replays the WHOLE live era (439 rows back to 2026-06-11). Those rows were
    written by pre-repair code and can never match, so its headline numbers stay red no
    matter what today's binary does. It cannot answer "is the bot on parity NOW?"
  * it also produces no engine value at the 16:00 mark (engine columns come back NaN),
    which is exactly where prereg sigma-provenance Amendment 1 applies.

This checks a single session, including 16:00, and reports the gates directly:

  G1  |live sigma - engine sigma| < 1e-9   at every mark
  G2  |live UB - engine UB| < 0.01 pt and |live LB - engine LB| < 0.01 pt

Engine semantics reproduced (study_mim_nb_catstop.py):
  * a session is ACCEPTED iff its first RTH bar is 09:31 AND a 16:00 bar exists
  * every mark of day d is evaluated against the 14 accepted sessions strictly BEFORE d,
    and against d-1's 16:00 close — the fold and the prev_close roll happen after the
    day's marks, never before
  * sigma is float(np.mean(window)); bands are
        ub = O*(1+sigma) + max(prev_close-O, 0)
        lb = O*(1-sigma) - max(O-prev_close, 0)

Usage:
    python tools/mim_parity_day.py [YYYY-MM-DD]     # default: today (ET)

Exit status: 0 all marks pass G1+G2 · 1 a gate failed · 2 nothing to compare.
"""
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytz

ET = pytz.timezone("America/New_York")
BASE = Path(__file__).resolve().parent.parent
WARMUP = BASE / "data" / "processed" / "dollar_bars" / "1_minute" / "mnq_1min_2026_ytd.csv"
BARS_RAW = BASE / "data" / "mim_nb" / "bars_raw.csv"
DECISIONS = BASE / "data" / "mim_nb" / "decisions.csv"

RTH_FIRST, RTH_LAST, LOOKBACK = "09:31", "16:00", 14
ENTRY_MARKS = {f"{h:02d}:{m}" for h in range(10, 16) for m in ("00", "30")} - {RTH_LAST}
CHECK_MARKS = sorted(ENTRY_MARKS | {RTH_LAST})

G1_TOL = 1e-9      # sigma must be EXACT — deterministic or it is not
G2_TOL = 0.01      # bands, points


def read_rth(path, ts_field):
    """{date: [(hm, open, close), ...]} for RTH bars, chronological."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get(ts_field)
            if not raw:
                continue
            try:
                ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                et = ts.astimezone(ET)
                hm = et.strftime("%H:%M")
                if not (RTH_FIRST <= hm <= RTH_LAST):
                    continue
                out.setdefault(et.date(), []).append(
                    (hm, float(row["open"]), float(row["close"])))
            except (ValueError, KeyError, TypeError):
                continue
    for d in out:
        out[d].sort(key=lambda r: r[0])
    return out


def engine_state_for(target):
    """(sigma_hist, prev_close, open_d) as the sealed engine would hold them while
    evaluating `target` — i.e. built from accepted sessions strictly before it."""
    sess = read_rth(WARMUP, "timestamp")
    sess.update(read_rth(BARS_RAW, "ts_utc"))          # live record wins on overlap
    accepted = [d for d in sorted(sess)
                if sess[d][0][0] == RTH_FIRST
                and any(hm == RTH_LAST for hm, _o, _c in sess[d])]
    prior = [d for d in accepted if d < target][-LOOKBACK:]
    if len(prior) < LOOKBACK:
        return None, None, None, prior
    hist = {}
    for d in prior:
        o = sess[d][0][1]
        for hm, _o, c in sess[d]:
            hist.setdefault(hm, []).append(abs(c / o - 1.0))
            hist[hm] = hist[hm][-LOOKBACK:]
    prev_close = next(c for hm, _o, c in reversed(sess[prior[-1]]) if hm == RTH_LAST)
    open_d = sess[target][0][1] if target in sess and sess[target][0][0] == RTH_FIRST else None
    return hist, prev_close, open_d, prior


def live_rows_for(target):
    rows = {}
    if not DECISIONS.exists():
        return rows
    with open(DECISIONS, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("ts_et", "")[:10] != str(target):
                continue
            rows[r["mark"]] = r
    return rows


def main():
    target = (datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1
              else datetime.now(ET).date())
    print(f"== MIM-NB per-session parity — {target} ==")

    live = live_rows_for(target)
    if not live:
        print(f"  no decision rows for {target} — nothing to compare")
        return 2

    hist, prev_close, open_d, prior = engine_state_for(target)
    if hist is None:
        print(f"  only {len(prior)} accepted sessions before {target} (need {LOOKBACK})")
        return 2
    if open_d is None:
        print(f"  {target} has no recorded {RTH_FIRST} bar — engine would not evaluate it")
        return 2

    print(f"  engine window: {prior[0]}..{prior[-1]} ({len(prior)} sessions)")
    print(f"  engine open_d={open_d:.2f}  prev_close={prev_close:.2f}")
    print()
    hdr = f"  {'mark':>5} {'live sigma':>11} {'engine':>11} {'|dsigma|':>10} " \
          f"{'|dUB|':>8} {'|dLB|':>8}  gate"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    worst_sigma = worst_band = 0.0
    n_checked = n_pass = 0
    skipped = []

    for mark in CHECK_MARKS:
        row = live.get(mark)
        if row is None:
            continue
        if not row.get("sigma"):                       # DEPTH_SKIP etc.
            skipped.append(f"{mark}({row.get('action', '?')})")
            continue
        window = hist.get(mark, [])
        if len(window) < LOOKBACK:
            skipped.append(f"{mark}(engine depth {len(window)}/{LOOKBACK})")
            continue

        e_sigma = float(np.mean(np.asarray(window, dtype=float)))
        e_ub = open_d * (1 + e_sigma) + max(prev_close - open_d, 0.0)
        e_lb = open_d * (1 - e_sigma) - max(open_d - prev_close, 0.0)

        d_sigma = abs(float(row["sigma"]) - e_sigma)
        d_ub = abs(float(row["ub"]) - e_ub)
        d_lb = abs(float(row["lb"]) - e_lb)

        ok = d_sigma < G1_TOL and d_ub < G2_TOL and d_lb < G2_TOL
        n_checked += 1
        n_pass += ok
        worst_sigma = max(worst_sigma, d_sigma)
        worst_band = max(worst_band, d_ub, d_lb)

        print(f"  {mark:>5} {float(row['sigma']):>11.6f} {e_sigma:>11.6f} "
              f"{d_sigma:>10.2e} {d_ub:>8.2f} {d_lb:>8.2f}  {'PASS' if ok else 'FAIL'}")

    if skipped:
        print(f"\n  not compared: {', '.join(skipped)}")

    print(f"\n  marks compared: {n_checked} · passing: {n_pass}")
    print(f"  worst |dsigma| = {worst_sigma:.3e}   (G1 needs < {G1_TOL:.0e})")
    print(f"  worst |dband|  = {worst_band:.4f} pt (G2 needs < {G2_TOL})")

    if n_checked == 0:
        print("\n  VERDICT: nothing comparable")
        return 2
    verdict = worst_sigma < G1_TOL and worst_band < G2_TOL
    print(f"\n  VERDICT: {'G1+G2 PASS' if verdict else 'G1/G2 FAIL'}")
    if not verdict:
        print("  NOTE: sessions 2026-07-29 and 2026-07-31 were contaminated by the "
              "pre-Amendment-2 fetch-anchored open_d and remain in the 14-day window "
              "until ~2026-08-20. A FAIL before then is expected, not a regression.")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
