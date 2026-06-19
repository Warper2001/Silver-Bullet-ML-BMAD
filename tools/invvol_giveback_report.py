#!/usr/bin/env python3
"""Compare giveback-from-high-water-mark: inverse-vol SIM paper-track vs the live combine.

Set 2026-06-19 (party-mode roundtable). Fires ~2026-07-21 ahead of the S25 decision to
test whether trimming YANK reduces giveback (Quinn vs Mary):
  SIM treatment  = YANK 1ct / MIM 1ct  (data/ts_sim_mirror/*_invvol_equity.csv)
  Combine control= YANK 2ct / MIM 1ct  (data/combine_joint/monitor.csv)

Self-contained (no repo imports) so it survives as a standalone scheduled reminder.
Accounts differ in size/funding, so the verdict uses giveback as % of equity (size-
normalized); raw $ is shown too. Small-sample: treat as directional, not significant.
"""
import csv
from datetime import datetime
from pathlib import Path

BASE = Path("/root/Silver-Bullet-ML-BMAD")
SIM_FILES = [BASE / "data/ts_sim_mirror/yank_invvol_equity.csv",
             BASE / "data/ts_sim_mirror/mim_invvol_equity.csv"]
COMBINE = BASE / "data/combine_joint/monitor.csv"


def _ts(s):
    return datetime.fromisoformat(s.strip().replace("Z", "+00:00"))


def max_giveback(equity):
    """Running-max minus equity, max over the path (mirrors capture_recon.max_giveback)."""
    if len(equity) < 2:
        return 0.0
    peak, mg = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        mg = max(mg, peak - v)
    return mg


def _load(path, ts_col, eq_col, since=None, extra=None):
    rows, last_extra = [], None
    if Path(path).exists():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                try:
                    ts, eq = _ts(r[ts_col]), float(r[eq_col])
                except (KeyError, ValueError):
                    continue
                if since and ts < since:
                    continue
                rows.append((ts, eq))
                if extra and r.get(extra) not in (None, ""):
                    last_extra = r[extra]
    rows.sort(key=lambda x: x[0])
    return rows, last_extra


def load_sim():
    merged = []
    for f in SIM_FILES:
        merged += _load(f, "ts_utc", "equity")[0]
    merged.sort(key=lambda x: x[0])
    seen, out = set(), []
    for ts, eq in merged:                       # both files poll the same shared SIM acct
        if ts.isoformat() in seen:
            continue
        seen.add(ts.isoformat())
        out.append((ts, eq))
    return out


def _pct(gb, eq):
    base = eq[0] if eq else 0.0
    return (gb / base * 100.0) if base else 0.0


def main():
    sim = load_sim()
    if not sim:
        print("No SIM invvol equity data — was SIM_INVVOL=1 actually running? Nothing to compare.")
        return
    since = sim[0][0]
    comb, n_trades = _load(COMBINE, "ts_utc", "equity", since=since, extra="n_trades")

    sim_eq = [e for _, e in sim]
    comb_eq = [e for _, e in comb]
    sim_gb, comb_gb = max_giveback(sim_eq), max_giveback(comb_eq)
    sim_p, comb_p = _pct(sim_gb, sim_eq), _pct(comb_gb, comb_eq)
    days = (sim[-1][0] - since).days

    if sim_p == 0 and comb_p == 0:
        verdict = "no giveback on either yet (flat / too early to tell)"
    elif sim_p < comb_p:
        verdict = "trimming YANK gave back LESS (% of equity) -> supports the trim"
    elif sim_p > comb_p:
        verdict = "trimming YANK gave back MORE -> does NOT support the trim"
    else:
        verdict = "tie"

    print(f"window: since {since:%Y-%m-%d} (~{days}d) | SIM polls={len(sim)} combine polls={len(comb)} "
          f"combine n_trades={n_trades or 0}")
    print(f"SIM treatment (YANK 1 / MIM 1): MaxGiveback ${sim_gb:,.0f} ({sim_p:.2f}% of equity)")
    print(f"Combine control (YANK 2 / MIM 1): MaxGiveback ${comb_gb:,.0f} ({comb_p:.2f}% of equity)")
    print(f"=> {verdict}")
    print("CAVEAT: accounts differ in size/leverage; metric is equity-poll based and the "
          "sample is small -> DIRECTIONAL ONLY, not significant (want N>=20 trades).")


if __name__ == "__main__":
    main()
