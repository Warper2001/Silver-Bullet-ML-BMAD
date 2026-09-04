#!/usr/bin/env python3
"""
BTC carry: re-derive the entry hurdle in LIVE funding units.

THE DEFECT (measured 2026-09-02)
--------------------------------
`backtest_btc_carry*.py` reads `data/kraken/PF_XBTUSD_funding_rate.csv`, which is
NOT Kraken's published funding rate. Its own downloader says so: Kraken's
historical funding endpoint is not public, so the series is a PROXY derived from
the mark/spot basis. The live executor, meanwhile, reads the actual funding rate
off the Kraken tickers API.

Over the 277 eight-hour periods the paper executor lived through (2026-06-02 ->
2026-09-02) the two series correlate at 0.95 -- same shape -- but differ in
level by ~7.3x:

    LIVE  (executor / tickers) : mean  30.19% ann, median 27.32%
    PROXY (backtest / basis)   : mean   4.13% ann, median  3.69%

The strategy's entry rule is `annualized > 10%`. Against the proxy that fires in
48 of 277 periods (17% -- only genuine spikes). Against live funding it fires in
175 of 277 (63% -- most of the time). So the backtest measured a highly selective
strategy and the shop deployed a permissive one under the same name and the same
"10%" label. The reported Sharpe 12.64 / 23.6% ann. describes a bot that was
never run; the bot that ran churned $30 round trips and made ~2%.

The hurdle is denominated in the wrong units. This script derives what it should
be, from the executor's OWN observed funding -- the only real funding data the
shop possesses.

METHOD
------
Read every `funding=...%ann` reading out of logs/btc_carry_executor.log, resample
to the 8h funding cadence, then sweep candidate hurdles through the v3 live rules
(exit: >=3 of last 5 negative payments, or 12 consecutive below hurdle) and
report round-trips, cost drag, net P&L and time-in-position for each.

This is OBSERVATION, not promotion: it runs on a paper record, risks nothing, and
its output is a candidate hurdle plus the evidence for it. Whether that hurdle
ever governs real money is a separate, sealed decision.
"""
from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path

import pandas as pd

LOG_DEFAULT = Path("logs/btc_carry_executor.log")

# Frozen to match the live executor (btc_carry_executor.py).
COST_BPS = 15.0                    # per transition; a round trip is 2x
COST_FRAC = COST_BPS / 10_000.0
NEG_THRESHOLD = -0.0001            # -0.01%/8h counts as a negative payment
NEG_WINDOW_SIZE = 5
NEG_WINDOW_MIN_NEG = 3
BELOW_HURDLE_EXIT_PERIODS = 12
PERIODS_PER_YEAR = 3 * 365

POLL_RE = re.compile(
    r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ INFO poll .*funding=([-\d.]+)%ann"
)


def load_live_funding(log_path: Path) -> pd.Series:
    """Observed annualized funding from the executor's own log, on the 8h grid."""
    rows = []
    with open(log_path) as fh:
        for line in fh:
            m = POLL_RE.match(line)
            if m:
                rows.append((pd.Timestamp(m.group(1), tz="UTC"), float(m.group(2))))
    if not rows:
        raise SystemExit(f"no funding readings found in {log_path}")
    s = (
        pd.DataFrame(rows, columns=["ts", "ann_pct"])
        .drop_duplicates("ts")
        .set_index("ts")
        .sort_index()["ann_pct"]
    )
    return s.resample("8h").mean().dropna()


def simulate(ann_pct: pd.Series, hurdle_pct: float, entry_confirm: int = 1) -> dict:
    """Replay the v3 live rules at a given hurdle. Returns summary stats."""
    per_period = ann_pct / 100.0 / PERIODS_PER_YEAR  # annualized % -> per-8h decimal

    in_pos = False
    negwin: deque = deque(maxlen=NEG_WINDOW_SIZE)
    below = above = 0
    cur = 0.0
    npay = 0
    trades: list[dict] = []
    cost = 0.0
    periods_in_pos = 0

    for ts, ann in ann_pct.items():
        rate = per_period.loc[ts]
        if not in_pos:
            if ann > hurdle_pct:
                above += 1
                if above >= entry_confirm:
                    in_pos = True
                    cost += COST_FRAC
                    cur, npay = 0.0, 0
                    negwin.clear()
                    below = above = 0
            else:
                above = 0
        else:
            periods_in_pos += 1
            cur += rate
            npay += 1
            negwin.append(1 if rate < NEG_THRESHOLD else 0)
            below = below + 1 if ann < hurdle_pct else 0
            if (len(negwin) == NEG_WINDOW_SIZE and sum(negwin) >= NEG_WINDOW_MIN_NEG) or (
                below >= BELOW_HURDLE_EXIT_PERIODS
            ):
                cost += COST_FRAC
                trades.append({"payments": npay, "net": cur - 2 * COST_FRAC})
                in_pos = False
                negwin.clear()
                below = above = 0

    open_net = (cur - COST_FRAC) if in_pos else 0.0
    closed_net = sum(t["net"] for t in trades)
    net = closed_net + open_net
    wins = [t for t in trades if t["net"] > 0]
    days = (ann_pct.index[-1] - ann_pct.index[0]).total_seconds() / 86400
    return {
        "hurdle_pct": hurdle_pct,
        "entry_confirm": entry_confirm,
        "round_trips": len(trades),
        "open": in_pos,
        "winners": len(wins),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "net_usd_10k": net * 10_000,
        "cost_usd_10k": cost * 10_000,
        "time_in_pos_pct": 100.0 * periods_in_pos / len(ann_pct),
        "ann_pct_return": (net * 365 / days * 100) if days else float("nan"),
        "median_hold_payments": (
            pd.Series([t["payments"] for t in trades]).median() if trades else float("nan")
        ),
    }


def _executor_ground_truth(
    state_path: Path = Path("data/carry_executor_state.json"),
) -> dict | None:
    """The executor's own tally -- the only ground truth for what actually ran.

    state.json stores P&L and cost as fractions of notional, so a $10k notional
    turns 0.005580 into $55.80.
    """
    import json

    if not state_path.exists():
        return None
    st = json.loads(state_path.read_text())
    notional = st.get("notional_usd", 10_000.0)
    scale = 10_000.0 / notional
    return {
        "n_trades": st.get("n_trades"),
        "net_usd": st.get("total_pnl", 0.0) * notional * scale,
        "cost_usd": st.get("total_cost", 0.0) * notional * scale,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=LOG_DEFAULT)
    ap.add_argument("--confirm", type=int, nargs="+", default=[1, 3],
                    help="entry-confirmation lengths to sweep (periods)")
    args = ap.parse_args()

    ann = load_live_funding(args.log)
    print(f"live funding observations: {len(ann)} 8h periods  "
          f"{ann.index.min()} -> {ann.index.max()}")
    print(f"  mean {ann.mean():.2f}%ann   median {ann.median():.2f}%   "
          f"p75 {ann.quantile(.75):.2f}%   p90 {ann.quantile(.90):.2f}%\n")

    # The deployed hurdle (10%) fires this often against live funding:
    fires = (ann > 10).mean() * 100
    print(f"deployed hurdle 10% fires in {fires:.1f}% of periods against LIVE funding")
    print("(the backtest's proxy series put that at ~17% -- the mismatch this fixes)\n")

    grid = [10, 20, 30, 40, 50, 60, 73, 80, 100, 125, 150]
    print(f"{'hurdle':>7}{'conf':>6}{'trips':>7}{'win':>6}{'net$/10k':>11}"
          f"{'cost$':>9}{'in-pos%':>9}{'ann%':>9}{'medHold':>9}")
    rows = []
    for confirm in args.confirm:
        for h in grid:
            r = simulate(ann, h, confirm)
            rows.append(r)
            print(f"{h:>7}{confirm:>6}{r['round_trips']:>7}{r['winners']:>6}"
                  f"{r['net_usd_10k']:>11,.2f}{r['cost_usd_10k']:>9,.0f}"
                  f"{r['time_in_pos_pct']:>9.1f}{r['ann_pct_return']:>9.2f}"
                  f"{r['median_hold_payments']:>9.0f}")
        print()

    # ---- PARITY GATE ------------------------------------------------------
    # Before any hurdle number from the table above is allowed to mean anything,
    # the replay must reproduce the executor it claims to replay. Ground truth
    # comes from the executor's own state file, not from this script.
    truth = _executor_ground_truth()
    if truth:
        deployed = next(r for r in rows if r["hurdle_pct"] == 10 and r["entry_confirm"] == 1)
        print("\n=== PARITY GATE: replay vs the executor's own record ===")
        print(f"{'':<26}{'replay':>12}{'executor':>12}")
        print(f"{'round trips':<26}{deployed['round_trips']:>12}{truth['n_trades']:>12}")
        print(f"{'net $ / $10k':<26}{deployed['net_usd_10k']:>12,.2f}{truth['net_usd']:>12,.2f}")
        print(f"{'costs $ / $10k':<26}{deployed['cost_usd_10k']:>12,.2f}{truth['cost_usd']:>12,.2f}")
        ratio = (deployed["net_usd_10k"] / truth["net_usd"]) if truth["net_usd"] else float("inf")
        print(f"\nnet ratio replay/executor: {ratio:.1f}x")
        if abs(ratio - 1.0) > 0.25 or deployed["round_trips"] != truth["n_trades"]:
            print(
                "\n*** PARITY FAILED ***\n"
                "This replay does not reproduce the live executor at the deployed\n"
                "hurdle. Every number in the sweep above is therefore a property of\n"
                "the MODEL, not of the strategy, and MUST NOT be used to pick a\n"
                "hurdle. Establish executor<->replay parity first; tuning a\n"
                "parameter inside a model that cannot reproduce its own subject is\n"
                "how MIM-NB got deployed on an engine that disagreed with it by\n"
                "$2,147 over the same bars (2026-07-07).\n"
                "Likely suspects, in order: (1) the executor evaluates its exit\n"
                "window on every ~60s poll while this replay evaluates on the 8h\n"
                "funding grid -- far more chances to trip 3-of-5; (2) accrual uses\n"
                "realized funding payments, not the quoted annualized rate this\n"
                "script resamples; (3) restarts reseed executor state."
            )
        else:
            print("\nparity OK -- sweep numbers may be interpreted.")

    best = max(rows, key=lambda r: r["net_usd_10k"])
    print(f"best net on this record: hurdle={best['hurdle_pct']}% "
          f"confirm={best['entry_confirm']} -> ${best['net_usd_10k']:,.2f}/10k "
          f"({best['ann_pct_return']:.2f}% ann), {best['round_trips']} round trips")
    print("\nNOTE: this is a single 3-month paper record. The number above is a")
    print("CANDIDATE, not a validated parameter -- picking the max of a sweep is")
    print("exactly the move the shop's own process exists to stop. It is written")
    print("down so a prospective rule can be sealed BEFORE it governs anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
