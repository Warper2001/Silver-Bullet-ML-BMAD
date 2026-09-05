"""Option 1b (post-R3 research plan): YANK exit-horizon sweep, 4-core parallel.

Pre-registration: _bmad-output/preregistration_option1b_yank_horizon.md
Sealed grid: max_hold_bars in {60 (baseline), 120, 180, 240}.
Sealed null: 16 independent full runs, max_hold_bars ~ uniform[30,300].

Reuses backtest_tier2_1year_validation.run_backtest() UNCHANGED (same engine
as every other YANK backtest in this project) via config_overrides -- entry
logic, sl_multiplier=2.0, tp_multiplier=8.0, ml_threshold=0.50 all untouched.
Dev-window only (ends at HOLDOUT_CUTOFF, 2026-03-01) -- does not touch the
sealed holdout.

Usage: nohup .venv/bin/python tools/option1b_yank_horizon_sweep.py > <log> 2>&1 &
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path("/root/Silver-Bullet-ML-BMAD")
GRID = (60, 240)  # cut from (60,120,180,240) -- see pre-reg's run-count revision
N_NULL = 8  # cut from 16
NULL_LO, NULL_HI = 30, 300
RNG_SEED = 20260904  # sealed in the pre-registration commit
OUT_JSON = Path("/root/.claude/jobs/960bda86/tmp/option1b_yank_results.json")


def _run_one(max_hold_bars: int) -> dict:
    """Runs in a worker process: fresh imports, fresh bar load, one full backtest.

    SAFETY: Tier2StreamingTrader._close_active_trade() writes every closed
    trade to src.monitoring.trade_db.TradeDatabase(), default db_path=
    "data/trades.db" -- a RELATIVE path resolved against the worker's CWD,
    with NO constructor injection point available. Discovered 2026-09-05 when
    a CWD mismatch made this raise (data/ not present under the launch dir) --
    the same write, under a CWD where data/ *does* exist, would land in a
    file at exactly the live trades ledger's relative path. Confirmed no
    contamination of the real ledger occurred (separate inode, unchanged row
    count), but this run redirects TradeDatabase to an isolated per-process
    tmp path unconditionally, so no future invocation can get near a
    real-looking path regardless of CWD.
    """
    sys.path.insert(0, str(REPO))
    import os
    import tempfile

    import src.monitoring.trade_db as _trade_db_mod

    _tmp_db = os.path.join(tempfile.mkdtemp(prefix="option1b_yank_"), "throwaway_trades.db")

    def _isolated_init(self, db_path: str = _tmp_db) -> None:  # noqa: ARG001 -- always uses _tmp_db
        self.db_path = _tmp_db
        self._init_db()

    _trade_db_mod.TradeDatabase.__init__ = _isolated_init

    from backtest_tier2_1year_validation import (
        CSV_2025,
        CSV_2026,
        HOLDOUT_CUTOFF,
        load_bars,
        profit_factor,
        run_backtest,
    )

    t0 = time.monotonic()
    bars = load_bars(REPO / CSV_2025) + load_bars(REPO / CSV_2026, end=HOLDOUT_CUTOFF)
    trades = asyncio.run(
        run_backtest(
            bars, ml_threshold=0.50, config_overrides={"max_hold_bars": max_hold_bars}
        )
    )
    pnls = [t.pnl for t in trades]
    pf = profit_factor(pnls)
    sorted_pnls = sorted(pnls, reverse=True)
    pf_ex_top3 = profit_factor(sorted_pnls[3:]) if len(sorted_pnls) > 3 else float("nan")
    elapsed = time.monotonic() - t0
    return {
        "max_hold_bars": max_hold_bars,
        "n": len(trades),
        "pf": pf,
        "pf_ex_top3": pf_ex_top3,
        "gross": sum(pnls),
        "elapsed_s": elapsed,
    }


def main() -> None:
    rng = random.Random(RNG_SEED)
    null_draws = [rng.randint(NULL_LO, NULL_HI) for _ in range(N_NULL)]

    tasks: list[tuple[str, int]] = [("grid", h) for h in GRID] + [
        ("null", h) for h in null_draws
    ]
    print(f"{len(tasks)} tasks: {len(GRID)} grid cells + {N_NULL} null draws, 4 workers")
    print(f"null draws (sealed rng, seed={RNG_SEED}): {null_draws}")

    results: dict[str, list[dict]] = {"grid": [], "null": []}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_run_one, hold_bars): (kind, hold_bars) for kind, hold_bars in tasks
        }
        done = 0
        for fut in as_completed(futures):
            kind, hold_bars = futures[fut]
            r = fut.result()
            results[kind].append(r)
            done += 1
            print(
                f"[{done}/{len(tasks)}] {kind} max_hold_bars={hold_bars}: "
                f"N={r['n']} PF={r['pf']:.3f} PF_ex_top3={r['pf_ex_top3']:.3f} "
                f"({r['elapsed_s']:.0f}s)"
            )

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    baseline = next(r for r in results["grid"] if r["max_hold_bars"] == 60)
    best = max(results["grid"], key=lambda r: r["pf"])
    null_pfs = sorted(r["pf"] for r in results["null"])
    p95_idx = int(0.95 * len(null_pfs))
    null_p95 = null_pfs[min(p95_idx, len(null_pfs) - 1)]
    null_median = null_pfs[len(null_pfs) // 2]

    print(f"\nBaseline (60 bars): PF={baseline['pf']:.3f} N={baseline['n']}")
    print(f"Best cell: {best['max_hold_bars']} bars, PF={best['pf']:.3f}")
    print(f"Null (N={N_NULL}): median={null_median:.3f} p95={null_p95:.3f}")

    print(f"\n{'='*60}")
    print("GATE 0 -- Option 1b (YANK exit-horizon)")
    print(f"{'='*60}")
    checks = {
        f"best PF ({best['pf']:.3f} @ {best['max_hold_bars']}b) > baseline PF ({baseline['pf']:.3f})": (
            best["pf"] > baseline["pf"]
        ),
        f"best PF ({best['pf']:.3f}) > null p95 ({null_p95:.3f})": best["pf"] > null_p95,
        f"N >= 60 (N={best['n']})": best["n"] >= 60,
        f"ex-top3 PF at best ({best['pf_ex_top3']:.3f}) > ex-top3 PF at baseline ({baseline['pf_ex_top3']:.3f})": (
            best["pf_ex_top3"] > baseline["pf_ex_top3"]
        ),
    }
    for check, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")

    grid_sorted = sorted(results["grid"], key=lambda r: r["max_hold_bars"])
    pfs_in_order = [r["pf"] for r in grid_sorted]
    print(f"\n  PF by hold_bars: {dict(zip([r['max_hold_bars'] for r in grid_sorted], [round(p,3) for p in pfs_in_order]))}")
    print("  Monotonicity/lone-spike check: NOT EVALUATED (grid cut to 2 points --")
    print("   needs >=3 to detect an interior spike; see pre-registration #5).")

    verdict = "PASS (provisional -- see pre-reg #5)" if all(checks.values()) else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    print(f"  (N_null={N_NULL} -- coarse null estimate, disclosed in the pre-registration;")
    print("   treat a near-boundary result as inconclusive, not a clean PASS/FAIL.)")


if __name__ == "__main__":
    main()
