"""Finishes the Option 1b sweep: baseline (60 bars) already confirmed standalone,
this runs only the remaining 9 tasks (240-bar cell + 8 null draws) across 4
workers and combines with the confirmed baseline for the Gate 0 verdict.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from option1b_yank_horizon_sweep import GRID, NULL_HI, NULL_LO, N_NULL, RNG_SEED, _run_one

CONFIRMED_BASELINE = {
    "max_hold_bars": 60,
    "n": 78,
    "pf": 2.124733650221275,
    "pf_ex_top3": 1.5665737857181883,
    "gross": 10293.0,
    "elapsed_s": 2927.5701798000373,
}
OUT_JSON = Path("/root/.claude/jobs/960bda86/tmp/option1b_yank_results.json")


def main() -> None:
    rng = random.Random(RNG_SEED)
    null_draws = [rng.randint(NULL_LO, NULL_HI) for _ in range(N_NULL)]
    remaining_grid = [h for h in GRID if h != 60]  # just 240

    tasks = [("grid", h) for h in remaining_grid] + [("null", h) for h in null_draws]
    print(f"{len(tasks)} remaining tasks (baseline 60 already confirmed): {tasks}")

    results = {"grid": [dict(CONFIRMED_BASELINE)], "null": []}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_one, h): (kind, h) for kind, h in tasks}
        done = 0
        for fut in as_completed(futures):
            kind, h = futures[fut]
            r = fut.result()
            results[kind].append(r)
            done += 1
            print(
                f"[{done}/{len(tasks)}] {kind} max_hold_bars={h}: "
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
    print("GATE 0 -- Option 1b (YANK exit-horizon, reduced grid/null)")
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

    verdict = "PASS (provisional -- 2-point grid, see pre-reg #5)" if all(checks.values()) else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    print(f"  (N_null={N_NULL} -- coarse null estimate, disclosed in the pre-registration.)")


if __name__ == "__main__":
    main()
