from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OPERATOR_ROOT = Path("/root/Silver-Bullet-ML-BMAD")
RUNS = BASE / "runs"
DEFAULT_SOURCE_RUN = (
    ROOT / "research/pf_improvement/runs/20260913T022924-run-26b118774d"
)
DEFAULT_DATA = OPERATOR_ROOT / "data/mim_x/mnq_1min_by_contract.csv"
DEFAULT_DIAGNOSTIC_RUN = (
    OPERATOR_ROOT / "research/mim_robustness/runs/20260912T151842-run-f8608e71fb"
)

CONFIG = {
    "arm": "A",
    "delay": 2,
    "quantity": 1,
    "point_value": 2.0,
    "side_cost": 1.12,
    "round_trip_cost": 2.24,
    "marks_et": [
        f"{hour:02d}:{minute:02d}"
        for hour, minute in [
            (10, 0),
            (10, 30),
            (11, 0),
            (11, 30),
            (12, 0),
            (12, 30),
            (13, 0),
            (13, 30),
            (14, 0),
            (14, 30),
            (15, 0),
            (15, 30),
        ]
    ],
    "threshold_quantiles": [i / 10 for i in range(1, 10)],
    "null_shifts": [31, 79, 157, 313, 631],
    "bootstrap_draws": 2000,
    "bootstrap_blocks": [5, 10, 20],
    "seed": 20260913,
    "alpha": 0.05,
    "power": 0.80,
    "target_pf": 1.40,
    "profit_retention": 0.90,
    "endpoint_sessions": 500,
    "horizon_months": 30,
    "baseline_sessions": 1323,
    "baseline_trades": 801,
    "baseline_net": 21889.76,
    "baseline_pf": 1.285668795960033,
    "baseline_exit_labels": {"CAT_STOP": 71, "EOD_CLOSE_PROXY": 723, "REVERSAL": 7},
    "accounting_tolerance": 1e-8,
}
