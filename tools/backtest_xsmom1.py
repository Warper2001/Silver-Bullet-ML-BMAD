"""XSMOM-1 evaluation — REFUSES TO RUN unless the power gate permits.

Pre-registration: _bmad-output/preregistration_xsmom1_cross_sectional_momentum.md §4.5(8)

This is the only program permitted to compute the aligned signal->return
statistic. It will not do so unless:
  1. power_verdict.json exists,
  2. its input_sha256 matches the data file this script is about to read,
  3. its verdict is POWERED or MARGINALLY_POWERED.

On UNDERPOWERED the study is terminal by the seal's stopping rule: the panel is
left unspent and this script exits non-zero without reading a single return.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path("/root/Silver-Bullet-ML-BMAD")
BARS = REPO / "data/commodity_curve/coverage-pilot-2025-20260906-v3/contract_bars.csv"
VERDICT = Path(__file__).resolve().parents[1] / "_bmad-output" / "power_verdict.json"
PERMITTED = {"POWERED", "MARGINALLY_POWERED"}


def gate_check() -> dict:
    if not VERDICT.exists():
        sys.exit("REFUSED: no power_verdict.json. Run tools/xsmom1_power_gate.py first (seal §4).")
    v = json.loads(VERDICT.read_text())

    actual = hashlib.sha256(BARS.read_bytes()).hexdigest()
    if v.get("input_sha256") != actual:
        sys.exit(
            "REFUSED: power verdict was computed on different input data.\n"
            f"  verdict input_sha256 = {v.get('input_sha256')}\n"
            f"  actual  input_sha256 = {actual}\n"
            "  Re-run the power gate on the current data (seal §4.5)."
        )

    verdict = v.get("verdict")
    if verdict not in PERMITTED:
        sys.exit(
            f"REFUSED: power verdict is {verdict} (achieved power "
            f"{v.get('achieved_power'):.1%} vs declared plausible effect "
            f"{v.get('theta_plaus')}).\n"
            "  Seal §4.4: an UNDERPOWERED verdict is TERMINAL. No strategy returns are\n"
            "  computed and the panel is left unspent for a future adequately-powered\n"
            "  seal. Seal §8 forbids re-sweeping k, changing the statistic, narrowing\n"
            "  the universe, or rebalancing faster to get around this.\n"
            f"  MDE80 = {v.get('mde80_mean_ic')} mean IC (gross SR "
            f"{v.get('mde80_gross_sr')}); years needed for 80% power: "
            f"{v.get('years_for_80pct_power')}"
        )
    return v


def main() -> None:
    v = gate_check()
    # --- Only reachable on POWERED / MARGINALLY_POWERED -------------------
    print(f"Gate permits evaluation (verdict={v['verdict']}, power={v['achieved_power']:.1%}).")
    print("Evaluation body intentionally not implemented: the gate returned")
    print("UNDERPOWERED on the sealed dataset, so this path was never authorised.")
    raise SystemExit(
        "STOP: implement the §5 construction and §6 stresses only under a seal "
        "whose gate actually permitted evaluation."
    )


if __name__ == "__main__":
    main()
