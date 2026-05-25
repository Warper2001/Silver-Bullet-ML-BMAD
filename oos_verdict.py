#!/usr/bin/env python3
"""oos_verdict.py — OOS verdict report generator (AR5, AR8, FR26).

Reads the sealed holdout data, computes OOS metrics, and produces a Go/No-Go verdict.

Usage:
    PYTHONPATH=. python oos_verdict.py --prereg <path-to-prereg-doc.md>

First action: calls checkpoint_or_abort() — aborts on any integrity failure (AR8).
Every holdout access is logged to data/sealed_holdout/ACCESS_LOG.md (AR8).
"""

import argparse
import itertools
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from oos_checkpoint import _parse_prereg, checkpoint_or_abort
from src.research.backtest_engine import BacktestEngine
from src.research.strategy_core import (
    calc_max_drawdown_pct,
    calc_profit_factor,
    calc_sharpe,
)

HOLDOUT_DIR = Path(__file__).parent / "data/sealed_holdout"
HOLDOUT_CSV = HOLDOUT_DIR / "mnq_1min_holdout_20260301_plus.csv"
ACCESS_LOG = HOLDOUT_DIR / "ACCESS_LOG.md"
REPORTS_DIR = Path(__file__).parent / "data/reports"

# Pre-registered thresholds (FR26, pre-registered before any holdout access)
THRESHOLD_PF = 2.0
THRESHOLD_SHARPE = 1.5
THRESHOLD_MAX_DD_PCT = 0.10   # fraction — 10%
THRESHOLD_N = 200

# Stopping rule: halt experiment if triggered regardless of total N
STOPPING_RULE_N = 100
STOPPING_RULE_PF = 1.1


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(trades: list) -> tuple:
    """Compute (pf, sharpe, max_dd_pct, n) from a list of trade objects.

    trades: objects with .pnl_usd (float) and .timestamp_exit (pd.Timestamp).
    Returns (0.0, 0.0, 0.0, 0) for an empty list without crashing.
    """
    n = len(trades)
    if n == 0:
        return 0.0, 0.0, 0.0, 0

    pnls = [t.pnl_usd for t in trades]

    pf = calc_profit_factor(pnls)

    # Daily returns: group by exit date and sum pnl per day
    daily: dict = defaultdict(float)
    for t in trades:
        daily[t.timestamp_exit.date()] += t.pnl_usd
    sharpe = calc_sharpe(list(daily.values()))

    # Max drawdown: pre-accumulate equity curve
    equity = list(itertools.accumulate(pnls))
    max_dd_pct = calc_max_drawdown_pct(equity)

    # Supplement: calc_max_drawdown_pct skips drawdown when peak <= 0 (initial losing
    # streak). If equity dips below zero, also measure abs(min) / max as a conservative
    # additional check so the 10% gate is not bypassed by strategies that lose first.
    if equity:
        eq_min = min(equity)
        eq_max = max(equity)
        if eq_min < 0 and eq_max > 0:
            max_dd_pct = max(max_dd_pct, abs(eq_min) / eq_max)
        elif eq_max <= 0:
            max_dd_pct = 1.0  # strategy never profitable

    return pf, sharpe, max_dd_pct, n


def _determine_verdict(pf: float, sharpe: float, max_dd_pct: float, n: int) -> str:
    """Apply decision rules in priority order.

    Priority 1 — stopping rule (overrides everything):
        N >= 100 AND PF < 1.1 → STOPPING_RULE_TRIGGERED
    Priority 2 — insufficient sample:
        N < 200 → INCONCLUSIVE
    Priority 3 — all thresholds pass:
        PF >= 2.0 AND Sharpe >= 1.5 AND MaxDD <= 10% → GO
    Priority 4 — any metric fails:
        → NO-GO
    """
    if n >= STOPPING_RULE_N and pf < STOPPING_RULE_PF:
        return "STOPPING_RULE_TRIGGERED"
    if n < THRESHOLD_N:
        return "INCONCLUSIVE"
    if pf >= THRESHOLD_PF and sharpe >= THRESHOLD_SHARPE and max_dd_pct <= THRESHOLD_MAX_DD_PCT:
        return "GO"
    return "NO-GO"


# ---------------------------------------------------------------------------
# ACCESS_LOG
# ---------------------------------------------------------------------------

def _append_access_log(access_log: Path, prereg_sha: str, prereg_name: str, result_summary: str) -> None:
    """Append a timestamped table row to ACCESS_LOG.md (AR8).

    Format matches the existing pipe-delimited Markdown table in ACCESS_LOG.md.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    row = (
        f"| {ts} | `{prereg_sha}` | oos_verdict.py | OOS verdict run — {prereg_name} "
        f"| {result_summary} |"
    )
    with open(access_log, "a", encoding="utf-8") as f:
        f.write(row + "\n")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    report_path: Path,
    prereg_path: Path,
    hashes: dict,
    pf: float,
    sharpe: float,
    max_dd_pct: float,
    n: int,
    v: str,
) -> None:
    """Write the tamper-evident verdict report to report_path."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _threshold_row(metric: str, realized: str, threshold: str, passed: bool) -> str:
        status = "PASS" if passed else "FAIL"
        return f"| {metric} | {realized} | {threshold} | {status} |"

    pf_pass = pf >= THRESHOLD_PF
    sharpe_pass = sharpe >= THRESHOLD_SHARPE
    dd_pass = max_dd_pct <= THRESHOLD_MAX_DD_PCT
    n_pass = n >= THRESHOLD_N
    stopping = n >= STOPPING_RULE_N and pf < STOPPING_RULE_PF

    content = f"""# OOS Verdict Report

**Generated:** {ts} UTC
**Script:** oos_verdict.py

---

## Pre-Registration Reference

| Field | Value |
|---|---|
| Pre-registration document | `{prereg_path}` |
| Sealed commit (hash_c) | `{hashes.get("hash_c", "unknown")}` |
| Config SHA (hash_a) | `{hashes.get("hash_a", "unknown")}` |
| Source SHA (hash_b) | `{hashes.get("hash_b", "unknown")}` |

---

## Realized Metrics

| Metric | Realized | Threshold | Status |
|---|---|---|---|
{_threshold_row("Profit Factor", f"{pf:.4f}", f">= {THRESHOLD_PF:.1f}", pf_pass)}
{_threshold_row("Annualized Sharpe", f"{sharpe:.3f}", f">= {THRESHOLD_SHARPE:.1f}", sharpe_pass)}
{_threshold_row("Max Drawdown", f"{max_dd_pct:.1%}", f"<= {THRESHOLD_MAX_DD_PCT:.0%}", dd_pass)}
{_threshold_row("Trade Count (N)", str(n), f">= {THRESHOLD_N}", n_pass)}

---

## Stopping Rule

| Condition | Value | Triggered |
|---|---|---|
| N >= {STOPPING_RULE_N} AND PF < {STOPPING_RULE_PF} | N={n}, PF={pf:.4f} | {"YES" if stopping else "NO"} |

---

## VERDICT: {v}

{"**Rationale:** All pre-registered thresholds met." if v == "GO" else ""}
{"**Rationale:** One or more metrics failed threshold at N >= 200." if v == "NO-GO" else ""}
{"**Rationale:** Insufficient sample size (N < 200)." if v == "INCONCLUSIVE" else ""}
{"**Rationale:** Stopping rule triggered — PF < 1.1 after 100+ trades. Halt experiment." if v == "STOPPING_RULE_TRIGGERED" else ""}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verdict(
    prereg_path: Path,
    holdout_csv: Path = HOLDOUT_CSV,
    access_log: Path = ACCESS_LOG,
    reports_dir: Path = REPORTS_DIR,
) -> int:
    """Run the full OOS verdict pipeline.

    AC #1 — AR8: checkpoint_or_abort() is the FIRST action.
    Returns 0 if verdict is GO, 1 otherwise.
    """
    # AR8 — FIRST ACTION: gate all holdout access; raises SystemExit(1) on failure
    checkpoint_or_abort(prereg_path)

    # Extract sealed commit SHA from prereg doc
    hashes = _parse_prereg(prereg_path)
    prereg_sha = hashes.get("hash_c") or "unknown"

    # Run backtest on holdout data; log access even on failure (AR8)
    engine = BacktestEngine(csv_path=str(holdout_csv))
    try:
        trades = engine.run()
    except Exception as e:
        _append_access_log(access_log, prereg_sha, prereg_path.name, f"ERROR: {e}")
        raise

    # Compute metrics
    pf, sharpe, max_dd_pct, n = _compute_metrics(trades)

    # Determine verdict
    v = _determine_verdict(pf, sharpe, max_dd_pct, n)

    # AR8 — log holdout access BEFORE writing report
    result_summary = f"{v}: PF={pf:.4f}, N={n}, Sharpe={sharpe:.3f}, MaxDD={max_dd_pct:.1%}"
    _append_access_log(access_log, prereg_sha, prereg_path.name, result_summary)

    # Write verdict report
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc)
    report_path = reports_dir / f"oos_verdict_{ts.strftime('%Y%m%d_%H%M%S')}.md"
    _write_report(report_path, prereg_path, hashes, pf, sharpe, max_dd_pct, n, v)

    print(f"VERDICT: {v}")
    print(f"Report: {report_path}")
    return 0 if v == "GO" else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOS verdict report generator — reads sealed holdout data and emits Go/No-Go verdict."
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        required=True,
        help="Path to pre-registration document (.md)",
    )
    args = parser.parse_args()
    sys.exit(verdict(args.prereg))


if __name__ == "__main__":
    main()
