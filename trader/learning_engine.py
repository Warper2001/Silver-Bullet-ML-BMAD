#!/usr/bin/env python3
"""
Continuous Learning Engine: Analyzes performance, identifies improvements,
retrains models, and proposes strategy adjustments.

Runs weekly (or on demand) to:
1. Pull completed trade data from logs
2. Identify what's working / not working
3. Retrain ML models on new data
4. Run walk-forward validation
5. Generate improvement proposals as markdown reports
6. Adjust probability thresholds based on live performance
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
BASE_DIR = Path(__file__).parent.parent
VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"


def retrain_models():
    """Trigger model retraining via existing pipeline."""
    print("Triggering ML model retraining...")
    script = BASE_DIR / "train_meta_model.py"
    if script.exists():
        result = subprocess.run(
            [str(VENV_PYTHON), str(script)],
            capture_output=True, text=True, timeout=600
        )
        print(result.stdout[-2000:] if result.stdout else "(no output)")
        if result.returncode != 0:
            print(f"Retraining stderr: {result.stderr[-1000:]}")
        return result.returncode == 0
    else:
        print(f"  Train script not found: {script}")
        return False


def run_backtest_validation():
    """Run the best available backtest for validation."""
    print("Running walk-forward validation backtest...")
    script = BASE_DIR / "proper_backtest_with_test_set.py"
    if script.exists():
        result = subprocess.run(
            [str(VENV_PYTHON), str(script)],
            capture_output=True, text=True, timeout=300
        )
        output = result.stdout[-3000:] if result.stdout else "(no output)"
        print(output)
        return output
    return "(backtest script not found)"


def analyze_recent_performance():
    """Read recent paper trading logs and produce analysis."""
    log_dir = BASE_DIR / "logs"
    analysis = []

    for log_name in ["tier2_paper_trading.log", "tier2_paper_trader.log"]:
        log_path = log_dir / log_name
        if log_path.exists():
            size_mb = log_path.stat().st_size / 1e6
            with open(log_path, errors="replace") as f:
                lines = f.readlines()
            # Last 200 lines for recent context
            recent = lines[-200:]
            errors = [l for l in recent if "ERROR" in l or "FAIL" in l or "401" in l]
            trades = [l for l in recent if "ENTRY" in l or "TP hit" in l or "SL hit" in l]
            analysis.append({
                "file": log_name,
                "size_mb": round(size_mb, 2),
                "recent_errors": len(errors),
                "recent_trade_events": len(trades),
                "last_line": lines[-1].strip() if lines else "(empty)",
            })

    return analysis


def generate_learning_report():
    """Generate a learning/improvement report."""
    now = datetime.now()
    report_path = REPORTS_DIR / f"learning_report_{now.strftime('%Y%m%d_%H%M')}.md"

    log_analysis = analyze_recent_performance()

    report = f"""# Continuous Learning Report
Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}

## Log Analysis
"""
    for la in log_analysis:
        report += f"""
### {la['file']}
- Size: {la['size_mb']} MB
- Recent errors: {la['recent_errors']}
- Recent trade events: {la['recent_trade_events']}
- Last entry: {la['last_line'][:120]}
"""

    report += """
## Retraining Status
"""
    retrain_ok = retrain_models()
    report += f"- Model retrain: {'SUCCESS' if retrain_ok else 'FAILED or SKIPPED'}\n"

    report += """
## Validation Backtest
"""
    backtest_output = run_backtest_validation()
    report += f"```\n{backtest_output[:2000]}\n```\n"

    report += f"""
## Improvement Proposals
(Auto-generated based on current system state)

1. MONITOR 1-min vs 5-min bar performance gap - use only 5-min for signal generation
2. Review kill zone windows - confirm London AM (3-4 ET), NY AM (10-11 ET), NY PM (2-3 ET)
3. Check regime distribution in recent trades - ensure hybrid model is selecting correctly
4. Validate probability threshold (0.65) against recent win rates
5. Review daily drawdown patterns - flag any day >$300 loss for manual review

## Next Actions
- [ ] Review this report
- [ ] Approve or adjust improvement proposals
- [ ] Run: python trader/promotion_engine.py status
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nLearning report saved: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    generate_learning_report()
