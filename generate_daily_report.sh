#!/bin/bash
###############################################################################
# Daily Trading Report Generator
###############################################################################

set -e

REPORT_DATE=${1:-$(date +%Y-%m-%d)}
REPORT_FILE="data/reports/hybrid_daily_report_${REPORT_DATE}.txt"

mkdir -p data/reports

cat > "$REPORT_FILE" << REPORT
==============================================================================
HYBRID REGIME-AWARE TRADING SYSTEM - DAILY REPORT
==============================================================================
Date: $REPORT_DATE
Generated: $(date '+%Y-%m-%d %H:%M:%S UTC')

==============================================================================
SYSTEM STATUS
==============================================================================

PROCESS: $(pgrep -f "start_paper_trading.py" > /dev/null && echo "RUNNING ✅" || echo "STOPPED ❌")

==============================================================================
SIGNAL GENERATION
==============================================================================

Total Signals: $(grep -c "Signal #" logs/hybrid_trading.log 2>/dev/null || echo "0")

Signals Today: $(grep "Signal #" logs/hybrid_trading.log 2>/dev/null | grep -c "$REPORT_DATE" || echo "0")

Target: 3-4 signals per day
Status: $( [ "$(grep "Signal #" logs/hybrid_trading.log 2>/dev/null | grep -c "$REPORT_DATE" || echo "0")" -ge 3 ] && echo "ON TARGET ✅" || echo "BELOW TARGET ⚠️")

==============================================================================
REGIME DISTRIBUTION
==============================================================================

$(grep "Regime=" logs/hybrid_trading.log 2>/dev/null | grep "$REPORT_DATE" | cut -d'=' -f2 | sort | uniq -c | sort -rn || echo "No regime data yet")

==============================================================================
RECENT SIGNALS
==============================================================================

$(grep "Signal #" logs/hybrid_trading.log 2>/dev/null | grep "$REPORT_DATE" | tail -10 || echo "No signals yet")

==============================================================================
ERROR LOG
==============================================================================

ERRORS: $(tail -10000 logs/hybrid_trading.log 2>/dev/null | grep "$REPORT_DATE" | grep -c "ERROR" || echo "0")

WARNINGS: $(tail -10000 logs/hybrid_trading.log 2>/dev/null | grep "$REPORT_DATE" | grep -c "WARNING" || echo "0")

==============================================================================
PERFORMANCE NOTES
==============================================================================

Expected Daily Performance (from backtest):
- Signals: 3-4 per day
- Win Rate: 51-52%
- Daily P&L: ~+0.08%

==============================================================================
NEXT STEPS
==============================================================================

Tomorrow:
- Monitor signal generation during market hours
- Track regime distribution
- Validate probability thresholds

End of Week:
- Compare actual vs expected performance
- Adjust threshold if needed
- Prepare Phase 2 decision

==============================================================================
Report Complete
==============================================================================
REPORT

echo "✅ Daily report generated: $REPORT_FILE"
cat "$REPORT_FILE"
