#!/bin/bash
# Continuous grid search monitoring

clear
echo "🔍 EXHAUSTIVE GRID SEARCH MONITOR"
echo "=================================="
echo ""
echo "Configuration:"
echo "  Process ID: 552041"
echo "  Total Combinations: 150"
echo "  Data: ALL 75,127 bars (no sampling)"
echo "  Mode: Exhaustive optimization"
echo ""

while true; do
    clear
    echo "🔍 EXHAUSTIVE GRID SEARCH MONITOR"
    echo "=================================="
    echo ""

    # Check process status
    if ps -p 552041 > /dev/null 2>&1; then
        echo "✅ Status: RUNNING"
        echo "   CPU: $(ps -p 552041 -o %cpu | tail -1)%"
        echo "   Memory: $(ps -p 552041 -o %mem | tail -1)%"
        echo "   Runtime: $(ps -p 552041 -o etime | tail -1)"
    else
        echo "❌ Status: COMPLETED or STOPPED"
        echo ""
        echo "Final Results:"
        echo "============="
        tail -100 /tmp/grid_search_full.log | grep -A 20 "TOP 10 PARAMETER SETS"
        break
    fi

    echo ""
    echo "Latest Progress:"
    echo "================"
    tail -10 /tmp/grid_search_full.log | tail -5

    echo ""
    echo "Estimates:"
    echo "=========="

    # Try to extract progress percentage
    progress=$(grep "Progress:" /tmp/grid_search_full.log | tail -1 | grep -o "[0-9]*\.[0-9]*" | head -1)
    if [ ! -z "$progress" ]; then
        pct=$(echo "$progress * 100" | bc)
        echo "Completed: $pct%"

        # Estimate remaining time
        completed=$(echo "$progress * 150" | bc | cut -d'.' -f1)
        remaining=$((150 - completed))
        echo "Remaining: $remaining combinations"
    else
        echo "Still initializing..."
    fi

    echo ""
    echo "Refresh: Every 30 seconds"
    echo "Press Ctrl+C to exit monitoring"

    sleep 30
done

echo ""
echo "✅ Grid search monitoring complete"
echo "📊 Results saved to:"
echo "   - data/reports/exit_parameter_optimization_1min.csv"
echo "   - data/reports/exit_parameter_optimization_1min.md"
