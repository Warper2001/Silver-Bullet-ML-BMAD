#!/bin/bash
# Monitor grid search progress

echo "Grid Search Progress Monitor"
echo "==========================="
echo ""
echo "Process ID: 552041"
echo "Log File: /tmp/grid_search_full.log"
echo ""

# Check if process is running
if ps -p 552041 > /dev/null 2>&1; then
    echo "✅ Process is RUNNING"
    echo "   CPU: $(ps -p 552041 -o %cpu | tail -1)%"
    echo "   Memory: $(ps -p 552041 -o %mem | tail -1)%"
    echo "   Runtime: $(ps -p 552041 -o etime | tail -1)"
else
    echo "❌ Process is NOT running"
    echo ""
    echo "Final output:"
    tail -50 /tmp/grid_search_full.log
    exit 1
fi

echo ""
echo "Latest Progress:"
tail -10 /tmp/grid_search_full.log | grep -E "(Progress:|✅|TOP|ERROR)" || echo "Still initializing..."

echo ""
echo "Estimates:"
echo "- Total combinations: 150"
echo "- Expected time: 2-4 hours"
echo "- Current status: Running"
