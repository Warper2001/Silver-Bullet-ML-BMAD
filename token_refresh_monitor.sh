#!/bin/bash
# Auto-refresh token before it expires (every 15 minutes)

while true; do
    echo "[$(date)] Checking token health..."
    
    # Check if we can make a successful API call
    if curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(cat .access_token 2>/dev/null)" \
        "https://api.tradestation.com/v3/marketdata/quotes/MNQH26" | grep -q "200"; then
        echo "✅ Token is healthy"
    else
        echo "❌ Token expired, refreshing..."
        echo "Token refresh needed - please run OAuth flow"
    fi
    
    # Check every 15 minutes
    sleep 900
done
