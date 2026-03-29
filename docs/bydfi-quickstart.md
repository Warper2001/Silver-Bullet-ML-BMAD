# BYDFI Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Get BYDFI API Credentials (2 minutes)

1. Go to https://www.bydfi.com
2. Create account (or login)
3. Navigate to **API Management** (https://www.bydfi.com/en/user/api-management)
4. Click **Create API Key**
5. Set permissions: **Read Permission** + **Spot Trading Permission**
6. **IMPORTANT**: Save your API Key and Secret Key immediately
7. Copy and save:
   - API Key
   - Secret Key

### 2. Configure Environment (1 minute)

```bash
# Copy example file
cp .env.bydfi.example .env.bydfi

# Edit with your credentials
nano .env.bydfi
```

Fill in your credentials:
```bash
BYDFI_API_KEY=your_actual_key_here
BYDFI_API_SECRET=your_actual_secret_here
BYDFI_ENVIRONMENT=production
BYDFI_TRADING_SYMBOL=BTC-USDT
```

### 3. Test Setup (1 minute)

```bash
python tests/integration/test_bydfi_implementation.py
```

Expected output:
```
✓ Configuration Loading: PASSED
✓ Client Initialization: PASSED
✓ API Connectivity: PASSED
...
All tests PASSED! ✓
```

### 4. Start Paper Trading (1 minute)

```bash
python start_bydfi_paper_trading.py
```

---

## 📁 File Structure

```
Silver-Bullet-ML-BMAD/
├── src/
│   ├── data/
│   │   └── bydfi_config.py          # Configuration loader
│   └── execution/
│       └── bydfi/
│           ├── auth/
│           │   └── signature.py       # HMAC SHA256 authentication
│           ├── client.py              # REST API client
│           ├── market_data/
│           │   └── streaming.py       # WebSocket client
│           └── orders/
│               └── submission.py      # Order execution
├── scripts/
│   └── acquire_bydfi_historical_data.py
├── start_bydfi_paper_trading.py
├── bydfi_config.yaml
├── .env.bydfi.example
└── .env.bydfi                        # Your credentials (create this)
```

---

## 🧪 Common Commands

### Test Connectivity
```bash
python tests/integration/test_bydfi_implementation.py
```

### Acquire Historical Data
```bash
python scripts/acquire_bydfi_historical_data.py \
    --symbol BTC-USDT \
    --interval 5m \
    --days 365
```

### Start Paper Trading
```bash
python start_bydfi_paper_trading.py
```

---

## ⚙️ Configuration

### Environment Variables (.env.bydfi)

| Variable | Description | Example |
|----------|-------------|---------|
| `BYDFI_API_KEY` | Your API key | `6abc123...` |
| `BYDFI_API_SECRET` | Your API secret | `xyz789...` |
| `BYDFI_ENVIRONMENT` | Environment | `production` |
| `BYDFI_TRADING_SYMBOL` | Trading pair | `BTC-USDT` |

### Risk Settings (bydfi_config.yaml)

| Setting | Default | Description |
|---------|---------|-------------|
| Daily Loss Limit | $500 | Maximum daily loss |
| Max Drawdown | 12% | Maximum drawdown |
| Position Close Time | 21:00 UTC | Close all positions |
| Dollar Bar Threshold | $10M | Aggregation threshold |

---

## 🔧 Troubleshooting

### "Invalid API credentials"
- Verify both credentials (key and secret)
- Check for extra whitespace in .env.bydfi
- Ensure API key has "Read" and "Spot Trading" permissions

### "Symbol not found"
- Use dash format: `BTC-USDT` (not `BTCUSDT`)
- Check symbol is supported by BYDFI

### "Rate limit exceeded"
- Add delays between requests: `await asyncio.sleep(0.1)`
- Check circuit breaker is working

---

## 📚 Documentation

- **Full Summary**: `docs/bydfi-implementation-summary.md`
- **API Docs**: https://developers.bydfi.com/en/public
- **API Management**: https://www.bydfi.com/en/user/api-management

---

## ✅ Checklist

- [ ] BYDFI account created
- [ ] API key generated
- [ ] `.env.bydfi` configured
- [ ] Integration tests passing
- [ ] Historical data acquired
- [ ] Paper trading started

---

**Questions?** Check `docs/bydfi-implementation-summary.md` for detailed documentation.
