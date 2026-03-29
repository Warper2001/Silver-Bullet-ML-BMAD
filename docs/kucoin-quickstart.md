# KuCoin Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Get KuCoin API Credentials (2 minutes)

1. Go to https://www.kucoin.com
2. Create account (or login)
3. Navigate to **API → API Management**
4. Click **Create API Key**
5. Set permissions: **Spot Trading** (read-only for paper trading)
6. **IMPORTANT**: Create a passphrase and save it immediately
7. Copy and save:
   - API Key
   - API Secret
   - Passphrase

### 2. Configure Environment (1 minute)

```bash
# Copy example file
cp .env.kucoin.example .env.kucoin

# Edit with your credentials
nano .env.kucoin
```

Fill in your credentials:
```bash
KUCOIN_API_KEY=your_actual_key_here
KUCOIN_API_SECRET=your_actual_secret_here
KUCOIN_API_PASSPHRASE=your_actual_passphrase_here
KUCOIN_ENVIRONMENT=production
KUCOIN_TRADING_SYMBOL=BTC-USDT
```

### 3. Test Setup (1 minute)

```bash
python tests/integration/test_kucoin_implementation.py
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
python start_kucoin_paper_trading.py
```

---

## 📁 File Structure

```
Silver-Bullet-ML-BMAD/
├── src/
│   ├── data/
│   │   └── kucoin_config.py          # Configuration loader
│   └── execution/
│       └── kucoin/
│           ├── auth/
│           │   └── signature.py       # HMAC SHA256 authentication
│           ├── client.py              # REST API client
│           ├── market_data/
│           │   └── streaming.py       # WebSocket client
│           └── orders/
│               └── submission.py      # Order execution
├── scripts/
│   └── acquire_kucoin_historical_data.py
├── start_kucoin_paper_trading.py
├── kucoin_config.yaml
├── .env.kucoin.example
└── .env.kucoin                        # Your credentials (create this)
```

---

## 🧪 Common Commands

### Test Connectivity
```bash
python tests/integration/test_kucoin_implementation.py
```

### Acquire Historical Data
```bash
python scripts/acquire_kucoin_historical_data.py \
    --symbol BTC-USDT \
    --interval 5m \
    --days 365
```

### Start Paper Trading
```bash
python start_kucoin_paper_trading.py
```

---

## ⚙️ Configuration

### Environment Variables (.env.kucoin)

| Variable | Description | Example |
|----------|-------------|---------|
| `KUCOIN_API_KEY` | Your API key | `6abc123...` |
| `KUCOIN_API_SECRET` | Your API secret | `xyz789...` |
| `KUCOIN_API_PASSPHRASE` | Your passphrase | `myPass123` |
| `KUCOIN_ENVIRONMENT` | Environment | `production` or `sandbox` |
| `KUCOIN_TRADING_SYMBOL` | Trading pair | `BTC-USDT` |

### Risk Settings (kucoin_config.yaml)

| Setting | Default | Description |
|---------|---------|-------------|
| Daily Loss Limit | $500 | Maximum daily loss |
| Max Drawdown | 12% | Maximum drawdown |
| Position Close Time | 21:00 UTC | Close all positions |
| Dollar Bar Threshold | $10M | Aggregation threshold |

---

## 🔧 Troubleshooting

### "Invalid API credentials"
- Verify all three credentials (key, secret, passphrase)
- Check for extra whitespace in .env.kucoin
- Ensure API key has "Spot Trading" permissions

### "Symbol not found"
- Use dash format: `BTC-USDT` (not `BTCUSDT`)
- Check symbol is supported by KuCoin

### "Rate limit exceeded"
- Add delays between requests: `await asyncio.sleep(0.1)`
- Check circuit breaker is working

---

## 📚 Documentation

- **Full Summary**: `docs/kucoin-implementation-summary.md`
- **API Docs**: https://docs.kucoin.com/
- **Sandbox**: https://sandbox.kucoin.com

---

## ✅ Checklist

- [ ] KuCoin account created
- [ ] API key generated (with passphrase)
- [ ] `.env.kucoin` configured
- [ ] Integration tests passing
- [ ] Historical data acquired
- [ ] Paper trading started

---

**Questions?** Check `docs/kucoin-implementation-summary.md` for detailed documentation.
