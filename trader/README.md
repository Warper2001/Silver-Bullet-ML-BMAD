# Trading System - Promotion Pipeline

## The Philosophy

This system operates as a continuous improvement machine. Every strategy
must EARN its way to live capital through a documented evidence trail.
No shortcuts. No gut feelings. Data decides.

## Promotion Stages

```
IDEA
  |-- hypothesis documented in strategy_registry.json
  |-- I (the AI) add ideas here autonomously based on research
  |
  v
BACKTESTING
  |-- run on 5-min MNQ historical data (1-min PROVEN catastrophic)
  |-- minimum 30 trades, 55%+ win rate, profit factor >= 1.5
  |-- walk-forward validation required (no in-sample overfitting)
  |
  v
BACKTEST_PASS
  |-- all criteria met, ready for paper trading
  |
  v
PAPER_TRADING   <-- currently here for Silver Bullet TIER2
  |-- running on TradeStation SIM account with live market data
  |-- REAL prices, ZERO real capital at risk
  |-- minimum: 20 trades, 10 days, 55% WR, PF 1.5, Sharpe 1.0
  |-- ZERO days hitting the $500 daily loss limit
  |
  v
AWAITING_LIVE
  |-- ALL paper criteria met
  |-- *** USER MUST EXPLICITLY APPROVE ***
  |-- Run: python trader/promotion_engine.py approve <strategy_id>
  |
  v
LIVE
  |-- Real capital. $20k allocated. $500/day max loss.
  |-- Continuous monitoring. Auto-retire if underperforms.
```

## Key Commands

```bash
# Check system status (who's winning, who's not ready)
cd /root/Silver-Bullet-ML-BMAD
.venv/bin/python trader/promotion_engine.py status

# Update paper metrics manually
.venv/bin/python trader/promotion_engine.py update-metrics <id> win_rate=0.62 trades=25

# Promote idea to backtesting
.venv/bin/python trader/promotion_engine.py backtest <idea_id>

# Approve live trading (YOU must run this - I cannot)
.venv/bin/python trader/promotion_engine.py approve <strategy_id>

# Run performance tracker now
.venv/bin/python trader/performance_tracker.py

# Run learning engine now (retrain + report)
.venv/bin/python trader/learning_engine.py
```

## Automated Jobs (Cron)

- Every 4 hours: performance_tracker.py - reads logs, updates metrics
- Weekdays 5pm: daily summary report
- Mondays 9am: weekly learning engine + model retrain

## The ONE Thing You Must Do

When a strategy reaches AWAITING_LIVE, you will be notified.
Review the paper trading record and run:

  python trader/promotion_engine.py approve <strategy_id>

That is the ONLY action I need from you to go live.
Everything else I handle autonomously.

## Files

- strategy_registry.json  - source of truth for all strategy states
- promotion_engine.py     - lifecycle management, approval gate
- performance_tracker.py  - log parsing, metrics computation, alerts
- learning_engine.py      - weekly retrain, backtest, improvement reports
- reports/                - generated learning reports (timestamped)
