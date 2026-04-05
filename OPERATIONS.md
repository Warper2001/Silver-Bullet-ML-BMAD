# Paper Trading Operations Guide

**Last Updated:** 2026-04-04
**Epic:** Epic 4 - Paper Trading Deployment
**Environment:** TradeStation SIM (Paper Trading)
**Purpose:** Daily operations and monitoring for extended paper trading validation

---

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Weekly Procedures](#weekly-procedures)
3. [Dashboard Usage](#dashboard-usage)
4. [Monitoring & Alerts](#monitoring--alerts)
5. [Risk Management](#risk-management)
6. [Performance Analysis](#performance-analysis)
7. [Emergency Procedures](#emergency-procedures)
8. [Extended Validation Checklist](#extended-validation-checklist)

---

## Daily Operations

### Morning Startup (Before Market Open)

**Time:** 9:00 AM - 9:25 AM ET (30 minutes before market open)

#### Step 1: System Health Check

```bash
# Check if system is running
./deploy_paper_trading.sh status

# Expected output: RUNNING
# If not running, start: ./deploy_paper_trading.sh start
```

#### Step 2: Review Overnight Logs

```bash
# Check for errors or warnings overnight
grep -E "ERROR|WARNING" logs/paper_trading.log | tail -20

# Check for risk events
cat logs/emergency_stop.csv
cat logs/daily_loss.csv
cat logs/drawdown.csv
```

#### Step 3: Verify WebSocket Connection

```bash
# Check WebSocket is connected
grep "WebSocket connected" logs/paper_trading.log | tail -1

# Verify market data flowing
grep "messages_received" logs/paper_trading.log | tail -5
```

#### Step 4: Check Dashboard

1. Open: http://localhost:8501
2. Review **Overview** page:
   - Account equity (should be $50,000 or current value)
   - Daily P&L (should be $0.00 at start of day)
   - Drawdown (should be 0% or current drawdown)
   - System uptime
3. Review **Positions** page:
   - Should show no open positions (or existing positions from overnight)
   - Verify no stale positions

#### Step 5: Reset Daily Limits (If Applicable)

If daily loss limit was breached previous day:

```bash
# Check daily loss
tail -1 logs/daily_loss.csv

# If loss is negative (breached), reset for new day
# Edit CSV and set current_loss to 0
# Or wait for automatic reset at midnight
```

**⚠️ CAUTION:** Only reset if you understand root cause of previous day's breach!

### During Market Hours (9:30 AM - 4:00 PM ET)

#### Continuous Monitoring

**Dashboard Checks (Every 30 minutes):**

1. **Overview Page:**
   - Account equity updating
   - Daily P&L changing
   - Drawdown within limits (< 12%)
   - No error indicators

2. **Positions Page:**
   - Open positions with real-time P&L
   - Stop-loss and take-profit levels
   - Position duration

3. **Signals Page:**
   - New signals appearing (expect 0-5 per day)
   - Signal confidence levels (should be ≥ 65%)
   - Which strategies triggered

4. **Logs Page:**
   - No ERROR messages
   - Normal trade flow messages
   - Risk validation messages

**Log Monitoring (Real-time):**

```bash
# Follow logs in real-time
tail -f logs/paper_trading.log | grep -E "ERROR|WARNING|Signal|Trade|Risk"

# Press Ctrl+C to stop following
```

### End of Day (4:00 PM - 5:00 PM ET)

#### Step 1: Generate Daily Report

```bash
# Run daily report generator
.venv/bin/python src/monitoring/daily_report_generator.py
```

This generates:
- Daily performance summary
- Win rate, profit factor, Sharpe ratio
- Trade-by-trade breakdown
- Risk metrics

Output saved to: `data/reports/daily_report_YYYY-MM-DD.md`

#### Step 2: Review Day's Performance

**Manual Checklist:**

- [ ] Total trades executed (target: 0-5 per day)
- [ ] Win rate for the day
- [ ] Total P&L for the day
- [ ] Maximum drawdown during day
- [ ] Any risk events (loss limit, drawdown breaches)
- [ ] Which strategies performed best
- [ ] Any anomalies or unexpected behavior

**Document observations:**

Create daily log entry:

```markdown
## YYYY-MM-DD Daily Log

**Market Conditions:** [Trending/Ranging/Volatile]
**Total Trades:** X
**Win Rate:** X%
**Daily P&L:** +$X.XX or -$X.XX
**Max Drawdown:** X%

**Observations:**
- Strategy ABC performed well because...
- Strategy XYZ underperformed due to...
- Market conditions affected...
- Anomalies noticed...

**Action Items (if any):**
- [ ] Follow up on...
```

#### Step 3: Backup Data

```bash
# Backup log files (if not automated)
cp logs/paper_trading.log logs/backup/paper_trading_$(date +%Y%m%d).log

# Backup state files
cp data/state/weight_history.csv data/backup/weight_history_$(date +%Y%m%d).csv
```

#### Step 4: Verify CSV Audit Trails

```bash
# Check CSV files are being updated
tail -5 logs/daily_loss.csv
tail -5 logs/drawdown.csv
tail -5 logs/per_trade_risk.csv

# Verify no missing data
wc -l logs/*.csv
```

#### Step 5: Prepare for Tomorrow

- Document any issues encountered
- Note any configuration changes needed
- Review tomorrow's economic calendar (news events)
- Plan any system adjustments

---

## Weekly Procedures

### Day of Week: Monday (Start of Week)

#### Weekly System Review

```bash
# Check system uptime
./deploy_paper_trading.sh status

# Review logs from weekend
grep -E "ERROR|WARNING" logs/paper_trading.log | grep "$(date +%Y-%m-%d)..."

# Verify disk space
df -h .
```

#### Weekly Weight Rebalancing Check

Weight optimization runs automatically every Sunday night. Check results:

```bash
# Check weight history
tail -10 data/state/weight_history.csv

# View weights in dashboard
# Open http://localhost:8501 → Settings → Ensemble Weights
```

**What to look for:**
- Weights changing (based on 4-week performance)
- Any strategy hitting floor (5%) or ceiling (40%)
- Consistent weight trends

### Day of Week: Friday (End of Week)

#### Generate Weekly Performance Report

```bash
# Run weekly analysis
.venv/bin/python scripts/weekly_analysis.py
```

This generates:
- Weekly win rate, profit factor, Sharpe
- Trade frequency analysis
- Strategy performance comparison
- Regime analysis (trend/range/volatile)
- Drawdown analysis

#### Weekly Performance Review

**Metrics to Track:**

| Metric | This Week | Last Week | Target | Status |
|--------|-----------|-----------|--------|--------|
| Total Trades | X | X | 10-20 | ✅/❌ |
| Win Rate | X% | X% | ≥ 55% | ✅/❌ |
| Profit Factor | X.X | X.X | ≥ 1.5 | ✅/❌ |
| Max Drawdown | X% | X% | ≤ 12% | ✅/❌ |
| Sharpe Ratio | X.X | X.X | ≥ 1.5 | ✅/❌ |
| Daily P&L | +$X | +$X | Positive | ✅/❌ |

**Weekly Analysis Questions:**

1. **Performance vs Expectations:**
   - Did system meet weekly targets?
   - How does this week compare to last week?
   - Any performance degradation or improvement?

2. **Trade Frequency:**
   - Are we getting enough trades (target: ≥ 2/day)?
   - If not, why? (Market conditions? Strategy constraints?)
   - Should probability threshold be adjusted?

3. **Strategy Performance:**
   - Which strategies performed best/worst?
   - Is weight optimization working as expected?
   - Any strategies consistently underperforming?

4. **Risk Events:**
   - Any risk limit breaches this week?
   - Emergency stop triggered?
   - Drawdown within acceptable limits?

5. **Market Regimes:**
   - What type of week? (Trending/Ranging/Volatile)
   - How did system perform in each regime?
   - Any regime-specific issues?

#### Document Weekly Findings

Create weekly summary:

```markdown
## Weekly Summary: YYYY-MM-DD to YYYY-MM-DD

**Performance:**
- Total Trades: X
- Win Rate: X% (target: ≥ 55%)
- Profit Factor: X.X (target: ≥ 1.5)
- Max Drawdown: X% (target: ≤ 12%)
- Sharpe Ratio: X.X (target: ≥ 1.5)

**Strategy Breakdown:**
- Triple Confluence: X trades, X% win rate
- Wolf Pack: X trades, X% win rate
- Adaptive EMA: X trades, X% win rate
- VWAP Bounce: X trades, X% win rate
- Opening Range: X trades, X% win rate

**Market Conditions:**
- Regime: [Trending/Ranging/Volatile]
- Volatility: [High/Medium/Low]
- News Events: [List major news]

**Risk Events:**
- [ ] Daily loss limit breached
- [ ] Max drawdown breached
- [ ] Emergency stop triggered
- [ ] Other risk events

**Observations:**
- What went well this week?
- What didn't go well?
- Any anomalies or unexpected behavior?

**Action Items:**
- [ ] Item 1
- [ ] Item 2

**Next Week Focus:**
- [ ] Focus area 1
- [ ] Focus area 2
```

---

## Dashboard Usage

### Dashboard Pages Overview

Access dashboard: **http://localhost:8501**

The dashboard auto-refreshes every 2 seconds.

#### Page 1: Overview

**Purpose:** High-level system status

**Key Metrics:**
- **Account Equity:** Current account value
- **Daily P&L:** Profit/Loss for current day
- **Maximum Drawdown:** Peak-to-trough decline
- **Win Rate:** Overall percentage of winning trades
- **System Uptime:** How long system has been running
- **Open Positions:** Number of current positions

**What to Check:**
- Equity updating (not stale)
- Drawdown < 12%
- No error indicators (red/orange)
- Uptime reasonable (should reset weekly)

#### Page 2: Positions

**Purpose:** Real-time position tracking

**For Each Position:**
- **Symbol:** MNQ (Micro Nasdaq-100)
- **Side:** LONG or SHORT
- **Entry Price:** Price where position entered
- **Current Price:** Live mark-to-market price
- **Stop Loss:** Risk barrier level
- **Take Profit:** Reward barrier level
- **Unrealized P&L:** Current profit/loss
- **Duration:** How long position open

**Actions:**
- **Manual Exit:** Click to exit position immediately
- **View Details:** Expand to see trade details

**What to Check:**
- Positions updating in real-time
- P&L calculating correctly
- Stop/take profit levels set correctly
- No stale positions (should be closed if stop/TP hit)

#### Page 3: Signals

**Purpose:** View Silver Bullet signals generated

**Signal Information:**
- **Timestamp:** When signal was generated
- **Confidence:** ML success probability (should be ≥ 65%)
- **Side:** LONG or SHORT recommendation
- **Entry Price:** Recommended entry level
- **Stop Loss:** Risk barrier
- **Take Profit:** Reward barrier
- **Strategies:** Which ICT strategies triggered
- **Status:** PENDING, EXECUTED, REJECTED, FILTERED

**Filtering:**
- Filter by confidence level (≥ 65%, ≥ 70%, ≥ 75%)
- Filter by status (only show executed, only show pending)
- Filter by time range (today, last 7 days, last 30 days)

**What to Check:**
- New signals appearing (expect 0-5 per day)
- Confidence levels ≥ 65%
- Status updates (PENDING → EXECUTED or REJECTED)
- Which strategies are triggering

#### Page 4: Charts

**Purpose:** Visualize performance and metrics

**Available Charts:**
- **Equity Curve:** Account equity over time
- **Daily P&L:** Profit/Loss per day
- **Drawdown:** Drawdown percentage over time
- **Win Rate:** Rolling win rate (7-day, 14-day, 30-day)
- **Trade Frequency:** Trades per day

**Time Range Selection:**
- Last 7 days
- Last 30 days
- Last 90 days
- Custom range

**What to Check:**
- Equity trending upward (or stable)
- Drawdown recovering (not continuously declining)
- Win rate ≥ 55% (rolling average)
- Trade frequency ≥ 2/day (on average)

#### Page 5: Settings

**Purpose:** View and modify configuration

**Sections:**

**Risk Limits:**
- Daily Loss Limit ($500)
- Max Drawdown (12%)
- Max Position Size (5 contracts)
- Per-Trade Risk (2%)

⚠️ **Password protected** - requires admin password to modify

**Ensemble Weights:**
- Weight for each of 5 strategies
- Auto-optimizes weekly based on performance
- Sum to 100%

⚠️ **Password protected** - requires admin password to modify

**ML Thresholds:**
- Probability threshold (default: 0.65)
- Lower threshold = more signals (but lower quality)
- Higher threshold = fewer signals (but higher quality)

⚠️ **Password protected** - requires admin password to modify

**What to Check:**
- Risk limits set correctly
- Weights sum to 100%
- Probability threshold ≥ 0.65

#### Page 6: Logs

**Purpose:** View system logs in real-time

**Features:**
- Real-time log streaming
- Filter by log level (DEBUG, INFO, WARNING, ERROR)
- Search for specific terms
- Export logs

**What to Check:**
- No ERROR messages
- No excessive WARNING messages
- Normal trade flow (messages received, bars created, signals generated)
- Risk validation messages (per-trade risk checks)

---

## Monitoring & Alerts

### Key Metrics to Monitor

#### Performance Metrics

| Metric | Target | Alert Threshold | Status |
|--------|--------|-----------------|--------|
| Win Rate | ≥ 55% | < 50% | 🟡 Warning |
| Profit Factor | ≥ 1.5 | < 1.2 | 🟡 Warning |
| Sharpe Ratio | ≥ 1.5 | < 1.0 | 🟡 Warning |
| Max Drawdown | ≤ 12% | > 10% | 🟡 Warning |
| Trade Frequency | ≥ 2/day | < 1/day | 🟡 Warning |

#### System Health Metrics

| Metric | Target | Alert Threshold | Status |
|--------|--------|-----------------|--------|
| WebSocket Connected | Yes | No | 🔴 Critical |
| Messages Received | > 0/min | 0 for 5 min | 🔴 Critical |
| Dashboard Accessible | Yes | No | 🔴 Critical |
| Process Running | Yes | No | 🔴 Critical |
| Log File Size | < 100MB | > 100MB | 🟡 Warning |

#### Risk Metrics

| Metric | Limit | Alert Threshold | Status |
|--------|-------|-----------------|--------|
| Daily Loss | $500 | > $400 | 🟡 Warning |
| Daily Loss | $500 | ≥ $500 | 🔴 Critical (Auto-Halt) |
| Max Drawdown | 12% | > 10% | 🟡 Warning |
| Max Drawdown | 12% | ≥ 12% | 🔴 Critical (Auto-Halt) |
| Open Positions | 5 | ≥ 4 | 🟡 Warning |

### Alert Actions

**🟡 Warning Alerts:**

1. **Win Rate < 50%:**
   - Check market regime (trending vs ranging)
   - Review which strategies underperforming
   - Consider adjusting probability threshold
   - Document and monitor

2. **Drawdown > 10%:**
   - Review recent trades
   - Check for consecutive losses
   - Verify risk management working
   - Consider reducing position sizes

3. **Trade Frequency < 1/day:**
   - Check if signals generated (Signals page)
   - Verify ML probability threshold not too high
   - Review market conditions
   - Consider lowering threshold if consistent issue

**🔴 Critical Alerts:**

1. **WebSocket Disconnected:**
   ```bash
   # Check connection
   grep "WebSocket" logs/paper_trading.log | tail -5

   # Attempt reconnection
   ./deploy_paper_trading.sh restart
   ```

2. **Daily Loss Limit Breached (System Halted):**
   - This is automatic protection
   - Review what caused losses
   - Do NOT reset without understanding
   - Document root cause

3. **Max Drawdown Breached (System Halted):**
   - This is automatic protection
   - Review drawdown progression
   - Consider systemic issues
   - Do NOT reset without investigation

4. **Process Not Running:**
   ```bash
   # Check status
   ./deploy_paper_trading.sh status

   # Restart if stopped
   ./deploy_paper_trading.sh start
   ```

### Setting Up Alerts

Currently, alerts are viewed in dashboard. For automated alerts, consider:

**Email Alerts (Future Enhancement):**
```yaml
# Add to config.yaml
alerts:
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    from_email: your-email@gmail.com
    to_email: your-email@gmail.com
    # Use app-specific password
    password: your-app-password

  alert_rules:
    - metric: daily_loss
      threshold: 400
      comparison: ">"
      severity: warning

    - metric: drawdown
      threshold: 10
      comparison: ">"
      severity: warning

    - metric: websocket_connected
      threshold: false
      comparison: "=="
      severity: critical
```

---

## Risk Management

### 8-Layer Risk System

The paper trading system has 8 layers of risk protection:

#### Layer 1: Emergency Stop (Manual)

**Purpose:** Manual shutdown capability

**How to Use:**
```bash
# CLI command
.venv/bin/python src/cli/emergency_stop.py --reason "Manual review required"

# Or via dashboard
# Settings → Emergency Stop → Click "STOP TRADING"
```

**When to Use:**
- You notice suspicious behavior
- Major market event (flash crash, news)
- System behaving unexpectedly
- Need to pause for investigation

#### Layer 2: Daily Loss Limit ($500)

**Purpose:** Prevent large daily losses

**How it Works:**
- Tracks cumulative daily P&L
- When loss ≥ $500, system auto-halts
- Requires manual review before reset

**Monitoring:**
```bash
# Check current daily loss
tail -1 logs/daily_loss.csv

# Output: date,current_loss,daily_limit,status
# 2026-04-04,-300,500,OK
```

**What to Do if Breached:**
1. Review trades that caused losses
2. Check for system errors
3. Verify risk management working
4. Document root cause
5. Wait for next day to auto-reset

#### Layer 3: Max Drawdown (12%)

**Purpose:** Limit peak-to-trough decline

**How it Works:**
- Tracks max account equity (peak)
- Calculates drawdown from peak
- When drawdown ≥ 12%, system auto-halts

**Monitoring:**
```bash
# Check current drawdown
tail -1 logs/drawdown.csv

# Output: date,peak_equity,current_equity,drawdown_percent,max_drawdown_percent
# 2026-04-04,50000,47000,6.0,11.8
```

**What to Do if Breached:**
1. Review overall performance
2. Check for strategy degradation
3. Consider parameter adjustments
4. Extended halt may be warranted

#### Layer 4: Max Position Size (5 contracts)

**Purpose:** Limit exposure per position

**How it Works:**
- Before opening trade, checks open position count
- Rejects trade if ≥ 5 positions open
- Prevents over-leveraging

**Monitoring:**
```bash
# Check open positions in dashboard
# Overview page shows: "Open Positions: X/5"
```

#### Layer 5: Circuit Breaker Detector

**Purpose:** Detect abnormal market conditions

**How it Works:**
- Monitors for rapid price moves
- Detects flash crashes
- Halts trading during extreme volatility

**Monitoring:**
```bash
# Check logs for circuit breaker events
grep "Circuit Breaker" logs/paper_trading.log
```

#### Layer 6: News Event Filter

**Purpose:** Avoid trading around major news

**How it Works:**
- Filters signals within 30 minutes of major news
- Prevents volatility-induced losses
- Uses economic calendar

**Monitoring:**
```bash
# Check for news filter events
grep "News Event" logs/paper_trading.log
```

#### Layer 7: Per-Trade Risk Limit (2% of equity)

**Purpose:** Limit risk per individual trade

**How it Works:**
- Calculates position size based on 2% of equity
- Sets stop-loss to limit loss to 2%
- Example: $50,000 equity → max risk $1,000 per trade

**Monitoring:**
```bash
# Check per-trade risk validation
tail -10 logs/per_trade_risk.csv

# Output: timestamp,entry_price,stop_loss,risk_amount,risk_percent,decision
# 2026-04-04 10:30:00,18250.00,18240.00,500.00,1.0,ACCEPT
```

#### Layer 8: Notification Manager

**Purpose:** Alert on risk events

**How it Works:**
- Logs all risk events to CSV files
- Displays alerts in dashboard
- (Future) Send email/SMS alerts

**Monitoring:**
```bash
# Check all risk event logs
ls -lh logs/*.csv
```

### Risk Management Best Practices

**Do:**
- ✅ Monitor dashboard daily
- ✅ Review risk event logs weekly
- ✅ Investigate all warning alerts
- ✅ Use emergency stop if concerned
- ✅ Document risk events
- ✅ Learn from breaches

**Don't:**
- ❌ Reset risk limits without investigation
- ❌ Increase risk limits during drawdown
- ❌ Disable risk layers
- ❌ Ignore warning alerts
- ❌ Override automated decisions

---

## Performance Analysis

### Daily Performance Analysis

At end of each day, analyze:

**1. Trade Count:**
- How many trades today? (Target: 0-5)
- vs expected frequency
- Trend over time

**2. Win Rate:**
- Today's win rate
- Rolling 7-day win rate
- vs 55% target

**3. Profit Factor:**
- Today's profit factor (gross profit / gross loss)
- vs 1.5 target
- Trend over time

**4. Drawdown:**
- Current drawdown
- Max drawdown today
- vs 12% limit

**5. Strategy Breakdown:**
- Which strategies won?
- Which strategies lost?
- Any strategy anomalies?

### Weekly Performance Analysis

At end of each week, generate comprehensive report:

```bash
.venv/bin/python scripts/weekly_analysis.py
```

**Report Contents:**

1. **Performance Summary**
   - Total trades
   - Win rate
   - Profit factor
   - Sharpe ratio
   - Max drawdown
   - Total P&L

2. **Strategy Comparison**
   - Performance by strategy
   - Win rate by strategy
   - Contribution to total P&L
   - Trade frequency by strategy

3. **Regime Analysis**
   - Performance in trending markets
   - Performance in ranging markets
   - Performance in volatile markets
   - Regime robustness score

4. **Ensemble Analysis**
   - Ensemble vs best individual strategy
   - Diversification benefit
   - Signal correlation analysis

5. **Risk Analysis**
   - Risk events summary
   - Drawdown analysis
   - Per-trade risk distribution

### Monthly Performance Analysis

At end of each month:

**1. Compare to Backtest Expectations**
- Backtest win rate: 60.7%
- Paper trading win rate: ?
- Difference explained?

**2. Stability Assessment**
- Parameter stability
- Performance stability
- Consistency across weeks

**3. Go/No-Go Criteria Check**
- [ ] Win rate ≥ 55%
- [ ] Max drawdown ≤ 12%
- [ ] Sharpe ratio ≥ 1.5
- [ ] Consistent performance
- [ ] No critical risk events
- [ ] Parameter stability demonstrated

---

## Emergency Procedures

### Emergency Stop

**When to Use:**
- System behaving erratically
- Flash crash detected
- Major technical issue
- Manual intervention required

**How to Execute:**

**Option 1: CLI**
```bash
.venv/bin/python src/cli/emergency_stop.py --reason "Describe reason"
```

**Option 2: Dashboard**
- Navigate to Settings page
- Click "EMERGENCY STOP" button
- Enter reason
- Confirm

**Option 3: Manual**
```bash
# Find process
ps aux | grep start_paper_trading

# Kill process
kill -9 <PID>

# Or use deployment script
./deploy_paper_trading.sh stop --force
```

### Post-Emergency Procedures

**Immediate Actions:**
1. Verify trading halted
2. Check open positions
3. Review logs for cause
4. Document event

**Investigation:**
1. What triggered emergency stop?
2. Were any trades in progress?
3. Any data corruption?
4. System integrity check?

**Recovery:**
1. Resolve root cause
2. Verify system integrity
3. Restart system (if safe)
4. Monitor closely

### Recovery Procedures

**Scenario 1: System Crash**

```bash
# Check if process running
./deploy_paper_trading.sh status

# If not running, check logs
tail -50 logs/paper_trading.log

# Identify crash cause
grep -E "ERROR|CRITICAL" logs/paper_trading.log

# Fix issue, then restart
./deploy_paper_trading.sh start
```

**Scenario 2: Data Corruption**

```bash
# Check data files
ls -lh data/processed/dollar_bars/
ls -lh data/state/

# If corrupted, restore from backup
cp data/backup/weight_history_YYYYMMDD.csv data/state/weight_history.csv

# Restart system
./deploy_paper_trading.sh restart
```

**Scenario 3: WebSocket Disconnection**

```bash
# Check connection
grep "WebSocket" logs/paper_trading.log | tail -10

# If disconnected, restart
./deploy_paper_trading.sh restart

# Verify reconnection
grep "WebSocket connected" logs/paper_trading.log | tail -1
```

---

## Extended Validation Checklist

This checklist guides you through the 4-8 week extended paper trading validation.

### Week 0: Preparation ✅

- [ ] TradeStation SIM credentials configured in `.env`
- [ ] WebSocket connection tested
- [ ] System deployed and running
- [ ] Dashboard accessible at http://localhost:8501
- [ ] Risk limits verified (daily loss $500, max drawdown 12%)
- [ ] CSV audit trails being written
- [ ] Daily report generator tested
- [ ] Weekly analysis script tested
- [ ] Operations guide reviewed
- [ ] Emergency procedures documented

### Week 1-2: Initial Validation

**Goals:**
- Verify system stability
- Confirm all components working
- Establish baseline metrics
- Document any initial issues

**Daily Tasks:**
- [ ] Morning system health check
- [ ] Monitor dashboard during market hours
- [ ] End-of-day review and log
- [ ] Generate daily report

**Weekly Tasks:**
- [ ] Weekly performance review
- [ ] Check weight optimization
- [ ] Review risk events
- [ ] Document findings

**Success Criteria:**
- ✅ System running continuously
- ✅ WebSocket connected throughout
- ✅ Market data flowing (messages_received > 0)
- ✅ Signals generated (if market conditions allow)
- ✅ No critical errors
- ✅ Dashboard updating correctly

### Week 3-4: Data Collection

**Goals:**
- Collect 100+ trades for statistical significance
- Monitor performance metrics
- Track drawdown behavior
- Validate risk management

**Daily Tasks:**
- [ ] All Week 1-2 tasks
- [ ] Track trade count (target: ≥ 50 trades by end of Week 4)
- [ ] Monitor win rate (target: ≥ 55%)
- [ ] Check drawdown (target: ≤ 12%)
- [ ] Verify risk management working

**Weekly Tasks:**
- [ ] Generate comprehensive weekly report
- [ ] Compare to backtest expectations
- [ ] Analyze strategy performance
- [ ] Review weight optimization

**Success Criteria:**
- ✅ ≥ 50 trades collected
- ✅ Win rate ≥ 55%
- ✅ Max drawdown ≤ 12%
- ✅ Sharpe ratio ≥ 1.5
- ✅ No critical risk events

### Week 5-6: Analysis

**Goals:**
- Analyze collected data
- Compare to backtest expectations
- Validate parameter stability
- Assess regime-specific performance

**Tasks:**
- [ ] Run comprehensive performance analysis
- [ ] Compare paper trading vs backtest metrics
- [ ] Calculate parameter stability scores
- [ ] Analyze performance by market regime
- [ ] Review strategy contributions
- [ ] Validate ensemble benefits
- [ ] Document deviations from expectations

**Success Criteria:**
- ✅ Performance matches backtest (± 10%)
- ✅ Parameter stability ≥ 0.65
- ✅ Performance stability ≥ 0.65
- ✅ Ensemble outperforms individuals
- ✅ No major anomalies

### Week 7-8: Final Assessment

**Goals:**
- Make final go/no-go decision
- Generate comprehensive report
- Document lessons learned
- Plan next steps

**Tasks:**
- [ ] Generate final performance report
- [ ] Apply go/no-go decision framework
- [ ] Compare all 6 criteria to thresholds
- [ ] Assess overall risk level
- [ ] Document recommendations
- [ ] Create live trading plan (if PROCEED)
- [ ] Create mitigation plan (if CAUTION/DO_NOT_PROCEED)

**Go/No-Go Decision:**

**PROCEED if:**
- [ ] Win rate ≥ 55%
- [ ] Max drawdown ≤ 12%
- [ ] Sharpe ratio ≥ 1.5
- [ ] Consistent performance over 4-8 weeks
- [ ] No critical risk events
- [ ] Parameter stability ≥ 0.65
- [ ] Performance stability ≥ 0.65

**CAUTION if:**
- [ ] Win rate 50-55%
- [ ] Max drawdown 12-15%
- [ ] Sharpe ratio 1.0-1.5
- [ ] Some inconsistency in performance
- [ ] Minor risk events occurred
- [ ] Stability scores 0.5-0.65

**DO_NOT_PROCEED if:**
- [ ] Win rate < 50%
- [ ] Max drawdown > 15%
- [ ] Sharpe ratio < 1.0
- [ ] Highly inconsistent performance
- [ ] Critical risk events occurred
- [ ] Risk management failures
- [ ] System doesn't match backtest expectations

### Ongoing: Documentation

Throughout the 8 weeks, maintain:

- [ ] Daily logs (observations, issues)
- [ ] Weekly summaries (performance, analysis)
- [ ] Risk event log (breaches, triggers)
- [ ] Anomaly log (unexpected behavior)
- [ ] Lesson learned log (what to improve)

---

## Summary

This operations guide provides comprehensive procedures for:

1. **Daily Operations:** Morning startup, during-day monitoring, end-of-day review
2. **Weekly Procedures:** System review, performance analysis, weight optimization check
3. **Dashboard Usage:** All 6 pages explained with monitoring guidance
4. **Monitoring & Alerts:** Key metrics, alert thresholds, response procedures
5. **Risk Management:** All 8 risk layers explained with best practices
6. **Performance Analysis:** Daily, weekly, monthly analysis procedures
7. **Emergency Procedures:** Emergency stop, recovery procedures
8. **Extended Validation:** 8-week checklist for go/no-go decision

**Key Success Factors:**
- Consistent daily monitoring
- Thorough documentation
- Proactive risk management
- Data-driven decisions
- Continuous learning

**Next Steps:**
1. Complete Week 0 preparation
2. Start extended paper trading (Week 1-2)
3. Follow daily/weekly procedures
4. Make go/no-go decision at Week 8

---

**Operations Guide Version:** 1.0
**Last Updated:** 2026-04-04
**Maintained By:** Operations Team
