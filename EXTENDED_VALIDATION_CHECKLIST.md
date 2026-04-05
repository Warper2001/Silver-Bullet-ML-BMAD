# Extended Paper Trading Validation Checklist

**Purpose:** 8-week extended validation checklist for go/no-go decision
**Start Date:** TBD (when WebSocket connection established)
**End Date:** TBD (8 weeks after start date)
**Decision:** PROCEED / CAUTION / DO_NOT_PROCEED (live trading)

---

## Quick Reference

**Dashboard:** http://localhost:8501
**Logs:** `tail -f logs/paper_trading.log`
**Status:** `./deploy_paper_trading.sh status`
**Start:** `./deploy_paper_trading.sh start`
**Stop:** `./deploy_paper_trading.sh stop`
**Health Check:** `python scripts/health_check.py`
**WebSocket Diagnostics:** `python scripts/diagnose_websocket.py`

**Key Limits:**
- Daily Loss Limit: $500 USD
- Max Drawdown: 12%
- Max Position Size: 5 contracts
- Per-Trade Risk: 2% of equity

**Target Metrics:**
- Win Rate: ≥ 55%
- Profit Factor: ≥ 1.5
- Sharpe Ratio: ≥ 1.5
- Trade Frequency: ≥ 2/day

---

## Week 0: Pre-Validation Preparation

### Prerequisites ✅

- [ ] Read DEPLOYMENT.md (deployment guide)
- [ ] Read OPERATIONS.md (operations guide)
- [ ] Read Epic 4 Completion Report
- [ ] Read Epic 4 Retrospective

### Setup Tasks

- [ ] TradeStation SIM credentials configured in `.env`
- [ ] Run WebSocket diagnostic: `python scripts/diagnose_websocket.py`
- [ ] All diagnostic tests pass ✅
- [ ] System deployed: `./deploy_paper_trading.sh start`
- [ ] Health check passes: `python scripts/health_check.py`
- [ ] Dashboard accessible: http://localhost:8501
- [ ] Risk limits verified in dashboard
- [ ] ML models present: `ls models/xgboost/5_minute/`

### Baseline Documentation

- [ ] Starting equity recorded: $________
- [ ] Start date: ___________
- [ ] Expected end date: ___________
- [ ] Weekly check-in time: ___________ (day/time)

---

## Week 1-2: Initial Validation

### Goals
- Verify system stability
- Confirm all components working
- Establish baseline metrics

### Daily Checklist (Weekdays Only)

**Morning (9:00 - 9:25 AM ET):**
- [ ] System running: `./deploy_paper_trading.sh status`
- [ ] Review overnight logs: `grep ERROR logs/paper_trading.log`
- [ ] Check WebSocket connected
- [ ] Dashboard loads: http://localhost:8501
- [ ] No open positions (or existing positions OK)

**During Market Hours (9:30 AM - 4:00 PM ET):**
- [ ] Check dashboard every 30 minutes
- [ ] Monitor account equity
- [ ] Check for signals generated
- [ ] Verify no ERROR messages

**End of Day (4:00 - 5:00 PM ET):**
- [ ] Count trades today: _____
- [ ] Today's P&L: +$_____ or -$_____
- [ ] Current drawdown: _____%
- [ ] Generate daily report: `python src/monitoring/daily_report_generator.py`
- [ ] Document observations in daily log

### Weekly Tasks

**End of Week 1:**
- [ ] Total trades this week: _____
- [ ] Weekly win rate: _____%
- [ ] Weekly P&L: +$_____ or -$_____
- [ ] Max drawdown this week: _____%
- [ ] Any risk events? Yes/No
- [ ] System stability summary: _________________________________

**End of Week 2:**
- [ ] Total trades this week: _____
- [ ] Weekly win rate: _____%
- [ ] Weekly P&L: +$_____ or -$_____
- [ ] Max drawdown this week: _____%
- [ ] Cumulative trades (Week 1+2): _____
- [ ] Cumulative win rate: _____%
- [ ] Any issues? Yes/No

### Week 1-2 Success Criteria

- [ ] System running continuously
- [ ] WebSocket connected throughout
- [ ] Market data flowing (messages_received > 0)
- [ ] Dashboard updating correctly
- [ ] No critical errors
- [ ] Daily reports generating

---

## Week 3-4: Data Collection

### Goals
- Collect ≥ 50 trades total
- Monitor performance metrics
- Track drawdown behavior
- Validate risk management

### Daily Checklist (Same as Week 1-2)

**Additional Tasks:**
- [ ] Track running trade count
- [ ] Note any consecutive losses
- [ ] Monitor drawdown progression
- [ ] Check which strategies trigger

### Weekly Tasks

**End of Week 3:**
- [ ] Total trades to date: _____
- [ ] Running win rate: _____%
- [ ] Current drawdown: _____%
- [ ] Any risk limit breaches? Yes/No
- [ ] Strategy performance summary:
  - Triple Confluence: _____ trades, _____% win rate
  - Wolf Pack: _____ trades, _____% win rate
  - Adaptive EMA: _____ trades, _____% win rate
  - VWAP Bounce: _____ trades, _____% win rate
  - Opening Range: _____ trades, _____% win rate

**End of Week 4:**
- [ ] Total trades to date: _____ (target: ≥ 50)
- [ ] Running win rate: _____% (target: ≥ 55%)
- [ ] Profit factor: _____ (target: ≥ 1.5)
- [ ] Max drawdown: _____% (target: ≤ 12%)
- [ ] Sharpe ratio: _____ (target: ≥ 1.5)
- [ ] On track for go/no-go? Yes/No/Maybe

### Week 3-4 Success Criteria

- [ ] ≥ 50 trades collected
- [ ] Win rate ≥ 55%
- [ ] Max drawdown ≤ 12%
- [ ] Sharpe ratio ≥ 1.5
- [ ] No critical risk events

---

## Week 5-6: Analysis

### Goals
- Analyze collected data
- Compare to backtest expectations
- Validate parameter stability
- Assess regime-specific performance

### Tasks

**Data Analysis:**
- [ ] Calculate parameter stability score
- [ ] Calculate performance stability score
- [ ] Compare to Epic 3 backtest results:
  - Backtest win rate: 60.7%
  - Paper trading win rate: _____%
  - Difference: _____%
  - Explainable? Yes/No

**Regime Analysis:**
- [ ] Performance in trending markets: _____%
  - Number of trending days: _____
  - Win rate: _____%
  - Profit factor: _____
- [ ] Performance in ranging markets: _____%
  - Number of ranging days: _____
  - Win rate: _____%
  - Profit factor: _____
- [ ] Performance in volatile markets: _____%
  - Number of volatile days: _____
  - Win rate: _____%
  - Profit factor: _____

**Strategy Analysis:**
- [ ] Which strategy performed best? _________________
- [ ] Which strategy performed worst? _________________
- [ ] Ensemble benefit quantified? Yes/No
- [ ] Diversification value demonstrated? Yes/No

**Risk Analysis:**
- [ ] Number of risk events: _____
- [ ] Type of risk events:
  - Daily loss breaches: _____
  - Drawdown breaches: _____
  - Emergency stops: _____
  - Other: _____
- [ ] Risk management working as expected? Yes/No

### Week 5-6 Success Criteria

- [ ] Performance matches backtest (± 10%)
- [ ] Parameter stability ≥ 0.65
- [ ] Performance stability ≥ 0.65
- [ ] Ensemble outperforms individuals
- [ ] No major anomalies

---

## Week 7-8: Final Assessment

### Goals
- Make final go/no-go decision
- Generate comprehensive report
- Document lessons learned
- Plan next steps

### Final Performance Metrics

**Overall Metrics:**
- [ ] Total trades: _____
- [ ] Total win rate: _____% (target: ≥ 55%)
- [ ] Profit factor: _____ (target: ≥ 1.5)
- [ ] Max drawdown: _____% (target: ≤ 12%)
- [ ] Sharpe ratio: _____ (target: ≥ 1.5)
- [ ] Total P&L: +$_____ or -$_____
- [ ] Average P&L per trade: +$_____

**Stability Metrics:**
- [ ] Parameter stability score: _____ (target: ≥ 0.65)
- [ ] Performance stability score: _____ (target: ≥ 0.65)
- [ ] Win rate std dev: _____%
- [ ] Weekly consistency: _____%

**Risk Metrics:**
- [ ] Daily loss limit breaches: _____
- [ ] Max drawdown breaches: _____
- [ ] Emergency stops: _____
- [ ] Circuit breaker triggers: _____
- [ ] Per-trade risk violations: _____

### Go/No-Go Decision Framework

**Criteria Assessment:**

| Criteria | Result | Threshold | Pass/Fail |
|----------|--------|-----------|-----------|
| Win Rate | _____% | ≥ 55% | [ ] |
| Max Drawdown | _____% | ≤ 12% | [ ] |
| Sharpe Ratio | _____ | ≥ 1.5 | [ ] |
| Consistent Performance | Yes/No | Yes | [ ] |
| No Critical Risk Events | Yes/No | Yes | [ ] |
| Parameter Stability | _____ | ≥ 0.65 | [ ] |
| Performance Stability | _____ | ≥ 0.65 | [ ] |

**Total Passing: _____ / 7**

**Decision:**

**PROCEED** if:
- [ ] All 7 criteria pass
- [ ] 6/7 criteria pass with no critical failures
- [ ] Confidence: HIGH

**CAUTION** if:
- [ ] 4-5 criteria pass
- [ ] Some concerns but no deal-breakers
- [ ] Confidence: MEDIUM
- [ ] Extended validation recommended

**DO_NOT_PROCEED** if:
- [ ] < 4 criteria pass
- [ ] Critical failures present
- [ ] Confidence: LOW
- [ ] Major issues to address

### Final Report Sections

- [ ] Executive Summary
- [ ] Performance Analysis (8 weeks)
- [ ] Comparison to Backtest
- [ ] Risk Assessment
- [ ] Strategy Breakdown
- [ ] Regime Analysis
- [ ] Stability Analysis
- [ ] Lessons Learned
- [ ] Recommendations
- [ ] Next Steps

### Decision Rationale

**If PROCEED:**
- Rationale: _________________________________________________
- Confidence Level: HIGH/MEDIUM/LOW
- Next Steps:
  - [ ] Create live trading deployment plan
  - [ ] Implement production safeguards
  - [ ] Train operations team
  - [ ] Set go-live date: ___________

**If CAUTION:**
- Rationale: _________________________________________________
- Concerns:
  1. _________________________________________________
  2. _________________________________________________
  3. _________________________________________________
- Recommendations:
  - [ ] Extend validation by _____ weeks
  - [ ] Address specific concerns: ______________________
  - [ ] Re-evaluate on: ___________

**If DO_NOT_PROCEED:**
- Rationale: _________________________________________________
- Deal-Breakers:
  1. _________________________________________________
  2. _________________________________________________
  3. _________________________________________________
- Recommendations:
  - [ ] Major issues to address
  - [ ] Return to development
  - [ ] Re-validate after fixes

---

## Daily Log Template

```markdown
## Daily Log: YYYY-MM-DD

**Market Day #: _____ (of ~40 trading days in 8 weeks)**

**Market Conditions:**
- Regime: [ ] Trending [ ] Ranging [ ] Volatile
- News Events: _________________
- Volatility: High/Medium/Low

**System Status:**
- System Running: Yes/No
- WebSocket Connected: Yes/No
- Dashboard Accessible: Yes/No

**Trading Activity:**
- Trades Today: _____
- Signals Generated: _____
- Signals Executed: _____
- Signals Rejected: _____

**Performance:**
- Today's P&L: +$_____ or -$_____
- Cumulative P&L: +$_____ or -$_____
- Current Drawdown: _____%
- Win Rate (Today): _____%
- Win Rate (Cumulative): _____%

**Positions:**
- Open Positions (EOD): _____
- Closed Today: _____
- Winning Trades: _____
- Losing Trades: _____

**Risk Events:**
- Daily Loss Breach: Yes/No
- Drawdown Breach: Yes/No
- Emergency Stop: Yes/No
- Other: _________________

**Observations:**
-
-
-

**Issues/Anomalies:**
-
-
-

**Action Items for Tomorrow:**
- [ ]
- [ ]
```

---

## Weekly Summary Template

```markdown
## Weekly Summary: Week # (YYYY-MM-DD to YYYY-MM-DD)

**Performance Summary:**
- Total Trades This Week: _____
- Cumulative Trades: _____
- Weekly Win Rate: _____%
- Cumulative Win Rate: _____%
- Weekly P&L: +$_____ or -$_____
- Cumulative P&L: +$_____ or -$_____
- Max Drawdown This Week: _____%
- Current Drawdown: _____%

**Strategy Breakdown:**
- Triple Confluence: _____ trades, _____% win rate
- Wolf Pack: _____ trades, _____% win rate
- Adaptive EMA: _____ trades, _____% win rate
- VWAP Bounce: _____ trades, _____% win rate
- Opening Range: _____ trades, _____% win rate

**Market Conditions:**
- Days Trending: _____
- Days Ranging: _____
- Days Volatile: _____
- Major News Events: _________________

**Risk Events:**
- Daily Loss Breaches: _____
- Drawdown Breaches: _____
- Emergency Stops: _____
- Other: _________________

**Observations:**
- What went well: _________________
- What didn't go well: _________________
- Anomalies: _________________
- Learnings: _________________

**Action Items:**
- [ ]
- [ ]
- [ ]

**Next Week Focus:**
- _________________
- _________________
```

---

## Quick Troubleshooting

**System not running?**
```bash
./deploy_paper_trading.sh status
./deploy_paper_trading.sh start
```

**WebSocket not connected?**
```bash
python scripts/diagnose_websocket.py
./deploy_paper_trading.sh restart
```

**Dashboard not loading?**
```bash
ps aux | grep streamlit
pkill -f streamlit
./deploy_paper_trading.sh restart
```

**System halted (risk limit)?**
```bash
grep HALT logs/paper_trading.log
tail -1 logs/daily_loss.csv
tail -1 logs/drawdown.csv
# Review what caused breach before resetting
```

**No trades generating?**
- Check market is open (Mon-Fri 9:30 AM - 4:00 PM ET)
- Check signals page in dashboard
- Review probability threshold (may be too high)
- Consider lowering threshold if consistent issue

**High memory usage?**
```bash
du -sh logs/paper_trading.log
# If > 100MB, rotate logs
mv logs/paper_trading.log logs/backup/paper_trading_$(date +%Y%m%d).log
./deploy_paper_trading.sh restart
```

---

## Important Reminders

**DO:**
✅ Monitor dashboard daily
✅ Review logs for errors
✅ Document observations
✅ Track performance metrics
✅ Investigate warnings
✅ Learn from mistakes

**DON'T:**
❌ Reset risk limits without understanding cause
❌ Increase risk limits during drawdown
❌ Disable risk layers
❌ Ignore warning signs
❌ Skip documentation
❌ Make emotional decisions

---

## Support & Resources

**Documentation:**
- DEPLOYMENT.md - Deployment guide
- OPERATIONS.md - Operations guide
- EPIC4_COMPLETION_REPORT.md - Epic 4 summary
- epic-4-retrospective.md - Lessons learned

**Scripts:**
- `scripts/health_check.py` - System health check
- `scripts/diagnose_websocket.py` - WebSocket diagnostics
- `src/monitoring/daily_report_generator.py` - Daily reports
- `scripts/weekly_analysis.py` - Weekly reports

**Commands:**
- `./deploy_paper_trading.sh {start|stop|status|restart|validate}`

**Log Files:**
- `logs/paper_trading.log` - Main system log
- `logs/daily_loss.csv` - Daily loss tracking
- `logs/drawdown.csv` - Drawdown tracking
- `logs/per_trade_risk.csv` - Per-trade risk
- `logs/emergency_stop.csv` - Emergency stops

---

**Checklist Version:** 1.0
**Created:** 2026-04-04
**Purpose:** Guide 8-week extended paper trading validation
**Goal:** Collect data for go/no-go live trading decision
