# Approved experiment protocol

- Keep frozen A/B source, protocol, data, orders, services and historical artifacts unchanged. No live changes, new prospective process, sizing changes, paid acquisition, parameter search or filter combinations. Existing history is exposed; results never authorize deployment. Preserve unrelated work.

- R: OLS closes on indices0..29, slope sign matches breakout and R2>=.30. Constant closes fail. E: abs(last-first)/sum(abs(diff))>=.30 and signed displacement matches breakout; zero denominator fails. All windows reset daily; short history blocks entries only.

- P: both current and immediately preceding completed minute closes strictly exceed their own respective upper bands for long or lower bands for short. Reversal gate failure still exits old position. All candidate gates run only at entry marks and are fixed at decision time.

- F: start session ready. After CAT_STOP or opposite-band exit (including exit portion of reversal), prohibit replacement entry. When flat, a strictly later completed bar closing inclusively inside its own bands rearms entries for a subsequent scheduled check with strict fresh breakout. No same-bar rearm/entry. Exit orders never gated. Reset at new session.

- Use existing contract CSV data/mim_x/mnq_1min_by_contract.csv bound to historical run20260910T210224-historical-f3950efb68. Same audited selected contracts, previous closes,14-session sigma and exclusions. Refuse changed input/source hashes. Preserve stop250 points, reference-gross daily guard-1000, one contract, EOD16:00 proxy, adverse gap fills, primary delay2 and optimistic delay1, costs2.24/3.24/6.24 per roundtrip.

- Primary screen: deltaSharpe>=.20, max daily equity drawdown<=.80*A, totalnet>=.75*A. Highcost at primary timing additionally candidate expectancy>0, deltaSharpe>0, MDD<=A. Thresholds calculated full precision. Undefined zero-variance Sharpe cannot pass.

- Bootstrap paired daily matrix together,20000 resamples seed7 stationary expected blocks5,10,20. Compute each draw Sharpe difference. Report95% and98.75% Bonferroni four-comparison percentile intervals. Exclude/count undefined draws; >1% undefined marks comparison inference unavailable. Historical shortlist requires point+cost screens and adjusted lower bound>0 for all blocks. Point-screen passes otherwise promising but uncertain; rest do not meet screen. No prospective validation claim.

- Report mean/totalnet,Sharpe,MDD,underwater duration,turnover,exposure,yearly results and candidate participation on baseline largest winning days. Diagnostics describe entry-feature distributions for baseline winners/losers without tuning. Rank qualified candidates by primary deltaSharpe then lowerMDD then ID. No winner if none qualifies.

- CLI audit/run accept --data and --baseline-run (labels end fixed); evaluate --run verifies immutable complete run before producing new output. New isolated research/mim_robustness package, all outputs under its runs/. Freeze before returns; record failures; each invocation unique. Read-only data utilities may be reused; adapt copied engine locally, don't import side-effectful study scripts.

- Verification covers baseline parity, formulas/equality, future mutation, reset/reversal state, protective exits, latency stops/gaps guardEOD, missingduplicateszeroVolumewarmupDSTrolls, PnL/turnover reconciliation, deterministic outputs, bootstrap undefined/boundaries, immutability and prohibited writes.

- Daily return is net PnL divided by fixed10000, not compounded equity; RF=0. Annualization uses252 eligible sessions and discloses missing dates. Maximum drawdown primary is daily closing equity including initial zero PnL peak; recovery measured eligible-session counts with unresolved drawdowns censored. DefaultsR/E .30 and30bars are predefined exploratory hypotheses, not empirical optima.

