# Pre-Registration: GC/MGC CPI Post-Catalyst — Prospective Test

**Generated:** 2026-06-09
**Experiment ID:** gc-cpi-prospective-v1
**Pre-registration commit:** (populate after `git commit`)

---

## ⚠️ Data-Observation Disclosure (CRITICAL — read before interpreting any result)

This test was designed **after** running `study_gc_post_catalyst.py` on historical
in-sample data (2025-05-01 → 2026-05-19). The following observations directly
motivated the "CPI only" filter:

| Event | N | WR | PF | Avg P&L | Decision |
|---|---|---|---|---|---|
| CPI | 7 | 57.1% | 1.77 | +$22 | Motivates this test |
| NFP | 8 | 25.0% | 0.30 | −$40 | Excluded (losing) |
| FOMC | 6 | 33.3% | 0.50 | −$29 | Excluded (losing) |
| **All (primary spec)** | **21** | **38.1%** | **0.668** | **−$16** | **Gate 0 FAIL** |

**The CPI subgroup was selected POST-HOC.** The in-sample CPI PF of 1.77 (N=7)
has ZERO independent validity. Any backtest that reproduces this result merely
confirms what was already observed — it adds no information.

**This pre-registration is the one mechanism that creates genuine validation.**
If the prospective PF (on events that occurred AFTER this commit date) exceeds
the thresholds below, that constitutes independent evidence of edge.
If it does not, CPI was a noise artefact of N=7 in-sample.

**The only methodological protection in this test:** the exact parameters
(WAIT=10, MIN_MOVE=2×ATR, SM=1.5×, TP=2:1) were declared as the "primary spec"
BEFORE running the historical study. They were not tuned to optimise the CPI
subgroup. This is explicitly documented in `study_gc_post_catalyst.py` docstring.

---

## Hypothesis

**H₁ (alternative):** Monthly CPI releases create a directional gold price move
that persists for at least 2 hours and generates positive expectancy when entered
10 minutes after the release, provided the initial move exceeds 2× ATR(14).
Specifically: prospective PF > 1.20, EV > $0, WR ≥ 40% over N ≥ 10 qualifying events.

**H₀ (null):** The in-sample CPI PF of 1.77 is a noise artefact of N=7 post-hoc
selection; prospective PF ≤ 1.0.

---

## Strategy Logic (Frozen — do NOT adjust after any prospective event)

| Element | Rule |
|---|---|
| Event scope | **CPI only** (monthly BLS Consumer Price Index release) |
| Release time | 08:30 ET (standard BLS schedule) |
| Pre-event reference | Close of 1-min GC bar immediately before 08:30 ET |
| Wait period | 10 bars = 10 minutes after 08:30 ET bar |
| Direction | Sign of (close[T+10] − close[T−1]): LONG if positive, SHORT if negative |
| Qualification | Skip if |net move| < 2.0 × ATR(14) at T+10 bar |
| Entry | Close of the 10th 1-min bar after 08:30 ET |
| ATR | Rolling 14-bar mean of (high − low) on GC 1-min bars (full session, not RTH-only) |
| Stop | 1.5 × ATR(14) at entry bar, clamped to $150/contract (= 15 pts on MGC) |
| TP | 2.0 × stop_dist in entry direction |
| Hold max | 120 bars (2 hours); close at market if neither TP nor stop hit |
| No session close | GC trades 23h/day; no forced session exit (except HOLD_MAX) |
| Contract | 1 MGC (Micro Gold, $10/pt, 10 troy oz) |
| Commission | $4.80 round-trip |

---

## Prospective Tracking Protocol

### Which events are in scope

All **CPI releases on or after the date of the pre-registration commit** that:
- Are released at 08:30 ET by the BLS
- Have GC 1-min data available for the event time and the following 130 bars

Events before the pre-registration commit date are **in-sample** and are excluded
from the prospective count regardless of outcome.

### How to log each event

After each CPI release:
1. Download fresh GC 1-min data covering the event date plus 3 hours
   (via `download_gc_1min.py` with updated end date and contract)
2. Run: `.venv/bin/python track_gc_cpi_event.py --date YYYY-MM-DD`
   The script will verify the config hash, compute the trade, and append to
   `data/reports/gc_cpi_prospective_log.csv`
3. Commit the updated log: `git add data/reports/gc_cpi_prospective_log.csv && git commit`
4. Check the early stopping rule (printed by the tracker)

### No adjustments allowed

After any prospective event is observed, the following are PROHIBITED:
- Changing WAIT_BARS, MIN_MOVE_ATR, STOP_ATR_MULT, TP_MULT, or any other parameter
- Restricting to a direction (long/short) based on what has been observed
- Excluding events retroactively
- "Testing" alternative parameters on the same prospective events

Any such change voids this pre-registration. Start a new one.

---

## Decision Rules (Pre-committed, Immutable After Seal)

### Early Stopping Rule (bearish)

After **N_qualifying ≥ 5** prospective events:
- If running prospective PF < **0.70** → HALT, declare FAIL, do not wait for N=10
- Strategy is clearly not working and continued testing wastes live capital

### Final Verdict (after N_qualifying = 10)

| Criterion | Threshold | Action if not met |
|---|---|---|
| N qualifying events | ≥ 10 | Continue tracking |
| Profit factor | ≥ 1.20 | FAIL — no combine deployment |
| Avg net P&L/trade | > $0 | FAIL |
| Win rate | ≥ 40.0% | FAIL |

**If all thresholds met:** PASS. The CPI post-catalyst strategy may advance to
full combine backtest (Gate 1) and, if that passes, a sealed OOS test (Gate 2).

**If any threshold missed:** FAIL. The CPI subgroup was a noise artefact. Record
the result and update memory. Do not re-run on another subgroup of CPI events.

### Combine deployment gate (only if prospective PASS)

Gate 1 (if prospective PASS): full combine accounting overlay backtest on all
historical + prospective GC data, applying trailing DD tracking, daily halt,
qualifying-day count, consistency cap. Minimum: MaxDD ≤ $2,000.

Gate 2 (if Gate 1 PASS): one-shot prospective OOS test — forward 20 CPI events
collected after Gate 1 pre-registration, subject to the same Go/No-Go rules.

---

## Strategy Parameters Snapshot

```yaml
# GC/MGC CPI Post-Catalyst Prospective Test — Frozen Parameters
# Pre-registration: _bmad-output/preregistration_gc_cpi_prospective.md
# DO NOT MODIFY after pre-registration commit — any change voids the prospective test.
version: "gc-cpi-v1"

# Event filter (only CPI releases are in scope)
event_type: CPI
event_time_et: "08:30"          # 08:30 ET — standard BLS CPI release time

# Entry timing
wait_bars: 10                   # wait 10 min after 08:30 before entering (spike settles)

# Qualification filter
min_move_atr: 2.0               # skip if |close[T+10] - close[T-1]| < 2.0 × ATR(14)
                                # (~60% of events qualify at this threshold)

# Trade geometry (ALL FROZEN)
stop_atr_mult: 1.5              # stop = 1.5 × ATR(14) at entry bar
tp_mult: 2.0                    # target = 2.0 × stop_dist (2:1 reward:risk)
hold_max_bars: 120              # 2-hour maximum hold; close at market if neither TP nor stop hit
atr_window: 14                  # rolling(14) of (high - low) on 1-min GC bars

# Direction
direction: both                 # enter LONG if gold rallied, SHORT if gold fell
                                # (no direction pre-selection — avoids double data-snooping)

# MGC contract economics (combine-sized)
mgc_pv: 10.0                    # $10 per point per MGC contract (10 troy oz)
commission: 4.80                # round-trip commission
stop_cap_usd: 150.0             # combine stop cap ($150/contract = 15 pts on MGC)

# Prospective test decision rules (frozen)
# Early stop  : HALT if PF < 0.70 after N_qualifying >= 5
# Final PASS  : N_qualifying >= 10 AND PF >= 1.20 AND avg_pnl > 0 AND WR >= 0.40
# Final FAIL  : otherwise
n_min_qualifying: 10
early_stop_n: 5
early_stop_pf: 0.70
final_pf_threshold: 1.20
final_ev_threshold: 0.0
final_wr_threshold: 0.40
```

---

## In-Sample Context

| Period | N events | N qualifying | WR | PF | Avg P&L |
|---|---|---|---|---|---|
| 2025-05-01 → 2026-05-19 (historical) | 13 CPI | 7 | 57.1% | 1.77 | +$22.37 |

**This table is the observation that motivated the test, not validation.**
The next CPI event after the pre-registration commit begins the prospective count.

---

## Integrity Hashes

| Hash | File | Value |
|---|---|---|
| (a) gc_cpi_config.yaml SHA-256 | `gc_cpi_config.yaml` | `c7be60e446629aa1d765c2156f0c84533ae463a15214f10f0f9d9d1b89cec870` |
| (b) study_gc_post_catalyst.py SHA-256 | `study_gc_post_catalyst.py` | `aa1af3539d203c082212a9d5cece79b71d4e96952840dd4f2f3696113c57865a` |
| (c) Git HEAD at seal time | — | `e52bd87d4c9e46b6eba821570d6a2b65993eefe7` |

*Hash (a): Proves parameters unchanged between pre-registration and any prospective event.*
*Hash (b): Proves the in-sample simulation logic that motivated this test.*
*Hash (c): Any prospective CPI event that occurred before this commit is excluded.*

---

## Scope Constraint

This pre-registration covers prospective tracking only. No live MGC trading on
a funded combine account until Gate 1 and Gate 2 pass. S25
(tier2_streaming_working.py on account 23884932) continues unchanged.
