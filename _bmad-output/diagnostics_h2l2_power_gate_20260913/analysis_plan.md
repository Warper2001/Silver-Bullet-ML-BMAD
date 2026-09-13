# H2/L2 second-entry power gate — analysis plan

**Written:** 2026-09-13, before any statistic below was computed. The SHA-256 of this file is pinned in `power_gate.py`, which refuses to run on a mismatch, and is recorded in `results.json`.

**Question:** could a test of Brooks-style H2/L2 second entries on MNQ 5-minute bars detect a plausible edge on the data this shop has? The gate is required by AGENTS.md policy before any new strategy test.

**Scope:** this gate decides whether a test is worth running. It does not test the strategy. It computes no P&L at a real signal, and it reads nothing under `data/sealed_holdout/`.

## 1. The construct

**Source:** Thomas Wade's published guide to his NinjaTrader "Price Action" strategy (thomaswadepriceactionindicators.com, user guide), which follows Al Brooks' bar counting.

**Translation to this shop:**
- The source uses a 2000-tick ES chart. This shop has no tick-bar history, so the gate uses **5-minute MNQ bars**. Five minutes is Brooks' canonical chart and the timeframe of the failed `study_5min_trend_pullback.py`.
- Relative to that study, the one knob this construct changes is **the entry trigger**: a second-attempt stop entry replaces a first-touch entry at the bar close.

### 1.1 Bars

- **Source files:** `data/processed/dollar_bars/1_minute/mnq_1min_2025.csv` and `mnq_1min_2026_ytd.csv`. The 1-minute bars are close-stamped.
- **Session:** RTH only. A 1-minute bar belongs to the session if its stamp falls in (09:30, 16:00] America/New_York.
- **Resampling:** to 5 minutes with `closed="right", label="right"`. That gives bars closing 09:35 … 16:00, 78 per full day.
- **Window:** 2025-01-01 through 2026-02-28 (ET dates). The script asserts that no bar on or after 2026-03-01 survives the filter.
  - This matters because `mnq_1min_2026_ytd.csv` runs to 2026-06-11 and so overlaps the sealed holdout window.

### 1.2 Bar count (swing strength 1, per session)

The count resets at the first bar of each session.

**Long side (H count):**
- **New leg high:** if `high[t] > leg_high`, then `leg_high = high[t]`, `count = 0`, `armed = False`.
- **Trigger:** if `high[t] > high[t-1]` and `armed`, then `count += 1` and `armed = False`. Bar t is an H{count} bar.
- **Arming:** if `high[t] < high[t-1]`, then `armed = True`. This is a lower-high bar, a leg of the pullback.
- **H2 event:** a trigger with count reaching 2.
  - The **signal bar** is s = t−1.
  - On a 0.25-point grid, `high[t] > high[s]` implies `high[t] ≥ high[s] + 0.25`, so the H2 trigger is exactly the fill of a buy stop placed one tick above the signal bar.
  - H3 and higher are not traded.

**Short side (L count):** the mirror image on lows.

### 1.3 Filters on the signal bar s

All four are known at the close of s, when the stop order would be placed:

| Filter | Long | Short | Source |
|---|---|---|---|
| F1 EMA proximity | `low[s] ≤ EMA21[s] + 1.00` | `high[s] ≥ EMA21[s] − 1.00` | Wade's upper bound of 4 ticks, taken literally (MNQ tick = 0.25) |
| F2 side of EMA | `close[s] > EMA21[s]` | `close[s] < EMA21[s]` | counter-trend filter |
| F3 bar colour | `close[s] > open[s]` | `close[s] < open[s]` | bar-colour filter |
| F4 not a doji | `abs(close−open) > 0.30·(high−low)` | same | doji ≤ 30% body |

- **EMA21** is an EMA with span 21 on 5-minute RTH closes, continuous across sessions and seeded on the first bar (`adjust=False`), as in the earlier study.
- The guide's shooting-star filter is not defined precisely enough to transcribe. It is dropped, and F4 covers most of it.

### 1.4 Order geometry (needed for the placebo in section 3)

- **Entry:** one tick beyond the signal bar.
- **Stop:** one tick beyond the signal bar's opposite extreme.
- **Risk:** `R_pts = range[s] + 0.50`, and `R$ = 2 · R_pts`.
- **Target:** 1R primary, 2R sensitivity. The source gives no target.
- **Time stop:** 24 bars.
- **Flatten:** at the close of the bar ending 15:55.
- **Time limit:** no trigger bar closes after 15:50.

### 1.5 Trade counts

| Count | Definition | Why it is safe |
|---|---|---|
| **N_upper** | every H2/L2 event that passes the filters | an upper bound |
| **N_lower** | events thinned greedily within each session so counted events are ≥ 24 bars apart | a guaranteed lower bound on one-at-a-time trades, because every trade exits within 24 bars; it uses no outcome |

## 2. Costs

- **Primary:** c = $5.80 per round trip = $4.80 commission (the repo figure) + one tick of slippage per side ($1.00).
- **Sensitivity:** c = $11.60.
- **Cost relative to risk:** reported as `mean(c / R$)`, along with the breakeven gross edge `c / mean(R$)`.

## 3. Firewall and null (mismatched pairing)

**The answer this gate must never see:** what price did after a real H2/L2 fill, in the signal's direction. That covers P&L, MFE, MAE and the close of the fill bar.

- The gate may see whether and when an order fills, because that is the event definition.
- It may not see what happened after the fill.

**Placebo trades carry the dispersion.** For each event i, keep its direction d_i, its risk R_i and its trigger-bar time of day.
- Pair it with the price path from **session j+k** (circular over the sorted list of sessions), where j is event i's session and k is a shift of whole sessions.
- **Placebo entry:** the close of the bar in session j+k stamped at the same time of day as event i's trigger bar.
- **Placebo exit:** the geometry of 1.4. If a bar touches both stop and target, count the stop.
- **Placebo P&L:** gross, in $.
- If that time of day is missing from session j+k (a short session), the event is dropped for that shift.

**Shifts:**
- Every k in [5, ND−5], where ND is the number of sessions.
- The pairing function asserts that k ≢ 0 (mod ND). The identity pairing is refused and never evaluated.

**Statistics from each shift:**
- the per-trade sd of gross $ (iid)
- a day-clustered SE, clustered on the event's *original* session. Events from one real session land together, so within-day correlation is preserved.
- the per-trade sd in R units

The medians over shifts are the gate's inputs.

## 4. Effect sizes (declared now; they may not be revised after the run)

**Gross edge per trade, in R:**

| θ | Label | Rationale |
|---|---|---|
| 0.05 | pessimistic | |
| **0.10** | **central** | half the optimistic value, following GAP-1's HALF convention |
| 0.20 | optimistic | about a 60% win rate at 1:1; roughly what discretionary price-action teaching claims for its best setups, with no audited source |

**Net edge per trade in $:** `μ_net(θ) = θ · mean(R$) − c`.

## 5. The test being sized

- A one-sided test of mean net $/trade > 0 at α = 0.05, with 80% power.
- **Power:** `π = Φ(μ_net · √N / σ_eff − 1.645)`, where `σ_eff = SE_cluster · √N` (the primary). The iid σ is also reported.
- **Windows:**
  - **W_IS:** the 2025-01-01 → 2026-02-28 Gate-0 window, with N measured.
  - **W_HOLD:** the sealed holdout, 2026-03-01 → 2026-05-19 per its ACCESS_LOG. N is projected at the W_IS rate per session over 55 sessions. The file is not opened.
- **Data needed:** the sessions and years (at 252 sessions a year) needed for 80% power at each θ, at both costs.

## 6. Verdict

Computed on W_IS, at the primary cost, at central θ = 0.10, with the primary σ.

**Cost check first:** if `μ_net(0.10) ≤ 0`, the verdict is **COST-BOUND**. The plausible edge does not cover costs, and no amount of data helps.

**Otherwise, from π at N_upper and N_lower:**

| Condition | Verdict |
|---|---|
| π(N_lower) ≥ 0.80 | **POWERED** |
| π(N_upper) ≥ 0.80 > π(N_lower) | **POWERED-IF-DENSE** (depends on overlap) |
| 0.50 ≤ π(N_upper) < 0.80 | **MARGINAL** |
| π(N_upper) < 0.50 | **UNDERPOWERED** |

COST-BOUND and UNDERPOWERED are terminal for this construct as specified. A variant needs its own plan.

**Also reported:**
- the same verdict at the 2R target and at the $11.60 cost
- the power of a holdout-only confirmation on W_HOLD

## 7. What this gate will not do

- Compute any statistic of price after a real event's fill.
- Read `data/sealed_holdout/`, or any MNQ bar on or after 2026-03-01.
- Change the spec, filters, effect sizes or thresholds after seeing output. Any deviation is reported as one.
- Recommend a parameter.

## 8. Outputs

All in this folder:
- `power_gate.py`
- `results.json`, with the input, plan and script SHA-256 hashes
- `results.md`
