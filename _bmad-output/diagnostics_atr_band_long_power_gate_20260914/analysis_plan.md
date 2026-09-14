# ATR-band long power gate — analysis plan

**Written:** 2026-09-14, before any event count, ATR value or price-path statistic was computed. Its SHA-256 is pinned in `power_gate.py`, which refuses to run on a mismatch. The plan and script are committed to git **before** the run (closing the H2/L2 gate's deviation 2).

**Computed before this plan, and only this:** a data-provenance profile of the raw input (record counts, contract labels per month, contract-switch times, minutes per session). Its numbers are quoted in section 1.2. No price level, range, ATR or event was examined.

**Question:** could a test of the "Trader MNQ" ATR-band long on MNQ 5-minute bars detect a plausible edge on the data this shop has? AGENTS.md requires this gate before any new strategy test.

**Scope:** this gate decides whether a test is worth running. It computes no P&L at a real fill and reads nothing under `data/sealed_holdout/`. The short (wick) leg and the prior-red-day filter are out of scope; they need their own gates.

**Source of the rules:** the imported user summary in `_bmad-output/planning-artifacts/research/academic-lit-trader-mnq-atr-band-longs-and-wick-short-2026-09-13/imports/user-strategy-summary.md`. The original video, code and trade list were never obtained. Every value the summary leaves open is fixed below as this shop's convention, not a claim about the presenter's implementation.

## 1. Data

### 1.1 Input

- **File:** `/root/mnq_historical.json`, the raw TradeStation 1-minute MNQ bars behind the repo's `mnq_1min_2025.csv`. Its SHA-256 is recorded in `results.json`.
- **Why not the CSVs:** `mnq_1min_2025.csv` is dollar-aggregated from this file, with 29,520 multi-minute bars and 5,583 mixed-contract bars (`docs/reports/yank-provenance-closure/README.md`). This construct fires on extreme short-horizon drops, which is exactly what a contract splice fakes.
- **Stamping:** TradeStation `TimeStamp` is the bar's close. The first record of each Globex session is stamped 18:01 ET, one minute after the 18:00 open.
- **Window:** records stamped before 2026-03-01 00:00 UTC. Later records are skipped during parsing, never stored. The script asserts that no bar on or after 2026-03-01 ET survives. The sealed holdout begins 2026-03-01 (`data/sealed_holdout/ACCESS_LOG.md`).

### 1.2 Contract purity (the profile, done before this plan)

- 788,780 pre-cutoff records, 2023-12-01 → 2026-02-27, no duplicate timestamps, one contract label per minute.
- **Roll weeks interleave contracts minute by minute:** 20,760 label switches, each a ±~240-point jump (the calendar spread). On an unadjusted continuous series, every switch is a fake multi-ATR move.
- RTH: 576 sessions; 40 contain more than one contract; 22 are holiday half-days (< 380 minutes).
- Globex: 578 sessions; 76 contain more than one contract; 474 are single-contract with ≥ 1,300 of 1,320 minutes.

### 1.3 Sessions and bars

| | **RTH (primary)** | Globex (sensitivity) |
|---|---|---|
| 1-minute stamps kept | (09:30, 16:00] ET | (18:00 prior day, 16:00] ET, keyed to the date of the 16:00 close |
| Eligible session | one contract label for every kept minute, and ≥ 380 minutes | one contract label, and ≥ 1,300 minutes |
| 5-minute bars | `closed="right", label="right"`: 09:35 … 16:00, 78 slots | 18:05 … 16:00, 264 slots |
| Last fill bar | closes 15:50 | closes 15:50 |
| Flatten | close of the bar ending 15:55 | close of the bar ending 15:55 |

Ineligible sessions are dropped whole, before ATR is computed. The 16:00–18:00 ET gap is never traded. No order spans a session boundary.

## 2. The construct

### 2.1 ATR

- **ATR(14), Wilder smoothing** (RMA, α = 1/14), TradingView's `ta.atr` default. It is seeded with the mean of the first 14 true ranges, runs continuously across eligible sessions, and includes the gap from the previous session's last close.
- **Contract reset:** if the previous bar belongs to a different contract, that bar's true range is `high − low`.
- **Warm-up:** no orders in the first 3 eligible sessions.

### 2.2 Order (long only)

At the close of bar t, with close C_t and ATR A_t:

- **Buy limit:** `L = floor((C_t − 3.1·A_t) / 0.25) · 0.25`. It rests during bar t+1 only and is replaced at that bar's close. It is placed only if bar t+1 is the next slot of the same session.
- **Fill, primary — trade-through:** `low[t+1] ≤ L − 0.25`. A resting limit at a touched price need not fill.
- **Fill, sensitivity — touch:** `low[t+1] ≤ L`.
- **Stop distance:** `1.5·A_t` in points, rounded to the nearest tick. **Target distance:** `2·A_t`, rounded the same way. Both are frozen at entry (entry-known ATR) and measured from the fill.
- **Risk:** `R_pts` = stop distance, and `R$ = 2 · R_pts` for one MNQ contract.
- **Exit:** stop, target, or flatten, whichever comes first. There is no other time stop; the source gives none.

### 2.3 Trade counts

| Count | Definition | Why it is safe |
|---|---|---|
| **N_upper** | every fill event | an upper bound; permits overlapping positions |
| **N_lower** | the first fill event of each session | a guaranteed lower bound on one-at-a-time trades, because a position can last until flatten; it uses no outcome |

## 3. Costs

| Name | $ per round trip | Basis |
|---|---|---|
| low | 2.22 | TopstepX's published $1.22 MNQ round trip (research report, section 2) plus one tick of slippage on the exit only |
| **primary** | **5.80** | the repo's $4.80 commission plus one tick per side; the H2/L2 and wedge/MM gates used the same figure |
| double | 11.60 | stress |

Reported: `mean(c / R$)` and the breakeven gross edge `c / mean(R$)`.

## 4. Firewall and dispersion

**The answer this gate must never see:** what price did after a real fill. That covers P&L, MFE, MAE, and the rest of the fill bar. The gate may see whether and when a limit fills, because that is the event definition.

### 4.1 Placebo (mismatched pairing), primary σ

For each event i, keep its fill-bar slot and its R_pts and target distance.
- Pair it with the path of eligible session **j+k** (circular over the sorted eligible sessions), where j is event i's session.
- **Placebo entry:** the close of the bar at the fill-bar slot in session j+k. **Path:** later bars up to the flatten bar. A bar touching both stop and target counts the stop. With neither, exit at the flatten bar's close (or the last available close).
- If the entry slot is missing in session j+k, drop the event for that shift.
- **Shifts:** every k in [5, ND − 5], where ND is the number of eligible sessions. The pairing function asserts k ≢ 0 (mod ND); the identity pairing is refused.
- **Per shift:** the iid sd of gross $; a day-clustered σ, clustered on the event's *original* session; the sd in R units. The gate uses the medians over shifts.

### 4.2 Bracket bound, conservative σ

**Why:** the placebo can understate dispersion. An event's A_t may be large relative to the random session it is laid on, so more placebo trades reach flatten instead of a bracket.

**Construction** (no outcome is used): suppose every trade ends at a bracket with a win probability of 0.5, so the outcome given A is +target$ or −R$ with equal probability. Then `σ_bb² = E[Var(X|A)] + Var(E[X|A])` over the events' A. It is scaled by the placebo's clustering ratio `max(1, σ_cluster / σ_iid)`.

**Verdict σ:** `σ_v = max(σ_placebo_cluster, σ_bb_scaled)`.

## 5. Effect sizes (declared now; they may not be revised after the run)

**Gross edge per trade, in R:**

| θ | Label | Win rate implied at the +1.33R / −1R brackets |
|---|---|---|
| 0.05 | pessimistic | 45.0% |
| **0.10** | **central** | **47.1%** |
| 0.20 | optimistic | 51.4% |

These are the H2/L2 and wedge/MM gates' values, following GAP-1's HALF convention, so the gates are comparable. The gross breakeven win rate is 42.86%. No audited figure for this setup exists.

**Net edge:**
- **Primary (fixed one contract):** `μ_net(θ) = θ · mean(R$) − c`.
- **Sensitivity (constant-risk sizing):** the source says contracts are reduced as ATR rises. Per trade in R: `μ_R(θ) = θ − mean(c / R$)`, with σ in R units taken from the placebo (and, for the bound, `σ_bb` recomputed in R). Integer MNQ lots make exact constant risk infeasible at small budgets, so this is a sensitivity only.

## 6. The test being sized

- A one-sided test of mean net $/trade > 0 at α = 0.05, with 80% power.
- **Power:** `π = Φ(μ_net · √N / σ − 1.645)`.
- **Windows:**
  - **W_IS:** every eligible pre-cutoff session, 2023-12-01 → 2026-02-27, with N measured.
  - **W_HOLD:** the sealed holdout, 2026-03-01 → 2026-05-19, projected at the W_IS event rate per eligible session over 55 sessions. The file is not opened.
- **Data needed:** the sessions and years (at 252 sessions a year) needed for 80% power at each θ and cost.

## 7. Verdict

**Primary cell:** RTH, trade-through fills, one contract, primary cost, central θ = 0.10, σ_v.

**Cost check first:** if `μ_net(0.10) ≤ 0`, the verdict is **COST-BOUND**.

**Otherwise:**

| Condition | Verdict |
|---|---|
| π(N_lower) ≥ 0.80 | **POWERED** |
| π(N_upper) ≥ 0.80 > π(N_lower) | **POWERED-IF-DENSE** |
| 0.50 ≤ π(N_upper) < 0.80 | **MARGINAL** |
| π(N_upper) < 0.50 | **UNDERPOWERED** |

COST-BOUND and UNDERPOWERED are terminal for this construct as specified. A variant needs its own plan.

**Also reported, without changing the primary verdict:** every cell of {RTH, Globex} × {trade-through, touch} × {one contract, constant risk} × {low, primary, double cost} × θ, under both σ_placebo and σ_v, plus the power of a holdout-only confirmation.

**If the gate says POWERED,** the next step is still a pre-registration. That later test needs a control that separates the downside excursion from simply being long Nasdaq in a rising market (research report, section 5, item 4). No window is clean relative to the presenter's own parameter choice (3.1 / 2 / 1.5), whose data is unknown and may overlap 2024–2026.

## 8. What this gate will not do

- Compute any statistic of price after a real event's fill.
- Read `data/sealed_holdout/`, or keep any MNQ record stamped on or after 2026-03-01.
- Change the spec, eligibility rules, effect sizes or thresholds after seeing output. Any deviation is reported as one.
- Recommend a parameter.

## 9. Outputs

All in this folder:
- `power_gate.py`
- `test_power_gate.py`: synthetic checks of event detection and the identity-pairing refusal
- `results.json`, with the input, plan and script SHA-256 hashes
- `results.md`
