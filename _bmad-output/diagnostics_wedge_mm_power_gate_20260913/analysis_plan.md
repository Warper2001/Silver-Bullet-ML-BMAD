# Wedge and measured-move power gates — analysis plan

**Written:** 2026-09-13, before any statistic below was computed. The SHA-256 of this file is pinned in `power_gate.py`, which refuses to run on a mismatch, and is recorded in `results.json`.

**Question:** could a test of either Brooks construct detect a plausible edge on 5-minute MNQ bars with the data this shop has?
- **W:** the wedge (three-push) reversal.
- **M:** the measured-move target fade.

These are two independent gates. Each gets its own verdict, and neither's result may inform the other's spec.

**Relationship to the H2/L2 gate:** everything except the construct definitions and the order geometry is inherited unchanged from the committed H2/L2 gate (commit 4482d37, `_bmad-output/diagnostics_h2l2_power_gate_20260913/`). That covers the window, bars, session handling, firewall, effect sizes, costs, power formula and verdict rule.
- The script imports that gate's `load_5min`, `grids`, `placebo`, `shift_stats`, `power` and `thin`.
- It records that file's SHA-256 and refuses to run if it is not `35bc2de4…87ef`.

**Scope:** this gate decides whether a test is worth running. It does not test either strategy.
- It computes no P&L at a real signal.
- It reads nothing under `data/sealed_holdout/`, and no MNQ bar on or after 2026-03-01.

## 1. Shared definitions

**Bars:** exactly as in the H2/L2 plan section 1.1:
- 5-minute RTH MNQ bars closing 09:35 … 16:00 ET, built from close-stamped 1-minute bars
- window 2025-01-01 → 2026-02-28, 297 sessions

**Pivots (strength s):**
- **Pivot high:** bar p is a pivot high if `high[p]` is strictly greater than the highs of the s bars on each side within the same session. It becomes known at the close of bar p+s, its **confirmation bar**.
- **Pivot low:** the mirror image on lows.
- All state resets at each session start.

**Pivot-strength arms, declared now:**
- **s = 1 is primary**, matching the "swing strength 1" of the source strategy.
- **s = 2 and s = 3 are pre-declared sensitivity arms.** They are reported for every table but cannot change the verdict.

**Timing:** no trigger (fill) bar closes after 15:50 (slot 75).

**Tick:** 0.25. All prices are on the tick grid; a computed level is rounded to the nearest tick.

## 2. Construct W: wedge (three-push) reversal

**Wedge top, a short. For each session:**
- **Pushes:** keep the run of consecutive confirmed pivot highs, each strictly higher than the one before. A pivot high at or below the previous one restarts the run at 1.
- **Signal:** when a pivot high P3 is confirmed and the run reaches **exactly 3**, the signal bar q is P3's confirmation bar (p3 + s). Runs of 4 or more do not signal again.
- **Signal-bar filters**, the same bar-quality rules as the H2/L2 construct:
  - it must be a bear bar (`close[q] < open[q]`)
  - it must not be a doji (`|close − open| > 0.30·(high − low)`)
- **Entry:** a sell stop at `low[q] − tick`, valid **on bar q+1 only**. It fills if `low[q+1] < low[q]`, and the fill bar is the trigger bar.
- **Stop:** one tick above P3's high.
- **Risk:** `R_pts = (high[P3] + tick) − (low[q] − tick)`.
- **Target:** 1R primary, 2R sensitivity.

**Wedge bottom, a long:** the mirror image, with consecutive strictly lower pivot lows, a bull signal bar, and a buy stop above it.

## 3. Construct M: measured-move target fade

**Choice of construct:**
- In Brooks, "leg 1 = leg 2" is used two ways. Traders enter the pullback expecting a second leg of equal size, or they fade the move when it reaches the measured-move (MM) target.
- The with-trend use enters on the same pullback-resumption bars as the H1/H2 count; only its exit differs. Its entries are therefore not a new construct.
- This gate tests **the fade**, which is the only use in which the MM defines the entry.

**Bull MM (the fade is a short). When a pivot low C is confirmed at bar c+s:**
- **A** is the most recent pivot low before C, and it must satisfy `low[A] < low[C]` (a higher low). If there is no such A, there is no setup.
- **B** is the highest high among the bars strictly between A and C.
- **Leg 1** is `B − low[A]`, and the target is `D = low[C] + leg1`, rounded to the tick. Because C is a higher low, D > B always.
- **Pullback depth** is `P = B − low[C]`.
- **Skip the setup** if any bar from c+1 to c+s already has a high ≥ D.
- **The order:** a limit sell at D is live from bar c+s+1.
  - It **fills** on the first bar with `high ≥ D + tick`, a trade-through of one tick because limit fills here are adverse-selected.
  - It is **cancelled** by any bar with `low < low[C]` before the fill, by the confirmation of a newer pivot low (which replaces it with that setup's order), or by slot 75.
  - One pending order per direction.
  - If a single bar both breaks C and trades through D, the cancel wins. The number of such bars is reported.
- **Stop:** `D + P`. **Target:** `D − P` (1R), with 2R = `D − 2P` as sensitivity. So `R_pts = P`.
  - The stop and target are set by the market's own pullback depth, not by a hand-set number.

**Bear MM (the fade is a long):** the mirror image. When a pivot high C is confirmed, A is the prior pivot high above C, B is the lowest low between them, and `D = high[C] − (high[A] − B)`. It fills on `low ≤ D − tick`, and `R_pts = high[C] − B`.

## 4. Counts, firewall, costs, effect sizes, power, verdict

All inherited from the H2/L2 plan, sections 1.5 and 2–6. Summary:

- **Counts:**
  - **N_upper** is every filled event.
  - **N_lower** thins events within a session to be at least 24 bars apart. It counts W's and M's events separately.
- **Firewall:**
  - Each event's direction, R and trigger-bar time of day are laid on the path of session j+k, for k in [5, ND−5]. The identity pairing is refused.
  - Placebo entry is the close of the bar at the trigger slot. The exit is stop, target, the 24-bar time stop, or the 15:55 flatten, and the stop wins a bar that touches both.
  - σ is the day-clustered SE × √N, taking the median over shifts.
- **Costs:** $5.80 primary, $11.60 sensitivity.
  - For M, the primary cost stays $5.80 even though a limit entry pays no entry slippage. This is conservative, and the difference is $0.50.
- **Effect sizes** (gross edge per trade, in R): 0.05 pessimistic, **0.10 central**, 0.20 optimistic. Then `μ_net = θ·mean(R$) − c`.
- **Power:** `π = Φ(μ_net·√N/σ − 1.645)`.
  - It is computed for the in-sample window (N measured) and for the holdout (55 sessions, projected at the in-sample rate; the file is not opened).
  - Also reported: the years of data needed for 80% power.
- **Verdict** for each construct, on the primary arm (s = 1, 1R target, $5.80, θ = 0.10):
  - **COST-BOUND** if `μ_net ≤ 0`. Otherwise:

| Condition | Verdict |
|---|---|
| π(N_lower) ≥ 0.80 | **POWERED** |
| π(N_upper) ≥ 0.80 > π(N_lower) | **POWERED-IF-DENSE** |
| 0.50 ≤ π(N_upper) < 0.80 | **MARGINAL** |
| π(N_upper) < 0.50 | **UNDERPOWERED** |

- **If W or M comes out POWERED or MARGINAL,** the next step is a sealed pre-registration. This gate does not run the test.

## 5. What this gate will not do

- Compute any statistic of price after a real event's fill.
- Change a definition, arm, threshold or effect size after seeing output. Any post-hoc analysis is labelled exploratory, and it cannot change a verdict.
- Recommend a parameter.

## 6. Outputs

All in this folder:
- `power_gate.py`
- `results.json`, with the input, plan, script and inherited-gate SHA-256 hashes
- `results.md`
