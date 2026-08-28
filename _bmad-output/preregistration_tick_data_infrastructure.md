# Pre-Registration: TICK-INFRA — MNQ Tick Data + Queue-Aware Fill Simulator

**Registered:** 2026-08-28
**Authored by:** Alex (warper2001@gmail.com), drafted in BMAD session
**Status:** SEALED at commit time. Append-only amendments.
**Upstream:** `_bmad-output/planning-artifacts/research/technical-higher-frequency-nq-mnq-intraday-scalpin-2026-08-28/research.md` — Recommendation R3, Open Question 1.

---

## 0. What this document seals, and why it exists

The research run concluded that **no ≥3-trades/day NQ/MNQ pattern-scalping family clears friction and can be validated on 1–12 months of 1-minute data**, and that the one honest way left to test whether a genuine fast edge exists is **at tick resolution with a queue-aware fill simulator** (R3). Both Mesfin preprints argue the surviving intraday edge "lives inside the bar" and is destroyed by bar-close→next-open execution.

Building that tool is not itself a hypothesis test. But three things about it are vulnerable to being quietly changed to flatter a later result, and this seal freezes them **before any tick data is purchased or examined**:

1. **The data purchase spec** — schema, symbol, contract handling, date range, and which slice is sealed as out-of-sample. So there is no "let me also grab 2019–2020" or "let me switch the roll rule" when a result disappoints.
2. **The simulator's modelling choices and its acceptance gate** — including what it deliberately does *not* model. So the fill model cannot be loosened toward front-of-queue optimism, and the simulator is not trusted for any research decision until it demonstrably reproduces real broker fills.
3. **The decision-rule template** that every tick-resolution strategy study must pass, and **the first sealed hypothesis** (H1) with its bounded parameter sweep and forbidden post-hoc moves.

This seal follows the project's standing rules: thresholds are anchored to external sources or measured instrument properties, not hand-picked (`memory/feedback_derive_dont_assert_one_knob.md`); the economic decision rule is stated separately from and above statistical significance (`preregistration_ofi_bar_level.md` §2); the minimum detectable effect at the planned N is computed here, before sealing (`preregistration_intraday_momentum_mnq.md` A1.5); and no result may be rescued by restricting to a favourable subset (`memory/feedback_iteration_loop_pattern.md`).

Nothing in this document authorises deployment, sizing, or any change to a live system.

---

# PART I — INFRASTRUCTURE SPECIFICATION (frozen)

## 1. Data acquisition — frozen spec

| Item | Value | Rationale |
|---|---|---|
| Vendor | Databento, dataset `GLBX.MDP3` (CME Globex) | Only retail-accessible source of full-depth MNQ history; named in the research run |
| Instrument | MNQ (Micro E-mini Nasdaq-100), all quarterly contracts in range | Matches the deployed system and every prior MNQ seal |
| Schema | **`mbo`** (market-by-order: every add / modify / cancel / trade with order ID) | Queue position cannot be reconstructed from `mbp-10` alone; the whole point of R3 is queue realism |
| Secondary schema | `mbp-1` (BBO) + `trades`, retained for cross-checks and for the cheap path if `mbo` cost is prohibitive | — |
| Date range purchased | **2023-01-01 → most recent available** | ≥ 2 full years for ≥2 volatility regimes + year-stability (§7); extends the existing sealed-holdout convention |
| Development window | **2023-01-01 → 2026-02-28 only** | The strategy and simulator may be fit, tuned, and inspected here |
| Sealed out-of-sample window | **2026-03-01 → present**, access-logged in `data/sealed_holdout/ACCESS_LOG.md` | Matches the project's existing sealed-holdout boundary (`data/sealed_holdout/mnq_1min_holdout_20260301_plus.csv`); the tick holdout is the OOS gate for every study under this seal |
| Contract handling | Per-contract, **no stitching**. Forward returns and signals are computed `groupby(contract)`; nothing crosses a roll boundary | Matches `preregistration_ofi_bar_level.md` §3 |
| Front-month definition | The contract with the greater cumulative volume on the prior session; switch takes effect at that session's open | Deterministic, no look-ahead |
| Storage | `data/tick/mnq_mbo_<contract>.<ext>`; raw vendor files retained unmodified alongside any parsed form | Reproducibility |
| Integrity check before use | On every contract file: monotonic non-decreasing timestamps; `bid ≤ ask` on 100.000% of book states; trade prices within [session low, session high]; report the exact pass rate | Same discipline as OFI-1's `upvol+downvol==volume` check |

**Cost:** the research run cited ~$42/quarter; the true `mbo` cost is to be confirmed at purchase. The *spec* above is frozen regardless of cost. If `mbo` for the full range exceeds a budget the user sets at purchase time, the pre-committed fallback is: buy `mbo` for **2024-01-01 → 2026-02-28** (dev) + **2026-03-01 → present** (holdout), and buy `mbp-1`+`trades` for 2023. A shorter `mbo` window is recorded as a power reduction (§9), not a silent change.

## 2. Fill simulator — frozen modelling contract

The simulator consumes the `mbo` stream and a strategy's order intents (submit / cancel / replace, each with a timestamp) and returns fills (price, size, timestamp) plus a per-order diagnostic (queue rank at submit, time-to-fill, adverse-selection marker).

### 2.1 What it models

| Aspect | Primary model (decision-bearing) | Secondary model (reported, never decision-bearing) |
|---|---|---|
| Queue position on a passive limit | **Back of queue** at the moment our order's timestamp + latency lands — i.e. every resting order already at that price is ahead of us, plus every order that arrives before us | Time-priority rank reconstructed from `mbo` order timestamps ("optimistic") |
| Latency (order intent → exchange) | **Fixed 250 ms round trip** (retail, no colo) | Fixed 50 ms ("near-colo") |
| Passive fill | Occurs only when cumulative trade volume at our price, after our arrival, exceeds the queue ahead of us | same, under the optimistic queue |
| Marketable / marketable-limit order | Fills by walking the resting book at arrival, price-time priority, partials allowed | — |
| Adverse selection | Marked (not penalised beyond the fill price itself) whenever a passive fill is followed within 1 s by a same-side quote move through our price | — |
| Cancel / replace | Takes one latency hop; queue priority is lost on any price change, kept on a size-decrease | — |
| Own-order market impact | **None** — assumed negligible at 1–5 micro contracts | A ±1-tick stress applied to every entry and exit (§2.3) |
| Fees | $0.72 exchange+regulatory round turn + a commission the user sets at seal time (default $0.58 RT, Tradovate base) | — |

### 2.2 What it deliberately does NOT model — and the direction of the bias

1. **Our true queue position is unknowable offline.** We were never in this book. Primary = back-of-queue is deliberately pessimistic; the gap between primary and secondary queue models bounds the uncertainty.
2. **Variable / bursty latency.** Real retail latency spikes under load exactly when it matters. A fixed 250 ms is optimistic in the tail. Disclosed, not corrected.
3. **Hidden and iceberg liquidity** that is in the true book but not expressed in `mbo` displayed size. Understates available passive liquidity → makes passive fills *harder* than reality → conservative for a passive strategy, optimistic for a strategy that assumes it is providing scarce liquidity.
4. **Matching-engine edge cases** (self-match prevention, credit controls, messaging throttles). MNQ is FIFO, so pro-rata effects are absent; the rest are minor at micro size.
5. **Our own order flow changing other participants' behaviour.** Not modellable offline. The §2.3 impact stress is the only guard.
6. **Exchange outages, halts, limit states, and the Sunday-open / holiday-thin sessions.** Bars inside a CME halt or the 4–5 pm CT maintenance window are excluded, not modelled.

### 2.3 Mandatory stress overlay on every result

Every strategy P&L figure under this seal is reported **three ways**: (a) primary model as specified; (b) primary model **plus a 1-tick adverse slip on every entry and every exit** (own-impact proxy); (c) primary model with the **optimistic** queue and latency. The decision rule (§6) is evaluated on (a) **and** must also hold under (b). Figure (c) is context only.

---

# PART II — SIMULATOR ACCEPTANCE GATE (must pass before Part III)

## 3. Parity against real fills

The simulator is **not trusted for any research decision** until it reproduces real broker fills from this project's own live ledger.

| Item | Value |
|---|---|
| Parity sample | All closed trades in `data/trades.db` from live combine / SIM-mirror execution: `trader-mim-nb`, `trader-yank` (live-combine subset only), `trader-gap-fade`, `trader-s26-combine` — every trade with a real broker fill price and timestamp. **Count at seal: ~150+, growing.** |
| Minimum sample to run the gate | **N ≥ 100** real closed trades (entry + exit = 200 fills) |
| What is replayed | Each trade's actual order intent (side, type, limit price, submit timestamp) fed to the simulator over the `mbo` stream for that instant |
| **Pass tolerance — primary** | Mean absolute fill-price error **≤ 1.0 tick** ($0.50 / 0.25 index pt) across all 200+ fills |
| **Pass tolerance — tail** | 90th-percentile absolute fill-price error **≤ 2.0 ticks** |
| **Pass tolerance — bias** | Mean *signed* fill-price error within **± 0.25 tick** (no systematic optimism or pessimism) |
| Reported alongside | Error distribution by order type (limit vs market), by session (RTH vs overnight), by trader; time-to-fill error for passive orders |

## 4. Infrastructure kill criterion

If the simulator **cannot pass §3 within a bounded effort — 3 revision cycles or 15 working days from first parity run, whichever first** — the R3 path is **abandoned and recorded as such**. The specific unmodelled effect that causes the miss is written up (magnitude, direction, why it could not be bounded), and no tick-resolution strategy study proceeds. A failure here is a real and publishable outcome: it would mean retail-accessible offline tick simulation cannot be made faithful enough to trust for MNQ scalping, which directly answers whether R3 was worth pursuing.

If §3 passes, the passing simulator commit SHA is recorded in an amendment and **frozen** — any later change to the fill model requires a new parity run and a new amendment before it may inform a decision.

---

# PART III — FIRST SEALED HYPOTHESIS (H1)

## 5. H1 — does an intra-bar edge exist?

### 5.1 Hypothesis

**H₁:** A breakout-continuation signal computed and acted on **intra-bar at tick resolution** on MNQ — entering the instant price trades through a rolling high/low by a threshold, rather than at the next bar open — produces a **≥ 3 trades/day** strategy that is **net-positive after $4.00 round-trip friction, with OOS walk-forward t ≥ 2.0, N ≥ 200 trades spanning ≥ 2 volatility regimes, direction-consistent across every test year, and permutation p < 0.05.**

**H₀:** No such configuration clears the rule. The 5-minute-bar "gross edge ceiling" of 0.07–1.50 index points per trade (Mesfin, research §3) is not materially exceeded by moving execution inside the bar.

### 5.2 Why this signal family and not another

The signal is **not** free-form. Mesfin (arXiv:2605.04004, §7 and the companion paper) states the surviving intraday edge is in "the breakout moves that clearly have momentum inside the bar but not between bars." H1 tests exactly that claim and no other. The family is fixed to: **on a rolling `W`-minute window, when the last trade price exceeds the window high by `k` ticks (or below the low by `k` ticks), submit a marketable-limit order in the breakout direction; exit after `H` seconds or at a `S`-tick stop, whichever first.** Direction is symmetric (long breakouts and short breakdowns both taken).

### 5.3 Pre-declared parameter sweep (bounded, OOS-validated)

Four parameters, swept over a **fixed grid declared here**, chosen by out-of-sample walk-forward performance — never by in-sample fit, never hand-picked:

| Param | Grid | Meaning |
|---|---|---|
| `W` | {5, 15, 30, 60} minutes | rolling breakout window |
| `k` | {1, 2, 4} ticks | breakout threshold beyond the extreme |
| `H` | {60, 180, 300} seconds | max hold |
| `S` | {4, 8, 16} ticks | hard stop |

That is 108 configurations. Per `memory/feedback_derive_dont_assert_one_knob.md` and the Bailey–López de Prado minimum-backtest-length result (research §2), a 108-cell grid over a ~3-year development window is **within budget only if deflated**: the reported statistic for the selected configuration is the **deflated Sharpe ratio** (Bailey & López de Prado 2014) with the trial count set to 108, and the probability of backtest overfitting (PBO via CSCV) is reported. A raw Sharpe or t-stat for the grid winner is **not decision-bearing** on its own.

### 5.4 Walk-forward protocol

Expanding window, matching Mesfin: train 2023 → test 2024; train 2023–24 → test 2025; train 2023–25 → test 2026-01-01…02-28. The configuration is re-selected by the grid at each training boundary. The **2026-03-01+ sealed tick holdout (§1) is touched exactly once**, after the walk-forward verdict is written, and its access is logged.

### 5.5 Volatility-regime split

"≥ 2 volatility regimes" is defined before the run using the project's existing tool: H1 P&L is partitioned by the `VVG` / rolling-ATR regime label (research §9; `arXiv:2605.11423` inputs) into "elevated" vs "normal" days. The rule requires N ≥ 200 total **and** a direction-consistent (same sign) mean net P&L in **both** partitions. A result that is positive only in one regime is a **FAIL**, not a regime-conditioned strategy (this is the documented iteration-loop failure pattern).

## 6. Decision rule — ECONOMIC first, statistical second

Friction is pre-declared at **$4.00 round trip (2.0 index points)** — the research run's conservative figure for market-order / marketable-limit execution (`research.md` §3, ref [13]), and the exact figure Mesfin applied. Secondary reporting at $2.00 RT (disciplined-limit) is shown alongside but is not decision-bearing.

Evaluated on simulator model (a) **and** required to also hold under model (b) (the 1-tick entry+exit stress, §2.3):

| Condition (on per-trade **net** P&L, $4 friction, OOS walk-forward) | Verdict | Pre-committed action |
|---|---|---|
| mean net ≥ **+1.0 index pt** ($2) **and** deflated-Sharpe t ≥ 2.0 **and** N ≥ 200 **and** both regime partitions same-sign positive **and** every test year same-sign **and** permutation p < 0.05 **and** all of the above still true under stress model (b) | **PASS** | Record. Triggers §8. **No deployment.** |
| mean net > 0 but any one statistical / robustness criterion fails | **POSITIVE, UNPROVEN** | Record. Eligible for a prospective seal (its own document). No deployment, no grid re-opening. |
| mean net ≤ 0 at $4 friction, OR fails under stress model (b) | **FAILS** | The intra-bar edge claim is not supported on MNQ over 2023–2026 at retail-accessible tick realism. Record; close H1. |

**Statistical significance is reported for completeness and is explicitly NOT sufficient for a PASS** — the economic bar (mean net ≥ +1.0 index pt after $4 friction) must clear first. This mirrors `preregistration_ofi_bar_level.md`: at tick resolution N is large and t-statistics are cheap; a p-value here is not evidence of a tradeable effect.

## 7. Power statement — computed before sealing

Target: a ≥ 3 trades/day strategy validated on the ~14 months of walk-forward test data (2024-01-01 → 2026-02-28, ~294 trading days). Per-trade P&L standard deviation is **estimated at $4.00** for this scalp profile (8-tick stop = $4; time exits cluster near zero; occasional larger winners) — this estimate is replaced by the measured value in the H1 result amendment, and the table re-run.

| Trades/day | N over the test window | Smallest per-trade Sharpe detectable at t = 2.0 (= 2/√N) | Smallest mean net detectable at sd ≈ $4 |
|---|---|---|---|
| 3 | ~880 | 0.067 | ~$0.27 (0.13 index pt) |
| 8 | ~2,350 | 0.041 | ~$0.17 (0.08 index pt) |
| 15 | ~4,400 | 0.030 | ~$0.12 (0.06 index pt) |

Two facts follow, and they define why the design is sound:

1. **Well-powered for a real edge.** The research run established that an economically relevant intraday edge on MNQ needs a per-trade Sharpe of roughly **0.08–0.12** (`research.md` §2). At ≥ 3 trades/day the test detects a per-trade Sharpe of 0.067, so an edge at the low end of that band is inside reach and anything above it is comfortably so.

2. **The statistical test is more sensitive than it is economically relevant — so the decision must be economic.** The §6 PASS bar of **+1.0 index point net** is roughly 5–8× the smallest mean the t-test can resolve. This is OFI-1's inverted-power situation in milder form: statistical significance will arrive for effects far below the economic bar. §6 handles it by making the economic threshold (mean net ≥ +1.0 index pt after $4 friction) the gate, with the t-statistic necessary but not sufficient, and by counting N in **round-trip trades — never per tick or per signal** — so a large tick sample cannot inflate the decision statistic.

## 8. Successor trigger

A **PASS** on H1 authorises drafting **one** successor pre-registration for a tradeable strategy built on that configuration. The successor must carry its own power statement, its own holding-period cost model, and a prospective (not backtested) evaluation plan. It does **not** authorise deployment, sizing, paper trading, or any change to a live system.

**Other hypotheses need their own amendment.** In particular:
- **H2 — reconstruct the RTH Confluence / London Session B regime signal (research R2) at tick resolution** — is a separate sealed test. It requires this seal's simulator (post-parity) plus its own amendment specifying the GMM feature vector, fit window, and label-alignment method chosen for the reconstruction (the five undisclosed choices from research §9.1), *before* any tick data is run through it.
- Any signal family other than §5.2 breakout-continuation requires a new amendment.

## 9. Disclosed limitations, stated before the run

- **Offline queue simulation is an approximation with no ground truth.** We were never in this book. §2.1's primary/secondary queue models bound the uncertainty; they do not eliminate it. A PASS that holds only under the optimistic queue (model c) is **not** a PASS (§6).
- **`mbo` displayed size omits hidden/iceberg liquidity.** For a passive strategy this is conservative; for H1 (marketable orders) it is close to neutral.
- **Fixed latency understates the tail.** Real 250 ms-nominal latency spikes under exactly the conditions a breakout strategy fires. The §2.3 stress overlay is a partial guard, not a fix.
- **MNQ only, 2023–2026.** No cross-instrument, pre-2023, or post-holdout claim is made.
- **If the `mbo` window is shortened for cost (§1 fallback), the walk-forward loses its first fold** and the power table (§7) shifts adverse by ~30%. This is recorded as a power reduction in the relevant amendment, not absorbed silently.
- **A FAIL on H1 does not refute Mesfin's "edge lives inside the bar" claim in general** — it refutes it for the breakout-continuation family, on MNQ, at retail tick realism, over this window. Other intra-bar mechanisms (absorption, sweep-reversal) are untested here and would need their own seals.
- **The parity sample (§3) is dominated by 1-minute-bar strategies** (yank, mim-nb, gap-fade). It validates the *fill model* well but contains few genuinely sub-minute orders; the simulator's behaviour on very fast cancel/replace sequences is therefore less well constrained by parity than its behaviour on bar-triggered orders. Disclosed.

## 10. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `f007a505d7f5854c5bb21d1f6d02ace19e73c547` |
| Working tree at seal | dirty (untracked skill dirs + the research run folder); this document's tamper-evidence rests on its own commit, not on a clean tree |
| Upstream research report | `_bmad-output/planning-artifacts/research/technical-higher-frequency-nq-mnq-intraday-scalpin-2026-08-28/research.md` (status: complete, deepened 2026-08-28) |
| Data vendor / schema | Databento `GLBX.MDP3`, schema `mbo` (fallback `mbp-1`+`trades` for 2023) |
| Development window | 2023-01-01 → 2026-02-28 |
| Sealed tick holdout | 2026-03-01 → present, access-logged |
| Simulator primary models | back-of-queue; 250 ms RT latency; no own-impact; +1-tick stress mandatory |
| Parity gate | N ≥ 100 real trades; MAE ≤ 1.0 tick; p90 ≤ 2.0 ticks; signed bias ≤ ±0.25 tick |
| Infra kill criterion | 3 revision cycles or 15 working days |
| H1 signal family | rolling-`W` breakout, threshold `k` ticks, hold `H` s, stop `S` ticks; symmetric |
| H1 grid | `W`∈{5,15,30,60}m · `k`∈{1,2,4}t · `H`∈{60,180,300}s · `S`∈{4,8,16}t = 108 configs |
| H1 friction (primary) | $4.00 round trip (2.0 index points) |
| H1 economic PASS bar | mean net ≥ +1.0 index pt/trade after $4 friction, under primary AND +1-tick-stress models |
| H1 statistical gate | deflated-Sharpe t ≥ 2.0 (trials = 108) · N ≥ 200 · both vol regimes same-sign · every test year same-sign · permutation p < 0.05 |

---

## 11. Commit instructions (for Alex)

This is a hand-written seal, not a `prereg_seal.py` output (that script is bound to `StrategyConfig` and does not fit an infrastructure pre-registration). To make it tamper-evident:

```bash
git add -f _bmad-output/preregistration_tick_data_infrastructure.md
git commit -m "pre-register TICK-INFRA: MNQ tick data + queue-aware fill simulator (research R3)"
```

Then, in order: (1) purchase the §1 data; (2) build the §2 simulator; (3) run the §3 parity gate and append an amendment with the result and the frozen simulator SHA; (4) only on a parity PASS, run H1 per Part III and append its result amendment.
