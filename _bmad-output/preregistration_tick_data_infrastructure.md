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

---

# Amendment 1 — Cost estimate (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. This amendment records what a spend is expected to look like **before** any data is bought. It changes **no** sealed spec — the §1 fallback path and the §4 kill criterion already bound the downside. Compiled from a web lookup on 2026-08-28; the data-price figures are estimates, flagged as such, because Databento does not publish its historical rate card openly.

## A1.1 Databento data cost — 3 years of MNQ (2023-01-01 → 2026), front-month quarterlies

| Purchase | Estimate | Range | Confidence |
|---|---|---|---|
| **`mbo` (the §1 primary schema)** | **~$3,000–3,600** | $1,900 – $5,000 | low |
| `mbp-1` + `trades` (the §1 fallback) | ~$800 | $550 – $1,450 | low |
| `ohlcv-1m` only (context, not a substitute) | ~$0 | — | covered by the $125 credit |
| New-account credit | **−$125** | one per team, expires 6 months after signup [S1][S3] | confirmed |
| Subscription required for pay-as-you-go historical | **none** [S2] | the $179–199/mo CME "Standard" plan is optional, live-data-oriented, and is **not** a prerequisite for buying historical MBO à la carte [S1][S2] | confirmed |

**Arithmetic (mbo):** `(GB/year × 3) × $/GB − $125`.
- Low: 120 GB × $17 ≈ $1,915
- Point: 195 GB × $19 ≈ $3,580 (less ~15–20% if volume tiers apply → ~$2,900–3,100)
- High: 270 GB × $19 ≈ $5,005

**Dominant uncertainty: the data-volume estimate.** No published or forum-reported GB figure for MNQ (or ES/NQ) MBO was found. The chain is: one hard anchor — a `trades` request for `ESH4`, 2024-02-12→17, "costs $2.17" [S3] → ES `trades` ≈ 25–30 MB/session compressed → ES `mbo` ≈ 15–25× that (every add/modify/cancel, not just fills) ≈ 0.4–0.7 GB/session → MNQ ≈ 30–60% of ES ≈ 0.15–0.35 GB/session → ~40–90 GB/year → ~120–270 GB over 3 years. Each link is unverified and could be ~2× off; they compound. The $/GB rate (~$19 for GLBX `mbo`) is also an unsourced estimate, ±~30%.

**The $42/quarter figure in the upstream research report (`research.md` §1, and echoed in §1 of this seal) is not reliable for `mbo`** — that is `ohlcv`-tier pricing. `mbo` is one to two orders of magnitude more.

## A1.2 The free way to replace this estimate with a firm number

Create a free Databento account and call, at no charge, before any purchase:

```python
metadata.get_cost(dataset="GLBX.MDP3", schema="mbo",
                  symbols=[<front-month list>], stype_in="raw_symbol",
                  start="2023-01-01", end="2026-03-01")
```

This returns the exact dollar cost of the development-window pull. Repeat for the `2026-03-01 → present` holdout and for the `mbp-1`+`trades` fallback. **Do this first.** If the real `mbo` figure lands above a budget the user sets, §1's fallback (mbo for 2024-02-28+holdout, mbp-1+trades for 2023) is the pre-committed response, recorded as a power reduction per §9 — not a silent change.

## A1.3 Compute and storage — the binding constraint on the current machine

Measured on the box this project runs on, 2026-08-28: **64 GB free disk, 7 GB RAM, 2 CPU cores.**

- **3 years of MNQ `mbo` (~120–270 GB) does not fit on disk** and cannot be loaded in memory. Even the `mbp-1`+`trades` fallback (~20–50 GB) is tight against 64 GB alongside the existing repo.
- A full pass of the H1 108-cell grid (Part III §5.3) over 3 years of tick data, with MBO book reconstruction, on 2 cores, is a **multi-day to multi-week** job per sweep. The project's memory already notes CPU is slow and heavy tasks take 1+ hour; a tick-grid sweep is orders beyond that.

Two pre-committed responses, either acceptable, chosen at build time and recorded in the Part II parity amendment:

1. **Stream-process on this box** — never hold a full contract in memory; the simulator and the grid both consume the `mbo` file as a forward iterator. Feasible for the simulator and the parity gate; slow but workable for the grid. Extra cost: ~$0, plus wall-clock measured in days per sweep.
2. **Rent a cloud box for the build + study window** — ~500 GB disk, 16–32 GB RAM, 8–16 cores, for the 1–2 months R3 is active. Roughly **$200–800** total at commodity VM rates. Data is downloaded once to that box.

Storage of the raw vendor files (§1) is included in whichever of the above is chosen; on own disk it is a non-cost, in the cloud it is part of the VM.

## A1.4 Build effort

Not a dollar cost, but the largest cost of R3: the §2 queue-aware simulator — `mbo` order-book reconstruction, back-of-queue position model, latency model, marketable-order fill engine, the §2.3 stress overlay, and the §3 parity harness against `trades.db` — is roughly **1–3 weeks of focused engineering**. Whether measured in the user's time or in assisted-development sessions, this dominates.

## A1.5 All-in expectation

| Path | Data | Compute/storage | Total out-of-pocket (excl. effort) |
|---|---|---|---|
| `mbo`, stream on this box | ~$3,000–3,600 | ~$0 (slow) | **~$3,000–3,600** |
| `mbo`, cloud box for the study | ~$3,000–3,600 | ~$200–800 | **~$3,200–4,400** |
| `mbp-1`+`trades` fallback, cloud box | ~$800 | ~$200–800 | **~$1,000–1,600** |

Plus 1–3 weeks to build the simulator. If it fails the §3 parity gate within the §4 window (15 working days), the spend stops there — data plus a few weeks, nothing else.

**Not included** (downstream of a PASS, not part of R3): a live CME data plan (~$179–199/mo) would be needed only if an H1 PASS led to a successor prospective test or deployment. This seal authorises neither (§8).

## A1.6 Sources

- **[S1]** Databento — Pricing page, `databento.com/pricing`, retrieved 2026-08-28: $125 historical-data credit per team, expires 6 months after signup; usage-based historical requires no subscription (pay-as-you-go); plan prices Standard $199/mo, Plus $1,750/mo, Unlimited $4,500/mo; Standard bundles only ~1 yr L1 / ~1 mo L2–L3 history.
- **[S2]** Databento Blog — "Introducing new CME pricing plans", `databento.com/blog/introducing-new-cme-pricing-plans`, dated 2025-04-16, retrieved 2026-08-28: usage-based *live* CME data discontinued 2025-04-16; "pay-as-you-go pricing for historical data will remain one of Databento's core features"; Standard plan $179/month.
- **[S3]** Databento Blog — "How to request historical market data using Python", `databento.com/blog/api-demo-python`, retrieved 2026-08-28: GLBX.MDP3 `trades` for `ESH4`, 2024-02-12→2024-02-17, "costs $2.17"; $125 historical credit per team.
- **[S4]** Databento Docs — "Metered pricing" and "Historical.metadata.list_unit_prices", `databento.com/docs/api-reference-historical/basics/metered-pricing`, retrieved 2026-08-28: historical responses metered by compressed bytes delivered; per-dataset/per-schema unit prices returned only via the (free) `metadata.list_unit_prices` / `metadata.get_cost` API; no numeric rate card on the public page.
- **[S5]** Databento — GLBX.MDP3 dataset page, `databento.com/datasets/GLBX.MDP3`, retrieved 2026-08-28: schema list (mbo, mbp-1, mbp-10, trades, ohlcv-*); $125 credit; no per-GB rates shown.
- Machine spec (§A1.3): measured locally 2026-08-28 — `df -h`, `free -g`, `nproc`.

---

# Amendment 2 — Staged acquisition plan (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. Refines *how* the §1 data is bought — into two tranches gated on the §3 parity result — so the bulk spend is contingent on the simulator working. The §1 data **spec** (schema, symbol, contract handling, windows) is unchanged. The §4 kill criterion is unchanged and now has teeth: a parity failure stops R3 after Tranche 1 only.

## A2.1 Acquisition method — pay-as-you-go, no subscription

Decided after pricing the Databento CME "Standard" plan ($199/mo new, $179/mo grandfathered): its historical entitlement is **trailing 1 month of L2/L3 (MBO) and trailing 1 year of L1** [S1] — everything older is à-la-carte at the standard rate. A multi-year MBO walk-forward is therefore **not** covered by any monthly plan short of "Unlimited" ($4,500/mo), and a subscription would be paid *on top of* the same historical cost. Pay-as-you-go, à la carte, is the method. No subscription is opened for R3.

## A2.2 Tranche 1 — parity slice only (buy now)

The smallest purchase that lets the §3 parity gate run.

| Item | Value |
|---|---|
| Schema | `mbo` (§1 primary) |
| Symbol | `MNQ.c.0` (front-month continuous by volume; spans the M6→U6 roll) |
| Date range | **2026-05-15 → 2026-09-01** (~3.5 months) |
| Why this range | The parity sample — all live-broker fills in `data/trades.db` from `trader-mim-nb`, `trader-gap-fade`, `trader-s26-combine`, and `trader-yank` (live-combine subset) — spans **2026-06-03 → 2026-08-28**, N ≈ 129 closed trades (above the §3 minimum of N ≥ 100). The range is padded ~3 weeks before the first trade for book warm-up and a few days after the last. |
| Mode | batch flat-files (`historical`) if cheaper per §probe; else streaming |
| Expected cost | to be set by the free `metadata.get_cost` probe (`tools`/scratch `probe_databento_cost.py`, candidate D) before purchase — estimate ~$300–600, low confidence |
| Delivered to | this box if it fits in the ~64 GB free (a 3.5-month `mbo` slice should); otherwise the cloud box per §A1.3 |

**On the parity slice falling inside the sealed holdout window (2026-03-01+):** this is deliberate and is not holdout contamination. The parity gate checks only whether the simulator reproduces *already-known* fill prices from `trades.db`; it develops and tests no strategy, and inspects no forward return. The sealed holdout's purpose — an untouched sample for strategy validation — is not touched. H1's walk-forward (§5.4) still touches the holdout exactly once, later, per the original seal.

## A2.3 Decision gate

Run the §2 simulator build and the §3 parity gate against Tranche 1.

| §3 outcome | Action |
|---|---|
| **PASS** (MAE ≤ 1 tick, p90 ≤ 2 ticks, signed bias ≤ ±0.25 tick) | Record the passing simulator commit SHA (frozen per §4). **Buy Tranche 2.** Proceed to H1. |
| **FAIL**, not fixable within the §4 window (3 revision cycles or 15 working days) | R3 is **closed** per §4. Total spend ≈ Tranche 1 only (~$300–600) plus build time. Write up the unmodelled effect that caused the miss. **Tranche 2 is not bought.** |

## A2.4 Tranche 2 — bulk dev window + holdout (buy only on a parity PASS)

| Item | Value |
|---|---|
| Schema | `mbo` (or the §1 fallback: `mbo` from 2024-02-28 + `mbp-1`+`trades` for 2023, if the §probe puts full `mbo` over the user's budget) |
| Symbol | `MNQ.c.0` |
| Date ranges | dev window **2023-01-01 → 2026-05-15** (butts against Tranche 1, no overlap) + any holdout not already covered by Tranche 1 (**2026-09-01 → present**) |
| Expected cost | §probe candidates A (full) / B (2024+ fallback) / G+H (schema fallback); estimate ~$2,000–3,600 for full `mbo`, ~$700 for the fallback |
| A shorter or cheaper-schema Tranche 2 | is the §1 pre-committed fallback and is recorded as a §9 power reduction — not a silent change |

## A2.5 Next concrete action

1. Create a free Databento account (databento.com → $125 credit).
2. Run `probe_databento_cost.py` (free `metadata.get_cost`) to replace every estimate here with firm figures — especially Tranche 1 (candidate D) and Tranche 2 options (A / B / G+H).
3. Buy Tranche 1.
4. Build §2 simulator → run §3 parity gate → append the result as Amendment 3.
5. Only on a PASS: buy Tranche 2, run H1, append as Amendment 4.

## A2.6 Sources

- **[S1]** Databento — Pricing page, `databento.com/pricing`, retrieved 2026-08-28: Standard plan $199/mo — historical entitlement "1 year of L1 history", "1 month of L2 and L3 history", "Pay as you go for more history"; Plus $1,750/mo adds "16+ years of L1 history" but L2/L3 stays 1 month; Unlimited $4,500/mo = "16+ years in all schemas".

---

# Amendment 3 — VPS / compute cost (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. Prices the machine upgrade the §A1.3 compute constraint implies, staged to match Amendment 2's two tranches. **Numbering note:** this shifts the forward references in §A2.5 — the §3 parity result is now **Amendment 4**, and the H1 result is **Amendment 5**.

## A3.1 Current machine

Measured 2026-08-28: KVM virt, 2 vCPU (AMD EPYC 9354P @ 2.0 GHz), **7.9 GB RAM (5.5 GB used at idle, ~0.24 GB free, no swap)**, 96 GB disk / 64 GB free. This matches **Hostinger KVM 2** (2 vCPU / 8 GB / 100 GB NVMe). RAM is already the binding constraint before any tick work.

## A3.2 Hostinger KVM plan ladder [S1][S2]

| Plan | vCPU | RAM | NVMe | Bandwidth | Promo /mo (24-mo term) | Renewal /mo |
|---|---|---|---|---|---|---|
| KVM 2 (current) | 2 | 8 GB | 100 GB | 2 TB | ~$7–9 | ~$16–20 (est.) |
| **KVM 4** | 4 | 16 GB | 200 GB | 4 TB | **~$13–15** | ~$30–40 (est.) |
| **KVM 8** | 8 | 32 GB | 400 GB | 8 TB | **~$30** | **~$74–78** |

Promo pricing requires a 1/2/4-year prepay; renewal is at standard rate regardless of term. Upgrades are in-place via hPanel at any time; **downgrades are restricted**. KVM 2/4 renewal figures are estimates ("renewal roughly doubles" per reviews); KVM 8 renewal ~$74–78 is reported [S3].

## A3.3 Phase A — now (Tranche 1 + simulator build + §3 parity gate)

**Upgrade KVM 2 → KVM 4.** Justified independently of this project — the box is RAM-starved today.

- 16 GB RAM: room for `mbo` book reconstruction and the streaming parity replay
- 200 GB disk: Tranche 1 `mbo` (~20–40 GB) + repo + parsed/intermediate forms, with headroom
- 4 cores: faster parity replay over the ~129 trades
- **Cost delta over KVM 2: ~+$6/mo during the promo term, ~+$15–20/mo at renewal.** Keep this plan permanently.

## A3.4 Phase B — only on a §3 parity PASS (Tranche 2 + the H1 108-cell grid)

Tranche 2 is 2–3 years of `mbo` (~120–270 GB raw, more with intermediates) that the H1 grid iterates over repeatedly. Two routes:

| Route | What | Cost | Trade-off |
|---|---|---|---|
| **B1 — in-place to KVM 8** | Upgrade KVM 4 → KVM 8 (8 vCPU / 32 GB / 400 GB) | promo ~$30/mo · **renewal ~$74–78/mo** | Simple, one machine. 400 GB may be tight for full `mbo` + intermediates → may force the §1 schema-fallback or the 2-year window. Downgrade is restricted, so you likely carry the ~$76/mo renewal after the project. |
| **B2 — burst box elsewhere, keep Hostinger at KVM 4** *(recommended)* | Rent a no-commitment VM for the 1–2 months H1 runs — e.g. Hetzner CPX51 / CCX (16 vCPU, 32 GB, 360–600 GB), hourly billing, cancel anytime. Download Tranche 2 to it, run the grid, tear it down. | **~$30–110 total** for the whole H1 study | More cores (faster grid), no 24-month lock-in, stable box stays lean. One extra data egress/transfer to set up. |

## A3.5 Revised all-in expectation (supersedes §A1.5)

| Stage | Data | VPS/compute | Running total |
|---|---|---|---|
| Phase A: Tranche 1 + sim build + parity | ~$300–600 (est., candidate D) | KVM 4 upgrade: +~$6/mo × ~1 mo ≈ **+$6–15** | **~$310–615** |
| — if parity FAILS (§4) | — | — | **stop here. ~$310–615 + build time.** |
| Phase B: Tranche 2 + H1 grid (on PASS only) | ~$2,000–3,600 full `mbo` / ~$700 fallback | B2 burst box ~$30–110 (or B1 ~$30 promo, then ~$76/mo ongoing) | **~$2,340–4,325** (B2, full data) |

Plus the KVM 4 upgrade as an ongoing ~+$6/mo (promo) / ~+$15–20/mo (renewal) line — kept regardless of the R3 outcome, since it fixes the current RAM shortage.

**Not included:** a live CME data plan (~$179–199/mo) and any always-on execution box — both downstream of an H1 PASS + a successor seal, neither authorised here (§8).

## A3.6 Sources

- **[S1]** Hostinger — VPS Hosting page, `hostinger.com/vps-hosting`, retrieved 2026-08-28: KVM 1/2/4/8 tiers, KVM virtualization, NVMe, dedicated IP, weekly backups; in-place upgrades via hPanel.
- **[S2]** "Hostinger VPS Pricing 2026: All Plans, Costs and What You Actually Pay", smarthostfinder.com/hostinger-vps-pricing, retrieved 2026-08-28: KVM 4 = 4 vCPU / 16 GB / 200 GB / 4 TB @ ~$14.99/mo; KVM 8 = 8 vCPU / 32 GB / 400 GB / 8 TB @ ~$29.99/mo; promo applies to first term only; upgrades any time via hPanel; no hourly billing.
- **[S3]** bestusavps.com / smarthostfinder / search aggregation, retrieved 2026-08-28: KVM 8 renewal reported ~$73.99–77.99/mo; renewal rates 140–232% of promo depending on plan/term.

---

# Amendment 4 — Actual costs from the Databento API (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. Replaces the **money figures** in §A1.1 and §A3.4/§A3.5 with values from `metadata.list_unit_prices` and `metadata.get_cost` / `metadata.get_billable_size` (all free calls, nothing purchased), run 2026-08-28 against MNQ front-month continuous (`MNQ.c.0`). The **method** (staged, §A2) and the **machine ladder** (§A3.2) are unchanged. Data spec unchanged.

**Numbering:** with this as Amendment 4, the §3 parity result becomes **Amendment 5** and the H1 result **Amendment 6** (superseding the shifts noted in §A3).

## A4.1 The real rate card — GLBX.MDP3 historical ($/GB, uncompressed)

| schema | $/GB | note |
|---|---|---|
| **`mbo`** | **$1.80** | ~10× lower than the §A1.1 estimate |
| `mbp-1` | $1.80 | |
| `mbp-10` | $0.50 | but emits a full 10-level snapshot per book change → far more GB (see A4.3) |
| `trades` | $28.00 | tiny volume, high unit price |
| `ohlcv-1m` | $70.00 | |

Batch and streaming are the **same price** for this dataset; delivery encoding (`dbn` vs `csv`) does not change the bill — Databento meters the **uncompressed record bytes**, not the download size.

## A4.2 Actual cost by candidate ( `metadata.get_cost`, MNQ.c.0 )

| # | Pull | Billable (uncompressed) | Cost |
|---|---|---|---|
| **D** | **`mbo` parity slice 2026-05-01 → 2026-09-01** | 240 GB | **$403** |
| A | `mbo` dev window 2023-01-01 → 2026-03-01 | 1,097 GB | $1,839 |
| C | `mbo` 2026-03-01 → 2026-08-28 (holdout) | 338 GB | $566 |
| — | **`mbo` EVERYTHING 2023-01-01 → 2026-08-28** | **1,435 GB** | **≈ $2,405** (A + C) |
| B | `mbo` 2024-01-01 → 2026-03-01 (2-yr fallback) | — | $1,393 |
| E | `mbp-10` dev window | — | **$2,851** — *more* than `mbo`, and less information |
| G / H | `mbp-1` / `trades` dev window | — | $1,664 / $686 |
| K | `ohlcv-1m` dev window (context only) | — | **$4** |

**The §1 fallbacks are dead.** `mbp-10` costs more than `mbo`; `mbp-1`+`trades` costs about the same as `mbo` and loses the full book. **`mbo` is both the best data and effectively the cheapest full-depth option.** The §1 "cheaper-schema fallback" is withdrawn; the only surviving lever is the date window (B) and the session filter (A4.4).

Credit: −$125 new-account, applied once.

## A4.3 RTH-only saves ~35%

Measured over a normal week (`get_billable_size`, per-day, 13:30–20:00 UTC = 09:30–16:00 ET): **RTH ≈ 62–65% of full-session `mbo` volume** (a normal day is 60–75%; holiday/half-days lower the weekly average). So an RTH-only download costs **~65%** of the full-session price.

H1 (Part III) is an RTH strategy, so H1 needs RTH-only. H2 (London/Asia regime reconstruction) would need the other sessions — bought later under H2's own amendment, not now.

## A4.4 Revised purchase plan (supersedes §A2.2 / §A2.4 dollar figures)

| Tranche | Pull | Cost | After −$125 credit |
|---|---|---|---|
| **1 — buy now** | `mbo` **full-session** (parity needs full book context), 2026-05-01 → 2026-09-01 | **$403** | **~$278** |
| **2 — on §3 parity PASS only** | `mbo` **RTH-only**, 2023-01-01 → 2026-05-01 (H1 is RTH) | ~$1,150–1,250 | — |
| (later, only if H2 is greenlit) | `mbo` non-RTH sessions, same span | ~$650 | — |

- **All-in for H1 (RTH path): ~$1,450–1,550 for data.**
- If you prefer to just buy everything full-session up front and skip the RTH bookkeeping: **~$2,280 net.**
- 2-year window (candidate B) instead of 3-year saves ~$450 and costs one walk-forward fold (§1 fallback, recorded as a §9 power reduction).

## A4.5 Storage — revised (supersedes the §A3 disk concern)

Billable size is **uncompressed**. On disk, `mbo` delivered as `dbn.zst` compresses roughly 6–10×. Databento's reader streams the compressed files, so the uncompressed 1.4 TB is never needed at rest.

| On disk (compressed `.dbn.zst`, est.) | ~size | Fits |
|---|---|---|
| Tranche 1 (240 GB billable) | **~25–40 GB** | **current KVM 2** (64 GB free) — no upgrade needed to start |
| Tranche 2 RTH-only (~700 GB billable) | **~70–120 GB** | **KVM 4** (200 GB) comfortably |
| Everything full-session (1,435 GB billable) | ~150–240 GB | KVM 4 tight / KVM 8 comfortable |

**Consequence for §A3.3:** the KVM 2 → KVM 4 upgrade is **not required to run Tranche 1, the simulator build, or the §3 parity gate** — those fit the current box on disk. The upgrade is still worth doing for **RAM** (the box is at ~0.2 GB free) and is needed for the H1 grid in Phase B. Treat it as: optional now, required before Tranche 2.

## A4.6 Revised all-in (supersedes §A1.5 and §A3.5)

| Stage | Data | Machine | Running total |
|---|---|---|---|
| Phase A: Tranche 1 + sim build + §3 parity | **~$278** | $0 (current box) — or +$6–15 if you take KVM 4 early | **~$280–295** |
| — if parity FAILS (§4) | — | — | **stop. ~$280 + build time.** |
| Phase B: Tranche 2 (RTH) + H1 grid (on PASS) | ~$1,150–1,250 | KVM 4 (~+$15–20/mo) + optional burst box ~$30–110 for the grid | **~$1,450–1,700** total |
| (full-session up-front variant) | ~$2,280 | as above | ~$2,550–2,800 |

Down from the §A1.5 / §A3.5 figure of ~$2.3–4.4k. The dominant cost is now the **~1–3 weeks of simulator build effort**, not the data.

## A4.7 Security note

The API key used for these free metadata calls was shared in plain text in the working session and is therefore in the transcript. It was used only for read-only `metadata.*` calls (no data purchased, no `batch.submit_job`). Recommend rotating it in the Databento portal before it is used for an actual purchase.

## A4.8 Sources

- Databento Historical API, `metadata.list_unit_prices` / `metadata.get_cost` / `metadata.get_billable_size`, `hist.databento.com/v0`, called 2026-08-28 with a valid key. Raw output retained at `~/.claude/jobs/960bda86/tmp/probe_out.txt` (session-local).

---

# Amendment 5 — Tranche 1 request locked + symbology clarification (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. Locks the exact Databento request for Tranche 1 so the purchase matches the seal, and resolves a symbology ambiguity in §1. Costs confirmed via the free `metadata.get_cost` API, 2026-08-28.

**Numbering:** with this as Amendment 5, the §3 parity result becomes **Amendment 6** and the H1 result **Amendment 7**.

## A5.1 Symbology clarification (resolves §1)

§1 lists two things that do not agree: the symbol `MNQ.c.0` (a *calendar*-roll continuous) and the front-month definition "the contract with the greater cumulative volume on the prior session" (a *volume* roll). The **volume roll is the substantive spec**; `MNQ.c.0` was imprecise shorthand. Combined with §1's "per-contract, no stitching … `groupby(contract)`", Tranche 1 is acquired as **explicit raw contracts**, not a continuous symbol.

## A5.2 Tranche 1 — locked request

| Parameter | Value |
|---|---|
| Dataset | `GLBX.MDP3` |
| Symbols | `MNQM6, MNQU6, MNQZ6` (Jun / Sep / Dec 2026; MNQZ6 is ~$15 insurance against an early roll — no live trade rolled that early) |
| `stype_in` | `raw_symbol` |
| Schema | `mbo` |
| Start | `2026-05-01` |
| End | `2026-08-28` (dataset availability ends 2026-08-28 22:30 UTC) |
| Encoding / compression | `dbn` / `zstd` |
| Split | by day |
| **Confirmed cost** | **$453.12** (billable 270.3 GB uncompressed) — **−$125 credit → $328.12 charged** |
| On-disk footprint | ~25–40 GB `.dbn.zst` — fits the current KVM 2 |

Cheaper alternatives on record, not chosen: `MNQM6,MNQU6` only = $437.80; `MNQ.v.0` volume-continuous = $412.65 (single front stream, still `instrument_id`-tagged, but no non-front book around the roll).

## A5.3 Known data-quality flags in the window

Databento `get_dataset_condition` marks **2026-05-24** and **2026-07-30** as `degraded` (still `available`). Both must be noted in the §5 integrity report; neither is excluded.

## A5.4 Operational guide

Step-by-step purchase and download instructions: `_bmad-output/tick_infra_tranche1_purchase_guide.md` (not part of the seal; editable).

## A5.5 Source

- Databento `metadata.get_cost` / `metadata.get_billable_size` / `metadata.get_dataset_condition`, `hist.databento.com/v0`, called 2026-08-28. `MNQM6,MNQU6,MNQZ6` raw, `mbo`, 2026-05-01→2026-08-28 → $453.12 / 270.3 GB.

---

# Amendment 6 — Correction: on-disk size of Tranche 1 (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. **Corrects an error in §A5.2 and §A4.5.**

**Numbering:** with this as Amendment 6, the §3 parity result becomes **Amendment 7** and the H1 result **Amendment 8**.

## A6.1 The error

§A4.5 and §A5.2 claimed Tranche 1 would be "~25–40 GB on disk", assuming `mbo` `.dbn.zst` compresses ~7–10×. **That ratio is wrong for MBO.** It is the ratio for `trades` / `ohlcv` (highly redundant). An MBO record is a fixed **56-byte** DBN struct, of which ~32 bytes (two nanosecond timestamps `ts_event`/`ts_recv`, the 64-bit `order_id`, `sequence`, `ts_in_delta`) are high-entropy and barely compress. Verified: `record_count × 56 B` reproduces every billable figure exactly (25.6 B records → 1,435 GB whole dataset; ~4.8 B records → 270 GB Tranche 1).

Realistic zstd on MBO DBN blends the compressible fields (constant `instrument_id`/`publisher_id`, clustered `price`, small `size`, 2-symbol `action`/`side`) with the incompressible ~32 bytes → **≈ 2.5–3.5× overall**.

## A6.2 Corrected figures

| | Uncompressed (billable) | On disk `.dbn.zst` (est., 3×) |
|---|---|---|
| Tranche 1 (`MNQM6,MNQU6,MNQZ6`, 2026-05-01→08-28) | 270 GB | **~80–110 GB** |
| Tranche 2 RTH-only (~700 GB billable) | 700 GB | ~200–280 GB |
| Everything full-session | 1,435 GB | ~410–575 GB |

The Databento portal's "270 GB" estimate is the **uncompressed** size. Setting the batch `compression` field to `zstd` yields the ~80–110 GB download; `compression=none` downloads the full 270 GB.

## A6.3 Consequence for the machine (corrects §A4.5 / §A3.3)

**Tranche 1 does NOT fit the current KVM 2 (64 GB free).** The KVM 2 → KVM 4 (200 GB) upgrade in §A3.3 is now **required before Tranche 1**, not optional. KVM 4 holds Tranche 1 (~100 GB) with room for parsed/intermediate forms. Tranche 2 RTH-only (~200–280 GB compressed) needs **KVM 8 (400 GB)** or an external volume.

Revised machine line: **KVM 4 from the start** (~+$6/mo promo, ~+$15–20/mo renewal); KVM 8 or a burst box for Phase B per §A3.4.

## A6.4 Mandatory de-risking step before the $328 purchase

Submit **one day, one contract** as a batch job first:

```python
job = c.batch.submit_job(
    dataset="GLBX.MDP3", symbols=["MNQM6"], stype_in="raw_symbol",
    schema="mbo", start="2026-06-16", end="2026-06-17",
    encoding="dbn", compression="zstd", split_duration="day")
```

Cost ~**$1–2**. It gives (a) the **exact** zstd ratio for MNQ MBO — multiply out to the real Tranche 1 disk size — and (b) an end-to-end test of account → billing → job → download → `DBNStore` read → the §5 integrity checks, on a trivial sample, before committing $328. Record the measured ratio and pipeline result in Amendment 7.

## A6.5 Revised all-in (corrects §A4.6)

| Stage | Data | Machine | Running total |
|---|---|---|---|
| Test download (§A6.4) | ~$2 | — | ~$2 |
| Phase A: KVM 4 upgrade + Tranche 1 + sim build + parity | ~$328 + ~$6–15 (1 mo KVM 4) | | **~$335–345** |
| — if parity FAILS (§4) | — | — | **stop. ~$335 + build time.** |
| Phase B: Tranche 2 (RTH) + H1 grid (on PASS) | ~$1,150–1,250 | KVM 8 or burst box $30–110 | **~$1,550–1,750** total (RTH path) |

Storage is now the constraint that forces the machine spend earlier, but the dollar totals are within ~$100 of §A4.6. Build effort still dominates.

---

# Amendment 7 — Tranche 1 re-scoped to the parity footprint (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. Re-scopes the Tranche 1 **purchase** (locked in §A5.2) down to the data the §3 parity gate actually touches. The parity gate's purpose, sample, and pass tolerances (seal §3) are unchanged. Machine is the just-upgraded **KVM 4 (193 GB disk, 162 GB free, 15 GB RAM, 4 cores)**.

**Numbering:** with this as Amendment 7, the test-download findings (§A7.4) become **Amendment 8**, the §3 parity result **Amendment 9**, and the H1 result **Amendment 10**.

## A7.1 The realisation

The §3 parity gate replays **129 fixed live trades** from `trades.db` (`trader-mim-nb`, `trader-gap-fade`, `trader-s26-combine`, `trader-yank` live-combine subset), 2026-06-03 → 2026-08-28. Each needs the MBO stream for a window around it. Merging ±90-minute windows over the 129 trades gives **87 windows totalling ~299 hours** of MBO — roughly **13 full-session-days-equivalent, ~6–16 GB** of data, versus the 270 GB / $453 contiguous slice locked in §A5.2.

The window boundaries are determined entirely by the fixed trade timestamps — there is **no window-selection freedom**, so this is not cherry-picking; it is the minimal sufficient data for the stated test.

## A7.2 Re-scoped Tranche 1 — two acquisition modes, chosen after the test (§A7.4)

| Mode | What | Est. cost | Est. disk (`.zst`) |
|---|---|---|---|
| **C — targeted windows** | `timeseries.get_range` (streaming), per merged window, `MNQM6`/`MNQU6` as active, ±90 min around each trade | **~$30–90** | ~3–8 GB |
| **C' — parent days** | full sessions for the ~13–20 distinct calendar days the windows fall on (use if Databento MBO does not prepend a book snapshot for intraday starts — see §A7.3) | **~$25–55** | ~5–12 GB |
| A — full contiguous slice (the §A5.2 lock) | `MNQM6,MNQU6,MNQZ6`, 2026-05-01 → 2026-08-28, batch | $453.12 | ~80–110 GB — **still fits the 162 GB free** |

Mode A remains valid and fits the upgraded box; it is the fallback if C / C' prove fiddly. The saving from C / C' is ~$350–420 and it keeps the box empty for Tranche 2.

## A7.3 The open question the test must answer

Databento MBO is an incremental stream. For a fill simulator to know the book at time *T* it must replay from a point where the book is **complete**. Databento reconstructs the book at a requested `start` for session-boundary starts; whether it prepends a full snapshot for an **arbitrary intraday `start`** must be confirmed, not assumed. If it does → Mode C (±90 min windows). If it does not → Mode C' (whole parent sessions, book builds from the open).

## A7.4 Revised test-download (replaces §A6.4) — one mid-session window, ~$2

```python
job = c.batch.submit_job(
    dataset="GLBX.MDP3", symbols=["MNQM6"], stype_in="raw_symbol",
    schema="mbo", start="2026-06-16T14:00", end="2026-06-16T15:00",   # mid-session
    encoding="dbn", compression="zstd", split_duration="day")
```

From the delivered file, record:
1. **zstd ratio** — file bytes ÷ (record_count × 56) → the real Tranche 1 disk size under Mode A.
2. **Book completeness at an intraday start** — do the first records rebuild a full book (`action='R'` then a burst of `action='A'`), or does the stream start mid-incremental? This picks Mode C vs C'.
3. **Pipeline** — billing → job → `batch.download` → `db.DBNStore.from_file` → the seal §5 integrity checks, end to end.

Append the three findings to the seal as **Amendment 8**, then proceed with the chosen mode.

## A7.5 Revised all-in (corrects §A6.5)

| Stage | Data | Machine | Running total |
|---|---|---|---|
| Test download (§A7.4) | ~$2 | KVM 4 already done | ~$2 |
| Phase A: Tranche 1 (Mode C/C') + sim build + parity | **~$30–90** | KVM 4 (~+$6/mo) | **~$40–100** |
| — if parity FAILS (§4) | — | — | **stop. ~$40–100 + build time.** |
| Phase B: Tranche 2 (RTH bulk) + H1 grid (on PASS) | ~$1,150–1,250 | KVM 8 or burst box $30–110 | **~$1,250–1,450** total (RTH path) |

Phase A drops from ~$335 to **under $100**. The §4 kill criterion now costs almost nothing to reach.

---

# Amendment 8 — Parity gate: sample corrected, threshold lowered, invariant battery added (2026-08-28, pre-purchase)

Append-only. Original sealed text unedited. **Corrects a defect in the seal §3 and in §A5/§A7:** the parity sample named in §3 is smaller and partly invalid. This amendment fixes the sample, lowers the real-fill N to what exists, adds a large-N invariant check as the second half of the gate, and re-scopes Tranche 1 to match.

**Future-amendment numbering is no longer pre-assigned** (the pre-assignments in §A2.5, §A3, §A6, §A7 kept shifting). Amendments are numbered in the order appended. The next appended will be the test-download findings, then the parity-gate result, then H1.

## A8.1 The defect

Inspection of `data/trades.db` on 2026-08-28:

| Named in seal §3 | Status | Real MNQ live-broker fills |
|---|---|---|
| `trader-mim-nb` | valid — real combine fills, hash-chained (`data/mim_nb/trades.csv`) | **22** |
| `trader-yank` (live-combine subset) | valid — combine account, 2026-06-17 onward | **8** |
| `trader-gap-fade` | **invalid** — every row carries `metadata.simulated = true`; its "fills" are computed by gap-fade's own backtest logic, not a broker. Checking a new simulator against an old simulator is not a ground-truth test. | 0 |
| `trader-s26-combine` | **invalid** — not MNQ. Prices 59,755–80,295 (vs ~28–30k for MNQ in the same window); `metadata.paper = true`. It is a paper strategy in the s26 crypto family. | 0 |

**Real parity sample: 30 fills** (`trader-mim-nb` all + `trader-yank` ≥ 2026-06-17), spanning 2026-06-11 → 2026-08-28. Not the ≥100 the seal §3 assumed.

## A8.2 Revised §3 — a two-part gate

### Part A — real-fill calibration (corrects §3)

- **Sample:** every closed `trader-mim-nb` fill + every `trader-yank` fill dated ≥ 2026-06-17, from `data/trades.db` (cross-checked against `data/mim_nb/trades.csv`). **N ≈ 30**, growing ~1–2/day as both bots trade live.
- **Minimum to run Part A: N ≥ 28.**
- **Tolerances — unchanged from §3:** mean absolute fill-price error ≤ 1.0 tick; 90th-pct absolute error ≤ 2.0 ticks; mean signed error within ±0.25 tick.
- **What N ≈ 30 can and cannot do:** at these tolerances it detects a systematically wrong or biased fill model (a 0.4-tick bias is ~4 standard errors at N=30) and catastrophic per-fill errors (they blow the MAE). It is weak on rare, conditional errors. That gap is covered by Part B and by the v2 re-run (§A8.4).

### Part B — synthetic invariant battery (new)

Independent of any real outcome. Generate **≥ 1,000** synthetic orders (mix of marketable and passive limit, both sides, sizes 1–5) at random timestamps across the Tranche 1 data. **Every one of these must hold, 100%:**

1. A buy never fills better than the best offer at (arrival + configured latency); a sell never better than the best bid.
2. A passive limit never fills at a price through its limit.
3. Fill timestamp ≥ submit timestamp + configured latency, always.
4. Reconstructed queue position is non-negative and non-increasing until the order fills or is cancelled.
5. Cumulative partial fills ≤ order size; no fill occurs when the book has no liquidity at or through the order's price.
6. Only book events with `ts ≤ simulated-now` are used to decide any fill (strict causality — no lookahead).

Any violation = **Part B FAIL**, regardless of Part A.

### Verdict

**PASS requires Part A (N ≥ 28, all three tolerances) AND Part B (all six invariants, 100%).** A Part A pass with a Part B failure is a FAIL — the fill model is structurally broken. A Part B pass with a Part A failure is a FAIL — the model runs but is miscalibrated.

The §4 kill criterion (3 revision cycles / 15 working days, then R3 closes) applies to the combined gate.

## A8.3 Tranche 1 re-scoped again (supersedes §A7.2)

The 30-fill sample merges into **28 windows (±90 min), 86 hours of MBO, 24 distinct calendar days**.

| | Value |
|---|---|
| Symbols | `MNQU6` for all windows except 2026-06-11 (`MNQM6`); pull **both** `MNQM6` and `MNQU6` for any window within ±3 days of the 2026-06-19 expiry and keep whichever contract the live order actually hit |
| Windows | 28 (list in `~/.claude/jobs/960bda86/tmp/parity_windows.py` output; regenerate from `trades.db`) |
| Est. cost | **~$40–95** (86 h × ~0.25–0.6 GB/h × $1.80/GB) — confirm with summed `metadata.get_cost` before pulling |
| Est. disk | ~10–25 GB uncompressed → ~4–10 GB `.dbn.zst` |
| Note | window 17 falls on 2026-07-30, a `degraded` day — flag in the integrity report, do not drop |

Modes C / C' / A from §A7 still apply (targeted `get_range` vs parent days vs full slice). The full slice (A, $453) remains the fallback.

## A8.4 Parity gate v2 — the deferred strong check

Part A at N ≈ 30 is provisional. When `trader-mim-nb` + `trader-yank` have produced **≥ 100** combined real fills (projected ~Nov–Dec 2026 at the current rate), re-run **Part A at N ≥ 100** on the same tolerances, buying the incremental windows (~$1–2 each).

- If v2 **passes**: the simulator is confirmed; nothing else changes.
- If v2 **fails** where v1 passed: any H1 result computed against Tranche 2 in the interim is **quarantined** pending a fix and a re-run. This is why Part B exists — to make an interim H1 result unlikely to be built on a silently broken simulator.

v2 gates the *interpretation* of H1, not the *purchase* of Tranche 2.

## A8.5 Not affected

H1's own acceptance gate (§5.5 / §6: N ≥ 200 trades across ≥ 2 volatility regimes, etc.) counts trades the *strategy* generates in the walk-forward — it is unrelated to the parity sample and does not change.

## A8.6 Revised all-in (supersedes §A7.5)

| Stage | Data | Machine | Running total |
|---|---|---|---|
| Test download | ~$4–5 (2026-06-22 window, doubles as 3 parity trades) | KVM 4 done | ~$5 |
| Phase A: Tranche 1 (28 windows) + sim build + parity gate (A+B) | ~$40–95 | KVM 4 (~+$6/mo) | **~$50–105** |
| — if the gate FAILS (§4) | — | — | **stop. ~$50–105 + build time.** |
| Phase B: Tranche 2 (RTH bulk) + H1 grid (on PASS) | ~$1,150–1,250 | KVM 8 or burst box $30–110 | **~$1,250–1,450** total |
| Parity gate v2 (~Nov–Dec) | ~$50–150 incremental windows | — | +~$100 |

---

# Amendment 9 — Test-download findings (2026-08-29)

Append-only. Original sealed text unedited. Results of the ~$2 test download mandated by §A6.4 / §A7.4 / guide Step 2. Job `GLBX-20260829-LVAEF8VMHE`: `GLBX.MDP3`, `MNQ.FUT` (parent), `mbo`, 2026-06-22 14:30–17:00 UTC, `dbn`/`zstd`, cost **$2.11**, 22,506,972 records.

## A9.1 zstd ratio — 3.46×

Uncompressed (billed): 1,260,390,432 B. Compressed (downloaded): 364,585,575 B. **Ratio 3.46×** — at the top of the Amendment 6 estimate (2.5–3.5×).

| | uncompressed | on disk |
|---|---|---|
| Re-scoped Tranche 1 (§A8.3, ~86 h MBO) | ~39 GB | **~11 GB** |
| Full slice (Mode A) | 270 GB | ~78 GB |
| Tranche 2 RTH bulk | ~700 GB | ~200 GB |

Measured rate: **~0.50 GB/hour** of `MNQ.FUT` MBO (0.48 front-month only).

## A9.2 Intraday book reconstruction — Mode C confirmed viable

The stream **does not** prepend a snapshot at an arbitrary intraday `start`: no `action='R'`, no `F_SNAPSHOT` flag; the first record is a normal incremental event 1 ms after the requested start, and early records include cancels for orders added before the window.

**However**, over the full 2.5 h window only **0.3 %** of cancel/modify events (34,514 of ~11.5 M) reference an unseen order, and these concentrate in the first ~60 s. At each real trade time — well inside the window — the reconstructed book is **deep and fully populated**: 6,810–8,585 resting orders, 2-tick spread, 3–11 orders on each of the top five levels per side.

**Verdict: Mode C (targeted ±90 min windows) is sufficient.** With the window's built-in ≥30–90 min of warm-up before each trade, the near-touch book — the only part that determines a marketable-order fill — is completely reconstructed. The residual ~0.3 % missing orders are deep and stale. **Mode C′ (full-session pulls) is not required.** §A8.3 stands.

## A9.3 Integrity (seal §5)

| Check | Result |
|---|---|
| Record count vs metadata | 22,506,972 = 22,506,972 ✓ |
| SHA-256 vs manifest | match ✓ |
| Timestamps non-decreasing | **0 violations** in 22.5 M records ✓ |
| Front-month trade-price range | 30,502.75 – 30,763.25 — tight, sensible ✓ |
| BBO non-crossing | 2,792 crossed instants / 19.67 M checks = **0.014 %**, all transient |

**The §5 check "`bid ≤ ask` on 100.000 % of book states" is too strict and is refined here:** raw CME MBO contains momentary locked/crossed markets (resolved by the matching engine in microseconds) that are *not* data errors; and a correct book requires processing `action='F'` (fill) events, not just A/C/M. The refined check: **no *persistent* cross (> a few ms), and the reconstruction must consume A/C/M/T/F.** The 0.014 % here is consistent with normal transient crossing plus this quick pass ignoring `F`.

## A9.4 Data shape (feeds the simulator design)

- Actions: A 8.95 M / C 8.95 M (balanced) / M 2.63 M / T 0.50 M / F 0.84 M / N 0.63 M.
- `MNQ.FUT` parent = the front month (instrument_id 42004800) is **96 %** of records; the remaining 4 % is one active spread/back-month plus negligible others. Pulling `MNQ.FUT` parent vs the explicit front-month raw symbol costs ~4 % more and **removes the roll-symbol logic entirely** — recommended for all Tranche 1 windows. This supersedes §A8.3's `MNQM6`/`MNQU6` switching.
- Records carry `instrument_id` (`stype_out=instrument_id`, `map_symbols=false`); the DBN metadata holds the id→contract map (`store.symbology` / `metadata.json`). The simulator must key the book by `instrument_id` and resolve the map.
- MNQ RTH spread in-window: **2 ticks (0.50 index pt)** — consistent with the seal's conservative $4 (2-pt) round-trip friction, not optimistic.

## A9.5 Pipeline

End-to-end works: free metadata calls → `batch.submit_job` (web) → `batch.list_files` → HTTPS download → SHA verify → `db.DBNStore.from_file` → streaming iteration → book reconstruction. `databento` **0.85.0** installed into `.venv` (via `pip`, for this analysis) — **must be added to `pyproject.toml`** before the simulator build.

## A9.6 Revised Tranche 1 cost

At the measured 0.50 GB/h: **~86 h × 0.5 GB/h × $1.80 = ~$77** for the 28 windows (Mode C, `MNQ.FUT` parent), ~$70 front-month-only. −$2.11 already spent on the test (which is 3 of the 28 windows: the 2026-06-22 window covering all three yank fills). Net remaining ~**$68–75**. On disk ~11 GB. Confirm with summed `metadata.get_cost` before pulling.

## A9.7 Next

Build the §2 simulator **via the BMAD method** (`bmad-architecture` → `bmad-build` → `bmad-review`; project standing rule 2026-08-29). Then pull the remaining 25 Tranche-1 windows, run the two-part parity gate (§A8.2), append the result.

# Amendment 10 — Parity gate result, cycle 1 (2026-09-03)

Append-only. Original sealed text unedited. First real run of the §A8.2 gate, on the completed Tranche-1 window map (38 windows: 28 base + 10 top-up, `_bmad-output/parity/gate_windows.json`) against simulator commit `f2769ee`.

**Verdict: FAIL.** Part B PASS. Part A FAIL — N=36 (32 `trader-mim-nb` + 4 `trader-yank`, ≥ the N≥28 floor): MAE 8.0556 ticks (tol 1.0), p90 13.5000 (tol 2.0), signed bias −1.7222 (tol ±0.25). Per-trader: mim-nb MAE 8.6562 / bias −1.9688 (32 legs); yank MAE 3.2500 / bias +0.2500 (4 legs, control). Part B: 1000 synthetic orders, 1188 fill events, zero invariant violations across all six labels.

Root cause, established by two follow-up diagnostics (book-vs-tape agreement check; `mim_nb_live.log` correlation): the reconstructed book and the real CME trade tape **agree** at every fill instant (MAE 14.0 vs 15.4 ticks, both ~14 ticks off the *recorded* mim-nb fill price) — a wrong book would disagree with the tape, so the book is not the defect. The defect is `orders.csv`'s FILL-row timestamp: mim-nb polls `/Trade/search` to detect fills and logs the FILL row when that poll returns, ~3.1s after the true PLACE instant on every market order. ProjectX `creationTimestamp` cross-reference (n=10, 2026-08-13+) confirms the true execution instant is the PLACE timestamp to within ~50ms — book-BBO-vs-fill MAE drops from 12.6 ticks (orders.csv FILL ts) to 1.5 ticks (ProjectX ts) on the same 10 fills, same book.

Full raw gate output (per-window integrity flags, coverage notes, fill-source manifest): `_bmad-output/parity/amendment_10_parity_gate_cycle1.md` (this run's CLI artifact, preserved verbatim).

Cycle 1 of 3 (§4 clock).

# Amendment 11 — Parity gate result, cycle 2 (2026-09-03)

Append-only. Original sealed text unedited. Re-run after the fix implied by Amendment 10's root cause: `reconstruct_mim_nb` now times every mim-nb market leg's `RealFill.ts_ns` by the order's PLACE instant (`intent.submit_ts_ns`), not the FILL-row poll-return; the FILL row still supplies price. The one otype-4 stop-out exit leg (no PLACE-equivalent trigger instant) is timed from its ProjectX `creationTimestamp` when available, via a caller-supplied mapping the CLI threads in from `projectx_fills.json` — kept in the sample rather than dropped, after a review round flagged dropping it as the project's documented restrict-to-favorable-subset pattern; the leg's own error (2 ticks on ProjectX ts vs 59 ticks on the poll-late `orders.csv` ts) argued for keeping and re-timing it, not for excluding it. Simulator commit `b2c12b3`.

**Verdict: FAIL.** Part B PASS (1000 orders, 1188 fills, zero violations, unchanged). Part A FAIL — same N=36: MAE 6.6111 ticks (tol 1.0), p90 12.0000 (tol 2.0), signed bias **+0.2778** (tol ±0.25). Per-trader: mim-nb MAE 7.0312 / bias −0.3438 (32 legs); yank MAE 3.2500 / bias +0.2500 (4 legs, unchanged — yank never touched the fix, it was never mistimed).

The fix worked exactly as diagnosed: signed bias moved from −1.7222 to +0.2778 — the ~3.1s poll-lag was the dominant source of the *directional* error, and removing it collapsed the bias to within 0.03 ticks of tolerance. **Dispersion did not clear**: MAE fell only 18% (8.06 → 6.61), still 6.6× tolerance; p90 still 6× tolerance. The remaining error is not a timing-source bug — it is investigated in Amendment 12.

Full raw gate output: `_bmad-output/parity/amendment_11_parity_gate_cycle2.md` (this run's CLI artifact, preserved verbatim).

Cycle 2 of 3 (§4 clock).

# Amendment 12 — §4 kill criterion invoked: R3 closed (2026-09-04)

Append-only. Original sealed text unedited. Per §4: *"If the simulator cannot pass §3 within a bounded effort — 3 revision cycles or 15 working days from first parity run, whichever first — the R3 path is abandoned and recorded as such. The specific unmodelled effect that causes the miss is written up (magnitude, direction, why it could not be bounded)."* This amendment is that write-up. **R3 is closed after cycle 2 of the allotted 3** — not because the clock ran out, but because no third-cycle hypothesis survived scrutiny; spending a cycle without a specific, falsifiable change to test would not be a revision, it would be running the clock for its own sake.

## A12.1 What was ruled out

**Not the calibration sample.** §3 (original) and §A8.2 (Amendment 8) are explicit that Part A's job is to reproduce *already-known* real fill prices from whichever live broker trades exist — "it develops and tests no strategy" (line 317) — not to be a representative slice of H1's future trading distribution; H1's own acceptance gate (§5.5/§6, N≥200 walk-forward trades) is unrelated to the parity sample and unaffected by this closure (line 716). The seal's own disclosed representativeness gap is about **order duration** (few genuinely sub-minute orders in a sample dominated by bar-triggered strategies, line 183), not about *time-of-day*. mim-nb's fixed clock-tick schedule is in the sample for the only reason the seal requires: it produced real, verifiable broker fills. Five of those fills landing in high-message-rate windows and missing tolerance is the frozen model's blind spot surfacing on contact with real data, not a sampling artifact — there is no cheaper, non-selective way to re-scope the sample that would change this finding.

**Not the book, not the fill logic, not the timestamp source.** The mechanism diagnostic (`_bmad-output/parity/` job artifacts, reproduced below) replayed each of the 36 fills against the purchased MBO tape at 10ms resolution over a ±1s window around its recorded time, searching for *any* latency offset that reconciles the simulator's fill against the real touch. **34 of 36 legs achieve exact 0-tick reconciliation somewhere in that window; 35 of 36 reach ≤1 tick** (median best-achievable error: 0.0 ticks). This rules out a data or book-construction defect — the correct fill is on the tape, reachable, for essentially every leg.

## A12.2 What could not be bounded

The offset that achieves each leg's best reconciliation is **not a constant**: across the 36 legs it ranges from **−40ms to −990ms**, spanning nearly a full second, with no clustering around any single value including the frozen 250ms. A companion offset sweep (0/25/50/100/250/500/1000/2000ms, applied uniformly to all 36 legs) confirms no single fixed offset works: MAE is minimized at **offset = 0ms (MAE 3.667, p90 9.0, bias +0.111)** — still **3.7× the MAE tolerance** with the *theoretical best-case* single latency constant — and MAE degrades monotonically as the assumed latency grows past that point (250ms: MAE 6.611, matching the real gate to 4 decimals; 2000ms: MAE 13.0). No fixed scalar, at any value, clears tolerance.

The legs with the widest own-market touch-range during their reconciliation window are also the legs the fixed model misses worst: `mimnb-3341188866#ENTRY` (touch range 2500 ticks — a 2026 macro-print spike), `mimnb-3280933244#ENTRY` (278 ticks), `mimnb-3359791379#ENTRY` (208 ticks) — versus a median touch-range of 27.5 ticks across the full sample. The residual concentrates exactly where real MNQ execution risk concentrates: scheduled news releases and the cash-close window, i.e. where **real retail latency itself is least likely to be a fixed constant** — congestion, requoting, and matching-engine load all spike together with market velocity.

This is precisely the effect the seal declined to model, in writing, before any of this data was purchased:

> §2.2, item 2: *"**Variable / bursty latency.** Real retail latency spikes under load exactly when it matters. A fixed 250 ms is optimistic in the tail. Disclosed, not corrected."*

The empirical signed-bias-vs-assumed-latency curve (Amendment 11 → this diagnostic) matches that prediction's shape: bias is small and unstable near the true latency, then grows increasingly negative and saturates as the assumed constant is pushed past it — exactly the pattern a right-tailed, load-dependent latency distribution produces against a fixed point estimate. **Magnitude:** MAE 6.61 ticks at the frozen 250ms primary, vs. 1.0 tolerance (6.6×); best case across any single constant, 3.67 ticks (3.7×). **Direction:** a fixed latency model is systematically too fast in the tail — it assigns queue position and touches the book before real bursty conditions would actually let the order arrive, which is optimistic for the trader exactly when it matters most. **Why it could not be bounded within this seal:** the fix is not a code defect but a modeling choice made and disclosed *before* purchase (§2.1 freezes a single fixed-latency primary by design, for auditability and to avoid a per-event tuning knob); correcting it requires either a conditional/variable latency model or a load-dependent latency distribution, either of which is a new primary model requiring its own pre-registration — not a revision cycle inside this one.

## A12.3 Filed forward, not smuggled in as a save

One reframing idea surfaced in review and is recorded here as a **future, separately pre-registered** direction — explicitly not invoked to rescue cycle 2: split calibration into two market-data-defined regimes (e.g. a pre-declared high-message-rate / high-touch-range flag, computed only from book activity, never from fill error) and report each regime's parity separately, mirroring how degraded trading days are already recorded rather than dropped elsewhere in this project. Any such redefinition must be sealed *before* the next purchase and evaluated on new data — applying it retroactively to the current 36-fill sample would be exactly the restrict-to-favorable-subset pattern this project has already documented and rejected three times.

## A12.4 Disposition

- **R3 (queue-aware offline tick fill simulation for MNQ) is closed.** No tick-resolution strategy study (H1) proceeds on this simulator.
- Spend to date: **$58.83** of the $125 Databento credit (Tranche-1 windows + the A9 test download). No further purchase is authorized under this seal.
- Simulator code (`src/ticksim/`), both gate cycles' raw output, and both diagnostics are retained as the publishable record §4 calls for — this *is* the "real and publishable outcome" the seal anticipated: a documented, mechanism-level finding that retail-accessible offline tick simulation, at a fixed-latency primary model, cannot be made faithful enough to trust for MNQ scalping during exactly the windows scalping edges would need to survive.
- Branch `feat/ticksim-fill-simulator` (and `prereg/ticksim-parity-gate`) stop here; neither merges to `main`.
