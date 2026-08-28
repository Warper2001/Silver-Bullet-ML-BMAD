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
