# Pre-Registration: MIM-NB Honest Expectations & Campaign Sizing (Correction)

**Generated:** 2026-06-14
**Experiment ID:** mim-nb-honest-expectations
**Base commit:** e0939d9 (worktree branch `worktree-mim-nb-honest-expectations`)
**Supersedes:** §5 "Honest Expectations" of `preregistration_mim_nb_live_deployment.md`
(experiment `mim-nb-live-combine`, base 1a67de3). That section quoted the **pooled**
result; this document replaces it with the **out-of-sample** result and adds the
size/campaign analysis.
**Status:** SEALED — no strategy or config change. The live bot (`trader-mim-nb.service`,
acct 23884932, 1× MNQ) keeps running **unchanged**. This document corrects the numbers we
quote when reasoning about MIM-NB; it changes no code, YAML, or position size.

---

## 0. Why this correction exists

While probing whether MIM-NB could be optimized, we found the deployment-authorizing
expectation was **inflated** in three ways. The strategy is **not** being abandoned — it is
the only setup in this project that survived OOS net of cost — but the expectation was
oversold. All numbers below are reproduced from the sealed engine; scripts are archived in
the job tmp (`mim_exit_sweep.py`, `mim_pyramid.py`, `mim_size_sweep.py`).

---

## 1. Frozen strategy (UNCHANGED — restated for the record)

Identical to sealed `mim-nb-v2-catstop` S-B (deployment prereg, base 1a67de3):

- **Instrument:** MNQ front month, **1 contract**, long & short.
- Noise bands per RTH minute label t: σ(t) = mean over prior 14 complete RTH days of
  |close(d,t)/open_d − 1|; UB = O·(1+σ)+max(C_prev−O,0); LB = O·(1−σ)−max(O−C_prev,0).
- Entries/reversals at HH:00/HH:30, 10:00–15:30 ET (close > UB → long, close < LB → short),
  filled next bar open.
- **Catastrophe stop** at entry ∓ 500 pts (intrabar). EOD flatten at 16:00 ET.
- DLL guard: once realized day P&L ≤ −$1,000, no new entries that session.

**Empirical note (dev-2025):** the "wide band-stop" exit specified in the original prereg
**never fires** — every exit is EOD or catastrophe-stop. The effective strategy is
*"enter on the half-hour breakout, hold to the close, catastrophe-stop at 500."* This is a
description, not a change.

---

## 2. CORRECTED honest expectations (replaces deployment §5)

### 2a. Forward expectancy — quote OOS, not pooled

| Window | role | net PF | net expectancy / contract / trade |
|---|---|---|---|
| Dev 2025 | selection (in-sample) | 1.484 | +$54.40 |
| **OOS 2026 (Jan–May)** | **honest forward** | **1.299** | **+$31.99** |
| Pooled (the figure cited at deployment) | — | 1.514 | +$53.74 |

**The forward expectation is ≈ $32/ct/trade, not $54.** The pooled figure is ~70% higher
because it blends 2025's fat tails into the forward estimate. Going forward we quote the
**OOS** number.

### 2b. The edge is a small number of fat-tail trend days

- Best single day = **+$3,647 = 41% of all pooled profit**; **top-3 days = 68%**
  (dev-2025 alone: top-3 = **97%**).
- ~160 of 163 days hover near break-even net of cost; rare big trend days held to EOD pay
  for everything. Only 2 catastrophe-stop hits in 163 days — the catstop is a backstop, not
  the driver.
- **Reassuring:** OOS-2026 (50 fresh days, never used in selection) shows the *same*
  fat-tail structure (its own top-3 = 114% of OOS profit) and still passes the MC. The edge
  is real and repeated, **not** overfit-and-dies. It is a genuine **positive-EV,
  high-variance lottery** on trend days, not a steady grind.

### 2c. Combine Monte Carlo — honest band, not a single number

Block-bootstrap of ET days (sampled with replacement, intraday order preserved), corrected
combine rules, 5,000 sims, 90-day cap, **1 contract**:

| data set | pass% | blow% | stall% | median days-to-pass |
|---|---|---|---|---|
| Pooled (sealed, authorizing) | 54.0% | 33.3% | 12.7% | 41 |
| **OOS-2026 only** | **54.5%** | **37.4%** | 8.2% | 42 |
| Pooled minus best day | 52.4% | 40.6% | — | 41 |
| **Pooled minus top-3 days** | **38.9%** | **49.4%** | — | 46 |

**Honest pass band ≈ 39–55% / blow ≈ 33–49%**, depending on regime. The headline "54%" is
not wrong, but it leans on the bootstrap re-sampling 1–3 fat-tail days; if 2026-H2 is choppy
and hands us few trend days, realized outcomes sit in the left tail. **Treat ~50–55% pass /
~35–40% blow as the central case and ~39% pass / ~49% blow as the bear case.**

### 2d. Methodological caveats (so we don't re-inflate later)

1. The day-block bootstrap samples days **IID with replacement** → it over-samples the rare
   fat-tail days and **understates** real-world regime clustering. Pass% is therefore an
   upper-ish estimate.
2. The MC checks trailing-DD/DLL on **realized exits only** — open-position MtM drawdown is
   not modeled. This **understates blow%** (matters most for any multi-unit sizing).
3. Live `data/mim_nb/trades.csv` logs **gross** P&L (no cost deducted); real balance runs
   ~$2.24/ct below logged. Trivial at 1ct, but don't read live P&L as net.

---

## 3. Sizing — 1 MNQ is the optimum; bigger is strictly worse

Same MC, OOS-2026 data, swept by size (1 NQ mini ≡ 10 MNQ in P&L):

| size | pass% | blow% | median days | note |
|---|---|---|---|---|
| **1 MNQ** | **54.5%** | **37.4%** | 42 | optimum |
| 2 MNQ | 47.9% | 52.1% | 14 | |
| 3 MNQ | 43.9% | 56.1% | 9 | |
| 5 MNQ | 35.0% | 65.0% | 6 | |
| **1 NQ (=10 MNQ)** | **31.5%** | **68.5%** | 6 | worst tested |

**Risk-of-ruin against a fixed $2,000 trailing drawdown makes the smallest size optimal.**
A 500-pt catastrophe stop costs $1,000/ct; at 5 MNQ that one stop is $5,000 (2.5× the DD),
at 1 NQ it is $10,000 (5× the DD). The worst OOS day at 1 NQ is −$6,627 — that single day
blows the account 3× over. Larger size only buys *speed if you happen to win* (median 6 days
vs 42) while making ruin the base case. **Decision: stay at 1 MNQ. No upsizing.**

### 3a. Pyramiding is also rejected (same reason)

Adding units on same-direction breakouts raises gross edge (net PF 1.61→1.87, mean day
$64→$197) but **lowers** combine pass (57%→49%) and **raises** blow (31%→45%) — the
drawdown-gated combine punishes the added variance, and the MC even flatters it (caveat §2d.2).
The drawdown wall, not the signal, is binding.

---

## 4. The combine is a campaign, not a single hero run

Combines are resettable. With a ~50–55% central per-attempt pass rate at 1 MNQ, cumulative
P(≥1 pass) over N independent attempts = 1 − (1 − p)^N:

| attempts | p = 0.55 (central) | p = 0.39 (bear) |
|---|---|---|
| 1 | 55% | 39% |
| 2 | 80% | 63% |
| 3 | **91%** | 77% |
| 4 | 96% | 86% |

**Plan:** run the combine at 1 MNQ, accept ~50–55% per attempt, and budget for ~2–3 resets
to push cumulative odds past 90%. Combine cost: $49/mo + $149 activation per attempt — small
relative to a funded 50K. A blown attempt at ~35–40% is a **priced, expected** outcome; it
does not invalidate the strategy unless a halt trigger (deployment prereg §4) fires.

---

## 5. What would change this verdict (pre-registered, so we can't move the goalposts)

- **Higher per-attempt odds** can only come from a **second, uncorrelated edge** (different
  instrument or session so its trend days do not coincide with MIM-NB's), run as an
  *additional* sleeve — never from a better exit/size on this strategy (exit and sizing are
  both exhausted: §1 note, §3, §3a).
- If live MIM-NB reaches **N ≥ 20–30 completed trades** with net PF tracking the OOS 1.30,
  the central-case expectation upgrades from hypothesis to evidence. If net PF < 0.70 over 30
  trades, the deployment halt trigger fires and this whole expectation is void.

---

## 6. Live status at sealing

Acct 23884932, 1× MNQ, `trader-mim-nb.service`, since 2026-06-11.
Trades to date (2): +$740.50 (6/11, EOD) and +$47.00 (6/12, EOD); the +$740 day being ~94%
of total live profit is the fat-tail signature of §2b, **expected, not anomalous**.
No config change. Live bot continues unchanged.
