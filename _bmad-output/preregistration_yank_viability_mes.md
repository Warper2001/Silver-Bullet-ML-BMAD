# Pre-Registration: YANK-VIA — Is YANK's Sealed Configuration Viable At All?

**Registered:** 2026-08-28
**Status:** SEALED at commit time. Append-only amendments.
**Arises from:** `preregistration_yank_gap_floor.md` §A1.4, where YANK's live configuration —
run as the control arm on four unseen years — came back at t = 0.514 and below 1× friction.

---

## 1. The question, and why it cannot be asked of existing data

§A1.4 was a by-product, not a test: the live floor was the *control* in a floor-comparison
experiment. Its result is already known and recorded. **A hypothesis test cannot be
pre-registered against data whose answer has been read**, so this seal does not re-run
MNQ 2021–2024, 2025 (derivation), or the 2026 holdout (33 logged accesses).

### 1.1 What is already known — accounting, not a new claim

YANK's live configuration (`min_gap_atr_ratio` 0.25), normalised per contract to the
notional **of each era**:

| source | N | bps/trade | 95% CI | t |
|---|---|---|---|---|
| 2021–2024 backtest | 500 | +0.633 | wide, spans 0 | 0.514 |
| 2025 backtest | 80 | −0.446 | wide, spans 0 | −0.231 |
| **2026 LIVE (Jun–Aug)** | **11** | **+14.20** | **[−3.73, +32.14]** | 1.552 |

**The "27× friction headroom" figure this project has quoted all week — including in
`friction_rescreen_20260827.md` — rests on N = 11, and its own 95% confidence interval does
not exclude zero.** Two backtests totalling N = 580 of the identical configuration put it
at +0.63 and −0.45 bps. The live figure is roughly an order of magnitude above both.

That is not a finding about YANK; it is a finding about **11 observations**. It is recorded
here so the 27× number is never again used as a premise.

## 2. The test — the only unspent data that exists

If YANK's edge is **structural** (H1 liquidity sweep → M15 CHoCH → M1 FVG, bearish-only),
it should appear in a sibling equity-index future sharing the same session and
microstructure. If it is an artifact of MNQ over one era, it should not.

> **Run the identical, unmodified YANK configuration on MES (Micro E-mini S&P 500),
> 2021-01-03 → 2024-12-19.**

**YANK has never touched MES.** No seal, sweep, backtest or holdout in this project has used
it. It is genuinely virgin data for this strategy.

**Data:** `data/mim_x/mes_1min_2021_2024_frontmonth.csv` — 1,404,796 bars, 16 quarterly
contracts, fetched 2026-08-28. Each contract contributes only its active window
`[3rd Friday −3 months, 3rd Friday]`; no bar crosses a roll.

## 3. Configuration — identical, one instrument change

Every parameter is the live sealed YANK config, unchanged: `min_gap_atr_ratio` 0.25,
`max_gap_atr_ratio` 0.426, `atr_threshold` 0.5, sl/tp 2.0/8.0, H1 sweep lookback 6,
M15 CHoCH on, vol regime 0.75/120, Tuesday exclusion on, bearish-only.

**Point value does not need changing and must not be.** Basis points are point-value
invariant: `bps = 10000 × points / index_level`, since P&L and notional both scale with it.
The daily-loss breaker therefore binds at the same *point* threshold in both instruments,
which is the correct equivalence. Dollar columns from the engine are meaningless for MES and
are not reported.

## 4. MES economics — different from MNQ, and worse

| | MNQ (2021–24) | **MES (2021–24)** |
|---|---|---|
| mean index level | 15,186 | **4,522** |
| point value | $2 | **$5** |
| notional/contract | $30,371 | **$22,610** |
| friction/RT | $2.50 | **$3.25** ($2 comm + 1 tick $1.25) |
| **friction in bps** | 0.823 | **1.437** |
| **3× economic bar** | 2.469 | **4.312** |

MES friction is **1.75× MNQ's in bps terms**. This independently reproduces this project's
own 2026-06-15 edge-headroom screen (MNQ 10.8× vs MES 3.2× at 5-min), and it means MES is a
**harder** venue. A pass here is strong evidence; a failure is partly attributable to venue
geometry and must be reported as such.

## 5. POWER STATEMENT — fixed before the run

Per `preregistration_intraday_momentum_mnq.md` §A1.5. Per-trade sd is proxied from the MNQ
2021–2024 run of the same configuration (recovered from its reported mean and t):
**≈ 27.6 bps per trade**. Expected N ≈ 500 (MNQ's rate over the same span).

| | value |
|---|---|
| Smallest effect detectable at t = 2.0, N=500 | **2.47 bps** |
| MES 3× economic bar | **4.31 bps** |

**Detectable (2.47) is below the bar (4.31)**, so the test can see an effect large enough to
matter. Declared in advance: if realised N or sd differ materially from these, the achieved
power is recomputed and reported **before** the verdict is read.

## 6. Decision rule — economic and statistical

| Condition | Verdict | Pre-committed action |
|---|---|---|
| mean > 0, t ≥ 2.0, **and ≥ 4.312 bps** | **STRUCTURAL** | The edge generalises to a sibling index future. Record. Triggers §8. |
| mean > 0, t ≥ 2.0, but < 4.312 bps | **REAL BUT SUB-SCALE** | Record. Present but cannot pay MES friction. No deployment. |
| mean > 0, t < 2.0 | **UNPROVEN** | Record. No claim either way. |
| mean ≤ 0 | **DOES NOT GENERALISE** | The edge is not present in MES. Record; close. |

**Reported, never decision-bearing:** per-year breakdown; trade count and frequency; exit-reason
mix; win rate; and the same figures at the MNQ friction benchmark for comparability.

**This test cannot prove YANK works.** A STRUCTURAL verdict says the pattern generalises
across instruments, which is evidence *for* the mechanism being real. It does not establish
that MNQ-specific live trading is profitable, and §7.4 forbids reading it that way.

## 7. What we will NOT do

1. No parameter is tuned for MES. Identical config or the test is void.
2. No second instrument if MES fails. One test, one instrument, one verdict.
3. No sweep of the floor, ceiling or any gate on MES data.
4. **No reading of a PASS as validating YANK on MNQ**, or of a FAIL as refuting it — MES is
   a different instrument with 1.75× the friction.
5. No dropping of years or contracts.
6. **No change to the live bot under any outcome.**

## 8. Successor trigger

**STRUCTURAL** authorises drafting a pre-registration for a properly-powered MNQ viability
test on genuinely prospective data — the only clean question left, given every MNQ era is
now spent. Nothing else.

## 9. Disclosed limitations

- Single sibling instrument. MYM and M2K were available and deliberately **not** run, to
  avoid a three-instrument multiple-comparison problem (§7.2).
- 2021–2024 excludes the 2025–2026 regime YANK was derived in and currently trades.
- Per-trade sd is proxied from MNQ, not measured on MES (§5).
- The engine, and any bug in it, is shared with every prior test in this program.

## 10. Values fixed at seal time

| Item | Value |
|---|---|
| git HEAD at seal | `a9e4595` |
| MES bars | `data/mim_x/mes_1min_2021_2024_frontmonth.csv`, 1,404,796 rows |
| Config | live YANK, unchanged, `min_gap_atr_ratio=0.25` |
| MES friction / 3× bar | 1.437 bps / **4.312 bps** |
| Expected N / detectable @ t=2 | ~500 / **2.47 bps** |

---

# Amendment 1 — Result (2026-08-28)

Append-only. Original sealed text unedited.
Script: `study_yank_viability_mes.py` | Report: `data/reports/yank_viability_mes_20260828.txt`
Run under seal `3f9167e`.

## A1.1 Verdict — DOES NOT GENERALISE

1,404,796 MES bars, 2021-01-03 → 2024-12-19, YANK's configuration unmodified.

| | value |
|---|---|
| N | **412** (0.285/day) |
| mean | **−1.827 bps/trade** |
| sd | 18.53 bps |
| **t** | **−2.002** |
| PF | **0.777** |
| win rate | 30.8% |
| net | −752.8 bps (−340 index points/contract) |

Mean ≤ 0 → **DOES NOT GENERALISE** per §6. Recorded; closed. Per §7.2 **no second
instrument is run** — MYM and M2K remain deliberately untouched.

## A1.2 This is the program's first SIGNIFICANT negative, not another null

Every prior test this week returned a null — MIM-X2 t = −1.921, OFI-1 economically
irrelevant, YANK-FLOOR t = −1.149, S26-EXIT p = 0.94. Each said "no effect found".

**t = −2.002 is different.** It is significantly negative at α = 0.05 two-sided. On MES,
YANK's structure does not merely fail to make money — it loses, with statistical support.

The exit mix shows why: **SL 247 / TIME_STOP 134 / TP 31.** Only **7.5%** of trades reach
target, against a 30.8% win rate. The 2.0/8.0 SL/TP geometry, calibrated on MNQ, is being
stopped out before the 8× target is reachable on a lower-volatility index.

## A1.3 Power was adequate, and better than sealed

§5 required recomputing achieved power before reading the verdict:

| | sealed estimate | realised |
|---|---|---|
| N | ~500 | **412** |
| sd | ~27.6 bps | **18.53 bps** |
| detectable @ t=2.0 | 2.47 bps | **1.825 bps** |

Realised sd came in well below the MNQ proxy, so despite the smaller N the test is **more**
sensitive than sealed: detectable 1.825 bps against a 4.312 bps bar. **ADEQUATE.** An effect
large enough to matter would have been seen.

## A1.4 Per-year (NOT decision-bearing, §6)

| year | N | mean | net |
|---|---|---|---|
| 2021 | 89 | +0.122 bps | +10.8 |
| 2022 | 98 | −2.867 bps | −281.0 |
| 2023 | 122 | −2.672 bps | −326.0 |
| 2024 | 103 | −1.521 bps | −156.6 |

Three of four years clearly negative; 2021 is flat. Not carried by one year.

## A1.5 A caveat that makes the result *worse*, not better

The engine deducts `commission_per_roundtrip = 4.0` — **an MNQ-calibrated figure**. At 5
contracts and the engine's internal $2/pt that is ≈0.885 bps at the MES index level, while
MES's actual round-trip friction is **1.437 bps** (§4). The reported −1.827 bps is therefore
**optimistic by roughly 0.55 bps**; the true figure is nearer −2.4 bps.

Disclosed rather than corrected, because correcting it would move a number after seeing the
verdict. It does not change the direction or the conclusion.

## A1.6 What this does and does not establish

**Establishes:** YANK's sealed structure (H1 sweep → M15 CHoCH → M1 FVG, bearish-only,
2.0/8.0) does not transfer to the nearest sibling equity-index future. On MES it is
significantly loss-making across four years at adequate power.

**Does NOT establish** — and §7.4 binds here: anything about YANK on MNQ, in either
direction. MES carries **1.75× the friction in bps**, a materially lower index volatility,
and the SL/TP geometry was never calibrated for it. A strategy can be instrument-specific
and still real.

**Does not trigger §8.** No successor is authorised.

## A1.7 The accumulated picture, stated without overclaiming

Placing this beside §1.1 rather than in place of it:

| evidence | N | result |
|---|---|---|
| MNQ 2021–2024 (unseen at the time) | 500 | +0.633 bps, **t = 0.514** |
| MNQ 2025 (derivation era) | 80 | −0.446 bps, t = −0.231 |
| **MES 2021–2024 (unseen)** | **412** | **−1.827 bps, t = −2.002** |
| MNQ 2026 LIVE | **11** | +14.20 bps, CI **[−3.73, +32.14]** |

**The only sample in which YANK's edge is large is the smallest one**, and its confidence
interval contains zero. Every larger sample — 992 backtested trades across two instruments
and four years — is at or below zero.

That is an observation about the evidence, not a verdict on the strategy. No test here was
designed to rule on YANK's MNQ viability, every MNQ era is spent, and §8 of this seal is not
triggered. But the "27× headroom" premise that motivated this entire line of work does not
survive contact with any of it.

**Live bot unmodified throughout.** Nothing deployed, no parameter changed, no service
restarted.
