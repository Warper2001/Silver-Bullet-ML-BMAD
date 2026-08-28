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
