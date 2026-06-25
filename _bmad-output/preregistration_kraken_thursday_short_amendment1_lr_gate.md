# Pre-Registration Amendment 1: LR-Regime Gate for Kraken Thursday Short

**Status:** PROSPECTIVE — SEALED 2026-06-25
**Amends:** `preregistration_kraken_thursday_short.md` (parent, sealed 2026-06-21)
**Strategy:** DOW-THU (Day-of-Week Thursday Short), PF_XBTUSD + PF_ETHUSD
**Date drafted:** 2026-06-25

---

## 0. Nature of this amendment (read first)

This amendment **does NOT change live trading.** The parent strategy (unconditional
Thursday short, every Thursday) keeps running exactly as sealed. This amendment
registers a **second, nested hypothesis** — that a linear-regression-channel trend
regime filter improves the edge — to be evaluated as a **pre-specified subset** of
the same prospective fills.

Because the live trader now logs the regime variable on every fill (commit
`fd6bd9c`, columns `lr_slope20_bpd` / `lr_slope40_bpd`), and because the gate rule
below is fully specified **before any prospective fill exists**, partitioning the
prospective Thursdays by it later is a *pre-registered conditional analysis*, not
post-hoc selection. No live trade is skipped; nothing about the parent test is
contaminated.

---

## 1. Motivation

The parent prereg flagged that the Thursday edge is ~7× stronger OOS (Sharpe 3.88)
than IS (Sharpe 0.56) — backwards from normal degradation, a regime-sensitivity red
flag. A regime-decomposition battery (2026-06-25, `tmp/regime_battery.py`) tested
whether the edge is structural or regime-conditional:

- **Day-of-week placebo:** Thursday is genuinely the best short-day in all segments,
  and "short every day" earns only Sharpe +0.34 in OOS — so it is *not* merely
  "short a down tape." The day-specific component survives.
- **Day-of-week dummy regression** with trend-slope + realized-vol controls: the
  Thursday coefficient is unchanged (t=+2.88 → +2.88), i.e. orthogonal to linear
  regime main-effects.
- **BUT the magnitude is heavily regime-amplified:** conditioning each Thursday on the
  BTC LR-channel slope at entry shows the edge lives in **down-trend** regimes.

This amendment moves that regime-interaction finding from in-sample observation to a
locked prospective test.

---

## 2. Discovery disclosure (transparent — this gate was found in-sample)

The LR-regime conditioning was **discovered post-hoc** on the Dec-2024–May-2026
sample (not pre-registered ab initio). In-sample, weekly-annualized (√52),
combined BTC+ETH Thursday short, net of 10 bps/asset:

| Bucket (regime at entry) | N | mean bps | wk-Sharpe | WR | worst Thu |
|---|---|---|---|---|---|
| Unconditional (all Thu) | 75 | +85 | +1.83 | 63% | **−14.27%** |
| **L=20 DOWN** (slope ≤ 0) | 38 | **+152** | **+3.34** | 68% | −4.27% |
| L=20 UP (slope > 0) | 37 | +16 | +0.34 | 57% | **−14.27%** |
| L=40 DOWN (slope ≤ 0) | 38 | +115 | +2.78 | 61% | −2.88% |
| L=40 UP (slope > 0) | 35 | +53 | +1.01 | 66% | **−14.27%** |

**Key in-sample observations (to be confirmed or refuted prospectively):**
1. The edge concentrates in DOWN regimes; in UP regimes it is near-flat (L=20: +16 bps).
2. The catastrophic −14.27% Thursday lands in the **UP** regime under *both* lengths —
   the gate removes exactly the tail that threatens a drawdown-limited funded eval.
3. ~51% of Thursdays are DOWN-regime, so a gated rule trades ~half as often.
4. **L=20 separates more cleanly than L=40** (UP is nearly dead at L=20; still +53 bps
   at L=40) → L=20 is pre-specified PRIMARY, L=40 SECONDARY robustness only.

**Multiple-comparisons note:** exactly two channel lengths (20, 40) were examined.
No other lengths, polarity definitions, or thresholds may be substituted without a
new amendment. The primary length is locked to 20 below.

---

## 3. Gate specification (SEALED)

**Regime variable.** BTC LR-channel slope, computed from **completed daily UTC BTC
closes** (Kraken `XBTUSD` OHLC, `interval=1440`), through the day **prior** to the
Thursday entry (no lookahead — the value is fully known at 00:00 UTC Thursday).
Normalized to bps/day: `slope = compute_lr_channel(closes, L).slope[-1] / last_close × 1e4`
(`src/research/lr_channel.py`). Logged live as `lr_slope20_bpd`, `lr_slope40_bpd`.

- **PRIMARY length L = 20 days.** SECONDARY (robustness) L = 40 days.
- **Regime definition:** DOWN ⇔ slope ≤ 0; UP ⇔ slope > 0.
- **Gated rule:** take the Thursday short **only in DOWN regime**; remain flat in UP.

The gated trade set is a strict subset of the unconditional trade set; both are
recovered from the same logged fills.

---

## 4. Prospective hypotheses

- **H_gate (primary, regime interaction):** prospective DOWN-regime Thursdays have a
  **higher mean net return than UP-regime Thursdays** (the in-sample interaction
  persists out-of-time). One-sided.
  - **H₀:** mean(DOWN) − mean(UP) ≤ 0.
- **H_improve (secondary):** the DOWN-only (gated) rule delivers **better
  risk-adjusted return than the unconditional rule** on the same prospective window
  (gated Sharpe > unconditional Sharpe) **and** does not worsen the tail.

---

## 5. Decision rule (SEALED)

**Evaluation trigger:** when **N_DOWN ≥ 20** prospective DOWN-regime Thursdays have
completed (primary L=20). At ~51% DOWN frequency this is ~40 calendar Thursdays
(~April–May 2027). An interim read is permitted at the parent's N≥30 unconditional
gate (~Jan 2027) but is **not** decision-bearing for the gate.

**GATE CONFIRMED — adds value (ALL must hold):**
1. N_DOWN ≥ 20.
2. mean(DOWN) − mean(UP) > 0, one-sided t-test **p < 0.10** (H₀ rejected).
3. Gated (DOWN-only) weekly Sharpe **> 1.0** AND gated Sharpe **>** unconditional Sharpe
   over the same prospective window.
4. Gated worst single Thursday is **not materially worse** than the unconditional
   worst (the gate must not enlarge the left tail).

**GATE REJECTED — keep unconditional (ANY triggers):**
- After N_DOWN ≥ 20: mean(DOWN) ≤ mean(UP) (no interaction — the IS finding was a
  regime artifact of the discovery window), **OR** gated Sharpe ≤ unconditional Sharpe.
- In that case the parent unconditional rule stands and any funded-eval uses it.

**On confirmation.** If the gate is CONFIRMED, a funded-account / scaled deployment
may trade the **LR-DOWN-gated** rule at higher size (the gate cut the worst Thursday
−14.3% → −4.3%, which an MC showed turns the Kraken Prop $100K **Advanced** eval from
~76% to ~100% pass) — pending its own separate deployment pre-registration. This
amendment authorizes **measurement only**, not any sizing or venue change.

---

## 6. What would falsify the gate

- UP-regime Thursdays prospectively perform **as well as** DOWN-regime Thursdays →
  the regime interaction was an artifact of the 2024–26 sample → reject, keep
  unconditional.
- The unconditional edge itself fails the parent decision rule (Sharpe ≤ 0.50 at
  N≥30) → the gate question is moot; the strategy is retired per the parent.

---

## 7. Implementation reference

- **Instrumentation:** commit `fd6bd9c` on `feat/yank-ml-canary` — behavior-preserving
  LR-slope logging in `kraken_thursday_short.py` (`fetch_btc_lr_slopes()`, firewalled;
  `_log_trade` writes `lr_slope20_bpd` / `lr_slope40_bpd` once per Thursday).
- **Regime math:** `src/research/lr_channel.py::compute_lr_channel`.
- **Discovery battery:** `tmp/regime_battery.py` (day-of-week placebo, rolling Sharpe,
  LR-regime conditioning, DoW dummy regression, cross-asset corr).
- **Funded-eval MC:** `tmp/kraken_prop_mc_gated.py` (gated vs unconditional pass rates).
- **Live logging active from:** Thursday **2026-07-02** (first Thursday after the
  fd6bd9c restart). The 2026-06-25 fills predate the restart and have blank slope.

---

## 8. Seal information

**Seal purpose:** Pre-register the LR-regime gate as a prospective conditional analysis
on the parent Thursday-short prospective sample, **before** the gated data accumulates.
**Sealed:** 2026-06-25 — before the 2026-07-02 first logged Thursday, preserving
prospective validity (no gated fills existed at seal time).
**Branch:** feat/yank-ml-canary (implementation); parent sealed on
worktree-kraken-strat-research.
**Cross-asset caveat (inherited):** corr(BTC, ETH) ≈ 0.85 — the two legs are ≈ one
bet; "both confirm" is ~1 independent witness, size accordingly.
