# Pre-Registration: GAP-1-C21 — one-shot confirmatory test of GAP-1 on unseen 2021–2024 MNQ

**Drafted:** 2026-09-25. **Status: SEALED 2026-09-25** on Alex's approval ("Approved as drafted"): §4
primary excludes the 29 roll-week setups, V_IN is dropped, and REFUTED leads to a recommendation to
archive. Committed before any replay code was written. No GAP-1 outcome on 2021–2024 had been computed.
**Power gate:** `_bmad-output/diagnostics_gap_fade_2021_2024_power_gate_20260925/` (script
pre-committed in `bb49f32`). Primary **POWERED** at the dev edge size; secondary V_IN **UNDERPOWERED**, so it
is dropped.

## 1. Why

- Live GAP-1 reached its sealed N=30 on 2026-09-21 and scaled to 2ct. That rule outputs SCALE 35.5% of
  the time with **zero** edge (power gate, 2026-09-13), so it is not evidence.
- A real confirmatory mean test needs about 88 trades at the full Gate-0 edge. Live gets there around 2027-03.
- 2021–2024 front-month MNQ holds **278** GAP-1 setups the rules were never run on. That is enough to
  confirm an edge the size of the dev estimate now, instead of in six months.

## 2. Disclosures (read first)

1. **What the author has seen.** Corrected Gate-0 (2025-01 → 2026-06, N=115, PF 1.646), all live GAP-1
   results (N=31), GAP-V's 2023 and 2024 Sep–Nov trades, and the outcome-blind setup list for
   2021–2024 (dates, sides, gap sizes, inside/outside the prior range; **no outcomes**).
2. **Why this test.** It follows the 2026-09-25 recon: published overnight-reversal effects decay, and
   MIM-NB's published counterpart faded after publication. Its purpose is to establish whether GAP-1's edge
   exists outside its own development window.
3. **Data property found by the power gate.** The 2021–24 file rolls at expiry, so roll weeks sit on the
   expiring contract, and front-month bars can't be rebuilt from the data on hand. Those 29 setups are
   excluded from the primary (§4). RTH sessions were verified single-contract.
4. **The engine's clock is fixed, not chosen.** Bars are close-stamped and the sealed engine's RTH is
   stamps 09:30–15:59. "RTH open" is therefore the open of the 09:29–09:30 bar, and "prior close" is the
   15:59 close. That is the sealed and live definition, kept as-is.
5. **Same-bar ambiguity.** If one bar touches both target and stop, the sealed engine takes the target
   first. This is optimistic and kept for fidelity. The count of such bars is reported.

## 3. Engine (frozen — identical to the sealed GAP-1 rules)

- **Sealed parameters** (`preregistration_gap_fade_panic_open.md`, seal 32da5d5; the constants in
  `src/research/gap_fade_live.py`):
  - |gap| / prior close ≥ 0.5%; fade the gap.
  - Enter at the RTH open; target = prior close; stop = entry ± 2.0 × |gap|.
  - Exit at the open of the first bar at or after 13:00 ET; skip Fridays; the prior session needs ≥ 300
    RTH bars.
- **Implementation:** the `replay()` logic of
  `diagnostics_gap_fade_gate0_rescore_20260916/rescore_gate0.py`, used unchanged. The prior close comes
  from the session's **own contract** (`data/mim_x/mnq_1min_by_contract.csv`), and each session's contract
  is identified by matching its bars.
- **Reproduction gate before touching the target:** the same replay must reproduce the corrected Gate-0
  (N=115, PF 1.646) on its dev bars. If it doesn't, stop and report.
- **Cost:** **$5.45 per trade**, which is $1.22 in fees plus the mean measured TS SIM slippage of $4.23
  over 26 fills. P&L is 1ct at $2/pt.

## 4. Window

- **Primary:** 2021-01-04 → 2024-12-31 **minus** 2023 and 2024 Sep–Nov (GAP-V) **minus** every setup
  within 8 calendar days before a quarterly expiry (the file's 16 contract-switch dates, listed in
  `addendum_rollweek.py`). **N = 278**, known before running.
- **Sensitivity (reported, never substituted for the primary):** all 307 unseen setups.

## 5. Test and decision rule

**H1:** mean net $/trade > 0. It is tested with a one-sided one-sample t-test at **α = 0.05** on the 278
primary trades. There is one look, and nothing is re-run after seeing the result.

| Outcome | Condition | Reading | Pre-committed action |
|---|---|---|---|
| **CONFIRMED** | p < 0.05 | GAP-1's rules carried a positive net edge on data they were never fitted to. | Cite as confirmatory *historical* evidence, always with the decay caveat and the per-year table. No automatic live change. It can support a **separate** pre-registration (e.g. ProjectX promotion). |
| **INCONCLUSIVE** | p ≥ 0.05 and mean > 0 | Not confirmed. At 0.75× the dev edge, power is only 0.72. | No edge claim, since ambiguous evidence counts as a FAIL for any claim. Live continues under its sealed rule. This window is spent for GAP-1. |
| **REFUTED** | mean ≤ 0 | Evidence against the edge outside its development window. | Recommend that Alex **archive GAP-1** (it is paper only). The decision is Alex's. The live rule's N=60 look is noted as weak by comparison. |

## 6. Reported alongside (descriptive, never a verdict)

- Gross PF, win rate, net total, and a bootstrap 95% CI of the mean.
- Per year (2021 through 2024) and per side, as a descriptive look at decay. **No decay test:** a decay
  claim would need its own pre-registered contrast.
- The sensitivity on all 307 setups.
- DEFF: the t-test with day-clustered errors by month, reported beside the plain one.
- The count of same-bar target-and-stop ambiguities.
- The share of P&L from the top 3 trades.

## 7. What is not tested

- **V_IN** (inside-prior-range filter): UNDERPOWERED even at its best case (power 0.42), so it is dropped
  and recorded as UNDERPOWERED. It is not to be tested on this window later.
- No other variant, parameter or filter. No live parameter changes. `data/sealed_holdout/` is untouched.

## 8. Sealing

Approved by Alex on 2026-09-25 and committed unchanged apart from this status text. The replay script
is written after this commit and run once.
