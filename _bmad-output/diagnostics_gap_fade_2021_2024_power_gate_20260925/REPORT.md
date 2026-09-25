# Power gate — one-shot test of the sealed GAP-1 rules on MNQ front-month 2021–2024 (2026-09-25)

**Verdict under the pre-committed rule:**

| Test | Verdict | Power at the dev edge size |
|---|---|---|
| **Primary** (sealed rules, mean net > 0) | **POWERED** | 0.93 on all 307 setups; **0.90 on the 278 outside roll weeks** |
| **Secondary** (V_IN: gap opens inside the prior range) | **UNDERPOWERED** | 0.46 / 0.42 at the *best-case* contrast |

**Method:**
- The script (`power_gate.py`) was committed in `bb49f32` **before it ran**, with the rule in its docstring.
- It is outcome-blind: no trade was simulated on 2021–2024, and only setup-defining prices were read.
- `data/sealed_holdout/` was not touched.

## Unseen setups (outcome-blind count)

| Stage | Count |
|---|---|
| RTH sessions in the file | 1,032 |
| Qualifying GAP-1 setups (sealed filters, same-contract prior close) | 344 |
| Minus the 2023 and 2024 Sep–Nov windows GAP-V already traded | −37 → **307** |
| Minus the 29 roll-week setups (see the data finding below) | → **278** |

- **By year and side (307):**

  | Year | Long | Short |
  |---|---|---|
  | 2021 | 31 | 42 |
  | 2022 | 65 | 49 |
  | 2023 | 27 | 34 |
  | 2024 | 21 | 38 |

  2022 alone supplies 37%, so this is regime-concentrated; see DEFF.
- **Inside vs outside the prior range:** 130 inside / 177 outside (307); 117 / 161 (278).
- **Contract assignment:** every one of the 1,415,732 bars matches exactly one contract, and no RTH session mixes contracts.

## Data finding: the file rolls at expiry, not at the volume crossover

- **Where it switches:** `mnq_1min_2021_2024_frontmonth.csv` switches contract on each quarterly **expiry Friday**, 16 of 16 times, and the switch happens outside RTH.
- **What that means in roll week:** the file sits on the **expiring** contract for about 5 sessions after volume has migrated. In March 2021, H21 volume fell from 1.1–1.7M a day to 0.2–0.3M.
- **No rebuild is possible:** `mnq_1min_by_contract.csv` has no next-contract bars before the switch.
- **Why this is low-risk but excluded anyway:** the prices stay liquid (about 200K contracts a day), and each gap is same-contract. But AGENTS.md says not to trust roll windows without rebuilt front-month bars, and this data can't be rebuilt. So the **proposed primary excludes the 29 setups within 8 calendar days before an expiry**; all 307 is a sensitivity. The choice was made before any outcome was seen (`addendum_rollweek.py`).
- **Worth noting:** the excluded setups include large gaps (3.3% on 2022-06-13, 3.9% on 2022-12-13).
- **Scope:** MIM-NB's 2021–2024 gate uses this same file, so its roll-week sessions carry the same property.

## Power (net of $5.45/trade: $1.22 fees plus $4.23 mean TS SIM slippage over 26 fills)

Anchors are in d = mean/SD per trade. The dev figure is the corrected Gate-0: N=115, net $66.56 ± $376.93.

| Anchor | d | N=278, DEFF 1 | N=278, DEFF 1.5 | N=307, DEFF 1 |
|---|---|---|---|---|
| **Ceiling: dev, in-sample** | 0.177 | **0.90** | 0.78 | 0.93 |
| 0.75 × ceiling | 0.132 | 0.72 | 0.56 | 0.75 |
| Live ledger (N=31, +$48/trade gross) | 0.113 | 0.59 | 0.46 | 0.64 |
| 0.5 × ceiling | 0.088 | 0.44 | 0.33 | 0.47 |
| 0.25 × ceiling | 0.044 | 0.18 | 0.15 | 0.19 |

At a $10 cost, the ceiling power for N=307 is 0.89 (see `results.json`).

**Reading:**
- The test can **confirm** an edge about as large as the in-sample dev estimate. At three-quarters of that size it's closer to a coin flip.
- The dev estimate is upward-biased: it is the window the spec was gated on, and the corrected figure is already 16% below the sealed one.
- So **a non-significant result does not show the edge is absent**, unless the point estimate itself is ≤ 0.
- **A pass shows an edge existed in 2021–24, not that it persists.** Published overnight-reversal evidence shows decay (Della Corte et al.; see the 2026-09-25 recon).

## Secondary: V_IN is UNDERPOWERED even at its best case

| Contrast (inside − outside) | Power, N=278 (117/161) | Power, N=307 (130/177) |
|---|---|---|
| 1.0 × ceiling (outside gaps have zero edge) | **0.42** | 0.46 |
| 1.5 × ceiling | 0.70 | 0.74 |
| 2.0 × ceiling (outside gaps lose as much as inside earn) | 0.90 | 0.92 |

Power reaches 0.80 only if outside-range gaps *lose* about as much as inside-range gaps *earn*. That is an extreme premise. **Recommendation: do not spend the unseen window on V_IN.** Record it as UNDERPOWERED, a valid verdict, and test only the baseline.

## Files

| File | Contents |
|---|---|
| `power_gate.py` | The gate, pre-committed in `bb49f32` |
| `results.json` | All numbers, with the sha256 of the target, the by-contract file, the dev file and the script |
| `unseen_setups_outcome_blind.csv` | 307 setups: date, side, gap %, inside/outside, contract — no outcomes |
| `addendum_rollweek.py` / `.json` | The roll-week exclusion and power at N=278 |
