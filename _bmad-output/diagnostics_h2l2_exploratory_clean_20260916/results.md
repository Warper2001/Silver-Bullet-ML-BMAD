# H2/L2 exploratory variants, re-run on corrected bars (2026-09-16)

The H2/L2 gate's exploratory variants originally ran on the contaminated CSVs, and the C1 correction explicitly left them out (`correction_plan.md` section 4). This re-runs them on the **C1b front-month rebuild**, the same bars the corrected gate used: 270 sessions, 20,675 bars, 2025-01-02 → 2026-02-27.

- **Still exploratory.** Not pre-registered, and it changes no verdict. All three constructs remain UNDERPOWERED.
- **Only the bars differ.** Constructs, filters, geometry, costs, effect sizes, placebo and power formula are the committed gates'.
- **Guard:** variant A reproduces the corrected gate's N = 47 exactly. The script exits if it does not.
- **Firewall:** unchanged. Dispersion comes only from placebo pairings shifted 5 or more sessions, no statistic of price after a real event, and nothing under `data/sealed_holdout/`.

## MNQ variants: contaminated → corrected

| Variant | N | Power at central 0.10R | Years for 80% at central | Power at optimistic 0.20R | Years at optimistic |
|---|---|---|---|---|---|
| A: plan primary (prox 1.00 pt) | 56 → **47** | 14.3% → **7.4%** | 21.7 → **173** | 47.5% → **22.1%** | 2.9 → **8.6** |
| B: proximity scaled (4.00 pt) | 61 → **51** | 13.9% → **7.3%** | 23.1 → **178** | 47.3% → **23.0%** | 2.9 → **8.1** |
| C: no proximity filter | 136 → **118** | 12.7% → **7.5%** | 28.6 → **152** | 56.4% → **35.9%** | 2.2 → **4.0** |
| D: raw H2/L2, no filters | 1,019 → **935** | 18.3% → **11.9%** | 13.3 → **30.9** | 99.6% → **98.1%** | 0.4 → **0.5** |

**The original reading holds, and hardens.**
- **No relaxation rescues the central case.** The floor across variants was 13.3 years; it is now **30.9**.
- **The pessimistic 0.05R is now cost-bound in every MNQ variant**, where before only the two ES rows were.
- **The one powered cell is still D at the optimistic edge.** That is not Wade's construct, and picking it now would be selection after seeing the gate.

**Why the numbers moved:** fake roll-week bars inflated risk, and the modelled edge is a fraction of risk, so it shrank against the fixed $5.80 cost. Median R falls from $76.50 to $69.00 in variant A, and σ from $102.9 to $82.1.

## ES rows: approximate, and unchanged in conclusion

**There is no raw ES source in this repo**, so a front-month rebuild is impossible. The ES rows are only approximately cleaned: 2026 is dropped entirely (the back-month defect cannot be checked without raw data), and the same 27 roll-week sessions are dropped. That leaves 149 sessions, 2025-05-01 → 2025-12-31.

| Variant | N | Power at central | Power at optimistic | Years at optimistic |
|---|---|---|---|---|
| A: literal 4 ticks, sized as MES | 34 | 2.1% | 9.0% | 40.2 |
| D: raw H2/L2, sized as MES | 540 | 0.0% | 8.9% | 40.7 |

**Both stay cost-bound at the central edge.** A 5-minute ES signal bar risks about $26 on MES, so $5.80 is about 22% of R.

**Incidental finding: the ES CSV shows no roll-week contamination signature.** Unlike MNQ, where every interleaved session holds a fake bar of 234+ points, ES roll-week sessions look like ordinary ones:

| | Roll weeks | Other sessions |
|---|---|---|
| Median of each session's largest 5-min range | 14.25 pts | 15.25 pts |

The largest clean-session range is 57.75 points. So `es_1min_2025_2026.csv` is probably single-contract over 2025, and dropping those sessions was merely conservative. This says nothing about whether its 2026 rows are the front month, which was not checked.

## Outputs

- `exploratory_clean.py`
- `results.json`
- this file
