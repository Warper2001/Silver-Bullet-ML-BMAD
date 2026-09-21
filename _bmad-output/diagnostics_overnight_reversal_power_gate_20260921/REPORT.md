# Power gate — overnight-intraday reversal on index futures (Della Corte, Kosowski, Wang, Nov 2015 draft)

**Verdict: UNDERPOWERED** under the pre-committed rule (tradable ceiling = the paper's own 1-minute entry retention).
**The verdict rests on one assumption that cannot be checked outcome-blind** (see "What decides it").
Files: `power_gate.py` (decision rule pre-committed in its docstring), `power_gate_output.md`, `power_verdict.json`.

## Design
Tradable form = a market-neutral 2-leg spread, 2 MNQ vs 3 MES (the stat-arb bot's basket): long the contract with the lower
overnight return, short the other, 09:30 ET open -> 16:00 ET close, about 1 trade per day. Firewall: the script never forms the
signal or pairs it with the open->close return. Inputs are the paper's published numbers plus signal-free properties of
`data/mim_x/{mnq,mes}_1min_2021_2024_frontmonth.csv` (992 days, 2021-01 to 2024-12; entirely after the paper's 1982-2014
sample, so it is unseen data a later test would need). `sealed_holdout` not touched.

## Result
| | value |
|---|---|
| paper per-day d (gross, same-open entry) | 0.257 (0.252 %/day / SD 0.980; Sharpe 4.08) |
| signal-free SD of the basket's open->close P&L | $257.64/day (robust $237.40) |
| round-trip cost, 2 MNQ + 3 MES (assumed cost card) | $14.70/day = 0.057 d ($15.10 with MNQ RT $2.24) |
| **paper as printed (0 delay), reference only** | net d 0.200 -> **0.6 years** (POWERED) |
| **tradable ceiling (1-min retention 0.306)** | gross d 0.079 ($20.24/day); no-cost **4.0 y**; net d 0.021 -> **53 y** (DEFF 1.0), 80 y (1.5) |
| 15-min retention 0.111 | gross $7.36/day, below cost: net d -0.029, **cost-bound** |
| smallest effect detectable in a 2y leash (N = 504) | d = 0.111 = 0.43x the paper's d |
| **retention needed for POWERED inside 2y** | **0.65** of the paper's gross effect (paper's own 1-min stock retention: 0.31) |

## What decides it
The entry-delay retention. The paper printed its number with signal and entry on the same open, and its own US-stock table
shows 0.36 %/day at +1 s, 0.11 at +1 min, 0.04 at +15 min. There is **no futures delayed-entry test** in the draft. The gate
applies the stock retention to futures as an assumption. Measuring the true futures retention *is* the aligned test, and that
would spend the 2021-2024 window. At retention 0.31 even a cost-free test needs 4.0 years of daily data; the 992 days on hand
(3.96 y) sit right at that edge and could not confirm a net edge. The look is informative only if retention is at least about 0.65.

## Caveats that push toward optimism (all favour the strategy, so the verdict is not fragile to them)
- The paper's d comes from a 5-index cross-section that includes the Nikkei. A 2-asset MNQ/MES pair has far less cross-sectional
  dispersion (overnight spread SD about 25-30 bp, roll days inflate the plain SD), so d transfers optimistically.
- The paper's index-futures lag-1 overnight-return regression has t < 2; its portfolio t = 13 is not reconciled with that.
- Costs are an assumed cost card (MES commission and slippage scaled from the MNQ memo), not measured fills.

## What this gate does not say
- It does not say the effect is absent, only that a test could not see a tradable, net-of-cost effect at the paper's own
  1-minute retention within 2 years.
- The 2.0y leash is an operator assumption carried over from the 09-20 and 09-21 gates.
- Correction to the recon: I suspected high correlation with GAP-1. That applies to a directional single-index version.
  This spread version is market-neutral, so correlation with GAP-1 is probably low, but it is a relative-value MNQ/MES trade,
  the family your 06-12 pair survey and 06-14 stat-arb review found had no edge on realistic execution.

## Consequence
Do not seal or run the aligned test on 2021-2024 on this evidence. If you want it anyway, the only defensible design is a
single-shot prereg whose pass bar is the derived retention of about 0.65 at the +1 min entry, citing this artifact and commit.
My recommendation is to park it: needing twice the retention the paper's own table shows is a poor bet for a one-shot spend of unseen data.
