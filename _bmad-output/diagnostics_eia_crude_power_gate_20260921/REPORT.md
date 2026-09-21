# Power gate — EIA-Wednesday crude intraday momentum (Wen et al., Energy Journal 2023)

**Verdict: UNDERPOWERED.** Outcome-blind: the paper's published statistics only, no bars read, no strategy built.
Files: `power_gate.py` (decision rule pre-committed in its docstring), `power_gate_output.md`, `power_verdict.json`.
Recon source: `_bmad-output/planning-artifacts/research/academic-lit-new-intraday-strategies-raising-weekly-t-2026-09-21/` (digest `eia-r2-1.md`).

## Result
| | value |
|---|---|
| paper effect per trade (USO, gross) | +1.64 bp, SD 21.0 bp, d = 0.078 (reproduces the paper's t: 1.90 vs 1.88) |
| trades needed, 80% power, one-sided 5% | **1,010** (DEFF 1.0); 1,515 (DEFF 1.5) |
| years to accrue at the paper's observed 44 events/yr | **22.9** (34.3 at DEFF 1.5) |
| years at the calendar maximum, 52 events/yr | **19.4** (29.1 at DEFF 1.5) |
| smallest effect detectable inside a 2y leash (N = 88–104) | d = 0.24–0.27, i.e. **3.1–3.4x the paper's own effect** |
| effect x0.75 / x0.50 / lower 1-SE | 35–41 y / 78–92 y / 86–102 y |

The verdict is one-sided by construction: it is computed at the paper's point estimate, which is the best case (the r3
window and the EIA-Wednesday subset were chosen in-sample; no out-of-sample test exists). Underpowered at the ceiling
means underpowered at every shrink and every lower rate.

## Cost flag (separate from power; does NOT change the verdict)
Gross edge per MCL contract = 1.64 bp x notional = **$0.82 at $50/bbl, $0.99 at $60, $1.31 at $80, $1.48 at $90 —
about one tick ($1.00).** So the net edge is positive only if the effective round-trip cost (spread + commission +
slippage) stays at or below roughly one tick. MCL spread and commission at 15:30–16:00 ET were **not retrieved**, so the
one-tick floor is an assumption; if the spread is one tick and any commission applies, net edge is at or below zero.

## What this gate does not say
- It does not say the effect is absent. It says a test could not see an effect this size within a 2-year leash.
- The leash (2.0y) is an operator-leash assumption carried over from the 2026-09-20 regime-hold gate, not derived.
- Inputs are USO ETF statistics (2006–2019). MCL, post-2019 and post-settlement (CL settles 14:30 ET) behaviour are untested.
- The volatility-half and first-half-of-month filters the paper found would cut the trade rate and were not tested for power;
  they are post-hoc splits and only worsen the arithmetic.

## Consequence
Per the project's own policy an UNDERPOWERED gate ends this hypothesis for confirmatory testing: no seal, no MCL data pull,
no backtest. It could only be reopened by a much larger effect than the paper reports, or by a different mechanism.
