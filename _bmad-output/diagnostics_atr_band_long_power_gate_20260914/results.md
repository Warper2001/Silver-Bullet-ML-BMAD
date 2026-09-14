# ATR-band long power gate — results (2026-09-14)

- **Plan:** `analysis_plan.md`, sha256 `a900bd88…5968`. It was committed with the script and tests in `ac65ac0` before the run. The script checks the plan's hash before running.
- **Script:** sha256 `105482cd…6184`, byte-identical to the pre-run commit.
- **Input:** `/root/mnq_historical.json`, the raw TradeStation 1-minute bars, with its sha256 in `results.json`.
  - 788,780 records were kept, 2023-11-30 19:01 → 2026-02-27 17:00 ET.
  - 62,277 records stamped on or after 2026-03-01 UTC were skipped during parsing and never stored.
- **Window:** RTH, 516 eligible sessions of 576 (2023-12-01 → 2026-02-27, 40,248 five-minute bars). Globex, 474 of 578.
- **Firewall:** nothing under `data/sealed_holdout/` was opened. No statistic of price after a real fill was computed.
  - Dispersion comes from placebo pairings: 507 shifts for RTH and 465 for Globex.
  - The identity pairing is refused, and a synthetic test confirms the refusal (7/7 tests pass).

## Verdict: UNDERPOWERED

| Primary cell | Value |
|---|---|
| Session and fill | RTH, trade-through |
| Sizing and cost | one contract, $5.80 per round trip |
| Effect | central θ = 0.10R |
| Events | N_upper = 79, N_lower = 68 (0.15 per session) |
| Mean risk R$ | $71.39 (median ATR 23.0 pts) |
| Net edge μ_net | **+$1.34 per trade** |
| σ_v | $91.69 (placebo $84.40; bracket bound $88.43 × 1.037) |
| **Power, in-sample (N_upper / N_lower)** | **6.5% / 6.4%** |
| Trades for 80% power | ~29,000, about **750 years** at the observed rate |
| Power of a holdout-only confirmation | 5.5% |

**This is terminal for the construct as specified** (plan section 7). No return was computed, so the window is unspent for this construct.

**All 48 central-θ cells** ({RTH, Globex} × {trade-through, touch} × {one contract, constant risk} × {low, primary, double cost} × {σ_placebo, σ_v}) come out UNDERPOWERED or COST-BOUND. None reaches MARGINAL or better.

## Why it fails

**It is almost cost-bound.**
- At the primary cost, the gross edge breaks even at **0.081R**. The central effect of 0.10R leaves $1.34 per trade.
- Even the optimistic 0.20R (a 51% win rate at the 4:3 brackets) has only **20.5%** power and would need about 19 years.
- The pessimistic 0.05R is net negative.

**It is rare.**
- A trade-through of a limit 3.1 ATR below the prior close fills 0.15 times per RTH session: 79 fills in 516 sessions.
- Holding one position at a time costs little (68 fills).
- In RTH, "touch" and "trade-through" produce identical counts. A 3.1-ATR plunge never stops exactly at the limit tick.

**Dispersion is not the binding constraint.**
- Placebo σ ≈ 1.1R, with no clustering penalty (ratio 1.04).
- Only 13.9% of placebo trades reach the flatten bar, so the conservative bracket bound lifts σ by just 8.6%.

| θ (gross, R) | Power, in-sample (RTH, $5.80, σ_v) | Years for 80% power |
|---|---|---|
| 0.05 pessimistic | 3.1% | net negative |
| **0.10 central** | **6.5%** | **~750** |
| 0.20 optimistic | 20.5% | ~19 |

## Sensitivities (pre-declared; none changes the verdict)

| Cell (central θ, σ_v) | N_upper | μ_net | Power | Verdict |
|---|---|---|---|---|
| RTH, low cost $2.22 | 79 | +$4.92 | 12.1% | UNDERPOWERED (~56 years) |
| RTH, double cost $11.60 | 79 | −$4.46 | — | COST-BOUND |
| RTH, constant-risk sizing, $5.80 | 79 | +0.009R | 5.7% | UNDERPOWERED |
| Globex, $5.80 | 462 | −$1.15 | — | COST-BOUND |
| Globex, low cost $2.22 | 462 | +$2.43 | 19.9% | UNDERPOWERED (~18 years) |

**Globex fills six times as often** (0.97 per session), but overnight ATR is small: median 13.9 pts, mean R$ $46.48. Cost therefore takes 16% of R, and the primary cell is cost-bound.
- 149 of its 462 fills fall in the 9 o'clock hour. That is the RTH-open burst measured against an overnight ATR, a different event from the construct's.

**The only cell near 80% is not the construct's claim.** Globex at low cost with the optimistic 0.20R reaches 75.4% power (N_upper). That combines the marketed edge with the cheapest fee schedule. Picking it now would be selection after seeing the gate.

## Data finding (affects other work)

The raw 1-minute file **interleaves contracts minute by minute during roll weeks.** Before the cutoff there are 20,760 label switches, each a ±~240-point calendar-spread jump. This gate excluded those sessions.
- **Any continuous series built from this file without contract handling contains fake ±240-point bars in roll weeks.** That includes `mnq_1min_2025.csv` (5,583 mixed-contract bars).
- **Unverified, but worth checking:** the H2/L2 gate ran on that CSV. Its risk tail (p90 R$ $466, "20% of events risk more than $150, mostly the high-volatility months of 2025") is the size of one splice. About 27 of its 297 RTH sessions fall in interleaved roll weeks.

## Deviations from the plan

None. The data-provenance profile ran before the plan, and the plan says so (section 1.2).

## Outputs

- `analysis_plan.md`
- `power_gate.py`
- `test_power_gate.py`
- `results.json`
- this file
