# Pre-Registration: HG (Copper) Gate-1 — One-Shot Sealed-Holdout Test

**Date sealed:** 2026-07-04 (committed BEFORE any holdout access; no bar of HG data on/after 2026-03-01 has been evaluated by any strategy code)
**Author:** Alex (session run per Alex's instruction: "write the Gate-1 holdout prereg")
**Lineage:**
- Exploratory result (frozen): YANK cross-instrument sweep 2026-06-25/26, seal 8f33f12
- Cost basis (measured): `precommit_hg_slippage_measurement_2026-06.md` + Amendment 1 (`0861e2f`) → verdict PASS `hg_slippage_verdict_20260704.md` — this prereg is the authorized consequence of that PASS.

---

## Hypothesis (H-HG1)

The frozen YANK structural engine (bearish-FVG + H1-sweep + M15-CHoCH, ML-off, structural mode), which produced gross PF 1.439 on HG micro copper in the exploratory window, retains a net edge ≥ the program bar on the unseen holdout period 2026-03-01 → 2026-06-12 at the **measured** all-in cost of $4.00/RT.

## Frozen inputs (nothing below may change between seal and run)

- **Engine/config:** the repo's frozen YANK structural path — `backtest_tier2_1year_validation.py --instrument hg --structural`, ML disabled. No parameter, gate, or exit differs from the run that produced the frozen IS trade list.
- **Frozen IS reference:** `data/reports/backtest_1year_20260625_225218.csv` — N=95, gross PF 1.439, gross total +$630, gross avg +$6.63/trade (1 micro contract, window 2025-05-19 → 2026-02-28).
- **Cost basis:** all-in **$4.00/RT** per contract (measured pooled median spread $2.50 + $1.50 commission, Amendment 1 PASS). Sensitivity cost **$5.25/RT** (worst qualifying session) — reported, non-binding.
- **Holdout window:** 2026-03-01 → 2026-06-12 (end of available data). Data source: `data/processed/dollar_bars/1_minute/hg_1min_2025_2026.csv`, whose ≥2026-03-01 rows must byte-match the sealed reference copy `data/sealed_holdout/hg_1min_holdout_20260301_plus.csv` (integrity check, step 2 below). Only the file's end-date metadata has been observed pre-seal; no prices/structure examined.

## Protocol (three steps, in order)

**Step 1 — Reproduction gate (no holdout involved).** Re-run the IS window first:

```
.venv/bin/python backtest_tier2_1year_validation.py --instrument hg --structural \
  --ml-threshold 0.0 --start 2025-05-19 --end 2026-02-28
```

Must reproduce the frozen IS result: N=95, gross PF 1.439 (±0.005), gross total +$630 (±$5), and a trade list matching `backtest_1year_20260625_225218.csv` on (entry_time, direction, pnl). **If it does not reproduce, ABORT — no holdout access.** Diagnose config drift, fix until reproduction passes, and only then proceed. Reproduction-gate iterations are unlimited (they touch no holdout data); the holdout run remains single-shot.

**Step 2 — Holdout integrity check (metadata only).** Verify the working file's rows timestamped ≥ 2026-03-01 are identical to `data/sealed_holdout/hg_1min_holdout_20260301_plus.csv` (row-count + content diff). Mismatch → ABORT and investigate; the sealed 444 copy is authoritative.

**Step 3 — One-shot holdout run.**

```
.venv/bin/python backtest_tier2_1year_validation.py --instrument hg --structural \
  --ml-threshold 0.0 --start 2026-03-01 --end 2026-06-12 \
  --preregistration <SHA of the commit sealing this document>
```

The script auto-verifies the SHA and appends the access record to `data/sealed_holdout/ACCESS_LOG.md` (commit that log change). Net P&L is then computed offline from the emitted trade list: `net_i = gross_i − $4.00` per trade (1 contract). **One run. No re-runs, no parameter changes, no subgroup selection, regardless of outcome.** Result is recorded in ACCESS_LOG and in a verdict doc either way.

## Sealed decision rule (evaluated at $4.00/RT all-in)

- **PASS:** net PF ≥ 1.10 **AND** N ≥ 15 **AND** ex-top-3-days net total > $0
  → authorizes drafting a **deployment pre-registration** for prospective paper trading (TS SIM, MHGU26, 1 contract). Deployment itself remains a separate gate requiring Alex's explicit go; nothing trades from this result.
- **INSUFFICIENT:** N < 15 (regardless of PF) → PARK; no verdict on the edge; option to log prospectively on paper.
- **MARGINAL:** net PF 1.00–1.10 (N ≥ 15) → PARK; no deployment path; may only be revisited with fresh prospective data under a new prereg.
- **FAIL:** net PF < 1.00 (N ≥ 15) → HG closed as a net candidate; cross-instrument claim logged as "structural edge did not survive the holdout at measured cost."

Reported but **non-binding**: net PF and expectancy at the $5.25/RT sensitivity cost; net $/trade vs the $2 economics bar (a deployment-prereg concern, not a Gate-1 criterion); win rate, exit-type mix, monthly P&L, max drawdown; bearish-only note (engine is bearish-only by freeze). Expected N for context: IS rate ~10.1 trades/month → ~34 expected in 3.4 months; N ≥ 15 guards against a thin-sample verdict.

## Known-contamination disclosures

- The operator knows the 2026 macro narrative (Iran war regime Mar–May 2026) and knows MNQ-instrument results on the same calendar window (YANK MNQ holdout PASS, S26/S27 results, Option B OOS FAIL). No HG price data in the window has been examined. Copper's holdout regime overlaps the war months — if the verdict is regime-lucky/unlucky, that is what prospective paper trading after a PASS is for.
- `hg_1min_2025_2026.csv` physically contains the holdout rows; only its header and final timestamp were viewed (to determine the window end). The script-enforced prereg gate has blocked strategy evaluation on ≥2026-03-01 bars throughout.
- The slippage measurement (2026-06-26 → 07-03) post-dates the holdout window entirely; measured cost is not fit to holdout prices.
- This prereg is committed on branch `worktree-innovation-strategy-trump-edge` (PR #2); the commit SHA is resolvable repo-wide (shared object store), which is what the script's `--preregistration` verification requires.

## Kill precedents acknowledged

S26 net-cost death (costs are inside the primary metric here, at measured not assumed values); MIM-NB fat-day fragility (ex-top-3 criterion); S26-KZ subgroup pattern (no segment/subgroup may rescue a failing full-window result); iteration-loop pattern (single sealed shot, no re-entry); Option B boundary honesty (a 1.09x PF is MARGINAL/PARK, not "almost passed").

## Out of scope

PL (platinum) — needs its own slippage amendment seal and its own Gate-1 prereg if pursued; combine-fit (account sizing, floor-monitor integration, correlation with the live book) — deployment-prereg territory, not Gate-1.
