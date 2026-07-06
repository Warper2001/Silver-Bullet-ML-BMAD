# HG (Copper) Gate-1 Holdout — VERDICT: FAIL, CLOSED

**Date:** 2026-07-04
**Prereg:** `_bmad-output/preregistration_hg_gate1_holdout.md`, sealed at commit `fbd7afe` before any holdout access.
**Access:** auto-logged by the backtest script's prereg gate + RESULT block in `data/sealed_holdout/ACCESS_LOG.md`. Single shot, executed verbatim.

---

## Protocol execution

**Step 1 — Reproduction gate: PASS.** IS re-run (`--instrument hg --structural --ml-threshold 0.0 --start 2025-05-19 --end 2026-02-28`) reproduced the frozen exploratory result exactly: N=95, gross PF 1.4386, total +$630.00, trade-list identical to `backtest_1year_20260625_225218.csv` on (entry_time, direction, pnl). No config drift; the holdout run tested the same engine that earned the slot.

**Step 2 — Holdout integrity: PASS.** Working file's 66,368 rows ≥ 2026-03-01 are semantically identical to the sealed 444 copy (OHLCV + volume byte-exact). Byte-level diffs were serialization artifacts only: `T`-vs-space timestamp separator and float-repr noise in the `notional` column (e.g. `442124.99999999994` vs `442125.0`). Recorded honestly; data content is the same.

**Step 3 — One-shot holdout run (2026-03-01 → 2026-06-12): FAIL.**

| Metric | Gross | Net @ $4.00/RT (binding) | Net @ $5.25/RT (sensitivity) |
|---|---|---|---|
| N | 26 | 26 | 26 |
| PF | 0.563 | **0.463** | 0.435 |
| Total | −$295.00 | **−$399.00** | −$431.50 |
| Exp/trade | −$11.35 | −$15.35 | −$16.60 |
| WR | 34.6% | 34.6% | 34.6% |
| Ex-top-3-days | −$483.75 | −$563.75 | −$588.75 |

Monthly net ($4.00): Mar +$10, Apr −$114, May −$229, Jun −$66. Max net drawdown −$470.50. Exit mix: 13 time-stops / 12 stop-losses / 1 take-profit.

**Sealed rule: N=26 ≥ 15 (valid verdict), net PF 0.463 < 1.00 → FAIL. HG is closed as a net candidate.** Logged per the prereg: "structural edge did not survive the holdout at measured cost." No re-runs, no subgroup rescue, no parameter changes.

## Post-mortem (why it died)

1. **Not a cost death.** Unlike S26 (gross edge eaten by costs), HG's gross holdout P&L was itself negative (PF 0.563). The $4.00 measured cost only deepened the hole. The slippage measurement did its job — the failure is in the signal, not the friction.
2. **The structural fingerprint flipped, not faded.** IS: WR 44%, broad non-lottery edge, TP-richer mix. Holdout: WR 34.6%, 12 SL vs 1 TP. The bearish-FVG aftermath on copper resolved *against* the trade in this window — the same follow-through failure shape as the OOF probe warned tuning couldn't fix.
3. **Regime overlap was disclosed in the seal:** the holdout months (Mar–Jun 2026) are the war regime. Copper in that window did not behave like copper in the exploratory window. Whether the IS edge was regime-specific or noise, the sealed test can't distinguish — and doesn't need to: either way it doesn't earn deployment.
4. **Program pattern confirmed again:** exploratory in-sample structural fingerprints (HG netPF 1.183 rough / 1.151 at measured cost) have now repeatedly failed to transfer forward out-of-window (S26, Option B, now HG Gate-1). The pipeline is working exactly as designed — this is the third seductive candidate killed at the cheapest possible gate, with zero dollars deployed.

## Consequences

- **HG (copper) port: CLOSED.** No deployment prereg, no paper trading, cross-instrument claim updated.
- **PL (platinum): prior substantially lowered.** Same engine, same calendar window, better IS netPF (1.274) but thinner market and full-contract sizing. Its slippage amendment + Gate-1 would cost another holdout spend; recommend treating PL as parked-by-default unless Alex explicitly wants the shot.
- **ES/YM correlated survivors:** unchanged status (duplicates of MNQ exposure, never candidates for independent deployment).
- **Banked assets:** measured MHGU26 cost card (2.0 ticks dead-stable, $4.00 all-in), the dead-front-month roll-detection lesson (bind by quote-staleness, not sample count), and the reproduction-gate protocol (worked perfectly — exact trade-list match before holdout spend).

Evidence files (committed): `data/reports/backtest_1year_20260704_180148.csv` (Step 1 reproduction), `data/reports/backtest_1year_20260704_180517.csv` (Step 3 holdout trades).

## Addendum 2026-07-06 — combine-fit confirms the death was signal, not risk

Ran `tools/combine_fit_gate.py` on the frozen N=95 HG trade list at the measured $4.00/RT,
against the Topstep 50K trailing-MLL math (floor $48K, EOD ratchet, $2K MLL). Micro copper
(MHG, $2,500/pt) is **drawdown-safe**: worst single SL **−$79**, max DD from HWM **$360**,
equity never comes within **$1,660** of the trailing floor — **zero bust risk**. It "fails"
the tool only on the profit-target gate (+$250/yr net at 1 micro ≪ the $3,000 target — too
small at safe size, not too dangerous). So copper's port died **purely on the OOS holdout
signal** (net PF 0.463 above), NOT on combine-fit — the exact opposite of the platinum port,
which passed slippage but is **combine-INcompatible** (1 full 50oz contract busts the $2K MLL;
`_bmad-output/pl_combine_fit_verdict_20260705.md`). Scaling micro copper toward the $3K target
(~10 contracts) would reintroduce bust risk (SL→−$750, DD→~$3.6K > $2K MLL) — moot, since one
does not scale a signal that already failed forward. This is confirmatory; the port remains
CLOSED on the holdout result. Tool: `tools/combine_fit_gate.py` (commit a42607c, on main).

## Program state

Live book unchanged: MIM-NB 1ct + YANK 2ct on the real combine, GAP-1 on TS SIM paper (NO-GO until N≥30, ~late Sep), S25 accrual (decision ~Jul 23). Research queue: no active candidate holds a pending holdout slot; next in line per earlier reviews — S26-KZ prospective, S27 IFVG (blocked on S25), and whatever survives the next exploratory sweep.
