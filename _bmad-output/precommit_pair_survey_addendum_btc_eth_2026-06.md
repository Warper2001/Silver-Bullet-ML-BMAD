# Pre-Commitment Addendum: BTC–ETH Divergence Fade (Gate 0)

**Date:** 2026-06-12
**Parent pre-registration:** `_bmad-output/precommit_pair_divergence_survey_2026-06.md`
(commit `b54fb08`); parent results: `db3abd0` (all five original pairs FAIL, closed).
**Status:** committed BEFORE the first BTC_ETH simulation runs.

## Scope

One additional pair under the identical frozen protocol (same dev window
2025-05-01 → 2026-02-28, same dollar grid {30,40,50,60} × {1.0,2.0}, same
primary spec $40/1.0×, same five Gate 0 criteria with per-pair breakeven WR,
same both-directions policy, same RTH 09:30–15:55 ET session, same decision
rule). This addendum does NOT reopen any closed pair.

## Pair specification

| Item | Value | Rationale |
|---|---|---|
| Leg A (traded) | BTC via **MBT** (0.1 BTC, $0.10/pt, tick 25 pts = $2.50) | $40 threshold = 400 BTC pts ≈ 0.5% at BTC ~$80k — reachable in 5 bars |
| Leg B (anchor) | ETH | MET as traded leg is structurally excluded: $0.10/pt PV with ETH ~$1.8–3k makes the frozen $40 threshold ≈ 15–25% of price — unreachable |
| Signal data | Kraken perps `PF_XBTUSD` / `PF_ETHUSD` 1-min (on disk) | Continuous — zero roll artifacts; CME MBT tracks within bps at 5-bar horizon |
| Commission RT | $2.84 (TopstepX MBT) | BE WR at $40 = 53.6% |
| Slippage stress | $5.00 RT (1 tick/side) | Stressed BE WR = 59.8% |
| Beta clip | **[0, 200]** (pair-level override; default [0,10] unchanged elsewhere) | BTC/ETH price-change beta runs ~20–45 (price ratio ~44×). The template clip [0,10] would silently under-hedge — the S26 "transplant" failure class. The clip is a numerical-stability bound, not a tuned parameter; frozen here before any run. |
| Roll dates | none | perpetuals |

## Disclosed weaknesses (frozen before results)

1. **Venue proxy:** signal = Kraken perps, economics = CME MBT. Acceptable for
   Gate 0 screening; any Gate 1 pre-registration MUST first replicate the
   signal on CME BTC/ETH futures data.
2. **No retroactive holdout for crypto.** Post-2026-03-01 Kraken data was
   already accessed by prior experiment families (FRRF, TSMOM-RF, BTC-CARRY,
   S26-crypto). No sealed slice is claimed; **any Gate 2 for this pair must be
   prospective** (data collected after 2026-06-12).
3. **Harness change:** `study_pair_divergence_survey.py` gains a per-pair
   `beta_clip` override (default-preserving). The MNQ/ES control is re-run in
   the same execution to confirm bit-identical reproduction.

## Decision rule

Identical to parent: qualifies for Gate 1 prereg consideration iff primary
spec passes all five Gate 0 criteria, ≥4/8 same-direction grid cells positive
EV, WR clears the stressed breakeven (59.8%), with the venue-proxy replication
requirement above stacked on top. Fail → pair closed, no re-sweeps.
