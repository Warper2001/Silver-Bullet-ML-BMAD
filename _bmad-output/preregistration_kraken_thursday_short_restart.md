# Pre-Registration RESTART: Thursday Short (DOW-THU) — prospective accrual reset to N=0

**Status:** PROSPECTIVE — SEALED 2026-09-11
**Supersedes:** `preregistration_kraken_thursday_short.md` (parent, sealed 2026-06-21),
Amendment 1 (LR gate, `0daf3a6`), Amendment 2 (venue CME MBT/MET via TS SIM, `cf0e7f8`),
Amendment 3 (evaluation series & counterfactual ledger, sealed 2026-07-10)
**Strategy:** DOW-THU (Day-of-Week Thursday Short), BTC + ETH, equal-notional
**Sample state at seal:** **N = 0.** All previously accrued Thursdays are VOID (§2).

**Tamper-evidence:**

| Artifact | SHA-256 |
|---|---|
| `thursday_short.py` (post-fix) | `b97b87151b76671cf3510391bb2d9264c77f9e1374c205de4d1c31f87d998a9c` |
| `tools/combine_ops_healthcheck.py` (post-fix) | `60ed2c52d00c7141045fe8b42cf429d886838197f2c66dea433d1f2382483424` |

Commit this document before the next Thursday (2026-09-17 00:00 UTC) to make it
tamper-evident. The two hashes above are the **fixed** files described in §3; verify
they match the deployed live checkout before the first counted Thursday.

---

## 1. Why this restart exists

The prior accrual is not a valid prospective sample. Between 2026-07-30 and 2026-08-27,
**five consecutive Thursdays produced no decision row and no trade**, and two further
Thursdays that *did* trade had their ledger rows destroyed. Neither failure was detected
until a fleet-wide audit on 2026-09-10. Two independent causes:

**(a) The bot was dead for 24 days.** It crashed 2026-07-27 23:21:42 UTC when a
shared-`.env` write race (two bots' token-refresh loops interleaving a non-atomic
read-modify-write) wiped `TRADESTATION_CLIENT_ID` and `TRADESTATION_CLIENT_SECRET`. The
race itself was fixed 15 minutes later by commit `017521c`. The bot was never revived,
because `main()` caught its own fatal exception, logged it, and returned normally —
exiting 0, which systemd's `Restart=on-failure` correctly reads as a clean shutdown.
Thursdays **2026-07-30, 08-06, 08-13, 08-20** passed with the process down: no log
output, no ledger rows, and no alarm. The bot's own in-loop "no entry attempted" alarm
cannot fire when the process itself is not running, and `trader-thursday-short` was
absent from `tools/combine_ops_healthcheck.py`.

**(b) Two real Thursdays were erased from the ledgers.** 2026-08-27 and 2026-09-03 both
traded. Commit `27e4e1e` (2026-09-08) recorded the 09-03 data onto a parent tree that
predated the 2026-09-02 reconstruction of 08-27, and never merged to `main`. The live
files became that stale content plus later appends — dropping 08-27 entirely and 09-03's
decision row. This is the "git-tracked while live-appended" failure class named in
`ChainedCsv`'s own docstring, recurring for at least the fourth time.

**Why the hash chain did not catch it:** `tools/verify_chain.py` proves no row was
*edited*; it does not prove the file is *complete*. A prefix of a valid chain is a valid
chain. Nothing was asking the completeness question.

## 2. What is voided, and what that costs

All **seven** genuinely-traded Thursdays are void for decision purposes:

| Thursday | Net (realized) | Status |
|---|---|---|
| 2026-07-02 | −$506.55 | void |
| 2026-07-09 | −$107.80 | void |
| 2026-07-16 | +$277.00 | void |
| 2026-07-23 | +$263.60 | void |
| 2026-08-27 | +$64.90 | void — early `shutdown` exit at 6h46m, not the 23h design hold |
| 2026-09-03 | −$783.70 | void |
| 2026-09-10 | +$223.50 | void |
| **Total** | **−$569.05** | — |

This sample is discarded **not because it is unfavourable** — it is, at PF 0.547 — but
because a sample with five unexplained absences is self-selected, and a self-selected
sample cannot support the parent's decision rule whichever way it points. Discarding a
losing sample is the direction that costs credibility least; recording that here so the
decision cannot later be read as favourable-subset selection (the failure pattern this
project has documented three times).

The voided rows are **not deleted**. The ledgers are reconstructed to their true
seven-Thursday state for audit completeness (`tools/reconstruct_thursday_ledger_20260911.py`),
and are excluded from the accrual by date, not by erasure.

## 3. Fixes that make the restart meaningful

A restart with the same blind spots would simply re-accumulate the same defect. Landed
before this seal:

1. **`thursday_short.py` `main()` now re-raises after logging**, so a fatal exits
   non-zero and the existing `Restart=on-failure` / `RestartSec=60` engages. A clean
   SIGTERM still exits 0. This alone would have prevented four of the five missed
   Thursdays.
2. **`trader-thursday-short` added to `tools/combine_ops_healthcheck.py`** (420s log
   staleness, no window — it idles 24/7 between Thursdays). This is the only check that
   can detect the process being *down*, as opposed to running-but-idle.

Audited and found clean: no sibling bot shares the swallow-and-exit-0 pattern; all call
`asyncio.run()` directly.

## 4. Trading rules — UNCHANGED

This restart changes **no trading behaviour whatsoever**. Every signal, sizing, venue and
exit rule carries forward exactly as sealed:

- **Entry:** Thursday 00:00 UTC, short both legs, equal notional (1 MBT + notional-matched MET).
- **Exit:** Thursday 23:05 UTC scheduled, or the 5% per-leg intraday stop (Amendment 2).
- **LR gate:** instrumentation only. `fetch_btc_lr_slopes()` is logged to both ledgers and
  **does not gate entry** (Amendment 1 status preserved; confirmed in code — its result is
  never referenced in any `if`).
- **Venue:** CME micro crypto futures MBT/MET on TradeStation SIM (Amendment 2).

## 5. Evaluation series — UNCHANGED from Amendment 3

- Scored on **realized TradeStation SIM fills, including the 5% per-leg stop**, as logged
  in `data/thursday_ts/trades.csv`.
- **Costs at evaluation: 10 bps per leg round-trip**, matching the sealed backtest pool.
- Per-Thursday return = equal-weight mean of the two legs' net returns.
- `data/thursday_ts/counterfactuals.csv` continues as the knowledge-only held-to-23:05
  ledger, with the resolution method fixed in Amendment 3 §3.

## 6. Decision rule — UNCHANGED

**PASS if Sharpe > 0.80 after N ≥ 30 prospective Thursdays**, counted from the first
Thursday after this seal.

- **First counted Thursday:** 2026-09-17.
- **N ≥ 30 reached:** approximately 2027-04-15, absent further absences.
- **Interim look:** none. No mean, no PF, no verdict before N = 30. Accrual is not progress.
- **Stopping date:** 2027-12-31. If N < 30 by then, the result is INCONCLUSIVE, not FAIL.

## 7. NEW: continuity requirement

This is the one substantive addition, and the reason a restart rather than an amendment.

**Every Thursday between the seal date and the verdict must produce a row in
`data/thursday_ts/decisions.csv`** — `ENTERED`, `SKIPPED_NOT_FLAT`, `NO_MARKS`, or
`REJECTED`. A Thursday with no decision row is a **protocol breach**, not a skipped
observation.

On a breach:
1. It must be investigated and its cause recorded in this document as a dated amendment
   before the next counted Thursday.
2. The missing Thursday counts as an **absence**, tracked explicitly.
3. **If cumulative absences exceed 3, the accrual is void and must restart again.** Three
   is the tolerance for genuine venue outages; beyond that the continuity claim fails.

**Verification cadence:** `tools/verify_chain.py` proves no row was edited. Completeness
is now checked separately — the expected decision-row count is one per Thursday elapsed
since the seal, and any shortfall is a breach under this section. Run both checks at each
monthly review.

**Absence log (append here):**

| Thursday | Decision row? | Cause | Cumulative absences |
|---|---|---|---|
| _(none yet — accrual begins 2026-09-17)_ | | | 0 |

## 8. What would falsify this restart's own premise

If the fixes in §3 hold, absences should be zero. If absences accumulate anyway, the
problem is not the bot's exception handling but the venue or the schedule, and §7's
3-absence trigger forces that conclusion rather than letting it be absorbed silently.
