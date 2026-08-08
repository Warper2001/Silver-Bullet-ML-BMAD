# Ruling V9 — which ledger is the record of truth for GAP-1

**Date:** 2026-08-08
**Decided by:** party-mode round-table (Winston / Mary / Dr. Quinn / Amelia), at Alex's direction
**Blocks:** Phase 2 of `buildspec_gap_fade_tradestation_live.md`
**Seal affected:** `preregistration_gap_fade_panic_open.md` (32da5d5) — *OOS / Live Decision Rule*

---

## 0. Why this needed a ruling

The sealed decision rule is a **PF computed over the live trades**. Two files claim to be
that set and they disagree: `data/gap_fade/trades.csv` says N=18 / +$1,874.50 / PF 1.595;
`data/trades.db` says N=18 / +$1,130.00 / PF **1.359**. The CSV double-counts 2026-06-25
and is missing 2026-08-06 (§8c of the build spec). One of them decides whether gap-fade
scales to 2 contracts. Until this is ruled, the seal has no input.

---

## 1. The measurement that decided it

```
$ .venv/bin/python tools/verify_chain.py --reconcile

[  OK  ] data/gap_fade/trades.csv: 18 rows, chain verifies
         INCOMPLETE: in trades.db but not in this file: ['2026-08-06']
         DUPLICATED rows for: ['2026-06-25']
[BROKEN] data/gap_fade/decisions.csv: 30 rows, first mismatch at row 30 (2026-08-07)
[  OK  ] data/gap_fade/fills.csv: 12 rows, chain verifies
```

**`trades.csv`'s hash chain verifies perfectly while the file is both incomplete and
duplicated.** That is not a bug in the chain — it is what a chain is. Reverting an
append-only file to an earlier commit yields a valid *prefix* of the same chain. The chain
proves *no row present was edited*. It says nothing whatsoever about *which rows are
present*.

`decisions.csv` did break, and the break is instructive: the live process kept appending
after the revert, chaining onto an in-memory head the file no longer contained. The break
is the **fingerprint of the loss**. `trades.csv` concealed the identical event only because
no trade happened to be appended afterwards.

**Conclusion: tamper evidence and completeness are different properties, and the audit
trail only ever had the first one.** The sealed decision rule needs the second.

---

## 2. What each store actually guarantees

| Property | `data/trades.db` | chained CSVs |
|---|---|---|
| Complete under restart-replay | ✅ `ux_trade_identity` UNIQUE + `INSERT OR IGNORE` | ❌ appends a duplicate (2026-06-25) |
| Durable against tree operations | ✅ not a git-tracked flat file | ❌ **proven failure** (2026-08-06) |
| Tamper-evident | ❌ no chain, no signature | ✅ SHA-256 chain |
| Records **non**-trades (`NO_SETUP`, `SKIPPED_FRIDAY`) | ❌ | ✅ `decisions.csv` only |
| Records realized broker fills vs modeled | ❌ | ✅ `fills.csv` only |

Neither store dominates. Neither is redundant.

---

## 3. Ruling

1. **`data/trades.db` is the record of truth for the sealed decision rule.** The PF, N, and
   calendar condition in `preregistration_gap_fade_panic_open.md` are computed from it and
   from nothing else. It holds the two properties the rule actually depends on — completeness
   under replay, and durability.

2. **The chained CSVs are retained as tamper evidence and as the sole source of two things
   the DB does not hold** — the decision record for sessions that produced no trade
   (`decisions.csv`, required for the trades-per-session sanity check) and realized-vs-modeled
   fill fidelity (`fills.csv`). They are **not** a P&L ledger and must not be quoted as one.

3. **The control is reconciliation, not either file alone.** `tools/verify_chain.py
   --reconcile` is the check: a broken chain means someone or something edited history; an
   INCOMPLETE/DUPLICATED result means the DB and the trail disagree. Either is an ops finding.
   Run it on the ops schedule alongside `combine_ops_healthcheck.py`.

4. **Nothing is repaired.** Per the 2026-08-06 sigma precedent — *corroborate, don't repair*:
   - The 2026-06-25 duplicate **stays** in `trades.csv` / `decisions.csv`. It is not deleted.
     `trades.db` already counts it once, so the sealed metric is unaffected.
   - The 2026-08-06 hole **stays**. Re-appending it would place it out of order behind
     2026-08-04 and would require re-chaining, i.e. manufacturing a record. `trades.db`
     carries the trade.
   - `decisions.csv`'s broken chain **stays broken**. Re-chaining it would erase the only
     surviving evidence that the 2026-08-06 revert happened. A chain rewritten when it is
     inconvenient was never evidence.

   This document is the correction. The files keep their holes; the holes have a citation.

5. **Every number this program has published for gap-fade from `trades.csv` is withdrawn**
   and superseded by the DB figures: **N=18, net +$1,130.00, PF 1.359**, mean $62.78,
   sd $545.13, maxDD −$1,560. Still inside the seal's *scale to 2 ct* band (PF > 1.20), by
   0.159 rather than the 0.395 first reported.

---

## 4. Consequences for the build

- **V9 is closed.** Phase 2 is no longer blocked by it.
- `data/gap_fade/*` is now gitignored (`.gitignore:180`, from PR #35), so the durability
  failure mode is closed going forward — but that fix landed *after* the loss, which is why
  the reconciliation control in §3.3 exists rather than relying on it.
- The live-TS deployment prereg (§6 of the build spec) must name `trades.db` as the evidence
  base and cite this ruling.
- **Open, not ruled here:** `trades.db` has no tamper evidence at all. That is acceptable
  while it records a paper strategy. Before it is the evidence base for a funded account, it
  should acquire one. Raised as **V10**.

---

## 5. Remaining blockers on Phase 2

| | Item | State |
|---|---|---|
| V7 | Funded live account `210MWN27` is not isolated | **open** — awaiting transfer to `210URF13` |
| V8 | 2026-06-25 double-append | closed — pre-guard commissioning artifact; guard now fails closed + tested |
| V9 | Record of truth undecided | **closed by this ruling** |
| V10 | `trades.db` has no tamper evidence | new, not blocking paper, blocking funded |
