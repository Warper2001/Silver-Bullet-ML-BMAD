# Ledger incident — gap-fade audit trail, 2026-08-06

**Status:** recovered 2026-08-12. Two injuries are permanent by design (see §5).
**Scope:** `data/gap_fade/{trades,decisions,fills}.csv`. No trading decision was affected;
the loss is to the record, not to the account.
**Related:** `ruling_v9_record_of_truth.md`, `tools/verify_chain.py`, `.gitignore` §"Live
append-only bot output".

---

## 1. What happened

The three gap-fade ledgers were git-tracked. Their last commit as tracked files was
`1910acb` (2026-08-04 22:52 UTC), whose content ends on 2026-08-04.

At **2026-08-06 20:00:36 UTC** a branch checkout (reflog: `main -> fix/s26-combine-unit-drift`,
logged 20:00:39 UTC) restored all three files to that committed state. Everything the bot
had appended on **2026-08-05** and **2026-08-06** was overwritten and lost.

`84bb7b3` (2026-08-06 21:14 UTC) untracked the files 74 minutes later, so the trigger is
gone. The hole it left was never filled — until now.

A file cannot be both a git-versioned artifact and a live append target. That lesson was
already written down in `.gitignore` at the time, drawn from the *same* checkout truncating
MIM-NB's chain. Only the MIM-NB half of the incident got a write-up; the gap-fade half was
observed on 2026-08-08 (it is described in `tools/verify_chain.py`'s docstring) and then
left alone for six days.

### The second, worse injury

The bot kept running through the revert. `ChainedCsv` held its chain head **in memory**, so
the next append chained onto a head the file no longer contained. That is why
`decisions.csv` fails verification at row 30 (2026-08-07) and always will.

`trades.csv` escaped chain corruption only by luck — no trade closed between the revert and
the next restart, so nothing was appended against the stale head.

## 2. What was lost

| File | Date | Content |
|---|---|---|
| `decisions.csv` | 2026-08-05 | `NO_SETUP` — gap 0.288% |
| `decisions.csv` | 2026-08-06 | `ENTERED long` |
| `trades.csv` | 2026-08-06 | closed trade, TP fill, **+$646.00** |
| `fills.csv` | 2026-08-06 | broker executions for that trade |

Six days of P&L reporting off `trades.csv` were therefore wrong by $646.00 in one direction
and $1,390.50 in the other (§5), for a net **$744.50 overstatement**, on the file the
service unit calls "the authoritative OOS P&L record" for a pre-registered decision gate.

## 3. How it was recovered

`journalctl` retention had already rolled past the window (oldest gap-fade entry:
2026-08-11), so the logs were no help. Three independent sources survived.

**`data/trades.db`** — written contemporaneously at 2026-08-06T15:01:01Z, never git-tracked,
never reverted. Supplies the entire 08-06 trade including its metadata blob (gap_pct,
gap_abs_pts, target, stop).

**TradeStation SIM `historicalorders`** — supplies the 08-06 broker executions: market entry
`965760604` @ 29368.00 at 13:30:02Z, TP limit `965760598` @ 29667.75 at 15:00:03Z, and the
`StopMarket` sibling `965760601` UROut. Realized $599.50 against $646.00 modeled, a
−$46.50 delta driven by 23.5pt of entry slippage against the modeled 09:30 open.

**TradeStation 1-min bars** — used only where it cannot change an outcome, because
TradeStation revises its own history. Re-deriving the **2026-08-04** decision today yields
`rth_open` 29224.5 against the 29223.25 the bot recorded live: 1.25pt of drift on a row we
can check.

That drift is why the two reconstructions are held to different standards:

- **2026-08-06** — every recomputed field reproduced the trades.db metadata *exactly*
  (gap_pct 1.089, gap_abs 323.0, prior_close 29667.5, rth_open 29344.5, target 29667.5,
  stop 28698.5). The row is taken from trades.db, and `gap_fade_ledger_repair.py`
  re-verifies all nine fields against it at apply time rather than trusting this document.
- **2026-08-05** — no trade, so no contemporaneous record exists and bars are all there is.
  The verdict survives anyway: gap 0.288% against a 0.500% threshold is a margin roughly
  170× the observed drift. **The action is certain; the two gap figures in that row are
  reconstructed**, and are marked as such in `data/gap_fade/ledger_repair_20260812.csv`.

## 4. What was done

Recovered rows were **appended at the tail** by `tools/gap_fade_ledger_repair.py`. Nothing
was rewritten, reordered, re-chained, or deleted.

They therefore sit *after* 2026-08-12 in the file. That out-of-order position is deliberate
provenance, not sloppiness: an append-only file showing an 08-05 row arriving last is
telling the truth about when it was written. The manifest records the source and the
verification status of each row.

The repair refuses to run while `trader-gap-fade` is active or holds an open trade —
appending under a running bot on the old code would leave its in-memory head stale and
break the chain again, which is the defect being undone.

## 5. What was NOT fixed, and why

**`decisions.csv` chain break at row 30.** Permanent. Re-chaining the file would make it
verify, and a chain you rewrite when it is inconvenient was never evidence. The break is
the fingerprint of the loss and it stays.

**`trades.csv` duplicate 2026-06-25** (+$1,390.50 counted twice; the first-day double append
across a restart, before the double-entry guard existed at `7c9bc0a`). Permanent for the
same reason — a row cannot be withdrawn from an append-only chain.

**Consequence:** `trades.csv` is now *complete* but still not *arithmetically correct*. It
overstates by $1,390.50. `data/trades.db` is the arithmetic authority: 18 distinct trades,
**+$1,130.00** net. Any P&L or N-count for the GAP-1 pre-registered gate must come from
there, or from `trades.csv` de-duplicated.

Both injuries are registered in `KNOWN_SCARS` in `tools/verify_chain.py`. They print on
every run, tagged `[SCAR]` with a pointer to this document, but do not set the exit code —
a monitor that cries every run stops being read. `--strict` reports them as findings again.

## 6. Why it cannot recur the same way

1. **The files are untracked** (`84bb7b3` + `.gitignore`). The trigger is gone.
2. **`ChainedCsv` re-reads its head from disk before every append**
   (`src/research/gap_fade_live.py`). If an outside writer truncates the file again, the
   chain stays valid and the accident degrades to a completeness question — which
   `verify_chain.py --reconcile` can answer. A head that moved under us is logged at ERROR,
   because a silent self-heal would be worse than the bug.
3. **Duplicate keys are refused.** `ChainedCsv.append` returns `False` and writes nothing
   when `date_et` is already present — INSERT-OR-IGNORE semantics, the same fix applied to
   `trades.db` on 2026-06-19.
4. **Something checks.** `tools/combine_ops_healthcheck.py` now runs `verify_chain.py
   --reconcile` and WARNs on any unregistered break, duplicate, or missing row. The
   six-day detection lag was not a tooling gap — `verify_chain.py` existed and was correct
   on 2026-08-08 — it was that nothing ran it.

Tests: `tools/test_gap_fade_ledger_repair.py`. `test_external_revert_does_not_corrupt_the_chain`
replays 2026-08-06 exactly and asserts the chain survives.

## 7. Still exposed

`ChainedCsv` exists in three copies.

| Copy | Status |
|---|---|
| `src/research/gap_fade_live.py` | fixed 2026-08-12 (single `key_field`) |
| `thursday_short.py` | fixed 2026-08-13 (composite `key_fields`, required) |
| `src/research/mim_nb_live.py` | **carries the defect** — frozen mid-Track-A by preregistration §7; needs its own change window |

MIM-NB's ledgers are untracked now, so the git trigger is gone there too; what remains is
that any *other* external writer would corrupt its chain rather than merely truncate it.

### The dedupe key is not portable between copies

gap-fade takes a single `key_field` (`date_et`) because its ledgers really are
one-row-per-session. thursday_short cannot: it writes **two legs per Thursday** (MBT and
MET), so a first-column guard on `thursday` would silently refuse every second leg — a
safety feature that deletes half the evidence. Its key is therefore
`("thursday", "symbol")`, and `decisions.csv` opts out entirely with `key_fields=None`,
because `current_thursday` is only set after a *successful* entry, so `SKIPPED_NOT_FLAT` /
`NO_MARKS` / `REJECTED` can each legitimately fire again on the next poll.

`key_fields` has no default in thursday_short's copy: every construction site must state
its key or `None` on purpose. `test_live_ledger_keys_are_actually_unique` checks the
chosen keys against the real files, not just synthetic rows.

The three copies have now drifted in signature. Consolidating them into one module is
worth doing and is blocked only by MIM-NB's change window.

`tools/verify_chain.py` now walks the three `data/thursday_ts/` ledgers as well, even
though that service has been inactive since 2026-07-27 — an unchecked chain is how this
incident stayed invisible for six days.

## 8. Verification

```bash
.venv/bin/python tools/verify_chain.py --reconcile          # scars registered, exit 0
.venv/bin/python tools/verify_chain.py --reconcile --strict # raw truth, exit 1
.venv/bin/python -m pytest tools/test_gap_fade_ledger_repair.py -q
```

Applied 2026-08-13 15:11 UTC, with `trader-gap-fade` stopped and flat (the 2026-08-12
short had closed at the time-stop for +$155.50, and 2026-08-13 was already logged
`NO_SETUP`). Result:

```
[  OK  ] data/gap_fade/trades.csv:    20 rows, chain verifies, complete vs trades.db
         DUPLICATED rows for: 2026-06-25                        [SCAR]
[BROKEN] data/gap_fade/decisions.csv: 36 rows, first mismatch at row 30 (2026-08-07)  [SCAR]
[  OK  ] data/gap_fade/fills.csv:     14 rows, chain verifies
exit 0
```

`INCOMPLETE: ['2026-08-06']` is gone; both breaks that remain are the registered scars,
unchanged in position. On restart the bot loaded the hardened `ChainedCsv` and its
double-entry guard correctly skipped the already-decided session.
