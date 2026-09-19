# MIM-NB hash chains: two pre-existing breaks, found 2026-09-19

Found by registering `data/mim_nb/*` in `tools/verify_chain.py`'s `DEFAULT_FILES`. MIM-NB
has written hash-chained records since it went live on 2026-06-11, and **no default run
had ever walked them** — the same blind spot that let the 2026-08-06 gap-fade damage sit
unnoticed for six days. Both breaks predate the registration; nothing in this session
wrote to these files.

Current state: `trades.csv` (28 rows) and `sessions.csv` verify clean. `decisions.csv`
and `orders.csv` are BROKEN.

## Break 1 — `data/mim_nb/decisions.csv`, row 128 (2026-06-29T10:00:00-04:00)

**Signature: rows removed after they were written.**

| Session | Rows recorded | Expected |
|---|---|---|
| 2026-06-24 | 12 | 13 |
| 2026-06-25 | **5** | 13 |
| 2026-06-26 | **0 — absent entirely** | 13 |
| 2026-06-29 | 13 | 13 |

The last surviving row before the break is `2026-06-25T12:00`, and the first row that
fails verification is the next one present, `2026-06-29T10:00`. A gap alone does not
break a chain — each row hashes its predecessor *as written*. A row that chains onto a
head no longer in the file means the intervening rows existed and were later removed.

This is the signature of the documented 2026-08-06 gap-fade incident: a checkout or
restore reverted the file while the bot held a stale in-memory head. 2026-06-25 was a
deploy day (the Thursday-short TradeStation SIM pivot).

## Break 2 — `data/mim_nb/orders.csv`, row 79 (2026-07-29T19:30:04)

**Signature: a row appended by hand, outside the writer.**

The row immediately before the break is:

```
2026-07-29T18:42:35.181643+00:00,FILL,111,4,1,1,29748.25,OK,fees=0.36 pnl=-503.5
```

`order_id=111` is not a broker id — every neighbouring row carries a ten-digit ProjectX
id (e.g. `3337022957`). A reconciliation entry was appended without going through
`ChainedCsv`, so it carries no valid chain value, and every row after it chains onto a
head the bot never produced. 2026-07-29 is the day floor gating was removed.

## What must not be done

Re-chaining either file would make the verifier green by destroying the evidence the
chain exists to preserve. `verify_chain.py`'s own doctrine: damage that already happened
cannot be undone without rewriting an append-only file.

## Decision (Alex, 2026-09-19): LEAVE THEM LOUD

No scar is registered for either break. The ops healthcheck will WARN on every run, and
`verify_chain.py` exits non-zero, until they are fixed or deliberately accepted later.
That is the intended state: the causes below are inferred from the files, not confirmed by
any contemporaneous note, and a scar is a claim the damage is understood and accepted.

Expect a standing `bot ledgers:` WARN from `combine-ops-healthcheck`. It is this, not a
new problem. Re-read this note before treating it as noise.

## Options that were weighed

1. **Register both as scars** (`SCARS` in `verify_chain.py`), each citing this note. The
   breaks still print, tagged `[SCAR]`, but stop failing the exit code — which is what
   the ops healthcheck reads. Registering a scar is a claim the damage is understood and
   accepted.
2. **Leave them BROKEN.** The healthcheck WARNs on every run until they are fixed or
   registered. Loud, and impossible to forget.

Option 1 is only honest once the causes above are confirmed rather than inferred. What is
written here is inferred from the files themselves; neither incident was investigated at
the time, and no contemporaneous note exists for either date.

## What this does not affect

- **No trade evidence is lost.** `trades.csv` verifies clean, as does the ledger in
  `data/trades.db`. The broken files are the decision log and the order log.
- Both breaks are more than seven weeks old and entirely predate the 2026-09-15 roll fix.
