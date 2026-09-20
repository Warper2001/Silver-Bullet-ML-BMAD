# MIM-NB: can the sealed N=30 halt rule fire, and does the capital outlast it? (2026-09-20)

Script `analyze.py`, raw output `results.json` (B=200,000, seed 20260920, deterministic on rerun).
Read-only: no config, unit file, bot, or `data/sealed_holdout/` was touched. No edge verdict is made.

## 0. Correction of record — the 09-17 diagnostic used the wrong ledger window

`diagnostics_mim_nb_live_variance_20260917` filtered `trades.db` to `write_mode='realtime'`, which
silently drops MIM-NB's first two live trades (06-11 is tagged `backfilled`; 06-12 is absent from the
DB). The authoritative hash-chained ledger is `data/mim_nb/trades.csv`.

| | 09-17 diagnostic | `trades.csv` (authoritative) |
|---|---|---|
| N | 26 | **28** |
| Net | −$1,540.50 | **−$753.00** |
| PF | 0.689 — "already below the 0.70 halt line" | **0.848 — above it** |
| Trades to N=30 | 4 | **2** |

The 09-17 statement "at the current live PF it would already fire if N=30 landed today" is
**withdrawn**. The AGENTS.md advice to filter `write_mode='realtime'` is right for *trader-wide* P&L
audits but wrong for a per-strategy counter that starts at deployment.

Sanity gate: this script's `reproduction_of_0917` block re-runs the 09-17 bootstrap on its own
(mis-windowed) N=26 and gets 3.51% / 5.47% vs the published 3.5% / 5.5% — so the corrected figures
below are comparable to it. On the correct window the tail is milder: **P(net ≤ observed) 8.4%,
P(PF ≤ observed) 10.1%** vs the sealed reference.

## 1. Three different "N=30" rules exist — they are not the same test

| | Definition | N now | PF now | Halts anything? |
|---|---|---|---|---|
| **D1** | Sealed deployment prereg §4: "30 completed trades with net PF < 0.70" — the strategy's own trades since 2026-06-11 (`trades.csv`) | 28 | 0.848 | Text says halt-and-review; **no automation behind it** |
| D2 | 09-17 window (`trades.db` realtime) | 26 | 0.689 | — (mis-windowed D1; superseded) |
| **D3** | `combine_floor_monitor.py`: MIM-NB **+ YANK** rows in `trades.db` since the *current* account's start (2026-08-13T16:54), string-compared timestamp, no `write_mode` filter | 12 | **0.447** | **Report-only** (Amendment 2, 2026-08-04) |

D3's window has no month-format defect today (string vs parsed compare: 0 row difference).
Status of the sealed triggers: halting authority was removed 2026-07-29 and the monitor restored
report-only 2026-08-04 (`preregistration_mim_nb_risk_mechanics_removal.md` §7); deployment §4's
`$48,400` equity trigger and `PF<0.70 @ 30` text were never named or repealed by it.

## 2. D1's PF rests on four trades

| D1 slice (descriptive — not a threshold, not a selection rule) | N | Net | PF |
|---|---|---|---|
| All | 28 | −$753.00 | 0.848 |
| 07-29 → 08-04 run alone | 4 | +$2,279.50 | ∞ |
| Excluding that run | 24 | −$3,032.50 | 0.388 |
| Retired acct 23884932 rows (day < 08-13) | 17 | +$681.00 | 1.232 |
| Current acct 26556101, MIM only | 10 | −$1,441.50 | 0.284 |

This is the seal's own disclosed structure (§2b: edge concentrated in ~1–3 fat-tail days), not new
evidence of failure — but it means D1's PF is one good run away from a completely different reading,
and it is why the counting rule must be fixed *before* trade 30. The 07-06→08-13 rows were real fills
on an account past its MLL but with `canTrade: true` (memory `project-combine-blown-20260706`): valid
strategy evidence, worthless for funding.

## 3. The sealed D1 rule is almost unable to fire at N=30

From N=28 (gross win $4,203.00 / gross loss $4,956.00) PF<0.70 at N=30 needs the next two trades to
lose **> $1,048.29 combined (> $524 each if both lose)**. The live cat-stop caps a stop-out at $500
(`CAT_STOP_PTS=250`, `mim_nb_live.py:52`). **Two full cat-stops give PF 0.7057 — still above 0.70.**
Only an EOD-held loser > $524 or slippage through the stop can do it.

| P(D1 halt condition true…) | R1 sealed 500-pt ref | R2 truncated −$500 (optimistic) | R3 live 250-pt config (N=25) |
|---|---|---|---|
| **exactly at N=30** | 0.4% | 0.0% | 0.0% |
| at *any* N in 30…60 (monitor semantics) | 13.1% | 11.0% | 48.4% |
| D3 at exactly N=30 (18 more combined trades) | 18.9% | 17.5% | 52.9% |
| D3 at any N in 30…60 | 26.4% | 24.2% | 72.2% |

So **N=30 will pass silently on ~2026-09-26 (5–95%: 09-21 → 10-07) with probability ≈ 99.6%**
whatever the strategy is. D3 reaches N=30 around 2026-11-24 (12-23 at the 95th pct) at the trailing
MIM rate of 1.875/wk (MIM-only rate; YANK adds ~0.4/wk, so D3 is slightly earlier).

## 4. Is PF<0.70 @ N=30 a real test? (fresh 30-trade sample, R1 shifted to the stated true mean)

| True net mean per trade | P(halt condition true) |
|---|---|
| +$31.99 (the sealed honest-expectations net edge) | **12.9%** — false-halt rate |
| $0 | 25.6% |
| −$31.99 | 42.1% |
| −$51.00 | **52.8%** — misses a losing strategy about half the time |

The rule's own rationale ("≈ below the MC's 5th-percentile path") does not match: a sealed-edge
strategy trips it 12.9% of the time, and a strategy losing $51/trade escapes it 47%. Same lesson as
GAP-1's N=30 power gate — an N=30 rule is a weak discriminator either direction. Caveat: R1 is the
500-pt variant, gross; the 250-pt prereg states a fresh benchmark was never built.

## 5. The capital race — the account probably decides before the rule can speak

Equity $48,917.42; floor $48,298.96 (buffer **$618.46**); sealed §4 halt-and-review equity line
$48,400 (distance **$517.42**). One cat-stop leaves $48,417.42 — $17.42 above that line. Bootstrap of
MIM-NB trades only, i.i.d., from the current equity:

| Within next … trades | Touch $48,400 (R1 / R2 / R3) | Touch MLL floor (R1 / R2 / R3) |
|---|---|---|
| 2 (≈ N=30) | 10.1% / 7.8% / 17.9% | 6.3% / 4.3% / 13.7% |
| 5 (≈ 2.7 wks) | 23.4% / 21.1% / 44.3% | 18.1% / 15.8% / 38.5% |
| 10 (≈ 5.3 wks) | 33.0% / 30.4% / 61.3% | 27.5% / 24.9% / 56.4% |

Not modelled (all disclosed): YANK's 2ct draw on the same buffer (2 trades in 5 weeks); the DLL guard
(only binds a second same-day loss); intraday vs EOD floor timing; the healthcheck's separate
$500 distance trigger (currently WARNing "approaching"; $118.46 above the line).

## 6. What this does and does not establish

- **Does:** D1 was mis-stated by 2 trades and 0.16 PF; the sealed halt cannot practically fire at
  N=30; the rule is a weak test even fresh; the equity trigger and the MLL floor are the near-term
  live risks, at roughly 10–18% within two trades and 25–56% within ten depending on the assumption.
- **Does not:** say MIM-NB has or lacks an edge (D1 = 28 trades, 4 of which carry the whole net);
  benchmark the live 250-pt config (R1 is the wrong variant, R2 is optimistic, R3 is 25 trades of
  the thing being tested); reopen the 2026-09-06 "brake stays OFF" decision.
- **Sizing option does not exist:** 1 MNQ is the smallest contract; the only levers are on or off.
