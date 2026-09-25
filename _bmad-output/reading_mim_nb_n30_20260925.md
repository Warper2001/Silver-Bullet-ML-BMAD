# MIM-NB N=30 Reading — per the 2026-09-20 pre-commitment

**Date:** 2026-09-25. **Trade 30 closed:** 2026-09-22; trade 31 closed 2026-09-23.
**Governing document:** `precommitment_mim_nb_n30_decision_20260920.md`. This reading applies it as
written. No parameter, threshold, config, unit file or sizing changes. Read-only evidence:
`_bmad-output/diagnostics_mim_nb_n30_reading_20260925/` (`reading.py`, `results.json`).

## 1. Result

**Branch: PF ≥ 0.70. No halt, and this is not evidence of an edge.** The sealed D1 trigger did not fire at
N=30 or at N=31. Under the any-time semantics of pre-commitment §1.3, evaluation continues with
every new trade.

| D1 counter (`data/mim_nb/trades.csv`) | N | Net | PF |
|---|---|---|---|
| **At trade 30, gross (PF of record)** | 30 | +$116.50 | **1.024** |
| At trade 30, fee-adjusted | 30 | +$79.90 | 1.016 |
| Now (trade 31), gross | 31 | +$270.00 | 1.054 |
| Now (trade 31), fee-adjusted | 31 | +$232.18 | 1.047 |

- **PF path, gross / fee-adjusted:** N=28 0.848/0.842 → N=29 0.996/0.989 → N=30 1.024/1.016 →
  N=31 1.054/1.047. No value at N ≥ 30 is below 0.70.
- **Fee basis:** $1.22 per contract round trip, which is $0.36 fees + $0.25 commission per side. This is
  the cost on every one of the 14 ProjectX fills in `data/mim_nb/projectx_fills.json`, normalised per
  contract. It is an observed broker cost, not the modelled 1-tick slippage.
- **What moved the PF:** trades 29–31 (09-21 → 09-23, all EOD exits) netted **+$1,023.00** and took
  PF from 0.848 to 1.054. Three trades moved the reading from its worst to its best. That is a picture
  of noise at this N, not a signal.

## 2. Why this is not evidence (pre-commitment §2–§3, unchanged)

1. **The rule has little power either way.** It trips 12.9% of the time when the sealed edge is real,
   and misses a −$51/trade strategy 47% of the time. Passing it proves nothing.
2. **PF 1.024 does not track the out-of-sample 1.30.** The upgrade is **not declared**, and
   "passed N=30" must never be cited as validation.
3. **No confirmatory N exists.** A claim that the edge is *confirmed* first needs a power calculation
   on the live 250-pt config's own trade list. That calculation has still not been done.

## 3. Disclosed splits (descriptive only; §5 of the pre-commitment)

| Split | N | Net | PF |
|---|---|---|---|
| D1 at 30 excluding the 07-29 → 08-04 run | 26 | −$2,163.00 | 0.564 |
| The 07-29 → 08-04 run alone | 4 | +$2,279.50 | all wins |
| D1 at 30 excluding the 09-15 AUTOROLL-bug trade (−$355) | 29 | +$471.50 | 1.102 |
| Current account 26556101, MIM-only (from 08-13 12:54 ET) | 13 | −$418.50 | 0.792 |
| Live 250-pt config (since 2026-06-25), all | 28 | +$482.50 | 1.122 |
| D3 monitor window (MIM-NB + YANK, `trades.db`, report-only) | 17 | +$174.50 | 1.082 |

The PF of record still depends on a handful of days. Without the four-trade 07-29 → 08-04 run, the
other 26 trades have PF 0.564, which is below the 0.70 line. That is a fragility disclosure, **not** a
trigger. The pre-commitment names the whole-ledger PF as the number that counts, and this reading
does not substitute a split for it. D3 is a separate tripwire with its own N and is not D1.

Both splits reconcile with the 09-20 diagnostic:
- Current account: 10 / −$1,441.50 at N=28, plus the three new trades (+$1,023) = 13 / −$418.50.
- Excluding the run: 24 / −$3,032.50 at N=28, plus trades 29–30 = 26 / −$2,163.00.

## 4. Capital (pre-commitment §4)

`data/combine_joint/floor_state.json` at 2026-09-25 13:13 UTC:

| | Amount |
|---|---|
| Equity | $49,823.60 |
| MLL floor | $48,298.96 |
| HWM | $50,298.96 |
| Headroom above the $48,400 halt-and-review line | $1,423.60 |
| Buffer to the floor | $1,524.64 |

No capital event. The buffer has grown from $618.46 on 09-20.

## 5. Outside context (not evidence about this bot)

A 2026-09-25 deep recon
(`planning-artifacts/research/technical-open-source-counterparts-to-live-strateg-2026-09-25/research.md`)
found two independent public replications of the published noise-area strategy. Both reproduce the
paper in-sample and then fade after publication:
- Out-of-sample Sharpe 0.39 on SPY over 2024–26 (PazSheimy/spy-intraday-momentum-oos).
- Sharpe about 0 in 2025–26 on both SPY **and** ES futures (giovannibrusco/zarattini-2024-momentum-spy).

In the second replication, re-selecting parameters every quarter did worse out of sample than the
fixed original settings (Sharpe 0.57 against 0.92). No public NQ/MNQ out-of-sample test exists.

This is a prior, not a measurement of MIM-NB. It points the same way as §2: nothing here shows an
edge. It is **not** grounds for a halt the sealed rule did not trigger. It also argues against re-tuning,
consistent with the one-knob policy.

## 6. What this reading does not do

- It does not change any parameter, threshold, unit, config, sizing or the "brake stays OFF" decision.
- It does not claim that MIM-NB's edge is confirmed or refuted.
- It does not reopen GAP-1's separate N=30 reading.

**Next look:** every new completed trade, under the §1.3 semantics. That continues until a derived
confirmatory N exists, or a sealed trigger or capital event fires.
