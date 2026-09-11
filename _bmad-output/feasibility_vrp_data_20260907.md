# VRP (Volatility Risk Premium) — DATA FEASIBILITY, 2026-09-07

**Status change: Option 5 moves from DATA-BLOCKED → TESTABLE.**

`project_post_r3_options_pass_20260906` closed Option 5 as *"DATA-BLOCKED, not failed — repo has no
options chains, no VIX/VX, no IV series. Needs NQ/QQQ options w/ IV **or VIX+VX term structure**."*
This probe tested both paths against the live TradeStation API using the same auth path the live
bots use (`TradeStationAuthV3.from_file(".access_token")`). Read-only; no orders; no token printed.

---

## 1. Result

| path | verdict |
|---|---|
| **Options chains + IV** | ❌ **BLOCKED** — `/v3/marketdata/options/expirations/{QQQ,SPY}` returns **HTTP 403 "Missing required scope"**. Historical option barcharts return 200 with **0 bars**. Unblocking requires re-authorizing the OAuth app with an options scope (and possibly an account options entitlement) — an interactive login only Alex can perform. |
| **VIX + VX term structure** | ✅ **FULLY OPEN** — see below. |

## 2. What is actually reachable (verified)

**Daily history — ~20 years, far deeper than anything this shop has tested:**

| symbol | bars | range |
|---|---|---|
| `$VIX.X` (S&P 500 volatility index) | 5,000 | 2006-10-25 → 2026-09-04 |
| `$VXN.X` (**Nasdaq-100** volatility index) | 5,000 | 2006-10-10 → 2026-09-03 |
| `$VVIX.X` (vol-of-vol) | 500+ | 2024-09-06 → 2026-09-03 |
| `@VX` (continuous VX future) | 5,001 | 2006-11-15 → 2026-09-07 |
| `$SPX.X`, `QQQ` (underlyings) | 5,000 each | 2006-10 → 2026-09 |

**Full term structure — 9 live VX contracts**, front through 9 months out: VXU26 (178 bars),
VXV26 (156), VXX26 (137), VXZ26 (117), VXF27 (98), VXG27 (73), VXH27 (55), VXJ27 (31), VXK27 (11).
A term-structure/roll-yield signal needs 3–4; there are 9.

**A correctly-sized instrument exists.** Full VX is $1,000/point — PL-sized risk, the exact
geometry that killed platinum. **Mini-VIX (VXM) is $100/point:** `VXMU26` (117 bars), `VXMV26`
(98), `@VXM` (401+). Symbol detail confirms `VXMU26` = *Mini-VIX Futures Sep 2026*, exchange
**CBOEF**, expiry 2026-09-16.

**Intraday exists** if a same-day signal is ever wanted: 1-min bars return for `$VIX.X` and `@VX`.

**Live quotes resolve** (`/v3/marketdata/quotes`): at probe time `$VIX.X` 15.30, `VXU26` 16.30,
`VXV26` 18.06, `$VXN.X` 20.04 — i.e. **spot < front < second month, textbook contango**, which is
the state the premium is harvested from. (Bid/ask returned 0 — outside CFE hours — so **no spread
was measured**; see the gates below.)

## 3. Why this is worth a seal — and the two reasons to be careful

**For it.** ~20 years of daily data is a genuine break from this shop's constraint. Every recent
negative verdict ran on ~1 year (fan-out), 5.4 years (calendar seasonality) or was closed
**UNDERPOWERED** outright (XSMOM-1, 10.6% power). A VRP study on 5,000 daily observations is the
first candidate here that can clear a power gate before spending anything. It is also genuinely
orthogonal to every live edge (all MNQ/BTC directional).

**Against it — record these before any seal.**

1. **Short vol is negative-skew: it is the opposite pathology to everything else in this book.**
   The fan-out re-run just established that all eight instruments' edges are *tail-carried* (they
   need their best 3 days). VRP instead collects small and loses big — it is *tail-exposed*. On a
   large-buffer SIM account that is survivable and the risk is legible; on a trailing-MLL combine
   it is the PL problem again in a new costume. **Any seal must state the vehicle up front** and
   must not reuse the fat-day (`ex-top-3-days > 0`) robustness clause unexamined — for a
   negative-skew strategy the meaningful robustness test is the *worst* days, not the best.
2. **This is one of the most published, most crowded trades in finance.** A high prior that the
   effect is real is *not* evidence it survives retail costs at 1 Mini-VIX contract. It gets the
   same Gate 0 as everything else, and the "well-known effect" status earns it no shortcut.

## 4. Gates that must be cleared before anything trades

1. **Power gate first** (per `project_xsmom1_power_gate_20260907`) — on 5,000 daily bars this
   should pass easily, but run it before spending data, not after.
2. **Slippage measurement on VXM, prospectively** — the HG/PL method exactly. No spread was
   measurable at probe time; a VXM cost card does not exist. **Do not assume a cost.** Note the
   lesson already paid for twice: detect contract rolls by quote-staleness/spread-sanity, never
   by sample count.
3. **Trade-permission check.** Data entitlement ≠ trade entitlement — the options 403 proves
   scopes bite here. Confirm CFE/VXM is tradeable on the TS SIM account **before** designing an
   execution path.
4. **Basis caveat if the MNQ link is used.** `$VXN.X` (Nasdaq vol) is the natural signal for a
   Nasdaq-focused shop, but VX/VXM futures track **VIX (S&P)**, not VXN. Signal and instrument
   would be on different underlyings — a real basis risk that must be pre-registered explicitly,
   not discovered later.

## 5. Recommendation

**Pre-register a VRP study on the VIX/VX term structure, VXM as the instrument, power gate first,
slippage measured second.** It is the only remaining candidate on the board that is both
data-rich and orthogonal, and it is now unblocked at zero data cost.

**Do not pursue the options-chain path** unless Alex wants to re-authorize with an options scope —
it is blocked, and the VIX/VX path answers the same question without it.

*Probe scripts: `tools/probe_vrp_feasibility.py`. Read-only; no orders placed; no credentials
printed or copied.*
