# Pre-Registration: bidirectional M15 CHoCH (`detect_m15_choch_bullish`)

**Status:** SEALED — no code written against this yet
**Date:** 2026-08-19
**Author:** party-mode round-table (Mary / Winston / Dr. Quinn / Level / Grumbal), at Alex's direction
**Supersedes:** nothing. **Amends:** nothing directly, but see §9 re: `preregistration_yank_sl2tp8_ml050.md`
**Predecessor evidence:** `tools/yank_gate_opening_battery.py`, `_bmad-output/s_bidir_15m_verdict_20260523.md`

---

## 0. The finding

`detect_m15_choch()` (`src/research/strategy_core.py:841-880`) implements one direction only:

```python
return last_close < swing_low - 0.3 * m15_atr     # bearish CHoCH — the only branch that exists
```

There is no bullish counterpart. `BacktestEngine.run()` (`src/research/backtest_engine.py`,
the M15 CHoCH scan block) only invokes it when the active H1 sweep is bearish:

```python
if (
    config.m15_confirmation
    and sweep_cached is not None
    and sweep_cached.direction == Direction.BEARISH      # <- bullish sweeps never reach this block
    and not m15_choch_active
):
    ...
```

**Consequence:** whenever `m15_confirmation=True` — the live YANK setting since S25 — bullish
entries are structurally unreachable, regardless of `bearish_only`. This was discovered
2026-08-19 running a `bearish_only=False` variant (`tools/yank_gate_opening_battery.py`) that
came back byte-identical to the bearish-only baseline (46 trades, same P&L, to the cent) —
the tell that something was silently vetoing the new population rather than evaluating and
rejecting it. Isolating the confound (`m15_confirmation=False` alongside `bearish_only=False`)
produced 35 bullish trades where there had been zero.

This is the same failure class named in `[[feedback_s26_golden_flip_lessons]]` lesson 1:
*"Bidirectional strategies break silent direction gates... Zero error, zero trades, no
warning."* Not a design decision recorded anywhere as deliberate — `bearish_only=True` was
already load-bearing when S25 added the M15 CHoCH requirement, so the one-directional formula
was consistent with the strategy at the time it was written. It became a landmine the moment
anyone tested bidirectional again without noticing the gate was still one-sided underneath.

### 0.1 What was seen, and why it is disclosed but not trusted

Running `bearish_only=False, m15_confirmation=False` on the pre-holdout derivation window
(2025-01-01..2026-02-28) and filtering to the bullish subset:

| | N | Net PnL | PF | WR | Avg win | Avg loss |
|---|---|---|---|---|---|---|
| Bullish, no M15 gate | 35 | +$2,328.75 | 1.286 | 37.1% (13/35) | $805.90 | −$370.36 |

Trades span 13 of the 14 months in the window — not clustered. This directly contradicts
`_bmad-output/s_bidir_15m_verdict_20260523.md`'s bullish-subset result (PF 0.826, Sharpe
−1.602, H₀) — but that test ran on 15-minute bars from the abandoned S12→S13 pivot lineage,
not the 1-minute strategy YANK actually trades, and it is the **only** other bidirectional
data point that has ever existed for this strategy family.

**This number is not evidence of a bullish edge.** It was produced with the M15 gate absent
entirely, not with a working bullish gate — there is no code to compare it against. It is the
directional evidence that justifies writing this pre-registration, per the standing rule
(`[[feedback_s26_golden_flip_lessons]]` lesson 7): explore first, then pre-register, never the
reverse. It must not be cited again once §5 has run.

---

## 1. The honest framing

This is not "restore intended behaviour" — no bidirectional 1-minute version of this strategy
has ever been validated, sealed, or lived. It is not "fix a bug that changes nothing" either —
adding a working bullish gate changes which bullish trades qualify, and by how much is exactly
what §5 exists to measure. Two honest possible outcomes, neither pre-favoured:

| | Response | Meaning |
|---|---|---|
| **A** | Ship the symmetric gate (§4) | Bullish trading becomes structurally possible again, gated by a real (not absent) M15 confirmation — conditional on §5 |
| **B** | Leave `detect_m15_choch` bearish-only | Document that bidirectional trading remains untested pending a real OOS sample; `bearish_only=True` stays the honest description of what's validated |

**§5 decides between them on evidence, not on how encouraging §0.1 looked.**

---

## 2. Hypothesis

**H₁:** a symmetric M15 CHoCH gate (bullish mirror of the existing bearish formula, same
constants, no new ones) produces a bullish trade population that clears the same bar the
2026-05-23 Epic 2 battery used to judge every other gate-opening variant (PF > 1.3, N ≥ 15) —
i.e., today's 1-minute strategy has a real bullish signal that the 15-minute lineage did not.

**H₀:** the gate stays symmetric-but-absent-of-edge — either N < 15 (the real gate is more
restrictive than "absent" and starves the population), or PF ≤ 1.3, in which case §0.1's
number was an artifact of having no filter at all, not a preview of what a working one finds.

H₁ is falsifiable and §5 is its only test. **A pass is not evidence the strategy should be
deployed bidirectionally** — it authorizes moving from "no bullish gate exists" to "a real one
does, and it clears a bar," nothing about the combine or live capital.

---

## 3. The change has no derived constant to sweep

Per the standing rule (*derive, don't assert* — `[[feedback_derive_dont_assert_one_knob]]*),
new thresholds must be derived, not hand-set. This change introduces **zero new constants.**
`detect_m15_choch_bullish` is a structural mirror of the existing, already-live bearish
formula — same `SWING_R=2`, same `period=min(20, n-1)` ATR window, same `0.3` multiplier —
substituting swing-high/close-above for swing-low/close-below. Nothing is tuned, swept, or
optimized. If §5 fails, the answer is Response B, not a search for a better multiplier.

---

## 4. Change spec (one structural fix, backtest scope only)

Conditional on §5 PASS and an explicit go from Alex. **Scope: `BacktestEngine` only.** The
live bot (`yank_streaming_working.py::_update_m15_choch`) is explicitly OUT of scope for this
seal — see §7. A pass here authorizes testing the live wiring in its own follow-on seal, not
deploying it.

1. `strategy_core.py` gains `detect_m15_choch_bullish(m15_completed: pd.DataFrame) -> bool`,
   adjacent to `detect_m15_choch`, mirroring its body exactly: swing-*high* scan (`highs[i+k]
   <= hi` in place of `lows[i+k] >= lo`), return `last_close > swing_high + 0.3 * m15_atr`.
2. `backtest_engine.py`'s M15 scan block gates on `sweep_cached.direction in (Direction.BEARISH,
   Direction.BULLISH)` instead of `== Direction.BEARISH`, and dispatches to
   `detect_m15_choch_bullish` when the active sweep is bullish, `detect_m15_choch` when bearish.
3. Nothing else. Not `SWING_R`, not the `0.3` multiplier, not `min_gap_atr_ratio`, not
   `max_gap_atr_ratio`, not any Epic 2 variant (KZ/VOL) — those stay separate questions.

---

## 5. Acceptance gate (binary, falsifiable, pre-committed)

Run on the pre-holdout derivation window only (2025-01-01..2026-02-28, same hard bound as the
gap-ceiling gate — asserted programmatically, `data/sealed_holdout/` never read), config
`bearish_only=False, m15_confirmation=True` (the real bidirectional-with-a-working-gate run —
**not** the M15-off exploratory config from §0.1), everything else at today's live values
including `max_gap_atr_ratio=0.426`.

**Bar reused from `_bmad-output/s_vol_15m_verdict_20260523.md` / the Epic 2 battery — chosen
before this document existed, not fit to §0.1's number:**

| | Criterion | Bar |
|---|---|---|
| **G1** | Bullish-subset trade count | **N ≥ 15** |
| **G2** | Bullish-subset PF | **> 1.3** |
| **G3** | Bearish-subset PF, same run, vs. today's bearish-only baseline (46 trades, PF 1.053) | **within 10%** — the new gate must not silently change bearish behaviour |
| **G4** | No H1-ATR-only structural artifact: bullish trades are not all concentrated in a single calendar month | **max 1 month ≤ 40% of N** |

**All four must pass.** A G1/G2 failure means H₀: the real gate (unlike its absence) does not
clear the bar Epic 2 already set for every other gate-opening variant — Response B. A G3
failure means the bidirectional wiring altered bearish behaviour, which §4 explicitly forbids —
implementation defect, not a strategy verdict, fix and re-run before judging H₁ at all.

---

## 6. What a PASS costs

If §5 passes and Response A ships (its own future go, not this seal):

1. This is new-strategy territory, same as the gap-ceiling change. **No prior bidirectional
   evaluation clock exists to reset** — there has never been a validated bidirectional sample,
   only the untrustworthy §0.1 look and the contradicted 15-minute precedent. A pass starts
   count from zero, honestly, rather than pretending a clock existed.
2. The sealed 2026 holdout is **not** touched by §5 and remains available for whatever future
   confirmatory test this strategy eventually earns — but reading it now, before H₁ clears its
   own in-sample bar, would be the same mistake as reading it for the gap ceiling.
3. Gate-Minus-One applies to any live deployment, exactly as it did for the gap-ceiling fix.

---

## 7. What this pre-registration does NOT authorize

- Wiring the bullish gate into the live bot (`yank_streaming_working.py` /
  `tier2_streaming_working.py`). §4 is `BacktestEngine`-only. Live wiring needs its own seal
  after §5 passes.
- Deploying anything to the combine account.
- Touching `bearish_only`, `m15_confirmation`, `max_gap_atr_ratio`, `min_gap_atr_ratio`, or any
  other sealed live value.
- Combining this test with any Epic 2 variant (KZ, VOL) in the same run — §5 tests this change
  alone.
- Tuning `SWING_R` or the `0.3` ATR multiplier for either direction (§3).
- Reading `data/sealed_holdout/` for any purpose in this work.
- Re-citing §0.1's PF 1.286 / N=35 as if it were §5's result.

---

## 8. Disclosures — a priori vs. found-while-exploring

| Item | Status |
|---|---|
| The one-directional M15 CHoCH gate | **Found 2026-08-19**, via a null BIDIR result during an unrelated gate-opening exploration Alex asked for |
| §0.1's PF 1.286 / N=35 exploratory number | **Found 2026-08-19**, with the M15 gate absent, not working — the directional evidence that justified writing this seal, per lesson 7 |
| Derivation window = 2025-01-01..2026-02-28 | A priori — inherited unchanged from the gap-ceiling seal's own derivation window |
| G1/G2 bar (N≥15, PF>1.3) | A priori — sourced from the 2026-05-23 Epic 2 battery, set before this document or §0.1's number existed |
| G3/G4 (bearish-invariance, no-single-month-concentration) | A priori — added by the room specifically because §0.1 could not check for these (M15 gate was absent, not real) |
| `detect_m15_choch_bullish`'s formula | A priori — mechanical mirror, zero new constants (§3) |

**Known asymmetry, disclosed:** the author saw the confounded exploratory number before
writing this seal. §5's own run has not been executed by anyone as of this seal.

---

## 9. Related open items (not in scope, recorded so they are not lost)

1. If §5 passes, `preregistration_yank_sl2tp8_ml050.md` (2026-06-15, the current live seal)
   will need its own amendment before any bidirectional deployment — it was sealed against a
   bearish-only population throughout.
2. The Epic 2 battery's other three variants (KZ, VOL) remain H₀ against today's config
   (`tools/yank_gate_opening_battery.py`, 2026-08-19) — KZ net −$3,590.50/PF 0.863, VOL net
   +$1,491.50/PF 1.071 (below the 1.3 bar). Not reopened by this seal.
3. `BacktestEngine`'s `m15_confirmation` tz bug (fixed 2026-08-19, PR #47, not yet merged) is
   a dependency of this work — §5 cannot run until PR #47 lands.

---

## 10. Integrity hashes (at seal)

```
git HEAD                     fb4c96f88b106694272a74ebca6fc5cb8d1d1041  (branch fix/backtest-engine-m15-tz-and-gap-ceiling-backtest, PR #47, unmerged)
strategy_core.py     SHA-256 3e8350f8b20ca477b68b3e9b436faa4c19441142220bf5512b9a2b9842d25cc2
backtest_engine.py   SHA-256 dc4f52c2004b3404a653af2f9e66fad74b83b982e68f97c7723757dd2e2ea757
strategy_config.yaml SHA-256 293c9d23c69f667564954e30bbb6360ca8af658288978ffa62de1b3c21391abb
```

Live YANK config at seal: `SL2.0x_TP8.0x_Midpoint_H1_M15CHoCH_M1FVG_g0.25/0.426`,
`ml_threshold=0.50`, `bearish_only=true`, `m15_confirmation=true`, 2 contracts, MNQU26.
`detect_m15_choch_bullish` does not exist in the codebase as of this seal.

**Commit this document before writing a line of code against it.**

---

## Amendment 1 — §5 executed, H1 confirmed (2026-08-19)

**Implemented exactly as specified in §4**, no deviation: `detect_m15_choch_bullish`
added to `strategy_core.py` as a pure structural mirror (same `SWING_R=2`, same ATR
window, same `0.3` multiplier — zero new constants), `BacktestEngine`'s M15 scan block
now dispatches by sweep direction instead of hardcoding bearish. Bearish-only behaviour
is unit-tested to be byte-identical to before (`tests/unit/test_m15_choch_bidirectional.py`,
7 tests). Scope held to `BacktestEngine` only, per §4/§7 — no live-bot file touched.

**§5 run** (`tools/yank_bidir_m15_choch_g5_gate.py`), config `bearish_only=False,
m15_confirmation=True` — the real gate, not §0.1's M15-absent exploratory config — on
the same derivation window:

| Gate | Result | Bar | Verdict |
|---|---|---|---|
| G1 (bullish N) | 23 | >= 15 | PASS |
| G2 (bullish PF) | 1.430 (net +$2,566.75) | > 1.3 | PASS |
| G3 (bearish PF vs. baseline) | 1.088 vs 1.053, diff 3.32% | <= 10% | PASS |
| G4 (no month > 40% of bullish N) | worst month 2026-01, 17.4% | <= 40% | PASS |

**All four pass. H1 confirmed.** Notably, the real working gate (N=23, PF 1.430) beats
the confounded §0.1 look (N=35, PF 1.286, M15 gate absent) on PF — filtering with an
actual bullish CHoCH check produces a *smaller, higher-quality* population, not just
fewer trades. Bullish trades span 12 of 14 months (worst concentration 17.4%), and the
bearish side moved only 3.32% off its pre-change baseline — the wiring did not disturb
existing behaviour, addressing the one failure mode §5 existed specifically to catch.

**What this does and does not authorize, restated from §6/§7:** this is new-strategy
territory with no prior evaluation clock to reset — a pass moves `detect_m15_choch`
from "bearish-only, one direction structurally unreachable" to "bidirectional, both
tested" **inside `BacktestEngine` only**. It does **not** authorize wiring this into
`yank_streaming_working.py`, does **not** authorize setting `bearish_only=False` on the
live bot, and does **not** authorize any combine deployment. Those each need their own
future seal, per §7, unchanged by this pass.

**Tests:** `tests/unit/test_m15_choch_bidirectional.py` (new, 7 tests: pure-function
symmetry checks + a `BacktestEngine`-level wiring guard confirming `bearish_only=True`
still produces bearish-only output). Full detect_fvg/config-override/backtest-engine/m15
sweep: 245 passed, 5 pre-existing unrelated failures in `src/detection/` (confirmed
untouched by this diff, same as the gap-ceiling and ml_proba changes this session).

**Not deployed. Not merged to main as of this amendment** — code lives on branch
`fix/backtest-engine-m15-tz-and-gap-ceiling-backtest` (PR #47).
