# Decision Note: Tune / Train / OOS Sequencing for YANK Cross-Instrument Candidates

**Date:** 2026-06-26
**Author:** Alex
**Scope:** HG (copper) and PL (platinum) — the two orthogonal, cost-pending
candidates from the YANK cross-instrument portability sweep. Applies by extension
to any future instrument the frozen YANK engine is fanned out to.
**Status:** Decision recorded BEFORE the sealed-holdout (OOS) step, so the
sequencing is on the record and cannot be rationalized after seeing holdout data.
**Related:** [[project_yank_cross_instrument_copper]],
`precommit_hg_slippage_measurement_2026-06.md`,
`precommit_pl_slippage_measurement_2026-06.md`.

## 1. What has and has NOT been done (state of evidence)

| Step | Done? | Detail |
|---|---|---|
| Per-instrument **tuning** | ❌ No | Frozen MNQ-derived config run unchanged (SL/TP mult, gap-ATR ratio, sweep lookback, CHoCH params, entry %). Zero per-instrument optimization. |
| Per-instrument **ML training** | ❌ No | Ran ML-off (`ml_threshold=0.0`). MNQ meta-labeler not used (won't transfer); nothing new trained. |
| **OOS** (sealed holdout) | ❌ Not yet | Backtested only the exploratory window 2025-05-19→2026-02-28 (in-sample: candidates were *selected* on it). Sealed holdouts (2026-03-01+) untouched. |

**Nuance (a real strength):** the parameters were tuned on MNQ and *never fit* to
copper/platinum, so the exploratory result is a **pseudo-OOS for the parameters**
(they never saw these instruments). It is NOT true OOS, because instrument
*selection* used this window across 7 instruments (selection bias / multiple
comparisons). True confirmation still requires the sealed slice.

## 2. Decision

1. **Do NOT tune per-instrument parameters before OOS.**
2. **Do NOT train per-instrument ML before OOS.**
3. **DO send the FROZEN, untuned, ML-off config to the sealed holdout**, gated
   behind (a) the slippage measurement passing and (b) a written Gate-1
   pre-registration with a frozen decision rule.

## 3. Rationale

### Why NOT tune (the load-bearing call)
- The **entire value** of this finding is that a frozen, untuned structure
  transferred to copper and platinum. That is evidence of a *real* structural
  edge. Tuning SL/TP/gap on the in-sample window destroys that evidence — we would
  no longer know whether the instrument "works" or whether we fit it.
- This is the program's documented failure mode: the wider-SL grid "2025 mirage,"
  and the iteration-loop / restrict-to-favorable-subset pattern logged three times.
- A **tuned** copper/platinum edge is worth LESS than the untuned one already in
  hand. Tuning trades away credibility for in-sample cosmetics.
- Multiple-comparisons inflation: tuning = trying many parameter sets = near-certain
  to surface something that looks good by chance.

### Why NOT train ML now
- ~95–101 trades is far too few to train a meta-labeler without overfitting (MNQ's
  model used far more history). On this sample, "training" ≈ memorizing noise.
- Premature: the *base* (no-ML) edge has not been OOS-confirmed. Adding ML before
  the base edge survives is building on sand (Winston's "walk first").

### Why OOS with the FROZEN config specifically
- OOS-validating an **untuned structure** is far stronger evidence than
  OOS-validating something we fit. The frozen config is the clean hypothesis.
- Each instrument has exactly **one** sealed holdout slice — non-renewable. It must
  be spent on the clean frozen hypothesis, never on a tuned/trained variant, or the
  one confirmatory shot is contaminated before it is fired.

## 4. Gated sequence (the only sanctioned path to OOS)

```
[done]    Exploratory frozen/ML-off sweep → ranked hypothesis (HG, PL)
[running] Real slippage measurement (capture_{hg,pl}_quotes.py)
            │  PASS (measured cost ≤ sealed ceiling)        FAIL → reclassify
            ▼                                                       gross-only, STOP
[next]    Write Gate-1 pre-registration (frozen config + decision rule + measured
          cost basis); commit BEFORE holdout access
            ▼
[then]    Run the FROZEN config ONCE on the sealed holdout slice (per-instrument)
            │  PASS                                          FAIL → edge did not
            ▼                                                       survive OOS, STOP
[gate]    Combine-fit check (notional vs 50K acct, per-trade SL $ risk) + Alex's
          deployment decision  (separate gates; nothing auto-deploys)
```

## 5. When tuning / training become permissible (future options, not now)

Only **after** an untuned, ML-off edge has **survived its sealed-holdout OOS test**,
and then only under fresh pre-registration:
- **Tuning:** a single pre-registered knob at a time (derive-don't-assert: sweep to
  find the value, seal before any further holdout/forward access). Never a grid on
  data already used for confirmation.
- **ML:** only once enough *forward/live* trades have accumulated to train without
  overfitting; the meta-labeler is itself pre-registered and validated on data it
  did not train on.

## 6. One-line summary

The restraint *is* the strength: not tuning or training is precisely what makes the
portability evidence credible. The next step is not more fitting — it is letting the
slippage measurement resolve, then spending the one-shot holdout on the frozen edge
exactly as-is.
