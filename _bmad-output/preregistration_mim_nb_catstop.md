# Pre-Registration: MIM-Noise-Bands V2 + Catastrophe Stop

**Generated:** 2026-06-11
**Experiment ID:** mim-nb-v2-catstop
**Pre-registration base commit:** 35aea36 (seal commit SHA recorded by the commit itself)
**Parent:** mim-noise-bands-mnq (50f111b) — V2 passed Gate 0 (dev net PF 1.484) and Gate 1 OOS (net PF 1.299) but failed Gate 2 combine MC at 48.6% pass vs the 50% gate, driven by single-trade tail risk
**Status:** SEALED — no catastrophe-stop variant has been computed on any data.

**Data files:** identical to parent seal (dev md5 `3ba83a32...`, OOS md5 `4ec175df...`).

---

## 1. Integrity Disclosure (read before the spec)

**This is a structural change motivated by an observed failure mode — the overfit risk is real and is mitigated as follows.**

What was observed before sealing:
- V2 pooled trade distribution including: worst dev trade **-977.75 pts** (a reversal day), worst OOS trade **-330.25 pts**; Gate 2 MC pass 48.6% / blow 41.1% at 1ct.
- Therefore any stop level chosen in the open interval (330, 978) would surgically remove exactly one known trade while leaving all others untouched — that would be curve-fitting wearing a risk-management hat.

Mitigation (binding):
- Stop levels are derived from **combine constants only**: the $1,000 Daily Loss Limit and the $2,000 MLL buffer at 1 contract ($2/pt). They are NOT derived from the trade distribution.
- **S-A = 250 pts** (= $500/ct = ½ DLL = ¼ MLL buffer). This level is BELOW the observed OOS worst (-330), so it actively modifies many trades in both windows — it cannot merely delete the known outlier.
- **S-B = 500 pts** (= $1,000/ct = full DLL). ⚠️ Disclosed: this level lies between the OOS worst (330) and the dev worst (978) and therefore in-sample it will primarily truncate the single known outlier. Its evidential value is weaker than S-A's; it is included because "one trade may not breach the DLL at intended size" is the canonical combine-derived rule, and it must survive the same gates.
- No other levels may be tested. If neither passes, this branch closes (no level search).

**OOS status disclosure:** the 2026 window was opened once for the parent V2 spec (Gate 1). It is no longer pristine for this family. Consequently this experiment's decision gate is the combine MC on pooled trades; the per-window re-runs are reported as sanity checks, not as fresh OOS validation.

## 2. Hypothesis

**H₁:** V2 with a combine-derived catastrophe stop caps single-trade tail loss enough to lift Topstep 50K MC pass% to ≥ 50% (with pass > blow) at some size 1–10, without destroying the validated edge (pooled net PF stays ≥ 1.10).

**H₀:** the stop either truncates enough winners' adverse excursions to kill the edge, or the tail was not the binding constraint. Both S-variants failing → branch closed; remaining options revert to the research-doc order (Candidate 3 HTF-MR) or a deployment prereg for plain V2 paper validation.

## 3. Spec (Frozen)

Identical to parent V2 in every respect (noise bands σ(t) 14-day, gap adjustments, HH:00/HH:30 entry+reversal checks 10:00–15:30, wide band-stop checks 10:00–16:00, fills next 1-min bar open, EOD exit at 16:00 close, $2.24/ct cost per completed trade) **plus one addition**:

| Element | Rule |
|---|---|
| Catastrophe stop | resting stop order at entry − S (long) / entry + S (short), S ∈ {250 (S-A), 500 (S-B)} points |
| Monitoring | every 1-min bar: long exits if bar low ≤ stop, short if bar high ≥ stop; fill AT the stop level (slippage already in cost model) |
| Same-bar rule | the stop is live from the entry bar itself (entry at open; that bar's low/high can trigger it) |
| After stop-out | flat; re-entry permitted at any subsequent HH:00/HH:30 entry check per normal band logic |
| Reversal stops | a reversal closes the old trade at the check fill as before; the new leg gets a fresh stop at its own entry ∓ S |

## 4. Gates (Frozen)

### Gate A — edge survival (both windows re-run, sanity)
- Pooled (dev+OOS) net PF ≥ **1.10** and pooled net expectancy > $0
- Each window separately: net PF ≥ 1.00 (the stop must not flip either window negative)

### Gate B — combine MC (decision gate)
Same harness and corrected rules as parent Gate 2 (EOD-ratchet MLL locking at $50k, DLL day-deactivation, consistency rule, 5,000 sims, 90-day cap, day-block bootstrap on pooled trades):
- Pass% ≥ **50%** at some integer size 1–10 AND Pass% > Blow% at that size
- Also report stall-resolved pass share (pass/(pass+blow)) for context; it does not gate

### Stop rule
Both S-variants fail Gate B → branch closed permanently. No third level, no re-derivation.

## 5. Untouched
Live bots, models, YAML configs, sealed holdout. A Gate-B pass authorizes a deployment pre-registration only — not combine entry by itself.
