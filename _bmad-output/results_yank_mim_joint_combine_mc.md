# Results: YANK + MIM-NB Joint Combine Monte Carlo

**Run date:** 2026-06-17
**Pre-registration:** `preregistration_yank_mim_joint_combine_mc.md` (re-seal v2, commit 75fc1eb)
**Engine:** `tools/joint_combine_mc.py` — 5,000 sims, 90-day cap, shared $50k account, $2k trailing ratchet, $48k floor, per-strategy DLL, both engines costed at $2.24/ct.

## Validation
MIM-only through this engine = **54.0% pass / 33.3% blow / median 41 days** — reproduces the deployed single-strategy baseline exactly (cap=500 cat-stop). Engine and inputs confirmed faithful.

## PRIMARY (common-ET-day block bootstrap, overlap window, governs the decision)
Pool: 145 union-traded days (115 MIM, 48 YANK, 18 both).

| Size (MIM:YANK) | Pass% | Blow% | Run-on% | Median days |
|---|---|---|---|---|
| baseline (MIM 1ct only) | 54.0% | 33.3% | — | 41 |
| 1 : 1 | 56.4% | 25.6% | 18.0% | 47 |
| 1 : 2 | 64.8% | 26.2% | 9.0% | 40 |
| 1 : 3 | 67.6% | 29.2% | 3.1% | 33 |

## SENSITIVITY (independent draw, optimistic bound — does NOT govern)

| Size (MIM:YANK) | Pass% | Blow% | Run-on% | Median days |
|---|---|---|---|---|
| 1 : 1 | 73.3% | 22.2% | 4.5% | 33 |
| 1 : 2 | 77.5% | 21.7% | 0.8% | 24 |
| 1 : 3 | 74.5% | 25.3% | 0.2% | 18 |

## Verdict (per sealed §5 gate)
**ADOPT — MIM 1ct : YANK 1ct.** Every tested size satisfies pass% > 54% AND blow% ≤ 33%; the gate adopts the smallest qualifying size (conservative, per the tail-correlation caveat). H₁ confirmed: the uncorrelated second stream raises pass AND cuts the blow tail — a path-variance reduction, not an edge addition.

## Reading / caveats
- The cleanest, most robust signal is the **blow-tail reduction** (33% → ~26%), stable across all sizes — diversification doing what it should against a trailing-drawdown kill switch.
- Pass% improvement scales with YANK size: minimal at 1:1 (+2.4pp), strong at the vol-balanced 1:2 (+10.8pp → 64.8%) at no blow cost. Choosing between the conservative 1:1 and the vol-balanced 1:2 for live deployment is a **separate deployment-prereg decision**, not a re-litigation of this gate.
- **Thin data:** only 18 days where both traded; both instruments are MNQ, so a genuine tail regime could raise the benign ~0.015 correlation. The primary arm (not the optimistic sensitivity) is the number to trust; the ~17pp gap between them is the diversification-fragility margin.
- This does NOT authorize a live change. Porting YANK onto the ProjectX combine account at a down-sized contract count, with joint halt triggers on the shared floor, requires its own deployment pre-registration.
