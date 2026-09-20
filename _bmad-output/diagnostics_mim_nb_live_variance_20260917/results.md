> **ERRATUM 2026-09-20 — the N=26 / PF 0.689 figures below are wrong.** They used `write_mode='realtime'`, which drops MIM-NB's first two live trades. The authoritative ledger (`data/mim_nb/trades.csv`) is N=28, net −$753.00, PF 0.848 — above the 0.70 line, 2 trades from N=30. The claim that the halt "would already fire" is withdrawn. See `_bmad-output/diagnostics_mim_nb_n30_race_20260920/REPORT.md` §0.

# MIM-NB live drawdown vs. its own modeled variance (2026-09-17)

**Question:** is MIM-NB's −$1,540.50 over 26 live trades inside the variance its own
sealed model expects, or is it a signal worth acting on before it reaches N=30?

**Inputs:** `data/trades.db` (`trader_id='trader-mim-nb'`, `write_mode='realtime'`) as of
2026-09-17; reference distribution `data/reports/mim_nb_gate1_v2_2026oos.csv` (the OOS-2026
gate1 v2 trade list cited in the seal itself, converted pts→USD at $2/pt, gross of
commission). Script: `check_variance.py`; raw output `results.json`.

## The sealed model's own decision rule

`preregistration_mim_nb_honest_expectations.md` §5 (sealed 2026-06-14, no config change
since — the live bot is running this exact strategy):

> If live MIM-NB reaches **N ≥ 20–30** completed trades with net PF tracking the OOS 1.30,
> the central-case expectation upgrades from hypothesis to evidence. **If net PF < 0.70
> over 30 trades, the deployment halt trigger fires** and this whole expectation is void.

Live is at **N=26 now**, squarely inside the window this trigger was written to watch.

## Live vs. reference

| | N | Net | Mean/trade | PF |
|---|---|---|---|---|
| OOS-2026 reference (gate1 v2, gross) | 62 | +$4,231 | +$68.24 | 1.659 |
| Honest-expectations §2a (net of cost, same window) | — | — | +$31.99 | 1.299 |
| **Live, all 26 trades** | 26 | **−$1,540.50** | **−$59.25** | **0.689** |
| Live, excl. 2026-09-15 (known contamination) | 25 | −$1,185.50 | −$47.42 | 0.742 |

**Live PF is already below the sealed halt threshold (0.70) at N=26, four trades early.**
Excluding the one known-bad trade below, it sits just above the threshold (0.742) — the
verdict currently hinges on that one trade.

## The known contamination

2026-09-15's −$355 is not organic: `project_mim_nb_roll_contamination_20260915` (memory,
unfixed live bug) — AUTOROLL read `open_d` from the outgoing U26 bar while marks had
already moved to Z26 (+293 pt), forcing a spurious LONG. Reported both ways above; the
bug is real and unfixed, but excluding it doesn't clear MIM-NB — PF stays under the OOS
reference and near the halt line.

## Is this within modeled variance?

IID day-block bootstrap (200,000 draws) of the OOS-2026 reference distribution, resampled
to the same N as live:

| | N | P(net ≤ observed) | P(PF ≤ observed) |
|---|---|---|---|
| vs. all 26 live trades | 26 | **3.5%** | **5.5%** |
| vs. 25 trades excl. contamination | 25 | 5.4% | 7.4% |

Not a coin flip either way — a run this bad happens roughly **1 time in 15–30** under the
model that authorized this deployment. That is a real tail draw, not "obviously broken,"
but also not comfortably inside the bulk of the distribution.

**One caveat that cuts against alarm:** the seal's own §2d.1 flags that this same IID
day-resampling *understates* real-world regime clustering — bad trades cluster in choppy
regimes more than IID sampling implies, so a streak this length is probably somewhat more
likely in reality than the 3.5–5.5% figures suggest. That cuts the same way as normal
variance, not as evidence of a broken edge — but it does not, on its own, get from "below
the halt line" to "the halt line was mis-set."

## Verdict

Not a clean "within variance, ignore it." Two things are true at once:
1. A drawdown this size is a genuine (if not extreme) tail event under MIM-NB's own model,
   consistent with the fat-tail structure the seal already disclosed (§2b: ~160 of 163 days
   near breakeven, edge concentrated in rare trend days that haven't shown up yet).
2. **The sealed halt trigger (PF < 0.70 at N≥30) is four trades from being live**, and
   at the current live PF (0.689) it would already fire if N=30 landed today. This is not
   a new judgment call — it is the pre-registered rule the seal exists to be read against
   before anyone is tempted to relitigate it after trade 30.

**Recommendation:** do not act yet — N=30 isn't reached and the rule is explicit about
that threshold — but flag it now, not after: the next ~4 MIM-NB trades decide whether
this deployment's own pre-committed halt condition fires. Watch it the way the GAP-1
N=30 trigger is being watched (see `precommitment_gap_fade_n30_decision_20260917.md`);
no config or code change is proposed here.
