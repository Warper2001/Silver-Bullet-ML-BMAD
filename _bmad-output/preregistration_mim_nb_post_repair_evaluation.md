# Pre-Registration: MIM-NB Post-Repair Evaluation — works / adjust / drop

**Generated:** 2026-07-28
**Experiment ID:** mim-nb-post-repair-evaluation
**Type:** Prospective evaluation with pre-committed decision rule. NO parameter tuning,
NO holdout access.
**Predecessor:** `preregistration_mim_nb_sigma_provenance.md` (§6 names this document)

**Status:** DRAFT — seals on merge to main. **Clock does not start until the predecessor's
gates G1–G3 all pass.**

> **Authored ahead of its trigger, deliberately.** The predecessor set a 5-business-day
> deadline after G1–G3 pass. Writing it now costs nothing — no content here depends on
> the repair's outcome — and removes the risk that a successful repair sits unevaluated.

---

## 0. The question, and an honest statement of what is answerable

Alex's question (2026-07-28): *does MIM-NB work, need adjustments, or get dropped?*

Before designing a test, the power of that test must be stated. Using the sealed pooled
trade distribution (`data/reports/mim_nb_catstop_s250_pooled.csv`, N=164, deployed S=250
config, net of `COST_PTS=1.12` at `PT_VAL=2.0`):

| | |
|---|---|
| Mean net per trade | **+$41.28** |
| SD per trade | **$434.05** |
| Sharpe per trade | **0.0951** |
| **N for 80% power to detect mean > 0** (two-sided, α=0.05) | **867 trades** |
| At the sealed cadence of 1.01 trades/session | **≈861 sessions ≈ 41 months** |

**Direct confirmation that MIM-NB has positive edge requires ~3.5 years of trading.**
That is not a timeline anyone is operating on. Any pre-registration promising a
works/adjust/drop verdict at N=20 or N=30 would be misrepresenting its own power — this
one will not.

This is a property of the edge's *shape*, not a flaw in the strategy: the sealed evidence
(`preregistration_mim_nb_honest_expectations.md`) records that ~160 of 163 days hover near
break-even and a handful of fat-tail trend days pay for everything. A distribution whose
mean lives in its top 2% of observations is intrinsically slow to measure.

**Therefore this experiment does not attempt to prove the edge.** It tests the two things
that *are* answerable on a useful timescale:

- **Track A — is the bot faithful?** (paired against the engine; high power, small N)
- **Track B — is the bot harmful?** (futility bound against the sealed distribution)

---

## 1. Venue — where MIM-NB runs to accrue N

**Decision: evaluation runs on TradeStation SIM paper (`SIM2797251F`), not the Topstep
combine account.**

Rationale — the combine account is structurally incapable of producing this data:

- The per-entry buffer gate (`buffer ≤ cat_stop_cost`) has blocked **every** entry since
  2026-07-24. At the time of writing, `buffer $499.12 ≤ cat_cost $500.00`.
- The buffer only recovers if the account makes money; MIM-NB is one of two things that
  could make it money; it cannot trade. **This is a deadlock, not a delay.**
- The $2,000 trailing MLL guarantees the gate re-closes on any drawdown, so even a
  recovery is temporary.

**Requirements on the paper instance:**

1. **Symbol isolation.** The SIM account is shared with the mirror and other bots. The
   evaluation instance must be identifiable by symbol + order tag, per the contamination
   fix in `72e8f1f`. Any ambiguity in attribution voids the affected trades.
2. **Buffer gate inert, and provably so.** With no trailing MLL the gate has no meaning.
   It must be disabled *for the paper instance only*, and the isolation must be
   structural — a shared constant edited in place would silently disarm the real-money
   path. Verified by asserting the real-money instance still logs `BUFFER_GATE` while the
   paper instance does not.
3. **The real-money combine instance is not modified by this prereg.** It remains blocked.
   Whether MIM-NB returns to real money is a *later* decision, gated on Track A + B.

**Disclosed cost of this choice:** paper fills are not real fills. MIM-NB has 12
broker-confirmed ProjectX round-trips, which is a thin basis for a slippage estimate and
is explicitly carried forward as an *assumption*, not a measurement (see §4). This project
has already been burned once by unmeasured execution cost — the SIL slippage verdict
(2026-06-19) closed an entire strategy family at $15/RT against an $8.74 bar. **A paper
result that passes both tracks is not authorization to trade real money; it is
authorization to run a fill-realism check.**

---

## 2. Track A — execution parity (primary, high power)

Once σ is deterministic, live and the sealed engine should take **the same trades on the
same bars**. The difference between them is therefore a *paired* quantity with almost no
variance — which is precisely why it can be measured at small N while the edge cannot.

**Metric.** For each live trade, the matched engine trade on the same session:

- direction, entry mark, and exit reason identical
- `Δ = live realized net − engine modeled net`

**Sample.** First **20 live trades** or **40 sessions** after G1–G3 pass, whichever comes
first.

**Pre-committed pass criteria:**

| | Criterion |
|---|---|
| **A1** | ≥ **95%** of live trades match the engine on direction + entry mark + exit reason |
| **A2** | mean \|Δ\| ≤ **$12.00** per round-trip (2× the modeled `COST_PTS=1.12` × `PT_VAL=2.0` × 2 legs ≈ $4.48, widened to absorb realistic slippage) |
| **A3** | no unexplained flatten events (cf. the 2026-07-06 `EXTERNAL_FLATTEN`) |

**A failure of Track A is an engineering verdict, not a verdict on the edge.** It sends
the work back to repair and this clock restarts. It must never be reported as "MIM-NB
doesn't work."

---

## 3. Track B — futility bound (the drop trigger)

Since the edge cannot be confirmed, the pre-registered question is inverted: **is realized
performance so poor that the sealed distribution is no longer a credible description of
this system?**

The rejection line is a lower percentile of cumulative net P&L at 1 contract under the
sealed day-level distribution (163 sealed days, `−$1,000` day cut applied, 20,000
bootstrap paths, `numpy.random.default_rng(7)`).

### 3.1 Multiplicity correction — why the line is not p5

Four checkpoints each tested at a nominal 5% is **four looks**, not one. Simulated
directly: a MIM-NB that genuinely draws from the sealed distribution breaches a p5 line at
**at least one** of sessions 40/80/120/163 with probability

> **12.19%** — a 1-in-8 chance of dropping a working strategy.

Per-checkpoint α is therefore set to **2%**, which yields a familywise false-drop rate of
**5.30%**:

| Sessions | **Rejection line (α=2%)** | median | P(cum < 0) |
|---|---|---|---|
| **40** | **−$3,222** | +$1,387 | 28.4% |
| **80** | **−$3,866** | +$3,104 | 19.4% |
| **120** | **−$4,022** | +$4,753 | 14.5% |
| **163** | **−$3,964** | +$6,405 | 11.0% |

**Evaluation checkpoints: sessions 40, 80, 120, 163. No interim looks.**

**Rule:** if cumulative realized net at a checkpoint is **below the line above**, the
sealed hypothesis is rejected → **DROP**.

**Disclosed trade-off:** controlling false-drops at ~5% familywise makes this bound
**lenient** — MIM-NB must lose roughly $4,000 at 1 contract before it is dropped. That is
accepted deliberately. The alternative (a tighter line) discards working strategies at an
unacceptable rate, and this evaluation runs on paper where no capital is at risk. **A
lenient drop trigger on paper is the correct trade; it would not be on real money.**

### 3.2 Cadence guard

The bound resamples sealed *days*, which carry **1.01 trades/session**. If the post-repair
live instance fires at a materially different rate, cumulative P&L accrues on a different
clock and a session-indexed bound is no longer comparable.

**Guard:** at each checkpoint, compute realized trades/session. If it falls outside
**[0.71, 1.31]** (±30% of 1.01), the checkpoint is **deferred** until the realized trade
count reaches what the sealed cadence would have produced by that session
(`1.01 × sessions`). Deferral is recorded; it is not a look.

This matters concretely: over the live era the sealed engine fired **21 trades in 33
sessions = 0.64/session**, well below the sealed sample's 1.01. If that persists, every
checkpoint defers and the calendar in §4 stretches accordingly.

**Note the shape, and note it now:** `P(cum < 0)` is still **11% at 163 sessions**. A
genuinely working MIM-NB has better than a 1-in-10 chance of being underwater after eight
months. **Being unprofitable at any checkpoint is therefore NOT, by itself, evidence of
anything** — only breaching the §3.1 rejection line is.

---

## 4. Decision rule — mapping to works / adjust / drop

| Outcome | Trigger | Meaning |
|---|---|---|
| **DROP** | Track B rejection line (§3.1) breached at any checkpoint, **or** Track A fails twice after two repair attempts | The sealed distribution does not describe this system |
| **ADJUST** | **Only** on a *named mechanism failure* — a parity break, a structural defect, an unexplained broker event | Requires its **own** prereg with its **own** out-of-sample evidence |
| **CONTINUE** | Track A passes and Track B stays above the line through session 163 | **Not rejected.** Explicitly **not** "works" |
| **WORKS** | — | **Cannot be established by this experiment.** Requires ~867 trades ≈ 41 months |

### The anti-iteration clause

**"ADJUST" may never be selected because performance was disappointing.** Disappointing
performance that does not breach the §3.1 line → **CONTINUE unchanged**. Disappointing performance
that breaches the §3.1 line → **DROP**.

This is the project's single most expensive documented failure mode
(`feedback_iteration_loop_pattern.md`: three instances). Tuning a parameter in response to
observed results and re-testing on the same data is not validation. If the room believes a
mechanism is wrong, that belief must be written as a *new hypothesis with a new prereg and
fresh data* — never as a mid-flight adjustment to this one.

### Timeline

At ~21 sessions/month, T0 = first session after G1–G3 pass (est. early Aug 2026):

| Checkpoint | Est. date |
|---|---|
| Track A (20 trades / 40 sessions) | ≈ 2026-09-30 |
| Session 40 | ≈ 2026-09-30 |
| Session 80 | ≈ 2026-11-30 |
| Session 120 | ≈ 2027-01-31 |
| Session 163 | ≈ 2027-03-31 |

**At session 163 the honest outcome is "not rejected," not "validated."** What to do with a
not-rejected-but-unconfirmed strategy after eight months is a **business decision, not a
statistical one**, and this prereg does not pre-empt it. It pre-commits only to holding
that review, on that date, with these numbers.

---

## 5. Disclosures

- **Pre-fix trades are excluded.** The 13 live trades from 2026-06-11 to 07-23 were
  produced by a bot with non-deterministic bands and **must not** be pooled into any N
  here. They are diagnostic material. N starts at zero at T0.
- **No sealed holdout data is accessed or spent.**
- **The cat-stop shadow ledger** (`preregistration_mim_nb_catstop_shadow_ledger.md`,
  SEALED 2026-07-07, fixed-N=10) runs independently and is unaffected. It has zero
  recorded events; its clock begins at the first post-fix cat-stop.
- **Slippage is assumed, not measured** — carried from 12 ProjectX round-trips. §1 states
  why a paper pass is not authorization for real money.
- **Track B's rejection line is derived from the sealed distribution**, which bootstraps
  day-level results and therefore under-models regime clustering
  (`preregistration_mim_nb_honest_expectations.md` discloses this). A real regime shift
  could breach the line for reasons the bootstrap treats as near-impossible — meaning the
  line is more likely to fire on *regime change* than on *absence of edge*, and the two
  are not distinguishable by this test. Accepted, and stated so the eventual DROP is not
  over-interpreted as "the edge was never there."
- **§3.1's α=2% and §3.2's ±30% cadence band are the only two chosen numbers in this
  document.** Both are derived, not asserted: α=2% is the value that returns the
  familywise false-drop rate to ~5% (simulated, 20,000 paths); the cadence band is set to
  the width at which a session-indexed bound and a trade-indexed bound stop agreeing.
  Neither was picked by looking at live results.
- **`A2`'s $12.00 threshold is asserted, not derived** — it is ~2.7× the modeled
  round-trip cost, chosen to absorb realistic slippage without a measurement to justify
  it. This is the weakest number in the document. If the first 20 trades show a stable
  \|Δ\| distribution, A2 should be re-derived from it under a fresh seal rather than
  quietly reused.
- **This prereg cannot be evaluated early.** Checkpoints are fixed at 40/80/120/163
  sessions.

---

## 6. Integrity hashes

| Hash | Value |
|---|---|
| (a) Sealed pooled trades (S=250) | `data/reports/mim_nb_catstop_s250_pooled.csv`, N=164, SHA-256 recorded at seal time |
| (b) `study_mim_nb_catstop.py` SHA-256 | `210518d63a76374f6b8bb79b8e60a2bb3dd94d3e00e1ac73c6b0a032846cc6a1` |
| (c) Predecessor seal | `preregistration_mim_nb_sigma_provenance.md` (PR #17) |
| (d) Bootstrap spec | 20,000 paths, `numpy.random.default_rng(7)`, day-level resample with replacement, `−$1,000` day cut, 1 contract, per-checkpoint α=2% |
| (d2) Familywise calibration | nominal p5 × 4 checkpoints → 12.19% false-drop; α=2% → 5.30% |
| (e) Git HEAD at draft | `0557696` |
| (f) Power calc | mean $41.28, sd $434.05, N=164 → 867 trades for 80% power at α=0.05 two-sided |

---

## 7. What this prereg does NOT authorize

- Returning MIM-NB to the real-money combine account.
- Any change to `CAT_STOP_PTS`, `DLL_GUARD_USD`, `CONTRACTS`, `LOOKBACK_DAYS`, or the
  band formula.
- Increasing size, on paper or otherwise.
- Declaring MIM-NB "validated" on any evidence produced here. The strongest available
  outcome is **not rejected**.
