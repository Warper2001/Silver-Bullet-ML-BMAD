# Native reclaim power feasibility — 2026-09-14

The gate is complete with **POWER_UNDETERMINED** and `evaluation_allowed=false`. The six eligible native sessions support a conditional planning calculation, but the inspected evidence does not establish the strategy's net effect, variability, costs or dependence. No signals, trades or returns were calculated, and profitability remains untested.

Under the prespecified iid normal, unknown-variance model, six independent sessions require **d=1.1858** to attain 80% power at a one-sided 5% significance level. Here **d is hypothetical mean net session dollars divided by the population standard deviation of net session dollars**. Neither quantity was estimated. Power is the chance that the proposed statistical test detects a specified positive mean under its assumptions; it is not win rate or probability of profit.

| Hypothetical d | Conditional power at six sessions | Minimum independent sessions for 80% power |
|---:|---:|---:|
| 0.10 | 7.62% | 620 |
| 0.20 | 11.17% | 156 |
| 0.30 | 15.74% | 71 |
| 0.50 | 27.99% | 27 |
| 1.00 | 67.69% | 8 |

These are all five preregistered scenarios. None is designated likely or preferred. Session counts are conditional on the model; they are not guaranteed data horizons, calibrated effective sample sizes or a recommendation to adopt an assumed effect. Dependence, nonnormality, missing coverage and sample selection remain unresolved.

| Independent sessions | Hypothetical d detectable at 80% power |
|---:|---:|
| 2 | 5.7928 |
| 3 | 2.2973 |
| 4 | 1.6497 |
| 5 | 1.3594 |
| 6 | 1.1858 |

The separate known-variance normal cross-check gives d=1.0151 at six sessions. Its lower requirement does not replace the unknown-variance t result. The model follows the [NIST sample-size discussion](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm) and the [SciPy noncentral-t definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html); neither source supplies a strategy effect estimate.

The pilot's original six eligible and four excluded sessions were preserved. The 8,274 dependent profile comparisons are not independent trials. Other inspected native archives contain short windows, including windows selected around earlier parity fills; they do not supply independent representative full-session calibration. This conclusion is limited to the inspected `data/yank` and `data/tick` metadata.

The next useful work is to obtain representative native full-session coverage with contract, calendar and feed evidence; assign calibration and validation roles before observing strategy outcomes; and establish defensible costs, a minimum worthwhile net effect and variability/dependence assumptions. Then pre-register the next power assessment and strategy test. Continue using native volume at price; the prior measurement showed that uniform OHLCV profiles often disagree. No filter or live parameter is adopted from these results.

The preregistration was committed as `42d905e2914a0d901e933dfa86651c223e0594f3`. Reviewed implementation/tests were committed as `a9509494b8f6b20456f75824c94833e05e188c98` before the actual gate. The [pre-run verification](../_bmad-output/valentini-power-20260914/pre-run-verification.json) checks ancestor history and committed file bytes. The [machine report](../_bmad-output/valentini-power-20260914/run-final/report.json), [manifest](../_bmad-output/valentini-power-20260914/run-final/manifest.json) and [independent verification](../_bmad-output/valentini-power-20260914/verification-final.json) bind inputs and calculations. The report SHA256 is `add7feaef6f5e9fee811a7529ef5de82a57eeb20c11c838eabf910bab4c02a17`.

Validation: 214 focused tests passed, with Black, flake8 and strict mypy. Independent chi-square integration checked all reported frontier powers and six-session scenario powers; exhaustive integer checks verified every required sample size. Review corrections were completed with no deferred findings. Three review passes used two independent-of-implementation agents because the host refused an additional fresh thread; blind and verification passes shared a reviewer context. See the [review record](../_bmad-output/valentini-power-20260914/review-context.md).

The [gate command and scope](valentini-power.md) document reproduction. No live strategy, service, sealed holdout or shared dependency environment changed.
