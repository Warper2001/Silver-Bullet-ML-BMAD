# Mathematical checks and conditional timeline

All examples in `calculate.py` are explicit assumptions; none estimates market performance. `calculation-results.json` is reproducible with the existing Python/SciPy environment. The verifier independently checks sample covariance arithmetic, a direct covariance-matrix sum, chi-square(df=2)'s closed form, union bounds, and the previous scenario ledger.

For paired session outcomes, `Var(D)=v_K+v_M-2*c_KM`. Synthetic `v_K=10000`, `v_M=6400`, `c=4000` gives `v_D=8400`, `sd_D≈91.652`. Positive semidefiniteness requires `|c|<=sqrt(v_K*v_M)=8000`. Without correlation information the variance upper bound is `(100+80)^2=32400`, not `10000+6400`. These identities do not supply actual arm variances.

Under a stationary finite-second-moment vector process `Y_t=(K_t,M_t)`, define `Gamma_h=Cov(Y_t,Y_(t-h))`. The *finite-N* covariance of its mean is

`V_N = [N*Gamma_0 + sum_(h=1..N-1) (N-h)*(Gamma_h+Gamma_h^T)]/N^2`.

For `a=(1,-1)`, `Var(mean D)=a^T V_N a`; cross-arm, cross-lag covariance matters. If covariance sums converge, the long-run covariance is `Omega=Gamma_0+sum_(h>=1)(Gamma_h+Gamma_h^T)` and `V_N≈Omega/N`. This convergence and any normal approximation need evidence. [Newey–West](https://www.nber.org/papers/t0055) provides primary methodological support for consistent HAC estimation under conditions, not a finite-sample assurance for this market process. A block-bootstrap label does not identify a valid block length or overcome too few effective blocks.

The synthetic stationary AR(1) example has exact mean variance `sigma²*[N+2*sum((N-h)*rho^h)]/N²`. At N=20 and rho=.5 its variance inflation is about 2.80 (SE inflation about 1.673), versus asymptotic variance inflation 3. At rho=.9 the exact inflation is about 11.094 (SE inflation 3.331). Thus the old hypothetical SE inflations 1/1.5/2 are sensitivities, not upper bounds. As rho approaches one, 20 observations can approach one effective observation. More seeds or overlapping forecasts cannot repair that.

For the inherited normal known-variance approximation, `z(.975)+z(.90)=3.241515`. Required sessions for a positive distance `d` from the tested boundary are `ceil(zsum²*variance*f²/d²)`. If covariance is uncertain, optimize both endpoint variances over a *joint* plausible region, with positive-semidefinite constraints and dependence uncertainty; do not choose favorable point estimates independently. A supported lower alternative effect and upper variance/dependence envelope are needed. When f already represents long-run risk, do not multiply by a second dependence factor.

The iid-normal pilot example uses n=20, nu=19, synthetic sample variance 8400, and a one-sided 95% upper variance limit. `nu/chi2(.05,nu)` inflates variance (and approximate required N), while its square root inflates SD. The helper prints both and the corresponding synthetic sample sizes for an arbitrary distance of $20. **That $20 is a unit-test example, not an economic alternative or target.** These normal-iid limits cannot be made dependence-robust by substituting an informal effective N for degrees of freedom.

A joint nuisance region with 95% coverage and conditional joint power at least .80 yields only a .76 lower bound on unconditional assurance (`.95*.80`) under valid planning/transport assumptions and a fresh independent evaluation population. It does not promise .80 unconditional power. Two separate 95% nuisance bounds guarantee only 90% joint coverage by the union bound, hence .72 assurance in the same illustration. These are conservative algebraic lower bounds, not estimated performance. Confidence coverage itself fails if iid normal assumptions or future transport fail. Repeated pilot peeking also invalidates fixed-n limits unless specifically accommodated.

Relative precision under iid normality can be expressed without an unknown scale: the ratio of upper to lower endpoints of the two-sided 95% variance interval is `chi2(.975,nu)/chi2(.025,nu)`. A future independently chosen precision tolerance would imply a sample size; this packet adopts neither tolerance nor sample size. Reported n=2/5/20/60 examples demonstrate sensitivity only. One observation cannot estimate variance, and a constant short pilot does not establish zero risk.

A nonnormal counterexample makes the last point exact: let X be 1000 with probability .001 and 0 otherwise. Its variance is 999, yet a 20-observation iid pilot is all zero with probability `.999^20≈.98019`. A zero observed variance would miss substantial tail risk. This is finite variance, so even finite-variance assumptions alone do not rescue the normal chi-square precision claim. The paired mismatch counterexamples are in [routes.md](routes.md). Synthetic success verifies arithmetic; synthetic failure refutes universal claims, not any measured Kronos result.

The prior capacities are preserved, not recalculated as admitted sessions:

| Scenario | 3 months | 6 months | 12 months |
|---|---:|---:|---:|
| Proxy no-roll ceiling, calibration independently completed beforehand | 59 | 119 | 246 |
| Assumed quarterly resets, calibration independently completed beforehand | 54 | 109 | 226 |
| Assumed resets/losses, 20 hypothetical calibration sessions removed | 15 | 46 | 119 |

These are copied from the September 23 design ledger and checked against it. The last row's conditional collection capacities are respectively 35, 66 and 139; the other rows reserve zero calibration. Proxy calendar, warmup, outages and roll assumptions remain unresolved. The 20-session allocation is not a statistical justification. No horizon is currently sufficient or admitted. Standardized detectable effects are reproduced for all nine rows in the JSON.

For an actual future programme, elapsed time is

`T_total = T_preparation + T_calibration + T_analysis_and_freeze + T_evaluation + T_final_analysis`.

Preparation includes economics, applicable evidence, separate admission and collection preregistrations, access isolation and operational/calendar checks. Calibration duration includes context warmup, completed eligible sessions, rolls, outages and missing outcomes. Analysis/freeze includes uncertainty estimation, power reassessment, cost freeze, review and the evaluation preregistration; this must finish before the first evaluation cutoff. Evaluation cannot begin merely because the twentieth calibration session has occurred. Final analysis also consumes time and operator resources.

If H months is an end-to-end deadline starting at planning, preparation and analysis/freeze consume part of H. Reserve only sessions remaining after the actual freeze date; rerun the dated ledger for that deadline, including resets and completeness losses. Do not subtract a generic fixed number of days or silently shift the endpoint. If H instead denotes the collection window, total elapsed time exceeds H by preparation, excluded delays and final analysis; label that explicitly. No numeric delay or finish date is justified now. Future budget must specify cash, operator hours, storage/compute and a hard calendar cap. This investigation's two-hour/no-purchase limit is not that future budget.
