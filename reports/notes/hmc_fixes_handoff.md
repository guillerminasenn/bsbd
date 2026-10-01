# Collapsed-HMC fixes and speedup: handoff

Last updated: 2026-10-01  
Branch: `hmc-fixes-and-speedup`

## Scope and current state

This branch implements Phases 0--5 of `plans/2026-09-30_hmc-fixes-and-speedup.md` and the reusable parts of Phase 6. The full Phase 6 sweeps and Phase 7 adjoint-gradient note remain deliberately unfinished. Do not launch the full sweeps without explicit approval: the current constraint is laptop-only work with less than 0.5 hours of computation at a time.

The central correctness and performance changes are complete:

- selectable `legacy`, `legacy_fixed`, and `adjoint` collapsed-HMC gradients;
- the legacy reshape-aliasing fix and an `m=0` guard;
- working identity, prior, and fixed posterior mass matrices;
- cached mass-matrix factorization;
- exactness-preserving cyclic Gibbs speedups;
- experiment, ESS-summary, and fixed-parameter HMC-tuning scripts;
- 17 tests covering gradients, potentials, Gibbs corrections, and mass matrices;
- a covariance-validity and separability investigation in `reports/notes/covariance_validity.tex`.

## Repository state at handoff

The last pushed implementation/report commit before this handoff was `4cdb208`. The following local commit, `dfaf3f9`, adds Reviewer 1 and Reviewer 2 reports; this handoff adds the posterior-tuning integration and durable project state. Before continuing, run:

```text
git status --short --branch
git log --oneline --decorate -10
```

Important earlier commits:

| Commit | Content |
|---|---|
| `6716610` | Phase 0 scaffolding |
| `2436543` | Phase 1 correctness fixes |
| `724a6ed` | Phase 2 adjoint gradient |
| `f9e2877` | Phase 3 mass matrices |
| `cbdad53` | Phase 4 Gibbs speedups |
| `01c0a53` | Phase 5 tests |
| `f73572f` | Phase 6 runner, summaries, and first pilot |
| `913b30e` | 24x1 pilot artifacts |
| `3d72e98` | fixed-parameter HMC tuner |
| `4cdb208` | covariance validity and separability note |

## Verified correctness

The last complete test run before this handoff had 17 passing tests. The numerical checks established:

- `legacy_fixed` and `adjoint` gradients agree to approximately `1e-13`;
- their potentials agree to approximately `1e-15`;
- both agree with central finite differences, while `legacy` reproduces the historical aliasing error;
- same-seed `legacy_fixed` and `adjoint` trajectories have the same accept/reject behavior;
- optimized and original Gibbs constraint corrections agree to numerical precision;
- identity, prior, and posterior momentum covariance, kinetic energy, and kinetic gradient are mutually consistent.

Use the `bpwave2-env` interpreter. The base conda interpreter has macOS code-signature import failures:

```text
/Users/guillers/anaconda3/envs/bpwave2-env/bin/python -m pytest tests/
```

## Tuning protocol

Reported HMC/Gibbs comparisons must use fixed HMC parameters selected by preliminary runs, not an adaptively changing production chain:

1. Generate one synthetic problem and keep it fixed across candidates.
2. If using posterior preconditioning, run a short prior-preconditioned pilot and estimate the inverse regularized covariance of the free blur coordinates.
3. Run a fixed `(epsilon, L)` grid with no adaptation.
4. Exclude stuck chains: require nonzero minimum blur ESS and acceptance rate at least 0.2 by default.
5. Select the candidate maximizing minimum blur-coordinate ESS per second.
6. Run fresh, longer fixed-parameter HMC and Gibbs chains for the comparison.

`scripts/tune_hmc.py` now performs this workflow, including posterior-mass estimation. Example for the weak-constraint case:

```text
/Users/guillers/anaconda3/envs/bpwave2-env/bin/python scripts/tune_hmc.py \
  --nv-obs 96 --nh-obs 12 -k 12 --m 12 --precondition posterior \
  --eps-grid 0.1,0.2,0.4,0.8 --L-grid 5,10,25 \
  --grid-N 300 --final-N-hmc 1500 --final-N-gibbs 1500
```

Acceptance rate is a diagnostic/feasibility guard, not the objective. The often quoted 0.234 optimum is for high-dimensional random-walk Metropolis (Roberts, Gelman, and Gilks, 1997), not HMC. The corresponding asymptotic HMC result is approximately 0.651 (Beskos et al., 2013). Here the defensible objective is measured ESS/s; an acceptance-only target can choose a slower trajectory. If a sensitivity analysis at AR near 0.234 is desired, report it separately rather than replacing ESS/s tuning.

## Valid preliminary efficiency results

All values below are laptop-scale preliminary results, not final paper estimates. ESS/s is the minimum over free blur coordinates after 20% burn-in. Short tuning runs are noisy, so final sweeps need longer independent replicates.

| Lattice and constraints | HMC mass | Fixed HMC result | Gibbs ESS_min/s | HMC/Gibbs |
|---|---|---:|---:|---:|
| 24x1, k=12, m=12 | prior | 46.38 ESS_min/s; `epsilon=0.05`, `L=25`, AR 0.97 | 9.45 | 4.91 |
| 96x1, k=12, m=48 | prior | 45.32 ESS_min/s | 72.37 | 0.63 |
| 96x12, k=12, m=96 | prior | ratio only retained from pilot | -- | 0.47 |
| 96x12, k=12, m=12 | prior | 1.64 ESS_min/s | 0.96 | 1.72 |
| 96x12, k=12, m=12 | posterior | 4.56 ESS_min/s; `epsilon=0.4`, `L=5`, AR 0.74, 7.5 ms/iter | 0.96 | 4.75 |

For the final posterior-preconditioned 96x12 run, the minimum and median blur ESS were 41.1 and 201.7 among 1200 retained draws. The posterior mass improved minimum ESS/s by about 2.8x over prior preconditioning. The short grid chose `(0.4, 5)`; candidates at `epsilon=0.8` had AR 0 and misleading raw ESS from constant chains, which is why the movement and AR guards are mandatory.

The earlier 96x1, k=12, m=96 result had an HMC/Gibbs ratio near 0.03, but it observes the entire one-dimensional image and is a degenerate comparison: Gibbs is then effectively drawing from highly informative full conditionals. Keep it only as a boundary case, not as evidence about the intended weak-well setting.

**Do not use or cite the earlier 96x12, k=24 pilot.** That wavelet configuration is not allowed for the intended model and the user explicitly rejected it. It is intentionally absent from the table above.

## Interpretation

The current evidence supports a qualified claim: correctly implemented and tuned marginal HMC can outperform Gibbs in a unimodal regime, but not uniformly over constraint strength or geometry.

- With sparse exact-image constraints, marginalizing the high-dimensional image can remove strong blur-image dependence that slows Gibbs.
- The prior precision is not always close to the local marginal posterior geometry. In the 96x12, m=12 case, a fixed empirical posterior precision materially improved HMC.
- With many exact image observations, the Gibbs full conditionals become much more informative and can approach independent draws, while HMC still pays for several gradients per proposal.
- Therefore performance is not monotone in image size or in `m/n`; posterior geometry, the exact-observation pattern, mass matrix, and trajectory length all matter.

The paper should claim a measured crossover/operating regime, not universal HMC superiority.

## Covariance and separability result

`reports/notes/covariance_validity.tex` contains the full investigation. The key conclusions are:

- a stationary covariance on a cyclic lattice is valid iff the DFT eigenvalues of its circulant base are nonnegative;
- composing a Euclidean powered exponential with geodesic distance is not generally valid near smoothness exponent 2;
- the published parameter values were numerically SPD, but the general recipe needs correction;
- periodization/wrapping is the recommended construction when smooth blur priors are required;
- core BCCB/FFT likelihood algebra needs toroidal stationarity, not Kronecker separability;
- separability is nevertheless used by the current directional prior specification, exact-well conditioning factorization, and gradient formulas.

## Next work, in priority order

1. Re-run the test suite after any change and preserve the current numerical equivalence checks.
2. Add independent replicates and Monte Carlo uncertainty to the valid k=12 pilot comparisons. Start with 24x1, m=12 and 96x12, m=12; keep each command below the 0.5-hour budget.
3. Decide a fixed, predeclared tuning grid and pilot length before the final runs. Record the cost of mass-matrix estimation in a separate setup-cost column and in an amortized measure.
4. Complete Sweep A only after explicit approval. The manuscript lattice is cyclic 24x1, no padding, k=12; sweep m over the intended well sizes.
5. Redesign Sweep B using only scientifically allowed `(lattice, k)` combinations. Never restore 96x12, k=24 merely because it was run once.
6. Write `reports/notes/adjoint_gradient.tex` (Phase 7), including the corrected complexity, the aliasing bug, the mass-matrix bug, and the finite-difference verification.
7. Implement the wrapped covariance and constructor-time DFT validity check only as a separately reviewed change; the current note recommends it but this branch does not yet alter the model.
8. Only after the evidence is stable, update the manuscript and Reviewer 3 response.

## Known cautions

- A constant chain can make the legacy ESS estimator return approximately the sample count; always apply the variance/movement guard in `omega_ess`.
- `cum_ar` is not populated reliably; summaries use the mean of `acc_iter`.
- Preliminary tuning and final evaluation currently reuse the same synthetic dataset, which is appropriate for algorithm comparison but does not measure across-dataset variability.
- The posterior mass pilot is an additional computational setup cost and must be disclosed.
- Full Sweep A/B calculations remain deferred; no handoff text should imply they were completed.