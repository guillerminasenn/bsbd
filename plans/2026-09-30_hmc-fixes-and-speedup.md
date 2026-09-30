# Plan: fix and speed up collapsed HMC, speed up Gibbs, re-run synthetic ESS experiments

Date: 2026-09-30
Branch: `hmc-fixes-and-speedup` (current code stays untouched on the current branch)
Status: planned, not started

## 1. Background

The manuscript *Bayesian Semi-Blind Deconvolution at Scale* (`reports/manuscript/...`) was rejected by JCGS
(`external/review_jcgs/editor_reviewer3.txt`). The relevant technical criticism is Reviewer 3, comment 7: the
benefits of HMC and of marginalisation should be assessed separately, and it is unclear when the cost of HMC
is justified at realistic sizes. The paper itself reports that HMC costs ~25x a Gibbs iteration for ~4x the
blur ESS (Table 1), so Gibbs wins on ESS/s.

The Claude Fable report (`external/andrew_claude/bsbd_report_source/`) explains why, with three findings that
I verified against the code:

1. **Aliasing bug in the gradient.** In `gradient_potential.py` (~line 219),
   `P = p.reshape((nv, nh), order='F')` is a *view* of `p`; `derivatives.py::gradient_SS_vectorized` then runs
   `np.fft.fft(P, axis=1, out=P)` etc. in place, so `p` is corrupted before it is used in `... @ p`. The
   Metropolis step uses the correct potential, so the chain is still exact, but leapfrog does not conserve
   energy, the adapted step size collapses (~300x), and ESS drops ~25x. Fix: `.copy()`.
2. **Forward-mode gradient.** The k free coordinates are handled by stacking (k, nv, nh), (k, m, m) and
   (k, nv, nv) tensors, giving O(k(n log n + m^3 + nv^2 nh)) per leapfrog step (the paper states a bound
   without the factor k). Every term is linear in the direction eigenvalue eps_i(f) = exp(-2 pi i f r_i / nv),
   so all k coordinates follow from one n-sized reduction and one length-nv FFT (reverse mode / adjoint).
   Cost drops to O(n log n + m^3 + nv^2 nh + nv^3) with no k-fold arrays; ~80x faster at paper size.
3. **Mass matrix bug.** `_create_momentum_object` reads `precondition` from the wrong dict level, so the
   mass matrix was always the identity; reading the right key would crash because `inv_R_wu_star` is a
   `csr_matrix`. The paper describes M = R_w^{-1} / sigma_w^2, which never ran.

I checked the adjoint derivation step by step (shift-evaluation lemma; tr(A^{-1} dA_i); tr(Y^{-1} dY_i) via
the lag histogram of M = L Y^{-1} L^T; the sum-of-squares term via K = V R_h V^T and column-wise cross
correlations). It is correct under the code's circulant convention (`linalg.circulant(c).T`, first row = base).

Additional issues found beyond the Fable list:

- `create_W` mutates its input (`w_star[np.abs(w_star) < 1e-11] = 0`), including the leapfrog position.
- `Covariance.__init__` allocates `np.eye(n)` (17 GB at n = 48000) to obtain one unit vector.
- `kinetic` / `grad_kinetic` re-factorise the mass matrix (Cholesky) at every leapfrog step.
- `MCMC.run(adapt_config=None)` raises.
- Legacy gradient raises `KeyError: 'L'` when m = 0 (no exact image observations); the potential handles it.
- `scipy.linalg.kron` was removed in SciPy >= 1.16 but is used in `covariance.py` and `gaussian.py`.
- `requirements.txt` is a conda export with local `file://` paths; not pip-installable.
- `_sample_circ` draws via `scipy.stats.multivariate_normal().rvs`, a different RNG stream from the rest.
- `mcmc_diagnostics.compute_autocorrelation` is an O(N * lag) Python loop.
- The paper cites Liu (2008) for ESS; the code implements Geyer's initial monotone sequence. Note for rewrite.

Gibbs hot path (cyclic topology; user asked that Gibbs be made as efficient as possible for a fair comparison):

- Image constraint correction (`step_gibbs.py` ~171-177): `subset_ABA` builds an nv x nv circulant;
  `subset_AB` builds a dense m x n matrix from nh circulants + `hstack` (~370 MB at paper size);
  `linalg.inv(ARAt)`; then `_correct_sample` computes the correction twice (inverse and Cholesky) and keeps
  the first. This is ~380 ms of the ~470 ms Gibbs iteration at paper size.
- Blur constraint correction (~94-109): `build_matrix_circ(base_R_cond)` builds the nv x nv matrix and a
  `copy.deepcopy`, to read a few columns.
- The wavelet full conditional (`sampling_utils.py`) is FFT-based and fine.

## 2. Decisions taken (with the user)

- Experiments: paper Sec. 5.1 sweep **and** a scaling study (ESS/s vs problem size); run here (synthetic data
  is small).
- Mass matrix: make `'prior'` work as the paper states **and** add a `'posterior'` option (k x k precision
  from a pilot run). `'prior'` stays the default.
- Keep three gradient variants selectable for comparison:
  `'legacy'` (today's code, byte-for-byte, only crash guards), `'legacy_fixed'` (aliasing fix), `'adjoint'`.
  Mass matrix options `None` / `'prior'` / `'posterior'` combine freely with them.
- Gibbs: speed up without changing its distribution; no legacy flag, equivalence covered by a test.
- Adjoint write-up: standalone LaTeX note `reports/notes/adjoint_gradient.tex`, reusable in the SM.
- Guideline: code and math as precise, concise, economical and simple as possible (repo goes to end users).
- Out of scope now: Euclidean topology, the real Egypt dataset, the manuscript rewrite (next stage).

## 3. Steps

### Phase 0 - Branch and scaffolding
1. `git checkout -b hmc-fixes-and-speedup`.
2. Replace `requirements.txt` with a minimal pinned list (numpy, scipy, matplotlib, pandas, pytest).
3. `tests/conftest.py`: fixture building small models like `setup()` in the Fable `check_gradient.py`
   (12x2 k=6 and 36x12 k=12; with a full well column and with m = 0).

### Phase 1 - Minimal correctness fixes on the legacy path (parallel with Phase 4)
4. `src/mcmc/update_blur_image/collapsedhmc/cyclic/gradient_potential.py`: kwarg `fix_aliasing=False`;
   when `True`, `P = p.reshape(...).copy()`. Guard the `constr_c` block so m = 0 does not raise.
   Default behaviour stays identical.
5. `src/mcmc/update_blur_image/collapsedhmc/initialize.py::_create_momentum_object`: read `precondition`
   from `mcmc_config['collapsed_hmc']`; densify `inv_R_wu_star`; store the Cholesky factor of the mass matrix
   once and use it in `kinetic` / `grad_kinetic` (`cyclic/utils.py`).
6. Small fixes: `covariance.py` `np.eye(1, n)`; `create_W` no input mutation; `MCMC.run` accepts
   `adapt_config=None`; `linalg.kron` -> `np.kron` in `covariance.py` and `gaussian.py`.

### Phase 2 - Adjoint gradient and potential (depends on 3)
7. New module `src/mcmc/update_blur_image/collapsedhmc/cyclic/adjoint.py`: `_common`, `potential_adjoint`,
   `gradient_potential_adjoint`. Based on Fable's `gradient_adjoint.py` but simplified:
   - Gamma^T v via one FFT cross-correlation of V with mu_c (drops the dense k x n
     `gamma_matrix_unconstrained_indexes` and all of `aux_derivatives`);
   - shifts r_i from the lattice wavelet positions instead of `bases_d_W0`;
   - vectorised cyclic diagonal sums (`kappa`).
   Variable names match the LaTeX note: `G1, G2, G3, Q, h1, q, M, mu, K, kappa, b, X, Ks`.
8. Dispatcher: `mcmc_config['collapsed_hmc']['gradient'] in {'legacy', 'legacy_fixed', 'adjoint'}` selects
   the (potential, gradient) pair in `leapfrog_collapsed_c` / `log_ar_hmc_collapsed_c`. For `'adjoint'`,
   reuse `common` from the first/last gradient in the two potential evaluations. Build
   `wavelet_derivatives` / `aux_derivatives` tensors only for the legacy variants.

### Phase 3 - Mass matrix (depends on 5)
9. `precondition in {None, 'prior', 'posterior'}`. `'prior'`: M = R_w^{-1} / sigma_w^2. `'posterior'`:
   k x k `mass_matrix` supplied in the config, produced by `estimate_mass_matrix(w_samples, _w)` (inverse of
   the regularised sample covariance of the free coordinates from a pilot chain). No online adaptation.

### Phase 4 - Gibbs speed-ups, exactness-preserving (parallel with Phase 1)
10. Cyclic image correction in `step_gibbs.py`: `ARAt` by gathering `base_R_cond[:, 0]` at
    `(rows[None, :] - rows[:, None]) % nv`; `RAt @ z` by scattering `z` onto `well_coords_vec` and one
    `multiply_matrix_vector_circ`; one Cholesky solve. Remove `subset_AB`/`subset_ABA`/`linalg.inv` from the
    per-iteration path and the duplicated computation in `sampling_utils._correct_sample`.
11. Blur correction: gather rolled base columns instead of `build_matrix_circ`; drop `deepcopy`.
    `_sample_circ`: `np.random.standard_normal` instead of scipy `rvs`. Audit `_update_ss` and
    `step_d_update` for dense work.

### Phase 5 - Tests (depends on 4-11)
12. `tests/test_gradient.py`: central finite differences of the legacy `potential` vs the three variants
    (m = 0 and full well, both lattices). Expect: `legacy` error O(1) (documents the bug),
    `legacy_fixed` and `adjoint` < 1e-6, `adjoint` vs `legacy_fixed` < 1e-10.
13. `tests/test_potential.py`: `potential_adjoint` == `potential` == dense log|Sigma| + quadratic form.
14. `tests/test_gibbs.py`: new vs old constraint correction give identical `c_star`, `w_star` for the same
    unconstrained draw.
15. `tests/test_mass_matrix.py`: momentum sampling, `kinetic`, `grad_kinetic` consistent with M for the three
    options.

### Phase 6 - Experiment runner and ESS tables (depends on Phase 5)
16. `scripts/run_experiment.py` (argparse): lattice `(nv_obs, nh_obs, k, mv, mh)`, `m`, algorithm, gradient
    variant, precondition, `L`, adaptation, `N`, burn-in, seed, init (`true`/`prior`). Reuses
    `build_model` / `get_image_well_subset` / `run_sampler` logic from `notebooks/run.ipynb` and the existing
    save layout `estimations/<folder_id>/...`.
17. `scripts/summarize_ess.py`: loads chains via `read_samples`; ESS (existing Geyer estimator, ACF via FFT),
    MSJD, ms/iter from `stats`, ESS/s; per parameter group (omega, c_u*, sigma_c^2, sigma_w^2, zeta; median
    and 2.5/97.5 pct) -> CSV + LaTeX in `reports/<sweep>/tables/`.
18. **Sweep A (paper Sec. 5.1):** cyclic 24x1, code `k=12` (10 free coordinates), m in {0, 2, ..., 24}
    centred, `L=40`, `marc` adaptation at 0.5, prior init, 45k post-burn-in. Gibbs and HMC-adjoint-prior for
    all 13 m; remaining variants (legacy, legacy_fixed x identity/prior/posterior) on m in {0, 12, 24} at 20k.
19. **Sweep B (scaling, full well column, `L=10`):** 36x12 (k=12), 90x24 (k=30), 200x50 (k=60), and paper
    size 480x100 (k=126) for ms/iter only. Gibbs vs each HMC variant; report ms/iter, ESS(omega) min/median,
    ESS/s.

### Phase 7 - Write-up
20. `reports/notes/adjoint_gradient.tex`: what "adjoint" means here (reverse mode:
    grad_omega U = J_{W0}^T grad_{W0} f; the transpose of omega -> W0 for a circulant is "evaluate at the
    shifts r_i", one FFT); conventions; the shift-evaluation lemma; the three gradient terms; complexity table
    correcting the paper's p. 23 claim; which mass matrix actually ran; verification results; map from
    symbols to code variables.

## 4. Relevant files

- `src/mcmc/update_blur_image/collapsedhmc/cyclic/gradient_potential.py`, `derivatives.py` - legacy gradient.
- `src/mcmc/update_blur_image/collapsedhmc/cyclic/potential.py`, `efficient_utils.py` -
  `compute_eigenvalues_A_invA_Z_B`, `project_down_with_Ac` / `project_up_with_Ac` reused by the adjoint.
- `src/mcmc/update_blur_image/collapsedhmc/cyclic/utils.py` - `leapfrog_collapsed_c`,
  `log_ar_hmc_collapsed_c`, `kinetic`, `grad_kinetic`.
- `src/mcmc/update_blur_image/collapsedhmc/initialize.py` - `_create_momentum_object`,
  `_initialize_collapsed_hmc`.
- `src/mcmc/update_blur_image/gibbs/step_gibbs.py`, `src/utils/sampling_utils.py`, `src/utils/model_utils.py`
  (`subset_AB`, `subset_ABA`, `create_W`) - Gibbs.
- `src/classes/covariance.py`, `src/classes/gaussian.py`, `src/mcmc/mcmc.py` - small fixes.
- `src/utils/mcmc_diagnostics.py`, `src/utils/file_utils.py` - ESS/MSJD, chain loading.
- New: `cyclic/adjoint.py`, `mass_matrix.py`, `tests/`, `scripts/run_experiment.py`,
  `scripts/summarize_ess.py`, `reports/notes/adjoint_gradient.tex`.
- Reference: `external/andrew_claude/bsbd_report_source/bsbd_report_source/{gradient_adjoint.py,
  check_gradient.py, bsbd_minimal_fixes.patch, bsbd_marginal_HMC_diagnosis.tex}`.

## 5. Verification

1. `pytest tests/` passes; FD relative errors: legacy ~1-6, legacy_fixed / adjoint <~ 1e-7,
   adjoint vs legacy_fixed <~ 1e-14 (reproduces Fable's table).
2. Same-seed chains with `legacy_fixed` and `adjoint` produce identical accept/reject sequences.
   Gibbs before/after speed-up produce identical `c_star` / `w_star` chains (<= 1e-10).
3. Timing table per gradient variant and Gibbs old/new at n = 432, 2160, 48000
   (expect ~80x gradient speed-up at paper size; Gibbs image update from ~380 ms to tens of ms).
4. Sweep A reproduces the qualitative Sec. 5.1 picture (bimodality for m <= 4, HMC advantage there);
   ESS/s of adjoint HMC vs Gibbs tabulated for all m.
5. Sweep B shows the ESS/s crossover vs problem size - the evidence Reviewer 3 (comment 7) asked for.

## 6. Open items to settle before Phase 6

1. **Lattice spec for Sweep A.** Paper text: "cyclic 24x1, no padding". Figure filenames:
   `k12_nv24_nh1_mv12`. Default to `Lattice(24, 1, 12, mv=0, mh=0)` per the text unless the user confirms
   `nv_ava=12, mv=12`.
2. **Budget.** With `L=40`, one HMC iteration on the 24x1 problem costs ~50-70 ms (41 gradients), so 50k
   iterations ~ 40-60 min per chain; 13 m-values x 6 HMC variants ~ 2 days sequential. Hence the split in
   step 18. Alternatives: all variants on all m at 15k iterations; or parallelise chains across processes.
3. **"Benefit of marginalisation alone" (R3 c.7)** would need a non-collapsed HMC on (omega, c) or a
   marginal Gibbs/MALA baseline. Excluded for now; the runner should not preclude adding it.

## 7. Key facts for later (so no re-exploration is needed)

- Circulant convention: `linalg.circulant(base).T` -> first row is the base; `C[p, q] = base[(q - p) mod nv]`.
- Eigenvalues arrays (`eigenvalues_*`) are `fft2(base)`; `lambda_{M^T} = conj(lambda_M)`;
  `lambda_{MN} = lambda_M * lambda_N`.
- Blur: `W0 = circ(P w*)`, `W = I_nh (x) W0`; d W0 / d w_i = circ(e_{r_i}); eps_i(f) = exp(-2 pi i f r_i / nv).
- Collapsed likelihood: A = W Sigma_c W^T + Sigma_d (BCCB); Z = Sigma_c W^T A^{-1} W Sigma_c;
  Y = I_m - L^T A_c Z A_c^T L / sigma_c^2 with L L^T = (A_c R_c A_c^T)^{-1};
  log|Sigma_{d|w}| = log|A| + log|Y|; Sigma^{-1} = A^{-1} + B A_c^T S^{-1} A_c B^T, B = A^{-1} W Sigma_c,
  S = A_c (Sigma_c - Z) A_c^T. Well pixels are one column, so A_c C A_c^T for BCCB C is a gather of the
  first base column at row lags.
- Config layout: `mcmc_config['collapsed_hmc'] = {epsilon, L, sigma2p, precondition, p_collapsed_hmc}`;
  `adapt_config['collapsed_hmc'][{'epsilon','L'}]` with types `None | 'marc' | 'rosenthal' | 'random'`.
- Notebook hyperparameters: alpha_w=2.01, beta_w=10, alpha_c=2.00001, beta_c=1/500, alpha_zeta=3,
  beta_zeta=0.1, zeta_init=0.05, rho_w=rho_v=rho_h=1.5, seed=20.
- Output layout: `estimations/<folder_id>/<algorithm>/constr_w_constr_c/<model_filename>_<alg>_N<N>_runID_<id>/`
  with `<name>.pkl`, `<name>_stats.pkl`, `samples/`; loaders `read_sampler_object`, `read_stats`,
  `read_samples` in `src/utils/file_utils.py`.
- Paper real example: 330x50 observed, margins (150, 50) -> 480x100, k=126, m=330, L=10; Gibbs 0.47 s/iter,
  HMC 13 s/iter; ESS(omega) 32 vs 133 out of 55k.
