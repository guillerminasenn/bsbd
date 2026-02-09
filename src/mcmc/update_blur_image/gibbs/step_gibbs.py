"""Gibbs updates for blur-kernel wavelet `w` and image reflectivity `c`.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.
"""

# Third-party library imports
import copy

import numpy as np
from scipy import linalg

# Algorithm imports
from src.utils.efficient_algebra_utils import build_matrix_circ
from src.utils.model_utils import subset_AB, subset_ABA
from src.utils.sampling_utils import _correct_sample, _sample_circ, _sample_scipy

def step_gibbs(sampler, i, update_blur=True, update_image=True):
    """Gibbs update for the blur-kernel wavelet and image reflectivity.

    Parameters
    ----------
    sampler : MCMC
        Sampler with current state, priors, and likelihood objects.
    i : int
        Iteration index (within chunk).
    update_blur : bool
        Whether to update the blur-kernel wavelet $w$ (stored as `w_star`).
    update_image : bool
        Whether to update the image reflectivity $c$ (stored as `c_star`).

        Notes
        -----
        This step relies on `recompute_mean_R` methods implemented on the Gaussian
        prior classes:
        - `Wavelet.recompute_mean_R` computes the conditional mean and correlation
            of $p(w\mid c, d)$ given the current sampler state.
        - `Reflectivity.recompute_mean_R` computes the conditional mean and
            correlation of $p(c\mid w, d)$ given the current sampler state.
        Both methods return `(mean_cond, base_R_cond, R_cond)` where `base_R_cond`
        is the circulant base (cyclic topology) and `R_cond` is the explicit
        correlation matrix (Euclidean topology).
    """
    verbose = sampler.mcmc_config['verbose']

    if verbose:
        print(f"In step_gibbs: update_blur={update_blur}, update_image={update_image}")

    # Gaussian objects
    _image = sampler.par_objs['c']
    _blur = sampler.par_objs['w']

    # Current state
    sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
    sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
    zeta = sampler.theta['zeta'][:, i + 1].item()
    psi = sampler.model.aux['psi']
    sigma2d_computed = psi * sigma2c * sigma2w * zeta
    sigma2d = sampler.aux['sigma2d'][:, i + 1].item()
    if verbose:
        print(
            "At the start of the Gibbs step: "
            f"sigma2c={sigma2c}, sigma2w={sigma2w}, zeta={zeta}, psi={psi}, "
            f"sigma2d={sigma2d}, sigma2d_computed={sigma2d_computed}"
        )

    # ---- Blur-kernel wavelet update ----
    if update_blur:
        # Conditional mean and correlation for p(w | c, d)
        # (computed from the current sampler state)
        mean_cond, base_R_cond, R_cond = _blur.recompute_mean_R(sampler, i)

        # Sample from the unconstrained Gaussian
        if sampler.lattice.topology == 'E':
            w_new = _sample_scipy(mean_cond, sigma2w, R_cond)
        elif sampler.lattice.topology == 'C':
            w_new = _sample_circ(mean_cond, sigma2w, base_R_cond)

        if verbose:
            center = w_new[(sampler.lattice.nv // 2 - 5):(sampler.lattice.nv // 2 + 5), :]
            print(f"Unconstrained blur sample (center):\n{center}\n")

        # Save the unconstrained sample (blur)
        sampler.aux['w'][:, i + 1] = np.squeeze(w_new)

        # Apply constraints if present
        w_new_star = copy.deepcopy(w_new)
        if _blur.wavelet_constraints['nr_constraints'] > 0:
            if verbose:
                print("Applying blur constraints...")

            # Load constraints
            b = _blur.wavelet_constraints['b']
            A = _blur.wavelet_constraints['A']

            # Compute RAt and ARAt
            if R_cond is None:
                # NOTE: can be done more efficiently with subset_AB
                R_cond = build_matrix_circ(base_R_cond)

            # Subset non-zero columns
            non_zero_cols = A.getnnz(axis=0) > 0
            RAt = R_cond[:, non_zero_cols]
            ARAt = RAt[non_zero_cols, :]

            # Correct the sample (same approach for both Euclidean and cyclic topologies because small dims)
            Aw_b = w_new[non_zero_cols, :].reshape(-1, 1) - b.reshape(-1, 1)  # Subset Aw_b to non-zero rows only
            cholesky_ARAt = linalg.cholesky(ARAt, lower=True)
            vector1 = linalg.solve_triangular(cholesky_ARAt, Aw_b, lower=True)
            vector2 = linalg.solve_triangular(cholesky_ARAt.T, vector1, lower=False)
            w_correction = RAt @ vector2
            w_new_star = w_new - w_correction  
            if verbose:
                center = w_new_star[(sampler.lattice.nv // 2 - 5):(sampler.lattice.nv // 2 + 5), :]
                print(f"Corrected blur sample (center):\n{center}\n")
            
            # Reshape x_star into (1,1) np.array if it only contains one element
            if not w_new_star.shape:
                w_new_star = w_new_star.reshape((1, 1))     

        # Save the corrected sample
        sampler.theta['w_star'][:, i + 1] = np.squeeze(w_new_star)
    
    # Print some values
    if verbose:
        center_prev = sampler.theta['w_star'][(sampler.lattice.nv // 2 - 5):(sampler.lattice.nv // 2 + 5), i]
        center_next = sampler.theta['w_star'][(sampler.lattice.nv // 2 - 5):(sampler.lattice.nv // 2 + 5), i + 1]
        print(f"Blur center (i): {center_prev.flatten()}")
        print(f"Blur center (i+1): {center_next.flatten()}")
        print(
            "After blur update: "
            f"sigma2c={sigma2c}, sigma2w={sigma2w}, zeta={zeta}, psi={psi}, "
            f"sigma2d={sigma2d}, sigma2d_computed={sigma2d_computed}"
        )

    # ---- Image reflectivity update ----
    if update_image:
        # Conditional mean and correlation for p(c | w, d)
        # (computed from the current sampler state)
        mean_cond, base_R_cond, R_cond = _image.recompute_mean_R(sampler, i)

        # Sample from the unconstrained Gaussian
        if sampler.lattice.topology == 'E':
            c_new = _sample_scipy(mean_cond, sigma2c, R_cond, verbose)
        elif sampler.lattice.topology == 'C':
            c_new = _sample_circ(mean_cond, sigma2c, base_R_cond, verbose)

        # Save the unconstrained sample (image)
        sampler.aux['c'][:, i + 1] = np.squeeze(c_new)

        # Apply constraints if present
        c_new_star = c_new
        if _image.reflectivity_constraints['constr']:

            # Load constraints
            b = _image.reflectivity_constraints['b']
            A = _image.reflectivity_constraints['A_co']

            # Correct sample (Euclidean vs cyclic)
            if sampler.lattice.topology == 'E':

                # Compute RAt and ARAt
                RAt = (A.dot(R_cond)).T
                ARAt = A.dot(RAt)
                ARAt = 0.5 * (ARAt + ARAt.T)  # Ensure symmetry
                inv_ARAt = linalg.inv(ARAt)
                inv_ARAt = 0.5 * (inv_ARAt + inv_ARAt.T)  # Ensure symmetry

                # Correct the sample
                c_new_star = _correct_sample(c_new, b, A, inv_ARAt, RAt, verbose)
            
            elif sampler.lattice.topology == 'C':
                if verbose:
                    print("Correcting image sample on cyclic lattice...")

                # Compute RAt and ARAt
                ARAt = subset_ABA(base_R_cond, _image)
                ARAt = 0.5 * (ARAt + ARAt.T)  # Ensure symmetry
                RAt = subset_AB(base_R_cond, _image).T
                inv_ARAt = linalg.inv(ARAt)  
                inv_ARAt = 0.5 * (inv_ARAt + inv_ARAt.T)  # Ensure symmetry              

                # Correct the sample
                c_new_star = _correct_sample(c_new, b, A, inv_ARAt, RAt, verbose)
                if verbose:
                    print(f"Corrected reflectivity sample (first 5): {c_new_star[:5]}")   

        # Save the corrected sample
        sampler.theta['c_star'][:, i + 1] = np.squeeze(c_new_star)

    # Print some values
    if verbose:
        print(f"Image head (i): {sampler.theta['c_star'][:5, i].flatten()}")
        print(f"Image head (i+1): {sampler.theta['c_star'][:5, i + 1].flatten()}")
        if sampler.theta_init is not None:
            print(f"Image init (head): {sampler.theta_init['c_star'][:5].flatten()}")
        if sampler.theta_true is not None:
            print(f"Image true (head): {sampler.theta_true['c_star'][:5].flatten()}")
        print(
            "After image update: "
            f"sigma2c={sigma2c}, sigma2w={sigma2w}, zeta={zeta}, psi={psi}, "
            f"sigma2d={sigma2d}, sigma2d_computed={sigma2d_computed}"
        )
     