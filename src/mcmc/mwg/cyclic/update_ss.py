"""Update the sum of squares for the cyclic MCMC sampler.
Functions:
- _update_ss: Main function to update the sum of squares.
- compute_likelihood_ss: Computes the likelihood sum of squares.
- compute_reflectivity_ss: Computes the reflectivity sum of squares.
- compute_wavelet_ss: Computes the wavelet sum of squares.
"""

# third-party imports
import numpy as np
from scipy import linalg

# Local library imports
from src.utils.model_utils import create_W
from src.utils.efficient_algebra_utils import multiply_matrix_vector_circ

def _update_ss(sampler, i):
    """Update sum of squares used to recompute the scale parameter in the 
    variance full conditionals. Cyclic case, FFT-based matrix algebra.
    
    Parameters
    ----------
    sampler: MCMC object
        Contains sampler configuration, data, and starting value.
    i: int
        Iteration number.

    Returns:
    -------
    Updates the following sums of squares in sampler.aux:
    - likelihood_ss_unconstrained: (d* - W*c*)' Q_d (d* - W*c*)
    - likelihood_ss_subset: (Ad* - AW*c*)' inv(AR_d At) (Ad* - AW*c*), or 0 if A=None because no constraints
    - likelihood_ss_constrained: (d - W*c*)' Q_d (d - W*c*) - (Ad - AW*c*)' inv(AR_d At) (Ad - AW*c*)
    """

    verbose = sampler.mcmc_config.get('verbose')
    if verbose:
        print(f"Current sum of squares at iteration {i}")
        print(f"  likelihood_ss_unconstrained: {sampler.aux['likelihood_ss_unconstrained'][:, i + 1]}")
        print(f"  likelihood_ss_subset: {sampler.aux['likelihood_ss_subset'][:, i + 1]}")
        print(f"  likelihood_ss_constrained: {sampler.aux['likelihood_ss_constrained'][:, i + 1]}\n")
        print(f"  reflectivity_ss_unconstrained: {sampler.aux['reflectivity_ss_unconstrained'][:, i + 1]}")
        print(f"  reflectivity_ss_subset: {sampler.aux['reflectivity_ss_subset'][:, i + 1]}")
        print(f"  reflectivity_ss_constrained: {sampler.aux['reflectivity_ss_constrained'][:, i + 1]}\n")
        print(f"  wavelet_ss_unconstrained: {sampler.aux['wavelet_ss_unconstrained'][:, i + 1]}")
        print(f"  wavelet_ss_subset: {sampler.aux['wavelet_ss_subset'][:, i + 1]}")
        print(f"  wavelet_ss_constrained: {sampler.aux['wavelet_ss_constrained'][:, i + 1]}\n")

    compute_likelihood_ss(sampler, i)
    compute_reflectivity_ss(sampler, i)
    compute_wavelet_ss(sampler, i)

    # Print debug information
    if verbose:
        print(f"  likelihood_ss_unconstrained: {sampler.aux['likelihood_ss_unconstrained'][:, i + 1]}")
        print(f"  likelihood_ss_subset: {sampler.aux['likelihood_ss_subset'][:, i + 1]}")
        print(f"  likelihood_ss_constrained: {sampler.aux['likelihood_ss_constrained'][:, i + 1]}\n")
        print(f"  reflectivity_ss_unconstrained: {sampler.aux['reflectivity_ss_unconstrained'][:, i + 1]}")
        print(f"  reflectivity_ss_subset: {sampler.aux['reflectivity_ss_subset'][:, i + 1]}")
        print(f"  reflectivity_ss_constrained: {sampler.aux['reflectivity_ss_constrained'][:, i + 1]}\n")
        print(f"  wavelet_ss_unconstrained: {sampler.aux['wavelet_ss_unconstrained'][:, i + 1]}")
        print(f"  wavelet_ss_subset: {sampler.aux['wavelet_ss_subset'][:, i + 1]}")
        print(f"  wavelet_ss_constrained: {sampler.aux['wavelet_ss_constrained'][:, i + 1]}\n")
        print(f"Updated likelihood sum of squares at iteration {i}.")


def compute_likelihood_ss(sampler, i):
    """Compute the likelihood sum of squares for the current iteration.
    """

    verbose = sampler.mcmc_config.get('verbose')

    # Load current state
    w_star = sampler.theta['w_star'][:, i + 1].reshape(-1, 1)   
    c_star = sampler.theta['c_star'][:, i + 1].reshape(-1, 1) 
    d_star = sampler.theta['d_star'][:, i + 1].reshape(-1, 1) # constrained extended data
    
    # Extract Gaussian objects 
    _d = sampler.par_objs['d']
    base_Q_d = _d.Sigma.base_Q # base of observational inverse correlation matrix

    # Compute the mean of the likelihood
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(sampler.lattice, w_star)
    mean_lik = multiply_matrix_vector_circ(base_W, c_star)
    if verbose:
        print(f"min(w_star) = {np.min(w_star)}, max(w_star) = {np.max(w_star)}")
        print(f"min(c_star) = {np.min(c_star)}, max(c_star) = {np.max(c_star)}")
        print(f"min(mean_lik) = {np.min(mean_lik)}, max(mean_lik) = {np.max(mean_lik)}")
        print(f"np.sum(mean_lik) = {np.sum(mean_lik)}") 

    # Centered d vector
    # d_centered = d - mean_lik # unconstrained d; before 09.08.2024
    d_centered = d_star - mean_lik # constrained d

    # Likelihood unconstrained SS
    Q_d_d = multiply_matrix_vector_circ(base_Q_d, d_centered)
    likelihood_ss_unconstrained = d_centered.T @ Q_d_d
    sampler.aux['likelihood_ss_unconstrained'][:, i + 1] = likelihood_ss_unconstrained
    print(f"The likelihood SS unconstrained = \n{likelihood_ss_unconstrained}") if verbose else None

    # if verbose:
    #     print(f"The likeihood SS unconstrained = \n{likelihood_ss_unconstrained}")
        # print(f"Verifying with traditional algebra: ")
        # Q_d = build_matrix_circ(base_Q_d)
        # likelihood_ss_unconstrained_naive = d_centered.T @ Q_d @ d_centered
        # print(f"likelihood_ss_unconstrained_naive = \n{likelihood_ss_unconstrained_naive}")

    # # Likelihood subset SS
    # constr_SSD = sampler.mcmc_config.get('constr_SSD', False)
    # if _d.data_constraints['constr'] and constr_SSD:

    #     # print(f"Data is constrained. Computing likelihood subset SS...") if verbose else None

    #     # Extract the constraint matrices
    #     X_matrix = _d.data_constraints['X_matrix'] 
    #     U_matrix = _d.data_constraints['U_matrix'] 
    #     n_ava = sampler.lattice.n_ava # number of nodes in the observed lattice
    #     nv_ava = sampler.lattice.nv_ava 
    #     nh_ava = sampler.lattice.nh_ava 
        
    #     # Compute the subset centered matrix
    #     D = d.reshape((nv, nh), order='F')  
    #     UDXt = U_matrix @ D @ X_matrix.T
    #     vector1 = UDXt.reshape((n_ava, 1), order='F')

    #     C = c_star.reshape((nv, nh), order='F')
    #     W0 = build_matrix_circ(base_W0)
    #     UW0CXt = U_matrix @ W0 @ C @ X_matrix.T
    #     vector2 = UW0CXt.reshape((n_ava, 1), order='F')
    #     d_subset_centered = vector1 - vector2

    #     # Compute the likelihood SS subset
    #     D_subset_centered = d_subset_centered.reshape((nv_ava, nh_ava), order='F')
    #     inv_X_R_d_h_Xt = _d.data_constraints['inv_X_R_d_h_Xt']
    #     inv_U_R_d_v_Ut = _d.data_constraints['inv_U_R_d_v_Ut']
    #     vector_right = D_subset_centered @ inv_X_R_d_h_Xt.T
    #     vector = inv_U_R_d_v_Ut @ vector_right
    #     vector = vector.reshape((n_ava, 1), order='F')
    #     likelihood_ss_subset = d_subset_centered.T @ vector
    #     # if verbose:
    #     #     print(f"shapes: d_subset_centered: {d_subset_centered.shape}, vector: {vector.shape}")
    #     #     print(f"The SS subset = \n{likelihood_ss_subset}")

    #     # if verbose:
    #     #     print(f"Verifying with traditional algebra:")
    #     #     A_d = sparse.kron(X_matrix, U_matrix)
    #     #     Q_subset = _d.data_constraints['inv_ARAt']
    #     #     d_subset = A_d @ d
    #     #     mean_subset = A_d @ mean_lik
    #     #     d_subset_centered = d_subset - mean_subset
    #     #     likelihood_ss_subset_naive = d_subset_centered.T @ Q_subset @ d_subset_centered
    #     #     print(f"likelihood_ss_subset_naive = \n{likelihood_ss_subset_naive}")

    #     # Unconstrained SSD with unconstrained d
    #     d_centered = d - mean_lik # unconstrained d; before 09.08.2024
    #     Q_d_d = multiply_matrix_vector_circ(base_Q_d, d_centered)
    #     likelihood_ss_unconstrained = d_centered.T @ Q_d_d

    # else:
    #     likelihood_ss_subset = 0  # No constraints, SS is zero
    #     # if verbose:
    #     #     print(f"Data is not constrained.")
    #     #     print(f"The SS subset = \n{likelihood_ss_subset}")
    # sampler.aux['likelihood_ss_subset'][:, i + 1] = likelihood_ss_subset

    # Likelihood constrained SS 
    # likelihood_ss_constrained = likelihood_ss_unconstrained - likelihood_ss_subset
    # sampler.aux['likelihood_ss_constrained'][:, i + 1] = likelihood_ss_constrained
    # if verbose:
    #     print(f"The likelihood SS constrained = \n{likelihood_ss_constrained}\n\n")

def compute_reflectivity_ss(sampler, i):
    """Compute the reflectivity sum of squares for the current iteration.
    """
    verbose = sampler.mcmc_config.get('verbose')

    # Load current state
    c = sampler.aux['c'][:, i + 1].reshape(-1, 1) # unconstrained extended reflectivity
    c_star = sampler.theta['c_star'][:, i + 1].reshape(-1, 1) # constrained reflectivity

    # Extract Gaussian objects 
    _c = sampler.par_objs['c']
    mean_c = _c.mean.reshape(-1, 1)  # Ensure mean_c is a column vector
    base_Q_c = _c.Sigma.base_Q # base of reflectivity inverse correlation matrix

    # Centered c vector
    # c_centered = c - mean_c # as it was before
    c_centered = c_star - mean_c # using the constrained reflectivity

    # Reflectivity unconstrained SS
    reflectivity_ss_unconstrained = c_centered.T @ multiply_matrix_vector_circ(base_Q_c, c_centered)
    # if verbose:
    #     print(f"The reflectivity SS unconstrained = \n{reflectivity_ss_unconstrained}")
    #     print(f"Verifying with traditional algebra: ")
    #     Q_c = build_matrix_circ(base_Q_c)
    #     reflectivity_ss_unconstrained = c_centered.T @ Q_c @ c_centered
    #     print(f"reflectivity_ss_unconstrained_naive = \n{reflectivity_ss_unconstrained}")
    sampler.aux['reflectivity_ss_unconstrained'][:, i + 1] = reflectivity_ss_unconstrained
    # if verbose:
    #     print(f"Reflectivity unconstrained SS = \n{reflectivity_ss_unconstrained}")

    # Reflectivity subset SS
    if _c.reflectivity_constraints['constr']:
        print(f"Reflectivity is constrained. Computing reflectivity subset SS...") if verbose else None
        A = _c.reflectivity_constraints['A_co'] # reflectivity constraint matrix
        inv_ARAt = _c.reflectivity_constraints['inv_ARAt'] # inverse of ARAt
        vector = A.dot(c) 
        right_term = inv_ARAt @ vector
        reflectivity_ss_subset = vector.T @ right_term
        # print(f"reflectivity_ss_subset = \n{reflectivity_ss_subset}") if verbose else None
        
        # if verbose:
        #     print(f"Verifying with traditional algebra:")
        #     c_subset = A @ c
        #     mean_c_subset = A @ mean_c
        #     c_subset_centered = c_subset - mean_c_subset
        #     Q_c_subset = _c.reflectivity_constraints['inv_ARAt']
        #     reflectivity_ss_subset = c_subset_centered.T @ Q_c_subset @ c_subset_centered
        #     print(f"reflectivity_ss_subset = \n{reflectivity_ss_subset}")

    else:
        reflectivity_ss_subset = 0  # No constraints, SS is zero
        # if verbose:
        #     print(f"Reflectivity is not constrained.")
        #     print(f"The SS subset = \n{reflectivity_ss_subset}")
    sampler.aux['reflectivity_ss_subset'][:, i + 1] = reflectivity_ss_subset

    # Reflectivity constrained SS 
    c_centered = c - mean_c # as it was before
    reflectivity_ss_unconstrained = c_centered.T @ multiply_matrix_vector_circ(base_Q_c, c_centered)
    reflectivity_ss_constrained = reflectivity_ss_unconstrained - reflectivity_ss_subset
    sampler.aux['reflectivity_ss_constrained'][:, i + 1] = reflectivity_ss_constrained
    # if verbose:
    #     print(f"The reflectivity SS constrained = \n{reflectivity_ss_constrained}\n")

def compute_wavelet_ss(sampler, i):
    """Compute the wavelet sum of squares for the current iteration.
    Distinguishes between collapsed and non-collapsed HMC algorithms, 
    since in the collapsed HMC case, we don't have access to the unconstrained
    wavelet sample w, just to the constrained sample w_star.
    """

    verbose = sampler.mcmc_config.get('verbose')

    # Extract Gaussian objects 
    _w = sampler.par_objs['w']
    base_Q_w = _w.Sigma.base_Q # base of wavelet inverse correlation matrix
    mean_w_star = _w.wavelet_constraints['mean_w_star']
    inv_R_w_star = _w.wavelet_constraints['inv_R_w_star']
    chol_R_wu_star = _w.wavelet_constraints['chol_R_wu_star']
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']

    if _w.wavelet_constraints['nr_constraints'] > 0:
        
        print(f"Wavelet is constrained with nr_constr={_w.wavelet_constraints['nr_constraints']}. Computing wavelet subset SS...") if verbose else None
        # Extract wavelet constraint matrix
        A = _w.wavelet_constraints['A'] 

        if sampler.mcmc_config['algorithm'] == 'collapsed_hmc':
            # In the collapsed HMC case, we don't have access to the unconstrained wavelet sample w
            print(f"Using collapsed HMC: cannot compute unconstrained or subset wavelet SS.") if verbose else None

            # Wavelet unconstrained SS
            wavelet_ss_unconstrained = None # not available in collapsed HMC
            sampler.aux['wavelet_ss_unconstrained'][:, i + 1] = wavelet_ss_unconstrained

            # Wavelet subset SS
            wavelet_ss_subset = None # not available in collapsed HMC
            sampler.aux['wavelet_ss_subset'][:, i + 1] = wavelet_ss_subset

            # Wavelet constrained SS (old)
            w_star = sampler.theta['w_star'][:, i + 1].reshape(-1, 1) # constrained sample
            print(f"w_star (center values) at i+1: \n{w_star[(sampler.lattice.nv//2-5):(sampler.lattice.nv//2 + 5), :]}\n") if verbose else None
            w_star_full = w_star[unconstrained_indices, :].reshape(-1, 1)
            mean_w_star_full = mean_w_star[unconstrained_indices, :].reshape(-1, 1)
            w_star_centered_full = w_star_full - mean_w_star_full
            # inv_R_w_star_full = inv_R_w_star[unconstrained_indices, :][:, unconstrained_indices] # old
            # wavelet_ss_constrained_full = w_star_centered_full.T @ inv_R_w_star_full @ w_star_centered_full
            inv_chol_wu = linalg.solve(chol_R_wu_star, w_star_centered_full, lower=True)
            wavelet_ss_constrained_full_chol = inv_chol_wu.T @ inv_chol_wu
            # print(f"Compared to HMC's method using the current value of w_star: \n{wavelet_ss_constrained_full}") if verbose else None
            # print(f"Compared to HMC's method using Cholesky of R_wu_star: \n{wavelet_ss_constrained_full_chol}\n") if verbose else None

            # # With the null space
            # base_Rv = _w.Sigma.base_Rv.flatten()
            # Rv = build_matrix_circ(base_Rv)
            # R_wu = Rv[unconstrained_indices, :][:, unconstrained_indices]
            # inv_R_wu = linalg.inv(R_wu)
            # inv_R_wu = (inv_R_wu.T + inv_R_wu) / 2 # Ensure symmetry
            # wavelet_ss_constrained_null = w_star_centered_full.T @ inv_R_wu @ w_star_centered_full
            # print(f"Compared to the null space method: \n{wavelet_ss_constrained_null}\n") if verbose else None

            sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_constrained_full_chol # go back to this
            # sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_constrained_null 

            if verbose:
                # print(f"min(inv_R_w_star_full) = {np.min(inv_R_w_star_full)}, max(inv_R_w_star_full) = {np.max(inv_R_w_star_full)}\n")
                print(f"SSW_constrained = {wavelet_ss_constrained_full_chol}\n")

        elif sampler.mcmc_config['algorithm'] == 'gibbs':
            
            print(f"Using Gibbs: We compute the unconstrained, subset, and constrained Wavelet SS, as their difference:") if verbose else None
            
            # Wavelet unconstrained SS
            w = sampler.aux['w'][:, i + 1].reshape(-1, 1) # unconstrained extended reflectivity
            print(f"Unconstrained wavelet w (center values) at i+1: \n{w[(sampler.lattice.nv//2-5):(sampler.lattice.nv//2 + 5), :]}\n") if verbose else None
            wavelet_ss_unconstrained = w.T @ multiply_matrix_vector_circ(base_Q_w, w)
            sampler.aux['wavelet_ss_unconstrained'][:, i + 1] = wavelet_ss_unconstrained

            # Wavelet subset SS
            non_zero_cols = A.getnnz(axis=0) > 0 # Extract indexes of non-zero columns of A
            w_subset = w[non_zero_cols, :].reshape(-1, 1)
            wavelet_ss_subset = w_subset.T @ _w.wavelet_constraints['inv_ARAt'] @ w_subset
            sampler.aux['wavelet_ss_subset'][:, i + 1] = wavelet_ss_subset

            # Wavelet constrained SS
            wavelet_ss_constrained = wavelet_ss_unconstrained - wavelet_ss_subset
            print(f"  unconstrained = {wavelet_ss_unconstrained}"
                  f"  subset = {wavelet_ss_subset}"
                  f"  constrained = {wavelet_ss_constrained}\n") if verbose else None
            
            # But we use HMC's method to compute the constrained SS:
            w_star = sampler.theta['w_star'][:, i + 1].reshape(-1, 1) # constrained sample
            print(f"w_star (center values) at i+1: \n{w_star[(sampler.lattice.nv//2-5):(sampler.lattice.nv//2 + 5), :]}\n") if verbose else None
            w_star_full = w_star[unconstrained_indices, :].reshape(-1, 1)
            mean_w_star_full = mean_w_star[unconstrained_indices, :].reshape(-1, 1)
            w_star_centered_full = w_star_full - mean_w_star_full
            inv_chol_wu = linalg.solve(chol_R_wu_star, w_star_centered_full, lower=True)
            wavelet_ss_constrained_full_chol = inv_chol_wu.T @ inv_chol_wu

            # # Old method using inv_R_w_star
            # inv_R_w_star_full = inv_R_w_star[unconstrained_indices, :][:, unconstrained_indices]
            # wavelet_ss_constrained_full = w_star_centered_full.T @ inv_R_w_star_full @ w_star_centered_full
            # print(f"Compared to HMC's method using the current value of w_star: \n{wavelet_ss_constrained_full}") if verbose else None
            
            # # With the null space
            # base_Rv = _w.Sigma.base_Rv
            # Rv = build_matrix_circ(base_Rv)
            # R_wu = Rv[unconstrained_indices, :][:, unconstrained_indices]
            # inv_R_wu = linalg.inv(R_wu)
            # inv_R_wu = (inv_R_wu.T + inv_R_wu) / 2 # Ensure symmetry
            # wavelet_ss_constrained_null = w_star_centered_full.T @ inv_R_wu @ w_star_centered_full
            # print(f"Compared to the null space method: \n{wavelet_ss_constrained_null}\n") if verbose else None

            sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_constrained_full_chol # go back to this
            # sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_constrained_null 
            
    else:
        # w_centered = w - mean_w # unnecesary because mean=0
        wavelet_ss_unconstrained = w.T @ multiply_matrix_vector_circ(base_Q_w, w)
        sampler.aux['wavelet_ss_subset'][:, i + 1] = 0
        sampler.aux['wavelet_ss_unconstrained'][:, i + 1] = wavelet_ss_unconstrained
        sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_unconstrained