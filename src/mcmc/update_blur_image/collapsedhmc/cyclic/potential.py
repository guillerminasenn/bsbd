"""Potential. FFT-based algebra.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions:
---------
potential
"""

# Standard library imports
import math

# Third-party library imports
import numpy as np
from scipy import linalg

# Local library imports
from src.utils.model_utils import create_W, subset_AB, subset_ABA
from src.mcmc.update_blur_image.collapsedhmc.cyclic.efficient_utils import compute_bases
from src.utils.efficient_algebra_utils import transpose_base_circ, multiply_matrix_vector_circ
from src.utils.math_utils import compute_log_det_circ, compute_log_det

pp = 4

def potential(sampler, i, w_star, verbose=False):
    """
    NOTE: Some quantities are already computed in the gradient, like Y, detY, invY, etc, 
    for the starting and end values of the trajectory. 
    It'd be more efficient to pass them as arguments, but this is not done for now.
    """
    
    # Extract from sampler
    _lik = sampler.par_objs['d']
    _c = sampler.par_objs['c']
    _w = sampler.par_objs['w']
    
    # Extract current state
    d_star = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
    sigma2w = sampler.theta['sigma2w'][0, i + 1].item()
    sigma2c = sampler.theta['sigma2c'][0, i + 1].item()

    # Extract dimensions
    n = _lik.lattice.n
    nv = _lik.lattice.nv
    if verbose: print(f'\nComputing the potential at w_star (center): \n{np.round(w_star[(nv//2-5):(nv//2+5), :], pp)}')

    # Contribution of the prior p(w*) to the potential
    if _w.wavelet_constraints['nr_constraints'] > 0:
        nr_constr = _w.wavelet_constraints['nr_constraints']
        term1 = 0.5 * (nv - nr_constr) * np.log(2 * math.pi)
        term2 = 0.5 * (nv - nr_constr) * np.log(sigma2w) + 0.5 * _w.wavelet_constraints['logdet_R_wu_star']
        mean_w_star = _w.wavelet_constraints['mean_w_star']
        w_centered = w_star - mean_w_star
        
        # With Cholesky and on wu
        unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
        w_star_full = w_star[unconstrained_indices, :].reshape(-1, 1)
        mean_w_star_full = mean_w_star[unconstrained_indices, :].reshape(-1, 1)
        w_star_centered_full = w_star_full - mean_w_star_full
        chol_R_wu_star = _w.wavelet_constraints['chol_R_wu_star']
        inv_chol_wu = linalg.solve_triangular(chol_R_wu_star, w_star_centered_full, lower=True)
        term3 = 0.5 * (inv_chol_wu.T @ inv_chol_wu)[0, 0] / sigma2w

    else: 
        term1 = 0.5 * nv * np.log(2 * math.pi)
        logdet_R_w = _w.Sigma.logdet_R
        term2 = 0.5 * nv * np.log(sigma2w) + 0.5 * logdet_R_w
        inv_cov = _w.Sigma.Q / sigma2w
        term3 = 0.5 * ((w_star.T) @ (inv_cov @ w_star))[0, 0]
            
    U_q_w = term1 + term2 + term3
    if verbose:
        print('U(q). Contribution from the prior: term1={0}, term2={1}, term3={2}.'.format(term1, term2, term3))

    # Contribution of the collapsed likelihood p(d|w*) to the potential
        
    # Blur-kernel wavelet convolutional matrix
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star) 

    # A
    base_A, base_inv_A, base_Z, base_B = compute_bases(sampler, i, base_W, base_WT, verbose)
    print(f"Computing logdet_A in the potential:") if verbose else None
    logdet_A = compute_log_det_circ(base_A) 

    # If c is constrained
    constr_c = _c.reflectivity_constraints['constr']
    if constr_c:

        n_w = _c.reflectivity_constraints['nr_constraints']
        rows_cyclic = _c.lattice.well_positions['rows_cyclic']
        
        # Constrained image reflectivity mean and covariance
        mean_c = _c.reflectivity_constraints['mean_c_star'] # length-n vector mu_c* and 
        L = _c.reflectivity_constraints['L']
        diagonal_block_Z = linalg.circulant(base_Z[:, 0]).T
        AZAt_temp = diagonal_block_Z[rows_cyclic, :] 
        AZAt = AZAt_temp[:, rows_cyclic]
        unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
        print(f"rows_cyclic = {rows_cyclic}, unconstrained indices = {unconstrained_indices}") if verbose else None
        print(f"AZAt.shape = {AZAt.shape}, AZAt (low dims) = \n{AZAt[:5, :5]}") if verbose else None
        print(f"det(AZAt) with linalg: {np.linalg.det(AZAt)}") if verbose else None
        print(f"AZAt symmetric? max diff = {np.max(np.abs(AZAt.T - AZAt))}") if verbose else None

        # new with Cholesky
        if np.round(np.sum(AZAt), 8)==0:
            print("AZAt is zero matrix.") if verbose else None
            Y = np.eye(n_w)
            logdet_Y = 0
        else:
            AZAt_lower_cholesky = linalg.cholesky(AZAt, lower=True)
            Lt_AZAt_lower_cholesky = L.T @ AZAt_lower_cholesky
            Lt_AZAt_L = Lt_AZAt_lower_cholesky @ (Lt_AZAt_lower_cholesky.T)
            Y = np.eye(n_w) - Lt_AZAt_L / sigma2c
            print(f'Y (low dims) = \n{Y[:5, :5]}') if verbose else None
            
            print(f"Computing logdet_Y...") if verbose else None
            logdet_Y = compute_log_det(Y)
            print(f"In potential, logdet_Y computed with compute_log_det = {logdet_Y}") if verbose else None
        
    else:
        mean_c = _c.mean
        logdet_Y = 0
        
    # Mean of the Gaussian collapsed likelihood
    mean_lik = multiply_matrix_vector_circ(base_W, mean_c)
    d_centered = d_star - mean_lik
    
    # Compute log-determinant of Sigma_dw
    logdet_Sigma_dw = logdet_Y + logdet_A

    # Compute sum of squares , old
    inv_A_d = multiply_matrix_vector_circ(base_inv_A, d_centered) # 
    ss_term1 = (d_centered.T) @ inv_A_d


    ss_term2 = 0
    if constr_c:
        base_BT = transpose_base_circ(base_B)
        ABt = subset_AB(base_BT, _c)
        base_Sigma_c = sigma2c * _c.Sigma.base_R
        base_matrix = base_Sigma_c - base_Z
        
        S = subset_ABA(base_matrix, _c)
        ABt_d = ABt @ d_centered # traditional algebra, no WA

        # Cholesky with solve_triangular
        S_lower_cholesky = linalg.cholesky(S, lower=True)
        S_lower_cholesky_inv_ABt_d = linalg.solve_triangular(S_lower_cholesky, ABt_d, lower=True)
        ss_term2 = (S_lower_cholesky_inv_ABt_d.T) @ S_lower_cholesky_inv_ABt_d

    ss = ss_term1 + ss_term2 
            
    # Evaluate contribution to the potential
    term1 = 0.5 * n * np.log(2 * math.pi)   
    term2 = 0.5 * logdet_Sigma_dw
    term3 = 0.5 * ss
    U_q_d = (term1 + term2 + term3)[0, 0]
    
    # Potential
    U = U_q_d + U_q_w
    if verbose:
        print('U(q). Contribution from the collapsed lik: term1={0}, term2={1}, term3={2}.'.format(term1, term2, term3))
        print(f'U(q). Potential: U_q_d={np.round(U_q_d, pp)}, U_q_w={np.round(U_q_w, pp)} ===> U={np.round(U, pp)}')

    return U
