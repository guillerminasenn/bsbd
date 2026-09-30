"""Gradient of the potential. Matrix operations are FFT-based.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions:
---------
gradient_potential
compute_gradient_for_index
"""

import time

# Third-party library imports
import numpy as np
from scipy import linalg
import scipy

from src.utils.model_utils import create_W, embed_wu, subset_ABA
from src.mcmc.update_blur_image.collapsedhmc.cyclic.efficient_utils import compute_Y_inv_det, compute_S, compute_eigenvalues_A_invA_Z_B, compute_r_s_p_Fourier_domain
from src.utils.efficient_algebra_utils import transpose_base_circ_vectorized

from src.mcmc.update_blur_image.collapsedhmc.cyclic.derivatives import gradient_SS_vectorized #, derivative_logdet_Sigma_dw_j_Fourier_domain, derivative_logdet_Sigma_dw_j_Fourier_domain_vectorized

pp = 6

def gradient_potential_Fourier_domain(sampler, i, w_star, fix_aliasing=False):
    """Gradient of the potential 

        p(w*|d) = p(w*) p(d|w*) / Z 
                = N(w*; \mu_w*, Sigma_w*) * N(d; W \mu_c, Sigma_dw),

    with

        Sigma_dw = W % Sigma_c* % W' + Sigma_d,

    allowing constraints w*=w|Ax=0 and c*=c|Ac=co. Vectorized and
    computed whenever possible in the Fourier domain.

    Params:
    ------
    sampler: Sampler object containing the MCMC configuration and parameters.
    i: int
        Current iteration index.
    w_star: np.array k x 1 (if cyclic, k = nv)
        Current constrained wavelet coefficients vector.
    fix_aliasing: bool (default=False)
        If True, copy p before reshaping it into P. With the default (legacy
        behaviour), P is a view of p and the in-place FFTs in
        gradient_SS_vectorized corrupt p before it is used in `... @ p`.
        
    Return:
    ------
    grad: np.array k x 1 
        Gradient of the potential evaluated at w_star.
    """
    verbose = sampler.mcmc_config['verbose']
    print(f'####### #######  Start of GradPotential (FFT) ####### ##############') if verbose else None       
    
    # 1. Extract configuration and pre-computed values from sampler

    ## Extract Gaussian objects
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    _c = sampler.par_objs['c']

    ## Extract configuration
    nv = _lik.lattice.nv
    nh = _lik.lattice.nh
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
    constr_c = _c.reflectivity_constraints['constr']

    ## Create empty structures to store quantities
    t0 = time.time()
    d_SS_1_part1_temp = sampler.aux_derivatives['d_SS_1_part1_temp']
    d_SS_1_part2_temp = sampler.aux_derivatives['d_SS_1_part2_temp']
    d_SS_2_vectorized_temp = sampler.aux_derivatives['d_SS_2_vectorized_temp']
    temp_array_nvk = sampler.aux_derivatives['temp_array_nvk']
    temp_array_knvnh = sampler.aux_derivatives['temp_array_knvnh']
    temp_array_knvnh_2 = sampler.aux_derivatives['temp_array_knvnh_2']
    temp_array_knvnh_3 = sampler.aux_derivatives['temp_array_knvnh_3']
    trace_invYdY = sampler.aux_derivatives['trace_invYdY']
    trace_invA_dA = sampler.aux_derivatives['trace_invA_dA']
    subset_base_dZ = sampler.aux_derivatives['subset_base_dZ']
    print('Initialization of temporary arrays completed in {:.4f} seconds.'.format(time.time() - t0)) if verbose else None

    ## Extract constrained wavelet mean and Cholesky decomposition of the correlation matrix 
    mean_w = _w.wavelet_constraints['mean_w_star'] 
    chol_R_wu_star = _w.wavelet_constraints['chol_R_wu_star']
    
    ## Extract precomputed wavelet derivative bases
    bases_d_W0 = sampler.par_objs['w'].wavelet_derivatives['bases_d_W0'][:, unconstrained_indices]
    row_indexes = np.argmax(bases_d_W0, axis=0)

    ## Extract precomputed eigenvalues for the wavelet derivative matrices
    eigenvalues_d_WT = _w.wavelet_derivatives['fft2_bases_d_WT'][unconstrained_indices]
    eigenvalues_dW0j_Rcv = sampler.aux_derivatives['eigenvalues_dW0_Rcv'][:, unconstrained_indices]
    
    ## Extract precomputed eigenvalues for the image correlation matrices
    eigenvalues_Rcv = _c.Sigma.fft_base_Rv 
    eigenvalues_Rch = _c.Sigma.fft_base_Rh 
    eigenvalues_Rc = _c.Sigma.fft_base_R 

    ## Extract the Cholesky decomposition of inv(A_c @ R_c @ A_c') (only exists with constraints on c)
    L = _c.reflectivity_constraints['L'] if constr_c else None
    
    ## Extract current state
    sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
    sigma2c = sampler.theta['sigma2c'][0, i + 1].item()
    d = sampler.theta['d_star'][:, i + 1].reshape((-1, 1)) # Extract current state

    # 2. Compute the gradient of log p(w*)
    w_star_full = w_star[unconstrained_indices, :].reshape(-1, 1)
    mean_w_full = mean_w[unconstrained_indices, :].reshape(-1, 1)
    w_star_centered_full = w_star_full - mean_w_full
    temp = linalg.solve_triangular(chol_R_wu_star, w_star_centered_full, lower=True)
    grad_prior_u = linalg.solve_triangular(chol_R_wu_star.T, temp, lower=False) / sigma2w
    grad_prior = embed_wu(grad_prior_u, _w)
    print(f'The gradient of the wavelet log-prior was computed with w_star (center values) =\n{w_star[(nv//2-5):(nv//2 + 5), :]}\n and is (center values) =\n{grad_prior[(nv//2-5):(nv//2 + 5), :] }\n') if verbose else None

    # 3. Compute the gradient of log p(d|w*)
    print(f"### Computing gradient of logdet Sigma_dw (VECTORIZED, Fourier domain) ###") if verbose else None
    
    ## Create base of wavelet convolutional matrices (for one column and for the whole lattice)
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star)

    ## Compute eigenvalues of the bases for W0, W0T, W, WT
    eigenvalues_W0 = scipy.fft.fft(base_W0.flatten()) 
    eigenvalues_W0T = np.conj(eigenvalues_W0)
    eigenvalues_W = eigenvalues_W0.reshape(-1, 1) @ np.ones((1, nh))
    eigenvalues_WT = np.conj(eigenvalues_W)
    print(f"Shape of eigenvalues_W: {eigenvalues_W.shape}") if verbose else None
    
    ## Compute eigenvalues of A, invA, Z, B, invA_W, WSigma_c, and the base of Z
    eigs, base_Z = compute_eigenvalues_A_invA_Z_B(sampler, i, eigenvalues_W, eigenvalues_WT, verbose=verbose)
    eigenvalues_invA = eigs['eigenvalues_inv_A'] # precomp

    ## Compute inv_Y, inv_S in the time domain
    Y, inv_Y, det_Y, logdet_Y = compute_Y_inv_det(sampler, i, base_Z, verbose)
    S, inv_S = compute_S(sampler, i, base_Z, verbose)

    ## Compute D1 for all coordinates in the time domain -> can this be done in the fourier domain?
    ### Permute base_Rv_W0T for unconstrained_indices with fancy indexing
    base_Rv_W0t = scipy.fft.ifft(eigenvalues_Rcv * eigenvalues_W0T)
    base_Rv_W0t_flat = base_Rv_W0t[:, 0] if base_Rv_W0t.ndim == 2 else base_Rv_W0t # Ensure base_Rv_W0t is 1D
    idx = (np.arange(nv)[None, :] - row_indexes[:, None]) % nv  # shape (m, nv)
    base_Dh = base_Rv_W0t_flat[idx]  # shape (m, nv); contains one Dh_j for each unconstrained j

    ### Transpose, sum, and FFT
    base_D1 = base_Dh + transpose_base_circ_vectorized(base_Dh) 
    eigenvalues_D1 = np.fft.fft(base_D1, axis=1) 

    ## 3.1 Compute the gradient of log|Sigma_dw|

    ### Compute eigenvalues of dWT_invA_W and WT_invA_dW (Fourier domain)
    idx = (np.arange(nv)[None, :] + row_indexes[:, None]) % nv  # shape (len(unconstrained_indices), nv)
    eigenvalues_dWT_invA_W =  np.multiply(eigenvalues_d_WT, eigs['eigenvalues_inv_A_W'], out=temp_array_knvnh)

    ### Compute eigenvalues of derivative of A (dA) for all coordinates 
    eigenvalues_dA_1 = np.multiply(
        (eigenvalues_dW0j_Rcv * eigenvalues_W0T[:, None]).T[:, :, None], 
        eigenvalues_Rch.reshape(1, 1, -1), 
        out=temp_array_knvnh_2
    )
    eigenvalues_dA = sigma2c * (eigenvalues_dA_1 + np.conj(eigenvalues_dA_1))  
    print(f"Shape of eigenvalues_dA: {eigenvalues_dA.shape}") if verbose else None

    ### Trace of invA_dA for all coordinates
    trace_invA_dA = np.sum(
        np.multiply(eigenvalues_invA[None, :, :], eigenvalues_dA, out=temp_array_knvnh_3).real,
        axis=(1,2), out=trace_invA_dA)
    
    ### Trace of invY_dY for all coordinates (non-zero only if constraints on c)
    if constr_c:    
        eigenvalues_WT_dinvA_j_W = np.multiply(
            np.multiply(eigenvalues_WT[None, :, :], 
                        np.multiply(
                            np.multiply(-eigenvalues_invA[None, :, :], eigenvalues_dA, out=temp_array_knvnh_2), 
                            eigenvalues_invA[None, :, :], out=temp_array_knvnh_2), out=temp_array_knvnh_2), 
            eigenvalues_W[None, :, :], out=temp_array_knvnh_2)

        ### Compute eigenvalues of d(WT @ invA @ W) for all coordinates
        eigenvalues_d_WT_invA_W = np.add(
            np.add(eigenvalues_dWT_invA_W, np.conj(eigenvalues_dWT_invA_W), out=temp_array_knvnh), 
            eigenvalues_WT_dinvA_j_W, out=temp_array_knvnh_2)
        print(f"Shape of eigenvalues_d_WT_invA_W: {eigenvalues_d_WT_invA_W.shape}") if verbose else None

        ### Compute eigenvalues of derivative of Z (dZ) for all coordinates 
        eigenvalues_dZ = sigma2c * np.multiply(
            np.multiply(eigenvalues_Rc[None, :, :], eigenvalues_d_WT_invA_W, out=temp_array_knvnh_2), 
            eigenvalues_Rc[None, :, :], out=temp_array_knvnh_2)
        print(f"Shape of eigenvalues_dZ: {eigenvalues_dZ.shape}") if verbose else None

        ### Compute base of Z for all coordinates
        base_dZ = scipy.fft.ifft2(eigenvalues_dZ, axes=(1,2)).real  # shape (len(unconstrained_indices), nv, nh
        print(f"shape of base_dZ: {base_dZ.shape}") if verbose else None
        
        ### Subset for all coordinates
        for idx in range(len(unconstrained_indices)):
            subset_base_dZ[idx, :, :] = subset_ABA(base_dZ[idx, :, :], _c)
        print(f"shape of subset_base_dZ: {subset_base_dZ.shape}") if verbose else None

        ### Compute derivative of Y for all coordinates (dY) in the time domain
        dY = -(L.T) @ subset_base_dZ @ L  # shape (len(unconstrained_indices), nr_constr_c, nr_constr_c)
        print(f"shape of dY: {dY.shape}") if verbose else None

        ### Compute trace of invY @ dY for all coordinates
        trace_invYdY = np.trace(inv_Y @ dY, axis1=1, axis2=2).real  # shape (len(unconstrained_indices), )
        print(f"shape of trace_invYdY: {trace_invYdY.shape}") if verbose else None

    ### Compute gradient of the log-determinant of Sigma_dw for all unconstrained coordinates
    trace_invA_dA += trace_invYdY  # Reuse variable to save memory
    d_logdet_Sigma_dw = embed_wu(trace_invA_dA, _w)
    print(f"Shape of d_logdet_Sigma_dw: {d_logdet_Sigma_dw.shape}, values for unconstrained indices:\n{d_logdet_Sigma_dw[unconstrained_indices, 0]}") if verbose else None

    ## 3.2 Compute the gradient of the sum of squares, vectorized

    ### Compute r, s, p in the Fourier domain
    r, s, p = compute_r_s_p_Fourier_domain(sampler, d, inv_S, eigenvalues_W, eigs, verbose=verbose)

    ### Compute P_R_ch_starT and R_cv_star_W0t in the time domain, used to compute the gradient of SS
    ### P is needed in gradient_SS_vectorized whether or not c is constrained.
    P = p.reshape((nv, nh), order='F')
    if fix_aliasing:
        P = P.copy()  # reshape returns a view of p; gradient_SS_vectorized FFTs P in place
    if _c.reflectivity_constraints['constr']:
        R_ch_star = _c.reflectivity_constraints['Rh_star']
        P_R_ch_starT = P @ (R_ch_star.T)
        W0T = linalg.circulant(base_W0T.reshape(-1)).T 
        R_cv_star_W0t = _c.reflectivity_constraints['Rv_star'] @ W0T
    else:
        R_ch_star = None
        P_R_ch_starT = None
        W0T = None
        R_cv_star_W0t = None

    ### Compute the gradient of the sum of squared, vectorized
    grad_ss = gradient_SS_vectorized(sampler, i, p, P, eigenvalues_D1, 
                                               R_cv_star_W0t, P_R_ch_starT, 
                                               d_SS_1_part1_temp, d_SS_1_part2_temp, d_SS_2_vectorized_temp,
                                               verbose=verbose)
    print(f"shape of grad_ss after vectorized computation: {grad_ss.shape}, values for unconstrained indices:\n{grad_ss[unconstrained_indices, 0]}") if verbose else None

    ### Combine the two terms of the gradient of p(d|w*)
    grad_lik = 0.5 * d_logdet_Sigma_dw + 0.5 * grad_ss
    print(f"shape of grad_lik after combining terms: {grad_lik.shape}, values for unconstrained indices:\n{grad_lik[unconstrained_indices, 0]}") if verbose else None

    ## Combine prior and likelihood gradients
    grad = grad_prior + grad_lik
    print(f"\n\nshape of gradient of the potential: {grad.shape}, values for unconstrained indices:\n{grad[unconstrained_indices, 0]}") if verbose else None
    print(f'####### #######  End of GradPotential ####### ##############') if verbose else None

    return grad