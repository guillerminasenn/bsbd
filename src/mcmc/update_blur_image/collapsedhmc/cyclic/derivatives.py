"""Functions for computing the derivatives in the
potential gradient in the cyclic case.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions:
---------

(used)
derivative_det_Sigma_dw_j
derivative_logdet_Sigma_dw_j
derivative_SS_first_term

(others) 
derivative_w_star_i
derivative_W0_i
derivative_Sigma_dw_i
"""

# Standard library imports
import time

# Third-party library imports
import numpy as np
from scipy import linalg, sparse
import scipy as scipy

from src.mcmc.update_blur_image.collapsedhmc.cyclic.efficient_utils import permute_matrix

pp = 6

def gradient_SS_vectorized(sampler, i, p, P, fft_base_D1_j_array, 
                           R_cv_star_W0t, P_R_ch_starT, 
                           d_SS_1_part1_vectorized, d_SS_1_part2_vectorized, d_SS_2_vectorized,
                           verbose=False):
    """Compute the gradient of SS = SS1 + SS2 with respect to all coordinates, 
    in the time domain."""

    print(f'\n### Computing derivative_SS_first_term for all coordinates (VECTORIZED):') if verbose else None

    # 1. Extract objects from sampler

    ## Extract Gaussian objets
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    _c = sampler.par_objs['c']

    ## Extract configuration
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
    nv = _lik.lattice.nv

    ## Extract eigenvalues
    fft_base_Rch = _c.Sigma.fft_base_Rh # precomputed

    ## Extract current state
    sigma2c = sampler.theta['sigma2c'][0, i + 1]

    ## Part 1 (always computed)

    # Multiply P on right by Rch (circular conv along axis=1)
    np.fft.fft(P, axis=1, out=P)  # Overwrite P with its FFT
    np.multiply(P, fft_base_Rch[None, :], out=P)  # In-place multiplication
    np.fft.ifft(P, axis=1, out=P)  # Overwrite P with IFFT result

    # Multiply result on left by D1 (circular conv along axis=0)
    np.fft.fft(P, axis=0, out=P)  # Reuse P for FFT along axis=0
    temp_array_knvnh = sampler.aux_derivatives['temp_array_knvnh']  # Preallocated temporary array
    P_after_D1_all = np.fft.ifft(P[None, :, :] * fft_base_D1_j_array[:, :, None], axis=1, out=temp_array_knvnh)  # Use a temporary array for output

    d_SS_1_part1_vectorized[unconstrained_indices] = (P_after_D1_all.reshape(len(unconstrained_indices), -1, order='F')) @ p  # shape: (len(unconstrained_indices), 1)

    # Part 2 (activates if c constr, else = 0); no Fourier domain here possible, are there other operations I can optimize?
    if _c.reflectivity_constraints['constr']:

        bases = sampler.par_objs['w'].wavelet_derivatives['bases_d_W0'][:, unconstrained_indices]
        row_indexes = np.argmax(bases, axis=0)
        
        # Build D1_star tensor with advanced indexing
        idx = (np.arange(nv)[None, :] + row_indexes[:, None]) % nv  # shape (len(unconstrained_indices), nv)
        Dh_star_array = R_cv_star_W0t[idx, :]  # shape (len(unconstrained_indices), nv, nv)
        products_all = (Dh_star_array + np.transpose(Dh_star_array, (0, 2, 1))) @ P_R_ch_starT

        # Reshape each product and compute dot product with p.T
        products_reshaped = products_all.reshape(len(unconstrained_indices), -1, order='F')  # shape: (len(unconstrained_indices), n)
        d_SS_1_part2_vectorized[unconstrained_indices] = products_reshaped @ p 

    # Putting part1 and part2 together
    d_SS_1_vectorized = -sigma2c * np.add(d_SS_1_part1_vectorized, -d_SS_1_part2_vectorized, out=d_SS_1_part1_vectorized)
    print(f'Vectorized d_SS_1 computation completed. Shape: {d_SS_1_vectorized.shape}') if verbose else None

    # VECTORIZED COMPUTATION OF d_SS_2 for all coordinates
    gamma_matrix_unconstrained_indexes = sampler.aux_derivatives['gamma_matrix_unconstrained_indexes']  # shape: (nv, p_size)
    d_SS_2_vectorized[unconstrained_indices] = (-2 * gamma_matrix_unconstrained_indexes @ p) # using only unconstrained coordinates
    print(f'Vectorized d_SS_2 computation completed. Shape: {d_SS_2_vectorized.shape}') if verbose else None

    # Final gradient computation
    d_SS_1_vectorized += d_SS_2_vectorized  # reuse d_SS_1_vectorized to save memory
    print(f'Vectorized d_SS computation completed. Shape: {d_SS_1_vectorized.shape}') if verbose else None
    return d_SS_1_vectorized.real

def derivative_SS_first_term(sampler, i, j, 
                             base_Rv_W0t, R_cv_star_W0t, p, P, P_R_ch_starT, 
                             base_D1_j_array, fft_base_D1_j_array,
                             verbose=False):
    """Compute the derivative of SS1 with respect to w*_j, in the time domain.

    Params:
    -------
    sampler: MCMC
        Object with the model.
    i: int
        Current iteration index.
    j: int
        Coordinate index in w_star.
    """

    # Extract from sampler
    _lik = sampler.par_objs['d']
    _c = sampler.par_objs['c']
    _w = sampler.par_objs['w']
    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    constr_c = _c.reflectivity_constraints['constr'] 
    n = _lik.lattice.n
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
    base_d_W0_j = sampler.par_objs['w'].wavelet_derivatives['bases_d_W0'][:, j].reshape(-1, 1)

    # Part 1 (always computed)
    t0 = time.time()
    # New way: with matrix-vector property of Kronecker products
    fft_base_Rch = _c.Sigma.fft_base_Rh # precomputed
    fft_base_D1_j = fft_base_D1_j_array[j - unconstrained_indices[0], :] # precomputed

    # Multiply on right by Rch (circular conv along axis=0)
    P_fft_axis1 = np.fft.fft(P, axis=1)
    P_after_Rch = np.fft.ifft(P_fft_axis1 * fft_base_Rch[None, :], axis=1)

    # Multiply on left by D1 (circular conv along axis=0)
    P_fft_axis0 = np.fft.fft(P_after_Rch, axis=0)
    P_after_D1 = np.fft.ifft(P_fft_axis0 * fft_base_D1_j[:, None], axis=0)
    product = P_after_D1.reshape((n, 1), order='F').real
    d_SS_1_part1 = np.squeeze((p.T) @ product)
    t1 = time.time() - t0

    # Part 2 activates if c constr, else = 0
    t0 = time.time()
    d_SS_1_part2 = 0
    if constr_c:
        Dh_star = permute_matrix(base_d_W0_j, R_cv_star_W0t) # vectorize!!

        D1_star = Dh_star + Dh_star.T 

        product = D1_star @ P_R_ch_starT
        d_SS_1_part2 = np.squeeze((p.T) @ (product.reshape((n, 1), order='F')))
    t1 = time.time() - t0

    if verbose and j in unconstrained_indices[:3]:
        print(f'd_SS_1_part1 = {d_SS_1_part1}\n')
        print(f'd_SS_1_part2 = {d_SS_1_part2}\n')
    d_SS_1 = -sigma2c * (d_SS_1_part1 - d_SS_1_part2)
    return d_SS_1

def derivative_w_star_i(i, _w, _lik, constr_w=True, verbose=False):
    """Derivative of w* with respect to coordinate i.
    
    Params:
    -------
    i: coordinate.
    """
    if verbose: print('Inside derivative_w_star_i')
    
    if _lik.lattice.topology == 'E':
        d_w_star_i = np.zeros((_lik.lattice.k, 1))
        d_w_star_i[i, 0] = 1
        if constr_w and ((i == 0) or (i == _lik.lattice.k - 1)):
            d_w_star_i[i, 0] = 0
            
    elif _lik.lattice.topology == 'C':
        d_w_star_i = np.zeros((_lik.lattice.nv, 1))
        
        unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
        if i in unconstrained_indices:
            d_w_star_i[i, 0] = 1
            
    if verbose: 
        print(f'd_w_star_i =\n{d_w_star_i}\n')
    
    return d_w_star_i

def derivative_W0_i(i, _w, _lik, verbose=False):
    """
    Compute the derivative of circulant matrix W0 with base w 
    with respect to w*_i.
    
    Parameters:
    -----------
    i (int): Index in the base vector (0 to n_v-1)
    _lik: contains the 2D lattice.
    
    Returns:
    --------
    numpy.ndarray: The derivative matrices dW0/dw*_i and d(W0.T)/dw*_i.
    """
    if verbose: print('Inside derivative_W0_i')

    nv = _lik.lattice.nv # because the convolutional matrix is for the full axis

    # By default, the derivative is a zero matrix
    base = np.zeros(nv)
    baseT = np.zeros(nv)

    # Change the corresponding element in the base vector to 1
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
    if i in unconstrained_indices:
        base[nv//2 - i] = 1
        baseT[-nv // 2 + i] = 1
    
    # Construct the derivative matrices
    DW0_i = sparse.csr_matrix(linalg.circulant(base).T)
    DW0T_i = sparse.csr_matrix(linalg.circulant(baseT).T) 
    
    if verbose: 
        print(f'D_W0_i:\n{DW0_i.toarray()}')
     
    return DW0_i, DW0T_i
