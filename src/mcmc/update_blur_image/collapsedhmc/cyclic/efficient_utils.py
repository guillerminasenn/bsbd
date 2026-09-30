"""Auxiliary functions for circulant algebra.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions: 
---------
compute_bases
compute_Y_inv_det
compute_S
compute_r_s_p
permute_circ
project_down_with_Ac
project_up_with_Ac
"""
# Third-party library imports
import numpy as np
from scipy import linalg
import scipy

# Local library imports
from src.utils.efficient_algebra_utils import multiply_circ, invert_circ, transpose_base_circ, multiply_matrix_vector_circ
from src.utils.model_utils import subset_ABA

pp = 4

# Aux
def compute_bases(sampler, i, base_W, base_WT, verbose=False):
    """
        A = W @ Sigma_c @ W' + Sigma_d,
        Z = Sigma_c @ (W.T) @ inv_A @ W @ Sigma_c,
    all matrices are BCCB."""
    
    # Extract from sampler
    _lik = sampler.par_objs['d']
    _c = sampler.par_objs['c']

    # Current state
    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    sigma2d = sampler.aux['sigma2d'][0, i + 1]
    
    # construct covariance matrices
    base_Sigma_c = sigma2c * _c.Sigma.base_R
    base_Sigma_d = sigma2d * _lik.Sigma.base_R
    
    # Construct A = W Sigma_c W' + Sigma_d, inv(A)
    base_WSc = multiply_circ(base_W, base_Sigma_c)
    base_WScWt = multiply_circ(base_WSc, base_WT)
    base_WScWt_t = transpose_base_circ(base_WScWt)
    base_WScWt = (base_WScWt + base_WScWt_t) / 2 # Ensure that W' Sigma_c W is symmetric
    base_A = base_WScWt + base_Sigma_d
    base_A = 0.5 * (base_A + transpose_base_circ(base_A))
    base_inv_A = invert_circ(base_A)
    base_inv_A = 0.5 * (base_inv_A + transpose_base_circ(base_inv_A))
    
    # Construct Z = Sigma_c W' inv(A) W Sigma_c
    base_ScWT = transpose_base_circ(base_WSc)
    base_B = multiply_circ(base_inv_A, base_WSc)
    base_Z = multiply_circ(base_ScWT, base_B)
    base_Z = (base_Z + transpose_base_circ(base_Z)) / 2 # Ensure that Z is symmetric
        
    return base_A, base_inv_A, base_Z, base_B

def compute_eigenvalues_A_invA_Z_B(sampler, i, eigenvalues_W, eigenvalues_WT, verbose=False):
    """Compute the eigenvalues of the BCCB matrices:
    
        A = W @ Sigma_c @ W' + Sigma_d,
        Z = Sigma_c @ (W.T) @ inv_A @ W @ Sigma_c,
        B = inv_A @ W @ Sigma_c,
        WSigma_c = W @ Sigma_c,
        inv_A_W = inv_A @ W, 
    
    and the base of Z."""
    
    # Extract from sampler
    _lik = sampler.par_objs['d']
    _c = sampler.par_objs['c']

    # Current state
    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    sigma2d = sampler.aux['sigma2d'][0, i + 1]
    
    # We start from the end result
    eigenvalues_Sigma_c = sigma2c * _c.Sigma.fft_base_R # FFT2 of base_Rc
    eigenvalues_Sigma_d = sigma2d * _lik.Sigma.fft_base_R # FFT2 of base_Rd
    eigenvalues_WSigma_c = eigenvalues_W * eigenvalues_Sigma_c
    eigenvalues_A = eigenvalues_WSigma_c * eigenvalues_WT + eigenvalues_Sigma_d
    eigenvalues_inv_A = eigenvalues_A ** (-1)  
    eigenvalues_inv_A_W = eigenvalues_inv_A * eigenvalues_W
    eigenvalues_B = eigenvalues_inv_A * eigenvalues_WSigma_c
    eigenvalues_BT = np.conj(eigenvalues_B)
    eigenvalues_Z = eigenvalues_Sigma_c * eigenvalues_WT * eigenvalues_B

    eigenvalues = {
        'eigenvalues_WSigma_c': eigenvalues_WSigma_c,
        'eigenvalues_inv_A_W': eigenvalues_inv_A_W,
        'eigenvalues_A': eigenvalues_A,
        'eigenvalues_inv_A': eigenvalues_inv_A,
        'eigenvalues_B': eigenvalues_B,
        'eigenvalues_BT': eigenvalues_BT,
        'eigenvalues_Z': eigenvalues_Z}
    base_Z = scipy.fft.ifft2(eigenvalues_Z)
    return eigenvalues, base_Z

def compute_Y_inv_det(sampler, i, base_Z, verbose=False):
    """Compute Y, its inverse, and its determinant."""
    _c = sampler.par_objs['c']
    constr_c = _c.reflectivity_constraints['constr'] 
    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    
    # Default values
    Y = None
    inv_Y = None
    det_Y = 0
    logdet_Y = None
    
    # Compute values
    if constr_c:
        n_w = _c.reflectivity_constraints['nr_constraints']
        rows_cyclic = _c.lattice.well_positions['rows_cyclic']
        L = _c.reflectivity_constraints['L']
        
        # Subset Z
        diagonal_block_Z = linalg.circulant(base_Z[:, 0].flatten()).T
        AZAt_temp = diagonal_block_Z[rows_cyclic, :] 
        AZAt = AZAt_temp[:, rows_cyclic]

        # Y
        LtAZAtL = (((L.T).dot(AZAt.T)).dot(L))
        Y = np.eye(n_w) - LtAZAtL / sigma2c
        Y = 0.5 * (Y + Y.T) # Ensure that Y is symmetric
        inv_Y = linalg.inv(Y)
        inv_Y = 0.5 * (inv_Y + inv_Y.T) # Ensure that inv_Y is symmetric
            
    return Y, inv_Y, det_Y, logdet_Y

def compute_S(sampler, i, base_Z, verbose=False):
    """Compute S and its inverse."""
    _c = sampler.par_objs['c']
    constr_c = _c.reflectivity_constraints['constr'] 
    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    
    if constr_c:
        base_Sigmac = sigma2c * _c.Sigma.base_R 
    
        # S
        base_matrix = base_Sigmac - base_Z
        base_matrix = 0.5 * (base_matrix + transpose_base_circ(base_matrix)) # Ensure symmetry
        S = subset_ABA(base_matrix, _c)
        inv_S = linalg.inv(S)
        inv_S = 0.5 * (inv_S + inv_S.T) # Ensure symmetry

    else:
        S = None
        inv_S = None

    return S, inv_S

def compute_r_s_p(sampler, d, base_W, base_inv_A, base_B, inv_S, verbose=False):
    """Compute the vectors r, s, and p."""
    _c = sampler.par_objs['c']
    constr_c = _c.reflectivity_constraints['constr'] 
    
    if constr_c:
        mean_c = _c.reflectivity_constraints['mean_c_star'] # length-n vector mu_c* and 
    else:
        mean_c = _c.mean   
        
    # Compute likelihood mean
    mean_lik = multiply_matrix_vector_circ(base_W, mean_c) # equal to Gw
    d_centered = d - mean_lik
    
    # r
    r = multiply_matrix_vector_circ(base_inv_A, d_centered)
    
    # s
    if constr_c:
        base_BT = transpose_base_circ(base_B)
        Btd = multiply_matrix_vector_circ(base_BT, d_centered)
        Ac_Btd = project_down_with_Ac(_c, Btd) # A_c @ vector
        inv_S_Ac_Btd = inv_S @ Ac_Btd # traditional algebra, no WA
        t = project_up_with_Ac(_c, inv_S_Ac_Btd) # (A_c.T) @ vector
        s = multiply_matrix_vector_circ(base_B, t)
    else:
        s = 0
    
    # p
    p = r + s
    
    return r, s, p


def compute_r_s_p_Fourier_domain(sampler, d, inv_S, eigenvalues_W, eigs, verbose=False):
    """Compute the vectors r, s, and p using the eigenvalues of the matrices."""

    _c = sampler.par_objs['c']
    constr_c = _c.reflectivity_constraints['constr'] 
    nv = sampler.lattice.nv
    nh = sampler.lattice.nh
    n = nv * nh
    
    if constr_c:
        mean_c = _c.reflectivity_constraints['mean_c_star'] # length-n vector mu_c* and 
        mean_c_matrix = mean_c.reshape((nv, nh), order='F')
        ifft2_mean_c_matrix = scipy.fft.ifft2(mean_c_matrix)
    else:
        mean_c = _c.mean   
        mean_c_matrix = mean_c.reshape((nv, nh), order='F')
        ifft2_mean_c_matrix = scipy.fft.ifft2(mean_c_matrix)
        
    # Extract eigenvalues
    eigenvalues_inv_A = eigs['eigenvalues_inv_A']
    eigenvalues_B = eigs['eigenvalues_B']
    eigenvalues_BT = eigs['eigenvalues_BT']

    # Compute likelihood mean
    mean_lik = scipy.fft.fft2(eigenvalues_W * ifft2_mean_c_matrix).reshape((n, 1), order='F')
    d_centered = d - mean_lik

    # r
    r = scipy.fft.fft2(eigenvalues_inv_A * scipy.fft.ifft2(d_centered.reshape((nv, nh), order='F'))).reshape((n, 1), order='F')

    # s
    if constr_c:
        Btd = scipy.fft.fft2(eigenvalues_BT * scipy.fft.ifft2(d_centered.reshape((nv, nh), order='F'))).reshape((n, 1), order='F')
        Ac_Btd = project_down_with_Ac(_c, Btd) # A_c @ vector
        inv_S_Ac_Btd = inv_S @ Ac_Btd # traditional algebra, no WA
        t = project_up_with_Ac(_c, inv_S_Ac_Btd) # (A_c.T) @ vector
        s = scipy.fft.fft2(eigenvalues_B * scipy.fft.ifft2(t.reshape((nv, nh), order='F'))).reshape((n, 1), order='F')
    else:
        s = 0
    
    # p
    p = r + s
    
    return r, s, p

def project_down_with_Ac(_c, vector):
    """Project nx1 vector down with A_c nw x n."""
    well_coords_vec = _c.lattice.well_positions['well_coords_vec']
    if well_coords_vec is not None:
        return vector[well_coords_vec]
    else:
        raise Exception(f'Cannot project down with A_c, no well coordinates. Check reflectivity constraints.')
    
def project_up_with_Ac(_c, vector):
    """Project nw x 1 vector up with A_c.T n x nw."""
    well_coords_vec = _c.lattice.well_positions['well_coords_vec']
    if well_coords_vec is not None:
        up_vector = np.zeros((_c.lattice.n, 1), dtype=vector.dtype)
        up_vector[well_coords_vec] = vector.reshape(-1, 1)
        return up_vector
    else:
        raise Exception(f'Cannot project up with A_c, no well coordinates. Check reflectivity constraints.')
    
def permute_circ(base_P, base_A, verbose=False):
    """Let P0 be a permutation matrix, P = I \kron P0 be the resulting circulant 
    or BCCB permutation matrix with base_P, such as in the wavelet convolutional matrix,
    and A a circulant or BCCB matrix with base_A.
    Return the base of the permutation PA=AP.
    """
    base_nonzero_block = base_P[:, 0].reshape(-1)
    j = [index for index, value in enumerate(base_nonzero_block) if value == 1][0]

    # Permute the base of A by rolling it j times
    permut_base_A = np.roll(base_A, j, axis=0)

    return permut_base_A

def permute_matrix(base_P, A, verbose=False):
    """Let P be a circulant permutation matrix with base_P, and A an arbitrary matrix.
    Return the product PA, which is a permutation of A.
    """
    # **NOTE**: Assumes that P = I \kron P0, such as in the wavelet convolutional matrix
    base_nonzero_block = base_P[:, 0].reshape(-1)
    j = [index for index, value in enumerate(base_nonzero_block) if value == 1][0]

    # Permute A by rolling it j times
    permut_A = np.vstack((A[j:], A[:j]))

    return permut_A
