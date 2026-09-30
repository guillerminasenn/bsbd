
"""
Functions included in the sampling_utils module:

- Gaussian sampling functions.
    _sample_circ
    _sample_scipy
    _correct_sample
"""

# Third-party imports
import numpy as np
from scipy import linalg
from scipy import stats

# Local imports
from src.utils.efficient_algebra_utils import multiply_circ, invert_circ, multiply_matrix_vector_circ

######### Gaussian sampling functions ##########
def _sample_circ(mean, sigma2, base_R, verbose=False):
        """Sample once from a multivariate lattice on a cyclic lattice 
        using the FFT."""
        nv = base_R.shape[0]
        nh = base_R.shape[1]
        n = nv * nh
        z = np.empty(n, dtype=np.complex128)
        z.real = np.random.standard_normal(n)
        z.imag = np.random.standard_normal(n)
        Z = z.reshape((nv, nh))                                                             
        Lambda = np.sqrt(n) * np.fft.fft2((sigma2 * base_R), norm='ortho')
        internal_term = np.power(Lambda, 0.5) * Z
        v = np.fft.fft2(internal_term, norm='ortho')
        v_real = np.real(v).reshape((n, 1), order='F')
        x = mean.reshape((n, 1)) + v_real

        # Reshape x into (1,1) np.array if it only contains one element
        if not x.shape:
            x = x.reshape((1, 1))
        else:
            x = x.reshape((n, 1))
        return x

def _sample_scipy(mean, sigma2, R, verbose=False):
    """."""
    n = mean.shape[0]
    cov = sigma2 * R
    gaussian = stats.multivariate_normal(mean=mean.reshape(-1), cov=cov)
    x = gaussian.rvs(1)
    # if verbose:
    #     print(f"Sampled from Euclidean lattice: {x.reshape((n, 1))}")

    # Reshape x into (1,1) np.array if it only contains one element
    if not x.shape:
        x = x.reshape((1, 1))
    else:
        x = x.reshape((n, 1))
    return x

def _correct_sample(x, b, A, inv_ARAt, RAt, verbose=False):
    """Correct a Gaussian sample from an unconstrained distribution
    using conditioning by Kriging.
    
    Params:
    -------
    x: (nv x 1) np.array.
        The sample to be corrected.
    b: (n_w x 1) np.array
        The constraint independent term.
    A: (n_w x n) np.array
        The constraint matrix.
    base_R: (nv x nh) np.array
        (optional) The base of the BCCB full conditional correlation matrix.
    inv_ARAt: (n_w x n_w) np.array
        (optional) The inverse of A @ R @ A'.
    RAt: (nv x n_w) np.array
        (optional) The matrix R @ A'.
    """
    
    # Compute the corrected sample
    vector = (A.dot(x.reshape(-1, 1))).reshape(-1, 1) - b.reshape(-1, 1)
    right_term = inv_ARAt @ vector
    x_correction = RAt @ right_term   
    x_star = x - x_correction  
    
    # Reshape x_star into (1,1) np.array if it only contains one element
    if not x_star.shape:
        x_star = x_star.reshape((1, 1))
    return x_star        

######## Wavelet full conditional #########
def _compute_bases_Gammat_inv_Sigma_d(sampler, i, bases_G): 
    """
    Compute base of the product Gamma' @ inv_Sigma_d.
    
    Params:
    -------
    sampler: MCMC object.
        Contains the current state.
    i: int
        Current iteration.
    bases_G: (nv x nh) np.array
        Bases of the vertical block-matrix reflectivity convolutional matrix. 
    
    Return:
    -------
    bases_Gammat_inv_Sigma_d: List with the nh bases of the product C' @ Q_d
    """
    
    # Extract configuration
    nh = sampler.lattice.nh
    dense_indexes = sampler.model.model['d'].dense_indexes
    # base_inv_R_d = sampler.model.model['d'].Sigma.base_R # old
    base_inv_R_d = sampler.model.model['d'].Sigma.base_Q

    # Extract current state
    # sigma2d = sampler.aux['sigma2d'][0, i + 1]
    psi = sampler.model.aux['psi']
    sigma2d = psi * sampler.theta['sigma2c'][0, i + 1] * sampler.theta['sigma2w'][0, i + 1] * sampler.theta['zeta'][0, i + 1]
    
    # Compute the base of the product
    base_Q_d = base_inv_R_d / sigma2d
    bases_G_inverted = bases_G[::-1] # inverted every base
    bases_G_inverted_shifted = np.roll(bases_G_inverted, 1, axis=0)

    # list comprehension code
    bases_Gammat_inv_Sigma_d = [np.sum(
        [multiply_circ(
            bases_G_inverted_shifted[:, (j + k) % nh].reshape(-1, 1), 
            base_Q_d[:, j].reshape(-1, 1)) 
        for j in dense_indexes], axis=0)
        for k in range(nh)]

    return bases_Gammat_inv_Sigma_d

def _compute_base_Gammat_inv_Sigma_d_Gamma(sampler, bases_G, bases_Gt_inv_Sigma_d): 
    """
    Compute the base of the circulant product Gamma' @ inv_Sigma_d @ Gamma.
    
    Params:
    -------
    sampler: MCMC object.
        Contains the current state.
    bases_G: (nv x nh) np.array
        Bases of the vertical block-matrix reflectivity convolutional matrix. 
    bases_Gt_inv_Sigma_d: (nv x nh) np.array
        Bases of the product Gamma' @ inv_Sigma_d.
    
    Return:
    -------
    base_Gammat_inv_Sigma_d_Gamma: Base of the product C' @ Q_d @ C.
    """
    
    # Compute the base of the product
    nh = sampler.lattice.nh
    base_Gammat_inv_Sigma_d_Gamma = np.sum(
        [multiply_circ(
            bases_G[:, j].reshape(-1, 1), 
            bases_Gt_inv_Sigma_d[j].reshape(-1, 1))
         for j in range(nh)],
         axis=0)
    
    return base_Gammat_inv_Sigma_d_Gamma

def _compute_wavelet_conditional_covariance(base_inv_Sigma_w, base_Gt_inv_Sigma_d_G): 
    """
    Compute the base of Gamma' @ Sigma_d^{-1} @ Gamma + Sigma_w^{-1}, 
    and the base of its inverse.
    
    Params:
    -------
    base_inv_Sigma_w: (nv x 1) np.array
        Base of the wavelet prior precision matrix.
    base_Gt_inv_Sigma_d_G: (nv x nh) np.array
        Base of the product Gamma' @ Sigma_d^{-1} @ Gamma.
    
    Return:
    -------
    base_Sigma_cond: 
        Base of the wavelet full conditional covariance matrix.
    base_inv_Sigma_cond: 
        Base of the wavelet full conditional precision matrix.
    """

    # Compute bases of conditional covariance and precision
    base_inv_Sigma_cond = base_inv_Sigma_w + base_Gt_inv_Sigma_d_G
    base_Sigma_cond = invert_circ(base_inv_Sigma_cond.reshape(-1, 1))
    return base_Sigma_cond, base_inv_Sigma_cond

def _compute_Gt_inv_Sigma_d_d(sampler, i, bases_Gt_inv_Sigma_d):
    """
    Compute the matrix-vector product Gamma' @ Sigma_d^{-1} @ d_star.

    Params:
    -------
    sampler: MCMC object.
        Contains the current state.
    i: int
        Current iteration.
    bases_Gt_inv_Sigma_d: (nh) list of (nv x 1) np.array
        Bases of the product Gamma' @ Sigma_d^{-1}
    """

    # Extract configuration and current state
    d_star = sampler.theta['d_star'][:, i + 1].reshape(-1, 1)
    nv = sampler.lattice.nv
    nh = sampler.lattice.nh

    # Compute the matrix-vector product [Gamma' @ Sigma_d^{-1}]_j @ [d_star]_j (column j of the lattice)
    products = [
        multiply_matrix_vector_circ(
            bases_Gt_inv_Sigma_d[j].reshape(-1, 1), 
            d_star[j * nv:(j + 1) * nv, 0].reshape(-1, 1)
        ) for j in range(nh)
    ]
    # Sum the products to get the final result
    product = np.sum(products, axis=0)
    return product

def _compute_wavelet_conditional_mean(sampler, i, Gammat_inv_Sigma_d_d, base_Gammat_inv_Sigma_d_Gamma, base_Sigma_cond):
    """
    Compute the wavelet full conditional mean.

    Params:
    -------
    sampler: MCMC object.
        Contains the current state.
    i: int
        Current iteration.
    Gammat_inv_Sigma_d_d: (nv x 1) np.array
        The matrix-vector product Gamma' @ Sigma_d^{-1} @ d_star.
    base_Gammat_inv_Sigma_d_Gamma: (nv x nh) np.array
        Base of the product Gamma' @ Sigma_d^{-1} @ Gamma.
    base_Sigma_cond: (nv x nh) np.array
        Base of the wavelet full conditional covariance matrix.
    """
    # Extract configuration and current state
    sigma2w = sampler.theta['sigma2w'][0, i + 1]
    base_Sigma_w = sigma2w * sampler.model.model['w'].Sigma.base_Rv
    base_I = np.zeros_like(base_Sigma_w)
    base_I[0, 0] = 1 # Identity matrix base

    # Compute conditional mean
    base1 = multiply_circ(base_Gammat_inv_Sigma_d_Gamma, base_Sigma_cond)
    base2 = base_I - base1
    base3 = multiply_circ(base_Sigma_w, base2)
    mean_cond = multiply_matrix_vector_circ(base3, Gammat_inv_Sigma_d_d)

    return mean_cond