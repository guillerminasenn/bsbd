"""Potential in Euclidean mode using direct linear algebra.

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
from src.utils.model_utils import create_W
from src.utils.math_utils import compute_log_det

verbose_d = False
pp = 4

def potential(sampler, i, w_star, verbose=False):
    """Potential energy of the target distribution. 
    Assuming w*=w|Ax=0 and c*=c|Ac=co, the target is 
        p(w*|d) = p(w*) p(d|w*) / Z 
               = N(w*; \mu_w*, Sigma_w*) * N(d; W \mu_c*, Sigma_dw),
        Sigma_dw = W % Sigma_c* % W' + Sigma_d.
          
    Params:
    ------
    w_star: k x 1 array containing the (constrained) wavelet sample.
    d: The data values, for the likelihood. 
    _lik: Instance of class SeismicData representing N(d; Wc, Sigma_d).
    _w: Instance of class Wavelet representing N(w; \mu_w, Sigma_w)
    _c: Instance of class Reflectivity representing N(c*; \mu_c*, Sigma_c*)
    
    Return:
    ------
    U_q: The potential energy evaluated at q.
    """
    
    if verbose: print('\nU(q). Inside potential')

    # Extract Gaussian objects
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    _c = sampler.par_objs['c']

    # extract current state
    d = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
    sigma2w = sampler.theta['sigma2w'][0, i + 1]
    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    sigma2d = sampler.aux['sigma2d'][0, i + 1]

    # Extract dimensions
    n = _lik.lattice.n
    nv = _lik.lattice.nv
    if verbose: 
        print(f'U(q). w_star: \n{np.round(w_star, pp)}')    

    ## 1. Contribution of the prior p(w*) = N(w*; \mu_w*, \Sigma_w*) to the potential
    if _w.wavelet_constraints['constr']:
        term1 = 0
        term2 = 0
        mean_w_star = _w.wavelet_constraints['mean_w_star']
        w_centered = w_star - mean_w_star
        
        # This is done efficiently because it's in low dimensions and sparse
        product = (_w.wavelet_constraints['inv_R_w_star']).dot(w_centered)
        term3 = 0.5 * ((w_centered.T) @ product)[0, 0] / sigma2w
        if verbose: 
            print(f'Calculating prior contribution to U(w*) with w constrained.')
            print(f"log_eigvals_R_w_star=\n{_w.wavelet_constraints['log_eigvals_R_w_star']}")
            print(f"inv_R_w_star=\n{_w.wavelet_constraints['inv_R_w_star']}")
            print(f"mean_w_star=\n{mean_w_star, pp}")

    if not _w.wavelet_constraints['constr']: # p(w*) = p(w)
        term1 = 0.5 * nv * np.log(2 * math.pi)
        logdet_R_w = _w.Sigma.logdet_R
        if verbose:
            print(f'logdet(R_w)={logdet_R_w}')
        term2 = 0.5 * nv * np.log(sigma2w) + 0.5 * logdet_R_w
        inv_cov = _w.Sigma.Q / sigma2w
        term3 = 0.5 * ((w_star.T) @ (inv_cov @ w_star))[0, 0]

    if verbose:
        print('U(q). Contribution from the prior: term1={0}, term2={1}, term3={2}.'.format(term1, term2, term3))

    U_q_w = term1 + term2 + term3

    ## 2. Contribution of collapsed likelihood p(d|w*) to the potential

    # Extract reflectivity mean and covariance 
    if _c.reflectivity_constraints['constr']:
        mean_c = _c.reflectivity_constraints['mean_c_star'] # length-n vector mu_c* and 
        Sigma_c = sigma2c * _c.reflectivity_constraints['R_c_star'] # n x n Sigma_c*
    else:
        mean_c = _c.mean
        Sigma_c = sigma2c * _c.Sigma.R 
    if verbose: 
        print(f'mean(c)=\n{mean_c}')

    # Update the mean of the Gaussian collapsed likelihood
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star, verbose=verbose)
    mean_lik = W.dot(mean_c)
    if verbose: 
        print(f'Mean of the collapsed lik as W @ mean_c = \n{mean_lik}.')

    # Update the covariance of the Gaussian collapsed likelihood
    Sigma_d = sigma2d * _lik.Sigma.R
    Sigma_dw = W @ Sigma_c @ (W.T) + Sigma_d
    if verbose: 
        print(f'U(q). Current covariance of the collapsed lik = \n{np.round(Sigma_dw, 4)}.')

    # Evaluate contribution
    
    # constant term
    term1 = 0.5 * n * np.log(2 * math.pi)
    
    # determinant
    try:
        logdet_Sigma_dw = compute_log_det(Sigma_dw) 
        if verbose: 
            print(f'U(q). *** Log-det of the collapsed lik covariance = \n{logdet_Sigma_dw}.')
    except:
        print(f'NON-PD Sigma_dw, discarding iter.')
        return None
    term2 = 0.5 * logdet_Sigma_dw
    
    # sum of squares
    d_centered = d.reshape((-1, 1)) - mean_lik.reshape((-1, 1))
    inv_cov = linalg.inv(Sigma_dw)
    term3 = 0.5 * (d_centered.T) @ (inv_cov @ d_centered)
    U_q_d = term1 + term2 + term3

    # Potential
    U = U_q_d + U_q_w 
    if verbose:
        print(f'U(q). Potential: U_q_d={np.round(U_q_d[0, 0], pp)}, U_q_w={np.round(U_q_w, pp)} ===> U={np.round(U[0, 0], pp)}')
    return U[0, 0]
