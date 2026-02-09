"""Gradient for collapsed HMC in Euclidean mode.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions:
---------
gradient_potential

"""

# Standard library imports

# Third-party library imports
import numpy as np
import pandas as pd
from scipy import linalg

# Local library imports
from src.utils.model_utils import pad, create_W, create_C
from src.mcmc.update_blur_image.collapsedhmc.euclidean.derivatives import derivative_w_star_j, derivative_W0_j, derivative_Sigma_dw_j, derivative_inv_Sigma_dw_j, derivative_logdet_Sigma_dw_j
from src.mcmc.update_blur_image.collapsedhmc.euclidean.potential import potential

pp = 4

def gradient_potential(sampler, i, w_star, verbose=False):
    """Gradient of the potential energy of the target

        p(w*|d) = p(w*) p(d|w*) / Z 
                = N(w*; \mu_w*, Sigma_w*) * N(d; W \mu_c, Sigma_dw),

    with

        Sigma_dw = W % Sigma_c* % W' + Sigma_d,

    allowing constraints w*=w|Ax=0 and c*=c|Ac=co.

    Params:
    ------
    w_star: k x 1. NOTE: if cyclic, k = nv.
    d: The data over the whole lattice.
    _lik: Instance of class SeismicData representing N(d; Wc, Sigma_d).
    _w: Instance of class Wavelet in k dims.
    _c: Instance of class Reflectivity in n dims.

    Return:
    ------
    grad: A k x 1 vector containing the gradient of the potential evaluated at w_star.
    """
    
    if verbose: 
        print(f'####### #######  Start of GradPotential ####### ##############')       
    
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    _c = sampler.par_objs['c']
    nv = _lik.lattice.nv
    constr_w = _w.wavelet_constraints['nr_constraints'] > 0 # is the wavelet constrained?
    constr_c = _c.reflectivity_constraints['constr'] # is the reflectivity constrained?

    # Extract current state
    d = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
    w_star_pad = pad(_lik.lattice, w_star) # for the convolution
    if verbose: 
        print(f'w_star_pad: \n{w_star_pad}') 

    # Create empty structures
    grad_logdet_Sigma_dw = np.zeros((nv, 1))
    grad_ss = np.zeros((nv, 1))
    
    # Extract blur-kernel wavelet mean and covariance
    sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
    if verbose:
        print(f"sigma2w in gradient_potential = {sigma2w}")
    if constr_w:
        w_star[0] = 0 # make exactly 0
        w_star[-1] = 0 
        mean_w = _w.wavelet_constraints['mean_w_star'] # length-k vector mu_w* and 
        inv_Sigma_w = _w.wavelet_constraints['inv_R_w_star'] / sigma2w
    else:
        mean_w = _w.mean
        inv_Sigma_w = _w.Sigma.Q / sigma2w
        
    # Create blur-kernel wavelet convolutional matrix
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star)
    if verbose:
        print(f'W =\n{pd.DataFrame(W.toarray())}.')
        print(f'Wavelet marginal variance =\n {sigma2w}')
        print(f'R_w =\n {pd.DataFrame(_w.Sigma.R)}')
        print(f'inv_Sigma_w =\n {pd.DataFrame(inv_Sigma_w)}')
        
    # Extract image reflectivity mean and covariance
    sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
    if constr_c:
        mean_c = _c.reflectivity_constraints['mean_c_star'] # length-n vector mu_c* and 
        Sigma_c = sigma2c * _c.reflectivity_constraints['R_c_star'] # n x n Sigma_c*
    else:
        mean_c = _c.mean
        Sigma_c = sigma2c * _c.Sigma.R 
    if verbose:
        print(f'Reflectivity mean =\n {np.round(mean_c, pp)}')
        print(f'Reflectivity covariance =\n {pd.DataFrame(Sigma_c)}')
        
    # Create reflectivity (image) convolutional matrix G: W \mu_c* = Gw
    bases_G, G = create_C(_lik.lattice, mean_c)
    
    # Pre-compute some useful quantities
    Gw = G @ w_star_pad
    try:
        sigma2d = sampler.aux['sigma2d'][:, i + 1] 
        Sigma_d = sigma2d * _lik.Sigma.R
        Sigma_wd = W.dot(Sigma_c) @ (W.T) + Sigma_d
        inv_Sigma_dw = linalg.inv(Sigma_wd)
    
        if verbose:
            print(f'Collapsed likelihood mean (Gamma @ w) = \n{Gw}.')
            print(f'd_cur = \n{np.round(d, pp)}.')
            print(f'Observational noise marginal variance =\n {sigma2d}')
            print(f'Observational noise correlation matrix =\n {pd.DataFrame(_lik.Sigma.R)}')
            print(f'Observational noise covariance =\n {pd.DataFrame(Sigma_d)}')
            print(f'Collapsed likelihood covariance = \n{pd.DataFrame(Sigma_wd)}.\n\n\n')
            
    except Exception as e:
        if verbose: 
            print('Could not update mean or Sigma_wd.')
            print(type(e), e)
        return None
      
    # Compute contribution of the prior to the gradient (works for constrained and unconstrained cases)
    grad_prior = (((w_star - mean_w).T) @ inv_Sigma_w).T
    if verbose:
        print(f'The gradient of the potential due to the prior is computed with w_star =\n{np.round(w_star, pp)}\n and inv_Sigma_w =\n{np.round(inv_Sigma_w, pp)}\n and is =\n{grad_prior}')

    # Compute contribution of the collapsed likelihood to the gradient
    
    # Compute gradient of log p(d|w*) for all free coordinates j
    unconstrained_indices = _w.wavelet_constraints['unconstrained_indices'] 
    for j in np.arange(nv): 
        if verbose: 
            print(f'\n\n### Coordinate {j} of the gradient (due to likelihood):\n')
        
        if j not in unconstrained_indices:
            if verbose:
                print(f'Coordinate {j} is constrained, continue to next coordinate.')
            continue

        # Derivative of the vector w*
        d_w_star_j = derivative_w_star_j(j, _w, constr_w=constr_w)
        d_w_star_j_padded = pad(_lik.lattice, d_w_star_j) 
        print(f'd_w_star_j_padded =\n{d_w_star_j_padded}\n') if verbose else None
        
        # Derivative of the wavelet convolutional matrix 
        if _w.wavelet_constraints['constr']:
            j_star = j - unconstrained_indices[0] - _lik.lattice.l + 1
        else:
            j_star = j - unconstrained_indices[0] - _lik.lattice.l
        d_W0_j, d_W0T_j = derivative_W0_j(j_star, _lik, constr_w=constr_w, verbose=verbose) 
        
        # Derivative of Sigma_dw
        d_Sigma_dw_j = derivative_Sigma_dw_j(W0, d_W0_j, d_W0T_j, Sigma_c, _lik, verbose=verbose)
        
        # Derivative of the inverse of Sigma_dw
        d_inv_Sigma_dw_j = derivative_inv_Sigma_dw_j(d_Sigma_dw_j, inv_Sigma_dw, verbose=verbose)
        
        # Derivative of the log-determinant of Sigma_dw
        d_logdet_Sigma_dw_j = derivative_logdet_Sigma_dw_j(d_Sigma_dw_j, inv_Sigma_dw, verbose=verbose) 
        grad_logdet_Sigma_dw[j, 0] = d_logdet_Sigma_dw_j
        if verbose: 
            print(f'Gradient of |Sigma_dw| for coordinate {j}={np.round(grad_logdet_Sigma_dw[j, 0], pp)}')

        # First term of the gradient of the SS = (d - W \mu_c*)' @ inv_Sigma_dw @ (d - W \mu_c*)
        grad_ss_term1_j = (d.T) @ d_inv_Sigma_dw_j @ d

        # Second term
        G_d_w_star_j = G @ d_w_star_j_padded
        grad_ss_term2_j_1 = (G_d_w_star_j.T) @ inv_Sigma_dw @ d
        grad_ss_term2_j_2 = (Gw.T) @ (d_inv_Sigma_dw_j.T) @ d
        grad_ss_term2_j = grad_ss_term2_j_1 + grad_ss_term2_j_2

        # Third term
        grad_ss_term3_j = ((G_d_w_star_j.T) @ inv_Sigma_dw @ Gw 
                           + (Gw.T) @ (d_inv_Sigma_dw_j.T) @ Gw 
                           + (Gw.T) @ inv_Sigma_dw @ G_d_w_star_j)
        
        # The gradient of the SS
        grad_ss_j = grad_ss_term1_j - 2 * grad_ss_term2_j + grad_ss_term3_j
        grad_ss[j, 0] = grad_ss_j
        if verbose: 
            print(f'Gradient of ss for coordinate {j}:\n term1 ={np.round(grad_ss_term1_j[0, 0], pp)}, \n term2 ={np.round(grad_ss_term2_j[0, 0], pp)} \n term3 ={np.round(grad_ss_term3_j[0, 0], pp)}\n')

    # Final gradient is the sum
    grad_lik = 0.5 * grad_logdet_Sigma_dw + 0.5 * grad_ss
    grad = grad_prior + grad_lik 
    
    if verbose: 
        print(f'####### #######  End of GradPotential ####### ##############')
        print(f'Gradient of prior = \n{np.round(grad_prior, pp)}')
        print(f'Gradient of log|Sigma_dw| = \n{np.round(grad_logdet_Sigma_dw, pp)}')
        print(f'Gradient of ss =\n{np.round(grad_ss, pp)}')
        print(f'Analytical gradient =\n{np.round(grad, pp)}')
        
    return grad
    
def grad_potential_fd(
    w_star, d, _lik, _lik_col, _w, _c, 
    epsilon=0.000001, compute = 'trad', verbose=False):
    """Compute the gradient of the potential at w_star using
    finite differences."""
    
    grad = np.zeros_like(w_star)
    K = w_star.shape[0]
    for k in np.arange(K):
        if _w.wavelet_constraints['constr'] == 0:
            std_vector = np.zeros_like(w_star)
            std_vector[k, 0] = epsilon
            w_star_plus = w_star + std_vector
            w_star_minus = w_star - std_vector
            U_q_plus = potential(w_star_plus, d, _lik, _lik_col, _w, _c, verbose=False)
            U_q_minus = potential(w_star_minus, d, _lik, _lik_col, _w, _c, verbose=False)
            if verbose: 
                print(f'U_q_plus = {np.round(U_q_plus, pp)}, U_q_minus = {np.round(U_q_minus, pp)}, Diff = {np.round(U_q_plus - U_q_minus, pp)}')
            grad[k, 0] = U_q_plus - U_q_minus
            
        if _w.wavelet_constraints['constr'] > 0:
            if (k > 0) or (k < (K - 1)):
                std_vector = np.zeros_like(w_star)
                std_vector[k, 0] = epsilon
                w_star_plus = w_star + std_vector
                w_star_minus = w_star - std_vector
                U_q_plus = potential(w_star_plus, d, _lik, _lik_col, _w, _c, verbose=False)
                U_q_minus = potential(w_star_minus, d, _lik, _lik_col, _w, _c, verbose=False)
                if verbose: 
                    print(f'U_q_plus = {np.round(U_q_plus, pp)}, U_q_minus = {np.round(U_q_minus, pp)}, Diff = {np.round(U_q_plus - U_q_minus, pp)}')
                grad[k, 0] = U_q_plus - U_q_minus 
    grad = grad / (2 * epsilon)
    return grad