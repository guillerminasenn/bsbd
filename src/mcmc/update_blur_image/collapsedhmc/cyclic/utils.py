"""Collapsed HMC utilities for sampling the blur-kernel wavelet.

Reflectivity parameters represent the image; wavelet parameters represent the
blur kernel. This module implements cyclic (FFT-based) updates.

Functions:
---------
get_grad_potential
get_potential
leapfrog_collapsed_c
log_ar_hmc_collapsed_c
kinetic
grad_kinetic
"""

# Standard library imports
import copy

# Third-party library imports
import numpy as np
from scipy import linalg

# Local library imports
from src.utils.model_utils import embed_wu
from src.mcmc.update_blur_image.collapsedhmc.cyclic.gradient_potential import gradient_potential_Fourier_domain
from src.mcmc.update_blur_image.collapsedhmc.cyclic.potential import potential

verbose_d = False
verbose_leap = False

pp = 4

# Utils (ongoing work)
def leapfrog_collapsed_c(sampler, i, w_star, p, L, epsilon,
                         verbose=False):
    """Leapfrog integrator for HMC. Takes current state and momentum and advances 
    them L time steps with step-size epsilon.
    
    Params:
    -------
    sampler: MCMC 
        Object containing
            _lik: Instance of class SeismicData representing N(d; Wc, Sigma_d).
            _w: Instance of class Wavelet representing N(w; \mu_w, Sigma_w)
            _c: Instance of class Reflectivity representing N(c*; \mu_c*, Sigma_c*)
    i: int
        Current iteration index.
    w_star: k x 1 array containing the (constrained) blur-kernel wavelet sample.
    p: the current momentum, with length K=(k - nr_constraints) and distribution p ~ N_K(0, Sigma).
    L: Number of time steps.
    epsilon: Step size.
    d: Current data values over the whole lattice.
    _p: Gaussian
        Object representing the distribution of the momentum p.
    
    """
    if verbose: 
        print('\n############    L. Inside leapfrog.   #############')
    
    # Import the gradient of the potential
    grad_potential = gradient_potential_Fourier_domain
    
    # Extract Gaussian objects to sample from priors and likelihood
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    _p = sampler.par_objs['p']
    nv = sampler.lattice.nv

    
    # Start and end of the wavelet in the extended wavelet
    wavelet_start_v = _lik.lattice.wavelet_positions['wavelet_start_v']
    wavelet_end_v = _lik.lattice.wavelet_positions['wavelet_end_v']

    # Embed the momentum to have same dimensions as the extended wavelet
    p_star = embed_wu(p, _w, verbose=verbose)

    # 1. Make a half step for momentum 
    print('Half step for momentum:') if verbose else None
    gp = grad_potential(sampler, i, w_star)
    p_star_t = p_star - 0.5 * epsilon * gp
    if verbose: 
        print(f'Momentum at t=0 (center values) =\n{p_star[(nv//2 - 5):(nv//2 + 5), :]}')
        print(f'epsilon={epsilon}, gradient of the potential at t=0 =\n{gp[(nv//2 - 5):(nv//2 + 5), :]}')
        print(f'Momentum at t=1/2 =\n{p_star_t[(nv//2 - 5):(nv//2 + 5), :]}\n')
        
    # 2. Alternate full steps for position and momentum
    w_star_t = copy.deepcopy(w_star)
    for j in np.arange(L):
        print(f'\n\n***Leapfrog iter= {j+1} of {L}...') if verbose else None
        
        # 2.1 Make a full step for the position
        if verbose: 
            print(f'Current position at t={j} (center values) =\n{w_star_t[(nv//2 - 5):(nv//2 + 5), :]}')
            
        # Subset the free elements in the extended momentum vector
        if _w.wavelet_constraints['constr']:
            p_t = p_star_t[(wavelet_start_v + 1):(wavelet_end_v - 1)]
        else:
            p_t = p_star_t[(wavelet_start_v):(wavelet_end_v)]
        
        # Advance the position
        gk = grad_kinetic(sampler, i, p_t, verbose=verbose)[1]
        w_star_t = w_star_t + epsilon * gk
        if verbose: 
            print(f'Kinetic gradient at t={j+1}=\n{gk[(nv//2 - 5):(nv//2 + 5), :]}')
            print(f'New position at t={j+1}: \n{w_star_t[(nv//2 - 5):(nv//2 + 5), :]}\n\n')

        # 2.2 Make a full step for the momentum, except at end of trajectory
        if j < L - 1:
            gp = grad_potential(sampler, i, w_star_t)
            print(f"Current momentum at t={j+1} (center values)=\n{p_star_t[(nv//2 - 5):(nv//2 + 5), :]}") if verbose else None
            p_star_t = p_star_t - epsilon * gp
            if verbose: 
                print(f'epsilon={epsilon}, Gradient of the potential at t=j+1 =\n{gp[(nv//2 - 5):(nv//2 + 5), :]}')
                print(f'Momentum at t={j+1}: \n{p_star_t[(nv//2 - 5):(nv//2 + 5), :]}\n')
        
    # 3. Make a half step for momentum at the end.
    if verbose: print('Make a half step for momentum at the end.\n')
    gp = grad_potential(sampler, i, w_star_t)
    print(f'Momentum at t=L-1/2 =\n{p_star_t[(nv//2 - 5):(nv//2 + 5), :]}') if verbose else None
    p_star_t = p_star_t - 0.5 * epsilon * gp
    if verbose: 
        print(f'Potential gradient at t=L-1/2 (first 5)=\n{gp[(nv//2 - 5):(nv//2 + 5), :]}')
        print(f'Momentum at t={L}: \n{p_star_t[(nv//2 - 5):(nv//2 + 5), :]}\n')

    if verbose: 
        print(f'\n\nFinal values at the end of the leapfrog: \n p_star = \n{p_star_t[(nv//2 - 5):(nv//2 + 5), :]}\nw_star = {w_star_t[(nv//2 - 5):(nv//2 + 5), :]}\n')
        
    # p_star_t is embedded to have length k, so if I have constraints it will be padded p_t. Return the unpadded version too
    if _w.wavelet_constraints['constr']:
        p_t = p_star_t[(wavelet_start_v + 1):(wavelet_end_v - 1)]
    else:
        p_t = p_star_t[(wavelet_start_v):(wavelet_end_v)]
            
    if verbose: print('\n############         L. End leapfrog.        #############')
    return w_star_t, p_star_t, p_t

def log_ar_hmc_collapsed_c(sampler, i, 
                           w_star_cur, p_cur, w_star_prop, p_prop, 
                           verbose=False):
    """Evaluate potential and kinetic energies at start and end of trajectory.
    
    Params:
    ------
    sampler: Sampler object containing the MCMC configuration and parameters.
    iter_nr: int    
        Current iteration index.
    w_star_cur: Current state.
    p_cur: Current momentum, in K dimensions.
    w_star_prop: Proposed state. 
    p_prop: Proposed momentum, in K dimensions.
    d: The data.
    _p: Instance of class Gaussian representing the distribution of the momentum p.
    _lik: Instance of class SeismicData representing N(d; Wc, Sigma_d).
    _w: Instance of class Wavelet representing N(w; 0, Sigma_w)
    _c: Instance of class Reflectivity representing N(c; 0, Sigma_c)
    
    Return:
    ------
    log_ratio: The log-acceptance ratio as defined in Eq. 3.6, p12, Neal's 
        Ch. 5 of MCMC Handbook.
    """
    if verbose: print('\n\n#####    AR. Inside log_ar_hmc_collapsed_c    #####')
    
    # Compute the ratio
    cur_U = float(potential(sampler, i, w_star_cur, verbose=verbose))
    cur_K = float(kinetic(sampler, i, p_cur, verbose=verbose))
    prop_U = float(potential(sampler, i, w_star_prop, verbose=verbose))
    prop_K = float(kinetic(sampler, i, p_prop, verbose=verbose))
    log_ratio = float(cur_U - prop_U + cur_K - prop_K)
    
    # Save values for control
    cur_H = cur_U + cur_K
    prop_H = prop_U + prop_K
    track_quantities = {
        'cur_H': float(cur_H),
        'prop_H': float(prop_H),
        'cur_U': float(cur_U),
        'prop_U': float(prop_U),
        'cur_K': float(cur_K),
        'prop_K': float(prop_K),
    }
    
    if verbose:
        print(f'H_cur = {cur_H},\nU(q_cur) = {cur_U},\nK(p_cur) = {cur_K}. ')
        print(f'H_prop = {prop_H},\nU(q_prop) = {prop_U},\nK(p_prop) = {prop_K}. ')
        print(f'Log-ratio = {log_ratio}. \n#####     End AR.   #####\n')
    return float(log_ratio), track_quantities

# Ready
def kinetic(sampler, i, p, verbose=False):
    """Kinetic energy of the momentum. Assuming the constraints 
        w* = w|Aw=0,
    only (k - nr_constraints) free elements remain. 
    Thus the momentum vector has length K=(k - nr_constraints) and distribution
        p ~ N_K(0, Sigma),
    and the potential energy is 
        K(p) = 0.5 * (p-mu)' Sigma^{-1} (p-mu).
    
    NOTE: this function could work for both Euclidean and cyclic.
    NOTE: This function works for both eff and trad.
                  
    Params:
    ------
    sampler: Sampler object containing the MCMC configuration and parameters.
    i: int
        Current iteration index.
    p: Array containing the K values for the momentum.
    
    Return:
    ------
    K_p: Kinetic energy evaluated at p.
    """
    if verbose: 
        print('\nK(p). Inside kinetic (cyclic)')
        
    # Extract momentum configuration
    _p = sampler.par_objs['p']

    # Marginal variance of the momentum
    if sampler.mcmc_config['collapsed_hmc']['precondition'] == 'prior':
        sigma2p = 1 / sampler.theta['sigma2w'][:, i + 1] 
    else:
        sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']


    # Compute the kinetic energy
    term1 = 0  # term1 = 0.5 * n * np.log(2 * math.pi) ; # cancels out in the acceptance ratio
    term2 = 0 # term2 = -0.5 * np.log(linalg.det(inv_Sigma)); cancels out in the acceptance ratio
    
    # With Cholesky
    chol_mass_matrix = linalg.cholesky(_p.Sigma.R, lower=True)
    y = linalg.solve_triangular(chol_mass_matrix, p, lower=True)
    term3 = 0.5 * y.T @ y / sigma2p
    
    K_p = term1 + term2 + term3
    K_p = float(np.asarray(K_p).reshape(-1)[0])
    if verbose:
        print(f"p (first 5)=\n{np.round(p.flatten()[:5], pp)} ")
        print(f'Kinetic(p)={np.round(K_p, pp)}.')
        
    return K_p

def grad_kinetic(sampler, i, p, verbose=False):
    """Gradient of the momentum. Assuming the constraints 
        w* = w|Aw=0,
    only (k - nr_constraints) free elements remain. 
    Thus the momentum vector has length K=(k - nr_constraints) and distribution
        p ~ N_K(0, Sigma),
    the kinetic energy is 
        K(p) = 0.5 * (p-mu)' Sigma^{-1} (p-mu),
    and its gradient has length K and is 
        K'(p) = -Sigma^{-1} (p-mu).
        
    NOTE: this function could work for both Euclidean and cyclic.
    NOTE: this function could work for both eff and trad.
                  
    Params:
    ------
    sampler: Sampler object containing the MCMC configuration and parameters.
    i: int
        Current iteration index.
    p: Array containing the K values for the momentum.

    Return:
    ------
    grad: A length-K vector containing K'(p).
    grad_ext: A length-k vector with K'(p) embedded.
    """
    if verbose: 
        print("\nK\'(p). Inside grad_kinetic (cyclic).")
        print(f"\np =\n{np.round(p.flatten()[:5], pp)} ")  
        
    # Extract Gaussian objects 
    _w = sampler.par_objs['w']
    _p = sampler.par_objs['p']

    # Marginal variance of the momentum
    if sampler.mcmc_config['collapsed_hmc']['precondition'] == 'prior':
        sigma2p = 1 / sampler.theta['sigma2w'][:, i + 1] 
    else:
        sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']
    
    # With Cholesky
    chol_mass_matrix = linalg.cholesky(_p.Sigma.R, lower=True)
    y = linalg.solve_triangular(chol_mass_matrix, p, lower=True)
    grad = linalg.solve_triangular(chol_mass_matrix.T, y, lower=False) / sigma2p
    grad = grad.reshape((-1, 1))
    
    # Embed the gradient
    grad_ext = embed_wu(grad, _w)
    if verbose: 
        print(f"Gradient of K at p (first 5)= \n{np.round(grad_ext.flatten()[:5], pp)}.")
        
    return grad, grad_ext
