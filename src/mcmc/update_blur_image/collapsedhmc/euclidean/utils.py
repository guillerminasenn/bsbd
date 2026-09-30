"""Collapsed HMC utilities for Euclidean updates.

Wavelet parameters represent the blur kernel and reflectivity parameters
represent the image.

Functions:
---------
get_grad_potential
get_potential
log_ar_hmc_collapsed_c
kinetic_euclidean
grad_kinetic_euclidean
leapfrog_collapsed_c
log_ar_hmc_collapsed_c
"""

# Standard library imports
import copy

# Third-party library imports
import numpy as np

# Local library imports
from src.utils.model_utils import embed_wu
from src.mcmc.update_blur_image.collapsedhmc.euclidean.gradient_potential import gradient_potential
from src.mcmc.update_blur_image.collapsedhmc.euclidean.potential import potential

verbose_d = False
verbose_leap = False
pp = 4

# Utils (ongoing work)
def leapfrog_collapsed_c(sampler, i, w_star, p, L, epsilon,
                         verbose=False):
    """Leapfrog integrator for HMC. Takes current state and momentum and advances 
    them L time steps with step size epsilon.
    
    Params:
    -------
    sampler: Sampler object containing the MCMC configuration and parameters.
    i: int
        Current iteration index.
    w_star: k x 1 array containing the (constrained) blur-kernel wavelet sample.
    p: the current momentum, with length K=(k - nr_constraints) and distribution p ~ N_K(0, Sigma).
    L: Number of time steps.
    epsilon: Step size.
    d: Current data values over the whole lattice.
    
    _p: Instance of class Gaussian representing the distribution of the momentum p.
    _lik: Instance of class SeismicData representing N(d; Wc, Sigma_d).
    _w: Instance of class Wavelet representing N(w; \mu_w, Sigma_w)
    _c: Instance of class Reflectivity representing N(c*; \mu_c*, Sigma_c*)
    
    """
    if verbose: print('\n############    L. Inside leapfrog.   #############')
    
    # Load lattice configuration
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    k = _lik.lattice.k
    
    # Import the gradient of the potential
    grad_potential = gradient_potential
    
    p_star = embed_wu(p, _w)

    # 1. Make a half step for momentum at the beginning
    if verbose: print('Half step for momentum at the beginning.')
    gp = grad_potential(sampler, i, w_star, verbose=verbose)
    p_star_t = p_star - 0.5 * epsilon * gp
    if verbose: 
        print(f'epsilon={epsilon}')
        print(f'gradient=\n{gp}')
        print(f'p_star=\n{p_star}')
        print(f'p_star_1/2 =\n{p_star_t}\n')
        
    # 2. Alternate full steps for position and momentum
    w_star_t = copy.deepcopy(w_star)
    if verbose: print('Starting iterator.')
    for j in np.arange(L):
        if verbose: print(f'\n\n\n     L. Leapfrog iter={j+1} of {L}.')
        
        if _w.wavelet_constraints['constr']:
            # p_t = p_star_t[1:(k-1)]
            unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
            p_t = p_star_t[unconstrained_indices]
        else:
            p_t = copy.deepcopy(p_star_t)
        gk = grad_kinetic_euclidean(sampler, i, p_t, verbose=verbose)[1]
        w_star_t = w_star_t + epsilon * gk
        if verbose: 
            print(f'The position after the full step: \nw_star_t =\n{w_star_t}\n')
            print(f'epsilon={epsilon}')
            print(f'gradient=\n{gk}')
            print(f'w_star (original)=\n{w_star}')

        # 2.2 Make a full step for the momentum, except at end of trajectory
        if j < L - 1:
            p_star_t = p_star_t - epsilon * grad_potential(
                sampler, i, w_star_t, verbose=verbose)
            if verbose: 
                print(f'The momentum after the full step: p_star_t =\n{p_star_t}\n')
        
    # 3. Make a half step for momentum at the end.
    if verbose: print('Make a half step for momentum at the end.\n')
    p_star_t = p_star_t - 0.5 * epsilon * grad_potential(
        sampler, i, w_star_t, verbose=verbose)
    if verbose: 
        print(f'Final values at the end of the leapfrog: \n p_star = \n{p_star_t}\nw_star = {w_star_t}\n')
        
    # p_star_t is embedded to have length k, so if I have constraints it will be padded p_t. Return the unpadded version too
    if _w.wavelet_constraints['nr_constraints'] > 0:
        # p_t = p_star_t[1:-1] # this should be updated for the correct nr of constraints
        unconstrained_indices = _w.wavelet_constraints['unconstrained_indices']
        p_t = p_star_t[unconstrained_indices] # this should be updated for the correct nr of constraints
        if verbose:
            print(f'p_t (unpadded) = \n{p_t}\n')
    else:
        p_t = p_star_t
    
    if verbose: print('\n\n############         L. End leapfrog.        #############')
    return (w_star_t, p_star_t, p_t)

def log_ar_hmc_collapsed_c(sampler, i, w_star_cur, p_cur, w_star_prop, p_prop, 
                           verbose=False):
    """Evaluate potential and kinetic energies at start and end of trajectory.
    
    Params:
    ------
    sampler: Sampler object containing the MCMC configuration and parameters.
    i: int
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
    cur_U = potential(sampler, i, w_star_cur, verbose=verbose)
    cur_K = kinetic_euclidean(sampler, i, p_cur, verbose=False)
    prop_U = potential(sampler, i, w_star_prop, verbose=verbose)
    prop_K = kinetic_euclidean(sampler, i, p_prop, verbose=verbose)
    log_ratio = cur_U - prop_U + cur_K - prop_K
    
    # Save values for control
    cur_H = cur_U + cur_K
    prop_H = prop_U + prop_K
    track_quantities = {'cur_H':cur_H, 'prop_H':prop_H, 'cur_U':cur_U, 
                        'prop_U':prop_U, 'cur_K':cur_K, 'prop_K':prop_K}
    
    if verbose:
        print(f'H_cur = {cur_H},\nU(q_cur) = {cur_U},\nK(p_cur) = {cur_K}. ')
        print(f'H_prop = {prop_H},\nU(q_prop) = {prop_U},\nK(p_prop) = {prop_K}. ')
        print(f'Log-ratio = {log_ratio}. \n#####     End AR.   #####\n')
    return log_ratio, track_quantities

# Ready
def kinetic_euclidean(sampler, i, p, verbose=False):
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
    if verbose: print('\nK(p). Inside kinetic_collapsed_c_euclidean')
        
    # Extract objects
    _p = sampler.par_objs['p']
    n = _p.lattice.n
    if verbose:
        print(f"p.shape = {p.shape}, _p.mean.shape = {_p.mean.shape}, n = {n}")
    
    # Center the momentum
    p_centered = p.reshape((n, 1)) - _p.mean.reshape((n, 1))
    
    # Marginal variance of the momentum
    sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']
    if sigma2p is None:
        sigma2p = sampler.theta['sigma2w'][:, i + 1] # pre-condition

    # Inverse of the mass matrix
    inv_mass_matrix = _p.Sigma.Q / sigma2p

    # Compute the kinetic energy
    term1 = 0  # term1 = 0.5 * n * np.log(2 * math.pi) ; # cancels out in the acceptance ratio
    term2 = 0 # term2 = -0.5 * np.log(linalg.det(inv_Sigma)); cancels out in the acceptance ratio
    term3 = 0.5 * (p_centered.T) @ (inv_mass_matrix @ p_centered)
    
    K_p = term1 + term2 + term3
    if verbose:
        print(f"p =\n{np.round(p, pp)} ")
        print(f"_p.sigma2 =\n{np.round(sigma2p, pp)} ")
        print(f"inv_M =\n{np.round(inv_mass_matrix, pp)} ")
        print(f'Kinetic(p)={np.round(K_p, pp)}.')
        
    return K_p

def grad_kinetic_euclidean(sampler, i, p, verbose=False):
    """Gradient of the momentum. Assuming the constraints 
        w* = w|Aw=0,
    only (k - nr_constraints) free elements remain. 
    Thus the momentum vector has length K=(k - nr_constraints) and distribution
        p ~ N_K(0, Sigma),
    the kinetic energy is 
        K(p) = 0.5 * (p-mu)' Sigma^{-1} (p-mu),
    and its gradient has length K and is 
        K'(p) = -Sigma^{-1} (p-mu).
        
    NOTE: Assume Sigma=I.
    NOTE: this function could work for both Euclidean and cyclic.
    NOTE: this function could work for both eff and trad.
                  
    Params:
    ------
    p: Array containing the K values for the momentum.
    _p: Instance of class Gaussian representing N(p; 0, Sigma).
    _lik: SeismicData object containing the 2D lattice, to embed the wavelet.
    _w: Wavelet object used to define the wavelet.
    
    Return:
    ------
    grad: A length-K vector containing K'(p).
    grad_ext: A length-k vector with K'(p) embedded.
    """
    if verbose: 
        print("\nK\'(p). Inside grad_kinetic (cyclic).")
        print(f"\np =\n{np.round(p, pp)} ")  

    # Extract objects
    _p = sampler.par_objs['p']
    _w = sampler.par_objs['w']

    # Code for R=I
    sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']
    if sigma2p is None:
        sigma2p = sampler.theta['sigma2w'][:, i + 1] 
    
    # Code for a generic R
    inv_mass_matrix = _p.Sigma.Q / sigma2p
    grad = (inv_mass_matrix @ p).reshape((-1, 1))
    
    # Embed the gradient
    grad_ext = embed_wu(grad, _w)

    if verbose: 
        print(f"K\'(p). inv_M =\n{np.round(inv_mass_matrix, pp)} ")
        print(f"K\'(p). Gradient of K at p = \n{np.round(grad_ext, pp)}.")
    return grad, grad_ext
