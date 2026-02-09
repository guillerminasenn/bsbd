"""Step function for collapsed HMC in Euclidean mode.

Wavelet parameters represent the blur kernel; reflectivity parameters
represent the image.

Function:
---------
step_collapsed_hmc

"""

# Standard library imports
import math

# Third-party library imports
import numpy as np

import src.mcmc.update_blur_image.collapsedhmc.euclidean.utils as cheucl

def step_collapsed_hmc(sampler, iter_nr):
    
    # Load algorithmic params
    verbose = sampler.mcmc_config['verbose']
    epsilon = sampler.adapt_tracking['collapsed_hmc']['epsilon']['history'][iter_nr]
    L = math.floor(sampler.adapt_tracking['collapsed_hmc']['L']['history'][iter_nr])

    # Compute the iteration index in the chunk
    chunk_size = sampler.file_config['chunk_size']
    i = iter_nr % chunk_size
    print(f'\n#######  HMC. Global iter = {iter_nr}, iter in chunck ={i}  ###############\n') if verbose else None
  
    # 1. Load current state 
    w_star_cur = sampler.theta['w_star'][:, i + 1].reshape((-1, 1))
    c_cur = sampler.theta['c_star'][:, i + 1].reshape((-1, 1))
    sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
    sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
    sigma2d = sampler.aux['sigma2d'][:, i + 1].item()
    if verbose: 
        print(f'w_star_cur:\n{w_star_cur}')
        print(f'c_cur=\n{c_cur}\n')
        
    # Extract Gaussian objects to sample from prior and likelihood
    _lik = sampler.par_objs['d']
    _c = sampler.par_objs['c']
    _w = sampler.par_objs['w']
    _p = sampler.par_objs['p']
    
    # Update their marginal variances
    _c.update_sigma(sigma2c)
    _w.update_sigma(sigma2w)
    _lik.update_sigma(sigma2d)

    # Sample the momentum
    sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']
    if sigma2p is None:
        sigma2p = sampler.theta['sigma2w'][:, i + 1] # pre-condition with wavelet marginal variance
    p_cur = _p.sample_scipy(sigma2=sigma2p, verbose=verbose)
    if verbose: 
        print(f'p_cur:\n{p_cur}')

    # Integrate the trajectory L steps
    result = cheucl.leapfrog_collapsed_c(sampler, i, 
                                         w_star_cur, p_cur, L, epsilon,  
                                         verbose=verbose)
    (w_star_prop, p_star_prop, p_prop) = result

    # Negate momentum at end of trajectory to make the proposal symmetric
    p_prop = -p_prop
    if verbose:
        print(f'p_prop:\n{p_prop}')
        print(f'w_star_prop:\n{w_star_prop}')

    # A/R
    u = np.random.uniform()
    log_u = np.log(u)
    log_acc_prob, track_quantities = cheucl.log_ar_hmc_collapsed_c(
        sampler, i, w_star_cur, p_cur, w_star_prop, p_prop, 
        verbose=verbose)
    if log_u < log_acc_prob:
        if verbose: print(f'Accept.')
        sampler.stats['wc_update']['acceptance']['acc_iter'][iter_nr] = 1
        sampler.theta['w_star'][:, i + 1] = w_star_prop.flatten()
    else:
        if verbose: print(f'Reject.')
        sampler.theta['w_star'][:, i + 1] = sampler.theta['w_star'][:, i].flatten()

    # Save some stats
    sampler.stats['wc_update']['acceptance']['log_acc_prob'][iter_nr] = log_acc_prob
    sampler.stats['wc_update']['hamiltonian'][iter_nr] = track_quantities['cur_H'] # of the current position, not accepted position
    sampler.stats['wc_update']['potential'][iter_nr] = track_quantities['cur_U']
    sampler.stats['wc_update']['kinetic'][iter_nr] = track_quantities['cur_K']
    print(f'\n#######  End HMC. ###############\n') if verbose else None
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    