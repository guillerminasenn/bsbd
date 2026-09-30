"""Step function for collapsed HMC in the cyclic case.

Wavelet parameters represent the blur kernel; reflectivity parameters
represent the image.

Function:
---------
step_collapsed_hmc

"""

# Standard library imports
import math
import pandas as pd

# Third-party library imports
import numpy as np

# Local application imports
import src.mcmc.update_blur_image.collapsedhmc.cyclic.utils as cyutils

def step_collapsed_hmc(sampler, iter_nr):
    
    # Load algorithmic params
    epsilon = sampler.adapt_tracking['collapsed_hmc']['epsilon']['history'][iter_nr]
    L = math.floor(sampler.adapt_tracking['collapsed_hmc']['L']['history'][iter_nr])
    verbose = sampler.mcmc_config['verbose']

    # Compute the iteration index in the chunk
    chunk_size = sampler.file_config['chunk_size']
    i = iter_nr % chunk_size
    print(f'\n#######  HMC. Global iter = {iter_nr}, iter in chunk ={i}  ###############\n') if verbose else None

    # 1. Load current state 
    w_star_cur = sampler.theta['w_star'][:, i + 1].reshape((-1, 1))
    

    # Sample the momentum; dim = #free elements in the blur-kernel wavelet
    _p = sampler.par_objs['p']
    if verbose:
        print(f"The covariance in the momentum is: \n{pd.DataFrame(_p.Sigma.R)}")
    if sampler.mcmc_config['collapsed_hmc']['precondition'] == 'prior':
        sigma2p = 1 / sampler.theta['sigma2w'][:, i + 1]
    else:
        sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']
    p_cur = _p.sample_scipy(sigma2=sigma2p, verbose=verbose)
    print(f'p_cur (first 5; sampled with sigma2p={sigma2p}):\n{p_cur[:5]}') if verbose else None
    
    # Integrate the trajectory L steps 
    result = cyutils.leapfrog_collapsed_c(sampler, i, 
                                          w_star_cur, p_cur, L, epsilon, 
                                          verbose=verbose)
    (w_star_prop, p_star_prop, p_prop) = result

    # Negate momentum at end of trajectory to make the proposal symmetric
    p_prop = -p_prop

    # A/R
    u = np.random.uniform()
    log_u = np.log(u)
    log_acc_prob, track_quantities = cyutils.log_ar_hmc_collapsed_c(
        sampler, i, 
        w_star_cur, p_cur, w_star_prop, p_prop, 
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    