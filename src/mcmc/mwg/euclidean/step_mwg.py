"""Step function for Metropolis-within-Gibbs (MWG) sampling in Euclidean mode.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.

Updates (w, c) and variance parameters (sigma2c, sigma2w, zeta) plus data `d`
at each iteration. The `step_wc_update` function selects Gibbs or collapsed HMC
for the joint (w, c) update. This function assumes Euclidean lattices.
"""

# Local library imports
from src.mcmc.mwg.euclidean.update_ss import _update_ss
from src.mcmc.mwg.log_posterior_density import _compute_log_posterior_density
from src.mcmc.updates.state import _copy_previous_state
from src.mcmc.updates.hyperparams import step_sigma2c_update, step_sigma2w_update, step_zeta_update
from src.mcmc.updates.data import step_d_update

def step_mwg(sampler, i):

        # Set the next state equal to the current state
        _copy_previous_state(sampler, i)

        # Update (w, c) with current adapted parameters + handle adaptation 
        if sampler.mcmc_config.get('verbose'):
            print(f"Updating (w,c) at iteration {i} using algorithm: {sampler.mcmc_config.get('algorithm') }; inside mwg.py")
        step_wc_update(sampler, i) # adaptation is handled inside this function

        # Re-compute sum of squares used in the variance updates
        _update_ss(sampler, i) 
        
        # Update sigma_c
        step_sigma2c_update(sampler, i)
        
        # Update sigma_w  
        step_sigma2w_update(sampler, i)
        
        # Update zeta
        step_zeta_update(sampler, i)

        # Update d
        step_d_update(sampler, i)   

        # Compute the posterior log-density at the new state
        _compute_log_posterior_density(sampler, i)

        # Save stats if requested
        if sampler.mcmc_config['save_stats']:
            sampler._update_stats(i) 

def step_wc_update(sampler, i):
    """Update (w,c) on Euclidean lattices using the configured algorithm with adaptation"""
    algorithm = sampler.mcmc_config.get('algorithm') 
    if sampler.mcmc_config.get('verbose'):
        print(f"Updating (w,c) at iteration {i} using algorithm: {algorithm}")

    if 'w_star' not in sampler.estimate or 'c_star' not in sampler.estimate:
        if sampler.mcmc_config.get('verbose'):
            print(f"Skipping (w,c) update at iteration {i} as they are not estimated.")
        sampler.theta['w_star'][:, i + 1] = sampler.theta['w_star'][:, i]
        sampler.theta['c_star'][:, i + 1] = sampler.theta['c_star'][:, i]
        sampler.aux['w'][:, i + 1] = sampler.aux['w'][:, i]
        sampler.aux['c'][:, i + 1] = sampler.aux['c'][:, i]
        return
    
    else:
        if algorithm == 'gibbs':
            from src.mcmc.update_blur_image.gibbs.step_gibbs import step_gibbs
            step_gibbs(sampler, i, update_blur=True, update_image=True)
            
        elif algorithm == 'hmc':
            raise NotImplementedError("HMC algorithm is not available in update_blur_image.")

        elif algorithm == 'collapsed_hmc':
            from src.mcmc.update_blur_image.collapsedhmc.euclidean.step_collapsed_hmc import step_collapsed_hmc
            from src.mcmc.update_blur_image.gibbs.step_gibbs import step_gibbs
            from src.mcmc.adapt import adapt
            try:
                step_collapsed_hmc(sampler, i)
            except Exception as e:
                print(f"There was a problem at iteration {i}, error \n {e}")

            adapt(sampler, i, param='epsilon', alg='collapsed_hmc')
            adapt(sampler, i, param='L', alg='collapsed_hmc')
            step_gibbs(sampler, i, update_blur=False, update_image=True)
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")