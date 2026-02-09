"""Step function for Metropolis-within-Gibbs (MWG) sampling in cyclic mode.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.

Updates (w, c) and variance parameters (sigma2c, sigma2w, zeta) plus data `d`
at each iteration. The `step_wc_update` function selects Gibbs or collapsed HMC
for the joint (w, c) update. This function assumes cyclic lattices.
"""

# Third-party library imports
import numpy as np

# Local library imports
from src.mcmc.adapt import adapt
from src.mcmc.mwg.cyclic.update_ss import _update_ss
from src.mcmc.mwg.log_posterior_density import _compute_log_posterior_density
from src.mcmc.updates.hyperparams import step_sigma2c_update, step_sigma2w_update, step_zeta_update
from src.mcmc.updates.data import step_d_update

def step_mwg(sampler, iter_nr):

        # Compute the iteration index in the chunk
        verbose = sampler.mcmc_config['verbose']
        chunk_size = sampler.file_config['chunk_size']
        i = iter_nr % chunk_size
        if verbose:
            nv = sampler.lattice.nv
            
            print(f"Cyclic step_mwg. State at beginning of iteration i:")
            print(f"w_star[i + 1] (center): \n{sampler.theta['w_star'][(nv // 2 - 2):(nv // 2 + 2), i + 1]}\n")
            print(f"variances: sigma2w={sampler.theta['sigma2w'][0, i + 1]}, sigma2c={sampler.theta['sigma2c'][0, i + 1]}, zeta={sampler.theta['zeta'][0, i + 1]}, sigmawd={sampler.aux['sigma2d'][0, i + 1]}")
        
        try:
            step_wc_update(sampler, iter_nr)
            if verbose:
                print(f"\nState at iteration i, after (w, c) update:")
                print(f"w_star[i] (center): \n{sampler.theta['w_star'][(nv // 2 - 2):(nv // 2 + 2), i]}\n")
                print(f"w_star[i + 1] (center): \n{sampler.theta['w_star'][(nv // 2 - 2):(nv // 2 + 2), i + 1]}\n")
                print(f"variances: sigma2w={sampler.theta['sigma2w'][0, i + 1]}, sigma2c={sampler.theta['sigma2c'][0, i + 1]}, zeta={sampler.theta['zeta'][0, i + 1]}, sigmawd={sampler.aux['sigma2d'][0, i + 1]}")
            
            # Update d
            step_d_update(sampler, i)  
            if verbose:
                print(f"\nState at iteration i, after d update:")
                print(f"d_star[i] (center): \n{sampler.theta['d_star'][(nv // 2 - 2):(nv // 2 + 2), i]}\n")
                print(f"d_star[i + 1] (center): \n{sampler.theta['d_star'][(nv // 2 - 2):(nv // 2 + 2), i + 1]}\n")

            # Re-compute sum of squares used in the variance updates
            _update_ss(sampler, i) 

            # Update sigma_c
            step_sigma2c_update(sampler, i)

            # Update sigma_w  
            step_sigma2w_update(sampler, i)

            # Update zeta
            step_zeta_update(sampler, i)
            if verbose:
                print(f"\nState at iteration i, after variances update:")
                print(f"variances: sigma2w={sampler.theta['sigma2w'][0, i]}, sigma2c={sampler.theta['sigma2c'][0, i]}, zeta={sampler.theta['zeta'][0, i]}")
                print(f"variances: sigma2w={sampler.theta['sigma2w'][0, i + 1]}, sigma2c={sampler.theta['sigma2c'][0, i + 1]}, zeta={sampler.theta['zeta'][0, i + 1]}, sigmawd={sampler.aux['sigma2d'][0, i + 1]}")

        except Exception as e:
            print(f"An error occurred at iteration {i}, global {iter_nr}: \n{e}")
            # Set a flag to indicate an error occurred
            sampler.stats['mwg']['error'][iter_nr + 1] = 1
       
        # Compute the posterior log-density at the new state
        _compute_log_posterior_density(sampler, iter_nr)

        # Adapt epsilon and L; uses the global iteration index
        if sampler.mcmc_config['algorithm'] == 'collapsed_hmc':
            adapt(sampler, iter_nr, param='epsilon', alg='collapsed_hmc')
            adapt(sampler, iter_nr, param='L', alg='collapsed_hmc')

def step_wc_update(sampler, iter_nr):
    """Update (w,c) in cyclic lattices using the configured algorithm with adaptation"""
    algorithm = sampler.mcmc_config.get('algorithm') 
    verbose = sampler.mcmc_config.get('verbose', False)
    
    # Compute the iteration index in the chunk
    chunk_size = sampler.file_config['chunk_size']
    i = iter_nr % chunk_size
    
    print(f"Updating (w,c) at iteration {i} using algorithm: {algorithm}") if verbose else None

    # Estimate w, c, both or none?
    update_blur = 'w_star' in sampler.estimate
    update_image = 'c_star' in sampler.estimate
    print(f"Update blur? {update_blur}, update image? {update_image}") if verbose else None
    
    if algorithm == 'gibbs':
        from src.mcmc.update_blur_image.gibbs.step_gibbs import step_gibbs
        step_gibbs(sampler, i, update_blur=update_blur, update_image=update_image)
        
    elif algorithm == 'hmc':
        raise NotImplementedError("HMC algorithm is not implemented in cyclic MWG.")

    elif algorithm == 'collapsed_hmc':
        from src.mcmc.update_blur_image.collapsedhmc.cyclic.step_collapsed_hmc import step_collapsed_hmc
        from src.mcmc.update_blur_image.gibbs.step_gibbs import step_gibbs
            
        # Update w* with HMC or Gibbs if w_star is in estimate
        if update_blur:
            
            # Randomly choose between collapsed HMC and Gibbs
            sample_hmc = np.random.rand() < sampler.mcmc_config['collapsed_hmc'].get('p_collapsed_hmc', 0.05)
            print(f"(**) Update wavelet with HMC? ={sample_hmc}\n") if verbose else None
            if sample_hmc:
                sampler.stats['wc_update']['iter_with_chmc'][iter_nr] = 1 
                step_collapsed_hmc(sampler, iter_nr)
            else:
                step_gibbs(sampler, i, update_blur=True, update_image=False)
        
        # Update c* with a Gibbs step if c_star is in estimate
        if update_image:
            step_gibbs(sampler, i, update_blur=False, update_image=update_image)
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

