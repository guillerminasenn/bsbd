"""Metropolis-within-Gibbs (MWG) driver and topology dispatch.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.
"""

# Standard library imports
import time

# Third-party library imports
import numpy as np

# Algorithm imports
from src.mcmc.updates.state import _copy_previous_state

# Get the correct step_mwg function based on topology
def get_functions(topology):
    step_mwg = None
    _update_ss = None
    _compute_log_posterior_density = None

    if topology == 'E':
        from src.mcmc.mwg.euclidean.step_mwg import step_mwg
        from src.mcmc.mwg.euclidean.update_ss import _update_ss
        from src.mcmc.mwg.log_posterior_density import _compute_log_posterior_density
    elif topology == 'C':
        from src.mcmc.mwg.cyclic.step_mwg import step_mwg
        from src.mcmc.mwg.cyclic.update_ss import _update_ss
        from src.mcmc.mwg.log_posterior_density import _compute_log_posterior_density
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    return step_mwg, _update_ss, _compute_log_posterior_density

def mwg(sampler, **kwargs):
    """Full MwG sampler with adaptation for (w,c) update method"""
    
    # Get the step_mwg function based on the topology
    functions = get_functions(sampler.lattice.topology)
    step_mwg = functions[0]
    _update_ss = functions[1]
    _compute_log_posterior_density = functions[2]

    # Initialize the sampler configuration
    N = sampler.mcmc_config['N']
    verbose = sampler.mcmc_config.get('verbose', False)
    algorithm = sampler.mcmc_config.get('algorithm') 

    # Initialize the chunk index and chunk size
    chunk_size = sampler.file_config['chunk_size']
    if chunk_size > N:
        # The implementation saves only full chunks. For debug runs (or any run)
        # where N < chunk_size, pin chunk_size=N so we still save exactly one
        # complete chunk and avoid stepping past the allocated stats arrays.
        chunk_size = N
        sampler.file_config['chunk_size'] = chunk_size

    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_size={chunk_size}. Must be >= 1.")

    if N % chunk_size != 0:
        raise ValueError(
            f"N={N} must be a multiple of chunk_size={chunk_size} for the current "
            "chunk-saving implementation. Choose chunk_size that divides N (or set chunk_size=N)."
        )
    chunk_index = 0

    # Initialize the sample and auxiliary dictionaries
    sampler._create_sample_aux_dicts(N=chunk_size + 1, verbose=verbose) # + 1 to account for the initial state, but each file saves chunk_size iters
    sampler._initialize_sample_aux_dicts(i=0, verbose=verbose)

    # Initialize the sum of squares and the stats dictionaries; -1 to indicate initialization with the initial state
    _update_ss(sampler, -1) # 
    if _compute_log_posterior_density is not None:
        _compute_log_posterior_density(sampler, -1)
    
    # Initialize algorithm-specific setup (like momentum objects for HMC)
    _initialize_joint_update_algorithm(sampler, **kwargs)
    if verbose:
        print(f"Initialized {algorithm} with parameters: {kwargs}")
        print(f"In mwg.py, the adapt config dict=\n{sampler.adapt_config}")
    
    # Main sampling loop
    for iter_nr in np.arange(N + 1):  # +1 so we enter the save branch at the end of the last chunk

        # Calculate the current iteration index within the chunk
        i = iter_nr % chunk_size
        if iter_nr % 1000 == 0 and algorithm == 'collapsed_hmc':
            print(f'iter={iter_nr} and i={i} of chunk_size={chunk_size}')
        elif iter_nr % 1000 == 0:
            print(f'iter={iter_nr} and i={i} of chunk_size={chunk_size}')
            
        # If it's the chunk boundary (including the final boundary at iter_nr == N)
        if i % chunk_size == 0 and iter_nr > 0:

            # Save the current chunk_size iterations to file
            sampler.save_samples(chunk_index=chunk_index)

            # Update the stats file
            sampler.stats['mwg']['exec_time'] = time.time() - sampler.stats['mwg']['init_time'] # calculate elapsed time
            sampler.save_stats()

            # Stop if we have reached the total number of iterations.
            # Keep `sampler.theta`/`sampler.aux` intact so callers (e.g. notebooks)
            # can inspect the last chunk in-memory.
            if iter_nr == N:
                break

            # Store last state in theta_temp
            theta_temp = {
                'd_star': sampler.theta['d_star'][:, chunk_size],
                'c_star': sampler.theta['c_star'][:, chunk_size],
                'w_star': sampler.theta['w_star'][:, chunk_size],
                'sigma2c': sampler.theta['sigma2c'][:, chunk_size],
                'sigma2w': sampler.theta['sigma2w'][:, chunk_size],
                'zeta': sampler.theta['zeta'][:, chunk_size],
            }
            aux_temp = {
                'd': sampler.aux['d'][:, chunk_size],
                'c': sampler.aux['c'][:, chunk_size],
                'w': sampler.aux['w'][:, chunk_size],
                'sigma2d': sampler.aux['sigma2d'][:, chunk_size]
            }
            sampler.theta_temp = theta_temp
            sampler.aux_temp = aux_temp

            # Replace theta with a new, empty chunk
            print(f"\n(mwg.py) Current theta dict., before saving: \n{sampler.theta}") if verbose else None
            sampler._create_sample_aux_dicts(N=chunk_size)
            sampler._initialize_sample_aux_dicts(i=iter_nr, verbose=verbose)
            print(f"\n(mwg.py) New theta dict., after saving: \n{sampler.theta}") if verbose else None

            # Set the new chunk index
            chunk_index += 1


        # Set the (i+1)-th state equal to the current i-th state
        _copy_previous_state(sampler, i)

        # Perform the Metropolis-within-Gibbs step; pass the global iteration number 
        step_mwg(sampler, iter_nr)
            
def _initialize_joint_update_algorithm(sampler, **kwargs):
    """Initialize method-specific objects (like momentum for HMC)"""
    algorithm = sampler.mcmc_config.get('algorithm')
    print(f"Initializing {algorithm}.")
    if algorithm == 'collapsed_hmc':
        from src.mcmc.update_blur_image.collapsedhmc.initialize import _initialize_collapsed_hmc, _create_momentum_object
        _initialize_collapsed_hmc(sampler, **kwargs) 
        _create_momentum_object(sampler)