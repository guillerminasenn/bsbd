"""This module contains functions for parameter adaptation. 
Functions:
adapt: driver to call adaptation functions
adapt_rosenthal
adapt_random
"""

# Third-party library imports
import numpy as np

def adapt(sampler, i, param, alg):
    """Perform adaptation step for given parameter of given algorithm.
    Params:
    -------
    param: beta, epsilon, L
    alg: pcn, mpcn, hmc, collapsed_hmc
    """
    
    try:
        config = sampler.adapt_config[alg][param]
    except KeyError:
        return  # nothing to adapt
    
    # If no adaptation
    if config.get('type') is None:
        try:
            # print(f'alg={alg}, param={param}')
            # print(f"i={i}; sampler.adapt_tracking[alg][param]['history'][i - 1] = {sampler.adapt_tracking[alg][param]['history'][i - 1]}")
            sampler.adapt_tracking[alg][param]['history'][i] = sampler.adapt_tracking[alg][param]['history'][i - 1]
        except:
            print(f'adapt_tracking was not updated with the current param value, iter={i}')
        return
    
    # verbose = sampler.mcmc_config.get('verbose', False)
    # print(f"In adapt.py, calling the adapt functions for param={param}, with adapt config dict=\n{sampler.adapt_config}") if verbose else None
    # print(f"Adapting {param} in {alg} at iter={i} with adaptation type = {sampler.adapt_config[alg][param]['type']}.")
        
    # Marc Suchard's adaptation
    if config['type'] == 'marc':
        adapt_marc(sampler, i, param, alg)

    # Rosenthal-type adaptation
    elif config['type'] == 'rosenthal':
        adapt_rosenthal(sampler, i, param, alg)

    # Random adaptation
    elif config['type'] == 'random':
        adapt_random(sampler, i, param, alg)

    # Other adaptation types could go here
    else:
        raise ValueError(f"Unknown adaptation type: {config['type']}")

def adapt_marc(sampler, i, param, alg, **kwargs):
    """Determine the next value for a parameter using Marc Suchard's
    group adaptation method.
    If the batch acceptance rate is larger than some optimal acceptance rate, 
    be more agressive and increase beta. Otherwise, be less agressive and decrease beta. 
    
    Params:
    -------
    sampler:
    i: Global iter number.
    param:
    alg:
    """
    verbose = sampler.mcmc_config.get('verbose', False)

    # Read the configuration for the adaptation and set default batch size 
    config = sampler.adapt_config[alg][param]
    config.setdefault('B', 50)
    verbose = sampler.mcmc_config.get('verbose', False)

    # If no adaptation at this iteration 
    p_collapsed_hmc = sampler.mcmc_config['collapsed_hmc'].get('p_collapsed_hmc', 0.05)
    if p_collapsed_hmc < 1:
        nr_chmc = np.sum(sampler.stats['wc_update']['iter_with_chmc'])  # use the nr of HMC iterations so far
        if (nr_chmc) % config['B'] > 0 or sampler.stats['wc_update']['iter_with_chmc'][i] == 0:
            adapted_value = sampler.adapt_tracking[alg][param]['history'][i]
            sampler.adapt_tracking[alg][param]['history'][i + 1] = adapted_value
            print(f'No adaptation at iter={i} for {param} in {alg}.') if verbose else None
            return None
    
    elif p_collapsed_hmc == 1:
        if (i + 1) % config['B'] > 0:
            adapted_value = sampler.adapt_tracking[alg][param]['history'][i]
            sampler.adapt_tracking[alg][param]['history'][i + 1] = adapted_value
            print(f'No adaptation at iter={i} for {param} in {alg}.') if verbose else None
            return None
        
    # Continue reading config
    config.setdefault('optimal_ar', 0.5)
    config.setdefault('verbose', False)
    
    # Compute batch acceptance rate, allowing for hybrid algorithm
    if p_collapsed_hmc < 1:
        chmc_index = np.where(sampler.stats['wc_update']['iter_with_chmc'] == 1)[0]
        was_proposal_accepted = sampler.stats['wc_update']['acceptance']['acc_iter'][chmc_index] # subset only those iterations done with HMC
        acc_vector = np.array(was_proposal_accepted[-(config['B']):])
    elif p_collapsed_hmc == 1:
        lower_endpoint = ((i + 1) // config['B'] - 1) * config['B']
        upper_endpoint = lower_endpoint + config['B']
        was_proposal_accepted = sampler.stats['wc_update']['acceptance']['acc_iter']
        acc_vector = np.array(was_proposal_accepted[lower_endpoint:upper_endpoint])
        nr_chmc = i

    accepted_proposals_count = config['B'] - np.sum(acc_vector == 0)
    batch_ar = (accepted_proposals_count / config['B']).item()
    if verbose:
        print(f'Adapting at iter = {i}, which corresponds to hybrid iter = {nr_chmc}.') 
        print(f"Iterations with hmc:\n{sampler.stats['wc_update']['iter_with_chmc']}\n")
        print(f"sampler.acceptance[acc_iter] = \n{sampler.stats['wc_update']['acceptance']['acc_iter']}\n")
        print(f'acc_vector:\n{acc_vector}\n')
        print(f"accepted_proposals_count={accepted_proposals_count}")
        print(f'batch_ar = {batch_ar}')

    # Current parameter value
    beta = sampler.adapt_tracking[alg][param]['history'][i] # current value
    print(f'Current beta:\n{beta}') if verbose else None
    
    # Compute acceptance rate ratio
    optimal_ar = config['optimal_ar']
    ratio = batch_ar / optimal_ar
    if ratio >= 1:
        print(f"Ratio = {ratio}") if verbose else None
        if ratio > 2:
            ratio = 2
            print(f"Ratio > 2, bounding it at 2.") if verbose else None
        beta_adapted = beta * ratio
        print(f"Batch acc. rate too high ({batch_ar} > {optimal_ar}); increasing param from {beta} to {beta_adapted}") if verbose else None
    if ratio < 1:
        print(f"Ratio = {ratio}") if verbose else None
        if ratio < 0.5:
            ratio = 0.5
            print(f"Ratio < 0.5, bounding it at 0.5.") if verbose else None
        beta_adapted = beta * ratio
        print(f"Batch acc. rate too low ({batch_ar} < {optimal_ar}); decreaseing param from {beta} to {beta_adapted}") if verbose else None
        
    # Save new beta and batch acceptance rate
    sampler.adapt_tracking[alg][param]['batch_ar'].append(batch_ar)
    sampler.adapt_tracking[alg][param]['history'][(i + 1):] = beta_adapted
    if verbose:
        print(f'Updated beta in the adapt_tracking history[{i}+1]= {sampler.adapt_tracking[alg][param]["history"][i + 1]}.\n')
        print(f"Updated batch acceptance rate in the adapt_tracking history[{i}+1]= {sampler.adapt_tracking[alg][param]['batch_ar'][-1]}.\n")
        print(f"The whole history of beta:\n{sampler.adapt_tracking[alg][param]['history']}\n")
        print(f"The whole history of batch acceptance rates:\n{sampler.adapt_tracking[alg][param]['batch_ar']}\n")    
    
def adapt_rosenthal(sampler, i, param, alg, **kwargs):
    """Determine the next value for a parameter using the
    algorithm in p.18 in Optimal Proposal Distributions and 
    Adaptive MCMC, Rosenthal (2008).
    If the batch acceptance rate is larger than some optimal acceptance rate, 
    be more agressive and increase beta. Otherwise, be less agressive and decrease beta. 
    
    Params:
    -------
    sampler:
    i: Global iter number.
    param:
    alg:
    """

    verbose = sampler.mcmc_config.get('verbose', False)
    # print(f"In adapt_rosenthal in adapt.py, adapt config dict=\n{sampler.adapt_config}") if verbose else None

    history = sampler.adapt_tracking[alg][param]['history']
    if i + 1 >= len(history):
        print(f"Skipping adaptation at iter={i} for {param} in {alg} (history length {len(history)}).") if verbose else None
        return None

    # Read the configuration for the adaptation and set default batch size 
    config = sampler.adapt_config[alg][param]
    config.setdefault('B', 50)

    # If no adaptation at this iteration 
    p_collapsed_hmc = sampler.mcmc_config['collapsed_hmc'].get('p_collapsed_hmc', 0.05)
    if p_collapsed_hmc < 1:
        nr_chmc = np.sum(sampler.stats['wc_update']['iter_with_chmc'])  # use the nr of HMC iterations so far
        if nr_chmc % config['B'] > 0 and sampler.stats['wc_update']['iter_with_chmc'][i] == 0:
            adapted_value = history[i]
            history[i + 1] = adapted_value
            print(f'No adaptation at iter={i} for {param} in {alg}.') if verbose else None
            return None
    
    elif p_collapsed_hmc == 1:
        if (i + 1) % config['B'] > 0:
            adapted_value = history[i]
            history[i + 1] = adapted_value
            print(f'No adaptation at iter={i} for {param} in {alg}.') if verbose else None
            return None
        
    # Continue reading config
    config.setdefault('optimal_ar', 0.5)
    config.setdefault('max_change_ros', 1e-3)
    config.setdefault('min_beta_ros', 1e-8)
    config.setdefault('max_beta_ros', 1)
    config.setdefault('verbose', False)
    config.setdefault('exponent_ros', 0.5)
    
    # Compute batch acceptance rate, allowing for hybrid algorithm
    if p_collapsed_hmc < 1:
        chmc_index = np.where(sampler.stats['wc_update']['iter_with_chmc'] == 1)[0]
        was_proposal_accepted = sampler.stats['wc_update']['acceptance']['acc_iter'][chmc_index] # subset only those iterations done with HMC
        acc_vector = np.array(was_proposal_accepted[-(config['B']):])
    elif p_collapsed_hmc == 1:
        lower_endpoint = ((i + 1) // config['B'] - 1) * config['B']
        upper_endpoint = lower_endpoint + config['B']
        was_proposal_accepted = sampler.stats['wc_update']['acceptance']['acc_iter']
        acc_vector = np.array(was_proposal_accepted[lower_endpoint:upper_endpoint])

    accepted_proposals_count = config['B'] - np.sum(acc_vector == 0)
    # print(f"accepted_proposals_count={accepted_proposals_count}")
    batch_ar = (accepted_proposals_count / config['B']).item()
    if verbose:
        print(f'Adapting at iter = {i}, which corresponds to hybrid iter = {nr_chmc}.')
        print(f"Iterations with hmc:\n{sampler.stats['wc_update']['iter_with_chmc']}\n")
        print(f"sampler.acceptance[acc_iter] = \n{sampler.stats['wc_update']['acceptance']['acc_iter']}\n")
        print(f'acc_vector:\n{acc_vector}\n')
        print(f"accepted_proposals_count={accepted_proposals_count}")
        print(f'batch_ar = {batch_ar}')

    # Adaptation rule Delta(i)
    exponent = config['exponent_ros']
    print(f"exponent = {exponent}") if verbose else None
    change = i ** (-exponent)
    delta = np.min([config['max_change_ros'], change])
    if verbose:
        print(f'proposed change:\n{change}; max_change = {config["max_change_ros"]}')
        print(f'delta (the effective change):\n{delta}.\n')

    # Adapt
    beta = history[i] # current value
    print(f'Current beta:\n{beta}') if verbose else None
        
    if batch_ar > config['optimal_ar']:
        beta_adapted = beta + delta
        beta_adapted = np.min([beta_adapted, config['max_beta_ros']])
        print(f'Batch AR = {batch_ar} > {config["optimal_ar"]}; increase beta. \nNew beta={beta_adapted}.\n') if verbose else None
    else: 
        beta_adapted = beta - delta
        beta_adapted = np.max([beta_adapted, config['min_beta_ros']])
        print(f'Batch AR = {batch_ar} <= {config["optimal_ar"]}; decrease beta. \nNew beta={beta_adapted}.\n') if verbose else None

    # Save new beta and batch acceptance rate
    sampler.adapt_tracking[alg][param]['batch_ar'].append(batch_ar)
    sampler.adapt_tracking[alg][param]['history'][(i + 1):] = beta_adapted
    if verbose:
        print(f'Updated beta in the adapt_tracking history[{i}+1]= {sampler.adapt_tracking[alg][param]["history"][i + 1]}.\n')
        print(f"Updated batch acceptance rate in the adapt_tracking history[{i}+1]= {sampler.adapt_tracking[alg][param]['batch_ar'][-1]}.\n")
        print(f"The whole history of beta:\n{sampler.adapt_tracking[alg][param]['history']}\n")
        print(f"The whole history of batch acceptance rates:\n{sampler.adapt_tracking[alg][param]['batch_ar']}\n")

def adapt_random(sampler, i, param, alg):
    """Adapt the parameter by sampling a new value from an uniform
    distribution."""
    config = sampler.adapt_config[alg][param]
    history = sampler.adapt_tracking[alg][param]['history']
    if i + 1 >= len(history):
        return None
    updated_value = np.random.uniform(config.get('min', 0.001), config.get('max', 0.01))
    history[i + 1] = updated_value
    # if config['verbose']: 
    #         print(f'Current beta = {sampler.adapt_tracking[alg][param]["history"][i]}.')
    #         print(f'New beta = {updated_value}.\n')

    