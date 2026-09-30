"""Sum-of-squares updates for Euclidean lattices.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.
"""

# third-party imports
from scipy import sparse

# Local library imports
from src.utils.model_utils import create_W

def _update_ss(sampler, i):
    """Update sum of squares used to recompute the scale parameter in the 
    variance full conditionals. Euclidean case, traditional matrix algebra.
    Parameters
    ----------
    sampler: MCMC object
        Contains sampler configuration, data, and starting value.
    i: int
        Iteration number.
    """

    verbose = sampler.mcmc_config.get('verbose')
    if verbose:
        print(f"Current sum of squares at iteration {i}, EUCLIDEAN case:")
        print(f"  likelihood_ss_unconstrained: {sampler.aux['likelihood_ss_unconstrained'][:, i + 1]}")
        print(f"  likelihood_ss_subset: {sampler.aux['likelihood_ss_subset'][:, i + 1]}")
        print(f"  likelihood_ss_constrained: {sampler.aux['likelihood_ss_constrained'][:, i + 1]}\n")
        print(f"  reflectivity_ss_unconstrained: {sampler.aux['reflectivity_ss_unconstrained'][:, i + 1]}")
        print(f"  reflectivity_ss_subset: {sampler.aux['reflectivity_ss_subset'][:, i + 1]}")
        print(f"  reflectivity_ss_constrained: {sampler.aux['reflectivity_ss_constrained'][:, i + 1]}\n")
        print(f"  wavelet_ss_unconstrained: {sampler.aux['wavelet_ss_unconstrained'][:, i + 1]}")
        print(f"  wavelet_ss_subset: {sampler.aux['wavelet_ss_subset'][:, i + 1]}")
        print(f"  wavelet_ss_constrained: {sampler.aux['wavelet_ss_constrained'][:, i + 1]}\n")
        print(f"Updating sum of squares for iteration {i}...\n")

    # Load current state
    w_star = sampler.theta['w_star'][:, i + 1].reshape(-1, 1)   
    c_star = sampler.theta['c_star'][:, i + 1].reshape(-1, 1) 
    d_star = sampler.theta['d_star'][:, i + 1].reshape(-1, 1) # constrained extended data
    w = sampler.aux['w'][:, i + 1].reshape(-1, 1) # unconstrained extended wavelet
    c = sampler.aux['c'][:, i + 1].reshape(-1, 1) # unconstrained extended reflectivity
    d = sampler.aux['d'][:, i + 1].reshape(-1, 1) # unconstrained extended data
    if verbose:
        print(f"Current state at iteration {i}:")
        print(f"w_star:\n{w_star}\n")
        print(f"c_star:\n{c_star}\n")
        print(f"d_star:\n{d_star}\n")
        print(f"d:\n{d}\n")

    # Compute likelihood sum of squares:
    _d = sampler.par_objs['d']
    W0, W, x1, x2, x3, x4 = create_W(sampler.lattice, w_star)
    mean_lik = W @ c_star
    Q_d = _d.Sigma.Q

    # Likelihood unconstrained SS
    d_centered = d - mean_lik # original
    # d_centered = d_star - mean_lik # testing
    likelihood_ss_unconstrained = d_centered.T @ Q_d @ d_centered
    sampler.aux['likelihood_ss_unconstrained'][:, i + 1] = likelihood_ss_unconstrained
    if verbose:
        print(f"The likeihood SS unconstrained = \n{likelihood_ss_unconstrained}")

    # Likelihood subset SS
    constr_SSD = sampler.mcmc_config.get('constr_SSD', False)
    if _d.data_constraints['constr'] and constr_SSD: 
        X_matrix = _d.data_constraints['X_matrix'] 
        U_matrix = _d.data_constraints['U_matrix'] 
        A_d = sparse.kron(X_matrix, U_matrix)
        Q_subset = _d.data_constraints['inv_ARAt']
        d_subset = A_d @ d
        mean_subset = A_d @ mean_lik
        d_subset_centered = d_subset - mean_subset
        likelihood_ss_subset = d_subset_centered.T @ Q_subset @ d_subset_centered
        if verbose:
            print(f"Data is constrained. The constrained part is Ad @ d = {d_subset} ")
            print(f"The SS subset = \n{likelihood_ss_subset}")
    else:
        likelihood_ss_subset = 0  # No constraints, SS is zero
        if verbose:
            print(f"Data is not constrained.")
            print(f"The SS subset = \n{likelihood_ss_subset}")
    sampler.aux['likelihood_ss_subset'][:, i + 1] = likelihood_ss_subset

    # Likelihood constrained SS 
    likelihood_ss_constrained = likelihood_ss_unconstrained - likelihood_ss_subset
    sampler.aux['likelihood_ss_constrained'][:, i + 1] = likelihood_ss_constrained
    if verbose:
        print(f"The likelihood SS constrained = \n{likelihood_ss_constrained}\n\n")


    # Reflectivity SS
    _c = sampler.par_objs['c']
    mean_c = _c.mean.reshape(-1, 1)  # Ensure mean_c is a column vector
    Q_c = _c.Sigma.Q

    # The reflectivity SS is the same for collapsed HMC and non-collapsed HMC (because I'm sampling c with a Gibbs step)
    c_centered = c - mean_c 
    # c_centered = c_star - mean_c # testing
    reflectivity_ss_unconstrained = c_centered.T @ Q_c @ c_centered
    sampler.aux['reflectivity_ss_unconstrained'][:, i + 1] = reflectivity_ss_unconstrained
    if verbose:
        print(f"Reflectivity unconstrained SS = \n{reflectivity_ss_unconstrained}")

    # Reflectivity subset SS
    if _c.reflectivity_constraints['constr']:
        A_c = _c.reflectivity_constraints['A_co'] # reflectivity constraint matrix
        c_subset = A_c @ c
        mean_c_subset = A_c @ mean_c
        c_subset_centered = c_subset - mean_c_subset
        Q_c_subset = _c.reflectivity_constraints['inv_ARAt']
        reflectivity_ss_subset = c_subset_centered.T @ Q_c_subset @ c_subset_centered
        if verbose:
            print(f"Reflectivity is constrained.")
            print(f"c_subset: {c_subset}")
            print(f"mean_c_subset: {mean_c_subset}")
            print(f"reflectivity_ss_subset = \n{reflectivity_ss_subset}")

    else:
        reflectivity_ss_subset = 0
        if verbose:
            print(f"Reflectivity is not constrained.")
            print(f"reflectivity_ss_subset = \n{reflectivity_ss_subset}")
    sampler.aux['reflectivity_ss_subset'][:, i + 1] = reflectivity_ss_subset

    # Reflectivity constrained SS 
    reflectivity_ss_constrained = reflectivity_ss_unconstrained - reflectivity_ss_subset
    sampler.aux['reflectivity_ss_constrained'][:, i + 1] = reflectivity_ss_constrained
    if verbose:
        print(f"The reflectivity SS constrained = \n{reflectivity_ss_constrained}\n\n")


    # Wavelet SS
    _w = sampler.par_objs['w']
    A_w = _w.wavelet_constraints['A'] # wavelet constraints matrix
    mean_w = _w.mean
    Q_w = _w.Sigma.Q

    # If algorithm is not collapsed HMC, I have the unconstrained SS, and optionally subset and constrained SS
    if sampler.mcmc_config['algorithm'] != 'collapsed_hmc':

        # Wavelet unconstrained SS
        w_centered = w - mean_w 
        # w_centered = w_star - mean_w # testing
        wavelet_ss_unconstrained = w_centered.T @ Q_w @ w_centered
        sampler.aux['wavelet_ss_unconstrained'][:, i + 1] = wavelet_ss_unconstrained
        if verbose:
            print(f"The wavelet SS unconstrained = \n{wavelet_ss_unconstrained}")

        # Wavelet subset SS
        if _w.wavelet_constraints['nr_constraints'] > 0 or sampler.lattice.topology == 'C':
            w_subset = A_w @ w
            mean_w_subset = A_w @ mean_w
            w_subset_centered = w_subset - mean_w_subset
            Q_w_subset = _w.wavelet_constraints['inv_ARAt']
            wavelet_ss_subset = w_subset_centered.T @ Q_w_subset @ w_subset_centered
            if verbose:
                print(f"Wavelet is constrained.")
                print(f"w_subset: {w_subset}")
                print(f"mean_w_subset: {mean_w_subset}")
                print(f"wavelet_ss_subset = \n{wavelet_ss_subset}")
        else:
            wavelet_ss_subset = 0
            if verbose:
                print(f"Wavelet is not constrained.")
                print(f"wavelet_ss_subset = \n{wavelet_ss_subset}")
        sampler.aux['wavelet_ss_subset'][:, i + 1] = wavelet_ss_subset

        # Wavelet constrained SS  
        wavelet_ss_constrained = wavelet_ss_unconstrained - wavelet_ss_subset
        sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_constrained
        if verbose:
            print(f"The wavelet SS constrained = \n{wavelet_ss_constrained}")

    # If algorithm is collapsed HMC, I sample w* from the (constrained) Gaussian
    if sampler.mcmc_config['algorithm'] == 'collapsed_hmc':

        # Wavelet subset SS 
        sampler.aux['wavelet_ss_subset'][:, i + 1] = None
        if verbose:
            print(f"Wavelet subset SS is None because the algorithm is collapsed HMC.")

        # Wavelet constrained SS
        if _w.wavelet_constraints['nr_constraints'] > 0 or sampler.lattice.topology == 'C':
            # Wavelet unconstrained SS
            sampler.aux['wavelet_ss_unconstrained'][:, i + 1] = None
            if verbose:
                print(f"Wavelet unconstrained SS is None because the wavelet is constrained and algorithm is collapsed HMC.")

            mean_w_star = _w.wavelet_constraints['mean_w_star']
            w_star_centered = w_star - mean_w_star
            inv_R_w_star = _w.wavelet_constraints['inv_R_w_star']
            wavelet_ss_constrained = w_star_centered.T @ inv_R_w_star @ w_star_centered

            # Only the constrained wavelet SS is available
            sampler.aux['wavelet_ss_constrained'][:, i + 1] = wavelet_ss_constrained
            if verbose:
                print(f"wavelet_ss_constrained = \n{wavelet_ss_constrained}")

        else:
            # Wavelet unconstrained SS
            w_centered = w_star - mean_w # The w_star is unconstrained
            # w_centered = w_star - mean_w # testing
            wavelet_ss_unconstrained = w_centered.T @ Q_w @ w_centered
            sampler.aux['wavelet_ss_unconstrained'][:, i + 1] = wavelet_ss_unconstrained
            sampler.aux['wavelet_ss_constrained'][:, i + 1] = None
            if verbose:
                print(f"Wavelet unconstrained SS = \n{wavelet_ss_unconstrained}")
                print(f"Wavelet constrained SS is None there are no constraints on wavelet.\n\n")

    # Print debug information
    if verbose:
        print(f"Updated likelihood sum of squares at iteration {i}:")
        print(f"  likelihood_ss_unconstrained: {sampler.aux['likelihood_ss_unconstrained'][:, i + 1]}")
        print(f"  likelihood_ss_subset: {sampler.aux['likelihood_ss_subset'][:, i + 1]}")
        print(f"  likelihood_ss_constrained: {sampler.aux['likelihood_ss_constrained'][:, i + 1]}\n")
        print(f"  reflectivity_ss_unconstrained: {sampler.aux['reflectivity_ss_unconstrained'][:, i + 1]}")
        print(f"  reflectivity_ss_subset: {sampler.aux['reflectivity_ss_subset'][:, i + 1]}")
        print(f"  reflectivity_ss_constrained: {sampler.aux['reflectivity_ss_constrained'][:, i + 1]}\n")
        print(f"  wavelet_ss_unconstrained: {sampler.aux['wavelet_ss_unconstrained'][:, i + 1]}")
        print(f"  wavelet_ss_subset: {sampler.aux['wavelet_ss_subset'][:, i + 1]}")
        print(f"  wavelet_ss_constrained: {sampler.aux['wavelet_ss_constrained'][:, i + 1]}\n")
