"""Gibbs updates for data-related parameters."""

# Third-party imports
import copy

# Local imports
from src.utils.efficient_algebra_utils import multiply_matrix_vector_circ
from src.utils.model_utils import create_W


def step_d_update(sampler, i):
    """Update d parameter."""

    if sampler.mcmc_config.get('verbose'):
        print(f"Updating d_star at iteration {i}...")

    # Extract SeismicData object
    _d = sampler.par_objs['d']
    constr = _d.data_constraints['constr']
    verbose = sampler.mcmc_config.get('verbose')

    # Recompute and store the observational noise variance
    psi = sampler.model.aux['psi']
    sigma2d = psi * sampler.theta['sigma2c'][0, i + 1] * sampler.theta['sigma2w'][0, i + 1] * sampler.theta['zeta'][0, i + 1]
    if verbose:
        print(f"psi = {psi}, sigma2d = {sigma2d}")

    if not constr and 'd_star' in sampler.estimate:
        raise Exception("If the lattice is not extended, 'd' must not be estimated. Do not pass d_star in the estimate list.")

    if 'd_star' in sampler.estimate:

        # Sample d from the unconstrained Gaussian distribution
        w_star = sampler.theta['w_star'][:, i + 1].reshape(-1, 1)
        c_star = sampler.theta['c_star'][:, i + 1].reshape(-1, 1)
        W0, W, base_W0, base_W, base_W0T, base_WT = create_W(sampler.lattice, w_star)
        convolution = multiply_matrix_vector_circ(base_W, c_star)
        d = _d.sample(mean=convolution, sigma2=sigma2d)  # handles cyclic and Euclidean
        sampler.aux['d'][:, i + 1] = d.reshape(-1)

        # Correct if constraints are present
        d_star = copy.deepcopy(d)
        if constr:
            d_star = _d.correct_sample(d, verbose=False)
        sampler.theta['d_star'][:, i + 1] = d_star.reshape(-1)

    if 'd_star' not in sampler.estimate:

        # Copy the previous value
        sampler.aux['d'][:, i + 1] = sampler.aux['d'][:, i]
        sampler.theta['d_star'][:, i + 1] = sampler.theta['d_star'][:, i]
