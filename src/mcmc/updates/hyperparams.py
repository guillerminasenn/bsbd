"""Gibbs updates for hyperparameters."""


def step_sigma2c_update(sampler, i):
    """Update sigma2c parameter."""

    verbose = sampler.mcmc_config.get('verbose')

    if 'sigma2c' in sampler.estimate:

        # Extract inverse-gamma object
        _sigma2c = sampler.model.model['sigma2c']

        # Recompute its parameters
        alpha, beta = _sigma2c.recompute_alpha_beta(sampler, i)

        # Sample from the inverse-gamma distribution with updated parameters
        sigma2c = _sigma2c.sample(alpha=alpha, beta=beta, verbose=verbose)
        if verbose:
            print(f"Recomputed alpha, beta for sigma2c at iteration {i}... alpha: {alpha}, beta: {beta}\n")

    else:
        sigma2c = sampler.theta['sigma2c'][0, i + 1]  # Copy previous value if not estimated

    # Save
    sampler.theta['sigma2c'][0, i + 1] = sigma2c

    # Recompute and store the observational noise variance
    psi = sampler.model.aux['psi']
    sigma2d = psi * sampler.theta['sigma2c'][0, i + 1] * sampler.theta['sigma2w'][0, i + 1] * sampler.theta['zeta'][0, i + 1]
    sampler.aux['sigma2d'][0, i + 1] = sigma2d

    if verbose:
        old_sigma2c = sampler.theta['sigma2c'][0, i]
        print(f"and updated sigma2c from {old_sigma2c} to sigma2c: {sigma2c} and sigma2d to: {sigma2d}\n")


def step_sigma2w_update(sampler, i):
    """Update sigma2w parameter."""
    verbose = sampler.mcmc_config.get('verbose')

    if 'sigma2w' in sampler.estimate:

        # Extract inverse-gamma object
        _sigma2w = sampler.model.model['sigma2w']

        # Recompute its parameters
        alpha, beta = _sigma2w.recompute_alpha_beta(sampler, i)

        # Sample from the inverse-gamma distribution with updated parameters
        sigma2w = _sigma2w.sample(alpha=alpha, beta=beta, verbose=verbose)

    else:
        sigma2w = sampler.theta['sigma2w'][0, i + 1]  # Copy previous value if not estimated

    # Save sigma2w
    sampler.theta['sigma2w'][0, i + 1] = sigma2w

    # Recompute and store the observational noise variance
    psi = sampler.model.aux['psi']
    sigma2d = psi * sampler.theta['sigma2c'][0, i + 1] * sampler.theta['sigma2w'][0, i + 1] * sampler.theta['zeta'][0, i + 1]
    sampler.aux['sigma2d'][0, i + 1] = sigma2d

    if verbose:
        old_sigma2w = sampler.theta['sigma2w'][0, i]
        print(f"and updated sigma2w from {old_sigma2w} to sigma2w: {sigma2w} and sigma2d to: {sigma2d}\n")


def step_zeta_update(sampler, i):
    """Update zeta parameter."""

    verbose = sampler.mcmc_config.get('verbose')

    if 'zeta' in sampler.estimate:
        # Extract inverse-gamma object
        _zeta = sampler.model.model['zeta']

        # Recompute its parameters
        alpha, beta = _zeta.recompute_alpha_beta(sampler, i)

        # Sample from the inverse-gamma distribution with updated parameters
        zeta = _zeta.sample(alpha=alpha, beta=beta, verbose=verbose)

    else:
        zeta = sampler.theta['zeta'][0, i + 1]  # Copy previous value if not estimated

    # Save zeta
    sampler.theta['zeta'][0, i + 1] = zeta

    # Recompute and store the observational noise variance
    psi = sampler.model.aux['psi']
    sigma2d = psi * sampler.theta['sigma2c'][0, i + 1] * sampler.theta['sigma2w'][0, i + 1] * sampler.theta['zeta'][0, i + 1]
    sampler.aux['sigma2d'][0, i + 1] = sigma2d

    if verbose:
        old_zeta = sampler.theta['zeta'][0, i]
        print(f"and updated zeta from {old_zeta} to zeta: {zeta} and sigma2d to: {sigma2d}\n")
