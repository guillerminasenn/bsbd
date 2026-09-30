"""State management helpers for MCMC sampling."""


def _copy_previous_state(sampler, i):
    """Copy state from iteration i to i+1."""
    sampler.theta['d_star'][:, i + 1] = sampler.theta['d_star'][:, i].reshape(-1)
    sampler.theta['w_star'][:, i + 1] = sampler.theta['w_star'][:, i].reshape(-1)
    sampler.theta['c_star'][:, i + 1] = sampler.theta['c_star'][:, i].reshape(-1)
    sampler.theta['sigma2c'][0, i + 1] = sampler.theta['sigma2c'][0, i]
    sampler.theta['sigma2w'][0, i + 1] = sampler.theta['sigma2w'][0, i]
    sampler.theta['zeta'][0, i + 1] = sampler.theta['zeta'][0, i]

    sampler.aux['d'][:, i + 1] = sampler.aux['d'][:, i].reshape(-1)
    sampler.aux['c'][:, i + 1] = sampler.aux['c'][:, i].reshape(-1)
    sampler.aux['w'][:, i + 1] = sampler.aux['w'][:, i].reshape(-1)
    sampler.aux['sigma2d'][0, i + 1] = sampler.aux['sigma2d'][0, i]


copy_previous_state = _copy_previous_state
