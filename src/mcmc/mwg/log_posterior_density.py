"""Compute the log-posterior density.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.

This implementation is topology-agnostic and uses quantities already
computed by the sampler (e.g., sum of squares and log-determinants).
"""
# Third-party imports
import numpy as np


def _compute_log_posterior_density(sampler, iter_nr):
    """Compute the log-posterior density at the current state.

    Parameters:
    ----------
    sampler: MCMC object
        Contains sampler configuration, data, and starting value.
    iter_nr: int
        Global iteration number.
    """

    # Extract mcmc configuration
    verbose = sampler.mcmc_config.get('verbose')

    if verbose:
        print(f"\nComputing log-posterior density at iteration {iter_nr}...")
        print("Current state:")
        if iter_nr == -1:
            print({key: sampler.theta[key][:5, 0] for key in sampler.theta})
        else:
            chunk_size = sampler.file_config.get('chunk_size')
            i = iter_nr % chunk_size
            print({key: sampler.theta[key][:5, i + 1] for key in sampler.theta})

    # Compute the log-likelihood at iter_nr
    ld_d = _compute_log_likelihood(sampler, iter_nr)

    # Compute the log-prior density for the reflectivity
    ld_c = _compute_log_prior_reflectivity(sampler, iter_nr)

    # Compute the log-prior density for the wavelet
    ld_w = _compute_log_prior_wavelet(sampler, iter_nr)

    # Compute the log-prior density for the reflectivity variance
    ld_sigma2c = _compute_log_prior_sigma2c(sampler, iter_nr)

    # Compute the log-prior density for the wavelet variance
    ld_sigma2w = _compute_log_prior_sigma2w(sampler, iter_nr)

    # Compute the log-prior density for the inverse SNR
    ld_zeta = _compute_log_prior_zeta(sampler, iter_nr)

    # log-posterior-density
    log_post = ld_d + ld_c + ld_w + ld_sigma2c + ld_sigma2w + ld_zeta

    # Store the log-posterior density in the stats dictionary
    sampler.stats['log_post']['likelihood'][:, iter_nr + 1] = ld_d
    sampler.stats['log_post']['prior_reflectivity'][:, iter_nr + 1] = ld_c
    sampler.stats['log_post']['prior_wavelet'][:, iter_nr + 1] = ld_w
    sampler.stats['log_post']['prior_sigma2c'][:, iter_nr + 1] = ld_sigma2c
    sampler.stats['log_post']['prior_sigma2w'][:, iter_nr + 1] = ld_sigma2w
    sampler.stats['log_post']['prior_zeta'][:, iter_nr + 1] = ld_zeta
    sampler.stats['log_post']['joint'][:, iter_nr + 1] = log_post

    if verbose:
        print(f"Log-posterior density computation at iteration {iter_nr}:")
        print(f"  log-likelihood = {ld_d}")
        print(f"  log-prior reflectivity = {ld_c}")
        print(f"  log-prior wavelet = {ld_w}")
        print(f"  log-prior sigma2c = {ld_sigma2c}")
        print(f"  log-prior sigma2w = {ld_sigma2w}")
        print(f"  log-prior snr = {ld_zeta}")
        print(f"  Total log-posterior density = {log_post}\n")


def _compute_log_likelihood(sampler, iter_nr):
    """Compute the log-likelihood at the current state.

    Parameters:
    ----------
    sampler: MCMC object
        Contains sampler configuration, data, and starting value.
    iter_nr: int
        Global iteration number.
    """
    # Extract mcmc configuration
    verbose = sampler.mcmc_config.get('verbose')

    # Compute the iteration index in the chunk
    if iter_nr == -1:
        i = -1
    else:
        chunk_size = sampler.file_config.get('chunk_size')
        i = iter_nr % chunk_size

    # Extract current state for the observational variance
    sigma2d = sampler.aux['sigma2d'][0, i + 1]  # data variance
    print(f"Computing log-likelihood at iter {iter_nr}, i={i}, sigma2d={sigma2d}") if verbose else None

    # Compute the logdensity
    n = sampler.lattice.n
    _d = sampler.par_objs['d']

    term1 = - n / 2 * np.log(2 * np.pi)
    term2_sigma = -0.5 * n * np.log(sigma2d)
    term2_logdets = (
        - 0.5 * _d.Sigma.logdet_R
    )
    term2 = term2_sigma + term2_logdets

    # Compute the term involving the sum of squares
    term3 = -0.5 * sampler.aux['likelihood_ss_unconstrained'][:, i + 1] / sigma2d

    # Total log-likelihood
    log_density = term1 + term2 + term3

    if verbose:
        print(f"\nLog-likelihood computation at iteration {iter_nr}:")
        print(f"sigma2d = {sigma2d}")
        print(f"term2 (log-determinants term) = {term2} (sigma part: {term2_sigma}, logdet part: {term2_logdets})")
        print(f"SSunc = {sampler.aux['likelihood_ss_unconstrained'][:, i + 1]}, term3 (sum of squares term) = {term3}")
        print(f"Total log-likelihood = {log_density}\n")
    return log_density


def _compute_log_prior_reflectivity(sampler, iter_nr):
    """Compute the log-prior reflectivity at the current state.

    Parameters:
    ----------
    sampler: MCMC object
        Contains sampler configuration, data, and starting value.
    iter_nr: int
        Global iteration number.
    """
    # Extract mcmc configuration
    verbose = sampler.mcmc_config.get('verbose')

    # Compute the iteration index in the chunk
    if iter_nr == -1:
        i = -1
    else:
        chunk_size = sampler.file_config.get('chunk_size')
        i = iter_nr % chunk_size

    # Extract current state for the observational variance
    sigma2c = sampler.theta['sigma2c'][0, i + 1]  # reflectivity variance

    # Compute the term involving 2*pi
    n = sampler.lattice.n
    _c = sampler.par_objs['c']
    term1 = - n / 2 * np.log(2 * np.pi)

    # Compute the term involving the log-determinants
    term2_sigma = -0.5 * n * np.log(sigma2c)
    term2_logdets = - 0.5 * _c.Sigma.logdet_R
    term2 = term2_sigma + term2_logdets

    # Compute the term involving the sum of squares
    term3 = -0.5 * sampler.aux['reflectivity_ss_unconstrained'][:, i + 1] / sigma2c

    log_density = term1 + term2 + term3

    if verbose:
        print(f"\nLog-prior reflectivity at iteration {iter_nr}:")
        print(f"sigma2c = {sigma2c}")
        print(f"term2 (log-determinants term) = {term2} (sigma part: {term2_sigma}, logdet part: {term2_logdets})")
        print(f"SSunc = {sampler.aux['reflectivity_ss_unconstrained'][:, i + 1]}, term3 (sum of squares term) = {term3}")
        print(f"Total log-prior reflectivity = {log_density}\n")

    return log_density


def _compute_log_prior_wavelet(sampler, iter_nr):
    """Compute the log-prior wavelet at the current state.

    Parameters:
    ----------
    sampler: MCMC object
        Contains sampler configuration, data, and starting value.
    iter_nr: int
        Global iteration number.
    """
    # Extract mcmc configuration
    verbose = sampler.mcmc_config.get('verbose')

    # Compute the iteration index in the chunk
    if iter_nr == -1:
        i = -1
    else:
        chunk_size = sampler.file_config.get('chunk_size')
        i = iter_nr % chunk_size

    # Extract current state for the observational variance
    sigma2w = sampler.theta['sigma2w'][0, i + 1]  # wavelet variance

    # Compute the term involving 2*pi
    n = sampler.lattice.nv
    _w = sampler.par_objs['w']
    term1 = - n / 2 * np.log(2 * np.pi)

    # Compute the term involving the log-determinants
    term2_sigma = -0.5 * n * np.log(sigma2w)
    term2_logdets = - 0.5 * _w.Sigma.logdet_R
    term2 = term2_sigma + term2_logdets

    # Compute the term involving the sum of squares
    ss = sampler.aux['wavelet_ss_constrained'][:, i + 1]
    if ss is None:
        ss = sampler.aux['wavelet_ss_unconstrained'][:, i + 1]
    term3 = -0.5 * ss / sigma2w

    log_density = term1 + term2 + term3

    if verbose:
        print(f"\nLog-prior wavelet at iteration {iter_nr}:")
        print(f"sigma2w = {sigma2w}")
        print(f"term2 (log-determinants term) = {term2} (sigma part: {term2_sigma}, logdet part: {term2_logdets})")
        print(f"SS = {ss}, term3 (sum of squares term) = {term3}")
        print(f"Total log-prior wavelet = {log_density}\n")

    return log_density


def _compute_log_prior_sigma2c(sampler, iter_nr):
    """Compute the log-prior for sigma2c at the current state."""
    # Compute the iteration index in the chunk
    if iter_nr == -1:
        i = -1
    else:
        chunk_size = sampler.file_config.get('chunk_size')
        i = iter_nr % chunk_size

    sigma2c = sampler.theta['sigma2c'][0, i + 1]
    alpha = sampler.model.model['sigma2c'].alpha
    beta = sampler.model.model['sigma2c'].beta

    # Inverse-Gamma prior
    return -(alpha + 1) * np.log(sigma2c) - beta / sigma2c


def _compute_log_prior_sigma2w(sampler, iter_nr):
    """Compute the log-prior for sigma2w at the current state."""
    # Compute the iteration index in the chunk
    if iter_nr == -1:
        i = -1
    else:
        chunk_size = sampler.file_config.get('chunk_size')
        i = iter_nr % chunk_size

    sigma2w = sampler.theta['sigma2w'][0, i + 1]
    alpha = sampler.model.model['sigma2w'].alpha
    beta = sampler.model.model['sigma2w'].beta

    # Inverse-Gamma prior
    return -(alpha + 1) * np.log(sigma2w) - beta / sigma2w


def _compute_log_prior_zeta(sampler, iter_nr):
    """Compute the log-prior for zeta (inverse SNR) at the current state."""
    # Compute the iteration index in the chunk
    if iter_nr == -1:
        i = -1
    else:
        chunk_size = sampler.file_config.get('chunk_size')
        i = iter_nr % chunk_size

    zeta = sampler.theta['zeta'][0, i + 1]
    alpha = sampler.model.model['zeta'].alpha
    beta = sampler.model.model['zeta'].beta

    # Inverse-Gamma prior
    return -(alpha + 1) * np.log(zeta) - beta / zeta
