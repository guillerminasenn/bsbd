"""Momentum sampling, kinetic and grad_kinetic consistent with the mass matrix
for precondition in {None, 'prior', 'posterior'}."""
import numpy as np
import pytest
from scipy import linalg

from src.mcmc.update_blur_image.collapsedhmc.initialize import (
    _create_momentum_object, _initialize_collapsed_hmc)
from src.mcmc.update_blur_image.collapsedhmc.mass_matrix import estimate_mass_matrix
from src.mcmc.update_blur_image.collapsedhmc.cyclic.utils import grad_kinetic, kinetic


@pytest.fixture
def small_sampler(build_sampler):
    return build_sampler('small', None)


def _posterior_mass(sampler, lattice, rng):
    _w = sampler.par_objs['w']
    w_samples = sampler.theta['w_star'][:, 1:2] + 0.1 * rng.standard_normal((lattice.nv, 500))
    return estimate_mass_matrix(w_samples, _w), w_samples, _w


def test_estimate_mass_matrix(small_sampler):
    sampler, lattice = small_sampler
    M, w_samples, _w = _posterior_mass(sampler, lattice, np.random.default_rng(3))
    unc = _w.wavelet_constraints['unconstrained_indices']
    kf = len(unc)
    C = np.cov(w_samples[unc, :])
    C = 0.5 * (C + C.T) + 1e-6 * (np.trace(C) / kf) * np.eye(kf)
    assert np.allclose(M, linalg.inv(C), rtol=1e-8)
    assert np.all(linalg.eigvalsh(M) > 0)


@pytest.mark.parametrize('precondition', [None, 'prior', 'posterior'])
def test_momentum_consistent_with_mass_matrix(small_sampler, precondition):
    sampler, lattice = small_sampler
    rng = np.random.default_rng(3)
    kwargs = {}
    if precondition == 'posterior':
        kwargs['mass_matrix'] = _posterior_mass(sampler, lattice, rng)[0]
    _initialize_collapsed_hmc(sampler, epsilon=0.008, L=5, sigma2p=1.0,
                              precondition=precondition, p_collapsed_hmc=1, **kwargs)
    _create_momentum_object(sampler)

    _p = sampler.par_objs['p']
    R = _p.Sigma.R
    sigma2p = (1 / sampler.theta['sigma2w'][:, 1]).item() if precondition == 'prior' else 1.0
    kf = R.shape[0]
    p = rng.standard_normal((kf, 1))

    # kinetic and grad_kinetic match 0.5 p' M^{-1} p and M^{-1} p with M = sigma2p * R
    K = kinetic(sampler, 0, p)
    gk, gk_ext = grad_kinetic(sampler, 0, p)
    assert np.isclose(K, 0.5 * (p.T @ np.linalg.solve(R, p)).item() / sigma2p)
    assert np.allclose(gk, np.linalg.solve(R, p) / sigma2p)

    # empirical momentum sampling covariance matches sigma2p * R
    np.random.seed(0)
    draws = np.hstack([_p.sample_scipy(sigma2=sigma2p) for _ in range(4000)])
    emp = np.cov(draws)
    assert np.max(np.abs(emp - sigma2p * R)) / np.max(np.abs(sigma2p * R)) < 0.15


def test_invalid_options_raise(small_sampler):
    sampler, lattice = small_sampler
    with pytest.raises(Exception, match='mass_matrix'):
        _initialize_collapsed_hmc(sampler, precondition='posterior')
    with pytest.raises(Exception, match='precondition'):
        _initialize_collapsed_hmc(sampler, precondition='bogus')
