"""Central finite differences of the legacy potential vs the three gradient variants.

Expected: 'legacy' has O(1) relative error (documents the aliasing bug),
'legacy_fixed' and 'adjoint' match FD to < 1e-6, and 'adjoint' matches
'legacy_fixed' to < 1e-10.
"""
import numpy as np

from src.mcmc.update_blur_image.collapsedhmc.cyclic.potential import potential
from src.mcmc.update_blur_image.collapsedhmc.cyclic.gradient_potential import (
    gradient_potential_Fourier_domain)
from src.mcmc.update_blur_image.collapsedhmc.cyclic.adjoint import gradient_potential_adjoint


def central_fd(sampler, i, w_star, unc, h=1e-6):
    g = np.zeros(len(unc))
    for j, idx in enumerate(unc):
        wp, wm = w_star.copy(), w_star.copy()
        wp[idx, 0] += h
        wm[idx, 0] -= h
        g[j] = (potential(sampler, i, wp) - potential(sampler, i, wm)) / (2 * h)
    return g


def test_gradient_variants(sampler_case):
    sampler, lattice = sampler_case
    unc = sampler.par_objs['w'].wavelet_constraints['unconstrained_indices']
    w = sampler.theta['w_star'][:, 1].reshape(-1, 1)

    g_fd = central_fd(sampler, 0, w, unc)
    g_legacy = gradient_potential_Fourier_domain(sampler, 0, w.copy())[unc, 0]
    g_fixed = gradient_potential_Fourier_domain(sampler, 0, w.copy(), fix_aliasing=True)[unc, 0]
    g_adjoint = gradient_potential_adjoint(sampler, 0, w.copy())[unc, 0]

    scale = np.max(np.abs(g_fd))
    assert np.max(np.abs(g_legacy - g_fd)) / scale > 1e-2   # the documented aliasing bug
    assert np.max(np.abs(g_fixed - g_fd)) / scale < 1e-6
    assert np.max(np.abs(g_adjoint - g_fd)) / scale < 1e-6
    assert np.max(np.abs(g_adjoint - g_fixed)) / np.max(np.abs(g_fixed)) < 1e-10
