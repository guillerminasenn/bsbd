"""potential_adjoint == potential == dense log|Sigma_{d|w}| + quadratic form."""
import math

import numpy as np
from scipy import linalg

from src.utils.model_utils import create_W
from src.utils.efficient_algebra_utils import multiply_matrix_vector_circ
from src.mcmc.update_blur_image.collapsedhmc.cyclic.potential import potential
from src.mcmc.update_blur_image.collapsedhmc.cyclic.adjoint import potential_adjoint


def dense_from_base(base):
    """Dense matrix of the circulant/BCCB with the given (nv, nh) base."""
    n = base.size
    M = np.empty((n, n))
    e = np.zeros((n, 1))
    for j in range(n):
        e[:] = 0.0
        e[j] = 1.0
        M[:, j] = multiply_matrix_vector_circ(base, e).ravel()
    return M


def dense_potential(sampler, i, w_star):
    """Potential from dense algebra: prior + N(d; W mu_c*, W Sigma_c* W' + Sigma_d)."""
    _lik = sampler.par_objs['d']
    _c = sampler.par_objs['c']
    _w = sampler.par_objs['w']
    n = _lik.lattice.n
    sigma2w = sampler.theta['sigma2w'][0, i + 1].item()
    sigma2c = sampler.theta['sigma2c'][0, i + 1].item()
    sigma2d = sampler.aux['sigma2d'][0, i + 1].item()
    d = sampler.theta['d_star'][:, i + 1].reshape(-1, 1)

    # Prior on the free wavelet coordinates (via the stored Cholesky factor)
    unc = _w.wavelet_constraints['unconstrained_indices']
    chol = _w.wavelet_constraints['chol_R_wu_star']
    Sigma_wu = sigma2w * (chol @ chol.T)
    wc = (w_star[unc, :] - _w.wavelet_constraints['mean_w_star'][unc, :]).reshape(-1, 1)
    kf = len(unc)
    U_w = 0.5 * (kf * math.log(2 * math.pi) + np.linalg.slogdet(Sigma_wu)[1]
                 + (wc.T @ linalg.solve(Sigma_wu, wc)).item())

    # Dense W, Sigma_c (constrained if applicable), Sigma_d
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star)
    W_dense = dense_from_base(base_W)
    Sigma_c = sigma2c * dense_from_base(_c.Sigma.base_R)
    Sigma_d = sigma2d * dense_from_base(_lik.Sigma.base_R)
    if _c.reflectivity_constraints['constr']:
        wcv = _c.lattice.well_positions['well_coords_vec']
        ScAt = Sigma_c[:, wcv]
        Sigma_c = Sigma_c - ScAt @ linalg.solve(ScAt[wcv, :], ScAt.T)
        mu_c = _c.reflectivity_constraints['mean_c_star']
    else:
        mu_c = _c.mean
    Sigma_dw = W_dense @ Sigma_c @ W_dense.T + Sigma_d
    Sigma_dw = 0.5 * (Sigma_dw + Sigma_dw.T)

    dbar = d - W_dense @ mu_c.reshape(-1, 1)
    U_d = 0.5 * (n * math.log(2 * math.pi) + np.linalg.slogdet(Sigma_dw)[1]
                 + (dbar.T @ linalg.solve(Sigma_dw, dbar)).item())
    return U_w + U_d


def test_potential_agreement(sampler_case):
    sampler, lattice = sampler_case
    w = sampler.theta['w_star'][:, 1].reshape(-1, 1)
    U = float(potential(sampler, 0, w.copy()))
    U_adj = float(potential_adjoint(sampler, 0, w.copy()))
    U_dense = dense_potential(sampler, 0, w.copy())
    assert abs(U_adj - U) / abs(U) < 1e-10
    assert abs(U_dense - U) / abs(U) < 1e-8
