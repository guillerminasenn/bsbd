"""Adjoint (reverse-mode) evaluation of the collapsed-HMC potential gradient
and a leaner potential, cyclic case.

The existing gradient_potential_Fourier_domain() evaluates, for *each* of the
k free blur coordinates, a directional derivative in the Fourier domain: it
materialises (k, nv, nh) complex tensors, k batched (m x m) sandwiches
L' dZ_i L, k batched (m x m) products inv_Y @ dY_i, k IFFT2s and a
(k, nv, nv) tensor for the constrained-image term.  Its cost is therefore
O(k * (n log n + m^3 + nv^2 nh)) per leapfrog step and it moves ~1 GB of
temporaries per call at the paper's problem size.

Every one of those k quantities is linear in the derivative direction
eps_i(f_v) = exp(-2 pi i f_v r_i / nv)  (the DFT of the shifted unit vector
that generates dW0/dw_i), so all k coordinates can be obtained from ONE
n-sized reduction followed by ONE length-nv FFT evaluated at the k shifts
r_i.  This is what this module does.  Cost per gradient:
O(n log n + m^3 + nv^2 nh) with no k-fold tensors at all.

Numerically identical (to ~1e-12 relative) to the fixed vectorised gradient.
"""
import math
import numpy as np
import scipy.fft
from scipy import linalg

from src.utils.model_utils import create_W, embed_wu
from src.utils.efficient_algebra_utils import transpose_base_circ, multiply_matrix_vector_circ
from src.utils.math_utils import compute_log_det_circ, compute_log_det
from src.mcmc.update_blur_image.collapsedhmc.cyclic.efficient_utils import (
    compute_eigenvalues_A_invA_Z_B, project_down_with_Ac, project_up_with_Ac)


def _common(sampler, i, w_star):
    """Quantities shared by potential and gradient (all O(n log n + m^3))."""
    _lik = sampler.par_objs['d']; _w = sampler.par_objs['w']; _c = sampler.par_objs['c']
    nv, nh = _lik.lattice.nv, _lik.lattice.nh
    n = nv * nh
    sigma2c = sampler.theta['sigma2c'][0, i + 1].item()
    d = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
    constr_c = _c.reflectivity_constraints['constr']

    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star)
    ev_W0 = scipy.fft.fft(base_W0.flatten())
    ev_W = np.repeat(ev_W0[:, None], nh, axis=1)
    ev_WT = np.conj(ev_W)
    eigs, base_Z = compute_eigenvalues_A_invA_Z_B(sampler, i, ev_W, ev_WT)
    base_Z = base_Z.real

    out = dict(nv=nv, nh=nh, n=n, sigma2c=sigma2c, d=d, constr_c=constr_c, eigs=eigs,
               base_Z=base_Z, ev_W0=ev_W0, ev_W=ev_W, ev_WT=ev_WT, base_W=base_W, base_W0T=base_W0T)

    # collapsed-likelihood mean and centred data  (mu_c is the constrained prior mean)
    mean_c = _c.reflectivity_constraints['mean_c_star'] if constr_c else _c.mean
    mean_c_hat = scipy.fft.ifft2(mean_c.reshape((nv, nh), order='F'))
    mean_lik = scipy.fft.fft2(ev_W * mean_c_hat).real.reshape((n, 1), order='F')
    dbar = d - mean_lik
    D = dbar.reshape((nv, nh), order='F')
    Dhat = scipy.fft.ifft2(D)
    out['dbar'] = dbar
    # r = A^-1 dbar
    r = scipy.fft.fft2(eigs['eigenvalues_inv_A'] * Dhat).real.reshape((n, 1), order='F')
    out['r'] = r

    if constr_c:
        rows = _c.lattice.well_positions['rows_cyclic']
        L = _c.reflectivity_constraints['L']
        m = _c.reflectivity_constraints['nr_constraints']
        # A_c Z A_c' and S = A_c (Sigma_c - Z) A_c' from the diagonal block base
        bz = base_Z[:, 0]
        AZAt = bz[(rows[None, :] - rows[:, None]) % nv]            # C[p,q] = base[(q-p) mod nv]
        AZAt = 0.5 * (AZAt + AZAt.T)
        bsc = sigma2c * _c.Sigma.base_R[:, 0]
        S = (bsc - bz)[(rows[None, :] - rows[:, None]) % nv]
        S = 0.5 * (S + S.T)
        Y = np.eye(m) - L.T @ AZAt @ L / sigma2c
        Y = 0.5 * (Y + Y.T)
        cS = linalg.cho_factor(S, lower=True)
        # A_c B' dbar  via one FFT mat-vec + gather (no dense m x n matrix)
        Btd = scipy.fft.fft2(eigs['eigenvalues_BT'] * Dhat).real.reshape((n, 1), order='F')
        q = project_down_with_Ac(_c, Btd)                            # m x 1
        Sinv_q = linalg.cho_solve(cS, q)
        t = project_up_with_Ac(_c, Sinv_q)
        s = scipy.fft.fft2(eigs['eigenvalues_B'] * scipy.fft.ifft2(t.reshape((nv, nh), order='F'))).real.reshape((n, 1), order='F')
        out.update(rows=rows, L=L, m=m, Y=Y, S=S, cS=cS, q=q, Sinv_q=Sinv_q, s=s, p=r + s)
    else:
        out.update(p=r)
    return out


def potential_adjoint(sampler, i, w_star, common=None):
    """Same value as potential(), without building any dense m x n matrix."""
    _w = sampler.par_objs['w']
    c = common if common is not None else _common(sampler, i, w_star)
    nv, n, sigma2c = c['nv'], c['n'], c['sigma2c']
    sigma2w = sampler.theta['sigma2w'][0, i + 1].item()

    # prior
    nr = _w.wavelet_constraints['nr_constraints']
    unc = _w.wavelet_constraints['unconstrained_indices']
    wc = (w_star[unc, :] - _w.wavelet_constraints['mean_w_star'][unc, :]).reshape(-1, 1)
    z = linalg.solve_triangular(_w.wavelet_constraints['chol_R_wu_star'], wc, lower=True)
    U_w = 0.5 * (nv - nr) * math.log(2 * math.pi) + 0.5 * (nv - nr) * math.log(sigma2w) \
        + 0.5 * _w.wavelet_constraints['logdet_R_wu_star'] + 0.5 * float((z.T @ z)[0, 0]) / sigma2w

    # likelihood: log|A| + log|Y|  and  dbar' A^-1 dbar + q' S^-1 q
    logdet_A = float(np.sum(np.log(c['eigs']['eigenvalues_A'].real)))
    ss = float((c["dbar"].T @ c["r"])[0, 0])
    logdet_Y = 0.0
    if c['constr_c']:
        logdet_Y = compute_log_det(c['Y'])
        ss += float((c["q"].T @ c["Sinv_q"])[0, 0])
    U_d = 0.5 * n * math.log(2 * math.pi) + 0.5 * (logdet_A + logdet_Y) + 0.5 * ss
    return U_d + U_w


def gradient_potential_adjoint(sampler, i, w_star, common=None, return_common=False):
    """Gradient of the potential w.r.t. w* (nv x 1, zero on constrained coords)."""
    _w = sampler.par_objs['w']; _c = sampler.par_objs['c']
    c = common if common is not None else _common(sampler, i, w_star)
    nv, nh, n, sigma2c = c['nv'], c['nh'], c['n'], c['sigma2c']
    eigs = c['eigs']
    unc = _w.wavelet_constraints['unconstrained_indices']
    # r_i: position of the single 1 in base(dW0/dw_i)  (same convention as the vectorised code)
    bases_d_W0 = _w.wavelet_derivatives['bases_d_W0'][:, unc]
    r_idx = np.argmax(bases_d_W0, axis=0)

    # ---------- prior ----------
    sigma2w = sampler.theta['sigma2w'][0, i + 1].item()
    wc = (w_star[unc, :] - _w.wavelet_constraints['mean_w_star'][unc, :]).reshape(-1, 1)
    chol = _w.wavelet_constraints['chol_R_wu_star']
    g_prior = linalg.solve_triangular(chol.T, linalg.solve_triangular(chol, wc, lower=True), lower=False) / sigma2w

    # ---------- log-determinant ----------
    invA = eigs['eigenvalues_inv_A'].real                    # A is symmetric BCCB -> real eigenvalues
    lam_Rc = _c.Sigma.fft_base_R                             # (nv, nh) eigenvalues of R_c (real)
    G1 = lam_Rc * c['ev_WT']                                 # dA_i = sigma2c (eps_i G1 + conj)
    h1 = (invA * G1).sum(axis=1)                             # length nv
    tr_invA_dA = 2.0 * sigma2c * scipy.fft.fft(h1).real[r_idx]

    tr_invY_dY = np.zeros(len(unc))
    if c['constr_c']:
        L, Y, rows, m = c['L'], c['Y'], c['rows'], c['m']
        inv_Y = linalg.inv(Y); inv_Y = 0.5 * (inv_Y + inv_Y.T)
        M = L @ inv_Y @ L.T                                  # m x m, once
        Mlag = np.zeros(nv)
        np.add.at(Mlag, (rows[None, :] - rows[:, None]) % nv, M)   # lag histogram of M (symmetric in lag)
        hM = scipy.fft.fft(Mlag).real                         # real because Mlag[d] = Mlag[-d]
        G2 = invA * c['ev_W']
        G3 = -sigma2c * (np.abs(c['ev_W']) ** 2) * invA ** 2 * G1
        Q = sigma2c * lam_Rc ** 2 * (np.conj(G2) + G3)       # eig(dZ_i) = eps_i Q + conj(eps_i Q)
        qv = Q.sum(axis=1)
        tr_invY_dY = -(2.0 / n) * scipy.fft.fft(hM * qv).real[r_idx]
    g_logdet = tr_invA_dA + tr_invY_dY

    # ---------- sum of squares ----------
    p = c['p'].real
    P = p.reshape((nv, nh), order='F')
    # part 1: p'(R_ch x D1_i)p = 2 <Dh_i, K>,  K = P R_ch P',  Dh_i = circ(roll(base(R_cv W0'), r_i))
    Rch = _c.Sigma.Rh
    K = P @ Rch @ P.T
    Kdiag = np.array([np.trace(K, offset=dlt) + (np.trace(K, offset=dlt - nv) if dlt > 0 else 0.0) for dlt in range(nv)])
    b = scipy.fft.ifft(_c.Sigma.fft_base_Rv.ravel() * np.conj(c['ev_W0'])).real     # base of R_cv W0'
    part1 = 2.0 * scipy.fft.ifft(np.conj(scipy.fft.fft(b)) * scipy.fft.fft(Kdiag)).real[r_idx]
    part2 = np.zeros(len(unc))
    if c['constr_c']:
        Rh_star = _c.reflectivity_constraints['Rh_star']; Rv_star = _c.reflectivity_constraints['Rv_star']
        W0T = linalg.circulant(c['base_W0T'].reshape(-1)).T
        X = Rv_star @ W0T                                    # nv x nv
        Ks = P @ Rh_star @ P.T
        cc = scipy.fft.ifft((scipy.fft.fft(X, axis=0) * np.conj(scipy.fft.fft(Ks, axis=0))).sum(axis=1)).real
        part2 = 2.0 * cc[r_idx]
    d_SS_1 = -sigma2c * (part1 - part2)
    gamma_u = sampler.aux_derivatives['gamma_matrix_unconstrained_indexes']
    d_SS_2 = -2.0 * (gamma_u @ p).ravel()
    g_ss = d_SS_1 + d_SS_2

    g_u = g_prior.ravel() + 0.5 * g_logdet + 0.5 * g_ss
    grad = embed_wu(g_u.reshape(-1, 1), _w)
    return (grad, c) if return_common else grad
