"""Standalone check of the collapsed-HMC gradient (cyclic case).

Compares (1) the repo gradient gradient_potential_Fourier_domain, (2) the adjoint
gradient in gradient_adjoint.py and (3) central finite differences of potential().
Also times each.  Run from the repo root:

    python check_gradient.py NV_OBS NH_OBS K MV MH [reps]

e.g.  python check_gradient.py 24 6 12 12 6        (n=432, quick)
      python check_gradient.py 330 50 126 150 50 2  (paper size; needs ~6 GB RAM)
"""
import os, sys, time, random, copy, math, contextlib, io
import numpy as np
from scipy import linalg

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
import scipy.linalg
if not hasattr(scipy.linalg, 'kron'):
    scipy.linalg.kron = np.kron

from src.classes.lattice import Lattice
from src.classes.model import ConvolutionalModel
from src.mcmc.mcmc import MCMC
from src.mcmc.update_blur_image.collapsedhmc.cyclic.potential import potential
from src.mcmc.update_blur_image.collapsedhmc.cyclic import gradient_potential as gpmod
from src.utils.model_utils import create_W, subset_AB, subset_ABA, embed_wu
from src.mcmc.update_blur_image.collapsedhmc.cyclic.efficient_utils import compute_bases
from src.utils.efficient_algebra_utils import transpose_base_circ, multiply_matrix_vector_circ
from src.utils.math_utils import compute_log_det_circ, compute_log_det

nv_obs, nh_obs, k, mv, mh = [int(a) for a in sys.argv[1:6]]
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
nwell = None
seed = 20
random.seed(seed); np.random.seed(seed)
alpha_w, beta_w = 2.01, 10
alpha_c, beta_c = 2.00001, 1 / 500
alpha_zeta, beta_zeta = 3, 0.1
zeta_init = 0.05
rho_w = rho_v = rho_h = 1.5

def build_model(lattice, observed_b=None, image_b=None):
    model = ConvolutionalModel(lattice)
    model.setup_wavelet_variance(alpha=alpha_w, beta=beta_w); model.initialize_wavelet_variance(init=None)
    model.setup_reflectivity_variance(alpha=alpha_c, beta=beta_c); model.initialize_reflectivity_variance(init=None)
    model.setup_inverse_snr(alpha=alpha_zeta, beta=beta_zeta); model.initialize_inverse_snr(init=zeta_init)
    model.setup_wavelet_prior(rho=rho_w, constr=True); model.initialize_wavelet()
    model.setup_reflectivity_prior(rho_v=rho_v, rho_h=rho_h, b=image_b); model.initialize_reflectivity()
    model.setup_seismic_model(rho_v=rho_v, rho_h=rho_h, b=observed_b); model.initialize_seismic()
    if os.environ.get('BSBD_SLIM', '0') == '1':  # drop precomputed arrays that are never used at iteration time (memory only)
        wd = model.model['w'].wavelet_derivatives
        for key in ['fft2_bases_d_W', 'bases_d_WT', 'bases_d_W', 'd_W0', 'd_W']:
            wd[key] = None
    return model

def setup(nv_obs, nh_obs, k, mv, mh, nwell):
    with contextlib.redirect_stdout(io.StringIO()):
        lattice = Lattice(nv_obs, nh_obs, k, mv=mv, mh=mh, topology='C')
        lattice.create_ava_positions(verbose=False); lattice.create_wavelet_positions(verbose=False)
        data_true = build_model(lattice)
        if nwell == 0:
            image_b = None
        else:
            if nwell is None:
                rows_obs = np.arange(lattice.nv_ava, dtype=int)
            else:
                mid = lattice.nv_ava // 2; start = mid - nwell // 2
                rows_obs = np.arange(start, start + nwell, dtype=int)
            lattice.create_obs_reflectivity_positions(well_column_ava=None, rows_ava=rows_obs, verbose=False)
            well_coords_vec = lattice.well_positions['well_coords_vec']
            image_b = data_true.theta['c_star'].reshape(-1)[well_coords_vec].reshape(-1, 1)
        data_model = copy.deepcopy(data_true)
        data_model.setup_reflectivity_prior(rho_v=rho_v, rho_h=rho_h, b=image_b)
        data_model.initialize_reflectivity(init=data_true.theta['c_star'])
        estim_root = os.path.join(root_dir, 'estimations', 'prof'); os.makedirs(estim_root, exist_ok=True)
        estimate = ['c_star', 'w_star', 'sigma2c', 'sigma2w', 'zeta']
        if data_model.lattice.n > data_model.lattice.n_ava: estimate.append('d_star')
        sampler = MCMC(model=data_model, theta_init=data_true.theta, estimate=estimate,
                       theta_true=data_model.theta, chunk_size=10, path=estim_root + '/',
                       model_filename='gradcheck', folder='gradcheck/')
        adapt_config = {'collapsed_hmc': {'epsilon': {'type': None}, 'L': {'type': None}}}
        sampler.mcmc_config = {'algorithm': 'collapsed_hmc', 'N': 10, 'verbose': False, 'save_stats': False,
                               'constr_SSD': False, 'constr_SSW': True, 'constr_SSC': False}
        sampler._create_par_objects(False)
        sampler.adapt_config = adapt_config
        sampler.adapt_tracking = {m: {p: {'history': np.zeros(11), 'batch_ar': []} for p in pd} for m, pd in adapt_config.items()}
        sampler.stats = {'mwg': {'init_time': time.time(), 'exec_time': 0, 'error': np.zeros(11)},
                         'wc_update': {'exec_time': None, 'iter_with_chmc': np.zeros(11), 'nr_iters': np.zeros(11),
                                       'loglik': np.zeros(11), 'log_prior_w': np.zeros(11), 'log_prior_c': np.zeros(11),
                                       'log_posterior_dens': np.zeros(11),
                                       'acceptance': {'log_acc_prob': np.zeros(11), 'acc_iter': np.zeros(11), 'cum_ar': np.zeros(11)}},
                         'log_post': {kk: np.zeros((1, 11)) for kk in ['likelihood', 'prior_reflectivity', 'prior_wavelet', 'prior_sigma2c', 'prior_sigma2w', 'prior_zeta', 'joint']}}
        sampler._create_sample_aux_dicts(N=11)
        sampler._initialize_sample_aux_dicts(i=0)
        from src.mcmc.mwg.cyclic.update_ss import _update_ss
        _update_ss(sampler, -1)
        from src.mcmc.update_blur_image.collapsedhmc.initialize import _initialize_collapsed_hmc, _create_momentum_object
        _initialize_collapsed_hmc(sampler, epsilon=0.008, L=10, sigma2p=1, precondition='prior', p_collapsed_hmc=1)
        _create_momentum_object(sampler)
        from src.mcmc.updates.state import _copy_previous_state
        _copy_previous_state(sampler, 0)
    return sampler, lattice

def _unused_potential_parts(sampler, i, w_star):
    """Copy of potential() returning the pieces: (prior, logdet_A, logdet_Y, ss1, ss2)."""
    _lik = sampler.par_objs['d']; _c = sampler.par_objs['c']; _w = sampler.par_objs['w']
    d_star = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
    sigma2w = sampler.theta['sigma2w'][0, i + 1].item(); sigma2c = sampler.theta['sigma2c'][0, i + 1].item()
    unc = _w.wavelet_constraints['unconstrained_indices']
    mean_w_star = _w.wavelet_constraints['mean_w_star']
    wc = (w_star[unc, :] - mean_w_star[unc, :]).reshape(-1, 1)
    chol = _w.wavelet_constraints['chol_R_wu_star']
    z = linalg.solve_triangular(chol, wc, lower=True)
    prior = 0.5 * (z.T @ z)[0, 0] / sigma2w
    W0, W, base_W0, base_W, base_W0T, base_WT = create_W(_lik.lattice, w_star.copy())
    base_A, base_inv_A, base_Z, base_B = compute_bases(sampler, i, base_W, base_WT)
    logdet_A = compute_log_det_circ(base_A)
    constr_c = _c.reflectivity_constraints['constr']
    if constr_c:
        n_w = _c.reflectivity_constraints['nr_constraints']; rows_cyclic = _c.lattice.well_positions['rows_cyclic']
        mean_c = _c.reflectivity_constraints['mean_c_star']; L = _c.reflectivity_constraints['L']
        AZAt = linalg.circulant(base_Z[:, 0]).T[rows_cyclic, :][:, rows_cyclic]
        Y = np.eye(n_w) - L.T @ AZAt @ L / sigma2c
        logdet_Y = compute_log_det(Y)
    else:
        mean_c = _c.mean; logdet_Y = 0.0
    mean_lik = multiply_matrix_vector_circ(base_W, mean_c)
    dc = d_star - mean_lik
    ss1 = (dc.T @ multiply_matrix_vector_circ(base_inv_A, dc))[0, 0]
    ss2 = 0.0
    if constr_c:
        ABt = subset_AB(transpose_base_circ(base_B), _c)
        S = subset_ABA(sigma2c * _c.Sigma.base_R - base_Z, _c)
        ABt_d = ABt @ dc
        ss2 = (ABt_d.T @ linalg.solve(S, ABt_d))[0, 0]
    return dict(prior=prior, logdet_A=float(logdet_A), logdet_Y=float(logdet_Y), ss1=float(ss1), ss2=float(ss2))


# ---------------------------------------------------------------------------
from src.mcmc.update_blur_image.collapsedhmc.cyclic.gradient_potential import gradient_potential_Fourier_domain
from src.mcmc.update_blur_image.collapsedhmc.cyclic.potential import potential
from src.mcmc.update_blur_image.collapsedhmc.cyclic.gradient_adjoint import gradient_potential_adjoint, potential_adjoint, _common

sampler, lattice = setup(nv_obs, nh_obs, k, mv, mh, nwell)
nv, nh, n = lattice.nv, lattice.nh, lattice.n
_w = sampler.par_objs['w']; _c = sampler.par_objs['c']
unc = _w.wavelet_constraints['unconstrained_indices']; m = _c.reflectivity_constraints['nr_constraints']
print(f"nv={nv} nh={nh} n={n} k={lattice.k} k_free={len(unc)} m={m}")
i = 0
w0 = sampler.theta['w_star'][:, i + 1].reshape((-1, 1)).copy()
w0[unc, 0] += 0.3 * np.std(w0[unc, 0]) * np.random.randn(len(unc))

U_ref = potential(sampler, i, w0.copy()); U_adj = potential_adjoint(sampler, i, w0.copy())
print(f"potential: repo {U_ref:.10g}  adjoint {U_adj:.10g}  rel diff {abs(U_ref-U_adj)/abs(U_ref):.2e}")
g_ref = gradient_potential_Fourier_domain(sampler, i, w0.copy())[unc, 0]
g_adj = gradient_potential_adjoint(sampler, i, w0.copy())[unc, 0]
print(f"gradient: max |adjoint - repo(fixed)| / max|repo| = {np.abs(g_adj - g_ref).max() / np.abs(g_ref).max():.2e}")
if n <= 3000:
    h = 1e-6 * max(1.0, np.abs(w0[unc]).max()); fd = np.zeros(len(unc))
    for jj, idx in enumerate(unc):
        wp = w0.copy(); wp[idx, 0] += h; wm = w0.copy(); wm[idx, 0] -= h
        fd[jj] = (potential(sampler, i, wp) - potential(sampler, i, wm)) / (2 * h)
    print(f"gradient: max |adjoint - FD| / max|FD| = {np.abs(g_adj - fd).max() / np.abs(fd).max():.2e}")

def bench(f, label):
    f(); t0 = time.time()
    for _ in range(reps): f()
    dt = (time.time() - t0) / reps
    print(f"  {label:40s} {dt*1000:9.1f} ms"); return dt
print("timings:")
t_gref = bench(lambda: gradient_potential_Fourier_domain(sampler, i, w0.copy()), "repo gradient (vectorised over k)")
t_gadj = bench(lambda: gradient_potential_adjoint(sampler, i, w0.copy()), "adjoint gradient")
t_pref = bench(lambda: potential(sampler, i, w0.copy()), "repo potential")
t_padj = bench(lambda: potential_adjoint(sampler, i, w0.copy()), "adjoint potential")
L = 10
print(f"HMC blur update, L={L}: repo ~{((L+1)*t_gref + 2*t_pref):.2f} s  ->  adjoint ~{((L+1)*t_gadj + 2*t_padj):.3f} s   (speedup {((L+1)*t_gref + 2*t_pref)/((L+1)*t_gadj + 2*t_padj):.0f}x)")
