"""Shared fixtures: small collapsed-HMC/Gibbs samplers for the test suite.

Model construction follows setup() in
external/andrew_claude/bsbd_report_source/bsbd_report_source/check_gradient.py.
"""
import contextlib
import copy
import io
import os
import random
import sys
import time

import numpy as np
import pytest
import scipy.linalg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# scipy.linalg.kron was removed in SciPy >= 1.16 but is still imported by src.
if not hasattr(scipy.linalg, 'kron'):
    scipy.linalg.kron = np.kron

from src.classes.lattice import Lattice
from src.classes.model import ConvolutionalModel
from src.mcmc.mcmc import MCMC
from src.mcmc.mwg.cyclic.update_ss import _update_ss
from src.mcmc.update_blur_image.collapsedhmc.initialize import (
    _create_momentum_object,
    _initialize_collapsed_hmc,
)
from src.mcmc.updates.state import _copy_previous_state

SEED = 20

# name -> (nv_obs, nh_obs, k, mv, mh)
LATTICES = {
    'small': (12, 2, 6, 0, 0),
    'medium': (36, 12, 12, 18, 6),
}

# Hyperparameters (as in check_gradient.py / the paper's synthetic experiments)
ALPHA_W, BETA_W = 2.01, 10
ALPHA_C, BETA_C = 2.00001, 1 / 500
ALPHA_ZETA, BETA_ZETA = 3, 0.1
ZETA_INIT = 0.05
RHO_W = RHO_V = RHO_H = 1.5


def _build_model(lattice, observed_b=None, image_b=None):
    model = ConvolutionalModel(lattice)
    model.setup_wavelet_variance(alpha=ALPHA_W, beta=BETA_W)
    model.initialize_wavelet_variance(init=None)
    model.setup_reflectivity_variance(alpha=ALPHA_C, beta=BETA_C)
    model.initialize_reflectivity_variance(init=None)
    model.setup_inverse_snr(alpha=ALPHA_ZETA, beta=BETA_ZETA)
    model.initialize_inverse_snr(init=ZETA_INIT)
    model.setup_wavelet_prior(rho=RHO_W, constr=True)
    model.initialize_wavelet()
    model.setup_reflectivity_prior(rho_v=RHO_V, rho_h=RHO_H, b=image_b)
    model.initialize_reflectivity()
    model.setup_seismic_model(rho_v=RHO_V, rho_h=RHO_H, b=observed_b)
    model.initialize_seismic()
    return model


def make_sampler(nv_obs, nh_obs, k, mv, mh, nwell, estim_root):
    """Build a small model and an MCMC sampler initialized for collapsed HMC.

    nwell: 0 -> no exact image observations (m = 0); None -> full well column;
    an int -> that many centred rows of the well column.
    Returns (sampler, lattice).
    """
    random.seed(SEED)
    np.random.seed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        lattice = Lattice(nv_obs, nh_obs, k, mv=mv, mh=mh, topology='C')
        lattice.create_ava_positions(verbose=False)
        lattice.create_wavelet_positions(verbose=False)
        data_true = _build_model(lattice)
        if nwell == 0:
            image_b = None
        else:
            if nwell is None:
                rows_obs = np.arange(lattice.nv_ava, dtype=int)
            else:
                mid = lattice.nv_ava // 2
                start = mid - nwell // 2
                rows_obs = np.arange(start, start + nwell, dtype=int)
            lattice.create_obs_reflectivity_positions(
                well_column_ava=None, rows_ava=rows_obs, verbose=False)
            well_coords_vec = lattice.well_positions['well_coords_vec']
            image_b = data_true.theta['c_star'].reshape(-1)[well_coords_vec].reshape(-1, 1)
        data_model = copy.deepcopy(data_true)
        data_model.setup_reflectivity_prior(rho_v=RHO_V, rho_h=RHO_H, b=image_b)
        data_model.initialize_reflectivity(init=data_true.theta['c_star'])

        os.makedirs(estim_root, exist_ok=True)
        estimate = ['c_star', 'w_star', 'sigma2c', 'sigma2w', 'zeta']
        if data_model.lattice.n > data_model.lattice.n_ava:
            estimate.append('d_star')
        sampler = MCMC(model=data_model, theta_init=data_true.theta, estimate=estimate,
                       theta_true=data_model.theta, chunk_size=10,
                       path=str(estim_root) + '/',
                       model_filename='test', folder='test/')
        adapt_config = {'collapsed_hmc': {'epsilon': {'type': None}, 'L': {'type': None}}}
        sampler.mcmc_config = {'algorithm': 'collapsed_hmc', 'N': 10, 'verbose': False,
                               'save_stats': False, 'constr_SSD': False,
                               'constr_SSW': True, 'constr_SSC': False}
        sampler._create_par_objects(False)
        sampler.adapt_config = adapt_config
        sampler.adapt_tracking = {m: {p: {'history': np.zeros(11), 'batch_ar': []}
                                      for p in pd} for m, pd in adapt_config.items()}
        sampler.stats = {
            'mwg': {'init_time': time.time(), 'exec_time': 0, 'error': np.zeros(11)},
            'wc_update': {
                'exec_time': None, 'iter_with_chmc': np.zeros(11), 'nr_iters': np.zeros(11),
                'loglik': np.zeros(11), 'log_prior_w': np.zeros(11),
                'log_prior_c': np.zeros(11), 'log_posterior_dens': np.zeros(11),
                'acceptance': {'log_acc_prob': np.zeros(11), 'acc_iter': np.zeros(11),
                               'cum_ar': np.zeros(11)}},
            'log_post': {kk: np.zeros((1, 11)) for kk in
                         ['likelihood', 'prior_reflectivity', 'prior_wavelet',
                          'prior_sigma2c', 'prior_sigma2w', 'prior_zeta', 'joint']}}
        sampler._create_sample_aux_dicts(N=11)
        sampler._initialize_sample_aux_dicts(i=0)
        _update_ss(sampler, -1)
        _initialize_collapsed_hmc(sampler, epsilon=0.008, L=10, sigma2p=1,
                                  precondition='prior', p_collapsed_hmc=1)
        _create_momentum_object(sampler)
        _copy_previous_state(sampler, 0)
    return sampler, lattice


@pytest.fixture
def build_sampler(tmp_path):
    """Factory: build_sampler(lattice_name, nwell) -> (sampler, lattice)."""
    def _factory(lattice_name, nwell):
        return make_sampler(*LATTICES[lattice_name], nwell, tmp_path / 'estimations')
    return _factory


@pytest.fixture(params=['small', 'medium'])
def lattice_name(request):
    return request.param


@pytest.fixture(params=[0, None], ids=['m0', 'full_well'])
def nwell(request):
    return request.param


@pytest.fixture
def sampler_case(build_sampler, lattice_name, nwell):
    """Parametrized over both lattices and m = 0 / full well column."""
    return build_sampler(lattice_name, nwell)
