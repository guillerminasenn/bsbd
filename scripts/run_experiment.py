"""Run one Gibbs or collapsed-HMC estimation on a simulated dataset.

Mirrors notebooks/run.ipynb: simulate a dataset from the SBD model on a cyclic
lattice, apply m exact image observations (centred), and run the sampler.
Outputs follow the existing layout estimations/<folder_id>/<algorithm>/...

Examples (paper Sec. 5.1 lattice):
  python scripts/run_experiment.py --nv-obs 24 --nh-obs 1 -k 12 --m 12 \
      --algorithm gibbs -N 5000
  python scripts/run_experiment.py --nv-obs 24 --nh-obs 1 -k 12 --m 12 \
      --algorithm collapsed_hmc --gradient adjoint --precondition prior \
      -N 1000 -L 40 --adapt marc
"""
import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from src.classes.lattice import Lattice
from src.classes.model import ConvolutionalModel
from src.mcmc.mcmc import MCMC

# Hyperparameters (paper / notebooks/run.ipynb)
ALPHA_W, BETA_W = 2.01, 10
ALPHA_C, BETA_C = 2.00001, 1 / 500
ALPHA_ZETA, BETA_ZETA = 3, 0.1
ZETA_INIT = 0.05
RHO_W = RHO_V = RHO_H = 1.5


def build_model(lattice, observed_b=None, image_b=None):
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


def get_image_well_subset(model, m, well_column_obs=None):
    """Exact image observations: m centred rows of the well column (m=0: none)."""
    lattice = model.lattice
    nv_obs = lattice.nv_ava
    if m == 0:
        return None, np.array([], dtype=int)
    if not 0 < m <= nv_obs:
        raise ValueError(f"m must be in [0, {nv_obs}], got {m}")
    mid = nv_obs // 2
    start = min(max(mid - m // 2, 0), nv_obs - m)
    rows_obs = np.arange(start, start + m, dtype=int)
    lattice.create_obs_reflectivity_positions(
        well_column_ava=well_column_obs, rows_ava=rows_obs, verbose=False)
    well_coords_vec = lattice.well_positions['well_coords_vec']
    image_b = model.theta['c_star'].reshape(-1)[well_coords_vec].reshape(-1, 1)
    return image_b, rows_obs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nv-obs', type=int, required=True)
    p.add_argument('--nh-obs', type=int, required=True)
    p.add_argument('-k', '--blur-length', type=int, required=True)
    p.add_argument('--mv', type=int, default=0)
    p.add_argument('--mh', type=int, default=0)
    p.add_argument('--m', type=int, required=True,
                   help='number of exact image observations (centred rows of the well column)')
    p.add_argument('--algorithm', choices=['gibbs', 'collapsed_hmc'], required=True)
    p.add_argument('--gradient', choices=['legacy', 'legacy_fixed', 'adjoint'], default='adjoint')
    p.add_argument('--precondition', choices=['none', 'prior', 'posterior'], default='prior')
    p.add_argument('--mass-matrix', type=str, default=None,
                   help=".npy file with the k_free x k_free precision (precondition='posterior')")
    p.add_argument('--epsilon', type=float, default=0.01)
    p.add_argument('-L', '--leapfrog-steps', type=int, default=40)
    p.add_argument('--adapt', choices=['none', 'marc', 'rosenthal'], default='marc',
                   help='step-size adaptation (L stays fixed)')
    p.add_argument('--optimal-ar', type=float, default=0.5)
    p.add_argument('--adapt-B', type=int, default=50)
    p.add_argument('-N', '--iterations', type=int, required=True)
    p.add_argument('--chunk-size', type=int, default=None,
                   help='default: 1000 if it divides N, else N')
    p.add_argument('--seed', type=int, default=20)
    p.add_argument('--init', choices=['true', 'prior'], default='true')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    folder_id = (f"k{args.blur_length}_nv{args.nv_obs}_nh{args.nh_obs}"
                 f"_mv{args.mv}_mh{args.mh}_seed{args.seed}")
    data_dir = os.path.join(ROOT, 'data', 'processed', folder_id)
    estim_root = os.path.join(ROOT, 'estimations', folder_id)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(estim_root, exist_ok=True)

    # Simulate the dataset (deterministic given the seed), then constrain the image
    lattice = Lattice(args.nv_obs, args.nh_obs, args.blur_length,
                      mv=args.mv, mh=args.mh, topology='C')
    lattice.create_ava_positions(verbose=False)
    lattice.create_wavelet_positions(verbose=False)
    data_true = build_model(lattice)
    data_true.save_model(folder=f"data/processed/{folder_id}/", path=ROOT + '/')

    image_b, rows_obs = get_image_well_subset(data_true, args.m)
    data_model = copy.deepcopy(data_true)
    data_model.setup_reflectivity_prior(rho_v=RHO_V, rho_h=RHO_H, b=image_b)
    data_model.initialize_reflectivity(init=data_true.theta['c_star'])

    if args.init == 'prior':
        observed_b = data_true.model['d'].data_constraints['b']
        init_model = build_model(lattice, observed_b=observed_b, image_b=image_b)
        theta_init = init_model.theta
    else:
        theta_init = data_true.theta

    estimate = ['c_star', 'w_star', 'sigma2c', 'sigma2w', 'zeta']
    if lattice.n > lattice.n_ava:
        estimate.append('d_star')

    N = args.iterations
    chunk_size = args.chunk_size or (1000 if N % 1000 == 0 else N)
    sampler = MCMC(model=data_model, theta_init=theta_init, estimate=estimate,
                   theta_true=data_model.theta, chunk_size=chunk_size,
                   path=estim_root + '/')

    run_kwargs = {'verbose': False, 'save_stats': True}
    if args.algorithm == 'collapsed_hmc':
        adapt_eps = ({'type': None} if args.adapt == 'none' else
                     {'type': args.adapt, 'optimal_ar': args.optimal_ar, 'B': args.adapt_B})
        run_kwargs.update(
            adapt_config={'collapsed_hmc': {'epsilon': adapt_eps, 'L': {'type': None}}},
            epsilon=args.epsilon, L=args.leapfrog_steps, sigma2p=1.0,
            precondition=None if args.precondition == 'none' else args.precondition,
            gradient=args.gradient, p_collapsed_hmc=1)
        if args.mass_matrix is not None:
            run_kwargs['mass_matrix'] = np.load(args.mass_matrix)

    t0 = time.time()
    filenames = sampler.run(N, args.algorithm, **run_kwargs)
    wall = time.time() - t0
    print(f"\nDone in {wall:.1f} s ({wall / N * 1e3:.2f} ms/iter).")
    print(f"Run directory: {os.path.join(sampler.file_config['path'], sampler.file_config['folder'])}")
    return filenames


if __name__ == '__main__':
    main()
