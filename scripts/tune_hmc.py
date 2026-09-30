"""Tune (epsilon, L) for collapsed HMC with preliminary runs, then compare
optimally-tuned HMC against Gibbs on ESS/s of the blur coordinates.

Grid of short fixed-(epsilon, L) chains (no adaptation); the best config
maximizes min-ESS/s of omega. Optionally follows with longer comparison runs.
All runs go to a temporary directory; only the printed table / CSV remain.

Example (paper Sec. 5.1 lattice):
  python scripts/tune_hmc.py --nv-obs 24 --nh-obs 1 -k 12 --m 12 \
      --grid-N 300 --final-N-hmc 2000 --final-N-gibbs 20000
"""
import argparse
import contextlib
import copy
import io
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import RHO_H, RHO_V, build_model, get_image_well_subset
from src.classes.lattice import Lattice
from src.mcmc.mcmc import MCMC
from src.utils.mcmc_diagnostics import estimate_effective_sample_size

NO_ADAPT = {'collapsed_hmc': {'epsilon': {'type': None}, 'L': {'type': None}}}


def make_data(nv_obs, nh_obs, k, mv, mh, m, seed, tmp):
    random.seed(seed)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        lattice = Lattice(nv_obs, nh_obs, k, mv=mv, mh=mh, topology='C')
        lattice.create_ava_positions(verbose=False)
        lattice.create_wavelet_positions(verbose=False)
        data_true = build_model(lattice)
        os.makedirs(os.path.join(tmp, 'data'), exist_ok=True)
        data_true.save_model(folder='data/', path=tmp + '/')
        image_b, _ = get_image_well_subset(data_true, m)
        data_model = copy.deepcopy(data_true)
        data_model.setup_reflectivity_prior(rho_v=RHO_V, rho_h=RHO_H, b=image_b)
        data_model.initialize_reflectivity(init=data_true.theta['c_star'])
    return data_true, data_model, lattice


def run_chain(data_true, data_model, lattice, algorithm, N, seed, tmp, **hmc_kwargs):
    """One in-process chain; returns (omega chains post state, seconds, AR)."""
    estimate = ['c_star', 'w_star', 'sigma2c', 'sigma2w', 'zeta']
    if lattice.n > lattice.n_ava:
        estimate.append('d_star')
    with contextlib.redirect_stdout(io.StringIO()):
        sampler = MCMC(model=data_model, theta_init=data_true.theta, estimate=estimate,
                       theta_true=data_model.theta, chunk_size=N,
                       path=os.path.join(tmp, 'estim') + '/')
    kwargs = {'verbose': False, 'save_stats': False}
    if algorithm == 'collapsed_hmc':
        kwargs.update(adapt_config=NO_ADAPT, sigma2p=1.0, p_collapsed_hmc=1, **hmc_kwargs)
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        sampler.run(N, algorithm, **kwargs)
    seconds = sampler.stats['mwg']['exec_time']
    acc = sampler.stats['wc_update']['acceptance']['acc_iter']
    ar = float(np.mean(acc[1:N + 1])) if algorithm == 'collapsed_hmc' else np.nan
    unc = sampler.par_objs['w'].wavelet_constraints['unconstrained_indices']
    chains = sampler.theta['w_star'][unc, 1:]  # (k_free, N)
    return chains, seconds, ar


def omega_ess(chains, burn):
    X = chains[:, burn:]
    # A (nearly) constant chain has ESS ~ 0, but zero variance makes the
    # estimator return n; guard against stuck chains.
    ess = np.array([float(estimate_effective_sample_size(X[j])) if np.var(X[j]) > 1e-12
                    else 0.0 for j in range(X.shape[0])])
    return ess


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nv-obs', type=int, required=True)
    p.add_argument('--nh-obs', type=int, required=True)
    p.add_argument('-k', '--blur-length', type=int, required=True)
    p.add_argument('--mv', type=int, default=0)
    p.add_argument('--mh', type=int, default=0)
    p.add_argument('--m', type=int, required=True)
    p.add_argument('--gradient', default='adjoint')
    p.add_argument('--precondition', default='prior')
    p.add_argument('--eps-grid', default='0.05,0.1,0.2,0.4')
    p.add_argument('--L-grid', default='10,25,50')
    p.add_argument('--grid-N', type=int, default=300)
    p.add_argument('--final-N-hmc', type=int, default=0,
                   help='if > 0, run the tuned HMC and a Gibbs baseline this long')
    p.add_argument('--final-N-gibbs', type=int, default=0)
    p.add_argument('--seed', type=int, default=20)
    args = p.parse_args()

    eps_grid = [float(v) for v in args.eps_grid.split(',')]
    L_grid = [int(v) for v in args.L_grid.split(',')]
    precondition = None if args.precondition == 'none' else args.precondition

    label = (f"{args.nv_obs}x{args.nh_obs} k={args.blur_length} m={args.m} "
             f"mv={args.mv} mh={args.mh}")
    print(f"=== Tuning collapsed HMC ({args.gradient}, precondition={args.precondition}) "
          f"on {label} ===")
    with tempfile.TemporaryDirectory() as tmp:
        data_true, data_model, lattice = make_data(
            args.nv_obs, args.nh_obs, args.blur_length, args.mv, args.mh,
            args.m, args.seed, tmp)

        burn = args.grid_N // 5
        rows = []
        for L in L_grid:
            for eps in eps_grid:
                chains, secs, ar = run_chain(
                    data_true, data_model, lattice, 'collapsed_hmc', args.grid_N,
                    args.seed, tmp, epsilon=eps, L=L,
                    precondition=precondition, gradient=args.gradient)
                ess = omega_ess(chains, burn)
                kept_secs = secs * (args.grid_N - burn) / args.grid_N
                rows.append((eps, L, secs / args.grid_N * 1e3, ar,
                             ess.min(), np.median(ess), ess.min() / kept_secs))
                print(f"eps={eps:6.3f} L={L:3d} | {rows[-1][2]:7.2f} ms/iter | AR {ar:4.2f} "
                      f"| ESS min {ess.min():6.1f} med {np.median(ess):6.1f} "
                      f"| ESS_min/s {rows[-1][6]:8.2f}")

        best = max(rows, key=lambda r: r[6] if r[3] >= 0.2 else 0.0)  # exclude stuck chains
        eps_b, L_b = best[0], best[1]
        print(f"\nBest: eps={eps_b}, L={L_b} (AR {best[3]:.2f}, ESS_min/s {best[6]:.2f})")

        if args.final_N_hmc > 0:
            N_h, N_g = args.final_N_hmc, args.final_N_gibbs or 10 * args.final_N_hmc
            print(f"\n=== Final comparison: tuned HMC (N={N_h}) vs Gibbs (N={N_g}) ===")
            out = {}
            for alg, N, kw in (('collapsed_hmc', N_h,
                                dict(epsilon=eps_b, L=L_b, precondition=precondition,
                                     gradient=args.gradient)),
                               ('gibbs', N_g, {})):
                chains, secs, ar = run_chain(data_true, data_model, lattice, alg, N,
                                             args.seed, tmp, **kw)
                b = N // 5
                ess = omega_ess(chains, b)
                kept_secs = secs * (N - b) / N
                out[alg] = ess.min() / kept_secs
                print(f"{alg:14s} | {secs / N * 1e3:7.2f} ms/iter | AR {ar:4.2f} "
                      f"| ESS(omega) min {ess.min():7.1f} med {np.median(ess):7.1f} of {N - b} "
                      f"| ESS_min/s {out[alg]:8.2f} | ESS_min/iter {ess.min() / (N - b):.4f}")
            print(f"\nESS_min/s ratio HMC/Gibbs: "
                  f"{out['collapsed_hmc'] / out['gibbs']:.2f} ({label})")


if __name__ == '__main__':
    main()
