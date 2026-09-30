"""Summarize ESS / MSJD / timing for one or more MCMC runs.

Takes run directories produced by scripts/run_experiment.py (the folder that
contains <name>.pkl, <name>_stats.pkl and samples/), computes per-coordinate
ESS (Geyer's initial monotone sequence, ACF via FFT), MSJD, ms/iter and ESS/s
per parameter group (omega, c_u*, sigma2c, sigma2w, zeta), and writes a CSV
and a LaTeX table.

Example:
  python scripts/summarize_ess.py estimations/<fid>/gibbs/*/C_* \
      estimations/<fid>/collapsed_hmc/*/C_* \
      --burn-in 500 --out-dir reports/<fid>/tables --name pilot \
      --labels gibbs,hmc_adjoint_prior
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

from src.utils.file_utils import read_sampler_object, read_samples, read_stats
from src.utils.mcmc_diagnostics import (
    compute_mean_squared_jumping_distance_vector, estimate_effective_sample_size)


def load_run(run_dir):
    run_dir = os.path.normpath(run_dir)
    name = os.path.basename(run_dir)
    sampler = read_sampler_object(run_dir, '', name)
    raw = read_stats(run_dir, '', name)
    stats = raw.get('stats', raw)  # the stats pkl nests under 'stats'
    adapt = raw.get('adapt_tracking', {})
    chunk_size = sampler['file_config']['chunk_size']
    samples, _ = read_samples(run_dir, '', name, chunk_size=chunk_size)
    return sampler, stats, adapt, samples


def parameter_groups(sampler):
    """Row indices of each parameter group in the sample arrays."""
    lattice = sampler['lattice']
    wp = lattice.wavelet_positions
    omega = np.arange(wp['wavelet_start_v'] + 1, wp['wavelet_end_v'] - 1)  # free coords
    well = getattr(lattice, 'well_positions', None) or {}
    observed = well.get('well_coords_vec')
    if observed is None:
        c_u = np.arange(lattice.n)
    else:
        c_u = np.setdiff1d(np.arange(lattice.n), np.asarray(observed).ravel())
    return {'omega': ('w_star', omega), 'c_u_star': ('c_star', c_u),
            'sigma2c': ('sigma2c', np.array([0])), 'sigma2w': ('sigma2w', np.array([0])),
            'zeta': ('zeta', np.array([0]))}


def summarize_run(run_dir, label, burn_in):
    sampler, stats, adapt, samples = load_run(run_dir)
    N = sampler['mcmc_config']['N']
    exec_time = stats['mwg']['exec_time']
    ms_iter = exec_time / N * 1e3
    wc = stats.get('wc_update', {})
    # cum_ar is never filled by the step function; average the accept indicators instead
    if np.any(wc.get('iter_with_chmc', 0)):
        ar = float(np.mean(wc['acceptance']['acc_iter'][1:]))
    else:
        ar = np.nan
    eps_hist = adapt.get('collapsed_hmc', {}).get('epsilon', {}).get('history')
    eps_final = float(eps_hist[-1]) if eps_hist is not None else np.nan

    rows = []
    for group, (key, idx) in parameter_groups(sampler).items():
        chains = samples[key][idx, burn_in:]
        kept = chains.shape[1]
        seconds = kept * ms_iter / 1e3
        ess = np.array([float(estimate_effective_sample_size(chains[j])) for j in range(len(idx))])
        rows.append({
            'label': label, 'group': group, 'n_coords': len(idx), 'N': N, 'kept': kept,
            'ms_per_iter': ms_iter, 'acc_rate': ar, 'eps_final': eps_final,
            'ess_min': ess.min(), 'ess_med': np.median(ess),
            'ess_q2.5': np.percentile(ess, 2.5), 'ess_q97.5': np.percentile(ess, 97.5),
            'ess_min_per_s': ess.min() / seconds, 'ess_med_per_s': np.median(ess) / seconds,
            'msjd': compute_mean_squared_jumping_distance_vector(chains.T),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='+')
    p.add_argument('--burn-in', type=int, default=0)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--name', default='ess_summary')
    p.add_argument('--labels', default=None,
                   help='comma-separated labels, one per run dir (default: folder names)')
    args = p.parse_args()

    labels = (args.labels.split(',') if args.labels
              else [os.path.basename(os.path.normpath(d)) for d in args.run_dirs])
    if len(labels) != len(args.run_dirs):
        raise SystemExit('Number of labels must match number of run dirs.')

    rows = []
    for run_dir, label in zip(args.run_dirs, labels):
        rows.extend(summarize_run(run_dir, label, args.burn_in))
    df = pd.DataFrame(rows)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'{args.name}.csv')
    tex_path = os.path.join(args.out_dir, f'{args.name}.tex')
    df.to_csv(csv_path, index=False, float_format='%.4g')
    df.to_latex(tex_path, index=False, float_format='%.3g')
    print(f'\nWrote {csv_path} and {tex_path}.')
    with pd.option_context('display.width', 200):
        print(df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))


if __name__ == '__main__':
    main()
