"""Mass-matrix estimation for collapsed HMC.

The momentum is p ~ N(0, M) with M = sigma2p * R (see _create_momentum_object).
`precondition` selects R and sigma2p:
- None: R = I, sigma2p from the config (identity mass matrix).
- 'prior': R = R_w*^{-1}, sigma2p = 1 / sigma_w^2 at the current iteration,
  i.e. M = R_w*^{-1} / sigma_w^2 as stated in the paper.
- 'posterior': R = fixed free-coordinate posterior precision estimated from a
  pilot chain with estimate_mass_matrix and passed as
  mcmc_config['collapsed_hmc']['mass_matrix']; sigma2p from the config.
No online adaptation.
"""

# Third-party library imports
import numpy as np
from scipy import linalg


def estimate_mass_matrix(w_samples, _w, ridge=1e-6):
    """Inverse of the regularised sample covariance of the free wavelet coordinates.

    Params:
    -------
    w_samples: np.array nv x N
        w* samples from a pilot chain (N > 1).
    _w: Gaussian
        Wavelet object (provides the unconstrained indices).
    ridge: float
        Relative ridge: ridge * mean(diag) is added to the covariance diagonal.

    Return:
    -------
    np.array k_free x k_free
        Precision matrix to pass as mcmc_config['collapsed_hmc']['mass_matrix'].
    """
    unc = _w.wavelet_constraints['unconstrained_indices']
    X = np.asarray(w_samples)[unc, :]
    C = np.cov(X)
    C = 0.5 * (C + C.T) + ridge * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    M = linalg.inv(C)
    return 0.5 * (M + M.T)
