"""MCMC diagnostic utilities (ESS, MSJD).

These are lightweight helpers for notebooks and analysis.
"""

from __future__ import annotations

import numpy as np


def compute_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for a centered 1D series up to max_lag."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array")

    n = len(x)
    if n == 0:
        return np.zeros(max_lag + 1)

    variance = np.var(x)
    if variance == 0:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(x[lag:] * x[:-lag]) / ((n - lag) * variance)
    return acf


def integrated_autocorrelation_time(acf: np.ndarray) -> float:
    """Compute integrated autocorrelation time using Geyer's monotone sequence."""
    acf = np.asarray(acf)
    if acf.ndim != 1:
        raise ValueError("ACF must be 1D")

    max_lag = len(acf) - 1
    if max_lag <= 0:
        return 1.0

    sums = np.zeros(max_lag // 2)
    for i in range(max_lag // 2):
        sums[i] = acf[2 * i + 1] + acf[2 * i + 2]

    cutoff = len(sums) - 1
    for i in range(1, len(sums)):
        if sums[i] < 0 or sums[i] > sums[i - 1]:
            cutoff = i
            break

    m = 2 * cutoff
    tau = 1.0 + 2.0 * np.sum(acf[1 : m + 1])
    return max(1.0, float(tau))


def estimate_effective_sample_size(chain: np.ndarray, max_lag: int | None = None) -> np.ndarray | float:
    """Estimate ESS for a 1D or 2D chain (samples x parameters)."""
    chain = np.asarray(chain)

    if chain.ndim == 2:
        return np.array([estimate_effective_sample_size(chain[:, i], max_lag=max_lag)
                         for i in range(chain.shape[1])])
    if chain.ndim != 1:
        raise ValueError("Chain must be 1D or 2D")

    n = len(chain)
    if n == 0:
        return 0.0

    if max_lag is None:
        max_lag = min(int(n / 5), 1000)

    centered = chain - np.mean(chain)
    acf = compute_autocorrelation(centered, max_lag)
    tau = integrated_autocorrelation_time(acf)
    return n / tau


def compute_squared_jumping_distance(chain: np.ndarray) -> np.ndarray:
    """Compute squared jumping distance for consecutive samples."""
    chain = np.asarray(chain)

    if chain.ndim == 1:
        return np.diff(chain) ** 2
    if chain.ndim == 2:
        diffs = np.diff(chain, axis=0)
        return np.sum(diffs ** 2, axis=1)

    raise ValueError("Chain must be 1D or 2D")


def compute_mean_squared_jumping_distance(chain: np.ndarray) -> np.ndarray | float:
    """Compute MSJD for a 1D or 2D chain (samples x parameters)."""
    chain = np.asarray(chain)

    if chain.ndim == 2:
        return np.array([compute_mean_squared_jumping_distance(chain[:, i])
                         for i in range(chain.shape[1])])
    if chain.ndim != 1:
        raise ValueError("Chain must be 1D or 2D")

    sjd = compute_squared_jumping_distance(chain)
    return float(np.mean(sjd))


def compute_mean_squared_jumping_distance_vector(chain: np.ndarray) -> float:
    """Compute MSJD for a vector-valued chain (samples x parameters).

    Returns the mean of squared Euclidean distances between consecutive samples.
    """
    chain = np.asarray(chain)
    if chain.ndim == 1:
        return float(np.mean(np.diff(chain) ** 2))
    if chain.ndim != 2:
        raise ValueError("Chain must be 1D or 2D")

    diffs = np.diff(chain, axis=0)
    sjd = np.sum(diffs ** 2, axis=1)
    return float(np.mean(sjd))
