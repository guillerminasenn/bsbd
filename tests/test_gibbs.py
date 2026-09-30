"""New vs old Gibbs constraint corrections: identical c_star, w_star for the
same unconstrained draw."""
import numpy as np
from scipy import linalg

from src.utils.model_utils import subset_AB, subset_ABA
from src.utils.efficient_algebra_utils import build_matrix_circ
from src.mcmc.update_blur_image.gibbs.step_gibbs import step_gibbs

TOL = 1e-10


def test_gibbs_corrections_match_old(build_sampler, lattice_name):
    sampler, lattice = build_sampler(lattice_name, None)  # full well column
    _image = sampler.par_objs['c']
    _blur = sampler.par_objs['w']

    # Conditional for the blur, at the pre-update state used inside step_gibbs
    mean_w, base_Rw, R_cond_w = _blur.recompute_mean_R(sampler, 0)

    np.random.seed(42)
    step_gibbs(sampler, 0)

    # ---- blur: old correction (full circulant build) on the saved unconstrained draw ----
    w_unc = sampler.aux['w'][:, 1].reshape(-1, 1)
    b_w = _blur.wavelet_constraints['b']
    A_w = _blur.wavelet_constraints['A']
    mask = A_w.getnnz(axis=0) > 0
    R_w = build_matrix_circ(base_Rw) if R_cond_w is None else R_cond_w
    RAt = R_w[:, mask]
    ARAt = RAt[mask, :]
    Aw_b = w_unc[mask, :].reshape(-1, 1) - b_w.reshape(-1, 1)
    chol = linalg.cholesky(ARAt, lower=True)
    v = linalg.solve_triangular(chol.T, linalg.solve_triangular(chol, Aw_b, lower=True), lower=False)
    w_star_old = w_unc - RAt @ v
    assert np.max(np.abs(sampler.theta['w_star'][:, 1] - w_star_old.ravel())) < TOL

    # ---- image: old correction (subset_ABA / subset_AB / inv) ----
    # The image conditional depends on w_star[:, i+1] (final) and the variances, so it
    # can be recomputed after the step.
    mean_c, base_Rc, _ = _image.recompute_mean_R(sampler, 0)
    c_unc = sampler.aux['c'][:, 1].reshape(-1, 1)
    b_c = _image.reflectivity_constraints['b']
    A_c = _image.reflectivity_constraints['A_co']
    ARAt_c = subset_ABA(base_Rc, _image)
    ARAt_c = 0.5 * (ARAt_c + ARAt_c.T)
    RAt_c = subset_AB(base_Rc, _image).T
    inv_c = linalg.inv(ARAt_c)
    inv_c = 0.5 * (inv_c + inv_c.T)
    c_star_old = c_unc - RAt_c @ (inv_c @ (A_c.dot(c_unc).reshape(-1, 1) - b_c.reshape(-1, 1)))
    assert np.max(np.abs(sampler.theta['c_star'][:, 1] - c_star_old.ravel())) < TOL


def test_gibbs_no_image_constraints(build_sampler, lattice_name):
    """With m = 0 the image sample is saved uncorrected."""
    sampler, lattice = build_sampler(lattice_name, 0)
    np.random.seed(42)
    step_gibbs(sampler, 0)
    assert np.array_equal(sampler.theta['c_star'][:, 1], sampler.aux['c'][:, 1])
