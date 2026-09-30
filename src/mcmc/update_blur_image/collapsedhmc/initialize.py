"""Initialization helpers for collapsed HMC.

Terminology: `c` is the image/reflectivity and `w` is the blur/wavelet.
"""

# Third-party library imports
import numpy as np
from scipy import linalg
import scipy

# Local library imports
from src.classes.lattice import Lattice
from src.classes.covariance import Covariance
from src.classes.gaussian import Gaussian
from src.utils.efficient_algebra_utils import multiply_circ, transpose_base_circ

def _initialize_collapsed_hmc(sampler, **kwargs):
    """Initialize the collapsed HMC sampler with necessary parameters and objects.
    Params:
    -------
    precondition: str, optional
        Preconditioning method for the momentum. Options are 'prior' or None.
        If 'prior', the momentum covariance matrix is the product of the wavelet
        marginal precision at the current iteration with the wavelet inverse prior
        precision."""
    
    # Extract parameters from kwargs with defaults
    verbose = sampler.mcmc_config.get('verbose', False)
    epsilon = kwargs.get('epsilon', 0.01)
    L = kwargs.get('L', 10)
    sigma2p = kwargs.get('sigma2p', 1)
    precondition = kwargs.get('precondition', None)
    p_collapsed_hmc = kwargs.get('p_collapsed_hmc', 0.5)
    gradient = kwargs.get('gradient', 'legacy')
    if gradient not in ('legacy', 'legacy_fixed', 'adjoint'):
        raise Exception(f"Unknown gradient variant '{gradient}'.")

    if verbose:
        print(f"Initializing collapsed HMC with epsilon={epsilon}, L={L}, sigma2p={sigma2p}.")

    # Save the collapsed HMC specific configuration
    sampler.mcmc_config['collapsed_hmc'] = {
        'epsilon': epsilon,
        'L': L,
        'sigma2p': sigma2p,
        'precondition': precondition,
        'p_collapsed_hmc': p_collapsed_hmc,
        'gradient': gradient
    }

    # Initialize the history of the params to the initial param value
    sampler.adapt_tracking['collapsed_hmc']['epsilon']['history'][:] = epsilon
    sampler.adapt_tracking['collapsed_hmc']['L']['history'][:] = L

    # Populate stats dictionary 
    N = sampler.mcmc_config['N']
    sampler.stats['wc_update'].update({
        'hamiltonian': np.zeros(N + 1),
        'potential': np.zeros(N + 1),
        'kinetic': np.zeros(N + 1),
    })
        
    # Create dictionary to store matrix products involving derivatives and constant across iterations
    sampler.aux_derivatives = {}

    # The adjoint gradient needs none of the derivative tensors below; free the
    # heavy model-level ones (built at setup) and skip the legacy-only arrays.
    if gradient == 'adjoint':
        wd = sampler.par_objs['w'].wavelet_derivatives
        for key in ('d_W0', 'd_W', 'bases_d_W', 'bases_d_WT', 'fft2_bases_d_W', 'fft2_bases_d_WT'):
            wd[key] = None
        return

    # Build gamma = G @ d_w_star_i (used in gradients for blur-kernel wavelet updates)
    _w = sampler.par_objs['w']
    wavelet_start_v = sampler.lattice.wavelet_positions['wavelet_start_v']
    wavelet_end_v = sampler.lattice.wavelet_positions['wavelet_end_v']
    gamma = []
    G = sampler.par_objs['c'].G
    bases_G = sampler.par_objs['c'].bases_G
    bases_GT = [transpose_base_circ(bases_G[:, j].reshape(-1, 1))
                for j in np.arange(sampler.lattice.nh)]

    for i in np.arange(sampler.lattice.nv):
        i_is_endpoint = (_w.wavelet_constraints['constr'] 
                         and ((i == wavelet_start_v) or (i == wavelet_end_v - 1)))
        i_is_padding = (i < wavelet_start_v or i >= wavelet_end_v)
        constrained = i_is_endpoint or i_is_padding
        
        # If the coordinate is constrained, gamma=0
        if constrained:
            gamma.append(np.zeros((sampler.lattice.n, 1)))
            
        # If it's free, we select the column i of the matrix G 
        else:
            if sampler.lattice.topology == 'E':
                # G is available explicitly
                gamma.append(G[:, i]) 

            elif sampler.lattice.topology == 'C':
                # If lattice is small enough, G is available explicitly
                try:
                    gamma.append(G[:, i]) 
                    
                # Otherwise, we work with the bases of the transpose of the circulant blocks in G
                except:
                    row_list = [np.roll(base, i) for base in bases_GT]
                    G_column = np.vstack(row_list)
                    gamma.append(G_column)
                    
            else:
                raise Exception(f'Unknown topology')

    sampler.aux['gamma'] = gamma
    sampler.aux_derivatives['gamma'] = gamma # this one is used in gradient computations
    nv = sampler.lattice.nv
    gamma_matrix = np.array([gamma[j].flatten() for j in range(nv)])  # shape: (nv, p_size)
    unconstrained_indices = sampler.par_objs['w'].wavelet_constraints['unconstrained_indices']
    gamma_matrix_unconstrained_indexes = gamma_matrix[unconstrained_indices, :]
    sampler.aux_derivatives['gamma_matrix'] = gamma_matrix
    sampler.aux_derivatives['gamma_matrix_unconstrained_indexes'] = gamma_matrix_unconstrained_indexes
    
    _c = sampler.par_objs['c']

    if sampler.lattice.topology == 'C':
        nv = sampler.lattice.nv
        base_Rv = _c.Sigma.base_Rv
        # sampler.aux_derivatives['bases_dW0_Rcv'] = np.array(bases_dW0_Rcv)

        # With an array instead of a list
        bases_dW0_Rcv = np.zeros((nv, nv))
        eigenvalues_dW0_Rcv = np.zeros((nv, nv), dtype=complex)
        for i in range(nv):
            base_d_W0_i = sampler.par_objs['w'].wavelet_derivatives['bases_d_W0'][:, i].reshape(-1, 1)
            base_dW0i_Rcv = np.squeeze(multiply_circ(base_d_W0_i, base_Rv))
            bases_dW0_Rcv[:, i] = base_dW0i_Rcv
            eigenvalues_dW0_Rcv[:, i] = scipy.fft.fft(base_dW0i_Rcv)
        sampler.aux['bases_dW0_Rcv'] = bases_dW0_Rcv
        sampler.aux_derivatives['bases_dW0_Rcv'] = bases_dW0_Rcv
        sampler.aux_derivatives['eigenvalues_dW0_Rcv'] = eigenvalues_dW0_Rcv

        # Build d_WO_i @ R_cv_star
        if sampler.par_objs['c'].reflectivity_constraints['constr']:
            dW0_Rcv_star = []
            Rv_star = _c.reflectivity_constraints['Rv_star']
            for i in np.arange(sampler.lattice.nv):
                base_d_W0_i = sampler.par_objs['w'].wavelet_derivatives['bases_d_W0'][:, i].reshape(-1, 1)
                d_W0_i = linalg.circulant(base_d_W0_i.reshape(-1)).T
                dW0i_Rcv_star = d_W0_i @ Rv_star
                dW0_Rcv_star.append(dW0i_Rcv_star)
            sampler.aux['dW0_Rcv_star'] = dW0_Rcv_star
            sampler.aux_derivatives['dW0_Rcv_star'] = dW0_Rcv_star

        # Build d_W_i @ R_c 
        bases_dW_Rc = []
        bases_dW_RcT = []
        _c = sampler.par_objs['c']
        base_R = _c.Sigma.base_R
        for i in np.arange(sampler.lattice.nv):
            base_d_W_i = sampler.par_objs['w'].wavelet_derivatives['bases_d_W'][i]
            base_dWi_Rc = multiply_circ(base_d_W_i, base_R)
            base_dWi_RcT = transpose_base_circ(base_dWi_Rc)
            bases_dW_Rc.append(base_dWi_Rc)
            bases_dW_RcT.append(base_dWi_RcT)
        sampler.aux['bases_dW_Rc'] = bases_dW_Rc
        sampler.aux['bases_dW_RcT'] = bases_dW_RcT
        sampler.aux_derivatives['bases_dW_Rc'] = bases_dW_Rc
        sampler.aux_derivatives['bases_dW_RcT'] = bases_dW_RcT

    # Create some temporary arrays for gradient computations
    d_SS_1_part1_temp = np.zeros((nv, 1), dtype=complex)
    d_SS_1_part2_temp = np.zeros((nv, 1), dtype=complex)
    d_SS_2_vectorized_temp = np.zeros((nv, 1), dtype=complex)
    temp_array_nvk = np.zeros((nv, len(unconstrained_indices)), dtype=complex) 
    temp_array_knvnh = np.zeros((len(unconstrained_indices), nv, sampler.lattice.nh), dtype=complex)
    temp_array_knvnh_2 = np.zeros((len(unconstrained_indices), nv, sampler.lattice.nh), dtype=complex)
    temp_array_knvnh_3 = np.zeros((len(unconstrained_indices), nv, sampler.lattice.nh), dtype=complex)
    trace_invYdY = np.zeros(len(unconstrained_indices))
    trace_invA_dA = np.zeros(len(unconstrained_indices))
    nr_constr_c = _c.reflectivity_constraints['nr_constraints']
    subset_base_dZ = np.zeros((len(unconstrained_indices), nr_constr_c, nr_constr_c))
    sampler.aux_derivatives['d_SS_1_part1_temp'] = d_SS_1_part1_temp
    sampler.aux_derivatives['d_SS_1_part2_temp'] = d_SS_1_part2_temp
    sampler.aux_derivatives['d_SS_2_vectorized_temp'] = d_SS_2_vectorized_temp
    sampler.aux_derivatives['temp_array_nvk'] = temp_array_nvk
    sampler.aux_derivatives['temp_array_knvnh'] = temp_array_knvnh
    sampler.aux_derivatives['temp_array_knvnh_2'] = temp_array_knvnh_2
    sampler.aux_derivatives['temp_array_knvnh_3'] = temp_array_knvnh_3
    sampler.aux_derivatives['trace_invYdY'] = trace_invYdY
    sampler.aux_derivatives['trace_invA_dA'] = trace_invA_dA
    sampler.aux_derivatives['subset_base_dZ'] = subset_base_dZ

def _create_momentum_object(sampler):
    """Create the momentum object for the collapsed HMC sampler."""

    # Extract Gaussian objects
    _lik = sampler.par_objs['d']
    _w = sampler.par_objs['w']
    verbose = sampler.mcmc_config.get('verbose', False)
    
    # Dimensions of the momentum
    dim_p = _lik.lattice.nv - _w.wavelet_constraints['nr_constraints'] 

    # Precondition?
    # NOTE: the option is stored under mcmc_config['collapsed_hmc'] (see _initialize_collapsed_hmc);
    # reading it from the top level always gave None, i.e. an identity mass matrix.
    precondition = sampler.mcmc_config['collapsed_hmc'].get('precondition', None)
    if precondition == 'prior':
        sigma2p = 1 / sampler.theta['sigma2w'][:, 0]
        mass_correlation_matrix = _w.wavelet_constraints['inv_R_wu_star']
        if scipy.sparse.issparse(mass_correlation_matrix):
            mass_correlation_matrix = mass_correlation_matrix.toarray()  # stored as csr; Covariance needs dense
        print(f"Using prior preconditioning with mass_correlation_matrix shape: {mass_correlation_matrix.shape}") if verbose else None
    else:
        sigma2p = sampler.mcmc_config['collapsed_hmc']['sigma2p']
        mass_correlation_matrix = np.eye(dim_p)
        
    # Instance the momentum object
    lattice_p = Lattice(nv_ava=dim_p, nh_ava=1, k=1, mv=0, mh=0, topology='E') 
    M = Covariance(lattice=lattice_p, sigma2=sigma2p, R=mass_correlation_matrix) 
    _p = Gaussian(Sigma=M)     

    # Factorize the mass matrix once; kinetic / grad_kinetic reuse it at every leapfrog step
    _p.Sigma.chol_R = linalg.cholesky(M.R, lower=True)

    # Store the momentum object in the sampler
    sampler.par_objs['p'] = _p