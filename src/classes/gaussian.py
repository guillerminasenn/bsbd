"""Gaussian priors and conditionals for blur and image fields.

Terminology: `w` is the blur/wavelet and `c` is the image/reflectivity.
"""

# Standard library imports
import math

# Third-party library imports
import numpy as np
import pandas as pd
from scipy import linalg, stats, sparse
import scipy

# Model imports
from src.classes.covariance import Covariance
from src.utils.model_utils import create_C, create_W, subset_AB, subset_ABA
from src.mcmc.derivs import derivative_W0_i

# Custom math and efficient algebra on cyclic lattices
from src.utils import math_utils 
from src.utils.efficient_algebra_utils import transpose_base_circ, invert_circ, multiply_circ, multiply_matrix_vector_circ

# Gaussian constrained and unconstrained sampling
from src.utils.sampling_utils import _sample_scipy, _sample_circ, _correct_sample

# Recomputing wavelet full conditional parameters
from src.utils.sampling_utils import _compute_bases_Gammat_inv_Sigma_d, _compute_base_Gammat_inv_Sigma_d_Gamma, _compute_wavelet_conditional_covariance, _compute_Gt_inv_Sigma_d_d, _compute_wavelet_conditional_mean


pp = 4

###################### ###################### ###################### 
######################       Gaussian         ######################
###################### ###################### ######################

class Gaussian():
    """My own multivariate Gaussian class for Euclidean and cyclic lattices. 
    Contains methods to evaluate the pdf, logpdf and to sample.
    """
    
    def __init__(self, mean=None, Sigma=None, verbose=False):
        """
        Params:
        -------
        mean: nx1 array
            Optional.
        Sigma: Covariance object.
        """

        # Save lattice dimensions
        try:
            self.nv = next(item.shape[0] for item in [Sigma.Rv, Sigma.base_Rv] if item is not None)
            self.nh = next(item.shape[0] for item in [Sigma.Rh, Sigma.base_Rh] if item is not None)
            self.n = self.nv * self.nh
        except:
            try:
                self.n = next(item.shape[0] for item in [mean, Sigma.R] if item is not None)
            except:
                raise Exception('Not enough information to assess the dimensions of the multivariate gaussian.')
        
        # Mean
        if mean is None: 
            if verbose: print('Gaussian init: mean=None, so assuming mean=0. ')
            self.mean = np.zeros((self.n, 1))
        else:
            if self.n != len(mean): 
                raise Exception("The desired dimensions are {0}x{1}={2} but the mean vector has length {3}.".format(nv, nh, self.n, len(self.mean)))
            self.mean = mean 
        
        # Sigma
        if not isinstance(Sigma, Covariance):
            raise ValueError("Sigma must be an instance of Covariance.")
        else:
            self.Sigma = Sigma
            self.lattice = self.Sigma.lattice
            
    def update_Sigma(self, Sigma):
        """Replace the covariance matrix."""
        if not isinstance(Sigma, Covariance):
            raise ValueError("Sigma must be an instance of Covariance.")
        else:
            self.Sigma = Sigma

    def update_mean(self, mean):
        """Replace the mean."""
        self.mean = mean
        
    def update_sigma(self, sigma2):
        """Replace the marginal variance."""
        self.Sigma.sigma2 = sigma2

    def pdf_scipy(self, x):
        "Euclidean. Evaluate pdf at x using the method from scipy's multivariate_normal."  
        cov = self.Sigma.sigma2 * self.Sigma.R
        gaussian = stats.multivariate_normal(np.squeeze(self.mean), cov)
        pdf_x = gaussian.pdf(np.squeeze(x))
        return pdf_x
    
    def logpdf_scipy(self, x, sigma2=None):
        "Euclidean. Evaluate logpdf at x using the method from scipy's multivariate_normal."  
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        cov = sigma2 * self.Sigma.R
        gaussian = stats.multivariate_normal(np.squeeze(self.mean), cov)
        logpdf_x = gaussian.logpdf(np.squeeze(x))
        return logpdf_x
    
    def pdf(self, x, sigma2=None):
        """Euclidean. Evaluate pdf at x using the formula for the Gaussian density 
        and traditional matrix algebra."""
        if x.shape[0] != self.n:
            raise Exception("The input vector has the wrong dimensions.")
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        cov = sigma2 * self.Sigma.R 
        x_centered = x.reshape((self.n, 1)) - self.mean.reshape((self.n, 1))
        inv_cov = linalg.inv(cov)
        ss = - 0.5 * (x_centered.T) @ (inv_cov @ x_centered) 
        k1 = (2 * math.pi) ** (-0.5 * self.n) 
        k2 = np.exp(math_utils.compute_log_det(cov, epsilon=0.00000001)) ** (-0.5) 
        k3 = np.exp(ss)
        pdf_x = k1 * k2 * k3
        return pdf_x
    
    def logpdf(self, x, sigma2=None):
        """Euclidean. Evaluate logpdf at x using the formula for the Gaussian density 
        and traditional matrix algebra."""
        if x.shape[0] != self.n:
            raise Exception("The input vector has the wrong dimensions.")
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        cov = sigma2 * self.Sigma.R
        x_centered = x.reshape((self.n, 1)) - self.mean.reshape((self.n, 1))
        inv_cov = linalg.inv(cov)
        ss = - 0.5 * (x_centered.T) @ (inv_cov @ x_centered)
        l1 = -0.5 * self.n * np.log(2 * math.pi)
        l2 = -0.5 * np.log(linalg.det(cov))
        logpdf_x = l1 + l2 + ss
        return logpdf_x
        
    def ss_circ(self, x, sigma2=None):
        """Cyclic. Computes the sum of squares x'Qx with the FFT, Q circulant."""
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        x_centered = x.reshape((self.n, 1)) - self.mean
        base_x = x_centered.reshape((self.nv, self.nh), order='F')
        internal = (np.fft.fft2(self.Sigma.base_Q / sigma2, norm='ortho') 
                    * np.fft.ifft2(base_x, norm='ortho'))
        temp = np.sqrt(self.n) * np.fft.fft2(internal, norm='ortho')
        temp = np.real(temp.reshape((self.n, 1), order='F'))
        sum_squares = (x_centered.T) @ temp
        return sum_squares
    
    def pdf_circ(self, x):
        """Cyclic. Evaluate pdf at x using the FFT."""
        pdf_x = np.exp(self.logpdf_circ(x)) # yes im cheating
        return pdf_x
    
    def logpdf_circ(self, x, sigma2=None):
        """Cyclic. Evaluate logpdf at x using the FFT."""
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        ss = self.ss_circ(x)[0]
        l1 = -0.5 * self.n * np.log(2 * math.pi)
        l2 = 0.5 * math_utils.compute_log_det_circ(
            self.Sigma.base_Q / sigma2, 
            discard=False, epsilon=0.000000001)
        l3 = -0.5 * ss
        logpdf_x = l1 + l2 + l3
        return logpdf_x

    def sample_scipy(self, mean=None, sigma2=None, R=None, verbose=False):
        """Euclidean. Draw random samples from a multivariate normal distribution 
        using scipy's multivariate_normal."""
        if mean is None:
            mean = self.mean
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        if R is None:
            R = self.Sigma.R
        if verbose:
            print(f"Sampling from Gaussian with sigma2={sigma2}.")
        x = _sample_scipy(mean, sigma2, R)
        return x

    def sample_cholesky(self, size=1, sigma2=None, verbose=False):
        """Euclidean. Draw random samples from a multivariate normal distribution 
        using the Cholesky decomposition."""
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        gaussian = stats.multivariate_normal()
        z = gaussian.rvs(self.n * size)
        Z = z.reshape((self.n, size))
        lower_chol = linalg.cholesky(sigma2 * self.Sigma.R, lower=True)
        x = self.mean.reshape((self.n, 1)) + lower_chol @ Z
        x = x.T
        
        # Reshape x into (1,1) np.array if it only contains one element
        if not x.shape:
            x = x.reshape((1, 1))
        else:
            x = x.reshape((self.n, 1))
            
        return x

    def sample_circ(self, mean=None, sigma2=None, base_R=None):
        """Sample from a multivariate normal distribution on a cyclic lattice
        using the FFT."""
        
        if self.lattice.topology != 'C':
            raise Exception('Trying to use a circulant method for a non-circulant topology.')
                          
        if mean is None:
            mean = self.mean
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2       
        if base_R is None:
            base_R = self.Sigma.base_R
        x = _sample_circ(mean, sigma2, base_R)
        return x

    def sample(self, mean=None, sigma2=None, base_R=None, R=None):
        """Automatically choose the sampling method."""

        if mean is None:
            mean = self.mean
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2    
        
        if self.lattice.topology == 'E':
            if R is None:
                R = self.Sigma.R
            x = _sample_scipy(mean, sigma2, R)
            return x
        
        if self.lattice.topology == 'C':
            if base_R is None:
                base_R = self.Sigma.base_R
            x = _sample_circ(mean, sigma2, base_R)
            return x

###################### ###################### ###################### 
######################     Reflectivity       ######################
###################### ###################### ###################### 

class Reflectivity(Gaussian):
    """The Gaussian field used to represent the image/reflectivity. The class includes 
    methods to do conditioning by Kriging and to compute the conditional parameters 
    of the reflectivity full conditional with the DFT."""
    
    def __init__(self, mean=None, Sigma=None, b=None, verbose=False):
        """
        Params:
        -------
        mean:
        Sigma: Covariance object.
        b: constraints
        
        """

        # Is the covariance matrix based on a lattice?
        if not hasattr(Sigma, 'lattice'):
            raise Exception('The Covariance matrix must be based on a Lattice object so I know if I need to apply constraints.')
        
        else:
            super().__init__(mean, Sigma)
            
            # Set the constraint flag
            constr = hasattr(self.lattice, 'well_positions') and (b is not None)
            if verbose: 
                print(f'Linear constraints on reflectivity? {constr}.')
            if constr:
                nr_constraints = b.shape[0]
            else: 
                nr_constraints = 0

            # Save constraint flag, nr of constraints, the cosntraints
            self.reflectivity_constraints = {
                'constr': constr, 
                'nr_constraints': nr_constraints,
                'b':b
            }
            
            # Compute attributes
            self._compute_attributes_c(verbose)    

    def _create_reflectivity_constraint_matrices(self, verbose=False):
        """Create constraint matrices that select the observed (c_o) or
        unobserved (c_u) reflectivities from the vectorized reflectivity c
        on the extended lattice.

        Return:
        -------
        A_co_h: Selection matrix for the column where the well is located.
        A_co_v: Selection matrix for the rows where the well is located.
        A_co: Contraint matrix to select the well c_o from c
        A_co_ext: Constraint matrix to select the well from (c, d).
        A_cu: Constraint matrix to subset c_u from c.
        A_co_one: Matrix that has a 1 for the elements in c corresponding to c_o.
        A_co_h_one:
        A_co_v_one:
        """
        
        # Initialize matrices to None
        A_co_h = None, 
        A_co_v = None, 
        A_co = None, 
        A_co_ext = None, 
        A_cu = None, 
        A_co_h_one = None, 
        A_co_v_one = None
        
        # Extract variables for readability
        nh = self.lattice.nh
        nv = self.lattice.nv
        n = self.lattice.n
        well_column = self.lattice.well_positions['well_column']
        well_start_v = self.lattice.well_positions['well_start_v']
        rows_ava = self.lattice.well_positions['rows_ava']
        nr_constraints = self.reflectivity_constraints['nr_constraints']

        # Helper matrices
        A_co_h = sparse.csr_matrix((1, nh)) 
        A_co_h[0, well_column] = 1
        zeros = sparse.csr_matrix((nr_constraints, n)) # Dont select elements of (d)
        A_co_h_one = np.zeros((nh, nh))
        A_co_h_one[well_column, well_column] = 1
        A_co_v_one = np.zeros((nv, nv))
        A_zero_co = sparse.identity(n, format='csr') # n x n
            
        # Default: the reflectivity is observed over the whole well_column
        if rows_ava is None:
            A_co_v = sparse.csr_matrix((nr_constraints, nv)) 
            for i in range(nr_constraints):
                A_co_v[i, i + well_start_v] = 1
            
            # Matrix that has a 1 in the diagonal for elements in c corresponding to c_o
            for i in np.arange(well_start_v, well_start_v):
                A_co_v_one[i, i] = 1
        
        # The reflectivity is observed at some rows within the well_column
        else:
            A_co_v = sparse.csr_matrix((nr_constraints, nv)) 
            for i in range(nr_constraints):
                A_co_v[i, self.lattice.mv // 2 + rows_ava[i]] = 1
                
        # Constraint matrix to select c_o from c 
        A_co = sparse.kron(A_co_h, A_co_v) # (nv_ava x n),
        
        # Constraint matrix to select c_o from (d, c)
        A_co_ext = sparse.hstack([zeros, A_co]) # (nv_ava x 2n)

        # Matrix with rows equal to zero for elements in c corresponding to c_o
        for i in self.lattice.well_positions['well_coords_vec']: 
            A_zero_co[i, i] = 0 

        # Constraint matrix to subset c_u from c
        keep_rows = [i for i in range(A_zero_co.shape[0]) 
                     if i not in self.lattice.well_positions['well_coords_vec']]
        A_cu = A_zero_co[keep_rows, :]            
        
        self.reflectivity_constraints.update({
            'A_co_h':A_co_h, 
            'A_co_v':A_co_v, 
            'A_co':A_co, 
            'A_co_ext':A_co_ext, 
            'A_cu':A_cu, 
            'A_co_h_one':A_co_h_one, 
            'A_co_v_one':A_co_v_one
        })
        
    def _compute_attributes_c(self, verbose=False):
        """Creates reflectivity convolutional matrix, constraint matrices and the 
        mean and covariance of the constrained prior. 
        """  
        
        if not self.reflectivity_constraints['constr']:
            
            # Create the reflectivity convolutional matrix and its base
            bases_G, G = create_C(self.lattice, self.mean) 
            self.bases_G = bases_G
            self.G = G 
            
        if self.reflectivity_constraints['constr']:
            
            # Build constraint matrices
            self._create_reflectivity_constraint_matrices(verbose)
            
            # Construct R @ A' and A @ R @ A' 
            A = self.reflectivity_constraints['A_co']
            if self.lattice.topology == 'C':
                    self._subset_R_constraints(verbose=verbose) 
            if self.lattice.topology == 'E':
                self.reflectivity_constraints['RAt'] = (A.dot(self.Sigma.R)).T
                self.reflectivity_constraints['ARAt'] = A.dot(self.reflectivity_constraints['RAt'])
            
            # Compute inverse, decomposition, and logdet of A @ R @ A' 
            inv_ARAt = linalg.inv(self.reflectivity_constraints['ARAt'])
            inv_ARAt = 0.5 * (inv_ARAt + inv_ARAt.T)  # Ensure symmetry
            self.reflectivity_constraints['inv_ARAt'] = inv_ARAt
            self.reflectivity_constraints['L'] = linalg.cholesky(self.reflectivity_constraints['inv_ARAt']).T
            self.reflectivity_constraints['logdet_ARAt'] = math_utils.compute_log_det(self.reflectivity_constraints['ARAt'])
            
            # Compute logdet of A @ A' 
            self.reflectivity_constraints['logdet_AAt'] = math_utils.compute_log_det((A.dot(A.T)).toarray())
            
            # Compute mu_c*, the mean of p(c|Ac=co)
            inv_ARAt_b = self.reflectivity_constraints['inv_ARAt'] @ self.reflectivity_constraints['b']
            mean_c_star = (
                self.reflectivity_constraints['RAt'] @ inv_ARAt_b)
            self.reflectivity_constraints['mean_c_star'] = mean_c_star

            # Compute the sum of squares of the correction term
            self.reflectivity_constraints['constrained_ss'] = (self.reflectivity_constraints['b'].T) @ inv_ARAt_b
            
            # Create the convolutional matrix G and its base
            bases_G, G = create_C(self.lattice, mean_c_star) 
            print(f"In _compute_attributes_c, mean_c_star.shape: {mean_c_star.shape}, and mean_c_star[:3]: {mean_c_star[:3]}")
            print(f"bases_G.shape = {bases_G.shape}, bases_G[:3, 0] = {bases_G[:3, 0]}")
            self.bases_G = bases_G
            self.G = G 
    
            # Compute R_c*, the correlation matrix of p(c|Ac=co), and its pseudo-inverse 
            if self.lattice.n < 300:
                try:
                    R_c_star = (self.Sigma.R
                                     - self.reflectivity_constraints['RAt']
                                     @ self.reflectivity_constraints['inv_ARAt'] 
                                     @ (self.reflectivity_constraints['RAt'].T))
                    R_c_star = 0.5 * (R_c_star + R_c_star.T)  # Ensure symmetry
                    log_pos_eigs, inv_R_c_star = math_utils.pseudo_inverse_RH(R_c_star)
                    inv_R_c_star = 0.5 * (inv_R_c_star + inv_R_c_star.T)  # Ensure symmetry
                    self.reflectivity_constraints['log_pos_eigs'] = log_pos_eigs
                except Exception as e:
                    print('Couldnt create R_c_star and its pinv', e)
                    R_c_star = inv_R_c_star = None
            else:
                print(f'Lattice is too big to create R_c_star explicitly. R_c_star and inv_R_c_star will be None.')
                R_c_star = inv_R_c_star = None
            self.reflectivity_constraints['R_c_star'] = R_c_star
            self.reflectivity_constraints['inv_R_c_star'] = inv_R_c_star # pseudoinverse
            
            # Compute the Kronecker factors Rh* and Rv* in the correction term of the constrained 
            Rh_star = None
            Rv_star = None
            try:
                Rh = self.Sigma.Rh
                A_co_h = self.reflectivity_constraints['A_co_h']
                RA = (A_co_h.dot(Rh)).T
                inv_ARA = linalg.inv(A_co_h.dot(RA))
                Rh_star = RA @ inv_ARA @ (RA.T)
                # Rh_star_old = (A_co_h.dot(Rh)).T @ (A_co_h.dot(Rh)) # This assumes only 1 well column
                
                
                Rv = self.Sigma.Rv
                A_co_v = self.reflectivity_constraints['A_co_v']
                RA = A_co_v.dot(Rv).T
                inv_ARA = linalg.inv(A_co_v.dot(RA))
                Rv_star = RA @ inv_ARA @ (RA.T)
                
            except:
                pass
            self.reflectivity_constraints['Rh_star'] = Rh_star
            self.reflectivity_constraints['Rv_star'] = Rv_star     

            # Compute mu_cu*, R_cu*, the parameters of p(cu|Ac=co), and inv R_cu*
            self.reflectivity_constraints['mean_cu_star'] = self.reflectivity_constraints['A_cu'].dot(mean_c_star)
            
            if R_c_star is not None and self.reflectivity_constraints['nr_constraints'] < self.lattice.nv_ava:
                R_cu_star = (self.reflectivity_constraints['A_cu'].dot(R_c_star)
                                 @ (self.reflectivity_constraints['A_cu'].T))
                self.reflectivity_constraints['R_cu_star'] = R_cu_star
                self.reflectivity_constraints['inv_R_cu_star'] = linalg.inv(R_cu_star)
            else:
                if R_c_star is None: 
                    print("Won't compute R_cu_star and inv_R_cu_star because R_c_star is None")
                else:
                    print("Won't compute R_cu_star and inv_R_cu_star because the well occupies a whole column.")
                self.reflectivity_constraints['R_cu_star'] = None
                self.reflectivity_constraints['inv_R_cu_star'] = None
            
    
    def _subset_R_constraints(self, verbose=False):
        """
        Given the base of the BCCB reflectivity correlation matrix, construct the 
        matrices R @ A' and A @ R @ A' by subsetting the right elements
        from the base.    

        Return:
        -------
        RAt: R @ A'.
        ARAt: A @ R @ A'. 
        """
        base_B = self.Sigma.base_R
        RAt = subset_AB(base_B, self, verbose=verbose).T
        ARAt = subset_ABA(base_B, self, verbose=verbose)
        ARAt = 0.5 * (ARAt + ARAt.T)  # Ensure symmetry
        self.reflectivity_constraints.update({'RAt':RAt, 'ARAt':ARAt})
    
    def recompute_mean_R(self, sampler, i):
        """Recompute the mean and correlation matrix in the reflectivity full 
        conditional using the current state stored in the sampler object. 
        Works for both Euclidean and cyclic lattices.

        Params:
        -------
        sampler: Sampler object
            The sampler containing the current state and model.
        i: int
            The current iteration index.
        """
        verbose = sampler.mcmc_config.get('verbose', False)
        nv = sampler.lattice.nv
        print("Recomputing conditional mean and correlation matrix in Reflectivity...") if verbose else None

        # Extract the Gaussian objects
        _lik = sampler.par_objs['d']
        _c = sampler.par_objs['c']

        # Extract current states
        w_star = sampler.theta['w_star'][:, i + 1].reshape((-1, 1))
        d_star = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
        sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
        sigma2d = sampler.aux['sigma2d'][:, i + 1].item()
        
        if sampler.lattice.topology == 'C':

            # Create wavelet convolutional matrix
            W0, W, base_W0, base_W, base_W0T, base_WT = create_W(sampler.lattice, w_star)

            # Extract bases of the prior correlation matrices for d and c
            base_Sigma_d = sigma2d * _lik.Sigma.base_R
            base_R = _c.Sigma.base_R

            # Compute the reflectivity full conditional correlation matrix
            base_RWt = multiply_circ(base_R, base_WT)
            base_WR = transpose_base_circ(base_RWt)
            base_WSigmacWt = multiply_circ(base_W, base_RWt)
            base_WSigmacWt = sigma2c * 0.5 * (base_WSigmacWt + transpose_base_circ(base_WSigmacWt))  # Ensure symmetry
            base_Z = base_WSigmacWt + base_Sigma_d
            base_Z = 0.5 * (base_Z + transpose_base_circ(base_Z))
            base_inv_Z = invert_circ(base_Z)
            base_inv_Z = 0.5 * (base_inv_Z + transpose_base_circ(base_inv_Z))
            base_correction_left = sigma2c * multiply_circ(base_RWt, base_inv_Z)
            base_correction = multiply_circ(base_correction_left, base_WR)
            base_R_cond = base_R - base_correction
            base_R_cond = 0.5 * (base_R_cond + transpose_base_circ(base_R_cond))  # Ensure symmetry
            R_cond = None
            
            # Compute the reflectivity full conditional mean
            mean_cond = multiply_matrix_vector_circ(base_correction_left, d_star)
        
        elif sampler.lattice.topology == 'E':

            # Create wavelet convolutional matrix
            W0, W, base_W0, base_W, base_W0T, base_WT = create_W(sampler.lattice, w_star, explicit=True)

            # Extract prior correlation matrices for d and c
            Sigma_d = sigma2d * _lik.Sigma.R
            R = _c.Sigma.R

            # Compute reflectivity full conditional correlation matrix
            RWt = R @ (W.T)
            Z = sigma2c * W @ RWt + Sigma_d
            inv_Z = linalg.inv(Z)
            correction_left = sigma2c * RWt @ inv_Z
            correction = correction_left @ (RWt.T)
            R_cond = R - correction
            base_R_cond = None

            # Compute reflectivity full conditional mean
            mean_cond = correction_left @ d_star

        # Return updated mean and base of correlation matrix
        print(f"Recomputed reflectivity conditional params using w_Star={sampler.theta['w_star'][(nv//2 - 5):(nv//2 + 5), i + 1].reshape((-1, 1))}") if verbose else None
        return mean_cond, base_R_cond, R_cond

    def correct_sample(self, x, base_R_cd=None, R_cd=None, verbose=False):
        """Euclidean and cyclic. Correct x with conditioning by Kriging so that Ax=b.
        Params:
        -------
        x: (nv x 1) np.array.
            The reflectivity sample to be corrected.
        base_R_cd: (nv x nh) np.array
            The base of the BCCB full conditional correlation matrix. Only used for the 
            full conditional reflectivity Gibbs step.
        """

        # Load the constraint matrices
        b = self.reflectivity_constraints['b'] 
        A = self.reflectivity_constraints['A_co']

        # Compute the corrected sample
        if self.lattice.topology == 'C':
            
            # If the base of the BCCB is passed, compute the matrices RAt, ARAt, and inv_ARAt by subsetting the base
            if base_R_cd is not None:
                ARAt = subset_ABA(base_R_cd, self)
                RAt = subset_AB(base_R_cd, self)
                inv_ARAt = linalg.inv(ARAt)       
                inv_ARAt = 0.5 * (inv_ARAt + inv_ARAt.T)  # Ensure symmetry         
            
            # Else, load the already computed matrices RAt, ARAt, and inv_ARAt from the reflectivity_constraints
            else:
                RAt = self.reflectivity_constraints['RAt']
                inv_ARAt = self.reflectivity_constraints['inv_ARAt']

            # Correct the sample
            x_star = _correct_sample(x, b, A, inv_ARAt, RAt, verbose)
        
        elif self.lattice.topology == 'E':
            
            # If the correlation matrix is passed, compute the matrices RAt, ARAt, and inv_ARAt
            if R_cd is not None:
                ARAt = A @ R_cd @ (A.T)
                RAt = R_cd @ (A.T)
                inv_ARAt = linalg.inv(ARAt)
                inv_ARAt = 0.5 * (inv_ARAt + inv_ARAt.T)  # Ensure symmetry
            
            # Else, use the already computed matrices RAt, ARAt, and inv_ARAt from the reflectivity_constraints
            else:
                RAt = self.reflectivity_constraints['RAt']
                inv_ARAt = self.reflectivity_constraints['inv_ARAt']
            
            # Correct the sample
            x_star = _correct_sample(x, b, A, inv_ARAt, RAt, verbose)
                

        # Reshape x_star into (1,1) np.array if it only contains one element
        if not x_star.shape:
            x_star = x_star.reshape((1, 1))
        return x_star        
        
    def logpdf_constrained(self, x, sigma2=None, verbose=False):
        """Evaluate the log-density of the constrained Gaussian at x."""
        n = x.shape[0]
        nr_constr = self.reflectivity_constraints['nr_constraints']
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2 # use the stored marginal variance if no value is passed
            
        # 2pi constant
        term1 = -0.5 * (n - nr_constr) * np.log(2 * math.pi)
        
        # logdet
        term2 = -0.5 * np.sum(self.reflectivity_constraints['log_pos_eigs'])
        term2 = term2 - 0.5 * (n - nr_constr) * np.log(sigma2)
        
        # sum of squares
        mean_c_star = self.reflectivity_constraints['mean_c_star']
        inv_Sigma_c_star = self.reflectivity_constraints['inv_R_c_star'] / sigma2
        x_centered = x - mean_c_star
        term3 = -0.5 * (x_centered.T) @ inv_Sigma_c_star @ x_centered
        
        # pdf
        ly = term1 + term2 + term3
        return ly.item()
        
    def _get_logpdf_unconstrained(self, x, method_name, verbose=False):
        method = getattr(self, method_name)  
        if verbose: 
            print('method=', method_name, 'at', method)
        if callable(method):  
            logpdf_x = method(x)  
            return logpdf_x 

###################### ###################### ###################### 
######################        Wavelet         ######################
###################### ###################### ###################### 
        
class Wavelet(Gaussian):
    """The Gaussian process used to represent a blur/wavelet on an Euclidean or cyclic lattice. 
    Includes methods to do conditioning by Kriging and to compute the conditional parameters 
    of the wavelet full conditional with the DFT. 
    """
    
    def __init__(self, mean=None, Sigma=None, constr=False, nh=None, verbose=False):
        """
        Params:
        -------
        mean:
        Sigma: Covariance object created on a 1D lattice.
        constr: True if the endpoints are set to zero.
        nh: (temporary) Used to compute derivatives for collapsed HMC. 
        """
        super().__init__(mean, Sigma)
        
        # Calculate the number of constraints
        if constr:
            nr_constraints = self.lattice.nv - self.lattice.k + 2 
        else:
            nr_constraints = self.lattice.nv - self.lattice.k

        # Save constraint flag and number of constraints
        self.wavelet_constraints = {
            'constr': constr, 
            'nr_constraints': nr_constraints
        }
        if verbose:
            print(f'Linear constraints on wavelet? {constr}.')
            
        # Only computes something if constr=True
        self._compute_attributes_w(verbose) 
        
        # This is temporary code to build the derivatives in collapsed HMC. Should be cleaner and the ders. shd have access to the 2D lattice object.
        self.nh = nh 
        
        # Compute derivatives for collapsed HMC
        self._compute_derivatives_w(verbose)

    def _create_wavelet_constraint_matrices(self, verbose=False):
        """Create constraint matrices used to select the free elements of 
        the wavelet from the extended wavelet vector w.
        
        NOTE: This method is called in the Euclidean case only if 
        wavelet_constraints['constr'], but always in the cyclic case.

        Return:
        -------
        A: Contraint matrix to select the endpoints and/or padding from w.
        b: Constraint independent term (a vector of zeros).
        """

        # Boolean flag indicating if endpoint constraints are active
        constr_w = self.wavelet_constraints['constr']
        nr_constraints = self.wavelet_constraints['nr_constraints']

        # Start with a matrix that selects all elements in the extended w
        A = np.eye(self.lattice.nv)
        
        # Locations in the extended w where the original w is located 
        wavelet_start_v = self.lattice.wavelet_positions['wavelet_start_v']
        wavelet_end_v = self.lattice.wavelet_positions['wavelet_end_v']
        
        # Constraint matrix if endpoints of original w are zero
        if self.wavelet_constraints['constr']:
            # nr_constraints = self.lattice.nv - self.lattice.k + 2 
            A_wu = A[(wavelet_start_v + 1):(wavelet_end_v - 1), :]
            A = np.vstack((A[:(wavelet_start_v + 1), :], 
                A[(wavelet_end_v - 1):, :]))
            
        # Constraint matrix if endpoints of original w are free
        else:
            # nr_constraints = self.lattice.nv - self.lattice.k
            A_wu = A[wavelet_start_v:wavelet_end_v, :]
            if nr_constraints > 0: # it's 0 if k=nv and no margin
                A = np.vstack((A[:wavelet_start_v, :], 
                    A[wavelet_end_v:, :]))
            else:
                A = None
                    
        # Constraint independent term
        b = np.zeros((nr_constraints, 1))  

        # Determine indexes of unconstrained elements
        wavelet_start_v = self.lattice.wavelet_positions['wavelet_start_v']
        wavelet_end_v = self.lattice.wavelet_positions['wavelet_end_v']
        i_all = np.arange(self.lattice.nv) # Create all possible indices
        i_is_endpoint = constr_w & ((i_all == wavelet_start_v) | (i_all == wavelet_end_v - 1))
        i_is_padding = (i_all < wavelet_start_v) | (i_all >= wavelet_end_v)
        constrained = i_is_endpoint | i_is_padding
        unconstrained_indices = i_all[~constrained] # Select only unconstrained indices

        self.wavelet_constraints.update({
            'A': sparse.csr_matrix(A),
            'A_wu': sparse.csr_matrix(A_wu),
            'b':b,
            'unconstrained_indices':unconstrained_indices
        })

    def _compute_attributes_w(self, verbose=False):
        """Compute constraint matrices and parameters of constrained prior."""

        # Build constraint matrices
        self._create_wavelet_constraint_matrices(verbose) # always runs

        # Build constrained covariance matrices if nr_constraints > 0
        if self.wavelet_constraints['nr_constraints'] > 0: 

            # Compute R @ At, and A @ R @ At, its inverse, and logdet 
            A = self.wavelet_constraints['A']
            non_zero_cols = A.getnnz(axis=0) > 0 # Extract indexes of non-zero columns of A
            Rv = self.Sigma.Rv
            RAt = Rv[:, non_zero_cols]
            ARAt = RAt[non_zero_cols, :]
            inv_ARAt = linalg.inv(ARAt)
            inv_ARAt = (inv_ARAt.T + inv_ARAt) / 2 # Ensure symmetry
            logdet_ARAt = math_utils.compute_log_det(ARAt)
            if verbose:
                print(f'A_w=\n{pd.DataFrame(A.toarray())}')
                print(f'RAt=\n{pd.DataFrame(RAt)}')
                print(f'ARAt=\n{pd.DataFrame(ARAt)}')
                
            # Compute logdet(AA')
            logdet_AAt = math_utils.compute_log_det(A.dot(A.T).toarray())

            # Compute the mean and correlation matrix of the constrained prior p(w|Aw=b)
            vector = A.dot(self.mean) - self.wavelet_constraints['b']
            mean_w_star = self.mean - RAt @ inv_ARAt @ vector
            R_w_star_old = (self.Sigma.Rv - RAt @ inv_ARAt @ (RAt.T))
            R_w_star_old = 0.5 * (R_w_star_old + R_w_star_old.T) # Ensure symmetry

            # Using Cholesky decomposition of ARAt
            chol_ARAt = linalg.cholesky(ARAt, lower=True)
            X = linalg.solve_triangular(chol_ARAt, RAt.T, lower=True)
            R_w_star = self.Sigma.Rv - X.T @ X
            R_w_star = 0.5 * (R_w_star + R_w_star.T) # Ensure symmetry
            print(f"Max diff in R_w_star old and Cholesky: {np.max(np.abs(R_w_star - R_w_star_old))}") if verbose else None
            
            log_eigvals_R_w_star, inv_R_w_star = math_utils.pseudo_inverse_RH(R_w_star, verbose=True)
            inv_R_w_star = 0.5 * (inv_R_w_star + inv_R_w_star.T) # Ensure symmetry
            # Print if elements are complex
            if np.iscomplexobj(R_w_star):
                print("R_w_star has complex elements!")
            if np.iscomplexobj(X):
                print("X has complex elements!")
            if np.iscomplexobj(chol_ARAt):
                print("chol_ARAt has complex elements!")
            if np.iscomplexobj(log_eigvals_R_w_star):
                print("log_eigvals_R_w_star has complex elements!")
            if np.iscomplexobj(inv_R_w_star):
                print("inv_R_w_star has complex elements!")
            
            if verbose:
                print(f'Compute attributes_w:')
                print(f'Wavelet marginal variance stored as class attr ={np.round(self.Sigma.sigma2, pp)}')
                print(f'Wavelet Rv stored as class attr =\n{np.round(self.Sigma.Rv, pp)}')
                print(f'R_w_star =\n{np.round(R_w_star, pp)}')

            # Compute the mean and covariance of the non-singular conditional Gaussian p(wu|Aw=b)
            mean_wu_star = None
            unconstrained_indices = self.wavelet_constraints['unconstrained_indices']
            R_wu_star = R_w_star[unconstrained_indices, :][:, unconstrained_indices]
            chol_R_wu_star = linalg.cholesky(R_wu_star, lower=True)
            inv_R_wu_star = np.linalg.inv(R_wu_star)
            inv_R_wu_star = 0.5 * (inv_R_wu_star + inv_R_wu_star.T) # ensure symmetry
            log_eigvals_R_wu_star, pinv_R_wu_star = math_utils.pseudo_inverse_RH(R_wu_star)
            if np.iscomplexobj(R_wu_star):
                print("R_wu_star has complex elements!")
            if np.iscomplexobj(chol_R_wu_star):
                print("chol_R_wu_star has complex elements!")
            if np.iscomplexobj(inv_R_wu_star):
                print("inv_R_wu_star has complex elements!")
            if np.iscomplexobj(log_eigvals_R_wu_star):
                print("log_eigvals_R_wu_star has complex elements!")

            # Convert to sparse objects 
            R_w_star = sparse.csr_matrix(R_w_star)
            inv_R_w_star = sparse.csr_matrix(inv_R_w_star)
            R_wu_star = sparse.csr_matrix(R_wu_star)
            inv_R_wu_star = sparse.csr_matrix(inv_R_wu_star)

            self.wavelet_constraints.update({
                'RAt': RAt, 
                'ARAt': ARAt, 
                'chol_ARAt': chol_ARAt,
                'inv_ARAt': inv_ARAt, 
                'logdet_ARAt': logdet_ARAt, 
                'logdet_AAt': logdet_AAt,
                'mean_w_star': mean_w_star,
                'R_w_star': R_w_star, 
                'log_eigvals_R_w_star': log_eigvals_R_w_star, 
                'inv_R_w_star': inv_R_w_star,
                'mean_wu_star': mean_wu_star,
                'R_wu_star': R_wu_star, 
                'log_eigvals_R_wu_star': log_eigvals_R_wu_star, 
                'logdet_R_wu_star': np.sum(log_eigvals_R_wu_star),
                'inv_R_wu_star': inv_R_wu_star,
                'chol_R_wu_star': chol_R_wu_star
            })    
    
        else:
            self.wavelet_constraints.update({
                'nr_constraints':0,
                'A':None,
                'b':None
            })

    def recompute_mean_R(self, sampler, i):
        """Recompute the mean and correlation matrix in the wavelet full 
        conditional using the current state stored in the sampler object. 
        Works for both Euclidean and cyclic lattices.

        Params:
        -------
        sampler: Sampler object
            The sampler containing the current state and model.
        i: int
            The current iteration index.
        """
        verbose = sampler.mcmc_config.get('verbose', False)
        if verbose:
            print("Recomputing conditional mean and correlation matrix in Wavelet...")

        # Extract the Gaussian objects
        _lik = sampler.par_objs['d']
        _w = sampler.par_objs['w']

        # Extract current states
        c_star = sampler.theta['c_star'][:, i + 1].reshape((-1, 1))
        d_star = sampler.theta['d_star'][:, i + 1].reshape((-1, 1))
        sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
        # sigma2d = sampler.aux['sigma2d'][:, i + 1].item()
        
        # Load current state 
        sigma2c = sampler.theta['sigma2c'][:, i + 1].item()
        # sigma2w = sampler.theta['sigma2w'][:, i + 1].item()
        zeta = sampler.theta['zeta'][:, i + 1].item()
        psi = sampler.model.aux['psi']
        sigma2d_computed = psi * sigma2c * sigma2w * zeta
        sigma2d = sampler.aux['sigma2d'][:, i + 1].item()
        print(f"At the start of the wavelet recompute_mean_R step, sigma2c={sigma2c}, sigma2w={sigma2w}, zeta={zeta}, psi={psi}, sigma2d={sigma2d}, sigma2d_computed={sigma2d_computed}") if verbose else None
    
        if sampler.lattice.topology == 'C':

            # Extract bases of correlation matrices
            # base_Sigma_w = sigma2w * _w.Sigma.base_R
            base_inv_Sigma_w = _w.Sigma.base_Qv / sigma2w
            # base_Sigma_d = sigma2d * _lik.Sigma.base_R

            # Create reflectivity convolutional matrix
            bases_G, Gamma = create_C(sampler.lattice, c_star) # gamma is old C_w

            # Compute bases in Gamma' @ Sigma_d^{-1}
            bases_Gt_inv_Sigma_d = _compute_bases_Gammat_inv_Sigma_d(sampler, i, bases_G) # runs ok, result ok?

            # Compute base of Gamma' @ Sigma_d^{-1} @ Gamma 
            base_Gt_inv_Sigma_d_G = _compute_base_Gammat_inv_Sigma_d_Gamma(sampler, bases_G, bases_Gt_inv_Sigma_d) # runs ok, result ok?
           
            # Compute base of the wavelet full conditional covariance matrix
            base_Sigma_cond, base_inv_Sigma_cond = _compute_wavelet_conditional_covariance(base_inv_Sigma_w, base_Gt_inv_Sigma_d_G) 
            base_Sigma_cond = 0.5 * (base_Sigma_cond + transpose_base_circ(base_Sigma_cond))  # Ensure symmetry
            base_R_cond = base_Sigma_cond / sigma2w # convert to correlation matrix
            R_cond = None
            
            # Compute the matrix-vector product Gamma' @ Sigma_d^{-1} @ d_star
            Gt_inv_Sigma_d_d = _compute_Gt_inv_Sigma_d_d(sampler, i, bases_Gt_inv_Sigma_d)

            # Compute the conditional mean 
            mean_cond = _compute_wavelet_conditional_mean(sampler, i, Gt_inv_Sigma_d_d, base_Gt_inv_Sigma_d_G, base_Sigma_cond)
        
        elif sampler.lattice.topology == 'E':
            
            # Extract covariance matrices
            R_w = _w.Sigma.R
            Sigma_d = sigma2d * _lik.Sigma.R

            # Compute conditional mean and covariance
            bases_G, Gamma = create_C(sampler.lattice, c_star)
            Z = sigma2w * Gamma @ R_w @ (Gamma.T) + Sigma_d
            Z = 0.5 * (Z + Z.T) # Ensure symmetry
            inv_Z = linalg.inv(Z)
            inv_Z = 0.5 * (inv_Z + inv_Z.T) # Ensure symmetry
            RGammaT = R_w @ (Gamma.T)
            correction_left = sigma2w * RGammaT @ inv_Z
            mean_cond = correction_left @ d_star
            R_cond = R_w - correction_left @ (RGammaT.T)
            base_R_cond = None
           
        # Return updated mean and base of correlation matrix
        return mean_cond, base_R_cond, R_cond

    def correct_sample(self, x, R_wd=None, verbose=False):
        """Euclidean and cyclic. Correct x with conditioning by Kriging so that Ax=b.
        Params:
        -------
        x: (nv x 1) np.array.
            The wavelet sample to be corrected.
        R_wd: (nv x nv) np.array
            The full conditional correlation matrix of the wavelet. 
            Only used for the full conditional wavelet Gibbs step.
        """

        # Load the constraint matrices
        b = self.wavelet_constraints['b'] 
        A = self.wavelet_constraints['A']

        # If the correlation matrix is passed, compute the matrices RAt, ARAt, and inv_ARAt
        if R_wd is not None:

            # new
            non_zero_cols = A.getnnz(axis=0) > 0 # Extract indexes of non-zero columns of A
            RAt = R_wd[:, non_zero_cols]
            ARAt = RAt[non_zero_cols, :]
            inv_ARAt = linalg.inv(ARAt)
            inv_ARAt = (inv_ARAt.T + inv_ARAt) / 2 # Ensure symmetry
        
        # Else, use the already computed matrices RAt, ARAt, and inv_ARAt from the wavelet
        else:
            RAt = self.wavelet_constraints['RAt']
            inv_ARAt = self.wavelet_constraints['inv_ARAt']
        
        # Correct the sample
        x_star = _correct_sample(x, b, A, inv_ARAt, RAt, verbose)
                
        return x_star        
        
    def logpdf_constrained(self, x, sigma2=None, verbose=False):
        """Euclidean or cyclic. Evaluate the log-density of the constrained Gaussian at x."""
        n = x.shape[0]
        nr_constr = self.wavelet_constraints['nr_constraints']
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2 # use the stored marginal variance if no value is passed
        
        # Constant term
        term1 = -0.5 * (n - nr_constr) * np.log(2 * math.pi)
        
        # determinant
        term2 = -0.5 * np.sum(self.wavelet_constraints['log_eigvals_R_w_star'])
        term2 = term2 - 0.5 * (n - nr_constr) * np.log(sigma2)
        
        # sum of squares
        mean_w_star = self.wavelet_constraints['mean_w_star']
        x_centered = x - mean_w_star
        inv_Sigma_w_star = self.wavelet_constraints['inv_R_w_star'] / sigma2
        term3 = -0.5 * (x_centered.T) @ inv_Sigma_w_star @ x_centered
        
        # pdf
        ly = term1 + term2 + term3
        return ly.item()
        
    def _get_logpdf_unconstrained(self, x, method_name, verbose=False):
        method = getattr(self, method_name) 
        if verbose: print('method=', method_name, 'at', method)
        if callable(method):  
            logpdf_x = method(x)  
            return logpdf_x 
        
    def _compute_derivatives_w(self, verbose=False):
        """Compute the derivative of the w* vector, the W0 matrix, 
        and the W matrix, with respect to each coordinate of the wavelet.
        
        Returns:
        -------
        d_w_star_matrix: np.array
            Column i contains the derivative of w* wrt coordinate i, i = 0, 1, ..., k (length of wavelet).
        """
        
        # Derivatives of the w* vector
        constr_w = self.wavelet_constraints['constr']
        if self.lattice.topology == 'E':
            d_w_star_matrix = np.eye(self.lattice.k)
            if constr_w: 
                d_w_star_matrix[0, 0] = d_w_star_matrix[-1, -1] = 0

        elif self.lattice.topology == 'C':
            d_w_star_matrix = np.eye(self.lattice.nv)
            wavelet_start_v = self.lattice.wavelet_positions['wavelet_start_v']
            wavelet_end_v = self.lattice.wavelet_positions['wavelet_end_v']
            if constr_w: 
                wavelet_start_v += 1
                wavelet_end_v -= 1
            d_w_star_matrix[:wavelet_start_v, :wavelet_start_v] = 0
            d_w_star_matrix[wavelet_end_v:, wavelet_end_v:] = 0

        # Derivatives of W0 and W (or bases, if Cyclic)
        d_W0 = []
        bases_d_W0 = np.zeros_like(d_w_star_matrix)
        bases_d_W0T = np.zeros_like(d_w_star_matrix)
        d_W = []
        base_d_W = []
        base_d_WT = []
        for i in np.arange(self.lattice.nv):
            d_W0_i, d_W0T_i = derivative_W0_i(i, self, constr_w=constr_w)
            d_W0.append(d_W0_i)

            # Derivative of W
            one_vector = np.zeros((1, self.nh))
            one_vector[0, 0] = 1
            if self.lattice.topology == 'C':

                bases_d_W0[:, i] = d_W0_i[0, :] 
                bases_d_W0T[:, i] = d_W0T_i[0, :] 
                base_d_W_i = np.kron(one_vector, bases_d_W0[:, i].reshape(-1, 1))
                base_d_WT_i = transpose_base_circ(base_d_W_i)
                base_d_W.append(base_d_W_i) 
                base_d_WT.append(base_d_WT_i) 

            if self.lattice.topology == 'E':
                d_W_i = np.kron(np.eye(self.lattice.nh), d_W0_i)
                d_W.append(d_W_i)

        # Defaults for non-cyclic case
        bases_d_W = None
        bases_d_WT = None
        fft_bases_d_W0 = None
        fft_bases_d_W0T = None
        fft2_bases_d_W = None
        fft2_bases_d_WT = None

        # Transform the base_d_W and base_d_WT lists to arrays (cyclic only)
        if self.lattice.topology == 'C':
            nv = self.nv
            nh = self.nh
            bases_d_W = np.zeros((nv, nv, nh))
            bases_d_WT = np.zeros((nv, nv, nh))

            # Fill with bases
            for idx in np.arange(nv):
                bases_d_W[idx, :, :] = base_d_W[idx]
                bases_d_WT[idx, :, :] = base_d_WT[idx]

            print(f"bases_d_W.shape = {bases_d_W.shape}") if verbose else None
            print(f"bases_d_WT.shape = {bases_d_WT.shape}") if verbose else None

            # Compute the FFT of the bases if Cyclic
            fft_bases_d_W0 = scipy.fft.fft(bases_d_W0, axis=0)
            fft_bases_d_W0T = np.conj(fft_bases_d_W0)
            fft2_bases_d_W = scipy.fft.fft2(bases_d_W, axes=(1, 2))
            fft2_bases_d_WT = np.conj(fft2_bases_d_W)


        print(f"Computed wavelet derivatives.") if verbose else None

        # Store
        self.wavelet_derivatives = {
            'd_w_star': d_w_star_matrix, 
            'd_W0': d_W0,
            'd_W': d_W,
            'bases_d_W0': bases_d_W0,
            'bases_d_W0T': bases_d_W0T,
            'bases_d_W': bases_d_W,
            'bases_d_WT': bases_d_WT,
            'fft_bases_d_W0': fft_bases_d_W0,
            'fft_bases_d_W0T': fft_bases_d_W0T,
            'fft2_bases_d_W': fft2_bases_d_W,
            'fft2_bases_d_WT': fft2_bases_d_WT
        }
        
###################### ###################### ###################### 
######################      SeismicData       ######################
###################### ###################### ######################        

class SeismicData(Gaussian):
    """The Gaussian field used to represent the seismic data on Euclidean or cyclic lattices. 
    Includes methods to do conditioning by Kriging and to compute the conditional parameters 
    of the extended seismic data full conditional with the DFT."""
    
    def __init__(self, mean=None, Sigma=None, b=None, verbose=False):
        """
        Params:
        -------
        mean:
        Sigma: Covariance object.
        b: The observed data on the original lattice.
        """
        if not hasattr(Sigma, 'lattice'):
            raise Exception('The Covariance matrix must be based on a Lattice object so I know if I need to apply constraints.')
        super().__init__(mean, Sigma)
        
        # Check that the passed observed data b has the right dimensions
        if b is not None:
            print(f"The passed b has shape {b.shape} and type {type(b)}.") 
            if b.shape[0] != self.lattice.n_ava: 
                raise Exception('The passed observed data has dimensions {0}x{1} but the extended lattice expects an {2}x1 vector.'.format(b.shape[0], b.shape[1], self.lattice.n_ava))    
        
        # Set the constraint flag
        constr = (self.lattice.mv > 0) or (self.lattice.mh > 0) # True if margins != 0
        if verbose: 
            print(f'Extended lattice? {constr}.')
            
        # Save the constraint flag, nr of constraints, and constraints
        if constr:
            nr_constraints = self.lattice.n - self.lattice.n_ava 
        else:
            nr_constraints = 0
        self.data_constraints = {
            'constr': constr, 
            'nr_constraints': nr_constraints, 
            'b':b
        }

        # Compute constraint matrices
        self._compute_attributes_d()    
    
    def _create_data_constraint_matrices(self):
        """Create constraint matrices to select the unobserved (d_u) or 
        observed (d_o) portions of the extended data vector d. 

        Return:
        -------
        X_matrix: Selection matrix for the columns occupied by the AVA lattice. 
        U_matrix: Selection matrix for the rows occupied by the AVA lattice. 
        A_do: Selects the elements corresponding to the observed AVA data. 
        A_do_ext: Constraint matrix to select d_o from (d, c_u).
        A_du: Constraint matrix to select d_u from d.
        """

        # Extract variables for readability
        nh_ava = self.lattice.nh_ava
        nv_ava = self.lattice.nv_ava
        nh = self.lattice.nh
        nv = self.lattice.nv
        ava_start_h = self.lattice.ava_positions['ava_start_h']
        ava_end_h = self.lattice.ava_positions['ava_end_h']
        ava_start_v = self.lattice.ava_positions['ava_start_v']
        ava_end_v = self.lattice.ava_positions['ava_end_v']
                                                   
        # Constraint matrix to select d_o from d
        X_matrix = sparse.lil_matrix((nh_ava, nh))
        X_matrix[:, ava_start_h:ava_end_h] = sparse.identity(nh_ava)
        U_matrix = sparse.lil_matrix((nv_ava, nv))
        U_matrix[:, ava_start_v:ava_end_v] = sparse.identity(nv_ava)
        
        self.data_constraints.update({
            'X_matrix':X_matrix, 
            'U_matrix':U_matrix, 
        })
        print(f"Created data_constraint matrices.") 
    
    def _compute_attributes_d(self):
        """Compute matrices for conditioning by Kriging, some determinants,
        and the non-zero elements in the observational precision matrices.
        """  
        inv_X_R_d_h_Xt = None
        inv_U_R_d_v_Ut = None
        X_star = None
        U_star = None
        logdet_AAt = None
        logdet_ARAt = None
        ARAt = None
        inv_ARAt = None
        # constrained_ss = 0

        # Build constraint matrices and related matrices and determinants
        if self.data_constraints['constr']:
            
            # Build the constraint matrices
            self._create_data_constraint_matrices()

            # Create helper matrices for the correction for the AVA data
            print(f"Create helper matrices for the correction for the AVA data")
            R_d_h = self.Sigma.Rh
            X_matrix = self.data_constraints['X_matrix']
            X_R_d_h = sparse.csr_matrix(sparse.csr_matrix(X_matrix).dot(R_d_h))
            X_R_d_h_Xt = X_R_d_h.dot(X_matrix.T)
            inv_X_R_d_h_Xt = sparse.csr_matrix(linalg.inv(X_R_d_h_Xt.toarray()))
            X_star = sparse.csr_matrix((X_R_d_h.T).dot(inv_X_R_d_h_Xt))

            R_d_v = self.Sigma.Rv
            U_matrix = self.data_constraints['U_matrix']
            U_R_d_v = sparse.csr_matrix(sparse.csr_matrix(U_matrix).dot(R_d_v))
            U_R_d_v_Ut = U_R_d_v.dot(U_matrix.T)
            inv_U_R_d_v_Ut = sparse.csr_matrix(linalg.inv(U_R_d_v_Ut.toarray()))
            U_star = sparse.csr_matrix((U_R_d_v.T).dot(inv_U_R_d_v_Ut))

            # Log-dets of correlation matrices
            logdet_AAt = 0
            logdet_X_R_d_h_Xt = math_utils.compute_log_det(X_R_d_h_Xt.toarray(), epsilon=0.00000001)
            logdet_U_R_d_v_Ut = math_utils.compute_log_det(U_R_d_v_Ut.toarray(), epsilon=0.00000001)
            nh_ava = self.lattice.nh_ava
            nv_ava = self.lattice.nv_ava
            logdet_ARAt = nv_ava * logdet_X_R_d_h_Xt + nh_ava * logdet_U_R_d_v_Ut
            ARAt = None
            inv_ARAt = None
            
        self.data_constraints.update({
            'inv_X_R_d_h_Xt':inv_X_R_d_h_Xt,
            'inv_U_R_d_v_Ut':inv_U_R_d_v_Ut,
            'X_star':X_star,
            'U_star':U_star,
            'logdet_AAt':logdet_AAt,
            'logdet_ARAt':logdet_ARAt,
            'ARAt': ARAt,
            'inv_ARAt': inv_ARAt
        })

        # List comprehension to get dense indexes
        if self.lattice.topology == 'C':
            base_Qh = self.Sigma.base_Qh
        elif self.lattice.topology == 'E':
            base_Qh = self.Sigma.Qh[0, :]
        dense_indexes = [index for index, value in enumerate(base_Qh) if np.abs(value) > 1e-8]  
        self.dense_indexes = dense_indexes
            
    def update_Sigma(self, Sigma):
        """Update Sigma and recompute derived attributes."""
        super().update_Sigma(Sigma)  
        self._compute_attributes_d()    

    def correct_sample(self, x, verbose=False):
        """Correct x with conditioning by Kriging so that Ax=b. 
        Works for Euclidean and cyclic lattices."""

        print(f"Correcting seismic data d with constraints Ad=do...") if verbose else None

        # Extract the constraint matrices
        X_matrix = self.data_constraints['X_matrix'] 
        U_matrix = self.data_constraints['U_matrix'] 
        nv = self.lattice.nv 
        nh = self.lattice.nh
        n = self.lattice.n 
        n_ava = self.lattice.n_ava # number of nodes in the observed lattice
        nv_ava = self.lattice.nv_ava 
        nh_ava = self.lattice.nh_ava 

        # Compute the subset centered matrix
        D = x.reshape((nv, nh), order='F')  
        UDXt = U_matrix @ D @ X_matrix.T
        vector1 = UDXt.reshape((n_ava, 1), order='F')
        v = vector1 - self.data_constraints['b']
        V = v.reshape((nv_ava, nh_ava), order='F')

        # Compute the correction term
        U_star = self.data_constraints['U_star']
        X_star = self.data_constraints['X_star']
        correction_term_matrix = U_star @ V @ X_star.T
        correction_term = correction_term_matrix.reshape((n, 1), order='F')
        x_star = x - correction_term
        print(f"Corrected sample d_star:\n{x_star}") if verbose else None
        
        # Reshape d_star into (1,1) np.array if it only contains one element
        if not x_star.shape:
            x_star = x_star.reshape((1, 1))
        
        if verbose:
            print('Constraints b:\n', self.data_constraints['b'])
            
        return x_star        
        
    def logpdf_constrained(self, x, method_name, sigma2=None, verbose=False):
        """Euclidean or cyclic. Evaluate log p(x|Ax=b).
        
        THIS IS A DISASTER, RE-WRITE.
        """
        b = self.data_constraints['b']
        nr_constr = self.lattice.n_ava
        A_mean = self.data_constraints['A_do'] @ self.mean
        if sigma2 is None:
            sigma2 = self.Sigma.sigma2
        b_centered = b - A_mean
        B = b_centered.reshape((self.lattice.nv_ava, self.lattice.nh_ava), order='F')
        
        # log p(Ax=b|x)
        l1 = -0.5 * self.data_constraints['logdet_AAt']
        
        # log p(x)
        l2 = self._get_logpdf_unconstrained(x, method_name)
        
        # log p(Ax=b)
        l3_3_temp = sparse.csr_matrix(self.data_constraints['inv_U_R_d_v_Ut'].dot(B))
        l3_3_temp1 = l3_3_temp.dot(self.data_constraints['inv_X_R_d_h_Xt'].T)
        l3_3_temp1_vector = l3_3_temp1.reshape((self.lattice.nv_ava * self.lattice.nh_ava, 1), order='F')

        l3_3_temp2 = (b.T) @ l3_3_temp1_vector
        ss = l3_3_temp2 / sigma2
        l3_3 = -0.5 * ss
        
        l3 = (-0.5 * (self.lattice.n - nr_constr) * np.log(2 * math.pi) 
              - 0.5 * (self.lattice.n - nr_constr) * np.log(sigma2) 
              - 0.5 * self.data_constraints['logdet_ARAt']
              - 0.5 * l3_3) 
        
        if np.sum(x == b) == self.n:
            l3 = np.array(0) # p(Ax=b)= p(Ib=b)=1.
        
        if verbose:
            print(f'logdet_AAt=', l1)
            print(f'with method {method_name}, logpdf(x)=', l2)
            print(f'dimensions of l3_3_temp1={l3_3_temp1.shape}')
            print(f'dimensions of l3_3_temp1_vector={l3_3_temp1_vector.shape}')
            print(f'dimensions of b={b.shape}')
            print(f'logpdf_Ax={l3}')
        
        ly = l1.item() + l2.item() - l3.item()
        return ly
        
    def _get_logpdf_unconstrained(self, x, method_name, verbose=False):
        method = getattr(self, method_name)  
        if verbose: print('method=', method_name, 'at', method)
        if callable(method):  
            logpdf_x = method(x)  
            return logpdf_x 
