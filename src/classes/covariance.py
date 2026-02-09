"""Covariance utilities for Gaussian priors.

Terminology: `w` is the blur/wavelet and `c` is the image/reflectivity.
"""

# Third-party library imports
import numpy as np
from scipy import linalg
import scipy

# Local library imports
from src.utils import math_utils 
from src.utils.model_utils import exp_cor
from src.utils.efficient_algebra_utils import invert_circ, compute_log_det_circ
from src.classes.circulant import Circulant

###################### ###################### ###################### 
######################      Covariance        ######################
###################### ###################### ###################### 


class Covariance():
    """Assumes a covariance function that can be written as the product cov(x, y) = sigma2 * cor(h).
    """
    
    def __init__(self, sigma2=None, lattice=None, n=None, R=None, base_R=None):
        """
        Params:
        ------
        sigma2: Scalar marginal variance.
        lattice: if not passed, i need to pass either n or R.
        n: Optional dimensions of the correlation matrix.
        R: Optional nxn correlation matrix.
        """

        # Check some basics
        if lattice is None and (n is None and R is None):
            raise Exception('I need either the lattice, or the dimensions of the covariance matrix, or the correlation matrix.')
            
        # Marginal variance
        if sigma2 is not None:
            if np.imag(sigma2) != 0 or sigma2 <= 0:
                raise Exception('Sigma2 must be real and positive.')
            else:
                self.sigma2 = sigma2
        else:
            self.sigma2 = 1 
            
        # Initialize Kronecker factors and bases to None. They get filled with the exp_cor method
        self.Rv = None
        self.Qv = None
        self.base_Rv = None
        self.base_Qv = None

        self.Rh = None
        self.Qh = None
        self.base_Rh = None
        self.base_Qh = None
        
        self.R = None
        self.Q = None
        self.base_R = None
        self.base_Q = None
            
        # Create n and correlation matrix without a lattice
        if lattice is None:
            
            # if R is passed
            if R is not None: 
                
                # Check that n and R have the correct dimensions
                if R.shape[0] != R.shape[1]:
                    raise Exception('Correlation matrix must be square.')
                if n is not None:
                    if n != R.shape[0]:
                        raise Exception("The given dimensions (n={0}) don\'t match the dimensions of the given correlation matrix ({1}x{1}).".format(n, R.shape[0]))
                
                # Initialize attributes
                self.n = R.shape[0]
                self.R = R
                self.logdet_R = math_utils.compute_log_det(self.R)
                
            # R = I
            else: 
                self.n = n
                self.R = np.eye(self.n)
                self.logdet_R = 0
                
        # Create n and correlation matrix on a lattice
        else:
            self.lattice = lattice
            self.n = self.lattice.n
                
            # For Euclidean lattices
            if self.lattice.topology=='E':  
                
                if self.n > 400: 
                    raise Exception('Euclidean correlation matrix with n>400 not allowed.')
                
                # If the lattice is small
                else:    
                    
                    # The user can pass R 
                    if R is not None: 
                    
                        # Check that R matches size of lattice
                        if R.shape[0] != R.shape[1]:
                            raise Exception('Correlation matrix must be square.')
                        if n is not None:
                            if n != R.shape[0]:
                                raise Exception("The given dimensions (n={0}) don\'t match the dimensions of the given correlation matrix ({1}x{1}).".format(n, R.shape[0]))

                        # Store R and compute its inverse and det
                        self.R = R
                        self.Q = linalg.inv(self.R)
                        self.logdet_R = math_utils.compute_log_det(self.R)

                    # If no R is passed, assume R=I
                    else:
                        self.R = np.eye(self.n)
                        self.Q = np.eye(self.n)
                        self.logdet_R = 0
            
            # For cyclic lattices
            if self.lattice.topology=='C':
                
                # The user can pass an arbitrarily large base_R
                if base_R is not None:
                    if (base_R.shape[0] != lattice.nv or base_R.shape[1] != lattice.nh):
                        raise Exception('Dimension mismatch in passed base_R.')
                    
                    # Store base_R and compute its inverse and det
                    self.base_R = base_R
                    self.base_Q = invert_circ(base_R)
                    self.logdet_R = compute_log_det_circ(base_R)
                    
                 # If no base_R is passed, assume R=I
                else:
                    self.base_R = np.eye(self.n)[0, :].reshape(-1, 1)
                    self.base_Q = invert_circ(self.base_R)
                    self.logdet_R = compute_log_det_circ(self.base_R)
                    
                # If lattice is small, build explicit R
                if self.n < 300:
                    print(f'Covariance class: Explicitly building R and Q because n < 300.')
                    self.R = linalg.circulant(self.base_R.reshape(-1)).T
                    self.Q = linalg.circulant(self.base_Q.reshape(-1)).T
            
    def create_exp_cor_matrix(self, rho, p, coord, explicit=False):
        """Create correlation matrix/base with the formula exp(-(h/rho)^p).
        For cyclic lattices: Set explicit=True to store the correlation and precision matrices
        in each coordinate and spatially, and not only their bases. 
        
        Params:
        ------
        rho: Correlation range.
        p: Exponent.
        coord: (h)orizontal or (v)ertical.
        """
        if coord not in ['v', 'h']:
            raise Exception('Coordinate must be "v" or "h".')
        
        # Store rho and p
        setattr(self, 'rho_' + coord, rho)
        setattr(self, 'p_' + coord, p)
        
        # Build name of the attributes I'll access/update 
        dist_name = 'dist_' + coord
        base_R_1d_name = 'base_R' + coord
        fft_base_R_1d_name = 'fft_base_R' + coord
        base_Q_1d_name = 'base_Q' + coord
        R_1d_name = 'R' + coord
        Q_1d_name = 'Q' + coord
        Q_2d_name = 'Q'
        R_2d_name = 'R'
        
        # Create correlation matrices in Euclidean grid
        if self.lattice.topology == 'E':
            H = self.lattice.dist[dist_name]['H']
            R_1d = exp_cor(rho, p, H=H)['R']
            Q_1d = linalg.inv(R_1d)
            setattr(self, R_1d_name,  R_1d)
            setattr(self, Q_1d_name,  Q_1d)
            
            if self.Rv is not None and self.Rh is not None:
                self.R = linalg.kron(self.Rh, self.Rv)
                self.Q = linalg.inv(self.R)
                self.logdet_R = math_utils.compute_log_det(self.R)
            
        # Create bases of correlation matrices in cyclic grid
        if self.lattice.topology == 'C':
            
            # For a 1D axis (e.g. horizontal or vertical)
            h = self.lattice.dist[dist_name]['base_H'] # load distance vector
            R_1d = exp_cor(rho, p, h=h) # create cor. matrix and its base
            setattr(self, base_R_1d_name,  R_1d['base_R']) # save base 
            fft_R_1d = scipy.fft.fft(R_1d['base_R'].flatten()) # compute fft of base
            setattr(self, fft_base_R_1d_name, fft_R_1d) # save fft of base
            cor_1d = Circulant(getattr(self, base_R_1d_name)) # instance circulant with base
            base_Q_1d = cor_1d.invert_circ() # compute base of inverse
            setattr(self, base_Q_1d_name, base_Q_1d) # save base of inverse

            # Spatial correlation matrix
            if self.base_Rv is not None and self.base_Rh is not None:
                self.base_R = (self.base_Rh @ (self.base_Rv.T)).T
                self.fft_base_R = scipy.fft.fft2(self.base_R)
                self.logdet_R = math_utils.compute_log_det_circ(self.base_R)
                cor_2d = Circulant(self.base_R)
                self.base_Q = cor_2d.invert_circ()
                print(f'Finished with the spatial correlation matrix.')
                
            # Store full correlation matrices
            print(f'Storing full 1D correlation matrices')
            setattr(self, R_1d_name,  R_1d['R'])
            setattr(self, Q_1d_name, cor_1d.build_matrix(base_Q_1d))
            if explicit:    
                try:
                    print(f"Storing R_2d h and/or v")
                    setattr(self, R_2d_name, cor_2d.build_matrix(self.base_R))
                    setattr(self, Q_2d_name, cor_2d.build_matrix(self.base_Q))
                except:
                    pass